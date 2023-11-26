import math
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig


def rotate_half(x):
   x1 = x[..., : x.shape[-1] // 2]
   x2 = x[..., x.shape[-1] // 2 :]
   return torch.cat((-x2, x1), dim=-1)


# 做的就是q矩阵、k矩阵和(sinmθ + cosmθ)的矩阵相乘 -> 返回的是注入了RoPE的q和k矩阵
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
   cos = cos[position_ids].unsqueeze(1)  # [seq_len, dim] -> [batch_size, 1, seq_len, head_dim]
   sin = sin[position_ids].unsqueeze(1)  # [seq_len, dim] -> [batch_size, 1, seq_len, head_dim]
   # xxw q矩阵 * cosmθ + 特殊的q矩阵 * sinmθ
   q_embed = (q * cos) + (rotate_half(q) * sin)
   # xxw k矩阵 * cosmθ + 特殊的k矩阵 * sinmθ
   k_embed = (k * cos) + (rotate_half(k) * sin)
   # 返回带RoPE的q和k
   return q_embed, k_embed


# xxw 实现GQA的关键
# repeat k/v heads if n_kv_heads < n_heads
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
   """
   This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
   num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
   """
   batch, num_key_value_heads, slen, head_dim = hidden_states.shape
   if n_rep == 1:
      return hidden_states
   hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
   return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# xxw 构造attention_mask的核心函数
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
   """
   Make causal mask used for bi-directional self-attention.
   """
   # xxw 这里否早的还是双向的self-attention
   # 假设 tgt_len=3
   bsz, tgt_len = input_ids_shape
   # xxw生成一个[tgt_len, tgt_len]的矩阵，矩阵中的每一个元素都是计算机最小值
   # tensor([[1.0000e-09, 1.0000e-09, 1.0000e-09],
   #         [1.0000e-09, 1.0000e-09, 1.0000e-09],
   #         [1.0000e-09, 1.0000e-09, 1.0000e-09]])
   mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
   # xxw mask_cond是tgt_len这么长的一个tensor
   # tgt_len = 3 -> tensor([0, 1, 2])
   mask_cond = torch.arange(mask.size(-1), device=device)
   # xxw mask最终变成了一个上三角阵
   # tensor([[0.0000e+00, 1.0000e-09, 1.0000e-09],
   #         [0.0000e+00, 0.0000e+00, 1.0000e-09],
   #         [0.0000e+00, 0.0000e+00, 0.0000e+00]])
   mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
   mask = mask.to(dtype)

   if past_key_values_length > 0:
      mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
   # 扩展mask的前两个维度，变为bsz和1
   # bsz=5，则最后返回的维度为[5, 1, 3, 3]
   # tensor([[[[0.0000e+00, 1.0000e-09, 1.0000e-09],
   #           [0.0000e+00, 0.0000e+00, 1.0000e-09],
   #           [0.0000e+00, 0.0000e+00, 0.0000e+00]]],
   #
   #         [[[0.0000e+00, 1.0000e-09, 1.0000e-09],
   #           [0.0000e+00, 0.0000e+00, 1.0000e-09],
   #           [0.0000e+00, 0.0000e+00, 0.0000e+00]]],
   #
   #         [[[0.0000e+00, 1.0000e-09, 1.0000e-09],
   #           [0.0000e+00, 0.0000e+00, 1.0000e-09],
   #           [0.0000e+00, 0.0000e+00, 0.0000e+00]]],
   #
   #         [[[0.0000e+00, 1.0000e-09, 1.0000e-09],
   #           [0.0000e+00, 0.0000e+00, 1.0000e-09],
   #           [0.0000e+00, 0.0000e+00, 0.0000e+00]]],
   #
   #         [[[0.0000e+00, 1.0000e-09, 1.0000e-09],
   #           [0.0000e+00, 0.0000e+00, 1.0000e-09],
   #           [0.0000e+00, 0.0000e+00, 0.0000e+00]]]])
   return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# xxw 扩展出一个[bsz, 1, tgt_seq_len, src_seq_len]的矩阵
def _expand_mask(mask, dtype, tgt_len = None):
   """
   Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
   """
   bsz, src_len = mask.size()
   tgt_len = tgt_len if tgt_len is not None else src_len

   expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

   inverted_mask = 1.0 - expanded_mask

   return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base # 10000
        # xxw 计算一组角度θi
        # 1.0 / 10000 ^ (2i/d) = 10000 ^ (-2i/d)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        # xxw 有多少个词就有多少个位置m
        # t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = torch.arange(self.max_seq_len_cached, device=device).to(dtype)
        # xxw m和θ做外积 -> mθ
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        # xxw 获取cos和sin
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            # 调用_set_cos_sin_cache
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        # xxw 返回cos和sin
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LlamaAttention(nn.Module):

   def __init__(self, config:LlamaConfig):
      super().__init__()
      self.hidden_size = config.hidden_size
      self.num_heads = config.num_attention_heads
      self.head_dim = config.hidden_size // self.num_heads
      self.num_key_value_heads = config.num_key_value_heads
      self.num_key_value_groups = self.num_heads // self.num_key_value_heads
      self.max_position_embeddings = config.max_position_embedding
      self.rope_theta = config.rope_theta

      self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
      self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
      self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
      self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

      def _init_rope(self):
         # 得到sinmθ、cosmθ -> 使用θ对位置m做偏移
         self.rotary_emb = LlamaRotaryEmbedding(
               self.head_dim,
               max_position_embeddings=self.max_position_embeddings,
               base=self.rope_theta,
         )

      def forward(
         self,
         hidden_states,
         attention_mask = None,
         position_ids = None,
         past_key_value = None,
         output_attentions = False,
         use_cache = False,
         padding_mask = None
      ):
         bsz, q_len, _ = hidden_states.size()
         query_states = self.q_proj(hidden_states)
         key_states = self.k_proj(hidden_states)
         value_states = self.v_proj(hidden_states)

         query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
         key_states = key_states.view(bsz, q_len, self.num_key_value_head, self.head_dim).transpose(1, 2)
         value_states = value_states.view(bsz, q_len, self.num_key_value_head, self.head_dim).transpose(1, 2)

         kv_seq_len = q_len

         # 更新长度
         if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

         # rope的计算跟长度有关
         cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

         # kv_cache，叠加上个时间步的kv
         if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], key_states], dim=2)

         # 更新past_key_value, 保留当前时间步的kv
         past_key_value = (key_states, value_states) if use_cache else None

         # GQA
         key_states = repeat_kv(key_states, self.num_key_value_groups)
         value_states = repeat_kv(value_states, self.num_key_value_groups)

         ### trick ###

         ### trick ###

         attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

         if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

         attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float32).to(query_states.dtype)
         attn_output = torch.matmul(attn_weights, value_states)
         attn_output = attn_output.transpose(1, 2).contiguous()
         attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
         attn_output = self.o_proj(attn_output)

         return attn_output, attn_weights, past_key_value


class LlamaMLP(nn.Module):

   def __init__(self, config:LlamaConfig):
      super().__init__()
      self.hidden_size = config.hidden_size
      self.intermediate_size = config.intermediate_size
      self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
      self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
      self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
      self.act_fn = ACT2FN[config.hidden_act]

   def forward(self, x):
      down_proj = self.down_proj(self.act_fn(self.gate_proj * self.up_proj(x)))
      return down_proj

class LlamaDecoderLayer(nn.Module):
   def __init__(self, config:LlamaConfig):
      super().__init__()
      self.hidden_size = config.hidden_size
      self.self_attn = LlamaAttention(config=config) # attention
      self.mlp = LlamaMLP(config) # mlp
      self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps) # norm
      self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

   def forward(
      self,
      hidden_states,
      attention_mask = None,
      position_ids = None,
      past_key_value = None,
      output_attentions = False,
      use_cache = False,
      padding_mask = None,
   ):
      residual = hidden_states # 第一次残差
      hidden_states = self.input_layernorm(hidden_states)

      # attention
      hidden_states, self_attn_weights, present_key_value = self.self_attn(
         hidden_states=hidden_states,
         attention_mask=attention_mask,
         position_ids=position_ids,
         past_key_value=past_key_value,
         output_attentions=output_attentions,
         use_cache=use_cache,
         padding_mask=padding_mask,
      )

      hidden_states = residual + hidden_states
      residual = hidden_states # 第二次残差
      hidden_states = self.post_attention_layernorm(hidden_states)
      hidden_states = self.mlp(hidden_states)
      hidden_states = residual + hidden_states

      output = (hidden_states, )

      if output_attentions:
        output += (self_attn_weights,)

      if use_cache:
         output += (present_key_value,)

      return output


class LlamaPreTrainedModel(PreTrainedModel):
   config_class = LlamaConfig
   base_model_prefix = "model"
   supports_gradient_checkpointing = True

   # 初始化w和bias
   def _init_weights(self, module):
      std = self.config.initializer_range
      # 线性层
      if isinstance(module, nn.linear):
         module.weight.data.normal_(mean=0.0, std=std)
         if module.bias is not None:
            module.bias.data.zero_()
      # embedding层
      elif isinstance(module, nn.Embedding):
         module.weight.data.normal_(mean=0.0, std=std)
         if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
   # 默认不开启gradient_checkponiting
   def _set_gradient_checkpointing(self, module, value=False):
      if isinstance(module, LlamaModel):
         module.gradient_checkponiting = value


class LlamaModel(LlamaPreTrainedModel):
   def __init__(self, config:LlamaConfig):
      super().__init__(config)
      self.padding_idx = config.pad_token_id
      self.vocab_size = config.vocab_size
      self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
      self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
      self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

      self.gradient_checkpointing = False
      self.post_init()

   def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):

      combined_attention_mask = None
      if input_shape[-1] > 1:
         combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length
         )
      if attention_mask is not None:
         expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(inputs_embeds.device)
         combined_attention_mask = expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask

      return combined_attention_mask

   def forward(
      self,
      input_ids = None,
      attention_mask = None,
      position_ids = None,
      past_key_values = None,
      inputs_embeds = None,
      use_cache = None,
      output_attentions = None,
      output_hidden_states = None,
      return_dict = None,
   ):
      output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
      output_hidden_states =output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
      use_cache = use_cache if use_cache is not None else self.config.use_cache
      return_dict = return_dict if return_dict is not None else self.config.return_dict

      # 针对input做判断 -> 不能都有，也不能都没有，要么是input_ids，要么是inputs_embeds
      if input_ids is not None and inputs_embeds is not None:
         raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
      elif input_ids is not None:
         batch_size, seq_length = input_ids.shape
      elif inputs_embeds is not None:
         batch_size, seq_length, _ = inputs_embeds.shape
      else:
         raise ValueError("You have to specify either input_ids or inputs_embeds")

      seq_length_with_past = seq_length
      past_key_values_length = 0

      if past_key_values is not None:
         past_key_values_length = past_key_values[0][0].shape[-2]

      if position_ids is None:
         device = input_ids.device if input_ids is not None else inputs_embeds.device
         position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
         position_ids = position_ids.unsqueeze(0)

      if inputs_embeds is None:
         inputs_embeds = self.embed_tokens(input_ids)

      if attention_mask is None:
         attention_mask = torch.one((batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device)
         padding_mask = None

      else:
         if 0 in attention_mask:
            padding_mask = attention_mask
         else:
            padding_mask = None

      attention_mask = self._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)
      hidden_states = inputs_embeds

      if self.gradient_checkpointing and self.training:
         if use_cache:
            use_cache = False

      # 根据是否要输出这些信息选择是否初始化
      all_hidden_states = () if output_hidden_states else None
      all_self_attns = () if output_attentions else None
      next_decoder_cache = () if use_cache else None

      for idx, decoder_layer in enumerate(self.layers):
         if output_hidden_states:
            all_hidden_states += (hidden_states, )
         # 这里的idx代表着时间步，代表当前时刻的key_value
         past_key_value = past_key_value[idx] if past_key_value is not None else None

         layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
         )

         hidden_states = layer_outputs[0]
         if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
         if output_attentions:
            all_self_attns += (layer_outputs[1],)

      hidden_states = self.norm(hidden_states)

      if output_hidden_states:
         all_hidden_states += (hidden_states,)

      next_cache = next_decoder_cache if use_cache else None

      if not return_dict:
         return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

      return BaseModelOutputWithPast(
         last_hidden_state=hidden_states,
         past_key_values=next_cache,
         hidden_states=all_hidden_states,
         attentions=all_self_attns,
      )


class LlamaForCasualLM(LlamaPreTrainedModel):
   def __init__(self, config):
      super().__init__(config)
      self.model = LlamaModel(config)
      self.vocab_size = config.vocab_size
      self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias = False)
      self.post_init()

   def forward(
      self,
      input_ids = None,
      attention_mask = None,
      position_ids = None,
      past_key_value = None,
      inputs_embeds = None,
      labels = None,
      use_cache = None,
      output_attentions = None,
      output_hidden_states = None,
      return_dict = None,
   ):
      output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
      output_hidden_states =output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
      return_dict = return_dict if return_dict is not None else self.config.return_dict

      outputs = self.model(
         input_ids=input_ids,
         attention_mask=attention_mask,
         position_ids=position_ids,
         past_key_value=past_key_value,
         inputs_embeds=inputs_embeds,
         use_cache=use_cache,
         output_attentions=output_attentions,
         output_hidden_states=output_hidden_states,
         return_dict=return_dict,
      )
      hidden_states = outputs[0]
      logits = self.lm_head(hidden_states)
      logits = logits.float()

      loss = None
      if labels is not None:
         shift_logits = logits[..., :-1, :].contiguous()
         shift_labels = labels[..., 1:].contiguous()

         loss_fct = CrossEntropyLoss()
         shift_logits = shift_logits.view(-1, self.config.vocab_size)
         shift_labels = shift_labels.view(-1)

         shift_labels = shift_labels.to(shift_logits.device)
         loss = loss_fct(shift_logits, shift_labels)

      if not return_dict:
         output = (logits, ) + output[1:]
         return (loss, ) + output if loss is not None else output

      return CausalLMOutputWithPast(
         loss=loss,
         logits=logits,
         past_key_values=output.past_key_values,
         hidden_states=output.hidden_states,
         attentions=outputs.attentions,
      )


class LlamaForSequenceClassification(LlamaPreTrainedModel):
   def __init__(self, config):
      super().__init__(config)
      self.num_labels = config.num_labels
      self.model = LlamaModel(config)
      self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

   def forward(
      self,
      input_ids=None,
      attention_mask=None,
      position_ids=None,
      past_key_values=None,
      inputs_embeds=None,
      labels=None,
      use_cache=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
   ):
      output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
      output_hidden_states =output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
      return_dict = return_dict if return_dict is not None else self.config.return_dict

      outputs = self.model(
         input_ids,
         attention_mask=attention_mask,
         position_ids=position_ids,
         past_key_values=past_key_values,
         inputs_embeds=inputs_embeds,
         use_cache=use_cache,
         output_attentions=output_attentions,
         output_hidden_states=output_hidden_states,
         return_dict=return_dict,
      )
      hidden_states = outputs[0]
      logits = self.score(hidden_states)

      if input_ids is not None:
         batch_size = input_ids.shape[0]
      else:
         batch_size = inputs_embeds.shape[0]

      if self.config.pad_token_id is None and batch_size != 1:
         raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
      if self.config.pad_token_id is None:
         sequence_lengths = -1
      else:
         if input_ids is not None:
               sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                  logits.device
               )
         else:
               sequence_lengths = -1

      # batch_size * sequence_lengths
      pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
      loss = None
      if labels is not None:
         labels = labels.to(logits.device)
         if self.config.problem_type is None:
            if self.num_labels == 1:
               self.config.problem_type = "regression"
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
               self.config.problem_type = "single_label_classification"
            else:
               self.config.problem_type = "multi_label_classification"

         if self.config.problem_type == "regression":
            loss_fct = MSELoss()
            if self.num_labels == 1:
               loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
            else:
               loss = loss_fct(pooled_logits, labels)

         elif self.config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))

         elif self.config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(pooled_logits, labels)

      if not return_dict:
         output = (pooled_logits,) + outputs[1:]
         return ((loss,) + output) if loss is not None else output
      # xxw 决定如何返回
      return SequenceClassifierOutputWithPast(
         loss=loss,
         logits=pooled_logits,
         past_key_values=outputs.past_key_values,
         hidden_states=outputs.hidden_states,
         attentions=outputs.attentions,
      )


class MyCausalLM(LlamaForCasualLM):
   def forward(
           self,
           input_ids,
           attention_mask,
           position_ids,
           past_key_values,
           inputs_embeds,
           labels,
           use_cache,
           output_attentions,
           output_hidden_states,
           return_dict):
      output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
      output_hidden_states = (
         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
      )
      return_dict = return_dict if return_dict is not None else self.config.use_return_dict

      outputs = self.model(
         input_ids=input_ids,
         attention_mask=attention_mask,
         position_ids=position_ids,
         past_key_values=past_key_values,
         inputs_embeds=inputs_embeds,
         use_cache=use_cache,
         output_attentions=output_attentions,
         output_hidden_states=output_hidden_states,
         return_dict=return_dict,
      )

      hidden_states = outputs[0]
      if self.config.pretraining_tp > 1:
         lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
         logits = [nn.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
         logits = torch.cat(logits, dim=-1)
      else:
         logits = self.lm_head(hidden_states)
      logits = logits.float()

      loss = None
      # xxw add new z_loss
      if labels is not None:
         # Shift so that tokens < n predict n
         shift_logits = logits[..., :-1, :].contiguous()
         shift_labels = labels[..., 1:].contiguous()
         # Flatten the tokens
         loss_fct = CrossEntropyLoss()
         shift_logits = shift_logits.view(-1, self.config.vocab_size)
         shift_labels = shift_labels.view(-1)
         softmax_normalizer = shift_logits.max(-1).values ** 2

         sorted_indices = torch.argsort(shift_logits, dim=-1)
         second_largest_index = sorted_indices[:, -2]

         second_largest_value = torch.gather(shift_logits, dim=-1, index=second_largest_index.unsqueeze(-1))
         second_largest_value = torch.squeeze(second_largest_value)

         z_loss = 2 * 10 ** (-4) * (softmax_normalizer.mean() - second_largest_value.mean())
         z_loss1 = 2 * 10 ** (-4) * (softmax_normalizer.mean())
         print("z_loss:", z_loss)
         print("z_loss1:", z_loss1)
         # Enable model parallelism
         shift_labels = shift_labels.to(shift_logits.device)
         print("ori_loss:", loss_fct(shift_logits, shift_labels))
         loss = loss_fct(shift_logits, shift_labels) + z_loss

      if not return_dict:
         output = (logits,) + outputs[1:]
         return (loss,) + output if loss is not None else output

      return CausalLMOutputWithPast(
         loss=loss,
         logits=logits,
         past_key_values=outputs.past_key_values,
         hidden_states=outputs.hidden_states,
         attentions=outputs.attentions,
      )


# longlora
class MyAttention(LlamaAttention):
    def forward(
         self,
         hidden_states,
         attention_mask=None,
         position_ids=None,
         past_key_value=None,
         output_attentions=False,
         use_cache=False,
         padding_mask = None):

        bsz, q_len, _ = hidden_states.size()
        group_size_ratio = 0.25
        group_size = int(q_len * group_size_ratio)

        if q_len % group_size > 0:
            raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
        num_group = q_len // group_size

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [nn.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [nn.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [nn.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            # xxw 得到qkv
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # xxw qkv变换维度 -> MQA / GQA
        # 使用view函数进行多头机制的拆分
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        print("ori query_states shape:", query_states.shape)
        print("ori key_states shape:", key_states.shape)
        print("ori value_states shape:", value_states.shape)


        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        # xxw 得到cosmθ, sinmθ矩阵
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # xxw 利用cosmθ+sinmθ矩阵，将相对位置信息注入到qk中
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # xxw 是否用了之前状态的kv
        # 用的话可以加速推理
        if past_key_value is not None:
            # xxw 前面的k加到当前的key_states上，得到当前时刻的key_states
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            # xxw 前面的v加到当前的key_states上，得到当前时刻的value_states
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        # xxw -> attention中使用的kv_cache
        past_key_value = (key_states, value_states) if use_cache else None

        # xxw 重复多组kv，实现Group Query Attention
        # 当self.num_key_value_groups=1时，相当于没变
        # 当self.num_key_value_groups=2时，num_heads的数量x2
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):
            # roll函数 -> 沿着给定轴滚动数组元素。超出最后位置的元素将会滚动到第一个位置
            qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(-group_size // 2, dims=2)
            # 将q_len分成4组,每一组的维度是group_size
            qkv = qkv.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim).transpose(1, 2)
            return qkv

        # 经过滚动处理的qkv
        query_states = shift(query_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
        key_states = shift(key_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
        value_states = shift(value_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
        print("new query_states shape:", query_states.shape)
        print("new key_states shape:", key_states.shape)
        print("new value_states shape:", value_states.shape)

        # xxw 最后计算attn_weights = q * k / √dk
        # [bsz, num_heads，q_len, head_dim]*[bsz, num_heads, head_dim, q_len]=[bsz, num_heads，q_len, q_len]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz * num_group, self.num_heads, group_size, group_size):
        #if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        attention_mask = attention_mask[:, :, :group_size, :group_size].repeat(num_group, 1, 1, 1)
        # xxw attn_weights叠加上attention_mask
        if attention_mask is not None:
            if attention_mask.size() != (bsz * num_group, 1, group_size, group_size):
            #if attention_mask.size() != (bsz, 1, q_len, kv_seq_len)dd:
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            # xxw 注意力做attention_mask
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        # xxw 做softmax将权重变为0-1
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # 最后和v相乘得到概率值
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz * num_group, self.num_heads, group_size, self.head_dim):
        #if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # xxw 修改维度
        attn_output = attn_output.transpose(1, 2).contiguous()
        # xxw 恢复形状
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # shift back
        attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            # xxw 最后再做一次线性变换
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value




