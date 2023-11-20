"""
Author: xxw
Date: 2023-11-20 01:15:25
LastEditTime: 2023-11-20 07:00:58
"""
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
   """Rotates half the hidden dims of the input."""
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



