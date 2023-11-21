import os
import torch

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase
)

from peft import (
    PeftModel,
    TaskType,
    LoraConfig,
    get_peft_model
)

from peft.utils import CONFIG_NAME, WEIGHTS_NAME
import argparse
import bitsandbytes as bnb
from transformers.deepspeed import is_deepspeed_zero3_enabled
from trl import AutoModelForCausalLMWithValueHead
from transformers.models.llama import modeling_llama_easy as LlamaModule
from transformers.models.llama.modeling_llama_easy import MyCausalLM, MyAttention
from transformers.trainer import WEIGHTS_NAME, WEIGHTS_INDEX_NAME
from transformers.modeling_utils import load_sharded_checkpoint

VALUE_HEAD_FILE_NAME = "value_head.bin"


# xxw add
def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


def init_adapter(model, model_args, data_args, training_args, finetuning_args, is_trainable, is_mergeable):
    if finetuning_args.finetuning_type == "none" and is_trainable:
        raise ValueError("You cannot use finetuning_type=none while training.")

    if finetuning_args.finetuning_type == "full" and is_trainable:
        print("Fine-tuning method: Full")
        model = model.float()

    if finetuning_args.finetuning_type == "lora":
        print("Fine-tuning method: LoRA")
        latest_checkpoint = None

        if model_args.checkpoint_dir is not None:
            assert os.path.exists(os.path.join(model_args.checkpoint_dir[0], WEIGHTS_NAME)), \
                "Provided path ({}) does not contain a LoRA weight.".format(model_args.checkpoint_dir[0])
            assert os.path.exists(os.path.join(model_args.checkpoint_dir[0], CONFIG_NAME)), \
                "The given checkpoint may be not a LoRA checkpoint, please specify\
                 `--finetuning_type full/freeze` instead."

            if (is_trainable and finetuning_args.resume_lora_training) or (not is_mergeable): # continually fine-tuning
                checkpoints_to_merge, latest_checkpoint = model_args.checkpoint_dir[:-1], model_args.checkpoint_dir[-1]
            else:
                checkpoints_to_merge = model_args.checkpoint_dir

            for checkpoint in checkpoints_to_merge:
                model = PeftModel.from_pretrained(model, checkpoint)
                model = model.merge_and_unload()

            if len(checkpoints_to_merge) > 0:
                print("Merged {} model checkpoint(s).".format(len(checkpoints_to_merge)))

            if latest_checkpoint is not None: # resume lora training or quantized inference
                model = PeftModel.from_pretrained(model, latest_checkpoint, is_trainable=is_trainable)

        if is_trainable and latest_checkpoint is None: # create new lora weights while training
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=finetuning_args.lora_rank,
                lora_alpha=finetuning_args.lora_alpha,
                lora_dropout=finetuning_args.lora_dropout,
                target_modules=finetuning_args.lora_target
            )
            model = get_peft_model(model, lora_config)

    # xxw add all linear lora
    if finetuning_args.finetuning_type == "lora_all_linear":
        print("Fine-tuning method: LoRA")
        latest_checkpoint = None
        args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))

        if model_args.checkpoint_dir is not None:
            assert os.path.exists(os.path.join(model_args.checkpoint_dir[0], WEIGHTS_NAME)), \
                "Provided path ({}) does not contain a LoRA weight.".format(model_args.checkpoint_dir[0])
            assert os.path.exists(os.path.join(model_args.checkpoint_dir[0], CONFIG_NAME)), \
                "The given checkpoint may be not a LoRA checkpoint, please specify \
                `--finetuning_type full/freeze` instead."

            if (is_trainable and finetuning_args.resume_lora_training) or (not is_mergeable): # continually fine-tuning
                checkpoints_to_merge, latest_checkpoint = model_args.checkpoint_dir[:-1], model_args.checkpoint_dir[-1]
            else:
                checkpoints_to_merge = model_args.checkpoint_dir

            for checkpoint in checkpoints_to_merge:
                model = PeftModel.from_pretrained(model, checkpoint)
                model = model.merge_and_unload()

            if len(checkpoints_to_merge) > 0:
                print("Merged {} model checkpoint(s).".format(len(checkpoints_to_merge)))

            if latest_checkpoint is not None: # resume lora training or quantized inference
                model = PeftModel.from_pretrained(model, latest_checkpoint, is_trainable=is_trainable)

        modules = find_all_linear_names(args, model)
        if is_trainable and latest_checkpoint is None: # create new lora weights while training
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=finetuning_args.lora_rank,
                lora_alpha=finetuning_args.lora_alpha,
                lora_dropout=finetuning_args.lora_dropout,
                target_modules=modules
            )
            model = get_peft_model(model, lora_config)

    if model_args.checkpoint_dir is not None:
        print("Loaded fine-tuned model from checkpoint(s): {}".format(",".join(model_args.checkpoint_dir)))
    return model


def load_trainable_params(model, checkpoint_dir):
    weights_file = os.path.join(checkpoint_dir, WEIGHTS_NAME)
    if os.path.exists(weights_file):
        model_state_dict = torch.load(weights_file, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False) # skip missing keys
    elif os.path.exists(os.path.join(checkpoint_dir, WEIGHTS_INDEX_NAME)):
        load_sharded_checkpoint(model, checkpoint_dir, strict=False)
    else:
        print("Provided path ({}) does not contain pre-trained weights.".format(checkpoint_dir))
        return False
    return True


def load_valuehead_params(model, checkpoint_dir):
    valuehead_file = os.path.join(checkpoint_dir, VALUE_HEAD_FILE_NAME)
    if not os.path.exists(valuehead_file):
        print("Provided path ({}) does not contain valuehead weights.".format(checkpoint_dir))
        return False
    valuehead_state_dict = torch.load(valuehead_file, map_location="cpu")
    model.register_buffer("reward_head_weight", valuehead_state_dict["summary.weight"])
    model.register_buffer("reward_head_bias", valuehead_state_dict["summary.bias"])
    model.register_buffer("default_head_weight", torch.zeros_like(valuehead_state_dict["summary.weight"]))
    model.register_buffer("default_head_bias", torch.zeros_like(valuehead_state_dict["summary.bias"]))

    return True


def prepare_model_for_training(model, finetuning_type, output_layer_name = "lm_head", use_gradient_checkpointing = True, layer_norm_names = LAYERNORM_NAMES):
    for name, param in model.named_parameters():
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)

    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model.gradient_checkpointing_enable()
        model.config.use_cache = False # turn off when gradient checkpointing is enabled

    if finetuning_type != "full" and hasattr(model, output_layer_name):
        output_layer: torch.nn.Linear = getattr(model, output_layer_name)
        input_dtype = output_layer.weight.dtype
        class CastOutputToFloat(torch.nn.Sequential):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return super().forward(x.to(input_dtype)).to(torch.float32)
        setattr(model, output_layer_name, CastOutputToFloat(output_layer))
    return model


def count_parameters(model):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def load_model_and_tokenizer(model_args, data_args, training_args, finetuning_args, is_trainable=False, stage="pt"):

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    # print("config_kwargs:", config_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        padding_side=model_args.padding_side,
        **config_kwargs
    )

    if finetuning_args.finetuning_type == "full" and model_args.checkpoint_dir is not None: # 全量
        model_to_load = model_args.checkpoint_dir[0]

    else: # lora
        model_to_load = model_args.model_name_or_path
    config = AutoConfig.from_pretrained(model_to_load, **config_kwargs)
    # print("config:", config)

    # Set RoPE scaling
    if model_args.rope_scaling is not None:
        scaling_factor = 2.0
        setattr(config, "rope_scaling", {"type": model_args.rope_scaling, "factor": scaling_factor})
        print("Using {} scaling strategy and setting scaling factor to {}".format(model_args.rope_scaling, scaling_factor))

    # xxw add patches
    # LlamaModule.LlamaAttention = MyAttention
    LlamaModule.LlamaForCausalLM = MyCausalLM

    # Quantization configurations (using bitsandbytes library).
    is_mergeable = True
    if model_args.quantization_bit is not None:
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        if model_args.quantization_bit == 8:
            print("xxw quantization_bit 8bits")
            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        elif model_args.quantization_bit == 4:
            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_args.compute_dtype,
                bnb_4bit_use_double_quant=model_args.double_quantization,
                bnb_4bit_quant_type=model_args.quantization_type
            )

        is_mergeable = False
        config_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))} if is_trainable else "auto"
        print("Quantizing model to {} bit.".format(model_args.quantization_bit))

    # xxw core -> full 微调的时候没走量化，需要再指定一下
    config_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))} if is_trainable else "auto"

    # Load and prepare pre-trained models (without valuehead).
    model = AutoModelForCausalLM.from_pretrained(
        model_to_load,
        config=config,
        torch_dtype=model_args.compute_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        **config_kwargs
    )
    # xxw add
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model,)

    for i, (name, layer) in enumerate(model.named_parameters()):
        print("xxw first: name:{}, layer.dtype:{}, layer.device:{}".format(name, layer.dtype, layer.device))
    # xxw add done

    # Fix LM head (for ChatGLM2)
    if not hasattr(model, "lm_head") and hasattr(model, "transformer"):
        setattr(model, "lm_head", model.transformer.output_layer)

    # Register auto class to save the custom code files.
    if isinstance(config, PretrainedConfig) and "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if isinstance(model, PreTrainedModel) and "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if isinstance(tokenizer, PreTrainedTokenizerBase) and "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()

    # peft model
    model = prepare_model_for_training(model, finetuning_args.finetuning_type) if is_trainable else model
    model = init_adapter(model, model_args, data_args, training_args, finetuning_args, is_trainable, is_mergeable)

    # xxw add
    for i, (name, layer) in enumerate(model.named_parameters()):
        print("xxw add second: name:{}, layer.dtype:{}, layer.device:{}".format(name, layer.dtype, layer.device))

    # Prepare model with valuehead for RLHF
    if stage == "rm" or stage == "ppo":
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        if stage == "rm" and model_args.checkpoint_dir is not None: # load valuehead weights to evaluate reward model
            print("Only the last checkpoint containing valuehead will be loaded as the valuehead.")
            if load_valuehead_params(model, model_args.checkpoint_dir[-1]):
                model.v_head.load_state_dict({
                    "summary.weight": getattr(model, "reward_head_weight"),
                    "summary.bias": getattr(model, "reward_head_bias")
                })

        if stage == "ppo": # load reward model
            print("Load reward model from {}".format(model_args.reward_model))
            model.pretrained_model.load_adapter(model_args.reward_model, "reward", is_trainable=False)
            assert load_valuehead_params(model, model_args.reward_model), "Reward model is not correctly loaded."

    # Prepare model for inference
    if not is_trainable:
        model.requires_grad_(False) # fix all model params
        infer_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 # detect cuda capability
        model = model.to(infer_dtype) if model_args.quantization_bit is None else model

    trainable_params, all_param = count_parameters(model)
    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    ))

    return model, tokenizer
