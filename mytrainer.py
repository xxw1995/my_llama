import os
import torch
from transformers import Seq2SeqTrainer
from transformers.trainer import TRAINING_ARGS_NAME, WEIGHTS_NAME, WEIGHTS_INDEX_NAME
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from peft import PeftModel
from trl import PreTrainedModelWrapper
from transformers.modeling_utils import load_sharded_checkpoint


FINETUNING_ARGS_NAME = "finetuning_args.json"
VALUE_HEAD_FILE_NAME = "value_head.bin"


def get_state_dict(model):
    state_dict = model.state_dict()
    filtered_state_dict = {}

    for k, v in model.named_parameters():
        if v.requires_grad:
            filtered_state_dict[k] = state_dict[k].cpu().clone().detach()

    return filtered_state_dict


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


class mytrainer(Seq2SeqTrainer):

    def __init__(self, finetuning_args, **kwargs):
        Seq2SeqTrainer.__init__(self, **kwargs)
        self.finetuning_args = finetuning_args,
        self.model = None
        self.tokenizer = None
        self.args = None
        self.state = None

    def _save(self, output_dir = None, state_dict = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model = unwrap_model(self.model)
        if isinstance(model, PreTrainedModelWrapper):
            model_state_dict = state_dict or model.state_dict()
            v_head_state_dict = {
                name.replace("v_head.", ""): model_state_dict[name].cpu().clone().detach()
                for name in model_state_dict.keys() if name.startswith("v_head.")
            }
            torch.save(v_head_state_dict, os.path.join(output_dir, VALUE_HEAD_FILE_NAME))
            model = model.pretrained_model

        state_dict = state_dict or get_state_dict(model)
        if isinstance(model, (PeftModel, PreTrainedModel)):
            model.config.use_cache = True
            model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors)
            model.config.use_cache = False
        else:
            torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        if self.finetuning_args.finetuning_type == "full" and self.tokenizer is not None:
            try:
                self.tokenizer.save_pretrained(output_dir)
            except:
                print("Cannot save tokenizer, copy the files manually.")

        with open(os.path.join(output_dir, TRAINING_ARGS_NAME), "w", encoding="utf-8") as f:
            f.write(self.args.to_json_string() + "\n")

        self.finetuning_args.save_to_json(os.path.join(output_dir, FINETUNING_ARGS_NAME))

    def _load_best_model(self):
        model = unwrap_model(self.model)
        if isinstance(model, PreTrainedModelWrapper):
            model.v_head.load_state_dict(torch.load(
                os.path.join(self.state.best_model_checkpoint, VALUE_HEAD_FILE_NAME), map_location="cpu"
            ))
            model = model.pretrained_model

        if isinstance(model, PeftModel):
            model.load_adapter(self.state.best_model_checkpoint, model.active_adapter)
        else:  # freeze/full-tuning
            load_trainable_params(model, self.state.best_model_checkpoint)
