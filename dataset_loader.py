import os
from datasets import load_dataset
import tiktoken
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
from itertools import chain
from argument import ModelArguments, DataArguments
from dataclasses import dataclass


IGNORE_INDEX = -100

EXT2TYPE = {
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "txt": "text"
}

def split_dataset(dataset, data_args, training_args):
    """
    分割数据集
    """
    if training_args.do_train:
        if data_args.val_size > 1e-6: # Split the dataset
            if data_args.streaming:
                val_set = dataset.take(int(data_args.val_size))
                train_set = dataset.skip(int(data_args.val_size))
                return {"train_dataset": train_set, "eval_dataset": val_set}
            else:
                val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
                dataset = dataset.train_test_split(test_size=val_size, seed=training_args.seed)
                return {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
        else:
            if data_args.streaming:
                dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
            return {"train_dataset": dataset}
    else: # do_eval or do_predict
        return {"eval_dataset": dataset}


def get_dataset(model_args: "ModelArguments", data_args: "DataArguments"):
    """
    获取数据集，加入all_datasets
    """
    max_samples = data_args.max_samples
    all_datasets = [] # support multiple datasets

    for dataset_attr in data_args.dataset_list:
        print("Loading dataset {}...".format(dataset_attr))
        if dataset_attr.load_from == "file":
            data_path = None
            data_files = [] # 数据append到data_files中
            if os.path.isdir(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)): # directory
                for file_name in os.listdir(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)):
                    data_files.append(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name, file_name))
                    if data_path is None:
                        data_path = EXT2TYPE.get(file_name.split(".")[-1], None)
                    else:
                        assert data_path == EXT2TYPE.get(file_name.split(".")[-1], None), "file type does not match."
            elif os.path.isfile(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)): # single file
                data_files.append(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name))
                data_path = EXT2TYPE.get(dataset_attr.dataset_name.split(".")[-1], None)
            else:
                raise ValueError("File not found.")

        # 调用load_dataset
        dataset = load_dataset(
            data_path,
            data_files=data_files,
            split=data_args.split,
            cache_dir=model_args.cache_dir,
            streaming=data_args.streaming,
            use_auth_token=True if model_args.use_auth_token else None
        )

        if max_samples is not None:
            max_samples_temp = min(len(dataset), max_samples)
            dataset = dataset.select(range(max_samples_temp))

        for column_name in ["prompt", "query", "response", "history"]: # align datasets
            if getattr(dataset_attr, column_name) and getattr(dataset_attr, column_name) != column_name:
                dataset = dataset.rename_column(getattr(dataset_attr, column_name), column_name)

        if dataset_attr.system_prompt: # add system prompt
            if data_args.streaming:
                dataset = dataset.map(lambda _: {"system": dataset_attr.system_prompt})
            else:
                dataset = dataset.add_column("system", [dataset_attr.system_prompt] * len(dataset))
        # 最终维护一个all_datasets
        all_datasets.append(dataset)

    if len(data_args.dataset_list) == 1:
        return all_datasets[0]


###################
# 数据格式处理
###################
@dataclass
class Template:

    prefix: List[Union[str, Dict[str, str]]]
    prompt: List[Union[str, Dict[str, str]]]
    system: str
    sep: List[Union[str, Dict[str, str]]]
    stop_words: List[str]
    use_history: bool

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        system, history = self._format(query, resp, history, system)
        encoded_pairs = self._encode(tokenizer, system, history)
        prompt_ids = []
        for query_ids, resp_ids in encoded_pairs[:-1]:
            prompt_ids = prompt_ids + query_ids + resp_ids
        prompt_ids, answer_ids = prompt_ids + encoded_pairs[-1][0], encoded_pairs[-1][1]
        return prompt_ids, answer_ids

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        system, history = self._format(query, resp, history, system)
        encoded_pairs = self._encode(tokenizer, system, history)
        return encoded_pairs

    def _format(
        self,
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None
    ) -> Tuple[str, List[Tuple[str, str]]]:
        r"""
        Aligns inputs to the standard format.
        """
        system = system or self.system # use system if provided
        history = history if (history and self.use_history) else []
        history = history + [(query, resp)]
        return system, history

    def _get_special_ids(
        self,
        tokenizer: "PreTrainedTokenizer"
    ) -> Tuple[List[int], List[int]]:
        if (
            tokenizer.bos_token_id is not None
            and getattr(tokenizer, "add_bos_token", True)
        ): # baichuan-13b has no bos token
            bos_ids = [tokenizer.bos_token_id]
        else:
            bos_ids = [] # bos token is optional

        if tokenizer.eos_token_id is not None:
            eos_ids = [tokenizer.eos_token_id]
        else:
            raise ValueError("EOS token is required.")

        return bos_ids, eos_ids

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        system: str,
        history: List[Tuple[str, str]]
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: bos + prefix + sep + query    resp + eos
        Turn t: sep + bos + query             resp + eos
        """
        bos_ids, eos_ids = self._get_special_ids(tokenizer)
        sep_ids = self._convert_inputs_to_ids(tokenizer, context=self.sep)
        encoded_pairs = []
        for turn_idx, (query, resp) in enumerate(history):
            if turn_idx == 0:
                prefix_ids = self._convert_inputs_to_ids(tokenizer, context=self.prefix, system=system)
                if len(prefix_ids) != 0: # has prefix
                    prefix_ids = bos_ids + prefix_ids + sep_ids
                else:
                    prefix_ids = bos_ids
            else:
                prefix_ids = sep_ids + bos_ids

            query_ids = self._convert_inputs_to_ids(tokenizer, context=self.prompt, query=query, idx=str(turn_idx))
            resp_ids = self._convert_inputs_to_ids(tokenizer, context=[resp])
            encoded_pairs.append((prefix_ids + query_ids, resp_ids + eos_ids))
        return encoded_pairs

    def _convert_inputs_to_ids(
        self,
        tokenizer: "PreTrainedTokenizer",
        context: List[Union[str, Dict[str, str]]],
        system: Optional[str] = None,
        query: Optional[str] = None,
        idx: Optional[str] = None
    ) -> List[int]:
        r"""
        Converts context to token ids.
        """
        if isinstance(getattr(tokenizer, "tokenizer", None), tiktoken.Encoding): # for tiktoken tokenizer (Qwen)
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=False)

        token_ids = []
        for elem in context:
            if isinstance(elem, str):
                elem = elem.replace("{{system}}", system, 1) if system is not None else elem
                elem = elem.replace("{{query}}", query, 1) if query is not None else elem
                elem = elem.replace("{{idx}}", idx, 1) if idx is not None else elem
                token_ids = token_ids + tokenizer.encode(elem, **kwargs)
            elif isinstance(elem, dict):
                token_ids = token_ids + [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            else:
                raise NotImplementedError

        return token_ids

@dataclass
class Llama2Template(Template):

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        system: str,
        history: List[Tuple[str, str]]
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: bos + prefix + query    resp + eos
        Turn t: bos + query             resp + eos
        """
        bos_ids, eos_ids = self._get_special_ids(tokenizer)
        encoded_pairs = []
        for turn_idx, (query, resp) in enumerate(history):
            if turn_idx == 0: # llama2 template has no sep_ids
                query = self.prefix[0].replace("{{system}}", system) + query
            query_ids = self._convert_inputs_to_ids(tokenizer, context=self.prompt, query=query)
            resp_ids = self._convert_inputs_to_ids(tokenizer, context=[resp])
            encoded_pairs.append((bos_ids + query_ids, resp_ids + eos_ids))
        return encoded_pairs


templates: Dict[str, Template] = {}


def register_template(
    name: str,
    prefix: List[Union[str, Dict[str, str]]],
    prompt: List[Union[str, Dict[str, str]]],
    system: str,
    sep: List[Union[str, Dict[str, str]]],
    stop_words: Optional[List[str]] = [],
    use_history: Optional[bool] = True
) -> None:
    template_class = Llama2Template if "llama2" in name else Template
    templates[name] = template_class(
        prefix=prefix,
        prompt=prompt,
        system=system,
        sep=sep,
        stop_words=stop_words,
        use_history=use_history
    )

r"""
Default template.
"""
register_template(
    name="default",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        "Human: {{query}}\nAssistant: "
    ],
    system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    sep=[
        "\n"
    ]
)

def get_template_and_fix_tokenizer(
    name: str,
    tokenizer: "PreTrainedTokenizer"
) -> Template:
    template = templates.get(name, None)
    assert template is not None, "Template {} does not exist.".format(name)

    additional_special_tokens = template.stop_words
    if len(template.stop_words): # inplace method
        if tokenizer.eos_token_id is not None:
            additional_special_tokens.append(tokenizer.eos_token)

        tokenizer.eos_token = additional_special_tokens[0] # use the first stop word as eos token
        additional_special_tokens.pop(0)
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "<|endoftext|>"
        logger.info("Add eos token: {}".format(tokenizer.eos_token))

    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    tokenizer.add_special_tokens(
        dict(additional_special_tokens=additional_special_tokens),
        replace_additional_special_tokens=False
    )
    return template


# stop_words eos pad ...
def get_template_and_fix_tokenizer(name, tokenizer):
    template = templates.get(name, None)
    print("templates:", templates)
    assert template is not None, "Template {} does not exist.".format(name)

    additional_special_tokens = template.stop_words
    if len(template.stop_words): # inplace method
        if tokenizer.eos_token_id is not None:
            additional_special_tokens.append(tokenizer.eos_token)

        tokenizer.eos_token = additional_special_tokens[0] # use the first stop word as eos token
        additional_special_tokens.pop(0)
        print("Replace eos token: {}".format(tokenizer.eos_token))

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "<|endoftext|>"
        print("Add eos token: {}".format(tokenizer.eos_token))

    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        print("Add pad token: {}".format(tokenizer.pad_token))

    tokenizer.add_special_tokens(
        dict(additional_special_tokens=additional_special_tokens),
        replace_additional_special_tokens=False
    )
    return template

# 利用tokenizer对dataset进行处理
def preprocess_dataset(dataset, tokenizer, data_args, training_args, stage):
    column_names = list(next(iter(dataset)).keys())
    print("data_args.template:", data_args.template)
    template = get_template_and_fix_tokenizer(data_args.template, tokenizer)
    print("data_args.template:", data_args.template)

    def construct_example(examples):
        for i in range(len(examples["prompt"])):
            query, response = examples["prompt"][i], examples["response"][i]
            query = query + "\n" + examples["query"][i] if "query" in examples and examples["query"][i] else query
            history = examples["history"][i] if "history" in examples else None
            system = examples["system"][i] if "system" in examples else None
            yield query, response, history, system

    def preprocess_pretrain_dataset(examples):
        # build grouped texts with format `X1 X2 X3 ...` (without <eos>)
        if isinstance(getattr(tokenizer, "tokenizer", None), tiktoken.Encoding): # for tiktoken tokenizer (Qwen)
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=False)

        tokenized_examples = tokenizer(examples["prompt"], **kwargs)
        print("xxw tokenized_examples", tokenized_examples)
        concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        block_size = data_args.max_source_length
        # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        total_length = (total_length // block_size) * block_size
        # split by chunks of max_source_length
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    def preprocess_supervised_dataset(examples):
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        max_length = data_args.max_source_length + data_args.max_target_length

        for query, response, history, system in construct_example(examples):
            input_ids, labels = [], []
            for source_ids, target_ids in template.encode_multiturn(tokenizer, query, response, history, system):
                if len(source_ids) > data_args.max_source_length:
                    source_ids = source_ids[:data_args.max_source_length]
                if len(target_ids) > data_args.max_target_length:
                    target_ids = target_ids[:data_args.max_target_length]

                if len(input_ids) + len(source_ids) + len(target_ids) > max_length:
                    break

                input_ids += source_ids + target_ids
                labels += [IGNORE_INDEX] * len(source_ids) + target_ids

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs

    def preprocess_unsupervised_dataset(examples):
        # build inputs with format `<bos> X` and labels with format `Y <eos>`
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for query, response, history, system in construct_example(examples):
            source_ids, target_ids = template.encode_oneturn(tokenizer, query, response, history, system)

            if len(source_ids) > data_args.max_source_length:
                source_ids = source_ids[:data_args.max_source_length]
            if len(target_ids) > data_args.max_target_length:
                target_ids = target_ids[:data_args.max_target_length]

            model_inputs["input_ids"].append(source_ids)
            model_inputs["attention_mask"].append([1] * len(source_ids))
            model_inputs["labels"].append(target_ids)

        return model_inputs

    def preprocess_pairwise_dataset(examples):
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = {"prompt_ids": [], "chosen_ids": [], "rejected_ids": []}
        for query, response, history, system in construct_example(examples):
            prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, query, response[0], history, system)
            _, rejected_ids = template.encode_oneturn(tokenizer, query, response[1], history, system)

            if len(prompt_ids) > data_args.max_source_length:
                prompt_ids = prompt_ids[:data_args.max_source_length]
            if len(chosen_ids) > data_args.max_target_length:
                chosen_ids = chosen_ids[:data_args.max_target_length]
            if len(rejected_ids) > data_args.max_target_length:
                rejected_ids = rejected_ids[:data_args.max_target_length]

            model_inputs["prompt_ids"].append(prompt_ids)
            model_inputs["chosen_ids"].append(chosen_ids)
            model_inputs["rejected_ids"].append(rejected_ids)
        return model_inputs

    def print_supervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(tokenizer.decode([
            token_id if token_id != IGNORE_INDEX else tokenizer.pad_token_id for token_id in example["labels"]
        ], skip_special_tokens=False)))

    def print_pairwise_dataset_example(example):
        print("prompt_ids:\n{}".format(example["prompt_ids"]))
        print("prompt:\n{}".format(tokenizer.decode(example["prompt_ids"], skip_special_tokens=False)))
        print("chosen_ids:\n{}".format(example["chosen_ids"]))
        print("chosen:\n{}".format(tokenizer.decode(example["chosen_ids"], skip_special_tokens=False)))
        print("rejected_ids:\n{}".format(example["rejected_ids"]))
        print("rejected:\n{}".format(tokenizer.decode(example["rejected_ids"], skip_special_tokens=False)))

    def print_unsupervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))

    if stage == "pt":
        dataset = dataset.filter(lambda example: example["prompt"])
        preprocess_function = preprocess_pretrain_dataset
        print_function = print_unsupervised_dataset_example
    elif stage == "sft" and not training_args.predict_with_generate:
        dataset = dataset.filter(lambda example: example["prompt"] and example["response"])
        preprocess_function = preprocess_supervised_dataset
        print_function = print_supervised_dataset_example
    elif stage == "rm":
        dataset = dataset.filter(lambda example: example["prompt"] and len(example["response"]) > 1)
        preprocess_function = preprocess_pairwise_dataset
        print_function = print_pairwise_dataset_example
    else:
        dataset = dataset.filter(lambda example: example["prompt"])
        preprocess_function = preprocess_unsupervised_dataset
        print_function = print_unsupervised_dataset_example

    with training_args.main_process_first(desc="dataset map pre-processing"):
        kwargs = {}
        if not data_args.streaming:
            kwargs = dict(
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset"
            )

        dataset = dataset.map(
            preprocess_function,
            batched=True,            
            remove_columns=column_names,
            **kwargs
        )

        print_function(next(iter(dataset)))
        return dataset


##########
# 还没用的register_template
##########
r"""
Supports: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
          https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
          https://huggingface.co/meta-llama/Llama-2-70b-chat-hf
"""
register_template(
    name="llama2",
    prefix=[
        "<<SYS>>\n{{system}}\n<</SYS>>\n\n"
    ],
    prompt=[
        "[INST] {{query}} [/INST] "
    ],
    system=(
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe.  "
        "Your answers should not include any harmful, unethical, "
        "racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n"
        "If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information."
    ),
    sep=[]
)



r"""
Supports: https://github.com/ymcui/Chinese-LLaMA-Alpaca-2
          https://huggingface.co/ziqingyang/chinese-alpaca-2-7b
"""
register_template(
    name="llama2_zh",
    prefix=[
        "<<SYS>>\n{{system}}\n<</SYS>>\n\n"
    ],
    prompt=[
        "[INST] {{query}} [/INST] "
    ],
    system="You are a helpful assistant. 你是一个乐于助人的助手。",
    sep=[]
)


r"""
Supports: https://huggingface.co/tatsu-lab/alpaca-7b-wdiff
          https://github.com/ymcui/Chinese-LLaMA-Alpaca
"""
register_template(
    name="alpaca",
    prefix=[
        "{{system}}"
    ],
    prompt=[
        "### Instruction:\n{{query}}\n\n### Response:\n"
    ],
    system=(
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    ),
    sep=[
        "\n\n"
    ]
)

r"""
Supports: https://huggingface.co/THUDM/chatglm2-6b
"""
register_template(
    name="chatglm2",
    prefix=[
        {"token": "[gMASK]"},
        {"token": "sop"},
        "{{system}}"
    ],
    prompt=[
        "[Round {{idx}}]\n\n问：{{query}}\n\n答："
    ],
    system="",
    sep=[
        "\n\n"
    ]
)
