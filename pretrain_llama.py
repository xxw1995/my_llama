from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers import DataCollatorForLanguageModeling

from argument import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments, GeneralArguments
from dataset_loader import get_dataset, preprocess_dataset, split_dataset
from load_tokenizer_model import load_model_and_tokenizer
from xxwTrainer import xxwTrainer

model_args, data_args, \
training_args, finetuning_args, \
generating_args, general_args = HfArgumentParser((
                                  ModelArguments,
                                  DataArguments,
                                  Seq2SeqTrainingArguments,
                                  FinetuningArguments,
                                  GeneratingArguments,
                                  GeneralArguments)).parse_args_into_dataclasses()

dataset = get_dataset(model_args, data_args) # 获取数据集
model, tokenizer = load_model_and_tokenizer(model_args, data_args, training_args, finetuning_args, training_args.do_train, stage="pt") # 获取model tokenizer
dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="pt") # 利用tokenizer处理数据集
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # 后处理data_collator

# trainer
trainer = xxwTrainer(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        **split_dataset(dataset, data_args, training_args))

# Training
if training_args.do_train:
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model()

