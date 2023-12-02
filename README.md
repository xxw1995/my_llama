## 前期准备
git clone https://github.com/xxw1995/my_llama  
pip3 install -r requirements.txt  
下载模型到meta-llama/Llama-2-7b-hf  
下载skywork预训练数据  
  
## 基于Llama-2-7b-hf拓展词表 & 合并得到llama-2-7b-extent
python3 my_llama_tokenizer.py

## 基于llama-2-7b-extent继续预训练
python pretrain_llama.py    
                              --stage pt \
                             --model_name_or_path meta-llama/llama-2-7b-extent \
                             --do_train --dataset 2021-49_zh_head_000x \
                             --template default \
                             --finetuning_type lora \
                             --lora_target q_proj,v_proj \
                             --output_dir path_to_pt_mytokenizer_myllama  \
                             --overwrite_cache \
                             --per_device_train_batch_size 1 \
                             --gradient_accumulation_steps 1 \
                             --lr_scheduler_type cosine \
                             --logging_steps 10 \
                             --save_steps 10000 \
                             --learning_rate 5e-5 \
                             --num_train_epochs 3.0 \
                             --plot_loss \
                             --fp16 \
                             --gpu_info GPU_INFO \
                             --quantization_bit 4

