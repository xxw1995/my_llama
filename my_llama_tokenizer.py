import os
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from transformers import LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

def train_spm(data_file, new_model_prefix, target_vocab_size):
    spm.SentencePieceTrainer.train(input=data_file,
                                   model_prefix=new_model_prefix,
                                   model_type="bpe",
                                   vocab_size=target_vocab_size,
                                   character_coverage=0.9995,
                                   num_threads=3,
                                   split_digits=True,
                                   byte_fallback=True,
                                   max_sentence_length=2000)

def add_vocab_2_llama(llama_tokenizer_dir, chinese_sp_model_file, output_sp_dir, output_hf_dir):

    llama_tokenizer_dir = llama_tokenizer_dir
    chinese_sp_model_file = chinese_sp_model_file

    llama_tokenizer = LlamaTokenizer.from(llama_tokenizer_dir) # len(llama_tokenizer) == 32000
    llama_spm = sp_pb2_model.ModelProto()
    llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())

    chinese_sp_model = spm.SentencePieceProcessor()
    chinese_sp_model.Load(chinese_sp_model_file) # len(chinese_sp_model) == 10000
    chinese_spm = sp_pb2_model.ModelProto()
    chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())

    """
    llama_tokenizer.all_special_tokens = ['<s>', '</s>', '<unk>']
    llama_tokenizer.all_special_ids = [1, 2, 0]
    llama_tokenizer.special_tokens_map = {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>'}
    """

    # 往llama词表里添加刚才训练chinese_spm.pieces中的词
    llama_spm_tokens_set=set(p.piece for p in llama_spm.pieces) # 32000
    for p in chinese_spm.pieces:
        piece = p.piece
        # 只有llama_spm_tokens_set中没有的（也就是刚才训练的8000词表中跟llama没有交集的）才放进来
        if piece not in llama_spm_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            llama_spm.pieces.append(new_p)
    """
    print(f"New model pieces: {len(llama_spm.pieces)}")
    38795 -> 我们中文词表原来有1万，去重添加后，添加了6759个词
    """
    # 保存合并后的模型 & tokenizer
    output_sp_dir = output_sp_dir
    output_hf_dir = output_hf_dir
    os.makedirs(output_sp_dir,exist_ok=True)

    with open(output_sp_dir+'/chinese_llama.model', 'wb') as f:
        f.write(llama_spm.SerializeToString())

    tokenizer = LlamaTokenizer(vocab_file=output_sp_dir+'/my_easy_llama.model')
    tokenizer.save_pretrained(output_hf_dir)

    print(f"my_easy_llama tokenizer has been saved to {output_hf_dir}")

def new_tokenizer(llama_tokenizer_dir, output_hf_dir, new_model_name):
    model_name = llama_tokenizer_dir
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

    # 原始tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 新的tokenizer
    new_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)

    """
    text = "麒麟，是中国古代神话中的一种瑞兽"
    print(f"Tokenized by tokenizer:{tokenizer.tokenize(text)}")
    print(f"Tokenized by new_tokenizer:{new_tokenizer.tokenize(text)}")
    """

    # 建立新增的token和在原来token相对应的字典
    token_mapping = {}
    for i in range(len(tokenizer), len(new_tokenizer)): # 32000 -> 
        token = new_tokenizer.convert_ids_to_tokens(i) # 获取当前的新token
        input_ids = tokenizer(token, return_tensors="pt").input_ids[0] # 用老的tokenizer，把新出现的token转换成input_ids（用于初始化）

        if input_ids[1] == 29871:
            new_input_ids = input_ids[2:]
        else:
            new_input_ids = input_ids[1:]

        token_mapping[i] = new_input_id

    """
    embeddings = model.get_input_embeddings()
    print("原始 embeddings: ", embeddings)
    print("原始 embeddings 31000: ", embeddings(torch.LongTensor([31000])))
    """
    new_vocab_size = len(new_tokenizer)
    embedding_dim = 4096
    new_embedding = torch.nn.Embedding(new_vocab_size, embedding_dim)

    """
    print("未赋值前 new_embedding: ", new_embedding)
    print("未赋值前 new_embedding 31000: ", new_embedding(torch.LongTensor([31000])))
    print("未赋值前 new_embedding 35000: ", new_embedding(torch.LongTensor([35000])))
    """

    num_to_copy = min(new_vocab_size, len(embeddings.weight))
    # 开始将现有Embeddiyyng层的权重赋值给新的Embedding层的前32000行
    new_embedding.weight.data[:num_to_copy, :] = embeddings.weight.data[:num_to_copy, :]

    # 开始给新增的token使用原来的tokenizer的input_ids的均值赋值
    for new_token, original_tokens in token_mapping.items():
        original_embeddings = embeddings(original_tokens)
        mean_embedding = torch.mean(original_embeddings, dim=0)
        new_embedding.weight.data[new_token] = mean_embedding

    # 开始更换嵌入层
    model.set_input_embeddings(new_embedding)
    """
    print("model 更换嵌入层:", model)
    print("赋值后 new_embedding", new_embedding)
    print("赋值后 new_embeddings 31000", new_embedding(torch.LongTensor([31000])))
    print("赋值后 new_embeddings 35000", new_embedding(torch.LongTensor([35000])))
    """


    # 开始处理lm_head
    output_size = len(tokenizer)
    new_output_size = len(new_tokenizer)
    lm_head = model.lm_head

    # 新的lm_head
    new_lm_head = torch.nn.Linear(in_features=4096, out_features=new_output_size, bias=False)
    # 前32000个向量不变
    new_lm_head.weight.data[:output_size, :] = lm_head.weight.data[:output_size, :]

    # lm_head开始新增
    for new_token, original_tokens in token_mapping.items():
        original = 0
        for i in original_tokens:
            original += lm_head.weight.data[i]
        mean_para = original / len(original_tokens)
        new_lm_head.weight.data[new_token] = mean_para

    model.lm_head = new_lm_head

    model.config.vocab_size = new_output_size
    model.save_pretrained(new_model_name, max_shard_size="8GB")




if __name__ == "__main__":
    data_file="./data/2021-49_zh_head_000x.txt"
    new_model_prefix="bpe_test"
    target_vocab_size=10000

    llama_tokenizer_dir="./model/Llama-2-7b-hf"
    chinese_sp_model_file="./bpe_test.model"
    output_sp_dir="merged_tokenizer_sp_test"
    output_hf_dir="merged_tokenizer_hf_test"

    new_model_name = "llama-2-7b-extent"

    # train
    train_spm(data_file, new_model_prefix, target_vocab_size)

    # add vocab
    add_vocab_2_llama(llama_tokenizer_dir, chinese_sp_model_file, output_sp_dir, output_hf_dir)

    # new tokenizer
    new_tokenizer(llama_tokenizer_dir, output_hf_dir, new_model_name)
