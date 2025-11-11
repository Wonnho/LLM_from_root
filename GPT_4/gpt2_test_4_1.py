import torch   
import tiktoken

tokenizer=tiktoken.get_encoding("gpt2")
batch=[]
txt1="Every effort moves you"
txt2="Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch=torch.stack(batch,dim=0)
print(batch)

from dummy_gpt_model_class_4_1 import DummyGPTModel,DummyLayerNorm,DummyTransformerBlock

# GPT-2 124M 설정 추가
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # 어휘 크기
    "context_length": 1024,   # 최대 입력 길이
    "emb_dim": 768,           # 임베딩 차원
    "n_heads": 12,            # 어텐션 헤드 수
    "n_layers": 12,           # 트랜스포머 블록 수
    "drop_rate": 0.1,         # 드롭아웃 비율
    "qkv_bias": False         # Query, Key, Value에 bias 사용 여부
}

torch.manual_seed(123)
model=DummyGPTModel(GPT_CONFIG_124M)
logits=model(batch)
print("output size:",logits.shape)
print(logits)
