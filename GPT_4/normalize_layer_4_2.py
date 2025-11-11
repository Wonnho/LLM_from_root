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


import torch.nn as nn

torch.manual_seed(123)
batch_example=torch.randn(2,5)
layer=nn.Sequential(nn.Linear(5,6),nn.ReLU())
out=layer(batch_example)
print("nomralizer layers:",out)


class LayerNorm(nn.Module):
   def __init__(self,emb_dim):
      super().__init__()
      self.eps=1e-5
      self.scale=nn.Parameter(torch.ones(emb_dim))
      self.shift=nn.Parameter(torch.zeros(emb_dim))

   def forward(self,x):
      mean=x.mean(dim=-1,keepdim=True)
      var=x.var(dim=-1,keepdim=True,unbiased=False)
      norm_x=(x-mean)/torch.sqrt(var+self.eps)
      return self.scale*norm_x+self.shift
   

ln=LayerNorm(emb_dim=5)
out_ln=ln(batch_example)
mean=out_ln.mean(dim=-1,keepdim=True)
var=out_ln.var(dim=-1,unbiased=False,keepdim=True)
print("mean\n",mean)
print("variance\n",var)
