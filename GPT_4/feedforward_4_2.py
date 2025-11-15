import torch
import torch.nn as nn


class FeedForward(nn.Module):
   def __init__(self,cfg):
      super().__init__()
      self.layers=nn.Sequential(
         nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
         nn.GELU(),
         nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"]),
      )

   def forward(self,x):
      return self.layers(x)
   
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

   
ffn=FeedForward(GPT_CONFIG_124M)
x=torch.rand(2,3,768)
out=ffn(x)
print(out.shape)
