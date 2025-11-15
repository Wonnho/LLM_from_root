import sys
import os

# Add the parent directory (SCRATCH_LLM) to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now the import should work:

from attention_mechanism_3.multihead_attention_w_weight_3_6_2 import MultiHeadAttention
import torch
import torch.nn as nn
from torch.nn import LayerNorm

from feedforward_4_2 import FeedForward

class TransformerBlock(nn.Module):
   def __init__(self,cfg):
      super().__init__()
      self.att=MultiHeadAttention(
         d_in=cfg["emb_dim"],
         d_out=cfg["emb_dim"],
         context_length=cfg["context_length"],
         num_heads=cfg["n_heads"],
         dropout=cfg["drop_rate"],
         qkv_bias=cfg["qkv_bias"]
      )
      self.ff=FeedForward(cfg)
      self.norm1=LayerNorm(cfg["emb_dim"])
      self.norm2=LayerNorm(cfg["emb_dim"])
      self.drip_shortcut=nn.Dropout(cfg["drop_rate"])
   
   def forward(self,x):
      shortcut=x
      x=self.norm1(x)
      x=self.att(x)
      x=self.drip_shortcut(x)
      x=x+shortcut

      shortcut=x
      x=self.norm2(x)
      x=self.ff(x)
      x=self.drip_shortcut(x)
      x=x+shortcut
      return x
   

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
x=torch.rand(2,4,768)
block=TransformerBlock(GPT_CONFIG_124M)
output=block(x)

print("input size:",x.shape)
print("output size:",output.shape)