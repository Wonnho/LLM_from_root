import torch
import torch.nn as nn

from causal_attention_class_3_5_3 import CausalAttention


class MultiHeadAttentionWrapper(nn.Module):
   def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_bias=False):
      super().__init__()
      self.heads=nn.ModuleList(
         [CausalAttention(
            d_in,d_out,context_length,dropout,qkv_bias
         )
         for _ in range(num_heads)
         ]
      )

   def forward(self,x):
      return torch.cat([head(x) for head in self.heads],dim=-1)

   
inputs=torch.tensor(
   [
      [0.43,0.15,0.89],
      [0.55,0.87,0.66],
      [0.57,0.85,0.64],
      [0.22,0.58,0.33],
      [0.77,0.25,0.10],
      [0.05,0.80,0.55]
   ]
)


d_in=inputs.shape[1]
d_out=2
batch=torch.stack((inputs,inputs),dim=0)
#dropout=torch.nn.Dropout(0.5)

torch.manual_seed(123)
context_length=batch.shape[1]
mha=MultiHeadAttentionWrapper(d_in,d_out,context_length,0.0,num_heads=2)
context_vecs=mha(batch)

print('context vectors:',context_vecs)
print("context_vecs.shape:",context_vecs.shape)
