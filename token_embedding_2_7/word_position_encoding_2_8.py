import torch
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

vocab_size=50257
output_dim=256

token_embedding_layer=torch.nn.Embedding(vocab_size,output_dim)

max_length=4

from sliding_windows_data_sampling_2_6.dataloader_2_6 import create_dataloader_v1

file_path = Path(__file__).parent.parent / "the-verdict.txt"
with open("../the-verdict.txt","r",encoding="utf-8") as file:
     raw_text=file.read()

dataloader=create_dataloader_v1(
   raw_text,batch_size=8,max_length=max_length,stride=max_length,shuffle=False
)
data_iter=iter(dataloader)
inputs,targets=next(data_iter)
print("Token ID\n",inputs)
print("\n input size:\n",inputs.shape)

token_embeddings=token_embedding_layer(inputs)
print(token_embeddings.shape)

context_length=max_length
pos_embedding_layer=torch.nn.Embedding(context_length,output_dim)
pos_embeddings=pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings=token_embeddings+pos_embeddings
print(input_embeddings.shape)