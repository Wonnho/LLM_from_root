import torch
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

query=inputs[1]
attn_scores_2=torch.empty(inputs.shape[0])
for k,x_k in enumerate(inputs):
   attn_scores_2[k]=torch.dot(x_k,query)
print(attn_scores_2)


#normalization of attention scores

attn_weights_2_tmp=attn_scores_2/attn_scores_2.sum()
print("attention weights:",attn_weights_2_tmp)
print("sum:",attn_weights_2_tmp.sum())

attn_weights_2=torch.softmax(attn_scores_2,dim=0)
print('attention weights:',attn_weights_2)
print("sum:",attn_weights_2.sum())

query=inputs[1]
context_vec_2=torch.zeros(query.shape)
for k,x_k in enumerate(inputs):
   context_vec_2+=attn_weights_2[k]*x_k
print(context_vec_2)

# attn_scores=torch.empty(6,6)
# for k, x_k in enumerate(inputs):
#    for j,x_j in enumerate(inputs):
#       attn_scores[k,j]=torch.dot(x_k,x_j)
# print(attn_scores)

attn_scores=inputs@inputs.T
print(attn_scores)


attn_weights=torch.softmax(attn_scores,dim=-1)
print(attn_weights)

all_context_vec=attn_weights@inputs
print(all_context_vec)

