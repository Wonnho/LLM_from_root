import torch
tensor1d=torch.tensor([5,3,4,9])
print(tensor1d.dtype)

floatvec=torch.tensor([1.9,4,5,9.2])
print(floatvec.dtype)

floatvec=tensor1d.to(torch.float32)
print(floatvec.dtype)