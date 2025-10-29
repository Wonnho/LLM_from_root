import torch
import torch.nn.functional as F
# y=torch.tensor([1.0])
# x1=torch.tensor([1.1])
# w1=torch.tensor([2.2])
# b=torch.tensor([0.0])
# z=x1*w1+b
# a=torch.sigmoid(z)
# loss=F.binary_cross_entropy(a,y)
# print('loss:',loss)

from torch.autograd import grad
y=torch.tensor([1.0])
x1=torch.tensor([1.1])
w1=torch.tensor([2.2],requires_grad=True)
b=torch.tensor([0.0],requires_grad=True)
z=x1*w1+b
a=torch.sigmoid(z)
loss=F.binary_cross_entropy(a,y)
print('loss:',loss)

grad_L_w1=grad(loss,w1,retain_graph=True)
grad_L_b=grad(loss,b,retain_graph=True)

print('weight:',grad_L_w1)
print('bias:',grad_L_b)

loss.backward()
print(w1.grad)
print(b.grad)
