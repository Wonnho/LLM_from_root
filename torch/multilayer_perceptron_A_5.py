import torch

class NeuralNetwork(torch.nn.Module):
   def __init__(self,num_inputs,num_outputs):
      super().__init__()

      self.layers=torch.nn.Sequential(
         torch.nn.Linear(num_inputs,30),
         torch.nn.ReLU(),

         torch.nn.Linear(30,20),
         torch.nn.ReLU(),

         torch.nn.Linear(20,num_outputs)
      )
   def forward(self,x):
      logits=self.layers(x)
      return logits

torch.manual_seed(123)
model=NeuralNetwork(50,3)
print(model,'model')

num_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of parameters in trainable:',num_params)

print(model.layers[0].weight)

print(model.layers[0].weight.shape)


X=torch.rand((1,50))
# out=model(X)
# print('out',out)

# if not backward gradient 
with torch.no_grad():
   out=model(X)
print('out',out)

with torch.no_grad():
   out=torch.softmax(model(X),dim=1)
print('softmax out',out)
