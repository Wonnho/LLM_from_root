import torch
import torch.nn as nn

class ExampleDeepNeuralNetwork(nn.Module):
   def __init__(self,layer_sizes,user_shortcut):
      super().__init__()
      self.user_shortcut=user_shortcut
      self.layers=nn.ModuleList([
         nn.Sequential(nn.Linear(layer_sizes[0],layer_sizes[1]),
         nn.GELU()),
         nn.Sequential(nn.Linear(layer_sizes[1],layer_sizes[2]),
         nn.GELU()),
         nn.Sequential(nn.Linear(layer_sizes[2],layer_sizes[3]),
         nn.GELU()),
         nn.Sequential(nn.Linear(layer_sizes[3],layer_sizes[4]),
         nn.GELU()),
         nn.Sequential(nn.Linear(layer_sizes[4],layer_sizes[5]),
         nn.GELU())
          ])
   def forward(self,x):
      for layer in self.layers:
         layer_output=layer(x)
         if self.user_shortcut and x.shape == layer_output.shape:
            x=x+layer_output
         else:
            x=layer_output
      return x

layer_sizes=[3,3,3,3,3,1]
sample_input=torch.tensor([[1.0,0.,-1.]])
torch.manual_seed(123)
#model_without_shortcut=ExampleDeepNeuralNetwork(layer_sizes,user_shortcut=False)
model_with_shortcut=ExampleDeepNeuralNetwork(layer_sizes,user_shortcut=True)

def print_gradients(model,x):
   output=model(x)
   target=torch.tensor([[0.]])

   loss=nn.MSELoss()
   loss=loss(output,target)

   loss.backward()
   for name,param in model.named_parameters():
      if 'weight' in name:
         print(f"average gradient of {name} is {param.grad.abs().mean().item()}")

#print_gradients(model_without_shortcut,sample_input)

print_gradients(model_with_shortcut,sample_input)
