import torch

X_train=torch.tensor([
   [-1.2,3.2],
   [-0.9,2.9],
   [-0.5,2.6],
   [2.3,-1.1],
   [2.7,-1.5]
])

y_train=torch.tensor([0,0,0,1,1])

X_test=torch.tensor([
   [-0.8,2.8],
   [2.6,-1.6]
])

y_test=torch.tensor([0,1])

from torch.utils.data import Dataset

class ToyDataset(Dataset):
   def __init__(self,x,y):
      self.features=x
      self.labels=y

   def __getitem__(self,index):
      one_x=self.features[index]
      one_y=self.labels[index]
      return one_x,one_y
   
   def __len__(self):
      return self.labels.shape[0]
   
train_ds=ToyDataset(X_train,y_train)
test_ds=ToyDataset(X_test,y_test)

print('length:',len(train_ds))
print('return an index,1:',train_ds[1]) 

from torch.utils.data import DataLoader
torch.manual_seed(123)

train_loader=DataLoader(
   dataset=train_ds,
   batch_size=2,
   shuffle=True,
   num_workers=0,
   drop_last=True
)

test_loader=DataLoader(
   dataset=train_ds,
   batch_size=2,
   shuffle=False,
   num_workers=0
)

for idx,(x,y) in enumerate(train_loader):
   print(f"batch {idx+1}:",x,y)

import torch.nn.functional as F
from pytorch.multilayer_perceptron_A_5 import NeuralNetwork

torch.manual_seed(123)
model=NeuralNetwork(num_inputs=2,num_outputs=2)
optimizer=torch.optim.SGD(
   model.parameters(),lr=0.5
)

num_epochs=3
for epoch in range(num_epochs):
   model.train()
   for batch_idx,(features,labels) in enumerate(train_loader):
      logits=model(features)

      loss=F.cross_entropy(logits,labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      print(f"epoch:{epoch+1:03d}/{num_epochs:03d}"
      f" | Batch (batch_idx:03d/{len(train_loader):03d})"
      f" | loss :{loss:2f}")
   
model.eval()
with torch.no_grad():
   outputs=model(X_train)
print(outputs)

torch.set_printoptions(sci_mode=False)
probas=torch.softmax(outputs,dim=1)
print(probas)

predictions=torch.argmax(probas,dim=1)
print('predictions:',predictions)

predictions=torch.argmax(outputs,dim=1)
print(predictions)

print(predictions==y_train)

torch.sum(predictions==y_train)


def compute_accuracy(model,dataloader):
   model=model.eval()
   correct=0.0
   total_examples=0

   for idx,(features,labels) in enumerate(dataloader):
      with torch.no_grad():
         logits=model(features)

         predictions=torch.argmax(logits,dim=1)
         compare=labels==predictions
         correct+=torch.sum(compare)
         total_examples+=len(compare)
   return (correct/total_examples).item()

print('accuracy of train',compute_accuracy(model,train_loader))

print('accuracy of test',compute_accuracy(model,test_loader))
