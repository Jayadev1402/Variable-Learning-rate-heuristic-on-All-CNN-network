
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Defining the model
class View(nn.Module):
    def __init__(self,o):
        super().__init__()
        self.o = o

    def forward(self,x):
        return x.view(-1, self.o)
    
class allcnn_t(nn.Module):
    def __init__(self, c1=96, c2= 192):
        super().__init__()
        d = 0.5

        def convbn(ci,co,ksz,s=1,pz=0):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
                nn.ReLU(True),
                nn.BatchNorm2d(co))

        self.m = nn.Sequential(
            nn.Dropout(0.2),
            convbn(3,c1,3,1,1),
            convbn(c1,c1,3,1,1),
            convbn(c1,c1,3,2,1),
            nn.Dropout(d),
            convbn(c1,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,2,1),
            nn.Dropout(d),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,10,1,1),
            nn.AvgPool2d(8),
            View(10))

        print('Num parameters: ', sum([p.numel() for p in self.m.parameters()]))

    def forward(self, x):
        return self.m(x)
def optimal_lr(model,trainloader):
  T=100
  T0=T/5
  i=0
  model=allcnn_t()
  criterion = nn.CrossEntropyLoss()
  l=10**(-5)
  lr_all=[]
  loss_all=[]
  model = model.to(device)
  optimizer = optim.SGD(model.parameters(), lr=l, momentum=0.9, weight_decay=0.001)
  for i, (images, labels) in enumerate(trainloader):
          # Move tensors to configured device
          images = images.to(device)
          labels = labels.to(device)
          #Forward Pass
          optimizer.zero_grad()
          outputs = model(images)
          loss = criterion(outputs, labels)
          loss.backward()
          
          l=1.1*l
          lr_all.append(l)
          optimizer = optim.SGD(model.parameters(), lr=l, momentum=0.9, weight_decay=0.0001, nesterov=True)
          optimizer.step()
          loss_all.append(loss.item())
          i+=1
          print(f'iter {i} : {loss.item()}')
          if i==T:
            break
  min_idx = np.argmin(np.array(loss_all))
  lr_s = lr_all[min_idx]
  lr_max=lr_s/10
  return lr_all,loss_all,lr_max
def train(net,criterion,optimizer,trainloader,testloader,e):
  # net = allcnn_t()
  net.to(device)

  T=19550
  T0=T/5
  i=0
  criterion = nn.CrossEntropyLoss()

  lr_all=[]
  lr_max=0.1



  optimizer = optim.SGD(net.parameters(), lr=lr_max, momentum=0.9, weight_decay=0.001)

  train_loss = []
  train_error = []
  val_loss = []
  val_error = []

  def alphas(i,T,T0,lr_max):
    if i <= T0:
      alpha = 10**(-4) + (i/T)*lr_max  
    else: 
      alpha = lr_max*np.cos((np.pi/2)*((i-T0)/(T-T0))) + 10**(-6) 
    return alpha
  optimizer = optim.SGD(net.parameters(), lr=lr_max, momentum=0.9, weight_decay=0.001)

  for epoch in range(e): 
    correct = 0
    total = 0.  
    for i, data in enumerate(trainloader): 
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        lr=alphas(i,T,T0,lr_max)
        lr_all.append(lr)
        for g in optimizer.param_groups:
          g['lr'] = lr
        
        optimizer.step()
        i+=1

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
      
        train_loss.append(loss.item()) 
        train_error.append(100-100 * correct / total)

    correct = 0
    total = 0.  
    for i, data in enumerate(testloader):

      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = net(inputs)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
    
      val_error.append(100-100 * correct / total )
      loss = criterion(outputs, labels)

      val_loss.append(loss.item())
    print('Accuracy of the network on the test images: {}:{} %'.format(epoch,100 * correct / total))
  return train_loss,train_error, val_loss,val_error
def plot(net,train_loss,train_error,val_loss,val_error):
  t_step = np.array(range(len(train_loss)))
  plt.plot(t_step, np.array(train_loss))
  plt.ylabel('Training loss')
  plt.xlabel('No. of iterations')
  plt.show()

  t_step = np.array(range(len(val_loss)))
  plt.plot(t_step, np.array(val_loss))
  plt.ylabel('Validation loss')
  plt.xlabel('No. of iterations')
  plt.show()

  t_step = np.array(range(len(train_error)))
  plt.plot(t_step, np.array(train_error))
  plt.ylabel('Training error')
  plt.xlabel('No. of iterations')
  plt.show()

  t_step = np.array(range(len(val_error)))
  plt.plot(t_step, np.array(val_error))
  plt.ylabel('Validation error')
  plt.xlabel('No. of iterations')
  plt.show()

  T=19550
  T0=T/5
  i=0
  criterion = nn.CrossEntropyLoss()

  lr_all=[]
  lr_max=0.071
  def alphas(i,T,T0,lr_max):
    if i <= T0:
      alpha = 10**(-4) + (i/T0)*lr_max  
    else: 
      alpha = lr_max*np.cos((np.pi/2)*((i-T0)/(T-T0))) + 10**(-6) 
    return alpha
  optimizer = optim.SGD(net.parameters(), lr=lr_max, momentum=0.9, weight_decay=0.001)

  al=[]
  for i in range(50000):
    al.append(alphas(i,T,T0,lr_max))

  plt.plot(range(len(al)), np.array(al))
  plt.ylabel('Learning rate')
  plt.xlabel('No. of iterations')
  plt.show()


def main():
  ######################################
  model=allcnn_t()
  lr_all,loss_all,lr_max=optimal_lr(model,trainloader)
  plt.semilogx(np.array(lr_all), np.array(loss_all))
  plt.xlabel('Learning Rate (log)')
  plt.ylabel('Training loss (log)')

  ######################################
  model=allcnn_t()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=lr_max, momentum=0.9, weight_decay=0.001)
  train_loss,train_error,val_loss,val_error=train(model,criterion,optimizer,trainloader,testloader,100)
  plot(model,train_loss,train_error,val_loss,val_error)

  ###################################### 
  model=allcnn_t()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=lr_max, momentum=0.9, weight_decay=0.001)
  train_loss,train_error,val_loss,val_error=train(model,criterion,optimizer,trainloader,testloader,50)
  plot(model,train_loss,train_error,val_loss,val_error)

  model=allcnn_t()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=5*lr_max, momentum=0.5, weight_decay=0.001)
  train_loss,train_error,val_loss,val_error=train(model,criterion,optimizer,trainloader,testloader,50)
  plot(model,train_loss,train_error,val_loss,val_error)


  model=allcnn_t()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=lr_max, momentum=0.5, weight_decay=0.001)
  train_loss,train_error,val_loss,val_error=train(model,criterion,optimizer,trainloader,testloader,50)
  plot(model,train_loss,train_error,val_loss,val_error)

if __name__=="__main__":
  main()
