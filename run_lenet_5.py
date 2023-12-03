import os
from glob import glob
import cv2
from sklearn.model_selection import train_test_split
from torch import nn
import torch
from torch.nn import functional as F
import numpy as np

def backprop_deep(xtrain, ltrain, net, T, B=10, gamma=.001, rho=.9):
    '''
    Backprop.
    
    Args:
        xtrain: training samples
        ltrain: testing samples
        net: neural network
        T: number of epochs
        B: minibatch size
        gamma: step size
        rho: momentum
    '''
    N = xtrain.size()[0]     # Training set size
    NB = N//B                # Number of minibatches
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=gamma, momentum=rho)
    
    for epoch in range(T):
        running_loss = 0.0
        shuffled_indices = np.random.permutation(NB)
        for k in range(NB):
            # Extract k-th minibatch from xtrain and ltrain
            minibatch_indices = range(shuffled_indices[k]*B, (shuffled_indices[k]+1)*B)
            inputs = xtrain[minibatch_indices]
            labels = ltrain[minibatch_indices]

            # Initialize the gradients to zero
            optimizer.zero_grad()

            # Forward propagation
            outputs = net(inputs)
            # Error evaluation
            loss = criterion(outputs, labels)

            # Back propagation
            loss.backward()

            # Parameter update
            optimizer.step()

            # Print averaged loss per minibatch every 100 mini-batches
            # Compute and print statistics
            with torch.no_grad():
                running_loss += loss.item()
            if k % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, k + 1, running_loss / 100))
                running_loss = 0.0
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    #return x
    def __call__(self, data): 

        x = self.conv1(data)

        x = F.relu(x)

        x = F.max_pool2d(x, (2, 2))

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        return x


    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)

    #preprocess input file to fit Lenet-5
    def preprocess(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = img / 255
        img=cv2.resize(img,(32,32), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(img).unsqueeze(0)
        return img 

    def __len__(self): 
        return 8 


data_path= 'myData'
if os.path.exists(data_path):
    if os.path.isdir(data_path):
        x=[]
        y=[]
        for i in range (10):
            new_data_path=data_path+'/'+str(i)
            if os.path.isdir(new_data_path):
                pathnames=glob(os.path.join(new_data_path, "*.png"))
            for file in pathnames:
                img = cv2.imread(file)
                x.append(LeNet5.preprocess(img))
                y.append(i)
            print("input the package:",i)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, train_size = .7)
model=LeNet5().to('cuda')
for i in range(len(x_train)):
    x_train[i] = x_train[i].to(dtype=torch.float32)
x_train= torch.stack(x_train)
y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)
for i in range(len(x_test)):
    x_test[i] = x_test[i].to(dtype=torch.float32)
x_test= torch.stack(x_test)
x_train=x_train.to('cuda')
x_test=x_test.to('cuda')
y_train=y_train.to('cuda')
y_test=y_test.to('cuda')
print()
print("training process: ")
backprop_deep(x_train, y_train, model,T=10)
y_result = model(x_test)
print()
print("percentage of the model:")
print(100 * (y_test==y_result.max(1)[1]).float().mean())

