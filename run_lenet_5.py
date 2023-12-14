import os
import math
from glob import glob
import cv2
from sklearn.model_selection import train_test_split
from torch import nn
import torch
import subprocess
from torch.nn import functional as F
import numpy as np


ZERO_TYPE = 'none' 
EPSILON = 0
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

def save_matrix(x, fpath):
    assert x.dim() == 2

    with open(fpath, "wb") as f:
        f.write(x.size(dim=0).to_bytes(4, byteorder="little"))
        f.write(x.size(dim=1).to_bytes(4, byteorder="little"))
        f.write(x.numpy().tobytes())


def load_matrix(fpath) -> torch.Tensor:
    with open(fpath, "rb") as f:
        h = int.from_bytes(f.read(4), byteorder="little")
        w = int.from_bytes(f.read(4), byteorder="little")

        buf = f.read(h * w * np.dtype(np.float32).itemsize)

        x = torch.from_numpy(np.frombuffer(buf, dtype=np.float32).copy())
        x = x.reshape(h, w)

        return x


def pad_matrix(x) -> torch.Tensor:
    assert x.dim() == 2

    h = x.size(dim=0)
    w = x.size(dim=1)

    if h % 16 != 0:
        padding = 16 - h % 16
        x = F.pad(x, pad=(0, 0, 0, padding), mode="constant", value=0.0)

    if w % 16 != 0:
        padding = 16 - w % 16
        x = F.pad(x, pad=(0, padding, 0, 0), mode="constant", value=0.0)

    return x

def conv2d(x, layer_name, module) -> torch.Tensor: 
    START_INDEX=1
    CURRENT_INDEX=1

    # Either the (1) output directory of the convolution operation or 
    # (2) location of the cached convolution from a previous execution. 
    gemm_file_path = f'bin/lenet5/gemm/{layer_name}.bin' 

    weight = module.weight.detach()
    x = x.detach()

    out_n = x.size(dim=0)
    out_c = weight.size(dim=0)
    out_h = math.floor(((x.size(dim=2) + 2 * module.padding[0] - module.dilation[0] * (
        module.kernel_size[0] - 1) - 1) / module.stride[0]) + 1)
    out_w = math.floor(((x.size(dim=3) + 2 * module.padding[1] - module.dilation[1] * (
        module.kernel_size[1] - 1) - 1) / module.stride[1]) + 1)

    weight = weight.flatten(start_dim=1)
    weight = weight.view(weight.size(dim=0), weight.size(dim=1))

    x = nn.Unfold(kernel_size=module.kernel_size, stride=module.stride,
                  padding=module.padding, dilation=module.dilation)(x)

    slices = []
    for i in range(x.size(dim=0)):
        slices.append(x[i])
    x = torch.cat(slices, dim=1)

    h = weight.size(dim=0)
    w = x.size(dim=1)

    weight = pad_matrix(weight)
    x = pad_matrix(x)

    if START_INDEX <= CURRENT_INDEX:
        # Preprocess the matrices given the passed arguments by removing values
        # from the tensors if they're less than or equal to EPSILON
        if ZERO_TYPE in ('both', 'weight'): 
            weight = torch.where(torch.abs(weight) <= EPSILON, 0, weight)
        if ZERO_TYPE in ('both', 'input'): 
            x = torch.where(torch.abs(x) <= EPSILON, 0, x) 

        # We don't need to use cached conv outputs, so calculate the conv.         
        weight_file_path = f'bin/lenet5/weight/{layer_name}.bin'
        save_matrix(weight, weight_file_path) 
        
        x_file_path = f'bin/lenet5/x/{layer_name}.bin'
        save_matrix(x, x_file_path)
    
        output_file_path = f'output/lenet5/result/{layer_name}.txt'
        output_file = open(output_file_path, 'w') 
        
        error_file_path = f'output/lenet5/error/{layer_name}.txt'
        error_file = open(error_file_path, 'w')
        
        print(f'Starting gemm on layer "{layer_name}"')
        subprocess.run(
            [
                "./build/gemm", 
                "--w", 
                weight_file_path,
                "--x", 
                x_file_path, 
                "--o", 
                gemm_file_path
            ], 
            stdout=output_file, 
            stderr=error_file
        )
        print(f'Finished gemm on layer "{layer_name}"') 
    
    x = load_matrix(gemm_file_path) 
    
    x = torch.stack(torch.chunk(x[:h, :w], chunks=out_n, dim=1))
    x = x.view(out_n, out_c, out_h, out_w)

    for name, _ in module.named_parameters():
        if name in ["bias"]:
            bias = module.bias.detach()
            bias = bias.view(1, bias.size(dim=0), 1, 1)
            bias = bias.tile(1, 1, out_h, out_w)

            x = x.add(bias)

    CURRENT_INDEX += 1 
    return x

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = torch.load('model_weights.pth')
        # self.model.eval()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        

    #return x
    def __call__(self, data):
        # self.model.load_state_dict(torch.load('model_weights.pth'))
        # self.model.eval() 
        self=torch.load('model_weights.pth')
        self.eval()
        x= data
        # x = self.conv1(data)
        x = conv2d(x, "cL1_", self.conv1)

        x = F.relu(x)

        x = F.max_pool2d(x, (2, 2))

        x = conv2d(x, "cL2_", self.conv2)
        # x = self.conv2(x)
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


data_path= 'data_lenet5/myData'
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
model=LeNet5()
# model = torch.load('model_weights.pth')
# model.eval()
for i in range(len(x_train)):
    x_train[i] = x_train[i].to(dtype=torch.float32)
x_train= torch.stack(x_train)
y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)
for i in range(len(x_test)):
    x_test[i] = x_test[i].to(dtype=torch.float32)
x_test= torch.stack(x_test)


# backprop_deep(x_train, y_train, model,T=10)
y_result = model(x_test)
print()
# torch.save(model, 'model_weights.pth')
print("percentage of the model:")
print(100 * (y_test==y_result.max(1)[1]).float().mean())

