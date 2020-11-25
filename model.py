import torch.nn as nn
import torch.nn.functional as F

class Net_PRACTICE(nn.Module): # model for testing the 12 hyperparameters, with 2 convolutional layers
    def __init__(self):
        super(Net_PRACTICE,self).__init__()
        self.conv1 = nn.Conv2d(3,4,3)
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2,padding=1)
        self.conv2 = nn.Conv2d(4,8,5)
        self.fc1 = nn.Linear(12*12*8,100)
        self.fc2 = nn.Linear(100,10)
    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 12*12*8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net1_2(nn.Module): # model for testing the 12 hyperparameters, with 2 convolutional layers
    def __init__(self, num_kernels, num_neurons):
        self.num_kernels = num_kernels
        super(Net1_2,self).__init__()
        self.conv1 = nn.Conv2d(3,num_kernels,3)
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2,padding=1)
        self.conv2 = nn.Conv2d(num_kernels,num_kernels,3)
        self.fc1 = nn.Linear(13*13*num_kernels,num_neurons)
        self.fc2 = nn.Linear(num_neurons,10)
    def forward(self,x):
        print(x.shape)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 13*13*self.num_kernels)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net1_4(nn.Module): # model for testing the 12 hyperparameters, with 4 convolutional layers
    def __init__(self, num_kernels, num_neurons):
        self.num_kernels = num_kernels
        super(Net1_4,self).__init__()
        self.conv1 = nn.Conv2d(3,num_kernels,3)
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2,padding=1)
        self.conv2 = nn.Conv2d(num_kernels,num_kernels,3)
        self.conv3 = nn.Conv2d(num_kernels,num_kernels,3)
        self.conv4 = nn.Conv2d(num_kernels,num_kernels,3)
        self.fc1 = nn.Linear(2*2*num_kernels,num_neurons)
        self.fc2 = nn.Linear(num_neurons,10)
    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool1(F.relu(self.conv4(x)))
        x = x.view(-1, 2*2*self.num_kernels)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net1_BN(nn.Module): # 2 convolutional layers with batch normalization
    def __init__(self, num_kernels, num_neurons):
        self.num_kernels = num_kernels
        super(Net1_BN,self).__init__()
        self.conv1 = nn.Conv2d(3,num_kernels,3)
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2,padding=1)
        self.conv2 = nn.Conv2d(num_kernels,num_kernels,3)
        self.conv3 = nn.Conv2d(num_kernels,num_kernels,3)
        self.conv4 = nn.Conv2d(num_kernels,num_kernels,3)
        self.conv_BN = nn.BatchNorm2d(num_kernels)
        self.conv_BNl = nn.BatchNorm1d(num_neurons)
        self.fc1 = nn.Linear(2*2*num_kernels,num_neurons)
        self.fc2 = nn.Linear(num_neurons,10)
    def forward(self,x):
        x = self.conv_BN(self.pool1(F.relu(self.conv1(x))))
        x = self.conv_BN(self.pool2(F.relu(self.conv2(x))))
        x = self.conv_BN(self.pool2(F.relu(self.conv3(x))))
        x = self.conv_BN(self.pool1(F.relu(self.conv4(x))))
        x = x.view(-1, 2*2*self.num_kernels)
        x = self.conv_BNl(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class Net_best(nn.Module): # my best model
    def __init__(self):
        super(Net_best,self).__init__()
        self.conv1 = nn.Conv2d(3,10,3)
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2,padding=1)
        self.conv2 = nn.Conv2d(10,10,3)
        self.conv3 = nn.Conv2d(10,10,3)
        self.fc1 = nn.Linear(6*6*10,100)
        self.fc2 = nn.Linear(100,30)
        self.fc3 = nn.Linear(30,10)
    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = x.view(-1, 6*6*10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net_bestsmall(nn.Module): # my best small model
    def __init__(self):
        super(Net_bestsmall,self).__init__()
        self.conv1 = nn.Conv2d(3,10,3)
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2,padding=1)
        self.conv2 = nn.Conv2d(10,10,3)
        self.conv3 = nn.Conv2d(10,10,3)
        self.conv4 = nn.Conv2d(10,10,3)
        self.fc1 = nn.Linear(2*2*10,30)
        self.fc2 = nn.Linear(30,10)
    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool1(F.relu(self.conv4(x)))
        x = x.view(-1, 2*2*10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x