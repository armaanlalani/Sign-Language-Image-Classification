import torch.nn as nn
import torch.nn.functional as F

class Net1(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,10,3)
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2,padding=1)
        self.conv2 = nn.Conv2d(10,10,3)
        # self.conv4 = nn.Conv2d(10,10,3)
        # self.conv_BN = nn.BatchNorm2d(10)
        # self.conv_BNl = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(13*13*10,32)
        self.fc2 = nn.Linear(32,10)
    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # x = self.pool2(F.relu(self.conv3(x)))
        # x = self.pool1(F.relu(self.conv4(x)))
        x = x.view(-1, 13*13*10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x