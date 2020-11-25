import argparse
from time import time

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

from torchsummary import summary

from sklearn.metrics import confusion_matrix
from model import Net

mean = [0.5801265923035882, 0.4923211367880485, 0.4326831149126318]
std = [0.21313345096241879, 0.2401031219607465, 0.24437690218961625]

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]) # transform compose containing the respective transforms
dataset = dset.ImageFolder('/Users/armaanlalani/Documents/Engineering Science Year 3/ECE324 - Intro to Machine Intelligence/Assignment 4/Lalani_1005023225/', transform = transform) # loads dataset using the initialized transforms

data = DataLoader(dataset, batch_size = 10, shuffle = True) # dataloader object

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5801265923035882, 0.4923211367880485, 0.4326831149126318])
    std = np.array([0.21313345096241879, 0.2401031219607465, 0.24437690218961625])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.pause(25)  # pause a bit so that plots are updated

inputs, classes = next(iter(data))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

class_names = dataset.classes
# imshow(out, title=[class_names[x] for x in classes])

def evaluate(outputs, labels):
    output = outputs.detach().numpy()
    label = labels.detach().numpy()
    count = 0
    for i in range (output.shape[0]):
        # predict.append(np.argmax(output[i]))
        # real.append(np.argmax(label[i]))
        if np.argmax(output[i]) == np.argmax(label[i]):
        # if np.argmax(output[i]) == label[i]:
            count = count + 1
    return count / output.shape[0]

def load_model(lr):
    model = Net()
    loss_fnc = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)

    return model, loss_fnc, optimizer

model, loss, opt = load_model(0.1)

for epoch in range(30):
    total_loss = []
    acc = []
    first_time = time()

    for i, data in enumerate(data, 0):
        inputs, labels = data
        inputs = inputs.float() # inputs of the training data
        labels = labels.float() # labels of the training data

        opt.zero_grad() # sets the gradient to 0

        outputs = model(inputs) # predicts labels based on the inputs

        labels_out = labels.detach().numpy()
        labels_new = np.zeros((10,10))
        labels_out = labels_out.astype(np.int)
        for j in range(10):
            labels_new[j,labels_out[j]] = 1
        labels_new = labels_new.astype(np.float32)
        labels = torch.from_numpy(labels_new)

        loss_in = loss(input=outputs, target=labels)
        loss_in.backward()
        opt.step()

        total_loss.append(loss_in.item()) # total running loss

        acc.append(evaluate(outputs, labels))
