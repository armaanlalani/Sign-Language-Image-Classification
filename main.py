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
from model import Net_PRACTICE
from model import Net1_2 # 2 convolution layer network with variable number of kernels and size of first fully connected layer
from model import Net1_4 # 4 convolution layer network with variable number of kernels and size of first fully connected layer
from model import Net1_BN # my best solution from the 12 choices of hyperparameters with batch normalization
from model import Net_best # my best overall model
from model import Net_bestsmall # my best overall small model


#------------------THIS CODE IS COMMENTED OUT BECAUSE THE MEAN AND STD IS ALREADY CALCULATED, TO AVOID DOING IT WHEN RUNNING THE PROGRAM EVERY TIME----------------

# dataset = dset.ImageFolder('/Users/armaanlalani/Documents/Engineering Science Year 3/ECE324 - Intro to Machine Intelligence/Assignment 4/training_v2/')

# def mean_std(dataset): 
#   mean = [0,0,0]
#    std = [0,0,0]
#    for img, _ in dataset:
#        image = img.convert()
#        data = np.asarray(image)
#        mean[0] = mean[0] + np.average(data[:,:,0])
#        mean[1] = mean[1] + np.average(data[:,:,1])
#        mean[2] = mean[2] + np.average(data[:,:,2])
#        std[0] = std[0] + np.std(data[:,:,0])
#        std[1] = std[1] + np.std(data[:,:,1])
#        std[2] = std[2] + np.std(data[:,:,2])

#    mean = [x / len(dataset) / 255 for x in mean]
#    std = [x / len(dataset) / 255 for x in std]
#    return mean, std

# print(mean_std(dataset))

mean = [0.687623792964106, 0.6189514144452015, 0.5673827742106011] # mean of dataset based on commented out code above
std = [0.12916601094530797, 0.16210342066441724, 0.17262787525450826] # std of dataset based on commented out code above

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]) # transform compose containing the respective transforms
dataset = dset.ImageFolder('/Users/armaanlalani/Documents/Engineering Science Year 3/ECE324 - Intro to Machine Intelligence/Assignment 4/training_v2/', transform = transform) # loads dataset using the initialized transforms

data = DataLoader(dataset, batch_size = 4, shuffle = True) # dataloader object to display the images

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.687623792964106, 0.6189514144452015, 0.5673827742106011]) # unnormalizes the image in order to display correctly
    std = np.array([0.12916601094530797, 0.16210342066441724, 0.17262787525450826])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp) # shows the plot of images
    plt.title(title) # updates the title based on the classes
    plt.pause(10)

inputs, classes = next(iter(data)) # obtains the data necessary to print
out = torchvision.utils.make_grid(inputs) # creates a grid based on the batch size
class_names = dataset.classes
# imshow(out, title=[class_names[x] for x in classes])

def load_data(batch_size):
    train_test_split = 0.2 # train_test split fraction
    dataset_size = len(dataset)
    ind = list(range(dataset_size))
    split = int(np.floor(train_test_split * dataset_size))
    np.random.seed(0) # consistent seed
    np.random.shuffle(ind) # shuffles the indices of the various images
    # develops the indices for training and validation training sets
    if batch_size == 4: # changes length of indices to ensure size of training data % batch size == 0
        train_ind, val_ind = ind[split:], ind[:split-3]
    elif batch_size == 32:
        train_ind, val_ind = ind[split+12:], ind[:split+5]

    train_sample = SubsetRandomSampler(train_ind)
    valid_sample = SubsetRandomSampler(val_ind)

    train_load = DataLoader(dataset, batch_size=batch_size, sampler = train_sample) # creates train_load based on the data subset
    val_load = DataLoader(dataset, batch_size=batch_size, sampler = valid_sample) # creates val_load based on the data subset

    return train_load, val_load

def load_model(lr, mseloss_true): # initiates the model, loss function, and optimizer
    model = Net1_2(10,32) # use this to change the model type
    if mseloss_true:
        loss_fnc = torch.nn.MSELoss() # sets loss function based on mseloss_true
    else:
        loss_fnc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)

    return model, loss_fnc, optimizer

def evaluate(outputs, labels, loss_fnc):
    output = outputs.detach().numpy() # creates a numpy array from output tensor
    label = labels.detach().numpy() # creates a numpy array from label tensor
    count = 0
    for i in range (output.shape[0]):
        if loss_fnc: # different evaluation if loss_fnc is mse or cross entropy
            if np.argmax(output[i]) == np.argmax(label[i]):
                count = count + 1
        elif not loss_fnc:
            if np.argmax(output[i]) == label[i]:
                count = count + 1
    return count / output.shape[0] # counts accurate predictions and divides by total predictions to get accuracy

def evaluate1(model, val_loader, batch_size, loss, val_loss, loss_fnc):
    eval = 0
    count = 0
    total_loss = 0
    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = inputs.float()
        labels = labels.float()
        outputs = model(inputs) # obtains outputs from validation data base on current model

        if loss_fnc: # changes the shape of labels to match outputs necessary for the mse loss function     
            labels_out = labels.detach().numpy()
            labels_new = np.zeros((batch_size,10))
            labels_out = labels_out.astype(np.int)
            for j in range(batch_size):
                labels_new[j,labels_out[j]] = 1
            labels_new = labels_new.astype(np.float32)
            labels = torch.from_numpy(labels_new)
        elif not loss_fnc: # some minor changes to ensure tensors are in the right form for mse loss
            outputs = outputs.type(torch.float)
            labels = labels.type(torch.LongTensor)

        eval = eval + evaluate(outputs, labels, loss_fnc)
        count = count + 1

        loss_in = loss(input=outputs, target=labels) # determines the loss
        total_loss = total_loss + loss_in # determines the accuracy of the validation data
    val_loss.append(total_loss/count) # adds the accuracy to the validation accuracy list
    return eval / count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default = 4)
    parser.add_argument('--lr', type=float, default = 0.1)
    parser.add_argument('--epochs', type=int, default= 30)
    parser.add_argument('--eval_every', type=int, default=763)
    parser.add_argument('--loss_fnc', type=bool, default = True) # true if mse loss

    args = parser.parse_args()

    train, val = load_data(args.batch_size) # loads training and validation data
    model, loss, opt = load_model(args.lr, args.loss_fnc) # instantiates the model, loss function, and optimization function

    valid_acc = []
    train_acc = []
    valid_loss = []
    train_loss = []
    time_total = []
    first_time = time() # starts the timer

    summary(model, (3, 56, 56))

    for epoch in range(0,args.epochs,1):
        predict = [] # used if a confusion matrix is needed
        real = [] # used if a confusion matrix is needed
        total_loss = 0
        total_corr = 0
        train_eval = 0
        for i, data in enumerate(train, 0):
            inputs, labels = data
            inputs = inputs.float() # inputs of the training data
            labels = labels.float() # labels of the training data

            opt.zero_grad() # sets the gradient to 0

            outputs = model(inputs) # predicts labels based on the inputs
            # the below sectioned code converts both outputs and labels to same dimension in order to compare properly

            if args.loss_fnc: # changes the shape of labels to match outputs necessary for the mse loss function 
                labels_out = labels.detach().numpy()
                labels_new = np.zeros((args.batch_size,10))
                labels_out = labels_out.astype(np.int)
                for j in range(args.batch_size):
                    labels_new[j,labels_out[j]] = 1
                labels_new = labels_new.astype(np.float32)
                labels = torch.from_numpy(labels_new)
            elif not args.loss_fnc: # some minor changes to tensors necessary for cross entropy loss
                outputs = outputs.type(torch.float)
                labels = labels.type(torch.LongTensor)

            loss_in = loss(input=outputs, target=labels) # determines the loss
            loss_in.backward()
            opt.step() # optimization step

            total_loss = total_loss + loss_in.item() # total running loss

            train_eval = train_eval + evaluate(outputs, labels, args.loss_fnc) # evaluates the total correct predictions based on a certain batch

            if i % args.eval_every == (args.eval_every - 1):
                acc = evaluate1(model, val, args.batch_size, loss, valid_loss, args.loss_fnc) # determines the validation accuracy
                print('Validation Accuracy: ' + str(acc))
                valid_acc.append(acc) # adds accuracy to the validation accuracy list
                train_acc.append(train_eval / args.eval_every) # adds training accuracy to list
                print('Training Accuracy: ' + str(train_eval / args.eval_every))
                time_diff = time() - first_time # determines the time increment
                print("   Elapsed Time: " + str(time_diff) + " seconds")
                train_loss.append(total_loss / args.eval_every) # adds training loss to the appropriate list
                train_eval = 0 # resets the loss for the next batch
                print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, total_loss / args.eval_every)) # prints the epoch and step
                total_loss = 0.0

    # plot of accuracy vs epochs
    plt.plot(valid_acc, label = "Validation")
    plt.plot(train_acc, label = "Training")
    plt.legend()
    plt.xlabel("Number of Steps")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Number of Steps")
    plt.show()

    # plot of loss vs epochs
    plt.plot(valid_loss, label = "Validation")
    plt.plot(train_loss, label = "Training")
    plt.legend()
    plt.xlabel("Number of Steps")
    plt.ylabel("Loss")
    plt.title("Loss vs. Number of Steps")
    plt.show()

    #torch.save(model.state_dict(), 'MyBest.pt') use to save model if required

if __name__ == "__main__":
    main()