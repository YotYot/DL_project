from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import os
import argparse
import numpy as np
from utils import *

# ours
from hw_models import comb_model1
from models import Net
from dataloader import Dataset

LR0 = 1e-2

def train(net, train_data_loader, test_data_loader,device,num_epochs=10):
    args = get_args()
    net.to(device)
    batch_size = train_data_loader.batch_size
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=LR0)
    optimizer.zero_grad()
    net.zero_grad()
    train_epochs_loss = []
    val_epochs_loss = []
    train_steps_per_e = len(train_data_loader.dataset) // batch_size
    val_steps_per_e   = len(test_data_loader.dataset) // batch_size
    best_loss = 1e5
    for e in range(num_epochs):
        print ("Epoch: ", e)
        net = net.train()
        val_loss_sum = 0.
        train_loss_sum = 0
        for i, data in enumerate(train_data_loader):
            x,y = data
            x = x.to(device)
            y = y.to(device)
            x = x.float()
            out = net(x)
            optimizer.zero_grad()
            loss = criterion(out, y.long())
            loss.backward()
            optimizer.step()
            if i%1000 == 0:
                print('Step: {:3} / {:3} Train loss: {:3.3}'.format(i, train_steps_per_e,loss.item()))
            train_loss_sum += loss.item()
        train_epochs_loss += [train_loss_sum / train_steps_per_e]
        conf_mat = torch.zeros((16, 16), dtype=torch.long)
        net = net.eval()
        total_acc = 0
        for i, val_data in enumerate(test_data_loader):
            x,y = val_data
            x, y = x.to(device), y.to(device)
            x = x.float()
            out = net(x)
            y = y.long()
            loss = criterion(out, y)
            val_loss_sum += loss.item()
            out = torch.argmax(out,dim=1)
            acc = np.count_nonzero(out == y) / len(y)
            total_acc = (total_acc * (i+1) + acc) / (i+2)
        val_epochs_loss += [val_loss_sum / val_steps_per_e]
        if val_epochs_loss[-1] < best_loss:
            print ("Saving Model")
            save_model(net, epoch=e, experiment_name=get_args().experiment_name)
            best_loss = val_epochs_loss[-1]
        print("\nepoch {:3} Train Loss: {:1.5} Val loss: {:1.5} Acc: {:1.5}".format(e, train_epochs_loss[-1],
                                                                        val_epochs_loss[-1], total_acc))
    return train_epochs_loss, val_epochs_loss


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch DepthNet Training')
    parser.add_argument('--load_model', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='load trained model')
    parser.add_argument('--load_path', type=str, default=None,
                        help='load trained model path')
    parser.add_argument('--train', type=lambda x: (str(x).lower() == 'true'), default=True, help='train model')
    parser.add_argument('--epochs', type=int, default=300, help='num of epochs')
    parser.add_argument('--experiment_name', type=str, default='default', help='exp name')
    parser.add_argument('--target', type=str, default='segmentation', help='target - classificaion or segmentation')
    parser.add_argument('--target_mode', type=str, default='discrete', help='target mode - cont or discrete')
    args = parser.parse_args()
    return args

def main():
    torch.set_printoptions(linewidth=320)
    args = get_args()
    print(args)

    # set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    net = Net()

    if args.load_model:
        load_model(net, device, fullpath=args.load_path)

    train_dataset = Dataset(train=True, pickle_name='pickles/comb1.pickle')
    test_dataset  = Dataset(train=False, pickle_name='pickles/comb1.pickle')

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=1)

    if args.train:
        train_loss, val_loss = train(net=net, train_data_loader=train_data_loader, test_data_loader=test_data_loader, device=device, num_epochs=args.epochs)
        _plot_fig(train_loss, val_loss, model_name+'-Losses')
        save_model(net, epoch=args.epochs, experiment_name=args.experiment_name)

if __name__ == '__main__':
    main()
