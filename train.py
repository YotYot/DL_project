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
# from hw_models import comb_model1
from hw_models import model_generater
from models import Net
from dataloader import Dataset
from hyperopt import STATUS_OK
from utils import _plot_fig

import nni

LR0 = 1e-3
epsilon = 1

class agent():
    def __init__(self, verbose=False, num_of_inputs=10, depth=10):
        torch.set_printoptions(linewidth=320)
        args = self.get_args()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        batch_size = 10

        self.hw_model = model_generater(inputs=num_of_inputs, depth=depth)


        train_dataset = Dataset(train=True, model=self.hw_model)
        test_dataset = Dataset(train=False, model=self.hw_model)

        self.train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                             num_workers=1)
        self.test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                                            num_workers=1)
        self.batch_size = batch_size
        self.verbose = verbose
        self.best_loss = 1e5
        # self.criterion = nn.MSELoss()
        self.criterion = nn.BCELoss()
        torch.manual_seed(42)


    def train(self, net, train_data_loader, test_data_loader,device,num_epochs, learning_rate):
        args = self.get_args()
        net.to(device)
        batch_size = self.batch_size
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        net.zero_grad()
        train_epochs_loss = []
        val_epochs_loss = []
        err_epochs = []
        train_steps_per_e = len(train_data_loader.dataset) // batch_size
        val_steps_per_e   = len(test_data_loader.dataset) // batch_size
        for e in range(num_epochs):
            if self.verbose:
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
                out = torch.squeeze(out)
                out = torch.clamp(out,0,1)
                loss = self.criterion(out.view(-1), y.view(-1))
                loss.backward()
                optimizer.step()
                if self.verbose:
                    if i%1000 == 0:
                        print('Step: {:3} / {:3} Train loss: {:3.3}'.format(i, train_steps_per_e,loss.item()))
                train_loss_sum += loss.item()
            train_epochs_loss += [train_loss_sum / train_steps_per_e]
            conf_mat = torch.zeros((16, 16), dtype=torch.long)
            net = net.eval()
            total_acc = 0
            err = 0
            for i, val_data in enumerate(test_data_loader):
                x,y = val_data
                x, y = x.to(device), y.to(device)
                x = x.float()
                with torch.no_grad():
                    out = net(x)
                out = torch.squeeze(out)
                y = y.float()
                loss = self.criterion(out.view(-1), y.view(-1))
                val_loss_sum += loss.item()
                # out = torch.argmax(out,dim=1)
                out = out.view(-1).detach().cpu().numpy()
                y = y.view(-1).detach().cpu().numpy()
                acc = np.count_nonzero(np.round(out) == y) / y.size
                err += np.sum(np.abs(out - y))
                total_acc = (total_acc * (i) + acc) / (i+1)
                total_err = 1 - total_acc
            val_epochs_loss += [val_loss_sum / val_steps_per_e]
            err_epochs = np.append(err_epochs, err)
            if val_epochs_loss[-1] < self.best_loss:
                if self.verbose:
                    print ("Saving Model")
                # save_model(net, epoch=e, experiment_name=self.get_args().experiment_name)
                save_latest_model(net, experiment_name=self.get_args().experiment_name)
                self.best_loss = val_epochs_loss[-1]
            if self.verbose:
                print("\nepoch {:3} Train Loss: {:1.5} Val loss: {:1.5} Acc: {:1.5} Err: {:1.5}".format(e, train_epochs_loss[-1],
                                                                            val_epochs_loss[-1], total_acc, str(err)))
        return train_epochs_loss, val_epochs_loss, err_epochs, total_acc, total_err

    def test(self, net, load_model=True):
        if load_model:
            load_latest_model(net, device=self.device, experiment_name=self.get_args().experiment_name)
        val_loss_sum = 0
        err = 0
        val_epochs_loss = []
        err_epochs = []
        total_acc = 0
        val_steps_per_e = len(self.test_data_loader.dataset) // self.batch_size
        for i, val_data in enumerate(self.test_data_loader):
            x, y = val_data
            x, y = x.to(self.device), y.to(self.device)
            x = x.float()
            with torch.no_grad():
                out = net(x)
            out = torch.squeeze(out)
            y = y.float()
            loss = self.criterion(out.view(-1), y.view(-1))
            val_loss_sum += loss.item()
            # out = torch.argmax(out,dim=1)
            out = out.view(-1).detach().cpu().numpy()
            y = y.view(-1).detach().cpu().numpy()
            acc = np.count_nonzero(np.round(out) == y) / len(y)
            err += np.sum(np.abs(out - y))
            total_acc = (total_acc * (i) + acc) / (i + 1)
        val_epochs_loss += [val_loss_sum / val_steps_per_e]
        err_epochs = np.append(err_epochs, err)
        return (1-total_acc)


    def get_args(self):
        parser = argparse.ArgumentParser(description='PyTorch DepthNet Training')
        parser.add_argument('--load_model', type=lambda x: (str(x).lower() == 'true'), default=False,
                            help='load trained model')
        parser.add_argument('--load_path', type=str, default=None,
                            help='load trained model path')
        parser.add_argument('--train', type=lambda x: (str(x).lower() == 'true'), default=True, help='train model')
        parser.add_argument('--epochs', type=int, default=100, help='num of epochs')
        parser.add_argument('--experiment_name', type=str, default='default', help='exp name')
        parser.add_argument('--target', type=str, default='segmentation', help='target - classificaion or segmentation')
        parser.add_argument('--target_mode', type=str, default='discrete', help='target mode - cont or discrete')
        args = parser.parse_args()
        return args

    def objective(self,args):
        num_of_neurons = int(args['num_of_neurons'])
        num_of_hidden_layers = int(args['num_of_hidden_layers'])
        learning_rate = args['learning_rate']
        epochs = int(args['epochs'])
        net = Net(num_of_neurones=num_of_neurons, num_of_hidden_layers=num_of_hidden_layers,num_of_inputs=self.hw_model.inputs, num_of_outputs=self.hw_model.outputs)
        train_loss, val_loss, err_loss, acc, err = self.train(net=net, train_data_loader=self.train_data_loader,
                                               test_data_loader=self.test_data_loader, device=self.device, num_epochs=epochs,
                                               learning_rate=learning_rate)
        return {
            'loss': val_loss[-1],
            'acc': acc,
            'status': STATUS_OK,
            'discrete_acc': acc,
        }




        return

    def main(self):
        num_of_neurons = 151
        num_of_hidden_layers = 2
        learning_rate = 0.001
        epochs = 10
        args = dict()
        args['num_of_neurons'], args['num_of_hidden_layers'], args['learning_rate'], args['epochs'] = num_of_neurons, num_of_hidden_layers,learning_rate, epochs
        loss = self.objective(args)
        print (loss)
        net = Net(num_of_neurones=num_of_neurons, num_of_hidden_layers=num_of_hidden_layers, num_of_inputs=self.hw_model.inputs, num_of_outputs=self.hw_model.outputs)
        net = net.to(self.device)
        print (self.test(net))
        # error_accumulated = []
        # train_loss_accumulated = []
        # val_loss_accumulated = []
        # if args.train:
            # net = Net(num_of_neurones=500 * i)
            # train_loss, val_loss, err_loss = train(net=net, train_data_loader=train_data_loader, test_data_loader=test_data_loader, device=device, num_epochs=args.epochs, learning_rate=params['learning_rate'])
            # _plot_fig(train_loss, val_loss,err_loss, 'Losses')
            # save_model(net, epoch=args.epochs, experiment_name=args.experiment_name)
        #     error_accumulated.append(err_loss[-1])
        #     train_loss_accumulated.append(train_loss[-1])
        #     val_loss_accumulated.append(val_loss[-1])
        # _plot_fig(train_loss_accumulated, val_loss_accumulated, error_accumulated, 'Accumulated Values')

if __name__ == '__main__':
    agent = agent(verbose=True)
    agent.main()
