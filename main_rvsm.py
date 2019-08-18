# Ensemble of 2 resnet20
# -*- coding: utf-8 -*-
"""
main pgd enresnet
"""
import argparse
import os
import shutil
# shutil.rmtree('C:/Users/Thu')
import time

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

import torch
import torch.nn as nn
import math
import numpy as np

from resnet_cifar import *
from utils import *
from models import *

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--model_name', default='en_resnet20_cifar10', type=str, help='name of the model')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--beta', default=1e-2, type=float, help='beta value')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--model', default='resnet86', type=str, help='type of the model')
parser.add_argument('--dev', '--device', default=0, type=int, metavar='N', help='GPU to run on')
parser.add_argument('--num_ensembles', '--ne', default=1, type=int, metavar='N') # Ensemble of 2
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--noise-coef', '--nc', default=0.1, type=float, metavar='W', help='forward noise (default: 0.1)')
parser.add_argument('--noise-coef-eval', '--nce', default=0.0, type=float, metavar='W', help='forward noise (default: 0.)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT',
                    help='10 for cifar10,100 for cifar10 (default: 10)')

args = parser.parse_args()
device = 'cuda:'+str(args.dev) if torch.cuda.is_available() else 'cpu'
cuda = torch.device('cuda:'+str(args.dev))


if __name__ == '__main__':
    global best_acc
    best_acc = 0
    best_cum = 0
    start_epoch = 0
    
    #--------------------------------------------------------------------------
    # Load Cifar data
    #--------------------------------------------------------------------------
    print('==> Preparing data...')
    root = '../data'
    download = True
    
    #normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    
    lamb = 0.000001
    beta = args.beta
    gamma = np.sqrt(2*lamb/beta)
    
    def thres(w):
        return w.where(torch.abs(w)>gamma, torch.zeros_like(w))
    
    def newgrad(weight):
        w = weight.data
        u = thres(w)
        weight.grad += beta*(w-u)
        
    def sparse_layer(layer):
        newgrad(layer.bn1.weight)
        newgrad(layer.conv1.weight)
        newgrad(layer.bn2.weight)
        newgrad(layer.conv2.weight)
        
    def replace_sparse_weight(layer):
        layer.bn1.weight.data = thres(layer.bn1.weight.data)
        layer.conv1.weight.data = thres(layer.conv1.weight.data)
        layer.bn2.weight.data = thres(layer.bn2.weight.data)
        layer.conv2.weight.data = thres(layer.conv2.weight.data)
    
    
    train_set = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #normalize,
        ]))
    
    test_set = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            #normalize,
        ]))
    
    
    kwargs = {'num_workers':1, 'pin_memory':True}
    batchsize_test = len(test_set)/40 #100
    print('Batch size of the test set: ', batchsize_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=int(batchsize_test),
                                              shuffle=False, **kwargs
                                             )
    batchsize_train = 128
    print('Batch size of the train set: ', batchsize_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=int(batchsize_train),
                                               shuffle=True, **kwargs
                                              )

    
    # net = PGDNet20(ensemble = args.num_ensembles) # default = 2, ensemble = -1 for no emsemble
    if args.model == 'resnet86':
        net = ResNet86()
    elif args.model == 'resnet38':
        net = ResNet38()
    
    net = net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    nepoch = 200
    for epoch in range(nepoch):
        print('Epoch ID', epoch)
        if epoch < 80:
            lr = 0.1
        elif epoch < 120:
            lr = 0.1/10
        elif epoch < 160:
            lr = 0.1/10/10
        else:
            lr = 0.1/10/10/10
        
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        
        #----------------------------------------------------------------------
        # Training
        #----------------------------------------------------------------------
        correct = 0; total = 0; train_loss = 0
        net.train()
        for batch_idx, (x, target) in enumerate(train_loader):
          #if batch_idx < 1:
            optimizer.zero_grad()
            x, target = Variable(x.to(device)), Variable(target.to(device))
            
            score, pert_x = net(x, target)
            loss = criterion(score, target)
            loss.backward()
            
            for i in range(1):
                all_layers = net.basic_net.ensemble[i]
                newgrad(all_layers.conv1.weight)
                for j in range(6):                
                    sparse_layer(all_layers.layer1[j])
                    sparse_layer(all_layers.layer2[j])
                    sparse_layer(all_layers.layer3[j])        
                newgrad(all_layers.bn.weight)
                newgrad(all_layers.fc.weight)       
                
            
            optimizer.step()
            
#             train_loss += loss.data[0]
            train_loss += loss.data.item()
            _, predicted = torch.max(score.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct.numpy()/total, correct, total))
            
        #----------------------------------------------------------------------
        # Testing
        #----------------------------------------------------------------------
        for i in range(args.num_ensembles):
            all_layers = net.basic_net.ensemble[i]
            all_layers.conv1.weight.data = thres(all_layers.conv1.weight.data)
            for j in range(6):
                replace_sparse_weight(all_layers.layer1[j])
                replace_sparse_weight(all_layers.layer2[j])
                replace_sparse_weight(all_layers.layer3[j])
            all_layers.bn.weight.data = thres(all_layers.bn.weight.data)
            all_layers.fc.weight.data = thres(all_layers.fc.weight.data)
        
        
        test_loss = 0; correct = 0; total = 0
        net.eval()
        for batch_idx, (x, target) in enumerate(test_loader):
            with torch.no_grad():
                x, target = Variable(x.to(device)), Variable(target.to(device))
                score, pert_x = net(x, target)
                
                loss = criterion(score, target)
                test_loss += loss.data.item()
                _, predicted = torch.max(score.data, 1)
                total += target.size(0)
                correct += predicted.eq(target.data).cpu().sum()
                progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct.numpy()/total, correct, total))
        
        #----------------------------------------------------------------------
        # Save the checkpoint
        #----------------------------------------------------------------------
        acc = 100.*correct.numpy()/total
        #if acc > best_acc:
        if correct > best_cum:
            print('Saving model...')
            state = {
                'net': net.basic_net, #net,
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, './weights/rvsm'+str(args.beta*100)+str(args.model)+'.pth')
            best_acc = acc
    print('The best acc: ', best_acc, best_cum)
