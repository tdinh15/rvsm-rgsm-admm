# -*- coding: utf-8 -*-
"""
main pgd resnet
Usage: python main_pgd_resNet.py
"""
import argparse
import os
import shutil
import time
import numpy as np

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

import torch
import torch.nn as nn
import math

from resnet_cifar import *
from utils import *
from models import *

from loss_func import gpls,gpT1,rgsm_ls,rgsm_l0


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--model_name', default='en_resnet20_cifar10', type=str, help='name of the model')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--dev', '--device', default=0, type=int, metavar='N', help='GPU to run on')
parser.add_argument('--lamb', default=1e-2, type=float, help='lambda value')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--num_ensembles', '--ne', default=2, type=int, metavar='N')
parser.add_argument('--model', default='pgd', type=str, help='type of the model')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--noise-coef', '--nc', default=0.1, type=float, metavar='W', help='forward noise (default: 0.0)')
parser.add_argument('--noise-coef-eval', '--nce', default=0.0, type=float, metavar='W', help='forward noise (default: 0.)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT',
                    help='10 for cifar10,100 for cifar100 (default: 10)')


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available
    global best_acc
    best_acc = 0
    start_epoch = 0
    args = parser.parse_args()
    device = 'cuda:'+str(args.dev) if torch.cuda.is_available() else 'cpu'
    
    #--------------------------------------------------------------------------
    # Load Cifar data
    #--------------------------------------------------------------------------
    print('==> Preparing data...')
    root = './data'
    download = True
    
    #normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    beta = 1
    lamb = args.lamb
    lamb2 = 1e-5    

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
    
    
    kwargs = {'num_workers':4, 'pin_memory':True}
    batchsize_test = int(len(test_set)/100)
    print('Batch size of the test set: ', batchsize_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batchsize_test, shuffle=False, **kwargs)
    batchsize_train = 128
    print('Batch size of the train set: ', batchsize_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batchsize_train, shuffle=True, **kwargs)

    criterion = nn.CrossEntropyLoss()

    if args.model == 'pgd':
        net = PGDNet20(ensemble = args.num_ensembles) # default = 2, ensemble = -1 for no emsemble
    elif args.model == 'resnet':
        net = Resnet20(ensemble = args.num_ensembles) # default = 2, ensemble = -1 for no emsemble
    elif args.model == 'resnet86':
        net = ResNet86()
    elif args.model == 'resnet38':
        net = ResNet38()
    net = net.to(device)
    
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
        weights = [ p for n,p in net.named_parameters() if 'weight' in n and len(p.size())==4]
        sp_wt = rgsm_l0(weights, lamb, beta)
        for batch_idx, (x, target) in enumerate(train_loader):
          #if batch_idx < 1:
            optimizer.zero_grad()
            x, target = x.to(device), target.to(device)
            
            ### RGSM
            try:
                score, pert_x = net(x, target)
            except:
                score = net(x)
            loss = criterion(score, target) + gpls(weights, lamb2)
            
            loss.backward()
            for j in range(len(weights)):
                weights[j].data -= lr*beta*(weights[j].data-sp_wt[j].data)
            optimizer.step()
            sp_wt = rgsm_l0(weights, lamb, beta)                   
            
            
            
            train_loss += loss.item()
            _, predicted = torch.max(score.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct.numpy()/total, correct, total))
            
        #----------------------------------------------------------------------
        # Testing
        #----------------------------------------------------------------------       
                
        test_loss = 0; correct = 0; total = 0
        net.eval()
        for j in range(len(sp_wt)):
            k_sp = sp_wt[j]
            k_wt = weights[j]
            k_sp.data, k_wt.data = k_wt.data, k_sp.data
        
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_loader):
                with torch.no_grad():
                    x, target = x.to(device), target.to(device)
                    try:
                        score, pert_x = net(x, target)
                    except:
                        score = net(x)
                    loss = criterion(score, target)
                    test_loss += loss.item()
                    _, predicted = torch.max(score.data, 1)
                    total += target.size(0)
                    correct += predicted.eq(target.data).cpu().sum()
                    progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f %% (%d/%d)'
                                 % (test_loss/(batch_idx+1), 100.*correct.numpy()/total, correct, total))
        
        #----------------------------------------------------------------------
        # Save the checkpoint
        #----------------------------------------------------------------------
        acc = 100.*correct.numpy()/total
        if acc > best_acc:
            print('Saving model...')
            try:
                net_saved = net.basic_net
            except:
                net_saved = net
            state = {
                'net': net_saved, #net,
                'acc': acc,
                'epoch': epoch,
                'param': [beta,lamb,lamb2]
            }
            # torch.save(state, './weights/rgsm'+str(args.num_ensembles)+str(args.model)+str(int(lamb*100))+'.pth')
            torch.save(state, './weights/rgsm'+str(int(lamb*100))+str(args.model)+'.pth')
            best_acc = acc
    
    print('The best acc: ', best_acc)  