# -*- coding: utf-8 -*-
"""
Attack EnResNet
"""
#------------------------------------------------------------------------------
# System module.
#------------------------------------------------------------------------------
import os
import random
import time
import copy
import argparse
import sys

#------------------------------------------------------------------------------
# Torch module.
# We used torch to build the WNLL activated DNN. Note torch utilized the
#dynamical computational graph, which is appropriate for our purpose, since
#in our model, we involves nearest neighbor searching, which is too slow by
#symbolic computing.
#------------------------------------------------------------------------------
from models import *
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

#------------------------------------------------------------------------------
# Numpy and other common module.
#------------------------------------------------------------------------------
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import pickle
#import cPickle

parser = argparse.ArgumentParser(description='Fool Standard Neural Nets')
ap = parser.add_argument
ap('-method', help='Attack Method', type=str, default="fgsm") # fgsm, ifgsm, cwl2
ap('-epsilon', help='Attack Strength', type=float, default=0.031) #1./255.
opt = vars(parser.parse_args())

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    """
    The main function, load the DNN, and fooled the correctly classified data,
    and save the fooled data to file.
    """
    #--------------------------------------------------------------------------
    # Load the neural nets
    #--------------------------------------------------------------------------
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('ckpt_normal.pth')
    
    net = EnResNet20()
    net = net.to(device)
    net.load_state_dict(checkpoint['net'])
    net = net.basic_net
    
    epsilon = opt['epsilon']
    attack_type = opt['method']
    #--------------------------------------------------------------------------
    # Load the original test set
    #--------------------------------------------------------------------------
    print('==> Preparing data...')
    root = './data'
    download = False
    
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    
    test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    
    kwargs = {'num_workers':1, 'pin_memory':True}
    batchsize_test = 200
    if attack_type == 'cw':
        batchsize_test = 1
    print('Batch size of the test set: ', batchsize_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batchsize_test,
                                              shuffle=False, **kwargs
                                             )
    
    
    #--------------------------------------------------------------------------
    # Testing
    #--------------------------------------------------------------------------
    # images: the original images
    # labels: labels of the original images
    # images_adv: the perturbed images
    # labels_pred: the predicted labels of the perturbed images
    # noise: the added noise
    images, labels, images_adv, labels_pred, noise = [], [], [], [], []
    
    total_fooled = 0; total_correct_classified = 0
    criterion = nn.CrossEntropyLoss()
    
    if attack_type == 'fgsm':
      for batch_idx, (x1, y1_true) in enumerate(test_loader):
        x_Test = x1.numpy()
        x_Test = ((x_Test - x_Test.min())/(x_Test.max() - x_Test.min()) - 0.5)*2
        y_Test = y1_true.numpy()
        
        x = Variable(torch.cuda.FloatTensor(x_Test.reshape(1, 3, 32, 32)), requires_grad=True)
        y = Variable(torch.cuda.LongTensor(y_Test), requires_grad=False)
        
        # Classification before perturbation
        pred_tmp = net(x)
        loss = criterion(pred_tmp, y)
        y_pred = np.argmax(pred_tmp.cpu().data.numpy())
        
        # Attack
        net.zero_grad()
        if x.grad is not None:
            x.grad.data.fill_(0)
        loss.backward()
        
        #x_val_min = 0.0
        x_val_min = -1.0
        x_val_max = 1.0 
        x.grad.sign_()
        #x_adversarial = x - epsilon*x.grad # This improves generazation when epsilon is small
        x_adversarial = x + epsilon*x.grad
        x_adversarial = torch.clamp(x_adversarial, x_val_min, x_val_max)
        x_adversarial = x_adversarial.data
        
        # Classify the perturbed data
        x_adversarial_tmp = Variable(x_adversarial)
        pred_tmp = net(x_adversarial_tmp)
        loss = criterion(pred_tmp, y)
        y_pred_adversarial = np.argmax(pred_tmp.cpu().data.numpy())
        
        if y_Test == y_pred_adversarial:
            total_correct_classified += 1
        
        # Save the perturbed data
        images.append(x_Test) # Original image
        images_adv.append(x_adversarial.cpu().numpy()) # Perturbed image
        noise.append(x_adversarial.cpu().numpy()-x_Test) # Noise
        labels.append(y_Test)
        labels_pred.append(y_pred_adversarial)
    
    elif attack_type == 'ifgsm':
      for batch_idx, (x1, y1_true) in enumerate(test_loader):
       #if batch_idx < 100:
        x_Test = x1.numpy()
        x_Test = ((x_Test - x_Test.min())/(x_Test.max() - x_Test.min()) - 0.5)*2
        y_Test = y1_true.numpy()
        
        x = Variable(torch.cuda.FloatTensor(x_Test.reshape(1, 3, 32, 32)), requires_grad=True)
        y = Variable(torch.cuda.LongTensor(y_Test), requires_grad=False)
        
        # Classification before perturbation
        pred_tmp = net(x)
        loss = criterion(pred_tmp, y)
        y_pred = np.argmax(pred_tmp.cpu().data.numpy())
        
        # Attack
        iteration = 20#5#6#7#8#9#3#10
        x_val_min = 0; x_val_max = 1.0
        alpha = 2./255.#0.03
        epsilon = 8./255.
        # Helper function
        def where(cond, x, y):
            """
            code from :
            https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
            """
            cond = cond.float()
            return (cond*x) + ((1-cond)*y)
        
        x_adv = Variable(x.data, requires_grad=True)
        for i in range(iteration):
            h_adv = net(x_adv)
            loss = criterion(h_adv, y)
            #loss = -loss
            net.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            
            loss.backward()
            
            x_adv.grad.sign_()
            #x_adv = x_adv - alpha*x_adv.grad
            x_adv = x_adv + epsilon*x_adv.grad
            x_adv = where(x_adv > x+alpha, x+alpha, x_adv)
            x_adv = where(x_adv < x-alpha, x-alpha, x_adv)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)
            
        x_adversarial = x_adv.data
        
        # Classify the perturbed data
        x_adversarial_tmp = Variable(x_adversarial)
        pred_tmp = net(x_adversarial_tmp)
        loss = criterion(pred_tmp, y)
        y_pred_adversarial = np.argmax(pred_tmp.cpu().data.numpy())
        
        if y_Test == y_pred_adversarial:
            total_correct_classified += 1
        
        # Save the perturbed data
        images.append(x_Test) # Original image
        images_adv.append(x_adversarial.cpu().numpy()) # Perturbed image
        noise.append(x_adversarial.cpu().numpy()-x_Test) # Noise
        labels.append(y_Test)
        labels_pred.append(y_pred_adversarial)
    elif attack_type == 'cwl2':
      for batch_idx, (x1, y1_true) in enumerate(test_loader):
        x_Test = x1.numpy()
        y_Test = y1_true.numpy()
        
        x = Variable(torch.cuda.FloatTensor(x_Test.reshape(1, 3, 32, 32)), requires_grad=True)
        y = Variable(torch.cuda.LongTensor(y_Test), requires_grad=False)
        
        # Classification before perturbation
        pred_tmp = net(x)
        loss = criterion(pred_tmp, y)
        
        y_pred = np.argmax(pred_tmp.cpu().data.numpy())
        
        # Attack
        cwl2_learning_rate = 0.01
        max_iter = 10# 100 # BW Change 100 to 10
        lambdaf = 10.0
        kappa = 0.0
        
        # The input image we will perturb 
        input = torch.FloatTensor(x_Test.reshape(1,3,32,32))
        input_var = Variable(input)
        
        # w is the variable we will optimize over. We will also save the best w and loss
        w = Variable(input, requires_grad=True) 
        best_w = input.clone()
        best_loss = float('inf')
        
        # Use the Adam optimizer for the minimization
        optimizer = optim.Adam([w], lr=cwl2_learning_rate)
        
        # Get the top2 predictions of the model. Get the argmaxes for the objective function
        probs = net(input_var.cuda())
        probs_data = probs.data.cpu()
        top1_idx = torch.max(probs_data, 1)[1]
        probs_data[0][top1_idx] = -1 # making the previous top1 the lowest so we get the top2
        top2_idx = torch.max(probs_data, 1)[1]
        
        # Set the argmax (but maybe argmax will just equal top2_idx always?)
        argmax = top1_idx[0]
        argmax = argmax.numpy()
        if argmax == y_pred:
            argmax = top2_idx[0]
        
        # The iteration
        for i in range(0, max_iter):
            if i > 0:
                w.grad.data.fill_(0)
            
            # Zero grad (Only one line needed actually)
            net.zero_grad()
            optimizer.zero_grad()
            
            # Compute L2 Loss
            loss = torch.pow(w - input_var, 2).sum()
            
            # w variable
            w_data = w.data
            w_in = Variable(w_data, requires_grad=True)
            
            # Compute output
            output = net.forward(w_in.cuda()) #second argument is unneeded
            
            # Calculating the (hinge) loss
            loss += lambdaf * torch.clamp( output[0][y_pred] - output[0][argmax] + kappa, min=0).cpu()
            
            # Backprop the loss
            loss.backward()
            
            # Work on w (Don't think we need this)
            w.grad.data.add_(w_in.grad.data)
            
            # Optimizer step
            optimizer.step()
            
            # Save the best w and loss
            #total_loss = loss.data.cpu()[0]
            total_loss = loss.cpu().item()
            
            if total_loss < best_loss:
                best_loss = total_loss
                
                ##best_w = torch.clamp(best_w, 0., 1.) # BW Added Aug 26
                
                best_w = w.data.clone()
        
        # Set final adversarial image as the best-found w
        x_adversarial = best_w
        
        ##x_adversarial = torch.clamp(x_adversarial, 0., 1.) # BW Added Aug 26
        
        #--------------- Add to introduce the noise
        noise_tmp = x_adversarial.cpu().numpy() - x_Test
        x_adversarial = x_Test + epsilon * noise_tmp
        #---------------
        
        # Classify the perturbed data
        x_adversarial_tmp = Variable(torch.cuda.FloatTensor(x_adversarial), requires_grad=False) #Variable(x_adversarial).cuda()
        pred_tmp = net(x_adversarial_tmp)
        loss = criterion(pred_tmp, y)
        y_pred_adversarial = np.argmax(pred_tmp.cpu().data.numpy())
        
        if y_Test == y_pred_adversarial:
            total_correct_classified += 1
    
    print('Number of correctly classified images: ', total_correct_classified)
    
    '''
    # Save data
    #with open("Adversarial" + attack_type + str(int(10*epsilon)) + ".pkl", "w") as f:
    with open("Adversarial" + attack_type + str(int(100*epsilon)) + ".pkl", "w") as f:
        adv_data_dict = {"images":images_adv, "labels":labels}
        cPickle.dump(adv_data_dict, f)
    '''
    
    '''
    with open("fooled_" + attack_type + str(int(100*epsilon)) + ".pkl", "w") as f:
        adv_data_dict = {
            "images" : images,
            "images_adversarial" : images_adv,
            "y_trues" : labels,
            "noises" : noise,
            "y_preds_adversarial" : labels_pred
            }
        pickle.dump(adv_data_dict, f)
   '''
