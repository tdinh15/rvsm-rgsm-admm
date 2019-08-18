import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
from models import *
from utils import progress_bar

from loss_func import gpls,gpT1,rgsm_ls,rgsm_l0



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--lamb', 
                    default=0.005, type=float, help='lambda value')
parser.add_argument('--beta', 
                    default=10, type=float, help='beta value')
parser.add_argument('--save_freq', 
                    default=20, type=int, help='saving frequency')
#parser.add_argument('--flname', 
#                    default='001', type=str, help='file name')
parser.add_argument('--maxepoch', 
                    default=200, type=int, help='file name')
parser.add_argument('--method', 
                    default='org', type=str, help='org, reg, spl')
parser.add_argument('--path', 
                    default='vgg_spl_1', type=str, help='path to ckpt')
parser.add_argument('--flname', default='admm-1pgd50', type=str, metavar='N', help='name of model file')
parser.add_argument('--num-ensembles', '--ne', default=1, type=int, metavar='N')
parser.add_argument('--noise-coef', '--nc', default=0.0, type=float, metavar='W', help='forward noise (default: 0.0)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
lr = args.lr
num = args.maxepoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
#net = VGG('VGG16')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
#net = net.to(device)
#net = EnResNet20()
#net = EnResNet2_20()
#if device == 'cuda':
#    net = torch.nn.DataParallel(net)
#    cudnn.benchmark = True


path = args.flname+'.pth'
checkpoint = torch.load('weights/'+path, map_location='cuda:0')  
net = checkpoint['net']
print(checkpoint['acc'])

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()


optimizer = optim.SGD(net.parameters(), lr, momentum=0.9, weight_decay=5e-4)


#path = args.path
#checkpoint = torch.load('./weights/'+path+'/ckpt_'+path+'.pth')

#best_acc =checkpoint['acc']
net.eval()
fname = 'net_rgsm'
f_c = open(fname+'.txt', 'w')
param = [ p for n,p in net.named_parameters() if 'weight' in n and len(p.size())==4]
name = [ n for n,p in net.named_parameters() if 'weight' in n and len(p.size())==4]
f_c.write('accuracy: '+str(best_acc)+'\n')
for i in range(len(name)):
    f_c.write(name[i] +'\n')
#    print(name[i] +'\n')
    f_c.write('sz:'+str(list(param[i].size()))+'\n')
#    print('sz:'+str(list(param[i].size()))+'\n')
    tmp = param[i].pow(2).sum(dim=[0,2,3]).pow(1/2.)
    tmp = tmp.data.cpu().numpy()
    np.set_printoptions(precision=9)
    f_c.write(str(tmp)+'\n')
#    print(str(tmp)+'\n')
f_c.close()    
