# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 11:17:25 2018

@author: 涂彦伦
"""
import torch.nn as nn
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
import random
import os
import h5py
from PIL import Image


MAX_ITER = 30000
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
GMMA = 1
BASE_LR = 0.0008
BATCH_SIZE =100

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,path:str):
        super().__init__()
        self.path = path
        self.label = h5py.File(path+'trainlabel.mat','r')
        self.label = self.label['trainlabel'][:]
    def __getitem__(self,index:int):
        data = Image.open(self.path+'train'+str(index+1)+'.jpg')
        label_ = self.label[index]
        data = np.array(data)
        data = torch.from_numpy(data)
        label_ = torch.from_numpy(label_)
        return data,label_
        
class CustomDataset_test(torch.utils.data.Dataset):
    def __init__(self,path:str):
        super().__init__()
        self.path = path
        self.label = h5py.File(path+'testlabel.mat','r')
        self.label = self.label['testlabel'][:]
    def __getitem__(self,index:int):
        data = Image.open(self.path+'test'+str(index+1)+'.jpg')
        label_ = self.label[index]
        data = np.array(data)
        data = torch.from_numpy(data)
        label_ = torch.from_numpy(label_)
        return data,label_
    
class myDCNN(nn.Module):
    def __init__(self):
        super(myDCNN,self).__init__()
        #input 3x256x256
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3,out_channels=16,
                          kernel_size=5,stride=1),
                nn.MaxPool2d(kernel_size=3,stride=2),
                nn.ReLU()
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=16,out_channels=16,
                          kernel_size=5,stride=1),
                nn.MaxPool2d(kernel_size=3,stride=2),
                nn.ReLU()
                )
        self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels=16,out_channels=16,
                          kernel_size=5,stride=1),
                nn.MaxPool2d(kernel_size=3,stride=2),
                nn.ReLU()
                )
        self.fc1 = nn.Linear(27*27*16,50)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(50,4)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x

def test_accuracy(pred_y,test_label):
    sumN,sumA,sumO,sumNoisy=[0,0,0,0]
    sumN1,sumA1,sumO1,sumNoisy1=[0,0,0,0]
    for i in range(test_label.size(0)):
        if test_label[i] == 0:
            sumN = sumN + 1
        elif test_label[i] == 1:
            sumA = sumA + 1
        elif test_label[i] == 2:
            sumO = sumO + 1
        else:
            sumNoisy = sumNoisy + 1
    for i in range(test_label.size(0)):
        if test_label[i]==0 and pred_y[i]==0:
            sumN1 = sumN1 + 1
        elif test_label[i]==1 and pred_y[i]==1:
            sumA1 = sumA1 + 1
        elif test_label[i]==2 and pred_y[i]==2:
            sumO1 = sumO1 + 1
        elif test_label[i]==3 and pred_y[i]==3:
            sumNoisy1 = sumNoisy1 + 1
    return sumN1,sumN,sumA1,sumA,sumO1,sumO,sumNoisy1,sumNoisy


def test(testset,loader_test,mynet):
    mynet.eval()
    N1,N,A1,A,O1,O,Noisy1,Noisy=[0,0,0,0,0,0,0,0]
    for step,(b_x,b_y) in enumerate(loader_test):
        b_x = Variable(b_x).cuda()
        b_y = Variable(b_y).cuda()
        output = mynet(b_x)
        b_y = b_y.long()
        pred_y = torch.max(output,1)[1].data.squeeze().cpu().numpy()
        sumN1,sumN,sumA1,sumA,sumO1,sumO,sumNoisy1,sumNoisy = test_accuracy(pred_y,b_y)
        N1,N,A1,A,O1,O,Noisy1,Noisy = [N1+sumN1,N+sumN,A1+sumA1,A+sumA,O1+sumO1,O+sumO,Noisys1+sumNoisy1,Noisy+sumNoisy]
    print('test accuary: N:',float(N1/N),' A:',float(A1/A),' O:',float(O1/O),' Noisy:',float(Noisy1/Noisy))
        
        
#load test
testset = CustomDataset_test(path='./Resize/Test/')
loader_test =Data.DataLoader(
        dataset = testset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 2
    )
#load train
dataset = CustomDataset(path='./Resize/Train/')
loader = Data.DataLoader(
        dataset = dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 2
    )

mynet = myDCNN()
if torch.cuda.is_available():
     mynet.cuda()
     
opt_SGD = torch.optim.SGD([
        {'params':mynet.parameters()}
        ],lr=BASE_LR,momentum=MOMENTUM,weight_decay=WEIGHT_DECAY)
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(MAX_ITER):
    mynet.train()
    print('Epoch:',epoch)
    for step,(b_x,b_y) in enumerate(loader):
        b_x = Variable(b_x).cuda()
        b_y = Variable(b_y).cuda()
        output = mynet(b_x)
        b_y = b_y.long()
        loss = loss_func(output,b_y)
        opt_SGD.zero_grad()
        loss.backward()
        opt_SGD.step()
        pred_y = torch.max(output,1)[1].data.squeeze().cpu().numpy()
        b_y = b_y.cpu().numpy()
        
        accuracy = float((pred_y == b_y).sum()) / float(b_y.size(0)) 
        print('Epoch:',epoch,'|step:',step,'|loss:%.4f'% loss.data[0],'test_accuracy:%.2f'% accuracy)
test(testset,loader_test,mynet)
torch.save(mynet.state_dict(), './model/mynet_'+str(epoch)+'_'+str(step)+'params.pkl')   #save parameters of net 
