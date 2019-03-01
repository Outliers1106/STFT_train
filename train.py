"""
Created on Sat Dec 29 11:17:25 2018
@author: 涂彦伦
"""
import torch.nn as nn
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
import random
import os
import h5py
from PIL import Image

MAX_ITER = 30000
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
GMMA = 1
BASE_LR = 0.0005
BATCH_SIZE = 128


class CustomDataset(torch.utils.data.Dataset):
    def __len__(self):
        return self.label.size
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.label = h5py.File(path + 'trainlabel.mat', 'r')
        self.label = self.label['trainlabel'][:]

    def __getitem__(self, index: int):
        data = Image.open(self.path + 'train' + str(index + 1) + '.jpg')
        label_ = self.label[index]
        label_ = label_.squeeze()
        data = np.array(data)
        data = np.transpose(data, (2, 0, 1))
        data = torch.from_numpy(data).float()
        label_ = torch.from_numpy(label_)
        return data, label_


class CustomDataset_test(torch.utils.data.Dataset):
    def __len__(self):
        return self.label.size
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.label = h5py.File(path + 'testlabel.mat', 'r')
        self.label = self.label['testlabel'][:]

    def __getitem__(self, index: int):
        data = Image.open(self.path + 'test' + str(index + 1) + '.jpg')
        label_ = self.label[index]
        label_ = label_.squeeze()
        data = np.array(data)
        data = np.transpose(data,(2,0,1))
        data = torch.from_numpy(data).float()
        label_ = torch.from_numpy(label_)
        return data, label_


class myDCNN(nn.Module):
    def __init__(self):
        super(myDCNN, self).__init__()
        # input 3x256x256
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(27 * 27 * 16, 50)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(50, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x

def F1score(pred_y,label,AA):
    n = label.shape[0]
    #AA = np.zeros((4,4))
    for i in range(n):
        if label[i] == 0:
            if pred_y[i] == 0:
                AA[0][0] = AA[0][0]+1
            elif pred_y[i] ==1:
                AA[0][1] = AA[0][1]+1
            elif pred_y[i] ==2:
                AA[0][2] = AA[0][2]+1
            elif pred_y[i] ==3:
                AA[0][3] = AA[0][3]+1
        elif label[i] == 1:
            if pred_y[i] == 0:
                AA[1][0] = AA[1][0]+1
            elif pred_y[i] ==1:
                AA[1][1] = AA[1][1]+1
            elif pred_y[i] ==2:
                AA[1][2] = AA[1][2]+1
            elif pred_y[i] ==3:
                AA[1][3] = AA[1][3]+1
        elif label[i] == 2:
            if pred_y[i] == 0:
                AA[2][0] = AA[2][0]+1
            elif pred_y[i] ==1:
                AA[2][1] = AA[2][1]+1
            elif pred_y[i] ==2:
                AA[2][2] = AA[2][2]+1
            elif pred_y[i] ==3:
                AA[2][3] = AA[2][3]+1
        elif label[i] == 3:
            if pred_y[i] == 0:
                AA[3][0] = AA[3][0]+1
            elif pred_y[i] ==1:
                AA[3][1] = AA[3][1]+1
            elif pred_y[i] ==2:
                AA[3][2] = AA[3][2]+1
            elif pred_y[i] ==3:
                AA[3][3] = AA[3][3]+1
    return AA
    F1n=2*AA[0][0]/(sum(AA[0][:])+sum(AA[:][0]))
    F1a=2*AA[1][1]/(sum(AA[1][:])+sum(AA[:][1]))
    F1o=2*AA[2][2]/(sum(AA[2][:])+sum(AA[:][2]))
    F1p=2*AA[3][3]/(sum(AA[3][:])+sum(AA[:][3]))
    F1=(F1n+F1a+F1o+F1p)/4
    print('f1n %1.4f,'%F1n,'f1a %1.4f,'%F1a,'flo %1.4f,'%F1o,'f1p %1.4f.'%F1p)
    print('f1 overall %1.4f'%F1)


def testdata(loader_test, mynet):
    mynet.eval()
    AA =np.zeros((4,4))
    for step, (b_x, b_y) in enumerate(loader_test):
        b_x = Variable(b_x).cuda()
        b_y = Variable(b_y).cuda()
        output = mynet(b_x)
        b_y = b_y.long()
        pred_y = torch.max(output, 1)[1].data.squeeze().cpu().numpy()
        AA1 = F1score(pred_y, b_y,AA)
        AA = AA +AA1
    F1n = 2 * AA[0][0] / (sum(AA[0][:]) + sum(AA[:][0]))
    F1a = 2 * AA[1][1] / (sum(AA[1][:]) + sum(AA[:][1]))
    F1o = 2 * AA[2][2] / (sum(AA[2][:]) + sum(AA[:][2]))
    F1p = 2 * AA[3][3] / (sum(AA[3][:]) + sum(AA[:][3]))
    F1 = (F1n + F1a + F1o + F1p) / 4
    print('f1n %1.4f,' % F1n, 'f1a %1.4f,' % F1a, 'flo %1.4f,' % F1o, 'f1p %1.4f.' % F1p)
    print('f1 overall %1.4f' % F1)
    return F1,F1n,F1a,F1o,F1p
    #print('test accuary: N:', float(N1 / N), ' A:', float(A1 / A), ' O:', float(O1 / O), ' Noisy:',float(Noisy1 / Noisy))
    #print('N1:%d/N:%d',N1,N,'A1:%d/A:%d',A1,A,'O1:%d/O:%d',O1,O,'N1:%d/N:%d',Noisy1,Noisy)


def restore_parameters(epoch):
    net = myDCNN()
    net.load_state_dict(torch.load('./model/mynet_'+str(int(epoch))+'_68params.pkl'))
    return net

# load test
testset = CustomDataset_test(path='./Resize/Test/')
loader_test = Data.DataLoader(
    dataset=testset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)
# load train
dataset = CustomDataset(path='./Resize/Train/')
loader = Data.DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

mynet = myDCNN()
#mynet = restore_parameters(300)
if torch.cuda.is_available():
    mynet.cuda()

opt_SGD = torch.optim.SGD([
    {'params': mynet.parameters()}
], lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
loss_func = torch.nn.CrossEntropyLoss()

if __name__ =='__main__':
    loss_save =[]
    F1_save = [[],[],[],[],[]]
    for epoch in range(1000):
        mynet.train()
        print('Epoch:', epoch)
        for step, (b_x, b_y) in enumerate(loader):
            b_x = Variable(b_x).cuda()
            b_y = Variable(b_y).cuda()

            #print('b_y',b_y.shape)
            output = mynet(b_x)
            #print('output',output.shape)
            b_y = b_y.long()
            loss = loss_func(output, b_y)
            loss_save.append(loss.data[0])
            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()
            pred_y = torch.max(output, 1)[1].data.squeeze().cpu().numpy()
            b_y = b_y.cpu().numpy()

            #testdata(loader_test, mynet)
            accuracy = float((pred_y == b_y).sum()) / float(b_y.size)
            print('Epoch:', epoch, '|step:', step, '|loss:%.4f' % loss.data[0], 'train_accuracy:%.2f' % accuracy)
        scoreAll,scoreN,scoreA,scoreO,scoreP = testdata(loader_test, mynet)

        F1_save[0].append(scoreAll)
        F1_save[1].append(scoreN)
        F1_save[2].append( scoreA)
        F1_save[3].append( scoreO)
        F1_save[4].append( scoreP)
        torch.save(mynet.state_dict(), './modellr0005relu/mynet_' + str(epoch) + '_' + '68params.pkl')  # save parameters of net
    plt.plot(loss_save)
    plt.savefig('./lossimage/loss_lr0005relu.png')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.close()
    labels=['F1','F1n','F1a','F1o','F1p']
    for i,l_his in enumerate(F1_save):
         plt.plot(l_his,label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.savefig('./lossimage/F1lr0005relu.png')
    plt.close()