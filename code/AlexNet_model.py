import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt



class AlexNet(nn.Module):
    """
    原网络为双GPU,此处全部只实现一半
    Input - 3x227x227
    <1>
    C1 - 48@55x55 (11x11 kernel)(4 stride)
    ReLU1
    S1 - 48@27x27 (3x3 kernel, stride 2) Subsampling
    LRN
    <2>
    C2 - 128@27x27 (5x5 kernel)(1 stride)(2 padding)
    ReLU2
    S2 - 128@13x13 (3x3 kernel, stride 2) Subsampling
    LRN
    <3>
    C3 - 192@13x13 (3x3 kernel)(1 stride)(1 padding)
    ReLU3
    <4>
    C4 - 192@13x13 (3x3 kernel)(1 stride)(1 padding)
    ReLU4
    <5>
    C5 - 128@13x13 (3x3 kernel)(1 stride)(1 padding)
    ReLU5
    S5 - 128@6x6 (3x3 kernel, stride 2) Subsampling
    <6>
    C6 - 2048@1x1 (6x6 kernel)
    ReLU6
    Dropout (p=0.5)
    <7>
    F7 - 2048
    ReLU7
    Dropout (p=0.5)
    <8>
    F8 - 6 (Output)
    """
    def __init__(self):
        super(AlexNet, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('C1   ', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
            ('ReLU1', nn.ReLU(inplace=True)),
            ('S1   ', nn.MaxPool2d(kernel_size=3, stride=2)),

            ('C2   ', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
            ('ReLU3', nn.ReLU(inplace=True)),
            ('S2   ', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('C3   ', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
            ('ReLU3', nn.ReLU(inplace=True)),
            ('C4   ', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
            ('ReLU4', nn.ReLU(inplace=True)),
            ('C5   ', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('ReLU5', nn.ReLU(inplace=True)),
            ('S5   ', nn.MaxPool2d(kernel_size=3, stride=2)),

            ('Avg6 ', nn.AdaptiveAvgPool2d((6, 6))),
            ('Drop6', nn.Dropout()),
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('F7   ', nn.Linear(256*6*6, 1024)),
            ('ReLU7', nn.ReLU(inplace=True)),
            ('Drop7', nn.Dropout()),
            ('F8   ', nn.Linear(1024, 6)),
        ]))

    def forward(self, x):
        x = self.convnet(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CarDataset(Dataset):
    """car dataset load"""

    def __init__(self, datapath, train=True):
        """
        Args:
            datapath:filename
            train(optional):traindata or testdata

        """
        self.label = []
        self.image = []
        self.train = train
        self.len = 0
        self.N = 227

        # 加载数据集
        tmp = []
        k = 0
        if train:
            with open(datapath, 'r', encoding='ascii') as fp:
                for line in fp:
                    k += 1
                    if k==1 :
                        self.label.append(int(line))
                    else:
                        img = np.array(line.split(','), dtype = float)
                        tmp.append(img.reshape(self.N, self.N))
                        if k==4:
                            self.image.append(tmp)
                            tmp = []
                            k = 0
        else:
            with open(datapath, 'r', encoding='ascii') as fp:
                for line in fp:
                    k += 1
                    img = np.array(line.split(','), dtype = float)
                    tmp.append(img.reshape(self.N, self.N))
                    if k==3:
                        self.image.append(tmp)
                        tmp = []
                        k = 0
        self.image = torch.from_numpy(np.array(self.image))
        self.len = len(self.image)


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.train:
            return (self.image[idx].float(), torch.tensor(self.label[idx]))
        else:
            return self.image[idx]

    def shape(self):
        print(self.image.shape)
    
    def show(self, idx):
        if self.train:
            print(self.label[idx])
        img = np.array(self.image[idx], dtype=int)
        img = np.transpose(img, (1,2,0))
        plt.imshow(img)
        plt.show()
    



if __name__ == '__main__':

    net = AlexNet()
    print(net)

