import torch.nn as nn
cfg=base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512]#vgg16:D
#3*300*300,64*300*300,64*150*150,128*150*150,128*150*150,128*75*75,256*75*75,256*75*75,256*75*75,256*38*38,512*38*38,512*38*38,512*38*38,512*19*19
#512*19*19,512*19*19,512*19*19

def vgg(channels):
    layers=[]
    in_channels=channels
    for v in cfg:
        if v=='M':
            layers+=[nn.MaxPool2d(2,2)]
        if v=='C':
            layers+=[nn.MaxPool2d(2,2,ceil_mode=True)]
        else:
            layers+=[nn.Conv2d(in_channels,v,kernel_size=3,padding=1),nn.ReLU()]
            in_channels=v
    pool5=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)#512*19*19
    #
    conv6=nn.Conv2d(512,1024,kernel_size=3,padding=1,dilation=6)
    conv7=nn.Conv2d(1024,1024,kernel_size=1)
    layers+=[pool5,conv6,nn.ReLU(),conv7,nn.ReLU()]
    return nn.ModuleList(layers)
