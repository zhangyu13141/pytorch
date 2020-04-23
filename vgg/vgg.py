#输入图象是224*224
#预处理是从每个像素减去平均RGB值
#3*3,1*1
#stride=1,padding=1.maxpooling:2*2,stride=2
#卷积层后跟三个完全连接FC层
#Relu()
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self,features):
        super(VGG,self).__init__()
        self.features=features#cnn
        self.classifiar=nn.Sequential(
            nn.Linear(4096,4096),
            nn.ReLU()
            nn.Linear(4096,1000),
            nn.ReLU()
            nn.Linear(1000,10)
        )
    def forward(self,x):
            x=self.features(x)
            x=x.view(x.size(0),-1)
            x=self.classifiar(x)
            return x
cfg={
    'vgg-A':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M']
    'vgg-B':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M']
    #'vgg-C':[64,64,'M',128,128,'M',256,256,'M',]
    'vgg-D':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']
    'vgg-E':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']

}
def make_layers(cf):
    layers=[]
    in_channels=3
    for v in cf:
        if v=='M':
            layers+=[nn.MaxPool2d(2,2)]
        else:
            c=nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
            layers+=[c,nn.ReLU()]
            in_channels=v
    return nn.Sequential(*layers)
make_layer(cfg['vgg-A'])
def vgg11():
    return VGG(make_layers(cfg['vgg-A']))
def vgg13():
    return VGG(make_layers(cfg['vgg-B']))
def vgg16():
    return VGG(make_layers(cfg['vgg-D']))
def vgg19():
    return VGG(make_layers(cfg['vgg-E']))

    




