import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
#要想写好这个程序,想要写个vgg
def extra_layers():
    layers=[]
    layers+=[nn.Conv2d(1024,256,kernel_size=1),nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1)]#1024*19*19->256*19*19-->512*10*10
    layers+=[nn.Conv2d(512,128,kernel_size=1),nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1)]#10*10*512--->5*5*256
    layers+=[nn.Conv2d(256,128,kernel_size=1),nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1)]#3*3*256
    layers+=[nn.Conv2d(256,128,kernel_size=1),nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1)]#1*1
    return nn.ModuleList(layers)
def cls_reg_block():
    #每个格多少个先验框,先验框预测位置和分类
    cls_blocks=nn.ModuleList()
    reg_blocks=nn.ModuleList()
    #前边一连串代表先验框数量,后边代表每个图层输出的通道数
    a=zip([4,6,6,6,4,4],[512,1024,512,256,256,256])
    for anchors_per_feature,c_out in a:
            cls_blocks.append(nn.Conv2d(c_out,anchers_per_feature*21,kernel_size=3,padding=1))#分类卷积
            reg_blocks.append(nn.Conv2d(c_out,anchers_per_feature*4,kernel_size=3,padding=1))#回归卷积,4代表坐标
    return cls_blocks,reg_blocks
class SSD(nn.Module):
    def __init__(self):
        super(SSD,self).__init__()
        self.vgg=vgg()
        self.extra=extra_layers()
        self.cls_block,self.reg_block=cls_reg_block()
    def forward(self,x):
        #++++++++++++++++++++++++++++++++++存放用得到的6个特征层,也就是图片分成多少格
        features=[]   
        for i in range(23):
            x=self.vgg[i](x)
        features.append(x)
        for k in range(k,len(self.vgg)):
            x=self.vgg[k](x)
        features.append(x)
        for k,v in enumerate(self.extra):
            x=F.relu(v(x),inplace=True)
            if k%2==1:
                features.append(x)
        #+++++++++++++++++++++++++++++++
        pred_cls=[]
        pred_locs=[]
        for feature,cls_block,reg_block in zip(features,self.cls_block,self.reg_block):
            pred_cls.append(cls_block(feature).permute(0,2,3,1))
            ored_locs.append(reg_block(feature).permute(0,2,3,1))#分类和卷积是并行的两步
        # 将六个特征图每个特征点上的不同anchor预测得出的各类置信度合并到一起
        # [batch_size, num_anchors*num_classes]) ->  [batch_size, num_anchors, num_classes]
        pred_cls = torch.cat([c.reshape(batch_size, -1) for c in pred_cls], dim=1).view(batch_size, -1, cfg_.num_classes)
        # 将六个特征图每个特征点上的不同anchor预测得出的各个修正系数合并到一起
        # [batch_size, num_anchors*4]  ->  [batch_size, num_anchors, 4]
        pred_locs = torch.cat([l.reshape(batch_size, -1) for l in pred_locs], dim=1).view(batch_size, -1, 4)
        return pred_locs, pred_cls



    