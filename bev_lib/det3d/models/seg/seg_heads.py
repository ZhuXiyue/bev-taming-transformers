import time
from enum import Enum
from functools import reduce
import contextlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import torchplus
# from second.pytorch.core import box_torch_ops
# from second.pytorch.core.losses import (WeightedSigmoidClassificationLoss,
#                                         WeightedSmoothL1LocalizationLoss,
#                                         WeightedSoftmaxClassificationLoss)
# from second.pytorch.models import middle, pointpillars, rpn, voxel_encoder
# from torchplus import metrics
# from second.pytorch.utils import torch_timer


import torch.nn as nn
import torchvision.models

import torch
import torch.nn as nn
import torchvision.models as models
################################
# modified from std res 50unet #
################################
def Deconv(n_input, n_output, k_size=4, stride=2, padding=1):
    Tconv = nn.ConvTranspose2d(
        n_input, n_output,
        kernel_size=k_size,
        stride=stride, padding=padding,
        bias=False)
    block = [
        Tconv,
        nn.BatchNorm2d(n_output),
        nn.LeakyReLU(inplace=True),
    ]
    return nn.Sequential(*block)
        

def Conv(n_input, n_output, k_size=4, stride=2, padding=0, bn=False, dropout=0):
    conv = nn.Conv2d(
        n_input, n_output,
        kernel_size=k_size,
        stride=stride,
        padding=padding, bias=False)
    block = [
        conv,
        nn.BatchNorm2d(n_output),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(dropout)
    ]
    return nn.Sequential(*block)


class Unet_res50(nn.Module):
    def __init__(self, n_classes=7):
        super().__init__()
        
        self.resnet = models.resnet50(pretrained=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)#self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        # get some layer from resnet to make skip connection
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        
        # convolution layer, use to reduce the number of channel => reduce weight number
        self.conv_5 = Conv(2048, 512, 1, 1, 0)
        self.conv_4 = Conv(1024, 512, 1, 1, 0)
        self.conv_3 = Conv(768, 256, 1, 1, 0)
        self.conv_2 = Conv(384, 128, 1, 1, 0)
        self.conv_1 = Conv(128, 64, 1, 1, 0)
        # self.conv_0 = Conv(32, 1, 3, 1, 1)
        
        self.out = nn.Conv2d(32, n_classes, 1)
        
        # deconvolution layer
        self.deconv4 = Deconv(512, 512, 4, 2, 1)
        self.deconv3 = Deconv(512, 256, 4, 2, 1)
        self.deconv2 = Deconv(256, 128, 4, 2, 1)
        self.deconv1 = Deconv(128, 64, 4, 2, 1)
        self.deconv0 = Deconv(64, 32, 4, 2, 1)

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip_1 = x
        
        x = self.maxpool(x)
        x = self.layer1(x)
        skip_2 = x

        x = self.layer2(x)
        skip_3 = x
        x4 = self.layer3(x)
        
        # skip_4 = x
        
        # x5 = self.layer4(x)
        # x5 = self.conv_5(x5)
        
        # x4 = self.deconv4(x5)
        # x4 = torch.cat([x4, skip_4], dim=1)

        x4 = self.conv_4(x4)
        
        x3 = self.deconv3(x4)
        x3 = torch.cat([x3, skip_3], dim=1)
        x3 = self.conv_3(x3)
        
        x2 = self.deconv2(x3)
        x2 = torch.cat([x2, skip_2], dim=1)
        x2 = self.conv_2(x2)
        
        x1 = self.deconv1(x2)
        x1 = torch.cat([x1, skip_1], dim=1)
        x1 = self.conv_1(x1)
        
        x0 = self.deconv0(x1)
        x0 = self.out(x0)
        
        x0 = self.sigmoid(x0)
        return x0
################################
# modified from std res 50unet #
################################

# # 6.9定稿版本
# # 参考：
# # arxiv 1505.04597
# # arxiv 1801.05746，官方实现：https://github.com/ternaus/TernausNet
# # https://blog.csdn.net/github_36923418/article/details/83273107
# # pixelshuffle参考: arxiv 1609.05158

# # modified from standard resnet50 unet
 
# backbone = 'resnet50'

# class DecoderBlock(nn.Module):
#     """
#     U-Net中的解码模块
#     采用每个模块一个stride为1的3*3卷积加一个上采样层的形式
#     上采样层可使用'deconv'、'pixelshuffle', 其中pixelshuffle必须要mid_channels=4*out_channles
#     定稿采用pixelshuffle
#     BN_enable控制是否存在BN，定稿设置为True
#     """
#     def __init__(self, in_channels, mid_channels, out_channels, upsample_mode='pixelshuffle', BN_enable=True):
#         super().__init__()
#         self.in_channels = in_channels
#         self.mid_channels = mid_channels
#         self.out_channels = out_channels
#         self.upsample_mode = upsample_mode
#         self.BN_enable = BN_enable
    
#         self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, bias=False)

#         if self.BN_enable:
#             self.norm1 = nn.BatchNorm2d(mid_channels)
#         self.relu1 = nn.ReLU(inplace=False)
#         self.relu2 = nn.ReLU(inplace=False)

#         if self.upsample_mode=='deconv':
#             self.upsample = nn.ConvTranspose2d(in_channels=mid_channels, out_channels = out_channels,

#                                                 kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
#         elif self.upsample_mode=='pixelshuffle':
#             self.upsample = nn.PixelShuffle(upscale_factor=2)
#         if self.BN_enable:
#             self.norm2 = nn.BatchNorm2d(out_channels)

#     def forward(self,x):
#         x=self.conv(x)
#         if self.BN_enable:
#             x=self.norm1(x)
#         x=self.relu1(x)
#         x=self.upsample(x)
#         if self.BN_enable:
#             x=self.norm2(x)
#         x=self.relu2(x)
#         return x

# class Resnet_Unet(nn.Module):
#     """
#     定稿使用resnet50作为backbone
#     BN_enable控制是否存在BN，定稿设置为True
#     """
#     def __init__(self, BN_enable=True, resnet_pretrain=False):
#         super().__init__()
#         self.BN_enable = BN_enable
#         # encoder部分
#         # 使用resnet34或50预定义模型，由于单通道入，因此自定义第一个conv层，同时去掉原fc层
#         # 剩余网络各部分依次继承
#         # 经过测试encoder取三层效果比四层更佳，因此降采样、升采样各取4次
#         if backbone=='resnet34':
#             resnet = models.resnet34(pretrained=resnet_pretrain)
#             filters=[64,64,128,256,512]
#         elif backbone=='resnet50':
#             resnet = models.resnet50(pretrained=resnet_pretrain)
#             filters=[64,256,512,1024,2048]
#         self.firstconv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.firstbn = resnet.bn1
#         self.firstrelu = resnet.relu
#         self.firstmaxpool = resnet.maxpool
#         self.encoder1 = resnet.layer1
#         self.encoder2 = resnet.layer2
#         self.encoder3 = resnet.layer3

#         # decoder部分
#         self.center = DecoderBlock(in_channels=filters[3], mid_channels=filters[3]*4, out_channels=filters[3], BN_enable=self.BN_enable)
#         self.decoder1 = DecoderBlock(in_channels=filters[3]+filters[2], mid_channels=filters[2]*4, out_channels=filters[2], BN_enable=self.BN_enable)
#         self.decoder2 = DecoderBlock(in_channels=filters[2]+filters[1], mid_channels=filters[1]*4, out_channels=filters[1], BN_enable=self.BN_enable)
#         self.decoder3 = DecoderBlock(in_channels=filters[1]+filters[0], mid_channels=filters[0]*4, out_channels=filters[0], BN_enable=self.BN_enable)
#         if self.BN_enable:
#             self.final = nn.Sequential(
#                 nn.Conv2d(in_channels=filters[0],out_channels=32, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(32), 
#                 nn.ReLU(inplace=False),
#                 nn.Conv2d(in_channels=32, out_channels=9, kernel_size=1),
#                 nn.Sigmoid()
#                 )
#         else:
#             self.final = nn.Sequential(
#                 nn.Conv2d(in_channels=filters[0],out_channels=32, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=False),
#                 nn.Conv2d(in_channels=32, out_channels=9, kernel_size=1), 
#                 nn.Sigmoid()
#                 )

#     def forward(self,x):
#         x = self.firstconv(x)
#         x = self.firstbn(x)
#         x = self.firstrelu(x)
#         x_ = self.firstmaxpool(x)

#         e1 = self.encoder1(x_)
#         e2 = self.encoder2(e1)
#         e3 = self.encoder3(e2)

#         center = self.center(e3)

#         d2 = self.decoder1(torch.cat([center,e2],dim=1))
#         d3 = self.decoder2(torch.cat([d2,e1], dim=1))
#         d4 = self.decoder3(torch.cat([d3,x], dim=1))

#         return self.final(d4)

# # modified from standard resnet18 unet

# def convrelu(in_channels, out_channels, kernel, padding):
#   return nn.Sequential(
#     nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
#     nn.ReLU(inplace=True),
#   )


# class ResNetUNet18(nn.Module):
#   def __init__(self, in_channels,n_class):
#     super().__init__()
    
#     self.base_model = torchvision.models.resnet18(pretrained=False)
#     self.base_layers = list(self.base_model.children())

#     # self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
#     self.layer0_1x1 = convrelu(64, 64, 1, 0)
#     self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
#     self.layer1_1x1 = convrelu(64, 64, 1, 0)
#     self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
#     self.layer2_1x1 = convrelu(128, 128, 1, 0)
#     self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
#     self.layer3_1x1 = convrelu(256, 256, 1, 0)
#     self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
#     self.layer4_1x1 = convrelu(512, 512, 1, 0)

#     self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#     self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
#     self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
#     self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
#     self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

#     self.conv_original_size0 = convrelu(in_channels, 64, 3, 1)
#     # this is the same measurement in lift slat shoot
#     self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
#                             bias=False)
#     self.bn1 = self.base_model.bn1
#     self.relu = self.base_model.relu
#     # this is the same measurement in lift slat shoot

#     self.conv_original_size1 = convrelu(64, 64, 3, 1)
#     self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

#     self.conv_last = nn.Conv2d(64, n_class, 1)

#   def forward(self, input):
#     print("input",input.size())
#     x_original = self.conv_original_size0(input)
#     x_original = self.conv_original_size1(x_original)
#     print("x_ori",x_original.size())
#     # this is the same measurement in lift slat shoot
#     # layer0 = self.layer0(input)
#     # layer0 = self.conv_original_size_self(input)
#     # layer0 = self.conv_original_size_self(input)
#     layer0 = self.conv1(input)
#     layer0 = self.bn1(layer0)
#     layer0 = self.relu(layer0)
#     print("layer0",layer0.size())


#     layer1 = self.layer1(layer0)
#     print("layer1",layer1.size())

#     layer2 = self.layer2(layer1)
#     print("layer2",layer2.size())
#     layer3 = self.layer3(layer2)
#     print("layer3",layer3.size())
#     layer3 = self.layer3_1x1(layer3)
#     print("layer3",layer3.size())
    
#     # layer4 = self.layer4(layer3)
#     # layer4 = self.layer4_1x1(layer4)
#     # x = self.upsample(layer4)

#     x = self.upsample(layer3)
#     print("x_after_up_3",x.size())
    
#     # x = torch.cat([x, layer3], dim=1)
#     # x = self.conv_up3(x)
#     # x = self.upsample(x)

#     layer2 = self.layer2_1x1(layer2)
#     x = torch.cat([x, layer2], dim=1)
#     x = self.conv_up2(x)

#     x = self.upsample(x)
#     layer1 = self.layer1_1x1(layer1)
#     x = torch.cat([x, layer1], dim=1)
#     x = self.conv_up1(x)

#     x = self.upsample(x)
#     layer0 = self.layer0_1x1(layer0)
#     x = torch.cat([x, layer0], dim=1)
#     x = self.conv_up0(x)

#     x = self.upsample(x)
#     x = torch.cat([x, x_original], dim=1)
#     x = self.conv_original_size2(x)

#     out = self.conv_last(x)
#     # use either sigmoid or use BCELOSSwithlogits in loss
#     m = nn.Sigmoid()
#     logits = m(out)
#     return logits
#     # return out


# ################################
# # segmentation head definition #
# ################################
# ## resnet 18 backbone using lift-splat-shoot
# # from efficientnet_pytorch import EfficientNet
# from torchvision.models.resnet import resnet18
# class liftUp(nn.Module):
#     def __init__(self, in_channels, out_channels, scale_factor=2):
#         super().__init__()

#         self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
#                               align_corners=True)

#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         x1 = torch.cat([x2, x1], dim=1)
#         return self.conv(x1)

# class BevEncode(nn.Module):
#     def __init__(self, inC, outC):
#         super(BevEncode, self).__init__()

#         trunk = resnet18(pretrained=False, zero_init_residual=True)
#         self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = trunk.bn1
#         self.relu = trunk.relu

#         self.layer1 = trunk.layer1
#         self.layer2 = trunk.layer2
#         self.layer3 = trunk.layer3

#         self.up1 = liftUp(64+256, 256, scale_factor=4)
#         self.up2 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear',
#                               align_corners=True),
#             nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, outC, kernel_size=1, padding=0),
#         )
        

#     def forward(self, x):
#         print(x.size())
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         print(x.size())
        
#         x1 = self.layer1(x)
#         x = self.layer2(x1)
#         x = self.layer3(x)

#         x = self.up1(x, x1)
#         x = self.up2(x)
#         # use either sigmoid or use BCELOSSwithlogits in loss
#         m = nn.Sigmoid()
#         logits = m(x)
#         return logits
        


# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)

# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)


# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         # if you have padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)

# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         #self.inc = DoubleConv(n_channels, 64)
#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         # self.down4 = Down(512, 1024 // factor)
#         # self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x):
#         # print("debug:",x.size())
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         # x5 = self.down4(x4)
#         # x = self.up1(x5, x4)
#         x = self.up2(x4, x3)
#         x = self.up3(x, x2)
#         # print("debug:",x.size())
#         # print("debugx1:",x1.size())
#         x = self.up4(x, x1)
        
#         ## TODO: make sure if sigmoid is necessary!
#         m = nn.Sigmoid()
#         logits = m(self.outc(x))
#         return logits
# import sys
# from collections import OrderedDict
# class Sequential(torch.nn.Module):
#     r"""A sequential container.
#     Modules will be added to it in the order they are passed in the constructor.
#     Alternatively, an ordered dict of modules can also be passed in.

#     To make it easier to understand, given is a small example::

#         # Example of using Sequential
#         model = Sequential(
#                   nn.Conv2d(1,20,5),
#                   nn.ReLU(),
#                   nn.Conv2d(20,64,5),
#                   nn.ReLU()
#                 )

#         # Example of using Sequential with OrderedDict
#         model = Sequential(OrderedDict([
#                   ('conv1', nn.Conv2d(1,20,5)),
#                   ('relu1', nn.ReLU()),
#                   ('conv2', nn.Conv2d(20,64,5)),
#                   ('relu2', nn.ReLU())
#                 ]))

#         # Example of using Sequential with kwargs(python 3.6+)
#         model = Sequential(
#                   conv1=nn.Conv2d(1,20,5),
#                   relu1=nn.ReLU(),
#                   conv2=nn.Conv2d(20,64,5),
#                   relu2=nn.ReLU()
#                 )
#     """

#     def __init__(self, *args, **kwargs):
#         super(Sequential, self).__init__()
#         if len(args) == 1 and isinstance(args[0], OrderedDict):
#             for key, module in args[0].items():
#                 self.add_module(key, module)
#         else:
#             for idx, module in enumerate(args):
#                 self.add_module(str(idx), module)
#         for name, module in kwargs.items():
#             if sys.version_info < (3, 6):
#                 raise ValueError("kwargs only supported in py36+")
#             if name in self._modules:
#                 raise ValueError("name exists.")
#             self.add_module(name, module)

#     def __getitem__(self, idx):
#         if not (-len(self) <= idx < len(self)):
#             raise IndexError("index {} is out of range".format(idx))
#         if idx < 0:
#             idx += len(self)
#         it = iter(self._modules.values())
#         for i in range(idx):
#             next(it)
#         return next(it)

#     def __len__(self):
#         return len(self._modules)

#     def add(self, module, name=None):
#         if name is None:
#             name = str(len(self._modules))
#             if name in self._modules:
#                 raise KeyError("name exists")
#         self.add_module(name, module)

#     def forward(self, input):
#         # i = 0
#         for module in self._modules.values():
#             # print(i)
#             input = module(input)
#             # i += 1
#         return input



# def kaiming_init(
#     module, a=0, mode="fan_out", nonlinearity="relu", bias=-2.19, distribution="normal"
# ):
#     assert distribution in ["uniform", "normal"]
#     if distribution == "uniform":
#         nn.init.kaiming_uniform_(
#             module.weight, a=a, mode=mode, nonlinearity=nonlinearity
#         )
#     else:
#         nn.init.kaiming_normal_(
#             module.weight, a=a, mode=mode, nonlinearity=nonlinearity
#         )
#     if hasattr(module, "bias") and module.bias is not None:
#         nn.init.constant_(module.bias, bias)

# class SepHead(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         classes = 9, 
#         num_conv = 3,
#         head_conv=64,
#         final_kernel=3,
#         bn=False,
#         init_bias=-2.19,
        
#         **kwargs,
#     ):
#         super(SepHead, self).__init__(**kwargs)

#         self.fc = Sequential()
#         for i in range(num_conv-1): ## 3 
#             self.fc.add(nn.Conv2d(in_channels, head_conv,
#                 kernel_size=final_kernel, stride=1, 
#                 padding=final_kernel // 2, bias=True))
#             if bn:
#                 self.fc.add(nn.BatchNorm2d(head_conv))
#             self.fc.add(nn.ReLU())

#         self.fc.add(nn.Conv2d(head_conv, classes,
#                 kernel_size=final_kernel, stride=1, 
#                 padding=final_kernel // 2, bias=True))    

       
#         # self.fc[-1].bias.data.fill_(init_bias)
#         for m in self.fc.modules():
#                     if isinstance(m, nn.Conv2d):
#                         kaiming_init(m)
#         # self.__setattr__(head, fc)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # print("head input size:",x.size())
#         ret_x = self.fc(x)
#         # print("head_out_size:",ret_x.size())
#         ret_x = self.sigmoid(ret_x)
#         return ret_x

if __name__ == "__main__":
    model = Unet_res50().cpu()
    inp = torch.rand((2, 384, 400, 400)).cpu()
    out = model(inp)
    print(out.size())