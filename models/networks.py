from typing_extensions import Self
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler
from torchvision import models
import torchvision.models as models
from collections import OrderedDict
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
import functools
import random
import sys
sys.path.append("..")
from util import util
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _quadruple
from torch.autograd import Variable
import re
import cupy
import cupy as cupyutil
from os import path as osp
import cv2
import math
import kornia



###############################################################################
# Helper Functions
###############################################################################

class Vgg19(nn.Module):
    """ Vgg19 for VGGLoss. """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class Sobel(nn.Module):
    """ Soebl operator to calculate depth grad. """

    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """x: depth map (batch_size,1,H,W)"""
        out = self.edge_conv(x) 
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
  
        return out

class FeatureExtraction(nn.Module):
    """ 
    size: 512-256-128-64-32-32-32
    channel: in_nc-64-128-256-512-512-512
    """
    def __init__(self, input_nc, ngf=64, n_layers=3, norm_layer=nn.InstanceNorm2d,  use_dropout=False ):
        super(FeatureExtraction, self).__init__()
        
        downconv = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        model = [downconv, nn.ReLU(inplace=True), norm_layer(ngf)]
        for i in range(n_layers):
            in_ngf = 2**i * ngf if 2**i * ngf < 512 else 512
            out_ngf = 2**(i+1) * ngf if 2**i * ngf < 512 else 512
            downconv = nn.Conv2d(in_ngf, out_ngf, kernel_size=4, stride=2, padding=1)
            model += [downconv, nn.ReLU(inplace=True)]
            model += [norm_layer(out_ngf)]
        model += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)]
        model += [norm_layer(512)]
        model += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)]

        self.model = nn.Sequential(*model)
        # init_weights(self.model, init_type='normal

    def forward(self, input):
        
        return self.model(input)



    

class VGG(nn.Module):

    def __init__(self, in_channels, init_weights=True ,batch_norm=False):
        super(VGG, self).__init__()
        cfgs = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        layers = []
        in_chan = in_channels
        for v in cfgs['D']:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_chan, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_chan = v
        self.features = nn.Sequential(*layers)       
        '''
        if init_weights:
            self._initialize_weights()
        '''
    def forward(self, input):
        
        return self.features(input)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    


'''
class Vgg16_net(nn.Module):
    def __init__(self ,in_channels):
        super(Vgg16_net, self).__init__()
 
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),  # 224 * 224 * 64
            nn.BatchNorm2d(64), # Batch Normalization强行将数据拉回到均值为0，方差为1的正太分布上，一方面使得数据分布一致，另一方面避免梯度消失。
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 224 * 224 * 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
 
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112 * 112 * 64
        )
 
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # 112 * 112 * 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # 112 * 112 * 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
 
            nn.MaxPool2d(2, 2)  # 56 * 56 * 128
        )
 
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # 56 * 56 * 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # 56 * 56 * 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # 56 * 56 * 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
 
            nn.MaxPool2d(2, 2)  # 28 * 28 * 256
        )
 
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1), # 28 * 28 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 28 * 28 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 28 * 28 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
 
            nn.MaxPool2d(2, 2)  # 14 * 14 * 512
        )
 
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 14 * 14 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 14 * 14 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 14 * 14 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
 
            nn.MaxPool2d(2, 2)  # 7 * 7 * 512
        )
 
        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )
 
    def forward(self, x):
        x = self.conv(x)

        return x
'''

class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)

'''
class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()
    
    def forward(self, feature_A, feature_B):
        b,c,h,w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
        feature_B = feature_B.view(b,c,h*w).transpose(1,2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B,feature_A)
        correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
        return correlation_tensor

'''


class FeatureCorrelation(torch.nn.Module):
    def __init__(self, shape='3D', normalization=True):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.shape = shape
        self.ReLU = nn.ReLU()

    def forward(self, feature_source, feature_target):
        # feature_A is source, B is target
        if self.shape == '3D':
            '''
            # the usual correlation
            b, c, h, w = feature_source.size()
            # reshape features for matrix multiplication
            feature_source = feature_source.transpose(2, 3).contiguous().view(b, c, h * w)
            feature_target = feature_target.view(b, c, h * w).transpose(1, 2)
            # perform matrix mult.
            feature_mul = torch.bmm(feature_target, feature_source)
            # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b ,h ,w , h *w).transpose(2 ,3).transpose(1 ,2)
            '''
            b,c,h,w = feature_target.size()
            # reshape features for matrix multiplication
            feature_target = feature_target.transpose(2,3).contiguous().view(b,c,h*w)
            feature_source = feature_source.view(b,c,h*w).transpose(1,2)
            # perform matrix mult.
            feature_mul = torch.bmm(feature_source,feature_target)
            correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
            
        elif self.shape =='4D':
            b, c, hsource, wsource = feature_source.size()
            b, c, htarget, wtarget = feature_target.size()
            # reshape features for matrix multiplication
            feature_source = feature_source.view(b, c, hsource * wsource).transpose(1, 2) # size [b,hsource*wsource,c]
            feature_target = feature_target.view(b, c, htarget * wtarget) # size [b,c,htarget*wtarget]
            # perform matrix mult.
            feature_mul = torch.bmm(feature_source, feature_target) # size [b, hsource*wsource, htarget*wtarget]
            correlation_tensor = feature_mul.view(b ,hsource ,wsource ,htarget ,wtarget).unsqueeze(1)
            # size is [b, 1, hsource, wsource, htarget, wtarget]

        if self.normalization:
            correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))

        return correlation_tensor




class FeatureRegression(nn.Module):
    def __init__(self, input_nc=640, output_dim=6):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(64 * 8 * 5, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x) # (batch_size,64,8,5)
        x = x.reshape(x.size(0), -1) # (batch_size,2560)
        x = self.linear(x) # (batch_size,output_dim)
        x = self.tanh(x)
        return x

class AffineGridGen(nn.Module):
    def __init__(self, out_h=512, out_w=320, out_ch = 3):
        super(AffineGridGen, self).__init__()        
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch
        
    def forward(self, theta):
        bs=theta.size()[0]
        if not theta.size()==(bs,2,3):
            theta = theta.view(-1,2,3)
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size,self.out_ch,self.out_h,self.out_w))
        return F.affine_grid(theta, out_size)

class TpsGridGen(nn.Module):
    def __init__(self, out_h=512, out_w=320, use_regular_grid=True, grid_size=9, device='cpu'):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w

        # create grid in numpy
        self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32) # (512,320,3)
        # sampling grid using meshgrid
        self.grid_X,self.grid_Y = np.meshgrid(np.linspace(-1,1,out_w),np.linspace(-1,1,out_h))
        # grid_X,grid_Y: size [1,H,W,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3) # (1,512,320,1)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3) # (1,512,320,1)
        if device != 'cpu':
            self.grid_X = self.grid_X.to(device)
            self.grid_Y = self.grid_Y.to(device)

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1,1,grid_size)
            self.N = grid_size*grid_size # 25 control points
            P_Y, P_X = np.meshgrid(axis_coords,axis_coords) # BUG: should return (P_X, P_Y)?
            # P_X, P_Y = np.meshgrid(axis_coords,axis_coords)
            P_X = np.reshape(P_X,(-1,1)) # size (N=25,1)
            P_Y = np.reshape(P_Y,(-1,1)) # size (N=25,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.P_X_base = P_X.clone() # size (N=25,1)
            self.P_Y_base = P_Y.clone() # size (N=25,1)
            self.Li = self.compute_L_inverse(P_X,P_Y).unsqueeze(0) # (1,N+3=28,N+3=28)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4) # (1,1,1,1,N=25)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4) # (1,1,1,1,N=25)
            if device != 'cpu':
                self.P_X = self.P_X.to(device)
                self.P_Y = self.P_Y.to(device)
                self.P_X_base = self.P_X_base.to(device)
                self.P_Y_base = self.P_Y_base.to(device)
                self.Li = self.Li.to(device)

            
    def forward(self, theta):
        # theta.size(): (batch_size, N*2=50)
        warped_grid = self.apply_transformation(theta,torch.cat((self.grid_X,self.grid_Y),3)) # (batch_size,512,512,2)
        
        return warped_grid
    
    def compute_L_inverse(self,X,Y):
        N = X.size()[0] # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N,N)
        Ymat = Y.expand(N,N)
        # a quick way to calculate distances between every control point pairs
        P_dist_squared = torch.pow(Xmat-Xmat.transpose(0,1),2)+torch.pow(Ymat-Ymat.transpose(0,1),2)
        P_dist_squared[P_dist_squared==0]=1 # make diagonal 1 to avoid NaN in log computation
        # the TPS kernel funciont $U(r) = r^2*log(r)$
        # K.size: (N,N)
        K = torch.mul(P_dist_squared,torch.log(P_dist_squared)) # BUG: should be torch.log(torch.sqrt(P_dist_squared))?
        # construct matrix L
        Z = torch.FloatTensor(N,1).fill_(1)
        O = torch.FloatTensor(3,3).fill_(0)       
        P = torch.cat((Z,X,Y),1) # (N,3)
        L = torch.cat((torch.cat((K,P),1),torch.cat((P.transpose(0,1),O),1)),0) # (N+3,N+3)
        Li = torch.inverse(L) # (N+3,N+3)

        return Li
        
    def apply_transformation(self,theta,points):
        if theta.dim()==2:
            theta = theta.unsqueeze(2).unsqueeze(3) # (batch_size, N*2=50, 1, 1)
        batch_size = theta.size()[0]
        # input are the corresponding control points P_i
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords  
        # and points[:,:,:,1] are the Y coords.  
        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]
        
        # split theta into point coordinates (extract the displacements Q_X and Q_Y from theta)
        Q_X=theta[:,:self.N,:,:].squeeze(3) # (batch_size, N=25, 1)
        Q_Y=theta[:,self.N:,:,:].squeeze(3) # (batch_size, N=25, 1)
        # add the displacements to the original control points to get the target control points
        Q_X = Q_X + self.P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + self.P_Y_base.expand_as(Q_Y)

        # compute weigths for non-linear part (multiply by the inverse matrix Li to get the coefficient vector W_X and W_Y)
        W_X = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_X) # (batch_size, N=25, 1)
        W_Y = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_Y) # (batch_size, N=25, 1)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        # compute weights for affine part (calculate the linear part $a_1 + a_x*a + a_y*y$)
        A_X = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_X) # (batch_size, 3, 1)
        A_Y = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_Y) # (batch_size, 3, 1)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1) 
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
        
        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1,points_h,points_w,1,self.N)) # (1,512,320,1,N=25)
        P_Y = self.P_Y.expand((1,points_h,points_w,1,self.N)) # (1,512,320,1,N=25)

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        # points: size [1,H,W,2]
        # points_X_for_summation, points_Y_for_summation: size [1,H,W,1,N]
        points_X_for_summation = points[:,:,:,0].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,0].size()+(1,self.N))
        points_Y_for_summation = points[:,:,:,1].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,1].size()+(1,self.N))
        
        if points_b==1:
            delta_X = points_X_for_summation-P_X # (1,512,320,1,N=25)
            delta_Y = points_Y_for_summation-P_Y # (1,512,320,1,N=25)
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation-P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation-P_Y.expand_as(points_Y_for_summation)
            
        dist_squared = torch.pow(delta_X,2)+torch.pow(delta_Y,2)  # (1,512,320,1,N=25)
        dist_squared[dist_squared==0]=1 # avoid NaN in log computation
        # pass the distances to the radial basis function U
        # U: size [1,H,W,1,N]
        U = torch.mul(dist_squared,torch.log(dist_squared)) 
        
        # expand grid in batch dimension if necessary
        points_X_batch = points[:,:,:,0].unsqueeze(3) # (1,512,320,1)
        points_Y_batch = points[:,:,:,1].unsqueeze(3) # (1,512,320,1)
        if points_b==1:
            points_X_batch = points_X_batch.expand((batch_size,)+points_X_batch.size()[1:]) # (batch_size,512,320,1)
            points_Y_batch = points_Y_batch.expand((batch_size,)+points_Y_batch.size()[1:]) # (batch_size,512,320,1)
        
        # points_X_prime, points_Y_prime: size [B,H,W,1]
        points_X_prime = A_X[:,:,:,:,0]+ \
                       torch.mul(A_X[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_X[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_X,U.expand_as(W_X)),4)
                    
        points_Y_prime = A_Y[:,:,:,:,0]+ \
                       torch.mul(A_Y[:,:,:,:,1],points_X_batch) + \
                       torch.mul(A_Y[:,:,:,:,2],points_Y_batch) + \
                       torch.sum(torch.mul(W_Y,U.expand_as(W_Y)),4)
        
        # concatenate dense array points points_X_prime and points_Y_prime into a grid
        return torch.cat((points_X_prime,points_Y_prime),3)

class DepthDec(nn.Module):
    """
    size: 32-32-32-64-128-256-512
    channel: in_nc-512-512-256-128-64-out_nc
    """
    def __init__(self, in_nc=1024, out_nc=2):
        super(DepthDec, self).__init__()
        self.upconv52 = nn.Conv2d(in_nc, 512, kernel_size=3, stride=1, padding=1)
        self.upnorm52 = nn.InstanceNorm2d(512)
        self.upconv51 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.upnorm51 = nn.InstanceNorm2d(512)

        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upconv4 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.upnorm4 = nn.InstanceNorm2d(256)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upconv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upnorm3 = nn.InstanceNorm2d(128)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upconv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.upnorm2 = nn.InstanceNorm2d(64)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upconv1 = nn.Conv2d(64, out_nc, kernel_size=3, stride=1, padding=1)
        self.upnorm1 = nn.InstanceNorm2d(out_nc)

    def forward(self, x):
        x52up = F.relu_(self.upnorm52(self.upconv52(x)))	        
        x51up = F.relu_(self.upnorm51(self.upconv51(x52up)))	        
        x4up = F.relu_(self.upnorm4(self.upconv4(self.upsample4(x51up))))	        
        x3up = F.relu_(self.upnorm3(self.upconv3(self.upsample3(x4up))))	        
        x2up = F.relu_(self.upnorm2(self.upconv2(self.upsample2(x3up))))	        
        x1up = self.upnorm1(self.upconv1(self.upsample1(x2up)))	        

        return x1up

class SegmtDec(nn.Module):
    """
    size: 32-32-32-64-128-256-512
    channel: in_nc-512-512-256-128-64-out_nc
    """
    def __init__(self, in_nc=1024, out_nc=20):
        super(SegmtDec, self).__init__()
        self.upconv52 = nn.Conv2d(in_nc, 512, kernel_size=3, stride=1, padding=1)
        self.upnorm52 = nn.InstanceNorm2d(512)
        self.upconv51 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.upnorm51 = nn.InstanceNorm2d(512)

        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upconv4 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.upnorm4 = nn.InstanceNorm2d(256)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upconv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upnorm3 = nn.InstanceNorm2d(128)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upconv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.upnorm2 = nn.InstanceNorm2d(64)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upconv1 = nn.Conv2d(64, out_nc, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x52up = F.relu_(self.upnorm52(self.upconv52(x)))	        
        x51up = F.relu_(self.upnorm51(self.upconv51(x52up)))	        
        x4up = F.relu_(self.upnorm4(self.upconv4(self.upsample4(x51up))))	        
        x3up = F.relu_(self.upnorm3(self.upconv3(self.upsample3(x4up))))	        
        x2up = F.relu_(self.upnorm2(self.upconv2(self.upsample2(x3up))))	        
        x1up = self.upconv1(self.upsample1(x2up))	        

        return x1up

class UnetSkipConnectionBlock(nn.Module):
    """Defines the submodule with skip connection.
    X -------------------identity---------------------- X
      |-- downsampling -- |submodule| -- upsampling --|
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(inplace=True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv]
            up = [uprelu, upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            upconv = nn.Conv2d(inner_nc*2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self._l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = self._l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _l2normalize(self, x, eps=1e-12):
        return x / (x.norm() + eps)

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self._l2normalize(u.data)
        v.data = self._l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return torch.nn.Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs_keep> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs_keep) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs_keep, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, std=0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type='normal', init_gain=init_gain)
    return net

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

def random_crop(reals, fakes, winsize=48):
    y, x = [random.randint(reals.size(i)//4, int(reals.size(i)*0.75)-winsize-1) for i in (2, 3)]
    return reals[:,:,y:y+winsize,x:x+winsize], fakes[:,:,y:y+winsize,x:x+winsize]

def define_SGN(input_nc_A=29, input_nc_B=3, ngf=64, n_layers=3, img_height=512, img_width=320, grid_size=5, add_tps=True, 
            add_depth=True, add_segmt=True, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create SGN model. 

    Parameters:
        input_nc_A (int)   -- the number of channels of agnostic input (default: 11)
        input_nc_B (int)   -- the number of channels of flat cloth mask input (default: 3)
        ngf (int)          -- the number of filters in the first conv layer (default: 64)
        img_height (int)   -- input image height (default: 512)
        img_width (int)    -- input image width (default: 320)
        norm (str)         -- the name of normalization layers used in the network: batch | instance | none (default: instance)
        use_dropout (bool) -- whether to use dropout in feature extraction module (default: False)
        init_type (str)    -- the name of our initialization method (default: normal)
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal (default: 0.02)
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2 (default: [])

    Returns:
        a generator, the generator has been initialized by <init_net>.
    """
    
    norm_layer = get_norm_layer(norm_type=norm)
    device = f'cuda:{gpu_ids[0]}' if len(gpu_ids) > 0 else 'cpu'
    net = SGN(input_nc_A, input_nc_B, ngf, n_layers, img_height, img_width, grid_size, add_tps, add_depth, add_segmt, norm_layer, use_dropout, device)
    
    return init_net(net, init_type, init_gain, gpu_ids)

def define_GLA(input_nc_A=29, input_nc_B=3, ngf=64, n_layers=3, img_height=512, img_width=320, grid_size=9, add_tps=True, 
            add_depth=True, add_segmt=True,add_GLU=True, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    
    norm_layer = get_norm_layer(norm_type='instance')
    device = f'cuda:{gpu_ids[0]}' if len(gpu_ids) > 0 else 'cpu'
    net = GLA(input_nc_A, input_nc_B, ngf, n_layers, img_height, img_width, grid_size, add_tps, add_depth, add_segmt,add_GLU, norm_layer, use_dropout, device)
    net=net.to(device)
    
    return init_net(net, init_type, init_gain, gpu_ids)

def define_P(input_nc=9, output_nc=4, num_downs=6, ngf=64, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    net = UnetGenerator(input_nc, output_nc, num_downs, ngf, norm_layer, use_dropout)

    return init_net(net, init_type, init_gain, gpu_ids)

def define_RDG(input_nc=4, output_nc=2, ngf=32, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    net = RDG(input_nc, output_nc, ngf, norm_layer)

    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

        [spectral_norm]: DCGAN-like spectral norm discriminator based on the SNGAN

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'spectral_norm':
        net = SNDiscriminator(input_nc, ndf)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)

##############################################################################
# Generators and Discriminators
##############################################################################
class SGN(nn.Module):
    def __init__(self, input_nc_A=29, input_nc_B=3, ngf=64, n_layers=3, img_height=512, img_width=320, grid_size=5, 
                add_tps=True, add_depth=True, add_segmt=True, norm_layer=nn.InstanceNorm2d, use_dropout=False, device='cpu'):
        super(SGN, self).__init__()
        self.add_tps = add_tps
        self.add_depth = add_depth
        self.add_segmt = add_segmt

        self.extractionA = FeatureExtraction(input_nc_A, ngf, n_layers, norm_layer, use_dropout)
        self.extractionB = FeatureExtraction(input_nc_B, ngf, n_layers, norm_layer, use_dropout)
        self.l2norm = FeatureL2Norm()
        self.correlation = FeatureCorrelation()
        self.regression_tps = FeatureRegression(input_nc=640, output_dim=2*grid_size**2)
        self.tps_grid_gen = TpsGridGen(img_height, img_width, grid_size=grid_size, device=device)

        if self.add_segmt:
            self.segmt_dec = SegmtDec()

        if self.add_depth:
            self.depth_dec = DepthDec(in_nc=1024)

    def forward(self, inputA, inputB):
        """ 
            input A: agnostic (batch_size,12,512,320)
            input B: flat cloth mask(batch_size,1,512,320)
        """
        output = {'theta_tps':None, 'grid_tps':None, 'depth':None, 'segmt':None}
        featureA = self.extractionA(inputA) # featureA: size (batch_size,512,32,20)
        featureB = self.extractionB(inputB) # featureB: size (batch_size,512,32,20)
        if self.add_depth or self.add_segmt:
            featureAB = torch.cat([featureA, featureB], 1) # input for DepthDec and SegmtDec: (batch_size,1024,32,20)
            if self.add_depth:
                depth_pred = self.depth_dec(featureAB)
                output['depth'] = depth_pred
            if self.add_segmt:
                segmt_pred = self.segmt_dec(featureAB)
                output['segmt'] = segmt_pred
        if self.add_tps:
            featureA = self.l2norm(featureA)
            featureB = self.l2norm(featureB)
            correlationAB = self.correlation(featureA, featureB) # correlationAB: size (batch_size, 640, 32, 32)
            theta_tps = self.regression_tps(correlationAB)
            grid_tps = self.tps_grid_gen(theta_tps)
            output['theta_tps'], output['grid_tps'] = theta_tps, grid_tps

        return output
    
class GLA(nn.Module):
    def __init__(self, input_nc_A=29, input_nc_B=3, ngf=64, n_layers=3, img_height=512, img_width=320, grid_size=9, 
                add_tps=True, add_depth=True, add_segmt=True, add_GLU=True, norm_layer=nn.InstanceNorm2d, use_dropout=False, device='cpu'):
        super(GLA, self).__init__()
        self.add_tps = add_tps
        self.add_depth = add_depth
        self.add_segmt = add_segmt
        self.add_GLU = add_GLU
        self.device = device

        #self.extractionA = FeatureExtraction(input_nc_A, ngf, n_layers, norm_layer, use_dropout)
        #self.extractionB = FeatureExtraction(input_nc_B, ngf, n_layers, norm_layer, use_dropout)
        #self.l2norm = FeatureL2Norm()
        #self.correlation = FeatureCorrelation(shape='3D', normalization=False)
        #self.regression_tps = FeatureRegression(input_nc=640, output_dim=2*grid_size**2)
        
        #self.tps_grid_gen = TpsGridGen(img_height, img_width, grid_size=grid_size, device=device)
        '''
        if self.add_segmt:
            self.segmt_dec = SegmtDec()
        
        if self.add_depth:
            self.depth_dec = DepthDec(in_nc=1024)
        '''
        if self.add_GLU:
            self.network = SemanticGLUNet_model(batch_norm=True, pyramid_type='VGG',
                                       div=1.0, evaluation= True, consensus_network=True,
                                       iterative_refinement=False)

    def forward(self, inputA, inputB):
        """ 
            input A: agnostic (batch_size,12,512,320)
            input B: flat cloth mask(batch_size,1,512,320)
        """

        output = {'theta_tps':None, 'grid_tps':None, 'depth':None, 'segmt':None}
        '''
        featureA = self.extractionA(inputA) # featureA: size (batch_size,512,32,20)
        featureB = self.extractionB(inputB) # featureB: size (batch_size,512,32,20)
        if self.add_depth or self.add_segmt:
            featureAB = torch.cat([featureA, featureB], 1) # input for DepthDec and SegmtDec: (batch_size,1024,32,20)
            if self.add_depth:
                depth_pred = self.depth_dec(featureAB)
                output['depth'] = depth_pred
            if self.add_segmt:
                segmt_pred = self.segmt_dec(featureAB)
                output['segmt'] = segmt_pred
        
        
        if self.add_tps:
            featureA = self.l2norm(featureA)
            featureB = self.l2norm(featureB)
            correlationAB = self.correlation(featureA, featureB) # correlationAB: size (batch_size, 640, 32, 32)
            theta_tps = self.regression_tps(correlationAB)
            grid_tps = self.tps_grid_gen(theta_tps)
            output['theta_tps'], output['grid_tps'] = theta_tps, grid_tps
        '''
        
        
        if self.add_GLU:
            
            source_img_copy, target_img_copy, source_img_256, target_img_256, ratio_x, ratio_y, h_original, w_original\
                = self.network.pre_process_data(inputB, inputA,apply_flip=False, device=self.device)
            '''
            source_img_copy, target_img_copy, source_img_256, target_img_256, ratio_x, ratio_y, h_original, w_original\
                = self.network.flipping_condition(inputB, inputA, device=self.device)
            '''
            
            estimated_flow = self.network(target_img_copy, source_img_copy,
                                      target_img_256, source_img_256)
            flow_original_reso = torch.nn.functional.interpolate(input=estimated_flow, size=(h_original, w_original),
                                                                 mode='bilinear', align_corners=False)
            flow_original_reso[:, 0, :, :] *= ratio_x
            flow_original_reso[:, 1, :, :] *= ratio_y

            flow_original_reso = flow_original_reso.permute(0,2,3,1)
            newgrid=kornia.utils.create_meshgrid(h_original, w_original, normalized_coordinates=True, device='cuda', dtype=torch.float32)
            b,_,_,_=inputA.shape
            newgrid=newgrid.expand(b,h_original, w_original,2)
            flow_original_reso[:,:,:,0]=newgrid[:,:,:,0]+flow_original_reso[:,:,:,0]
            flow_original_reso[:,:,:,1]=newgrid[:,:,:,1]+flow_original_reso[:,:,:,1]

            output['grid_tps'] = flow_original_reso               

        return output

class SemanticGLUNet_model(nn.Module):
    """
    Semantic-GLU-Net
    """
    def __init__(self, evaluation, div=1.0, batch_norm=True, pyramid_type='VGG', md=4,
                 cyclic_consistency=False, consensus_network=True, iterative_refinement=False):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping

        """
        super(SemanticGLUNet_model, self).__init__()
        self.div=div
        self.pyramid_type = pyramid_type
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.iterative_refinement = iterative_refinement
        self.cyclic_consistency = cyclic_consistency
        self.consensus_network = consensus_network
        if self.cyclic_consistency:
            self.corr = FeatureCorrelation(shape='4D', normalization=False)
        elif consensus_network:
            ncons_kernel_sizes = [3, 3, 3]
            ncons_channels = [10, 10, 1]
            self.corr = FeatureCorrelation(shape='4D', normalization=False)
            # normalisation is applied in code here

            self.NeighConsensus = NeighConsensus(use_cuda=True,
                                                 kernel_sizes=ncons_kernel_sizes,
                                                 channels=ncons_channels)
        else:
            self.corr = CorrelationVolume()
        # L2 feature normalisation
        self.l2norm = FeatureL2Norm()

        dd = np.cumsum([128,128,96,64,32]) # dd = [128 256 352 416 448]
        # weights for decoder at different levels
        nd = 16*16 # global correlation
        od = nd + 2
        self.decoder4 = CMDTop(in_channels=od, bn=batch_norm)
        # initialize the deconv to bilinear weights speeds up the training significantly
        self.deconv4 = BilinearConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
        # self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        nd = (2*md+1)**2 # constrained correlation, 4 pixels on each side
        od = nd + 2
        self.decoder3 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)

        # weights for refinement module
        self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1, batch_norm=batch_norm)
        self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2, batch_norm=batch_norm)
        self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4, batch_norm=batch_norm)
        self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8, batch_norm=batch_norm)
        self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
        self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1, batch_norm=batch_norm)
        self.dc_conv7 = predict_flow(32)

        # 1/8 of original resolution
        nd = (2*md+1)**2 # constrained correlation, 4 pixels on each side
        od = nd + 2  # only gets the upsampled flow
        self.decoder2 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)

        # initialize the deconv to bilinear weights speeds up the training significantly
        self.deconv2 = BilinearConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
        # self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        self.upfeat2 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)

        # 1/4 of original resolution
        nd = (2*md+1)**2 # constrained correlation, 4 pixels on each side
        od = nd + 4
        self.decoder1 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)

        self.l_dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1, batch_norm=batch_norm)
        self.l_dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2, batch_norm=batch_norm)
        self.l_dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4, batch_norm=batch_norm)
        self.l_dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8, batch_norm=batch_norm)
        self.l_dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
        self.l_dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1, batch_norm=batch_norm)
        self.l_dc_conv7 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()


        self.pyramid1 = VGGPyramid(in_channels=29,train=True)
        self.pyramid2 = VGGPyramid(in_channels=3,train=True)

        self.evaluation=evaluation

    def pre_process_data(self, source_img, target_img, device, apply_flip=False):
        '''

        :param source_img:
        :param target_img:
        :param apply_flip:
        :param device:
        :return:
        '''

        # img has shape bx3xhxw
        b, _, h_original, w_original = target_img.shape
        '''
        mean_vector = np.array([0.485, 0.456, 0.406])
        std_vector = np.array([0.229, 0.224, 0.225])
        
        mean_vector = np.array([0.8892, 0.8769, 0.8799])
        std_vector = np.array([0.3872, 0.4099, 0.3995])
    
        mean_29 = np.array([-0.6266, -0.9062, -0.9176, -0.9178, -0.9985, -0.9985, -0.9985, -0.9985,
        -0.9986, -0.9985, -0.9986, -0.9986, -0.9985, -0.9985, -0.9985, -0.9985,
        -0.9985, -0.9985, -0.9985, -0.9985, -0.9985, -0.9986, -0.9987, -0.9985,
        -0.9986, -0.9985, -0.9985, -0.9986, -0.9985])
        std_29 = np.array([0.6686, 0.3351, 0.2936, 0.2899, 0.0542, 0.0542, 0.0542, 0.0541, 0.0536,
        0.0542, 0.0535, 0.0529, 0.0542, 0.0542, 0.0542, 0.0541, 0.0542, 0.0542,
        0.0541, 0.0541, 0.0541, 0.0522, 0.0518, 0.0539, 0.0536, 0.0540, 0.0540,
        0.0535, 0.0541])
        '''

        # original resolution
        # if image is smaller than our base network, calculate on this and then downsample to the original size
        if h_original < 256:
            int_preprocessed_height = 256
        else:
            int_preprocessed_height = int(math.floor(int(h_original / 8.0) * 8.0))

        if w_original < 256:
            int_preprocessed_width = 256
        else:
            int_preprocessed_width = int(math.floor(int(w_original / 8.0) * 8.0))
        '''
        if apply_flip:
            # flip the target image horizontally 将数组在左右方向上翻转
            target_img_original = target_img
            target_img = []
            for i in range(b):
                transformed_image = np.fliplr(target_img_original[i].cpu().permute(1,2,0).numpy())
                target_img.append(transformed_image)

            target_img = torch.from_numpy(np.uint8(target_img)).permute(0, 3, 1, 2)
        '''
        source_img_copy = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                          size=(int_preprocessed_height, int_preprocessed_width),
                                                          mode='area')
        target_img_copy = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                          size=(int_preprocessed_height, int_preprocessed_width),
                                                          mode='area')
        '''
        #source_img_copy = source_img_copy.float().div(255.0) #.div(255.0)  除以255.0
        #target_img_copy = target_img_copy.float().div(255.0) 
        
        mean = torch.as_tensor(mean_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
        std = torch.as_tensor(std_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
        mean29 = torch.as_tensor(mean_29, dtype=target_img_copy.dtype, device=target_img_copy.device)
        std29 = torch.as_tensor(std_29, dtype=target_img_copy.dtype, device=target_img_copy.device)


        source_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])
        target_img_copy.sub_(mean29[:, None, None]).div_(std29[:, None, None])
        '''

        # resolution 256x256 将分辨率采样成256x256
        source_img_256 = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                         size=(256, 256), mode='area')
        target_img_256 = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                         size=(256, 256), mode='area')
        '''
        #source_img_256 = source_img_256.float().div(255.0)
        #target_img_256 = target_img_256.float().div(255.0)
        source_img_256.sub_(mean[:, None, None]).div_(std[:, None, None])
        target_img_256.sub_(mean29[:, None, None]).div_(std29[:, None, None])
        '''
        ratio_x = float(w_original)/float(int_preprocessed_width)
        ratio_y = float(h_original)/float(int_preprocessed_height)

        return source_img_copy.to(device), target_img_copy.to(device), source_img_256.to(device), target_img_256.to(device), \
               ratio_x, ratio_y, h_original, w_original
    '''
    def flipping_condition(self, im_source_base, im_target_base, device):
        # should only happen during evaluation
        target_image_is_flipped = False # for training
        if not self.evaluation:
            raise ValueError('Flipping condition should only happen during evaluation')
        else:
            list_average_flow = []
            false_true = [False, True]
            for apply_flipping in false_true:
                im_source, im_target, im_source_256, im_target_256, ratio_x, ratio_y, h_base, w_base = \
                    self.pre_process_data(im_source_base, im_target_base, apply_flip=apply_flipping, device=device)
                b, _, h_256, w_256 = im_target_256.size()

                with torch.no_grad():
                    # pyramid, 256 reso
                    im1_pyr_256 = self.pyramid1(im_target_256)
                    im2_pyr_256 = self.pyramid2(im_source_256)
                    c14 = im1_pyr_256[-3]
                    c24 = im2_pyr_256[-3]
                    c15 = im1_pyr_256[-2]
                    c25 = im2_pyr_256[-2]
                    c24_concat = torch.cat(
                        (c24, F.interpolate(input=c25, size=(16, 16), mode='bilinear', align_corners=False)), 1)
                    c14_concat = torch.cat(
                        (c14, F.interpolate(input=c15, size=(16, 16), mode='bilinear', align_corners=False)), 1)

                flow4 = self.coarsest_resolution_flow(c14_concat, c24_concat, h_256, w_256)
                average_flow = torch.mean(torch.abs(flow4[:, 0, :, :]), dim=(1, 2))
                list_average_flow.append(average_flow.item())
                
            target_image_is_flipped = false_true[np.argmin(np.float32(list_average_flow))]
            if target_image_is_flipped:
                list_average_flow = []
                # if previous way found that target is flipped with respect to the source ==> check that the
                # other way finds the same thing
                # ==> the source becomes the target and the target becomes source
                for apply_flipping in false_true:
                    im_source, im_target, im_source_256, im_target_256, ratio_x, ratio_y, h_base, w_base = \
                        self.pre_process_data(im_target_base, im_source_base, apply_flip=apply_flipping, device=device)
                    b, _, h_256, w_256 = im_target_256.size()

                    with torch.no_grad():
                        # pyramid, 256 reso
                        im1_pyr_256 = self.pyramid1(im_target_256)
                        im2_pyr_256 = self.pyramid2(im_source_256)
                        c14 = im1_pyr_256[-3]
                        c24 = im2_pyr_256[-3]
                        c15 = im1_pyr_256[-2]
                        c25 = im2_pyr_256[-2]
                        c24_concat = torch.cat(
                            (c24, F.interpolate(input=c25, size=(16, 16), mode='bilinear', align_corners=False)), 1)
                        c14_concat = torch.cat(
                            (c14, F.interpolate(input=c15, size=(16, 16), mode='bilinear', align_corners=False)), 1)

                    flow4 = self.coarsest_resolution_flow(c14_concat, c24_concat, h_256, w_256)
                    average_flow = torch.mean(torch.abs(flow4[:, 0, :, :]), dim=(1, 2))
                    list_average_flow.append(average_flow.item())
                 
                target_image_is_flipped = false_true[np.argmin(np.float32(list_average_flow))]
                # if the right direction found that it is flipped, either the other direction finds the same,
                # then it is flipped, otherwise it isnt flipped

        self.target_image_is_flipped = target_image_is_flipped
        im_source, im_target, im_source_256, im_target_256, ratio_x, ratio_y, \
        h_original, w_original = self.pre_process_data(im_source_base, im_target_base,
                                                       apply_flip=target_image_is_flipped, device=device)
        return im_source.to(device).contiguous(), im_target.to(device).contiguous(), \
               im_source_256.to(device).contiguous(), im_target_256.to(device).contiguous(), \
               ratio_x, ratio_y, h_original, w_original
    '''
    def coarsest_resolution_flow(self, c14, c24, h_256, w_256):
        ratio_x = 16.0 / float(w_256)
        ratio_y = 16.0 / float(h_256)
        b = c14.shape[0]
        if self.cyclic_consistency:
            corr4d = self.corr(self.l2norm(c24), self.l2norm(c14))  # first source, then target
            # run match processing model
            corr4d = MutualMatching(corr4d)
            corr4 = corr4d.squeeze(1).view(b, c24.shape[2] * c24.shape[3], c14.shape[2], c14.shape[3])
        elif self.consensus_network:
            corr4d = self.corr(self.l2norm(c24), self.l2norm(c14))  # first source, then target
            # run match processing model
            corr4d = MutualMatching(corr4d)
            corr4d = self.NeighConsensus(corr4d)
            corr4d = MutualMatching(corr4d)  # size is [b, 1, hsource, wsource, htarget, wtarget]
            corr4 = corr4d.squeeze(1).view(c24.shape[0], c24.shape[2] * c24.shape[3], c14.shape[2], c14.shape[3])
        else:
            corr4 = self.corr(self.l2norm(c24), self.l2norm(c14))
        corr4 = self.l2norm(F.relu(corr4))
        b, c, h, w = corr4.size()
        if torch.cuda.is_available():
            init_map = torch.FloatTensor(b, 2, h, w).zero_().cuda()
        else:
            init_map = torch.FloatTensor(b, 2, h, w).zero_()
        est_map4 = self.decoder4(x1=corr4, x3=init_map)
        # conversion to flow and from there constrained correlation
        flow4 = unnormalise_and_convert_mapping_to_flow(est_map4) / self.div
        flow4[:, 0, :, :] /= ratio_x
        flow4[:, 1, :, :] /= ratio_y
        return flow4

    def forward(self, im_target, im_source, im_target_256, im_source_256):
        # all indices 1 refer to target images
        # all indices 2 refer to source images

        b, _, h_original, w_original = im_target.size()
        b, _, h_256, w_256 = im_target_256.size()
        div = self.div

        # pyramid, original reso

        im1_pyr = self.pyramid1(im_target, eigth_resolution=True)
        im2_pyr = self.pyramid2(im_source, eigth_resolution=True)
        c11 = im1_pyr[-2] # size original_res/4xoriginal_res/4
        c21 = im2_pyr[-2]
        c12 = im1_pyr[-1] # size original_res/8xoriginal_res/8
        c22 = im2_pyr[-1]

        # pyramid, 256 reso
        im1_pyr_256 = self.pyramid1(im_target_256)
        im2_pyr_256 = self.pyramid2(im_source_256)
        c13 = im1_pyr_256[-4]
        c23 = im2_pyr_256[-4]
        c14 = im1_pyr_256[-3]
        c24 = im2_pyr_256[-3]
        c15 = im1_pyr_256[-2]
        c25 = im2_pyr_256[-2]

        # RESOLUTION 256x256
        # level 16x16
        c24_concat = torch.cat((c24, F.interpolate(input=c25, size=(16, 16), mode='bilinear', align_corners=False)), 1)
        c14_concat = torch.cat((c14, F.interpolate(input=c15, size=(16, 16), mode='bilinear', align_corners=False)), 1)
        flow4 = self.coarsest_resolution_flow(c14_concat, c24_concat, h_256, w_256)
        up_flow4 = self.deconv4(flow4)

        # level 32x32
        ratio_x = 32.0 / float(w_256)
        ratio_y = 32.0 / float(h_256)
        up_flow_4_warping = up_flow4 * div
        up_flow_4_warping[:, 0, :, :] *= ratio_x
        up_flow_4_warping[:, 1, :, :] *= ratio_y
        c23_concat = torch.cat((c23, F.interpolate(input=c24, size=(32, 32), mode='bilinear', align_corners=False),
                                F.interpolate(input=c25, size=(32, 32), mode='bilinear', align_corners=False)), 1)
        c13_concat = torch.cat((c13, F.interpolate(input=c14, size=(32, 32), mode='bilinear', align_corners=False),
                                F.interpolate(input=c15, size=(32, 32), mode='bilinear', align_corners=False)), 1)
        warp3 = warp(c23_concat, up_flow_4_warping)
        # constrained correlation now
        corr3 = FunctionCorrelation(tensorFirst=c13_concat, tensorSecond=warp3)
        corr3 = self.leakyRELU(corr3)
        corr3 = torch.cat((corr3, up_flow4), 1)
        x, res_flow3 = self.decoder3(corr3)
        flow3 = res_flow3 + up_flow4
        # flow 3 refined (at 32x32 resolution)
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow3 = flow3 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        if self.evaluation and self.iterative_refinement:
            # from 32x32 resolution, if upsampling to 1/8*original resolution is too big,
            # do iterative upsampling so that gap is always smaller than 2.
            R_w = float(w_original)/8.0/32.0
            R_h = float(h_original)/8.0/32.0
            if R_w > R_h:
                R = R_w
            else:
                R = R_h

            minimum_ratio = 3.0
            nbr_extra_layers = max(0, int(round(np.log(R/minimum_ratio)/np.log(2))))

            if nbr_extra_layers == 0:
                flow3[:, 0, :, :] *= float(w_original) / float(256)
                flow3[:, 1, :, :] *= float(h_original) / float(256)
                # ==> put the upflow in the range [Horiginal x Woriginal]
            else:
                # adding extra layers
                flow3[:, 0, :, :] *= float(w_original) / float(256)
                flow3[:, 1, :, :] *= float(h_original) / float(256)
                for n in range(nbr_extra_layers):
                    ratio = 1.0 / (8.0 * 2 ** (nbr_extra_layers - n ))
                    up_flow3 = F.interpolate(input=flow3, size=(int(h_original * ratio), int(w_original * ratio)),
                                             mode='bilinear', align_corners=False)
                    c23_bis = torch.nn.functional.interpolate(c22, size=(int(h_original * ratio), int(w_original * ratio)), mode='area')
                    c13_bis = torch.nn.functional.interpolate(c12, size=(int(h_original * ratio), int(w_original * ratio)), mode='area')
                    warp3 = warp(c23_bis, up_flow3 * div * ratio)
                    corr3 = FunctionCorrelation(tensorFirst=c13_bis, tensorSecond=warp3)
                    corr3 = self.leakyRELU(corr3)
                    corr3 = torch.cat((corr3, up_flow3), 1)
                    x, res_flow3 = self.decoder2(corr3)
                    flow3 = res_flow3 + up_flow3

            # ORIGINAL RESOLUTION
            up_flow3 = F.interpolate(input=flow3, size=(int(h_original / 8.0), int(w_original / 8.0)), mode='bilinear',
                                     align_corners=False)
        else:
            # ORIGINAL RESOLUTION
            up_flow3 = F.interpolate(input=flow3, size=(int(h_original / 8.0), int(w_original / 8.0)), mode='bilinear',
                                     align_corners=False)
            up_flow3[:, 0, :, :] *= float(w_original) / float(256)
            up_flow3[:, 1, :, :] *= float(h_original) / float(256)
            # ==> put the upflow in the range [Horiginal x Woriginal]

        # ORIGINAL RESOLUTION
        # level 1/8 of original resolution
        ratio = 1.0/8.0
        warp2 = warp(c22, up_flow3*div*ratio)
        corr2 = FunctionCorrelation(tensorFirst=c12, tensorSecond=warp2)
        corr2 = self.leakyRELU(corr2)
        corr2 = torch.cat((corr2, up_flow3), 1)
        x, res_flow2 = self.decoder2(corr2)
        flow2 = res_flow2 + up_flow3
        up_flow2 = self.deconv2(flow2)
        up_feat2 = self.upfeat2(x)

        # level 1/4 of original resolution
        ratio = 1.0 / 4.0
        warp1 = warp(c21, up_flow2*div*ratio)
        corr1 = FunctionCorrelation(tensorFirst=c11, tensorSecond=warp1)
        corr1 = self.leakyRELU(corr1)
        corr1 = torch.cat((corr1, up_flow2, up_feat2), 1)
        x, res_flow1 = self.decoder1(corr1)
        flow1 = res_flow1 + up_flow2
        x = self.l_dc_conv4(self.l_dc_conv3(self.l_dc_conv2(self.l_dc_conv1(x))))
        flow1 = flow1 + self.l_dc_conv7(self.l_dc_conv6(self.l_dc_conv5(x)))

        if self.evaluation:
            return flow1
        else:
            return [flow4, flow3], [flow2, flow1]
'''
class GLU_Net:
    def __init__(self, model_type='DPED_CityScape_ADE', path_pre_trained_models='pre_trained_models/',
                 apply_flipping_condition=False, pyramid_type='VGG', iterative_refinement=True,
                 feature_concatenation=True, decoder_inputs='corr_flow_feat', up_feat_channels=2,
                 cyclic_consistency=True, consensus_network=False, dense_connections=True):

        self.apply_flipping_condition = apply_flipping_condition
        # semantic glu-net
        if feature_concatenation:
            net = SemanticGLUNet_model(batch_norm=True, pyramid_type=pyramid_type,
                                       div=1.0, evaluation=True, consensus_network=consensus_network,
                                       iterative_refinement=iterative_refinement)

            if consensus_network:
                checkpoint_fname = osp.join(path_pre_trained_models, 'Semantic_GLUNet_' + model_type + '.pth')
            else:
                raise ValueError('there are no saved weights for this configuration')

        else:
            net = GLUNet_model(batch_norm=True,
                                pyramid_type=pyramid_type,
                                div=1.0, evaluation=True,
                                refinement_at_adaptive_reso=True,
                                decoder_inputs=decoder_inputs,
                                upfeat_channels=up_feat_channels,#2
                                dense_connection=dense_connections,#True
                                cyclic_consistency=cyclic_consistency,#True
                                consensus_network=consensus_network,#False
                                iterative_refinement=iterative_refinement)#True

            if cyclic_consistency and dense_connections and decoder_inputs == 'corr_flow_feat' and up_feat_channels == 2:
                checkpoint_fname = osp.join(path_pre_trained_models, 'GLUNet_' + model_type + '.pth')
            else:
                raise ValueError('there are no saved weights for this configuration')

        if not osp.isfile(checkpoint_fname):
            raise ValueError('check the snapshots path, checkpoint is {}'.format(checkpoint_fname))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            net.load_state_dict(torch.load(checkpoint_fname))
        except:
            net.load_state_dict(torch.load(checkpoint_fname)['state_dict'])

        print("loaded the weights")
        net.eval()
        self.net = net.to(device) # load on GPU

    def estimate_flow(self, source_img, target_img, device, mode='channel_first'):
        if self.apply_flipping_condition:
            source_img_copy, target_img_copy, source_img_256, target_img_256, ratio_x, ratio_y, h_original, w_original\
                = self.net.flipping_condition(source_img, target_img, device)
            estimated_flow = self.net(target_img_copy, source_img_copy,
                                      target_img_256, source_img_256)

            flow_original_reso = torch.nn.functional.interpolate(input=estimated_flow, size=(h_original, w_original),
                                                                 mode='bilinear', align_corners=False)
            flow_original_reso[:, 0, :, :] *= ratio_x
            flow_original_reso[:, 1, :, :] *= ratio_y

            if self.net.target_image_is_flipped:
                flipped_mapping = convert_flow_to_mapping(flow_original_reso, output_channel_first=True).permute(0, 2, 3, 1).cpu().numpy()
                b = flipped_mapping.shape[0]
                mapping_per_batch = []
                for i in range(b):
                    map = np.copy(np.fliplr(flipped_mapping[i]))
                    mapping_per_batch.append(map)

                mapping = torch.from_numpy(np.float32(mapping_per_batch)).permute(0, 3, 1, 2).to(device)
                flow_original_reso = convert_mapping_to_flow(mapping, device)

        else:
            source_img_copy, target_img_copy, source_img_256, target_img_256, ratio_x, ratio_y, h_original, w_original\
                = self.net.pre_process_data(source_img, target_img, device)
            estimated_flow = self.net(target_img_copy, source_img_copy,
                                      target_img_256, source_img_256)

            flow_original_reso = torch.nn.functional.interpolate(input=estimated_flow, size=(h_original, w_original),
                                                                 mode='bilinear', align_corners=False)
            flow_original_reso[:, 0, :, :] *= ratio_x
            flow_original_reso[:, 1, :, :] *= ratio_y

        if mode == 'channel_first':
            return flow_original_reso
        else:
            return flow_original_reso.permute(0,2,3,1)
'''



class UnetGenerator(nn.Module):
    """Defines the Unet generator."""
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True) # add the innermost layer
        for i in range(num_downs - 5): # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class RDG(nn.Module):
    def __init__(self, in_channel, out_channel, ngf=32, norm_layer=nn.InstanceNorm2d):
        super(RDG, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.ngf = ngf
        
        # size -> size / 2
        self.l0 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf * 2, 3, padding=1, stride=2),
            nn.ELU(),
            norm_layer(self.ngf * 2)
        )

        # size / 2 -> size / 4
        self.l1 = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 3, padding=1, stride=2),
            nn.ELU(),
            norm_layer(self.ngf * 4)
        )

        # size / 4 -> size / 8
        self.l2 = nn.Sequential(
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 8, 3, padding=1, stride=2),
            nn.ELU(),
            norm_layer(self.ngf * 8)
        )

        # size / 8 -> size / 16
        self.l3 = nn.Sequential(
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 16, 3, padding=1, stride=2),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            norm_layer(self.ngf * 16)
        )

        self.block1 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        self.block2 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        # size / 16 -> size / 8
        self.l3u = nn.Sequential(
            nn.Conv2d(self.ngf * 24, self.ngf * 8, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            norm_layer(self.ngf * 8)
        )

        # size / 8 -> size / 4
        self.l2u = nn.Sequential(
            nn.Conv2d(self.ngf * 12, self.ngf * 4, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            norm_layer(self.ngf * 4)
        )

        # size / 4 -> size / 2
        self.l1u = nn.Sequential(
            nn.Conv2d(self.ngf * 6, self.ngf * 2, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            norm_layer(self.ngf * 2)
        )

        # size / 2 -> size
        self.l0u = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.out_channel, 3, padding=1, stride=1),
            nn.Tanh()
        )

    def forward(self, input_data, inter_mode='bilinear'):
        x0 = self.l0(input_data)
        x1 = self.l1(x0)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x3 = self.block1(x3) + x3
        x3 = self.block2(x3) + x3
        x3u = nn.functional.interpolate(x3, size=x2.shape[2:4], mode=inter_mode)
        x3u = self.l3u(torch.cat((x3u, x2), dim=1))
        x2u = nn.functional.interpolate(x3u, size=x1.shape[2:4], mode=inter_mode)
        x2u = self.l2u(torch.cat((x2u, x1), dim=1))
        x1u = nn.functional.interpolate(x2u, size=x0.shape[2:4], mode=inter_mode)
        x1u = self.l1u(torch.cat((x1u, x0), dim=1))
        x0u = nn.functional.interpolate(x1u, size=input_data.shape[2:4], mode=inter_mode)
        x0u = self.l0u(x0u)
        return x0u


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

class SNDiscriminator(nn.Module):
    """Defines a DCGAN-like spectral norm discriminator (SNGAN)"""
    def __init__(self, input_nc, ndf=64):
        super(SNDiscriminator, self).__init__()

        self.conv1 = SpectralNorm(nn.Conv2d(input_nc, ndf, 3, stride=1, padding=(1,1)))

        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))
        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        self.conv7 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(1,1)))


        self.fc = SpectralNorm(nn.Linear(4 * 4 * 512, 1))

    def forward(self, x):
        m = x
        m = nn.LeakyReLU(0.1)(self.conv1(m))
        m = nn.LeakyReLU(0.1)(self.conv2(m))
        m = nn.LeakyReLU(0.1)(self.conv3(m))
        m = nn.LeakyReLU(0.1)(self.conv4(m))
        m = nn.LeakyReLU(0.1)(self.conv5(m))
        m = nn.LeakyReLU(0.1)(self.conv6(m))
        m = nn.LeakyReLU(0.1)(self.conv7(m))

        return self.fc(m.view(-1, 4 * 4 * 512))

##############################################################################
# Loss Functions
##############################################################################

class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()

    def forward(self, depth_pred, depth_gt):
        loss_depth = torch.log(torch.abs(depth_pred - depth_gt) + 1).mean()
        
        return loss_depth

class DepthGradLoss(nn.Module):
    def __init__(self):
        super(DepthGradLoss, self).__init__()

    def forward(self, depth_grad_pred, depth_grad_gt):
        depth_grad_gt_dx = depth_grad_gt[:, 0, :, :].unsqueeze(1)
        depth_grad_gt_dy = depth_grad_gt[:, 1, :, :].unsqueeze(1)
        depth_grad_pred_dx = depth_grad_pred[:, 0, :, :].unsqueeze(1)
        depth_grad_pred_dy = depth_grad_pred[:, 1, :, :].unsqueeze(1)
        
        loss_dx = torch.log(torch.abs(depth_grad_pred_dx - depth_grad_gt_dx) + 1).mean()
        loss_dy = torch.log(torch.abs(depth_grad_pred_dy - depth_grad_gt_dy) + 1).mean()
        
        loss_grad = loss_dx + loss_dy
    
        return loss_grad

class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-12)

    def forward(self, normal_pred, normal_gt):
        
        loss_normal = (1 - self.cos(normal_pred, normal_gt)).mean()
        
        return loss_normal

class WeakInlierCount(nn.Module):
    def __init__(self, tps_grid_size=9, h_matches=15, w_matches=15, dilation_filter=0, device=[]):
        super(WeakInlierCount, self).__init__()
        self.normalize=normalize_inlier_count

        self.geometricTnf = TpsGridGen(out_h=h_matches, out_w=w_matches, use_regular_grid=True, grid_size=tps_grid_size, device=self.device)

        # define identity mask tensor (w,h are switched and will be permuted back later)
        mask_id = np.zeros((w_matches,h_matches,w_matches*h_matches))
        idx_list = list(range(0, mask_id.size, mask_id.shape[2]+1))
        mask_id.reshape((-1))[idx_list]=1
        mask_id = mask_id.swapaxes(0,1)

        # perform 2D dilation to each channel 
        if not (isinstance(dilation_filter,int) and dilation_filter==0):
            for i in range(mask_id.shape[2]):
                mask_id[:,:,i] = binary_dilation(mask_id[:,:,i],structure=dilation_filter).astype(mask_id.dtype)
            
        # convert to PyTorch variable
        self.mask_id = torch.from_numpy(mask_id.transpose(1,2).transpose(0,1).unsqueeze(0).float()).to(self.device)

    def forward(self, theta, matches, return_outliers=False):
        batch_size=theta.size()[0]
        theta=theta.clone()
        mask = self.geometricTnf(util.expand_dim(self.mask_id,0,batch_size),theta)
        mask = self.geometricTnf(self.mask_id.expand())
        if return_outliers:
            mask_outliers = self.geometricTnf(util.expand_dim(1.0-self.mask_id,0,batch_size),theta)

        # normalize inlier conunt
        epsilon=1e-5
        mask = torch.div(mask, torch.sum(torch.sum(torch.sum(mask+epsilon,3),2),1).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(mask))
        if return_outliers:
            mask_outliers = torch.div(mask_outliers, torch.sum(torch.sum(torch.sum(mask_outliers+epsilon,3),2),1).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(mask_outliers))
        
        # compute score
        score = torch.sum(torch.sum(torch.sum(torch.mul(mask,matches),3),2),1)
        if return_outliers:
            score_outliers = torch.sum(torch.sum(torch.sum(torch.mul(mask_outliers,matches),3),2),1)
            return (score,score_outliers)

        return score
    

class ThetaLoss(nn.Module):
    def __init__(self, grid_size=9, device='cpu'):
        super(ThetaLoss, self).__init__()
        self.device = device
        self.grid_size = grid_size
        
    def forward(self, theta):
        batch_size = theta.size()[0]
        coordinate = theta.view(batch_size, -1, 2) # (4,25,2)
        # coordinate+=torch.randn(coordinate.shape).cuda()/10
        row_loss = self.get_row_loss(coordinate, self.grid_size)
        col_loss = self.get_col_loss(coordinate, self.grid_size)
        # row_x, row_y, col_x, col_y: size [batch_size,15]
        row_x, row_y = row_loss[:,:,0], row_loss[:,:,1]
        col_x, col_y = col_loss[:,:,0], col_loss[:,:,1]
        # TODO: what does 0.08 mean?
        if self.device != 'cpu':
            rx, ry, cx, cy = (torch.tensor([0.08]).to(self.device) for i in range(4))
        else:
            rx, ry, cx, cy = (torch.tensor([0.08]) for i in range(4))
        rx_loss = torch.max(rx, row_x).mean()
        ry_loss = torch.max(ry, row_y).mean()
        cx_loss = torch.max(cx, col_x).mean()
        cy_loss = torch.max(cy, col_y).mean()
        sec_diff_loss = rx_loss + ry_loss + cx_loss + cy_loss
        slope_loss = self.get_slope_loss(coordinate, self.grid_size).mean()

        theta_loss = sec_diff_loss + slope_loss

        return theta_loss
    
    def get_row_loss(self, coordinate, num):
        sec_diff = []
        for j in range(num):
            buffer = 0
            for i in range(num-1):
                # TODO: should be L2 distance according to ACGPN paper,  but not L1?
                diff = (coordinate[:, j*num+i+1, :]-coordinate[:, j*num+i, :]) ** 2
                if i >= 1:
                    sec_diff.append(torch.abs(diff-buffer))
                buffer = diff

        return torch.stack(sec_diff, dim=1)
    
    def get_col_loss(self, coordinate, num):
        sec_diff = []
        for i in range(num):
            buffer = 0
            for j in range(num - 1):
                # TODO: should be L2 distance according to ACGPN paper, but not L1?
                diff = (coordinate[:, (j+1)*num+i, :] - coordinate[:, j*num+i, :]) ** 2
                if j >= 1:
                    sec_diff.append(torch.abs(diff-buffer))
                buffer = diff
                
        return torch.stack(sec_diff,dim=1)
    
    def get_slope_loss(self, coordinate, num):
        slope_diff = []
        for j in range(num - 2):
            x, y = coordinate[:, (j+1)*num+1, 0], coordinate[:, (j+1)*num+1, 1]
            x0, y0 = coordinate[:, j*num+1, 0], coordinate[:, j*num+1, 1]
            x1, y1 = coordinate[:, (j+2)*num+1, 0], coordinate[:, (j+2)*num+1, 1]
            x2, y2 = coordinate[:, (j+1)*num, 0], coordinate[:, (j+1)*num, 1]
            x3, y3 = coordinate[:, (j+1)*num+2, 0], coordinate[:, (j+1)*num+2, 1]
            row_diff = torch.abs((y0 - y) * (x1 - x) - (y1 - y) * (x0 - x))
            col_diff = torch.abs((y2 - y) * (x3 - x) - (y3 - y) * (x2 -x))
            slope_diff.append(row_diff + col_diff)
            
        return torch.stack(slope_diff, dim=0)

class GridLoss(nn.Module):
    def __init__(self, image_height, image_width, distance='l1'):
        super(GridLoss, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.distance = distance

    def forward(self, grid):
        gx = grid[:,:,:,0]
        gy = grid[:,:,:,1]
        gx_ctr = gx[:, 1:self.image_height-1, 1:self.image_width-1]
        gx_up = gx[:, 0:self.image_height-2, 1:self.image_width-1]
        gx_down = gx[:, 2:self.image_height, 1:self.image_width-1]
        gx_left = gx[:, 1:self.image_height-1, 0:self.image_width-2]
        gx_right = gx[:, 1:self.image_height-1, 2:self.image_width]

        gy_ctr = gy[:, 1:self.image_height-1, 1:self.image_width-1]
        gy_up = gy[:, 0:self.image_height-2, 1:self.image_width-1]
        gy_down = gy[:, 2:self.image_height, 1:self.image_width-1]
        gy_left = gy[:, 1:self.image_height-1, 0:self.image_width-2]
        gy_right = gy[:, 1:self.image_height-1, 2:self.image_width]

        if self.distance == 'l1':
            grid_loss_left = self._l1_distance(gx_left, gx_ctr)
            grid_loss_right = self._l1_distance(gx_right, gx_ctr)
            grid_loss_up = self._l1_distance(gy_up, gy_ctr)
            grid_loss_down = self._l1_distance(gy_down, gy_ctr)
        elif self.distance == 'l2':
            grid_loss_left = self._l2_distance(gx_left, gy_left, gx_ctr, gy_ctr)
            grid_loss_right = self._l2_distance(gx_right, gy_right, gx_ctr, gy_ctr)
            grid_loss_up = self._l2_distance(gx_up, gy_up, gx_ctr, gy_ctr)
            grid_loss_down = self._l2_distance(gx_down, gy_down, gx_ctr, gy_ctr)

        grid_loss = torch.sum(torch.abs(grid_loss_left-grid_loss_right) + torch.abs(grid_loss_up-grid_loss_down))

        return grid_loss
    
    def _l1_distance(self, x1, x2):

        return torch.abs(x1 - x2)
    
    def _l2_distance(self, x1, y1, x2, y2):
        
        return torch.sqrt(torch.mul(x1-x2, x1-x2) + torch.mul(y1-y2, y1-y2))

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class VGGLoss(nn.Module):
    def __init__(self, layids = None, device = 'cpu'):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        if device != 'cpu':
            self.vgg.to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        
        return loss

class VGGVector(nn.Module):
    def __init__(self, layids = None, device = 'cpu'):
        super(VGGVector, self).__init__()
        self.vgg = Vgg19()
        if device != 'cpu':
            self.vgg.to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        vgg_vector = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            if i == 0:
                vgg_vector += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach()).expand(1,1)
            else:
                vgg_vector = torch.cat([vgg_vector, self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach()).expand(1,1)], 1)

        return vgg_vector

####GLU-net
class BilinearConvTranspose2d(nn.ConvTranspose2d):
    """A conv transpose initialized to bilinear interpolation."""

    def __init__(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1, groups=1):
        """Set up the layer.
        Parameters
        ----------
        channels: int
            The number of input and output channels
        stride: int or tuple
            The amount of upsampling to do
        groups: int
            Set to 1 for a standard convolution. Set equal to channels to
            make sure there is no cross-talk between channels.
        """
        if isinstance(stride, int):
            stride = (stride, stride)
        super().__init__(
            in_planes, out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the weight and bias."""
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.weight, 0)
        bilinear_kernel = self.bilinear_kernel(self.kernel_size)

        # for each output channel, applied bilinear_kernel on the same input channel, and 0 on the other input channels.
        for i in range(self.in_channels):
            if self.groups == 1:
                j = i
            else:
                j = 0
            self.weight.data[i, j] = bilinear_kernel

    @staticmethod
    def bilinear_kernel(kernel_size):
        """Generate a bilinear upsampling kernel."""
        num_dims = len(kernel_size)

        shape = (1,) * num_dims
        bilinear_kernel = torch.ones(*shape)

        # The bilinear kernel is separable in its spatial dimensions
        # Build up the kernel dim by dim
        for dim in range(num_dims):
            kernel = kernel_size[dim]
            factor = (kernel + 1) // 2
            if kernel % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            delta = torch.arange(0, kernel) - center
            channel_filter = (1 - torch.abs(delta / factor))

            # Apply the dim filter to the current dim
            shape = [1] * num_dims
            shape[dim] = kernel
            bilinear_kernel = bilinear_kernel * channel_filter.view(shape)
        # if kenel_size is (4,4), bilinear kernel is (4,4)
        # channel_filter is [0.25, 0.75, 0.75, 0.25]
        return bilinear_kernel

def Softmax1D(x,dim):
    x_k = torch.max(x,dim)[0].unsqueeze(dim)
    x -= x_k.expand_as(x)
    exp_x = torch.exp(x)
    return torch.div(exp_x,torch.sum(exp_x,dim).unsqueeze(dim).expand_as(x))


def conv4d(data, filters, bias=None, permute_filters=True, use_half=False):
    b, c, h, w, d, t = data.size()

    data = data.permute(2, 0, 1, 3, 4, 5).contiguous()  # permute to avoid making contiguous inside loop

    # Same permutation is done with filters, unless already provided with permutation
    if permute_filters:
        filters = filters.permute(2, 0, 1, 3, 4, 5).contiguous()  # permute to avoid making contiguous inside loop

    c_out = filters.size(1)
    if use_half:
        output = Variable(torch.HalfTensor(h, b, c_out, w, d, t), requires_grad=data.requires_grad)
    else:
        output = Variable(torch.zeros(h, b, c_out, w, d, t), requires_grad=data.requires_grad)

    padding = filters.size(0) // 2
    if use_half:
        Z = Variable(torch.zeros(padding, b, c, w, d, t).half())
    else:
        Z = Variable(torch.zeros(padding, b, c, w, d, t))

    if data.is_cuda:
        Z = Z.cuda(data.get_device())
        output = output.cuda(data.get_device())

    data_padded = torch.cat((Z, data, Z), 0)

    for i in range(output.size(0)):  # loop on first feature dimension
        # convolve with center channel of filter (at position=padding)
        output[i, :, :, :, :, :] = F.conv3d(data_padded[i + padding, :, :, :, :, :],
                                            filters[padding, :, :, :, :, :], bias=bias, stride=1, padding=padding)
        # convolve with upper/lower channels of filter (at postions [:padding] [padding+1:])
        for p in range(1, padding + 1):
            output[i, :, :, :, :, :] = output[i, :, :, :, :, :] + F.conv3d(data_padded[i + padding - p, :, :, :, :, :],
                                                                           filters[padding - p, :, :, :, :, :],
                                                                           bias=None, stride=1, padding=padding)
            output[i, :, :, :, :, :] = output[i, :, :, :, :, :] + F.conv3d(data_padded[i + padding + p, :, :, :, :, :],
                                                                           filters[padding + p, :, :, :, :, :],
                                                                           bias=None, stride=1, padding=padding)

    output = output.permute(1, 2, 0, 3, 4, 5).contiguous()
    return output


class Conv4d(_ConvNd):
    """Applies a 4D convolution over an input signal composed of several input
    planes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, pre_permuted_filters=True):
        # stride, dilation and groups !=1 functionality not tested
        stride = 1
        dilation = 1
        groups = 1
        # zero padding is added automatically in conv4d function to preserve tensor size
        padding = 0
        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)
        if float(torch.__version__[:3]) >= 1.3:
            super(Conv4d, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                transposed=False, output_padding=_quadruple(0), groups=groups, bias=bias,
                padding_mode='zeros')
        else:
            super(Conv4d, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                transposed=False, output_padding=_quadruple(0), groups=groups, bias=bias)

        # weights will be sliced along one dimension during convolution loop
        # make the looping dimension to be the first one in the tensor,
        # so that we don't need to call contiguous() inside the loop
        self.pre_permuted_filters = pre_permuted_filters
        if self.pre_permuted_filters:
            self.weight.data = self.weight.data.permute(2, 0, 1, 3, 4, 5).contiguous()
        self.use_half = False

    def forward(self, input):
        return conv4d(input, self.weight, bias=self.bias, permute_filters=not self.pre_permuted_filters,
                      use_half=self.use_half)  # filters pre-permuted in constructor


def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature,norm)




class NeighConsensus(torch.nn.Module):
    def __init__(self, use_cuda=True, kernel_sizes=[3 ,3 ,3], channels=[10 ,10 ,1], symmetric_mode=True):
        super(NeighConsensus, self).__init__()
        self.symmetric_mode = symmetric_mode
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):
            if i== 0:
                ch_in = 1
            else:
                ch_in = channels[i - 1]
            ch_out = channels[i]
            k_size = kernel_sizes[i]
            nn_modules.append(Conv4d(in_channels=ch_in, out_channels=ch_out, kernel_size=k_size, bias=True))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)
        if use_cuda:
            self.conv.cuda()

    def forward(self, x):
        if self.symmetric_mode:
            # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
            # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
            x = self.conv(x) + self.conv(x.permute(0, 1, 4, 5, 2, 3)).permute(0, 1, 4, 5, 2, 3)
            # because of the ReLU layers in between linear layers,
            # this operation is different than convolving a single time with the filters+filters^T
            # and therefore it makes sense to do this.
        else:
            x = self.conv(x)
        return x


def MutualMatching(corr4d):
    # mutual matching
    batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()

    corr4d_B = corr4d.view(batch_size, fs1 * fs2, fs3, fs4)  # [batch_idx,k_A,i_B,j_B] #correlation target
    corr4d_A = corr4d.view(batch_size, fs1, fs2, fs3 * fs4)

    # get max
    corr4d_B_max, _ = torch.max(corr4d_B, dim=1, keepdim=True)
    corr4d_A_max, _ = torch.max(corr4d_A, dim=3, keepdim=True)

    eps = 1e-5
    corr4d_B = corr4d_B / (corr4d_B_max + eps)
    corr4d_A = corr4d_A / (corr4d_A_max + eps)

    corr4d_B = corr4d_B.view(batch_size, 1, fs1, fs2, fs3, fs4)
    corr4d_A = corr4d_A.view(batch_size, 1, fs1, fs2, fs3, fs4)

    corr4d = corr4d * (corr4d_A * corr4d_B)  # parenthesis are important for symmetric output

    return corr4d


def maxpool4d(corr4d_hres, k_size=4):
    slices = []
    for i in range(k_size):
        for j in range(k_size):
            for k in range(k_size):
                for l in range(k_size):
                    slices.append(corr4d_hres[:, 0, i::k_size, j::k_size, k::k_size, l::k_size].unsqueeze(0))
    slices = torch.cat(tuple(slices), dim=1)
    corr4d, max_idx = torch.max(slices, dim=1, keepdim=True)
    max_l = torch.fmod(max_idx, k_size)
    max_k = torch.fmod(max_idx.sub(max_l).div(k_size), k_size)
    max_j = torch.fmod(max_idx.sub(max_l).div(k_size).sub(max_k).div(k_size), k_size)
    max_i = max_idx.sub(max_l).div(k_size).sub(max_k).div(k_size).sub(max_j).div(k_size)
    # i,j,k,l represent the *relative* coords of the max point in the box of size k_size*k_size*k_size*k_size
    return (corr4d, max_i, max_j, max_k, max_l)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False):
    if batch_norm:
        return nn.Sequential(
                            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, bias=True),
                            nn.BatchNorm2d(out_planes),
                            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Sequential(
                            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, bias=True),
                            nn.LeakyReLU(0.1))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    deconv_ = nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

    nn.init.kaiming_normal_(deconv_.weight.data, mode='fan_in')
    if deconv_.bias is not None:
        deconv_.bias.data.zero_()
    return deconv_


def unnormalise_and_convert_mapping_to_flow(map):
    # here map is normalised to -1;1
    # we put it back to 0,W-1, then convert it to flow
    B, C, H, W = map.size()
    mapping = torch.zeros_like(map)
    # mesh grid
    mapping[:,0,:,:] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0 # unormalise
    mapping[:,1,:,:] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0 # unormalise

    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if mapping.is_cuda:
        grid = grid.cuda()
    flow = mapping - grid
    return flow


class CorrelationVolume(nn.Module):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """

    def __init__(self):
        super(CorrelationVolume, self).__init__()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()

        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)  # shape (b,c,h*w)
        feature_B = feature_B.view(b, c, h * w).transpose(1, 2)  # shape (b,h*w,c)
        feature_mul = torch.bmm(feature_B, feature_A)  # shape (b,h*w,h*w)
        correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        return correlation_tensor  # shape (b,h*w,h,w)




class OpticalFlowEstimator(nn.Module):

    def __init__(self, in_channels, batch_norm):
        super(OpticalFlowEstimator, self).__init__()

        dd = np.cumsum([128,128,96,64,32])
        self.conv_0 = conv(in_channels, 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_1 = conv(in_channels + dd[0], 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_2 = conv(in_channels + dd[1], 96, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_3 = conv(in_channels + dd[2], 64, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_4 = conv(in_channels + dd[3], 32, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.predict_flow = predict_flow(in_channels + dd[4])

    def forward(self, x):
        # dense net connection
        x = torch.cat((self.conv_0(x), x),1)
        x = torch.cat((self.conv_1(x), x),1)
        x = torch.cat((self.conv_2(x), x),1)
        x = torch.cat((self.conv_3(x), x),1)
        x = torch.cat((self.conv_4(x), x),1)
        flow = self.predict_flow(x)
        return x, flow


class OpticalFlowEstimatorNoDenseConnection(nn.Module):

    def __init__(self, in_channels, batch_norm):
        super(OpticalFlowEstimatorNoDenseConnection, self).__init__()
        self.conv_0 = conv(in_channels, 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_1 = conv(128, 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_2 = conv(128, 96, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_3 = conv(96, 64, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_4 = conv(64, 32, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.predict_flow = predict_flow(32)

    def forward(self, x):
        x = self.conv_4(self.conv_3(self.conv_2(self.conv_1(self.conv_0(x)))))
        flow = self.predict_flow(x)
        return x, flow


# extracted from DGCNet
def conv_blck(in_channels, out_channels, kernel_size=3,
              stride=1, padding=1, dilation=1, bn=False):
    if bn:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation),
                             nn.ReLU(inplace=True))


def conv_head(in_channels):
    return nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)


class CorrespondenceMapBase(nn.Module):
    def __init__(self, in_channels, bn=False):
        super().__init__()

    def forward(self, x1, x2=None, x3=None):
        x = x1
        # concatenating dimensions
        if (x2 is not None) and (x3 is None):
            x = torch.cat((x1, x2), 1)
        elif (x2 is None) and (x3 is not None):
            x = torch.cat((x1, x3), 1)
        elif (x2 is not None) and (x3 is not None):
            x = torch.cat((x1, x2, x3), 1)

        return x


class CMDTop(CorrespondenceMapBase):
    def __init__(self, in_channels, bn=False):
        super().__init__(in_channels, bn)
        chan = [128, 128, 96, 64, 32]
        self.conv0 = conv_blck(in_channels, chan[0], bn=bn)
        self.conv1 = conv_blck(chan[0], chan[1], bn=bn)
        self.conv2 = conv_blck(chan[1], chan[2], bn=bn)
        self.conv3 = conv_blck(chan[2], chan[3], bn=bn)
        self.conv4 = conv_blck(chan[3], chan[4], bn=bn)
        self.final = conv_head(chan[-1])

    def forward(self, x1, x2=None, x3=None):
        x = super().forward(x1, x2, x3)
        x = self.conv4(self.conv3(self.conv2(self.conv1(self.conv0(x)))))
        return self.final(x)


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # makes a mapping out of the flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)

    if float(torch.__version__[:3]) >= 1.3:
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    else:
        output = nn.functional.grid_sample(x, vgrid)
    return output

class VGGPyramid(nn.Module):
    def __init__(self, in_channels, train=False):
        super().__init__()
        self.n_levels = 5
        source_model = VGG(in_channels=in_channels)

        modules = OrderedDict()
        tmp = []
        n_block = 0
        first_relu = False

        for c in source_model.features.children():
            if (isinstance(c, nn.ReLU) and not first_relu) or (isinstance(c, nn.MaxPool2d)):
                first_relu = True
                tmp.append(c)
                modules['level_' + str(n_block)] = nn.Sequential(*tmp)
                for param in modules['level_' + str(n_block)].parameters():
                    param.requires_grad = train

                tmp = []
                n_block += 1
            else:
                tmp.append(c)

            if n_block == self.n_levels:
                break

        self.__dict__['_modules'] = modules

    def forward(self, x, quarter_resolution_only=False, eigth_resolution=False):
        outputs = []
        x = x.to('cuda')
        if quarter_resolution_only:
            x_full = self.__dict__['_modules']['level_' + str(0)](x)
            x_half = self.__dict__['_modules']['level_' + str(1)](x_full)
            x_quarter = self.__dict__['_modules']['level_' + str(2)](x_half)
            outputs.append(x_quarter)
        elif eigth_resolution:
            x_full = self.__dict__['_modules']['level_' + str(0)](x)
            outputs.append(x_full)
            x_half = self.__dict__['_modules']['level_' + str(1)](x_full)
            x_quarter = self.__dict__['_modules']['level_' + str(2)](x_half)
            outputs.append(x_quarter)
            x_eight = self.__dict__['_modules']['level_' + str(3)](x_quarter)
            outputs.append(x_eight)
        else:
            for layer_n in range(0, self.n_levels):
                x = self.__dict__['_modules']['level_' + str(layer_n)](x)
                outputs.append(x)

            if float(torch.__version__[:3]) >= 1.6:
                x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area', recompute_scale_factor=True)
            else:
                x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area')
            outputs.append(x)

            if float(torch.__version__[:3]) >= 1.6:
                x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area', recompute_scale_factor=True)
            else:
                x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area')
            outputs.append(x)
        return outputs


class Stream:
	ptr = torch.cuda.current_stream().cuda_stream
# end

kernel_Correlation_rearrange = '''
	extern "C" __global__ void kernel_Correlation_rearrange(
		const int n,
		const float* input,
		float* output
	) {
	  int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	  if (intIndex >= n) {
	    return;
	  }

	  int intSample = blockIdx.z;
	  int intChannel = blockIdx.y;

	  float dblValue = input[(((intSample * SIZE_1(input)) + intChannel) * SIZE_2(input) * SIZE_3(input)) + intIndex];

	  __syncthreads();

	  int intPaddedY = (intIndex / SIZE_3(input)) + 4;
	  int intPaddedX = (intIndex % SIZE_3(input)) + 4;
	  int intRearrange = ((SIZE_3(input) + 8) * intPaddedY) + intPaddedX;

	  output[(((intSample * SIZE_1(output) * SIZE_2(output)) + intRearrange) * SIZE_1(input)) + intChannel] = dblValue;
	}
'''

kernel_Correlation_updateOutput = '''
	extern "C" __global__ void kernel_Correlation_updateOutput(
	  const int n,
	  const float* rbot0,
	  const float* rbot1,
	  float* top
	) {
	  extern __shared__ char patch_data_char[];
	  
	  float *patch_data = (float *)patch_data_char;
	  
	  // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
	  int x1 = blockIdx.x + 4;
	  int y1 = blockIdx.y + 4;
	  int item = blockIdx.z;
	  int ch_off = threadIdx.x;
	  
	  // Load 3D patch into shared shared memory
	  for (int j = 0; j < 1; j++) { // HEIGHT
	    for (int i = 0; i < 1; i++) { // WIDTH
	      int ji_off = (j + i) * SIZE_3(rbot0);
	      for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
	        int idx1 = ((item * SIZE_1(rbot0) + y1+j) * SIZE_2(rbot0) + x1+i) * SIZE_3(rbot0) + ch;
	        int idxPatchData = ji_off + ch;
	        patch_data[idxPatchData] = rbot0[idx1];
	      }
	    }
	  }
	  
	  __syncthreads();
	  
	  __shared__ float sum[32];
	  
	  // Compute correlation
	  for (int top_channel = 0; top_channel < SIZE_1(top); top_channel++) {
	    sum[ch_off] = 0;
	  
	    int s2o = top_channel % 9 - 4;
	    int s2p = top_channel / 9 - 4;
	    
	    for (int j = 0; j < 1; j++) { // HEIGHT
	      for (int i = 0; i < 1; i++) { // WIDTH
	        int ji_off = (j + i) * SIZE_3(rbot0);
	        for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
	          int x2 = x1 + s2o;
	          int y2 = y1 + s2p;
	          
	          int idxPatchData = ji_off + ch;
	          int idx2 = ((item * SIZE_1(rbot0) + y2+j) * SIZE_2(rbot0) + x2+i) * SIZE_3(rbot0) + ch;
	          
	          sum[ch_off] += patch_data[idxPatchData] * rbot1[idx2];
	        }
	      }
	    }
	    
	    __syncthreads();
	    
	    if (ch_off == 0) {
	      float total_sum = 0;
	      for (int idx = 0; idx < 32; idx++) {
	        total_sum += sum[idx];
	      }
	      const int sumelems = SIZE_3(rbot0);
	      const int index = ((top_channel*SIZE_2(top) + blockIdx.y)*SIZE_3(top))+blockIdx.x;
	      top[index + item*SIZE_1(top)*SIZE_2(top)*SIZE_3(top)] = total_sum / (float)sumelems;
	    }
	  }
	}
'''

kernel_Correlation_updateGradFirst = '''
	#define ROUND_OFF 50000

	extern "C" __global__ void kernel_Correlation_updateGradFirst(
	  const int n,
	  const int intSample,
	  const float* rbot0,
	  const float* rbot1,
	  const float* gradOutput,
	  float* gradFirst,
	  float* gradSecond
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
	  int n = intIndex % SIZE_1(gradFirst); // channels
	  int l = (intIndex / SIZE_1(gradFirst)) % SIZE_3(gradFirst) + 4; // w-pos
	  int m = (intIndex / SIZE_1(gradFirst) / SIZE_3(gradFirst)) % SIZE_2(gradFirst) + 4; // h-pos
	  
	  // round_off is a trick to enable integer division with ceil, even for negative numbers
	  // We use a large offset, for the inner part not to become negative.
	  const int round_off = ROUND_OFF;
	  const int round_off_s1 = round_off;
	  
	  // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
	  int xmin = (l - 4 + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4)
	  int ymin = (m - 4 + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4)
	  
	  // Same here:
	  int xmax = (l - 4 + round_off_s1) - round_off; // floor (l - 4)
	  int ymax = (m - 4 + round_off_s1) - round_off; // floor (m - 4)
	  
	  float sum = 0;
	  if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(gradOutput)-1) && (ymin<=SIZE_2(gradOutput)-1)) {
	    xmin = max(0,xmin);
	    xmax = min(SIZE_3(gradOutput)-1,xmax);
	    
	    ymin = max(0,ymin);
	    ymax = min(SIZE_2(gradOutput)-1,ymax);
	    
	    for (int p = -4; p <= 4; p++) {
	      for (int o = -4; o <= 4; o++) {
	        // Get rbot1 data:
	        int s2o = o;
	        int s2p = p;
	        int idxbot1 = ((intSample * SIZE_1(rbot0) + (m+s2p)) * SIZE_2(rbot0) + (l+s2o)) * SIZE_3(rbot0) + n;
	        float bot1tmp = rbot1[idxbot1]; // rbot1[l+s2o,m+s2p,n]
	        
	        // Index offset for gradOutput in following loops:
	        int op = (p+4) * 9 + (o+4); // index[o,p]
	        int idxopoffset = (intSample * SIZE_1(gradOutput) + op);
	        
	        for (int y = ymin; y <= ymax; y++) {
	          for (int x = xmin; x <= xmax; x++) {
	            int idxgradOutput = (idxopoffset * SIZE_2(gradOutput) + y) * SIZE_3(gradOutput) + x; // gradOutput[x,y,o,p]
	            sum += gradOutput[idxgradOutput] * bot1tmp;
	          }
	        }
	      }
	    }
	  }
	  const int sumelems = SIZE_1(gradFirst);
	  const int bot0index = ((n * SIZE_2(gradFirst)) + (m-4)) * SIZE_3(gradFirst) + (l-4);
	  gradFirst[bot0index + intSample*SIZE_1(gradFirst)*SIZE_2(gradFirst)*SIZE_3(gradFirst)] = sum / (float)sumelems;
	} }
'''

kernel_Correlation_updateGradSecond = '''
	#define ROUND_OFF 50000

	extern "C" __global__ void kernel_Correlation_updateGradSecond(
	  const int n,
	  const int intSample,
	  const float* rbot0,
	  const float* rbot1,
	  const float* gradOutput,
	  float* gradFirst,
	  float* gradSecond
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
	  int n = intIndex % SIZE_1(gradSecond); // channels
	  int l = (intIndex / SIZE_1(gradSecond)) % SIZE_3(gradSecond) + 4; // w-pos
	  int m = (intIndex / SIZE_1(gradSecond) / SIZE_3(gradSecond)) % SIZE_2(gradSecond) + 4; // h-pos
	  
	  // round_off is a trick to enable integer division with ceil, even for negative numbers
	  // We use a large offset, for the inner part not to become negative.
	  const int round_off = ROUND_OFF;
	  const int round_off_s1 = round_off;
	  
	  float sum = 0;
	  for (int p = -4; p <= 4; p++) {
	    for (int o = -4; o <= 4; o++) {
	      int s2o = o;
	      int s2p = p;
	      
	      //Get X,Y ranges and clamp
	      // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
	      int xmin = (l - 4 - s2o + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4 - s2o)
	      int ymin = (m - 4 - s2p + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4 - s2o)
	      
	      // Same here:
	      int xmax = (l - 4 - s2o + round_off_s1) - round_off; // floor (l - 4 - s2o)
	      int ymax = (m - 4 - s2p + round_off_s1) - round_off; // floor (m - 4 - s2p)
          
	      if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(gradOutput)-1) && (ymin<=SIZE_2(gradOutput)-1)) {
	        xmin = max(0,xmin);
	        xmax = min(SIZE_3(gradOutput)-1,xmax);
	        
	        ymin = max(0,ymin);
	        ymax = min(SIZE_2(gradOutput)-1,ymax);
	        
	        // Get rbot0 data:
	        int idxbot0 = ((intSample * SIZE_1(rbot0) + (m-s2p)) * SIZE_2(rbot0) + (l-s2o)) * SIZE_3(rbot0) + n;
	        float bot0tmp = rbot0[idxbot0]; // rbot1[l+s2o,m+s2p,n]
	        
	        // Index offset for gradOutput in following loops:
	        int op = (p+4) * 9 + (o+4); // index[o,p]
	        int idxopoffset = (intSample * SIZE_1(gradOutput) + op);
	        
	        for (int y = ymin; y <= ymax; y++) {
	          for (int x = xmin; x <= xmax; x++) {
	            int idxgradOutput = (idxopoffset * SIZE_2(gradOutput) + y) * SIZE_3(gradOutput) + x; // gradOutput[x,y,o,p]
	            sum += gradOutput[idxgradOutput] * bot0tmp;
	          }
	        }
	      }
	    }
	  }
	  const int sumelems = SIZE_1(gradSecond);
	  const int bot1index = ((n * SIZE_2(gradSecond)) + (m-4)) * SIZE_3(gradSecond) + (l-4);
	  gradSecond[bot1index + intSample*SIZE_1(gradSecond)*SIZE_2(gradSecond)*SIZE_3(gradSecond)] = sum / (float)sumelems;
	} }
'''

def cupy_kernel(strFunction, objectVariables):
	strKernel = globals()[strFunction]

	while True:
		objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

		if objectMatch is None:
			break
		# end

		intArg = int(objectMatch.group(2))

		strTensor = objectMatch.group(4)
		intSizes = objectVariables[strTensor].size()

		strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))
	# end

	while True:
		objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

		if objectMatch is None:
			break
		# end

		intArgs = int(objectMatch.group(2))
		strArgs = objectMatch.group(4).split(',')

		strTensor = strArgs[0]
		intStrides = objectVariables[strTensor].stride()
		strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs) ]

		strKernel = strKernel.replace(objectMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
	# end

	return strKernel
# end

@cupyutil.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
	return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)
# end

class _FunctionCorrelation(torch.autograd.Function):
	@staticmethod
	def forward(self, first, second):
		rbot0 = first.new_zeros([ first.size(0), first.size(2) + 8, first.size(3) + 8, first.size(1) ])
		rbot1 = first.new_zeros([ first.size(0), first.size(2) + 8, first.size(3) + 8, first.size(1) ])

		self.save_for_backward(first, second, rbot0, rbot1)

		assert(first.is_contiguous() == True)
		assert(second.is_contiguous() == True)

		output = first.new_zeros([ first.size(0), 81, first.size(2), first.size(3) ])

		if first.is_cuda == True:
			n = first.size(2) * first.size(3)
			cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
				'input': first,
				'output': rbot0
			}))(
				grid=tuple([ int((n + 16 - 1) / 16), first.size(1), first.size(0) ]),
				block=tuple([ 16, 1, 1 ]),
				args=[ n, first.data_ptr(), rbot0.data_ptr() ],
				stream=Stream
			)

			n = second.size(2) * second.size(3)
			cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
				'input': second,
				'output': rbot1
			}))(
				grid=tuple([ int((n + 16 - 1) / 16), second.size(1), second.size(0) ]),
				block=tuple([ 16, 1, 1 ]),
				args=[ n, second.data_ptr(), rbot1.data_ptr() ],
				stream=Stream
			)

			n = output.size(1) * output.size(2) * output.size(3)
			cupy_launch('kernel_Correlation_updateOutput', cupy_kernel('kernel_Correlation_updateOutput', {
				'rbot0': rbot0,
				'rbot1': rbot1,
				'top': output
			}))(
				grid=tuple([ output.size(3), output.size(2), output.size(0) ]),
				block=tuple([ 32, 1, 1 ]),
				shared_mem=first.size(1) * 4,
				args=[ n, rbot0.data_ptr(), rbot1.data_ptr(), output.data_ptr() ],
				stream=Stream
			)

		elif first.is_cuda == False:
			raise NotImplementedError()

		# end

		return output
	# end

	@staticmethod
	def backward(self, gradOutput):
		first, second, rbot0, rbot1 = self.saved_tensors

		assert(gradOutput.is_contiguous() == True)

		gradFirst = first.new_zeros([ first.size(0), first.size(1), first.size(2), first.size(3) ]) if self.needs_input_grad[0] == True else None
		gradSecond = first.new_zeros([ first.size(0), first.size(1), first.size(2), first.size(3) ]) if self.needs_input_grad[1] == True else None

		if first.is_cuda == True:
			if gradFirst is not None:
				for intSample in range(first.size(0)):
					n = first.size(1) * first.size(2) * first.size(3)
					cupy_launch('kernel_Correlation_updateGradFirst', cupy_kernel('kernel_Correlation_updateGradFirst', {
						'rbot0': rbot0,
						'rbot1': rbot1,
						'gradOutput': gradOutput,
						'gradFirst': gradFirst,
						'gradSecond': None
					}))(
						grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
						block=tuple([ 512, 1, 1 ]),
						args=[ n, intSample, rbot0.data_ptr(), rbot1.data_ptr(), gradOutput.data_ptr(), gradFirst.data_ptr(), None ],
						stream=Stream
					)
				# end
			# end

			if gradSecond is not None:
				for intSample in range(first.size(0)):
					n = first.size(1) * first.size(2) * first.size(3)
					cupy_launch('kernel_Correlation_updateGradSecond', cupy_kernel('kernel_Correlation_updateGradSecond', {
						'rbot0': rbot0,
						'rbot1': rbot1,
						'gradOutput': gradOutput,
						'gradFirst': None,
						'gradSecond': gradSecond
					}))(
						grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
						block=tuple([ 512, 1, 1 ]),
						args=[ n, intSample, rbot0.data_ptr(), rbot1.data_ptr(), gradOutput.data_ptr(), None, gradSecond.data_ptr() ],
						stream=Stream
					)
				# end
			# end

		elif first.is_cuda == False:
			raise NotImplementedError()

		# end

		return gradFirst, gradSecond
	# end
# end

def FunctionCorrelation(tensorFirst, tensorSecond):
	return _FunctionCorrelation.apply(tensorFirst, tensorSecond)
# end

class ModuleCorrelation(torch.nn.Module):
	def __init__(self):
		super(ModuleCorrelation, self).__init__()
	# end

	def forward(self, tensorFirst, tensorSecond):
		return _FunctionCorrelation.apply(tensorFirst, tensorSecond)
	# end
# end



def load_flo(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)

    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (h, w, 2))
    return data2D


def split2list(images, split, default_split=0.9):
    if isinstance(split, str):
        with open(split) as f:
            split_values = [x.strip() == '1' for x in f.readlines()]
        assert(len(images) == len(split_values))
    elif split is None:
        split_values = np.random.uniform(0,1,len(images)) < default_split
    else:
        try:
            split = float(split)
        except TypeError:
            print("Invalid Split value, it must be either a filepath or a float")
            raise
        split_values = np.random.uniform(0,1,len(images)) < split
    train_samples = [sample for sample, split in zip(images, split_values) if split]
    test_samples = [sample for sample, split in zip(images, split_values) if not split]
    return train_samples, test_samples


def center_crop(img, size):
    """
    Get the center crop of the input image
    Args:
        img: input image [HxWxC]
        size: size of the center crop (tuple) (width, height)
    Output:
        img_pad: center crop
        x, y: coordinates of the crop
    """

    if not isinstance(size, tuple):
        size = (size, size)
        #size is W,H

    img = img.copy()
    h, w = img.shape[:2]

    pad_w = 0
    pad_h = 0
    if w < size[0]:
        pad_w = np.int(np.ceil((size[0] - w) / 2))
    if h < size[1]:
        pad_h = np.int(np.ceil((size[1] - h) / 2))
    img_pad = cv2.copyMakeBorder(img,
                                 pad_h,
                                 pad_h,
                                 pad_w,
                                 pad_w,
                                 cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])
    h, w = img_pad.shape[:2]

    x1 = w // 2 - size[0] // 2
    y1 = h // 2 - size[1] // 2

    img_pad = img_pad[y1:y1 + size[1], x1:x1 + size[0], :]

    return img_pad, x1, y1


def convert_flow_to_mapping(flow, output_channel_first=True):
    if not isinstance(flow, np.ndarray):
        #torch tensor
        if len(flow.shape) == 4:
            if flow.shape[1] != 2:
                # size is BxHxWx2
                flow = flow.permute(0, 3, 1, 2)

            B, C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if flow.is_cuda:
                grid = grid.cuda()
            map = flow + grid # here also channel first
            if not output_channel_first:
                map = map.permute(0,2,3,1)
        else:
            if flow.shape[0] != 2:
                # size is HxWx2
                flow = flow.permute(2, 0, 1)

            C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float() # attention, concat axis=0 here

            if flow.is_cuda:
                grid = grid.cuda()
            map = flow + grid # here also channel first
            if not output_channel_first:
                map = map.permute(1,2,0).float()
        return map.float()
    else:
        # here numpy arrays
        if len(flow.shape) == 4:
            if flow.shape[3] != 2:
                # size is Bx2xHxW
                flow = flow.permute(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = flow.shape[:3]
            map = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                map[i, :, :, 0] = flow[i, :, :, 0] + X
                map[i, :, :, 1] = flow[i, :, :, 1] + Y
            if output_channel_first:
                map = map.transpose(0,3,1,2)
        else:
            if flow.shape[0] == 2:
                # size is 2xHxW
                flow = flow.permute(1,2,0)
            # HxWx2
            h_scale, w_scale = flow.shape[:2]
            map = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            map[:,:,0] = flow[:,:,0] + X
            map[:,:,1] = flow[:,:,1] + Y
            if output_channel_first:
                map = map.transpose(2,0,1).float()
        return map.astype(np.float32)


def convert_mapping_to_flow(map, output_channel_first=True):
    if not isinstance(map, np.ndarray):
        # torch tensor
        if len(map.shape) == 4:
            if map.shape[1] != 2:
                # size is BxHxWx2
                map = map.permute(0, 3, 1, 2)

            B, C, H, W = map.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if map.is_cuda:
                grid = grid.cuda()
            flow = map - grid # here also channel first
            if not output_channel_first:
                flow = flow.permute(0,2,3,1)
        else:
            if map.shape[0] != 2:
                # size is HxWx2
                map = map.permute(2, 0, 1)

            C, H, W = map.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float() # attention, concat axis=0 here

            if map.is_cuda:
                grid = grid.cuda()

            flow = map - grid # here also channel first
            if not output_channel_first:
                flow = flow.permute(1,2,0).float()
        return flow.float()
    else:
        # here numpy arrays
        if len(map.shape) == 4:
            if map.shape[3] != 2:
                # size is Bx2xHxW
                map = map.permute(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = map.shape[:3]
            flow = np.copy(map)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                flow[i, :, :, 0] = map[i, :, :, 0] - X
                flow[i, :, :, 1] = map[i, :, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(0,3,1,2)
        else:
            if map.shape[0] == 2:
                # size is 2xHxW
                map = map.permute(1,2,0)
            # HxWx2
            h_scale, w_scale = map.shape[:2]
            flow = np.copy(map)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            flow[:,:,0] = map[:,:,0]-X
            flow[:,:,1] = map[:,:,1]-Y
            if output_channel_first:
                flow = flow.transpose(2,0,1).float()
        return flow.astype(np.float32)