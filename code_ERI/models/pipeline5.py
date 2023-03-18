import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.facenet2 import InceptionResnetV1
import math
import torch

class Pipeline(nn.Module):
    def __init__(self,freeze=False):
        super(Pipeline,self).__init__()
        # self.faceNet= InceptionResnetV1(pretrained="vggface2")a
        # self.R_net = InceptionResnetV1(pretrained="vggface2")
        #self.Dconv = ConvOffset2D(3)
        self.faceNet = InceptionResnetV1(pretrained="vggface2").eval()
        for param in self.faceNet.parameters():
            param.requires_grad = False
        self.R_net = InceptionResnetV1(pretrained="vggface2")
        self.BN1 = nn.BatchNorm1d(512)
        self.BN2 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512,16,bias=False)

    def forward(self, x ):
        #x = self.Dconv(x)
        with torch.no_grad():
            id_feature = self.faceNet(x)
        id_feature = torch.sigmoid(id_feature)
        x = self.R_net(x)
        x = torch.sigmoid(x)
        x = x-id_feature
        x = self.linear2(x)
        x = F.normalize(x, dim=1)
        return x


    def forward_no_norm(self,x,return_skip =False):
        with torch.no_grad():
            if return_skip:
                id_feature,s2 = self.faceNet(x,return_skip)
            else:
                id_feature = self.faceNet(x)
        id_feature = torch.sigmoid(id_feature)
        x = self.R_net(x)
        x = torch.sigmoid(x)
        x = x-id_feature
        x = self.linear2(x)
        if return_skip:
            return x,s2
        return x

    def get_emb_vector(self,x):
        with torch.no_grad():
            id_feature = self.faceNet(x)
            id_feature = torch.sigmoid(id_feature)
            x = self.R_net(x)
            x = torch.sigmoid(x)
            x = x - id_feature
            return x

    def get_id_vector(self,x):
        with torch.no_grad():
            id_feature = self.faceNet(x)
            id_feature = torch.sigmoid(id_feature)
            return id_feature

    def forward_no_norm2(self,x,return_skip =False):
        with torch.no_grad():
            if return_skip:
                id_feature,s2 = self.faceNet(x,return_skip)
            else:
                id_feature = self.faceNet(x)
        id_feature = torch.sigmoid(id_feature)
        x = self.R_net(x)
        x = torch.sigmoid(x)
        x = x-id_feature
  
        return x


# net = Pipeline().cuda()
# x = torch.rand([16,3,224,224]).cuda()
# res= net(x)
# print(res.shape)
# print(sum(param.numel() for param in net.parameters()))