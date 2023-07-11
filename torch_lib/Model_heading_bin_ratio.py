import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def residual_loss(orient_residual,truth_bin,truth_orient_resdiual,device):#truth_orient_resdiual:B,truth_bin:B,orient_residual:B,12

    one_hot_map=torch.zeros((orient_residual.shape)).to(device).scatter_(dim=1,index=truth_bin.view(-1,1),value=1)#(batch,bin_class)
    heading_res=torch.sum(orient_residual*one_hot_map,dim=1)
    reg_loss=F.l1_loss(heading_res,truth_orient_resdiual,reduction='mean')
    
    return reg_loss,heading_res

class Model(nn.Module):
    def __init__(self, features=None, bins=4):
        super(Model, self).__init__()
        self.bins = bins #2
        self.features = features
        self.orientation = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins) # to get sin and cos
                )
        self.confidence = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins),
                    # nn.Softmax()
                    #nn.Sigmoid()
                )
        self.dimension = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 2) # Height, Width, Length
                )
        
        self.length_ratio = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 2) # H, W, HW
                )
        

    def forward(self, x):
        x = self.features(x) #vgg output 512 x 7 x 7
        x = x.view(-1, 512 * 7 * 7)
        orientation = self.orientation(x)
        confidence = self.confidence(x)
        dimension = self.dimension(x)
        ratio = self.length_ratio(x)
        return orientation, confidence, dimension , ratio
