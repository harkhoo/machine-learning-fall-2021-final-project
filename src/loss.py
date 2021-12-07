import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss( nn.Module ):
    """ Implementation of Focal Loss

        From:
        Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár,"Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002

    """

    def __init__( self, gamma ):
        self.gamma = gamma

    # __init__

    def forward( self, input, target ):
        BCELoss = F.binary_cross_entropy_with_logits( input, target, reduction='none' )
        pt = torch.exp( -BCELoss )

        if self.gamma == 0:
            loss = BCELoss
        else:
            loss = (1 - pt) ** self.gamma * BCELoss

        loss = loss.mean()

        return loss

    # forward

# class: FocalLoss
