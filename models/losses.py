import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __int__(self,smooth = 1e-6):
        super(DiceLoss,self).__init__()
        self.smooth = smooth

    def forward(self,predictions,targets):
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2. *intersection +self.smooth) / (predictions.sum()+targets.sum()+self.smooth)
        return 1-dice

class BCEDiceLoss(nn.Module):
    def __int__(self,alpha=0.5,smooth=1e-6):
        super(BCEDiceLoss, self).__int__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self,predictions,targets):
        bce_loss = self.bce(predictions,targets)
        dice_loss = self.dice(predictions,targets)
        return self.alpha * bce_loss + (1-self.alpha) * dice_loss