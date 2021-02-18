import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.005,
                 reduction='mean',
                 lb_ignore=255,
                 weight=None,
                 use_focal_loss=False
                 ):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.weight = weight
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.focal_loss = FocalLoss()
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        if self.use_focal_loss:
            floss = self.focal_loss(logits, label)
        logs = self.log_softmax(logits)
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        label[[a, torch.arange(label.size(1)), *b]] = 0

        if self.weight is not None:
            sum_loss = -torch.sum(torch.sum((logs*label)*self.weight, dim=1))
        else:
            sum_loss = -torch.sum(torch.sum(logs*label, dim=1))

        if self.reduction == 'mean':
            loss = sum_loss / n_valid
        elif self.reduction == 'sum':
            loss = sum_loss
        if self.use_focal_loss:
            loss += 0.5*floss
        return loss

class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=1.,
                 gamma=2,
                 reduction='sum',
                 ignore_lb=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_lb = ignore_lb

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        ignore = label.data.cpu() == self.ignore_lb
        n_valid = (ignore == 0).sum()
        label[ignore] = 0

        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        mask = torch.ones_like(logits)
        mask[[a, torch.arange(mask.size(1)), *b]] = 0

        # compute loss
        probs = torch.sigmoid(logits)
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        pt = torch.where(lb_one_hot == 1, probs, 1-probs)
        alpha = self.alpha*lb_one_hot + (1-self.alpha)*(1-lb_one_hot)
        loss = -alpha*((1-pt)**self.gamma)*torch.log(pt + 1e-12)
        loss[mask == 0] = 0
        if self.reduction == 'mean':
            loss = loss.sum(dim=1).sum()/n_valid
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, alpha=1., reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha, (float, int)):
#             self.alpha = torch.Tensor([alpha, 1 - alpha])
#         if isinstance(alpha, list):
#             self.alpha = torch.Tensor(alpha)
#         self.reduction = reduction

#     def forward(self, input, target):
#         if input.dim() > 2:
#             input = input.view(
#                 input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1, 1)

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1, target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type() != input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0, target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1 - pt)**self.gamma * logpt
#         if self.reduction == 'mean':
#             return loss.mean()
#         else:
#             return loss.sum()









