import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
class BCELogitsLossWith(nn.Module):

    def __init__(self, size_average=True):
        super(BCELogitsLossWith, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        '''
        :param input: Variable of shape (N, C, H, W)  logits
        :param target:  Variable of shape (N, C, H, W)  0~1 float
        :param mask: Variable of shape (N, C)  0. or 1.  float
        :return:
        '''
        target=target.unsqueeze(1)
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        if self.size_average:
            return loss.sum()/(target.size(2)*target.size(3))
        else:
            return loss.sum()


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-100):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, y_preds, y_true):
        '''
        :param y_preds: (N, C, H, W), Variable of FloatTensor
        :param y_true:  (N, H, W), Variable of LongTensor
        # :param weights: sample weights, (N, H, W), Variable of FloatTensor
        :return:
        '''
        logp = F.log_softmax(y_preds,dim=1)    # (N, C, H, W)
        ylogp = torch.gather(logp, 1, y_true.view(y_true.size(0), 1, y_true.size(1), y_true.size(2))) # (N, 1, H, W)
        return -(ylogp.squeeze(1)).mean()


class BCELogitsLossWithMask(nn.Module):

    def __init__(self, size_average=True):
        super(BCELogitsLossWithMask, self).__init__()
        self.size_average = size_average

    def forward(self, input, target, mask):
        '''
        :param input: Variable of shape (N, C, H, W)  logits
        :param target:  Variable of shape (N, C, H, W)  0~1 float
        :param mask: Variable of shape (N, C)  0. or 1.  float
        :return:
        '''
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        loss = loss * mask.unsqueeze(2).unsqueeze(3).expand_as(input)

        if self.size_average:
            return loss.sum() / (mask.sum()+1)
        else:
            return loss.sum()



class CrossEntropyLoss2d_sigmod_withmask(nn.Module):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w
    # label shape batch_size x channels x h x w
    def __init__(self):
        super(CrossEntropyLoss2d_sigmod_withmask, self).__init__()
        self.Sigmoid = nn.Sigmoid()
    def forward(self, inputs, targets,masks):
        if not (targets.size() == inputs.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), input.size()))

        inputs = self.Sigmoid(inputs)
        loss = -targets * torch.log(inputs + 1e-7) - (1 - targets) * torch.log(1 - inputs + 1e-7)
        loss = loss * masks.unsqueeze(2).unsqueeze(3).expand_as(inputs)
        return torch.sum(loss)/inputs.size(0)


class MESLossWithMask(nn.Module):
    def __init__(self, size_average):
        super(MESLossWithMask, self).__init__()
        self.size_average = size_average
    def forward(self, inputs, targets, masks):
        if not (targets.size() == inputs.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), input.size()))
        loss = (targets - inputs) ** 2 * masks.unsqueeze(2).unsqueeze(3).expand_as(inputs)

        if self.size_average:
            return loss.sum() / (masks.sum()+1)
        else:
            return loss.sum()
from torch.nn import MSELoss
class Matting_Loss(nn.Module):
    def __init__(self, size_average=True):
        super(Matting_Loss, self).__init__()
        self.size_average = size_average
        self.smooth_l1 =torch.nn.SmoothL1Loss(reduce=True, size_average=True)
    def forward(self, inputs, targets, imgs):
        targets=targets.unsqueeze(1).float()
        if not (targets.size() == inputs.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), input.size()))
        loss = self.smooth_l1(inputs, targets)
        loss += self.smooth_l1(inputs*imgs, targets*imgs)
        return loss
if __name__ == '__main__':
    # a = np.random.rand(4,3,5,5)
    # a= Variable(torch.from_numpy(a))
    # print a.size()
    #
    # ll = nn.Softmax2d()
    #
    # p = ll(a)
    # print p[0].sum(0)
    # loss_fn = torch.nn.MSELoss(reduce=False, size_average=True)
    # loss_fn = torch.nn.SmoothL1Loss(reduce=True, size_average=True)

    input = torch.autograd.Variable(torch.rand(1, 2,224,224))
    # _, preds = torch.max(input, 1)
    # print(preds)
    # print(torch.sum(preds))
    # target = torch.autograd.Variable(torch.randn(3,1,224,224))
    # img = torch.autograd.Variable(torch.randn(3,3,224,224))
    #
    # # loss = loss_fn(input, target)
    # loss_fn=Matting_Loss()
    # loss = loss_fn(input, target,img)
    #
    # print(loss)
    # # loss=torch.mean(torch.sqrt(loss))
    # # loss=torch.mean(loss)
    # print(loss)
    #
    # print(input.size(), target.size(), loss.size())

    # _,idx = torch.max(a.view(4,3,-1),2)
    # idx = idx.numpy()
    # print idx.shape
    # idx1,idx2 = np.unravel_index(idx, dims=(5,5))
    # idx_x_y = np.zeros((4,3,2))
    # idx_x_y[:,:,0] = idx1
    # idx_x_y[:,:,1] = idx2
    #
    # print a[1]
    # print idx_x_y[1]

    # inputs = np.random.rand(3,5,10,10)
    # inputs = Variable(torch.from_numpy(inputs)).float()
    #
    # targets = np.random.rand(3,5,10,10)
    # targets = Variable(torch.from_numpy(targets)).float()
    #
    # mask = np.array([[1,1,0,0,0],[0,0,1,1,0],[0,0,0,0,1]])
    # mask = Variable(torch.from_numpy(mask).float())
    # mask = mask
    #
    #
    #
    # crt = BCELogitsLossWithMask()
    # loss = crt(inputs, targets, mask)
    # print loss
    #
