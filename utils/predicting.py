import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from metrics import cal_IOU

def predict(model, data_set, data_loader, num_classes,counting=False,):
    """ Validate after training an epoch
    Note:
    """
    model.eval()
    n = len(data_set)
    ious = np.zeros(n,dtype=float)
    idx = 0
    val_acc =[]
    for bc_cnt, bc_data in enumerate(data_loader):
        if counting:
            print('%d/%d' % (bc_cnt, len(data_set)//data_loader.batch_size))
        imgs, masks = bc_data
        imgs = Variable(imgs).cuda()
        masks = Variable(masks).cuda()
        # imgs.volatile =True

        outputs = F.softmax(model(imgs), dim=1)
        if outputs.size() != masks.size():
            outputs = F.upsample(outputs, size=masks.size()[-2:], mode='bilinear')


        # cal pixel acc
        _, outputs = torch.max(outputs, dim=1)
        batch_corrects = torch.sum((outputs == masks).long()).data[0]
        batch_acc = 1. * batch_corrects / (masks.size(0) * masks.size(1) * masks.size(2))
        val_acc.append(batch_acc)
        outputs = outputs.cpu().data.numpy()
        masks = masks.cpu().data.numpy()
        ious[idx: idx+imgs.size(0)] = cal_IOU(outputs, masks,num_classes=num_classes)
        idx += imgs.size(0)

    return ious.mean(),np.array(val_acc).mean()
