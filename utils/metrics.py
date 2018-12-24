import numpy as np

def cal_IOU(pred_b, gt_b, num_classes):
    '''
    :param pred_b: shape of (bs, H, W), per pixel class
    :param gt_b:  shape of (bs, H, W), per pixel class
    :param num_classes:
    :return:
    '''
    bs = pred_b.shape[0]
    ious = np.zeros((bs, num_classes),dtype=float)
    cnt = np.zeros(bs)
    for i in xrange(num_classes):
        inter = (((pred_b==i) & (gt_b==i))>0).sum(axis=1).sum(axis=1)
        union = (((pred_b==i) + (gt_b==i))>0).sum(axis=1).sum(axis=1)
        valid = (union>0).astype(int)
        ious[:,i] = 1. * inter  / (union+(1-valid))
        cnt += valid

    miou = ious.sum(axis=1) / cnt
    return miou


