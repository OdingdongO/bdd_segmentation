import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from dataset.bdd_seg_dataset import BDD_data_test,collate_fn_test
import argparse
import glob
import tqdm
from models.deeplabv3_resnet import Deeplab_v3, Deeplab_v3_d
import torch.utils.data as torchdata


def tensor_flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
                 else x.new(torch.arange(x.size(i) - 1, -1, -1).tolist()).long()
                 for i in range(x.dim()))
    return x[inds]
parser = argparse.ArgumentParser()
parser.add_argument('--test_root', type=str, default= '/media/hszc/model/detao/data/bdd/test/testB', help='whether to train img root')
# parser.add_argument('--test_root', type=str, default= '/media/hszc/model/detao/data/bdd/bdd_testB/testB_img100', help='whether to train img root')
parser.add_argument('--save_root', type=str, default= '/media/hszc/model/detao/data/bdd/bdd_testB/testB_result_100', help='whether to train img root')

parser.add_argument('--use_d_deeplabv3', type=bool, default=True)
parser.add_argument('--use_deeplabv3_flip', type=bool, default=False)
parser.add_argument('--use_deeplabv3', type=bool, default=True)

parser.add_argument('--n_cpu', type=int, default=24, help='number of cpu threads to use during batch generation')
parser.add_argument('--batch_size', type=int, default=8, help='size of each image batch')
parser.add_argument('--cuda_device', type=str, default="3", help='whether to use cuda if available')
opt = parser.parse_args()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] =opt.cuda_device
    img_list=glob.glob(opt.test_root+"/*.jpg")

    data_set= BDD_data_test(img_list)
    data_loader = torchdata.DataLoader(data_set, opt.batch_size, num_workers=opt.n_cpu,
                                              shuffle=False, pin_memory=True, collate_fn=collate_fn_test)
    assert opt.use_d_deeplabv3 or opt.use_deeplabv3
    # two model 7970
    resume1="weights/deeplabv3z_101/weights-1-166-[0.84165]-[0.96885].pth"  #77.56 online 78.1909
    resume="weights/deeplabv3_resnet101_d_flip/weights-3-666-[0.85061]-[0.97098].pth"  #78.90 online 79.0976 flip 79.33

    if not os.path.exists(opt.save_root):
        os.makedirs(opt.save_root)

    if opt.use_deeplabv3:
        model_1 = Deeplab_v3(num_classes=3, layers=101)
        model_1.eval()
        print ('resuming finetune from %s' % resume1)
        try:
            model_1.load_state_dict(torch.load(resume1))
        except KeyError:
            model_1 = torch.nn.DataParallel(model_1)
            model_1.load_state_dict(torch.load(resume1))

    if opt.use_d_deeplabv3:
        model = Deeplab_v3_d(num_classes=3, layers=101)
        model.eval()
        print ('resuming finetune from %s' % resume)

        try:
            model.load_state_dict(torch.load(resume))
        except KeyError:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(resume))


    for data in tqdm.tqdm(data_loader):
        mask_list=[]
        imgs,img_flips,img_paths = data

        imgs = Variable(imgs.cuda())
        imgs.volatile = True
        if opt.use_d_deeplabv3:
            outputs = model(imgs)
            softmax_mask = F.softmax(outputs).data
            mask_list.append(softmax_mask)

        if opt.use_deeplabv3_flip:
            img_flips = Variable(img_flips.cuda())
            img_flips.volatile = True

            outputs_flip = model(img_flips)
            softmax_mask_flip = F.softmax(outputs_flip)
            softmax_mask_flip = tensor_flip(softmax_mask_flip.data,3)
            mask_list.append(softmax_mask_flip)

        if opt.use_deeplabv3:
            outputs_1 = model_1(imgs)
            softmax_mask_1 = F.softmax(outputs_1).data
            mask_list.append(softmax_mask_1)

        softmax_mask_merge =sum(mask_list)/len(mask_list)

        softmax_mask_merge = F.upsample_bilinear(softmax_mask_merge, size=(720, 1280))
        softmax_mask_merge = softmax_mask_merge.data.cpu().numpy()

        for mask,img_path in zip(softmax_mask_merge,img_paths):
            mask_out = np.argmax(mask, axis=0)
            cv2.imwrite(os.path.join(opt.save_root,os.path.basename(img_path).replace(".jpg", "_drivable_id.png")), mask_out)