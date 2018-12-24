# coding=utf8
from __future__ import division
import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt

class BDD_data(data.Dataset):
    def __init__(self, anno_pd,transforms=None,debug=False,test=False,img_root=None,label_root=None):
        self.img_list = anno_pd['img_path'].tolist()
        self.transforms = transforms
        self.debug =debug
        self.test =test
        self.img_root=img_root
        self.label_root=label_root
        self.num_classes=3

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        mask_path = self.img_list[index]
        image_path=mask_path.replace(self.label_root,self.img_root).replace("_drivable_id.png",".jpg")
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)  # [h,w,3]  RGB

        if not self.test:
            # mask_path = image_path.replace(self.img_root,self.label_root).replace(".jpg",".png")
            # mask_img = cv2.imread(mask_path)
            # mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            # mask =np.where(mask_img[:,:,1]>0,1,0)
            # mask +=np.where(mask_img[:,:,2]>0,2,0)
            mask = cv2.imread(mask_path,0)
            mask[mask>2] = 2
            mask = np.array(mask,dtype=np.uint8)
        else:
            mask = np.ones((img.shape[0],img.shape[1]),dtype=np.uint8)

        mask_teacher=None
        if self.transforms:
            img, mask, mask_teacher = self.transforms(img, mask, mask_teacher)
        # print(np.unique(mask))
        if not self.debug:
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).long()
        if not self.test:
            return img, mask
        else:
            return img, mask,image_path
class BDD_data_test(data.Dataset):
    def __init__(self, img_list,transforms=None):
        self.img_list = img_list
        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        image_path = self.img_list[index]
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)  # [h,w,3]  RGB

        mask_teacher=None
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)

        if self.transforms:
            img, mask, mask_teacher = self.transforms(img, mask, mask_teacher)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img,image_path

# def collate_fn_test(batch):
#     imgs = []
#     masks = []
#     file_id = []
#     depth = []
#
#     for sample in batch:
#         imgs.append(sample[0])
#         masks.append(sample[1])
#         file_id.append(sample[2])
#         depth.append(sample[3])
#
#     return torch.stack(imgs, 0), \
#            torch.stack(masks, 0), \
#            file_id,depth

def collate_fn_test(batch):
    imgs = []
    image_path = []

    for sample in batch:
        imgs.append(sample[0])
        image_path.append(sample[1])

    return torch.stack(imgs, 0), \
           image_path

def collate_fn(batch):
    imgs = []
    masks = []

    for sample in batch:
        imgs.append(sample[0])
        masks.append(sample[1])

    return torch.stack(imgs, 0), \
           torch.stack(masks, 0)

def plot2x2Array(image, mask):
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(image)
    axarr[0].imshow(mask*127,alpha=0.4)

    axarr[1].imshow(mask)
    axarr[0].grid()
    axarr[1].grid()
    axarr[0].set_title('Image')
    axarr[1].set_title('Mask')

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from dataset.data_aug import *
    from utils.salt_submission  import RLenc
    import glob
    import json
    from matplotlib.path import Path
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.image as mpimg
    class trainAug(object):
        def __init__(self, size=(360, 640)):
            self.augment = Compose([
                # RandomSelect([
                #     RandomSmall(ratio=0.1),
                #     RandomRotate(angles=(-20, 20), bound='Random'),
                #     RandomResizedCrop(size=size),
                # ]),
                RandomBrightness(delta=30),
                ResizeImg(size=size),
                RandomHflip(),
                # Normalize(mean=None, std=None)
            ])

        def __call__(self, *args):
            return self.augment(*args)


    class valAug(object):
        def __init__(self,  size=(360, 640)):
            self.augment = Compose([
                ResizeImg(size=size),
                Normalize(mean=None, std=None)
            ])

        def __call__(self, *args):
            return self.augment(*args)


    # img_root ="/media/hszc/model/detao/data/bdd/bdd/bdd100k_images/bdd100k_images/bdd100k/images/100k"
    # label_root = "/media/hszc/model/detao/data/bdd/bdd/bdd100k/labels_seg/"
    # label_list =glob.glob(label_root+"/val/*.png")
    # print(len(label_list))
    # for label_path in   label_list:
    #     img_path=label_path.replace(label_root,img_root).replace(".png",".jpg")
    #     img=cv2.imread(img_path)
    #
    #     img=cv2.imread(label_path)
    #     mask=np.zeros((720, 1280),dtype=int)
    #     mask=np.where(img[:,:,1]>0,1,0)
    #     mask +=np.where(img[:,:,2]>0,2,0)
    #
    #     print(img.shape)
    #     plt.subplot(121)
    #     plt.imshow(img)
    #     plt.subplot(122)
    #     plt.imshow(mask*127)
    #     plt.show()
    # print(val_pd["labels"].iloc[0])
    img_root = "/media/hszc/model/detao/data/bdd/bdd100k/images/100k"
    label_root = "/media/hszc/model/detao/data/bdd/bdd100k/drivable_maps/labels"
    train_root = os.path.join(label_root,"train")
    val_root = os.path.join(label_root,"val")

    train_pd =pd.DataFrame(glob.glob(train_root+"/*.png"),columns=["img_path"])
    # # train_pd["img_path"] = train_pd["label_path"].apply(lambda x:x.replace("labels","images").replace(".png",".jpg"))
    val_pd =pd.DataFrame(glob.glob(val_root+"/*.png"),columns=["label_path"])
    # val_pd["img_path"] = val_pd["label_path"].apply(lambda x:x.replace("labels","images").replace(".png",".jpg"))
    #
    print(val_pd.head())

    dataset = BDD_data(train_pd,trainAug(),debug=True,img_root=img_root,label_root=label_root)

    for data in dataset:
        image, mask =data
        print(image.shape)
        plot2x2Array(image, mask)

        plt.show()