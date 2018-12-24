from __future__ import division
import cv2
import numpy as np
from numpy import random
import math
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
__all__ = ['Compose','ResizeImg',"Normalize","RandomResizedCrop","RandomHflip",'deNormalize',"RandomVflip","RandomSelect",
           "RandomSmall","RandomRotate","RandomBrightness"]

def rotate_bound(image, angle, borderValue=0, borderMode=None):
    # grab the dimensions of the image and then determine the
    # center
    h, w = image.shape[:2]

    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    if borderMode is None:
        rotated = cv2.warpAffine(image, M, (nW, nH), borderValue=borderValue, borderMode=0)
    else:
        rotated = cv2.warpAffine(image, M, (nW, nH),borderValue=borderValue, borderMode=borderMode)

    return rotated


def rotate_nobound(image, angle,borderValue=0, borderMode=None):
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.)
    if borderMode is None:
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=borderValue, borderMode=0)
    else:
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=borderValue, borderMode=borderMode)

    return rotated

def fixed_crop(src, x0, y0, w, h, size=None):
    out = src[y0:y0 + h, x0:x0 + w]
    if size is not None and (w, h) != size:
        out = cv2.resize(out, (size[0], size[1]), interpolation=cv2.INTER_CUBIC)
    return out
def scale_down(src_size, size):
    w, h = size
    sw, sh = src_size
    if sh < h:
        w, h = float(w * sh) / h, sh
    if sw < w:
        w, h = sw, float(h * sw) / w
    return int(w), int(h)

def center_crop(src, size):
    h, w = src.shape[0:2]
    new_w, new_h = scale_down((w, h), size)

    x0 = int((w - new_w) / 2)
    y0 = int((h - new_h) / 2)

    out = fixed_crop(src, x0, y0, new_w, new_h, size)
    return out


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        # print("Compose")

    def __call__(self, *args):
        # print("Compose run")

        for t in self.transforms:
            args = t(*args)
            # print('  %s' % (t.__class__.__name__), len(args))
        return args


class RandomSelect(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, mask, mask_teacher=None):
        idx = random.randint(len(self.transforms))
        # print('  %s %s' % (self.transforms[idx].__class__.__name__, self.transforms[idx].__dict__))
        return self.transforms[idx](image, mask, mask_teacher)

class Center_Crop(object):
    def __init__(self, size,inter=cv2.INTER_LINEAR):
        self.size = size

    def __call__(self, img, mask,mask_teacher=None):

        if mask_teacher is not None:
            return center_crop(img, self.size), center_crop(mask, self.size), mask_teacher
        else:
            return  center_crop(img, self.size), center_crop(mask, self.size),None

class ResizeImg(object):
    def __init__(self, size,inter=cv2.INTER_LINEAR):
        self.size = size
        self.inter = inter

    def __call__(self, image, mask,mask_teacher=None):
        if mask_teacher is not None:
            mask_teacher = cv2.resize(mask_teacher, (self.size[1], self.size[0]), interpolation=self.inter)
        return cv2.resize(image, (self.size[1], self.size[0]), interpolation=self.inter), \
               cv2.resize(mask, (self.size[1], self.size[0]), interpolation=self.inter), \
               mask_teacher

class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.25, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self,img, mask, mask_teacher=None):
        h, w, _ = img.shape
        area = h * w
        for attempt in range(10):
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            new_w = int(round(math.sqrt(target_area * aspect_ratio)))
            new_h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                new_h, new_w = new_w, new_h

            if new_w < w and new_h < h:
                x0 = random.randint(0, w - new_w)
                y0 = random.randint(0, h - new_h)
                out_img = fixed_crop(img, x0, y0, new_w, new_h, self.size)
                out_mask = fixed_crop(mask, x0, y0, new_w, new_h, self.size)
                if mask_teacher is not None:
                    mask_teacher = fixed_crop(mask_teacher, x0, y0, new_w, new_h, self.size)
                return out_img, out_mask, mask_teacher
        # Fallback
        if mask_teacher is not None:
            mask_teacher = center_crop(mask_teacher,self.size)
        return center_crop(img, self.size), center_crop(mask, self.size), mask_teacher


class RandomHflip(object):
    def __call__(self, image, mask, mask_teacher=None):
        if random.randint(2):
            if mask_teacher is None:
                return cv2.flip(image, 1), cv2.flip(mask, 1), None
            else:
                return cv2.flip(image, 1), cv2.flip(mask, 1), cv2.flip(mask_teacher, 1)
        else:
            return image, mask, mask_teacher
class Hflip(object):
    def __call__(self, image, mask, mask_teacher=None):
            if mask_teacher is None:
                return cv2.flip(image, 1), cv2.flip(mask, 1), None
            else:
                return cv2.flip(image, 1), cv2.flip(mask, 1), cv2.flip(mask_teacher, 1)
class RandomVflip(object):
    def __call__(self, image,mask, mask_teacher=None):
        if random.randint(2):
            if mask_teacher is None:
                return  cv2.flip(image, 0),cv2.flip(mask, 0), None
            else:
                return cv2.flip(image, 0), cv2.flip(mask, 0), cv2.flip(mask_teacher, 0)
        else:
            return image,mask,mask_teacher

class RandomRotate(object):
    def __init__(self, angles, bound, borderMode='REFLECT', borderValue=None):
        self.angles = angles
        self.bound = bound
        if borderMode=='REFLECT':
            self.borderMode = cv2.BORDER_REFLECT
        else:
            self.borderMode = borderMode
        self.borderValue = borderValue

    def __call__(self,img, mask, mask_teacher=None):
        angle = np.random.uniform(self.angles[0], self.angles[1])
        if isinstance(self.bound, str) and self.bound.lower() == 'random':
            bound = random.randint(2)
        else:
            bound = self.bound

        if bound:
            img = rotate_bound(img, angle, borderMode=self.borderMode, borderValue=self.borderValue)
            mask = rotate_bound(mask, angle, borderValue=0)
            if mask_teacher is not None:
                mask_teacher = rotate_bound(mask_teacher, angle,borderMode=cv2.BORDER_REFLECT)
        else:
            img = rotate_nobound(img, angle, borderMode=self.borderMode, borderValue=self.borderValue)
            mask = rotate_nobound(mask, angle,borderValue=0)
            if mask_teacher is not None:
                mask_teacher = rotate_nobound(mask_teacher, angle,borderMode=cv2.BORDER_REFLECT)
        return img, mask, mask_teacher


class RandomBrightness(object):
    def __init__(self, delta=10):
        assert delta >= 0
        assert delta <= 255
        self.delta = delta

    def __call__(self, image, mask , mask_teacher=None):
        delta = random.uniform(-self.delta, self.delta)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        image[:,:,2] += delta
        image = image.clip(0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image, mask,mask_teacher


class RandomSmall(object):
    def __init__(self, ratio=0.1):
        self.ratio = ratio

    def __call__(self, image, mask,mask_teacher=None):
        h,w = image.shape[0:2]

        ratio = random.uniform(0, self.ratio)
        dw = max(1,int(w*ratio))
        dh = max(1,int(h*ratio))


        w_shift = random.randint(-dw, dw)
        h_shift = random.randint(-dh, dh)

        pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
        pts2 = np.float32([[dw, dh],
                         [w-dw, dh],
                         [dw, h-dh],
                         [w-dw, h-dh]])
        pts2[:,0] += w_shift
        pts2[:,1] += h_shift


        M = cv2.getPerspectiveTransform(pts1, pts2)
        image = cv2.warpPerspective(image, M, (w,h), borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpPerspective(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if mask_teacher is not None:
            mask_teacher = cv2.warpPerspective(mask_teacher, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return image, mask, mask_teacher

class Normalize(object):
    def __init__(self,mean=None, std=None):
        '''
        :param mean: RGB order
        :param std:  RGB order
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        '''
        if mean is not None:
            self.mean = np.array(mean).reshape(1,1,3)
        else:
            self.mean = None

        if std is not None:
            self.std = np.array(std).reshape(1,1,3)
        else:
            self.std = None
    def __call__(self, img, mask,mask_teacher=None):
        '''
        :param image:  (H,W,3)  RGB
        :return:
        '''
        if self.mean is None and self.std is None:
            return  img / 255., mask, mask_teacher
        elif self.mean is not None and self.std is not None:
            return  (img / 255. - self.mean) / self.std, mask, mask_teacher
        elif self.mean is None:
            return img / 255. / self.std, mask, mask_teacher
        else:
            return (img / 255. - self.mean) , mask, mask_teacher


class deNormalize(object):
    def __init__(self,mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        '''
        :param mean: RGB order
        :param std:  RGB order
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        '''
        if mean is not None:
            self.mean = np.array(mean).reshape(1,1,3)
        else:
            self.mean = None

        if std is not None:
            self.std = np.array(std).reshape(1,1,3)
        else:
            self.std = None
    def __call__(self, img):
        '''
        :param image:  (H,W,3)  RGB
        :return:
        '''
        if self.mean is None and self.std is None:
            return  (img * 255.).clip(0,255).astype(np.uint8)
        elif self.mean is not None and self.std is not None:
            return (255*(img*self.std + self.mean)).clip(0,255).astype(np.uint8)
        elif self.mean is None:
            return ((img*self.std) * 255.).clip(0,255).astype(np.uint8)
        else:
            return (255*(img*self.std)).clip(0,255).astype(np.uint8)
class DownCrop():
    def __init__(self, size, scale=(0.36,0.81)):
        self.size = size
        self.scale = scale

    def __call__(self,img, mask,mask_teacher=None):
        h, w, _ = img.shape
        area = h * w

        s = (self.scale[0]+self.scale[1])/2.0

        target_area = s * area

        new_w = int(round(math.sqrt(target_area)))
        new_h = int(round(math.sqrt(target_area)))

        if new_w < w and new_h < h:
            dw = w-new_w
            x0 = int(0.5*dw)
            y0 = h-new_h
            if mask_teacher is not None:
                return fixed_crop(img, x0, y0, new_w, new_h, self.size), \
                       fixed_crop(mask, x0, y0, new_w, new_h, self.size), \
                       fixed_crop(mask_teacher, x0, y0, new_w, new_h, self.size)
            else:
                return fixed_crop(img, x0, y0, new_w, new_h, self.size), \
                       fixed_crop(mask, x0, y0, new_w, new_h, self.size), \
                       None

        if mask_teacher is not None:
            return center_crop(img, self.size), center_crop(mask, self.size), mask_teacher
        else:
            return  center_crop(img, self.size), center_crop(mask, self.size),None

        # Fallback
class Cbct_random_crop():
    def __init__(self, x0,y0,size=386):
        self.size = size
        self.x0=x0
        self.y0=y0
    def __call__(self, img, mask, mask_teacher=None):
        x0=random.randint(self.x0-25,self.x0+25)
        y0=random.randint(self.y0-25,self.y0+25)

        if mask_teacher is not None:
            return fixed_crop(img,x0,y0, w=self.size,h=self.size), fixed_crop(mask,x0,y0, w=self.size,h=self.size), fixed_crop(mask_teacher,x0,y0, w=self.size,h=self.size)
        else:
            return fixed_crop(img,x0,y0, w=self.size,h=self.size), fixed_crop(mask,x0,y0, w=self.size,h=self.size),None


class Cbct_crop():
    def __init__(self, x0,y0,size=386):
        self.size = size
        self.x0=x0
        self.y0=y0
    def __call__(self, img, mask, mask_teacher=None):
        if mask_teacher is not None:
            return fixed_crop(img,self.x0,self.y0, w=self.size,h=self.size), fixed_crop(mask,self.x0,self.y0, w=self.size,h=self.size), fixed_crop(mask_teacher,self.x0,self.y0, w=self.size,h=self.size)
        else:
            return fixed_crop(img,self.x0,self.y0, w=self.size,h=self.size), fixed_crop(mask,self.x0,self.y0, w=self.size,h=self.size),None

class UpperCrop():
    def __init__(self, size,ratio=1,crop=2, scale=(0.09, 0.81)):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.crop = crop

    def __call__(self,img,mask,mask_teacher=None):
        h, w, _ = img.shape
        area = h * w

        if self.crop==1:
            target_area = (4 * self.scale[0] + self.scale[1]) / 5.0 * area
        elif self.crop==0:
            target_area = (self.scale[0] + 4 * self.scale[1]) / 5.0 * area
        elif self.crop==2:
            target_area = (self.scale[0] +  self.scale[1]) / 2.0 * area
        else:
            target_area = self.scale[1]* area


        aspect_ratio = self.ratio

        new_w = int(round(math.sqrt(target_area * aspect_ratio)))
        new_h = int(round(math.sqrt(target_area / aspect_ratio)))

        if new_w < w and new_h < h:
            dw = w-new_w
            x0 = int(0.5*dw)
            y0 = 0
            print(x0, y0, new_w, new_h)
            if mask_teacher is not None:
                return fixed_crop(img, x0, y0, new_w, new_h, self.size), \
                       fixed_crop(mask, x0, y0, new_w, new_h, self.size), \
                       fixed_crop(mask_teacher, x0, y0, new_w, new_h, self.size)
            else:
                return fixed_crop(img, x0, y0, new_w, new_h, self.size), \
                       fixed_crop(mask, x0, y0, new_w, new_h, self.size), \
                       None
        # Fallback
        if mask_teacher is not None:
            return center_crop(img, self.size), center_crop(mask, self.size), mask_teacher
        else:
            return  center_crop(img, self.size), center_crop(mask, self.size),None
class RandomUpperCrop(object):
    def __init__(self, size, scale=(0.49, 0.81)):
        self.size = size
        self.scale = scale

    def __call__(self,img, mask,mask_teacher=None):
        if random.random() < 0.2:
            return img,mask,mask_teacher

        h, w, _ = img.shape
        area = h * w
        for attempt in range(10):
            s = random.uniform(self.scale[0], self.scale[1])
            d = 0.1 + (0.3 - 0.1) / (self.scale[1] - self.scale[0]) * (s - self.scale[0])
            target_area = s * area
            new_w = int(round(math.sqrt(target_area)))
            new_h = int(round(math.sqrt(target_area)))
            # new_w = int(round(math.sqrt(target_area)))
            # new_h = int(round(math.sqrt(target_area)))

            if new_w < w and new_h < h:
                dw = w-new_w
                x0 = random.randint(int((0.5-d)*dw), int((0.5+d)*dw)+1)
                y0 = (random.randint(0, h - new_h))//10
                if mask_teacher is not None:
                    return fixed_crop(img, x0, y0, new_w, new_h, self.size), \
                           fixed_crop(mask, x0, y0, new_w, new_h, self.size), \
                           fixed_crop(mask_teacher, x0, y0, new_w, new_h, self.size)
                else:
                    return fixed_crop(img, x0, y0, new_w, new_h, self.size), \
                           fixed_crop(mask, x0, y0, new_w, new_h, self.size), \
                           None
        # Fallback
        if mask_teacher is not None:
            return center_crop(img, self.size), center_crop(mask, self.size), mask_teacher
        else:
            return center_crop(img, self.size), center_crop(mask, self.size),None

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import os
    from glob import glob
    import pandas as pd
    from portrait_dataset import Sdata
    import logging


    class valAug(object):
        def __init__(self, size=(224, 224)):
            self.augment = Compose([
                RandomSelect([
                    RandomSmall(ratio=0.1),
                    RandomRotate(angles=(-20, 20), bound='Random'),
                    RandomResizedCrop(size=size),
                ]),
                RandomBrightness(delta=30),
                # ExpandBorder(value=0),
                ResizeImg(size=size),
                # Normalize(mean=None, std=None)
            ])

        def __call__(self, *args):
            return self.augment(*args)

    # prepare data
    eval_data_root = '/media/hszc/data1/seg_data/'
    img_paths = sorted(glob(os.path.join(eval_data_root, 'seg_img/*.jpg')))
    mask_paths = ['/'.join(x.split('/')[:-2]) + '/seg_mask/' + x.split('/')[-1].replace('.jpg', '.png') for x in
                  img_paths]
    mask_teacher_paths = ['/'.join(x.split('/')[:-2]) + '/pred_mask(T20)/' + x.split('/')[-1].replace('.jpg', '.npy') for x in
                  img_paths]

    anno = pd.DataFrame({'image_paths': img_paths,
                         'mask_paths': mask_paths,
                         'mask_teacher_paths': mask_teacher_paths,
                         })
    print anno.info()
    data_set = {}
    # data_set['val'] = Sdata(anno_pd=anno, transforms=valAug(size=(448, 448)))
    data_set['val'] = Sdata(anno_pd=anno, transforms=valAug(size=(256, 256)),dis=True)
    img, mask, mask_teacher = data_set['val'][1234]
    img = img.astype(np.uint8)
    print img.shape, mask.shape, img.mean()

    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(mask[:,:,0])
    plt.subplot(133)
    plt.imshow(mask_teacher[:, :, 0])
    plt.show()
