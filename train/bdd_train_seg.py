from utils.seg_train_util import train, trainlog
from torch.optim import lr_scheduler,Adam,RMSprop
from utils.seg_losses import CrossEntropyLoss2d
from dataset.data_aug import Compose,ResizeImg,RandomHflip,Normalize,RandomVflip
from models.deeplabv3_resnet import Deeplab_v3,Deeplab_v3_d
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.utils.data as torchdata
import logging
import argparse
from dataset.data_aug import *
import torch
from dataset.bdd_seg_dataset import BDD_data,collate_fn
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=9, help='size of each image batch')
parser.add_argument('--learning_rate', type=float, default=0.000001, help='learning rate')
parser.add_argument('--checkpoint_dir', type=str, default='/media/hszc/model/detao/models/bdd/deeplabv3_resnet101_d_flip', help='directory where model checkpoints are saved')
parser.add_argument('--cuda_device', type=str, default="0,2,3", help='whether to use cuda if available')
parser.add_argument('--net', dest='net',type=str, default='deeplabv3_resnet101_d',help='gcn, drn_gcn,mobile_unet,deeplabv3_resnet101,deeplabv3_resnet50,deeplabv3_resnet101_d')
parser.add_argument('--optim', dest='optim',type=str, default='Adam',help='Adam,RMSprop')

# parser.add_argument('--resume', type=str, default="/media/hszc/model/detao/models/bdd/deeplabv3z_101/weights-9-1494-[0.83046]-[0.96626].pth", help='path to resume weights file')
# parser.add_argument('--resume', type=str, default='/media/hszc/model/detao/models/bdd/deeplabv3_resnet101_d/weights-10-6219-[0.84933]-[0.97039].pth', help='path to resume weights file')
parser.add_argument('--resume', type=str, default="/media/hszc/model/detao/models/bdd/deeplabv3_resnet101_d_flip/weights-12-6663-[0.85003]-[0.97086].pth", help='path to resume weights file')

parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--start_epoch', type=int, default=0, help='number of start epoch')
# parser.add_argument('--img_size', type=int, default=256, help='size of each image dimension')

parser.add_argument('--save_checkpoint_val_interval', type=int, default=4000, help='interval between saving model weights')
parser.add_argument('--print_interval', type=int, default=100, help='interval between print log')
parser.add_argument('--img_root', type=str, default= '/media/hszc/model/detao/data/bdd/bdd100k/images/100k', help='whether to train img root')
parser.add_argument('--label_root', type=str, default= '/media/hszc/model/detao/data/bdd/bdd100k/drivable_maps/labels', help='whether to val img root')
# /media/hszc/model/detao/models/bdd/deeplabv3/weights-19-3381-[0.68856]-[0.96513].pth
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_device
'''
deeplabv3_50 13.5598 /media/hszc/model/detao/models/bdd/deeplabv3/weights-19-3381-[0.68856]-[0.96513].pth
'''
class trainAug(object):
    def __init__(self, size=(360, 640)):
        self.augment = Compose([
            RandomBrightness(delta=30),
            ResizeImg(size=size),
            RandomHflip(),
            Normalize(mean=None, std=None)
        ])

    def __call__(self, *args):
        return self.augment(*args)

class valAug(object):
    def __init__(self,size=(360, 640)):
        self.augment = Compose([
            ResizeImg(size=size),
            Normalize(mean=None, std=None)
        ])
    def __call__(self, *args):
        return self.augment(*args)

def lr_lambda(epoch):
    if epoch < 10:
        return 1
    elif epoch < 20:
        return 0.1
    elif epoch < 40:
        return 0.05
    else:
        return 0.01

if __name__ == '__main__':
    import hashlib
    import glob
    # saving dir
    save_dir = opt.checkpoint_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logfile = '%s/trainlog.log' % save_dir
    trainlog(logfile)

    train_root = os.path.join(opt.label_root,"train")
    val_root = os.path.join(opt.label_root,"val")

    train_pd =pd.DataFrame(glob.glob(train_root+"/*.png"),columns=["img_path"])
    val_pd =pd.DataFrame(glob.glob(val_root+"/*.png"),columns=["img_path"])

    data_set={}
    data_set["train"] = BDD_data(train_pd,trainAug(),img_root=opt.img_root,label_root=opt.label_root)
    data_set["val"] = BDD_data(val_pd,trainAug(),img_root=opt.img_root,label_root=opt.label_root)
    data_loader={}
    data_loader['train'] = torchdata.DataLoader(data_set['train'], opt.batch_size, num_workers=opt.n_cpu,
                                                shuffle=True, pin_memory=True, collate_fn=collate_fn)
    data_loader['val'] = torchdata.DataLoader(data_set['val'], 3, num_workers=opt.n_cpu,
                                              shuffle=False, pin_memory=True, collate_fn=collate_fn)

    print len(data_set['train']), len(data_set['val'])

    # gcn, drn_gcn,mobile_unet
    if opt.net == 'deeplabv3_resnet101':
        model = Deeplab_v3(num_classes=3, layers=101)
    elif opt.net == 'deeplabv3_resnet50':
        model = Deeplab_v3(num_classes=3, layers=50)
    elif opt.net == 'deeplabv3_resnet101_d':
        model = Deeplab_v3_d(num_classes=3, layers=101)
    else:
        print("network is not defined")
    # logging.info(model)

    criterion = CrossEntropyLoss2d()

    if opt.optim == 'Adam':
        optimizer = Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)
    elif opt.optim == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)

    exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if opt.resume:
        model.eval()
        logging.info('resuming finetune from %s' % opt.resume)
        try:
            model.load_state_dict(torch.load(opt.resume))
        except KeyError:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(opt.resume))
        # optimizer.load_state_dict(torch.load(os.path.join("/media/hszc/model/detao/models/portrait/Mobile_sU_Adam_true_bs8_optim/", 'optimizer-state.pth')))
    # model = torch.nn.DataParallel(model)
    model.cuda()

    train(model,
          epoch_num=opt.epochs,
          start_epoch=opt.start_epoch,
          optimizer=optimizer,
          criterion=criterion,
          exp_lr_scheduler=exp_lr_scheduler,
          data_set=data_set,
          data_loader=data_loader,
          save_dir=save_dir,
          print_inter=opt.print_interval,
          val_inter=opt.save_checkpoint_val_interval,
          )
