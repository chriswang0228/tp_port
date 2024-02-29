import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import cv2
import albumentations as A
import time
import os
from tqdm.notebook import tqdm
import random

import segmentation_models_pytorch_v2 as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

IMAGE_PATH = '../dataset/road/tptc_img/'
MASK_PATH = '../dataset/road/tptc_mask/'

mask_list=os.listdir(MASK_PATH)
sum_0=0
sum_1=0
sum_2=0
for m in mask_list:
    mask = cv2.imread(MASK_PATH+m, 0)
    mask_0 = np.ones(mask.shape)
    mask_1 = np.zeros(mask.shape)
    mask_2 = np.zeros(mask.shape)
    mask_1[mask==128]=1
    mask_2[mask==255]=1
    mask_0[ mask_1 == 1 ] = 0
    mask_0[ mask_2 == 1 ] = 0
    sum_0=sum_0+mask_0.sum()
    sum_1=sum_1+mask_1.sum()
    sum_2=sum_2+mask_2.sum()
weights = [1.0, sum_0/sum_1, sum_0/sum_2]

X_train, X_test = train_test_split(os.listdir(IMAGE_PATH), test_size=0.1, random_state=19)
X_train, X_val = train_test_split(X_train, test_size=2/9, random_state=19)

class DroneDataset(Dataset):
    
    def __init__(self, img_path, mask_path, X, mean, std, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.mean = mean
        self.std = std
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = np.array(cv2.imread(self.mask_path + self.X[idx][:-4]+'.png', 0))
        label[ label == 128 ] = 1
        label[ label == 255 ] = 2
        if self.transform is not None:
            aug = self.transform(image=img, mask=label)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']
        
        if self.transform is None:
            img = Image.fromarray(img)
        
        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()
            
        return img, mask

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
height=480
width=480

t_train = A.Compose([A.Resize(height, width, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(),
                     A.VerticalFlip(),  A.GridDistortion(p=0.2), A.GaussNoise()])

t_val = A.Compose([A.Resize(height, width, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(),
                   A.VerticalFlip(), A.GridDistortion(p=0.2)])

#datasets
train_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std, t_train)
val_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std, t_val)

#dataloader
batch_size= 3 
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)

n_classes=3
model = smp.create_model(arch='UnetPlusPlus',encoder_name='efficientnet-b6', encoder_weights='imagenet', classes=n_classes, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
#model = torch.load('../model_weight/pretrain/crack_UPP_pretrain.pt')
def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=n_classes):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return iou_per_class

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs, model, train_loader, val_loader, criterion, criterion_lov, optimizer, scheduler, patch=False):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    for x in range(n_classes):
      globals()['val_iou%s' % x] = []
    val_acc = []
    train_iou = []; train_acc = []
    lrs = []
    val_miou0=[]
    val_miou1=[]
    val_miou2=[]

    train_miou0=[]
    train_miou1=[]
    train_miou2=[]
    best_model=model

    min_iou = 0
    increase = 1 ; not_improve=0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        #training loop
        model.train()
        train_iou_score = np.zeros([len(train_loader),n_classes])
        for i, data in enumerate(train_loader):
            #training phase
            image_tiles, mask_tiles = data
            image = image_tiles.to(device); mask = mask_tiles.to(device);
            #forward
            output = model(image)
            train_iou_score[i] =  mIoU(output, mask)
            loss = criterion(output, mask)
            loss_lov = criterion_lov(output, mask)
            loss=(loss+loss_lov)/2
            #evaluation metrics
            #iou_score += mIoU(output, mask)
            accuracy += pixel_accuracy(output, mask)
            #backward
            loss.backward()
            optimizer.step() #update weight          
            optimizer.zero_grad() #reset gradient
            
            #step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step() 
            
            running_loss += loss.item()



        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = np.zeros([len(val_loader),n_classes])
            #validation loop
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    #reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data
                    image = image_tiles.to(device); mask = mask_tiles.to(device);
                    output = model(image)
                    #evaluation metrics
                    val_iou_score[i] =  mIoU(output, mask)
                    test_accuracy += pixel_accuracy(output, mask)
                    #loss
                    loss = criterion(output, mask)  
                    loss_lov = criterion_lov(output, mask)     
                    loss=loss+loss_lov                         
                    test_loss += loss.item()
            
            #calculatio mean for each batch
            train_losses.append(running_loss/len(train_loader))
            test_losses.append(test_loss/len(val_loader))


            
            #iou
            per_class_iou_train=np.nanmean(train_iou_score,axis=0)
            #for i,score in enumerate(per_class_iou_train):
              #globals()['train_iou%s' % i].append(score)
            #train_iou.append(iou_score/len(train_loader))
            train_acc.append(accuracy/len(train_loader))
            val_acc.append(test_accuracy/ len(val_loader))
            print("Epoch:{}/{}..".format(e+1, epochs),
                  "Train Loss: {:.3f}..".format(running_loss/len(train_loader)),
                  "Val Loss: {:.3f}..".format(test_loss/len(val_loader)),
                  "Train Acc:{:.3f}..".format(accuracy/len(train_loader)),
                  "Val Acc:{:.3f}..".format(test_accuracy/len(val_loader)),
                  "Time: {:.2f}m".format((time.time()-since)/60))
            for i in range(n_classes):
              print("Train mIoU[{:.0f}]: {:.3f}..".format(i,per_class_iou_train[i]))
            train_miou0.append(per_class_iou_train[0])
            train_miou1.append(per_class_iou_train[1])
            train_miou2.append(per_class_iou_train[2])


            per_class_iou=np.nanmean(val_iou_score,axis=0)
            for i,score in enumerate(per_class_iou):
              globals()['val_iou%s' % i].append(score)
            #train_iou.append(iou_score/len(train_loader))
            for i in range(n_classes):
              print("Val mIoU[{:.0f}]: {:.3f}..".format(i,per_class_iou[i]))
            val_miou0.append(per_class_iou[0])
            val_miou1.append(per_class_iou[1])
            val_miou2.append(per_class_iou[2])
        
            if min_iou < (per_class_iou[1]):
                print('IoU Increasing.. {:.3f} >> {:.3f} '.format(min_iou, (per_class_iou[1])))
                min_iou = (per_class_iou[1])
                increase += 1
                not_improve=0
                print('saving model...')
                best_model=model
                    

            if (per_class_iou[1]) > min_iou:
                not_improve += 1
                print(f'Loss Not Decrease for {not_improve} time')

    history = {'train_loss' : train_losses, 'val_loss': test_losses,'train_iou0':train_miou0,'train_iou1':train_miou1,'train_iou2':train_miou2,
              'val_miou0':val_miou0,'val_miou1':val_miou1,'val_miou2':val_miou2,
               'train_acc':train_acc, 'val_acc':val_acc
               }
    print('Total time: {:.2f} m' .format((time.time()- fit_time)/60))
    return history ,best_model

max_lr = 1e-3
epoch = 80
weight_decay = 1e-4
class_weights = torch.FloatTensor(weights).to(device)
loss3 = nn.CrossEntropyLoss(weight=class_weights)
loss_lov=smp.losses.LovaszLoss(mode='multiclass')
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                            steps_per_epoch=len(train_loader))

history ,best_model= fit(epoch, model, train_loader, val_loader, loss3, loss_lov, optimizer, sched)
torch.save(best_model, '../model_weight/triple_class/crack_UPP(TL)_merge.pt')
pd.DataFrame(history).to_csv('../l_curve/history_curve_crack_UPP(TL)_triple_class_merge')