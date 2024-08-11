# -*- coding: utf-8 -*-

import numpy as np 
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2, os, wandb, torch, configs, glob
from statistics import median
import segmentation_models_pytorch_v2 as smp
from tqdm.auto import tqdm
from dataset import *
import albumentations as A

class Trainer:
    def __init__(self, train_img_path, val_img_path, save_path, max_lr = 5e-4, epochs = 80, batch = 3, classes = ['pothole', 'expand', 'crack'], device = 'cuda'):
        self.train_img_path = train_img_path
        self.val_img_path = val_img_path
        self.save_path = save_path
        self.max_lr = max_lr
        self.epochs = epochs
        self.batch = batch
        self.classes = classes
        self.n_classes = len(classes)+1
        self.device = device
      
      
    def train_one_epoch(self):
        running_loss = 0
        self.model.train()
        y_pred = []
        y = []
        prog = tqdm(enumerate(self.train_loader),total=len(self.train_loader))
        for i, data in prog:
            #training phase
            image_tiles, mask_tiles = data
            image = image_tiles.to(self.device); mask = mask_tiles.to(self.device)#; mask_ood = mask_ood_tiles.to(self.device)
            #forward
            output = self.model(image)
            loss_3 = self.criterion_3(output, mask)
            loss_lov = self.criterion_lov(output, mask)
            #loss_ood = self.criterion_ood(output, mask_ood)*5
            loss=(loss_lov + loss_3)          
            #backward
            loss.backward()
            self.optimizer.step() #update weight          
            self.optimizer.zero_grad() #reset gradient
            self.scheduler.step() 
            running_loss += loss.item()/3
            pred_mask = torch.argmax(output, dim=1).contiguous().view(-1)
            y_pred.append(pred_mask)
            mask = mask.contiguous().view(-1)
            y.append(mask)
        y_pred = torch.cat(y_pred)
        y = torch.cat(y)
        iou_list = self.mIoU(y_pred, y, smooth=1e-10, n_classes=self.n_classes)
        return running_loss/len(self.train_loader), iou_list
    
    def val_one_epoch(self):
        self.model.eval()
        running_loss = 0
        y_pred = []
        y = []
        prog = tqdm(enumerate(self.val_loader),total=len(self.val_loader))
        with torch.no_grad():
            for i, data in prog:
                image_tiles, mask_tiles = data
                image = image_tiles.to(self.device); mask = mask_tiles.to(self.device)#; mask_ood = mask_ood_tiles.to(self.device)
                output = self.model(image)
                loss_3 = self.criterion_3(output, mask)
                loss_lov = self.criterion_lov(output, mask)
                #loss_ood = self.criterion_ood(output, mask_ood)*5
                loss=(loss_lov + loss_3)
                running_loss +=  loss.item()/2
                pred_mask = torch.argmax(output, dim=1).contiguous().view(-1)
                y_pred.append(pred_mask)
                mask = mask.contiguous().view(-1)
                y.append(mask)
            y_pred = torch.cat(y_pred)
            y = torch.cat(y)
            iou_list = self.mIoU(y_pred, y, smooth=1e-10, n_classes=self.n_classes)
        return running_loss/len(self.val_loader), iou_list
    
    def fit(self):
        self.train_loader, self.val_loader = self.get_dl()
        #self.model = smp.create_model(arch='DeepLabV3Plus',encoder_name='mit_b0', encoder_weights='imagenet',
        #                                n_classes=2, activation=None, encoder_output_stride=32, encoder_depth=5, decoder_channels=256).to(self.device)
        self.model = smp.create_model(arch='UnetPlusPlus',encoder_name='efficientnet-b6', encoder_weights='imagenet',
                                              classes=self.n_classes, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16]).to(self.device)
        self.criterion_3 = nn.CrossEntropyLoss(ignore_index = 100)
        self.criterion_lov=smp.losses.LovaszLoss(mode='multiclass', ignore_index = 100)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.max_lr, momentum=0.9)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.max_lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.max_lr, epochs=self.epochs,steps_per_epoch=len(self.train_loader))
        self.model.to(self.device)
        for e in range(self.epochs):
            train_loss, train_iou = self.train_one_epoch()
            val_loss, val_iou = self.val_one_epoch()
            self.save_model(e)
            performance_dict = {f'train_IoU{c}':train_iou[i] for i, c in enumerate(self.classes)}
            for i, c in enumerate(self.classes):
                performance_dict[f'val_IoU{c}'] = val_iou[i]
            performance_dict['train_loss'] = train_loss
            performance_dict['val_loss'] = val_loss
            wandb.log(performance_dict)
            
    def get_dl(self):
        height=480
        width=480
        t_train = A.Compose([A.Resize(height, width, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(),
                     A.GridDistortion(p=0.2), A.RandomBrightnessContrast((-0.5,0.5),(-0.5,0.5)),
                     A.GaussNoise()])

        t_val = A.Compose([A.Resize(height, width, interpolation=cv2.INTER_NEAREST),
                        ])
        #datasets
        X_train = glob.glob(self.train_img_path)
        X_val = glob.glob(self.val_img_path)
        train_set = C_P_Dataset(X_train, t_train, classes=self.classes)
        val_set = C_P_Dataset(X_val, t_val, classes=self.classes)

        #dataloader
        batch_size= 3 
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, drop_last=True)

        return train_loader, val_loader
        
    def mIoU(self, pred_mask, mask, smooth=1e-10, n_classes=2):
        with torch.no_grad():
            #pred_mask = F.softmax(pred_mask, dim=1)
            #pred_mask = torch.argmax(pred_mask, dim=1)
            #pred_mask = pred_mask.contiguous().view(-1)
            #mask = mask.contiguous().view(-1)

            iou_per_class = []
            for clas in range(1, n_classes): #loop per pixel class
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
        
    def save_model(self, e):
        if os.path.isdir(self.save_path)==False:
            os.mkdir(self.save_path)
        "Save model locally and on wandb"
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f"{e}.pth"))
        
def main():
    cfg = configs.pothole.train
    model = Trainer(
        train_img_path = cfg.train_img_path,
        val_img_path = cfg.val_img_path,
        save_path = cfg.save_path,
        max_lr = cfg.max_lr,
        epochs = cfg.epochs,
        classes = cfg.classes
        )
    with wandb.init(project="tp_port"):
        model.fit()

    
if __name__ == '__main__':
    main()
