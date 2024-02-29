import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import PIL
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import cv2
import albumentations as A
import time
from tqdm.notebook import tqdm
import random
import segmentation_models_pytorch_v2 as smp
import shutil
from skimage import io
from util import *
import torch
from torchvision import transforms as T
import tifffile as tifi
import math
import scipy
import pickle
import config as cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

import segmentation_models_pytorch_v2

otho_img = cv2.imread('../data_ckpt/dataset/clip/clip.jpg')
clip_mask = cv2.imread('../data_ckpt/dataset/clip/clip_mask.png')
road_id_mask = cv2.imread('../data_ckpt/dataset/clip/road_id_mask_compression.png', 0)
detected_area_map = np.zeros(road_id_mask.shape)
road_id_dict = {}
for i in np.unique(road_id_mask)[1:]:
    road_id_dict[i] = list()
score_dict = {}
width = int(clip_mask.shape[1]/10) # keep original width
height = int(clip_mask.shape[0]/10)
dim = (width, height)
clip_mask_compression = cv2.resize(clip_mask, dim) #壓縮十倍減少運算時間
otho_pix_per_meter = 1/(math.dist([2784295.23560,289479.57382], [2784417.268382,289637.447092])/math.dist([5442, 0], [574, 6298])) #每公尺的正射圖像素數
total_area_dict = {}
for i in np.unique(road_id_mask)[1:]:
    a = (road_id_mask==i).sum()/(otho_pix_per_meter/10)**2   #m^2
    total_area_dict[i] = a 
model=torch.load('../data_ckpt/ckpt/crack_UPP(TL).pt')
model.eval().to(device)   

IMAGE_PATH = cfg.file_path
for img_file in os.listdir(IMAGE_PATH):
    try:
        f_path = IMAGE_PATH+img_file
        img = np.array(cv2.imread(f_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = pred_alg(img, model)
        width = int(mask.shape[1]/5) # keep original width
        height = int(mask.shape[0]/5)
        dim = (width, height)
        mask_compression = cv2.resize(mask, dim)
        pil_img = PIL.Image.open(f_path)
        exif = pil_img._getexif()
        width = int(pil_img.size[1]/5) # 縮小src以利sift檢測
        height = int(pil_img.size[0]/5)
        dim = (height, width)
        pil_img = pil_img.resize(dim)
        dst, H_img2otho, scale_rate, gps_point_x, gps_point_y = img2otho(pil_img, exif, otho_img)
        scale_rate/=5
        road_mask = get_road_mask(mask_compression, clip_mask_compression, H_img2otho, gps_point_x, gps_point_y)
        id_mask = get_road_mask(mask_compression, road_id_mask, H_img2otho, gps_point_x, gps_point_y)
        detected_area = get_road_mask(mask_compression, detected_area_map, H_img2otho, gps_point_x, gps_point_y)
        width = int(id_mask.shape[1]*5) # keep original width
        height = int(id_mask.shape[0]*5)
        dim = (width, height)
        id_mask[detected_area==1]=0
        detected_area[id_mask!=0]=1
        id_mask_ori = cv2.resize(id_mask, dim, interpolation=cv2.INTER_NEAREST)
        road_mask_ori = cv2.resize(road_mask, dim, interpolation=cv2.INTER_NEAREST)
        mask_expan = mask.copy()
        mask_expan[road_mask_ori[:,:,1]==0]=0
        mask_expan[mask_expan==2]=0
        w_problem, line_image = pred_expan(H_img2otho, mask_expan)
        if w_problem:
           print(img_file) 
        cor_H, cor_w, cor_h = correct_H(H_img2otho, detected_area.shape[1], detected_area.shape[0])
        dst_cor = cv2.warpPerspective(detected_area, cor_H, (cor_w, cor_h))
        dist = H_img2otho.dot(np.array([0,0,1])) - cor_H.dot(np.array([0,0,1]))
        dist = dist[:2]
        width = int(dst_cor.shape[1]/10) # keep original width
        height = int(dst_cor.shape[0]/10)
        dim = (width, height)
        dst_cor_com = cv2.resize(dst_cor, dim)
        gps_point_x_new = (gps_point_x+dist[1])/10
        gps_point_y_new = (gps_point_y+dist[0])/10
        dst_cor_com_ext = np.zeros(detected_area_map.shape)
        dst_cor_com_ext[int(gps_point_x_new-50):int(gps_point_x_new-50+dst_cor_com.shape[0]), int(gps_point_y_new-50):int(gps_point_y_new-50+dst_cor_com.shape[1])] = dst_cor_com
        detected_area_map += dst_cor_com_ext
        detected_area_map[detected_area_map>0] = 1
        for i in np.unique(id_mask_ori)[1:]:
            crack_mask = np.zeros(mask.shape)
            crack_mask[mask==2]=1
            crack_mask[id_mask_ori!=i]=0
            scale = 1/otho_pix_per_meter*scale_rate*100   #像素單位轉公制單位 #1m 約為正射圖的 40pixel  1cm = 1/scale pix
            pix_number = (id_mask_ori==i).sum()
            road_area = pix_number/(1/scale**2*10000)   #pix轉m^2
            pci = estimate_pci(crack_mask, scale, road_area, 5)
            road_id_dict[i].append([pci, road_area])    
            pci_arr = np.array(road_id_dict[i])
            try:
                ratio = pci_arr.T[1]/total_area_dict[i]
                covered = ratio.sum()
                score = (pci_arr.T[0]*ratio).sum()/covered
                if covered>1:
                    covered = 1
                print(score)
                print(covered)
            except:
                p=0
            score_dict[i] = [score, covered]
        with open(cfg.pci_path, 'wb') as fp:
            pickle.dump(score_dict, fp)
    except:
        print(img_file)
    cv2.imwrite(cfg.detected_area_path, detected_area_map)

