import numpy as np 
import torch, argparse
import PIL
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import cv2
import random
from utils import *
import torch
import math
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--otho_image_path', type=str, default='../dataset/clip/clip.jpg', help='orthophotograph path')
parser.add_argument('--clip_mask_path', type=str, default='../dataset/clip/clip_mask.png', help='orthophotograph mask path')
parser.add_argument('--road_id_mask_compression', type=str, default='../dataset/clip/road_id_mask_compression.png', help='road id mask path')
parser.add_argument('--image_path', type=str, default='../dataset/road/20230530/100FTASK/', help='prediction image file path')
parser.add_argument('--mask_ph_path', type=str, default='../output/prediction_c_p/', help='pothole mask file path')
parser.add_argument('--mask_crack_path', type=str, default='../output/prediction_c/', help='crack mask file path')
parser.add_argument('--save_path', type=str, default='../output/pci_map/', help='pci map save path')
parser.add_argument('--output_type', type=list, default=['total'], help='output type')
parser.add_argument('--otho_pix_per_meter', type=float, default=39.89206980085115, help='pixel number per meter in orthophotograph')

args = parser.parse_args()

otho_img = cv2.imread(args.otho_image_path)
clip_mask = cv2.imread(args.clip_mask_path)
road_id_mask = cv2.imread(args.road_id_mask_compression, 0)
detected_area_map = np.zeros(road_id_mask.shape)
crack_road_id_dict = {}
ph_road_id_dict = {}
area_road_id_dict = {}
for i in np.unique(road_id_mask)[1:]:
    crack_road_id_dict[i] = list()
    ph_road_id_dict[i] = list()
    area_road_id_dict[i] = list()

width = int(clip_mask.shape[1]/10) # keep original width
height = int(clip_mask.shape[0]/10)
dim = (width, height)
clip_mask_compression = cv2.resize(clip_mask, dim) #壓縮十倍減少運算時間
#otho_pix_per_meter = 1/(math.dist([2784295.23560,289479.57382], [2784417.268382,289637.447092])/math.dist([5442, 0], [574, 6298])) #每公尺的正射圖像素數
otho_pix_per_meter = args.otho_pix_per_meter
total_area_dict = {}
for i in np.unique(road_id_mask)[1:]:
    a = (road_id_mask==i).sum()/(otho_pix_per_meter/10)**2   #m^2
    total_area_dict[i] = a 

image_path = args.image_path
mask_ph_path = args.mask_ph_path
mask_crack_path = args.mask_crack_path
for img_file in sorted(os.listdir(image_path)):
    try:
        f_path = image_path+img_file
        mask_file = img_file.replace('JPG', 'png')
        c_path = mask_crack_path+mask_file
        p_path = mask_ph_path+mask_file
        img = np.array(cv2.imread(f_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        c_mask = cv2.imread(c_path)
        c_mask = cv2.cvtColor(c_mask, cv2.COLOR_BGR2RGB)
        c_mask = rgb2idx(c_mask, classes=['expand', 'crack'])
        p_mask = cv2.imread(p_path)
        p_mask = cv2.cvtColor(p_mask, cv2.COLOR_BGR2RGB)
        p_mask = rgb2idx(p_mask, classes=['pothole', 'expand', 'crack'])
        width = int(c_mask.shape[1]/5) # keep original width
        height = int(c_mask.shape[0]/5)
        dim = (width, height)
        mask_compression = cv2.resize(c_mask, dim)
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
        mask_expan = c_mask.copy()
        mask_expan[road_mask_ori[:,:,1]==0]=0
        mask_expan[mask_expan==2]=0
        #w_problem, line_image = pred_expan(H_img2otho, mask_expan)
        #if w_problem:
        #   print(img_file) 
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
            crack_mask = np.zeros(c_mask.shape)
            crack_mask[c_mask==2]=1
            crack_mask[id_mask_ori!=i]=0
            ph_mask = np.zeros(p_mask.shape)
            ph_mask[p_mask==1]=1
            ph_mask[id_mask_ori!=i]=0
            scale = 1/otho_pix_per_meter*scale_rate*100   #像素單位轉公制單位 #1m 約為正射圖的 40pixel  1cm = 1/scale pix
            pix_number = (id_mask_ori==i).sum()
            road_area = pix_number/(1/scale**2*10000)   #pix轉m^2
            c_hml = crack_hml(crack_mask, scale, 5)
            p_hml = ph_hml(ph_mask, scale)
            crack_road_id_dict[i].append(c_hml)
            ph_road_id_dict[i].append(p_hml)
            area_road_id_dict[i].append(road_area)
    except:
        print(img_file)
    #cv2.imwrite(pci.detected_area_path, detected_area_map)


total_score_dict = {}
crack_score_dict = {}
ph_score_dict = {}
for i in area_road_id_dict.keys():
    if len(area_road_id_dict[i])>0:
        c_hml = np.array(crack_road_id_dict[i])
        c_hml = np.sum(c_hml, axis = 0)
        p_hml = np.array(ph_road_id_dict[i])
        p_hml = np.sum(p_hml, axis = 0)
        road_area = np.sum(area_road_id_dict[i])
        c_todo_list = crack_dens(c_hml, road_area)
        p_todo_list = ph_dens(p_hml, road_area)
        crack_score_dict[i] = 100 - estimate_cdv(c_todo_list)
        ph_score_dict[i] = 100 - estimate_cdv(p_todo_list)
        total_score_dict[i] = 100 - estimate_cdv(c_todo_list+p_todo_list)

if os.path.isdir(args.save_path)==False:
    os.mkdir(args.save_path)

path = args.save_path
if 'crack' in args.output_type:
    score2map(crack_score_dict, road_id_mask, path+'crack.png', color_bar = False)
if 'ph' in args.output_type:
    score2map(ph_score_dict, road_id_mask, path+'ph.png', color_bar = False)
if 'total' in args.output_type:
    score2map(total_score_dict, road_id_mask, path+'total.png', color_bar = False)
mask = cv2.imread(path+'total.png')
mask = cv2.resize(mask, list(reversed(otho_img.shape[:2])), cv2.INTER_NEAREST)
cv2.imwrite(path+'scale_mask.png', mask)
overlapped_image = (1 - 0.5) * otho_img + 0.5 * mask
overlapped_image = np.clip(overlapped_image, 0, 255).astype(np.uint8)
cv2.imwrite(path+'overlap_mask.png', overlapped_image)
#with open(pci.crack_pci_path, 'wb') as fp:
#    pickle.dump(crack_score_dict, fp)
#with open(pci.ph_pci_path, 'wb') as fp:
#    pickle.dump(ph_score_dict, fp)
#with open(pci.total_pci_path, 'wb') as fp:
#    pickle.dump(total_score_dict, fp)
