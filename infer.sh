#!/bin/bash

cd ./code
image_path='../dataset/road/20230530/100FTASK/*.JPG'
ph_save_path='../output/prediction_c_p/'
crack_save_path='../output/prediction_c/'
python3 prediction.py --image_path $image_path --save_path $ph_save_path --ckpt '../ckpt/ph_ckpt.pth'\
        --classes 'pothole' 'expand' 'crack'
python3 prediction.py --image_path $image_path --save_path $crack_save_path --ckpt '../ckpt/crack_ckpt.pth'\
        --classes 'expand' 'crack'
python3 PCI.py --otho_image_path '../dataset/clip/clip.jpg' --clip_mask_path '../dataset/clip/clip_mask.png' \
        --road_id_mask_compression '../dataset/clip/road_id_mask_compression.png' --image_path $image_path\
        --mask_ph_path $ph_save_path --mask_crack_path $crack_save_path --save_path '../output/pci_map/'\
        --output_type 'total' --otho_pix_per_meter 39.89206980085115