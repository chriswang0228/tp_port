import pickle, os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import cv2
import numpy as np
from configs.pci import pci
from utils import score2map

with open(pci.crack_pci_path, 'rb') as fp:
    crack_pci = pickle.load(fp)
with open(pci.ph_pci_path, 'rb') as fp:
    ph_pci = pickle.load(fp)
with open(pci.total_pci_path, 'rb') as fp:
    total_pci = pickle.load(fp)
id_map = cv2.imread('../dataset/clip/road_id_mask_compression.png', 0)
path = pci.map_path
score2map(crack_pci, id_map, path+'crack.png', color_bar = False)
score2map(ph_pci, id_map, path+'ph.png', color_bar = False)
score2map(total_pci, id_map, path+'total.png', color_bar = False)
score2map(crack_pci, id_map, path+'crack.svg', color_bar = True)
score2map(ph_pci, id_map, path+'ph.svg', color_bar = True)
score2map(total_pci, id_map, path+'total.svg', color_bar = True)
image = cv2.imread('../dataset/clip/clip.jpg')
mask = cv2.imread(path+'total.png')
mask = cv2.resize(mask, list(reversed(image.shape[:2])), cv2.INTER_NEAREST)
cv2.imwrite(path+'scale_mask.png', mask)
overlapped_image = (1 - 0.5) * image + 0.5 * mask
overlapped_image = np.clip(overlapped_image, 0, 255).astype(np.uint8)
cv2.imwrite(path+'overlap_mask.png', overlapped_image)
