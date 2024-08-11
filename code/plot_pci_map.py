import pickle, cv2
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

