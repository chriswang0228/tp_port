from configs.crack import prediction as c_pred
from configs.pothole import prediction as p_pred
from types import SimpleNamespace

path = '../output/pci_score/20230530/'
pci = SimpleNamespace(
    image_path = c_pred.image_path.replace('*.JPG', ''),
    mask_ph_path = p_pred.save_path,
    mask_crack_path = c_pred.save_path,
    detected_area_path = '../output/detected_area_20230530.png',
    crack_pci_path = path+'crack_pci_data_20230530.pkl',
    ph_pci_path = path+'ph_pci_data_20230530.pkl',
    total_pci_path = path+'total_pci_data_20230530.pkl',
    map_path = path.replace('score', 'map')
    )