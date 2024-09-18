from torch.utils.data import Dataset
import cv2, torch
from PIL import Image
import numpy as np
from torchvision import transforms as T
from utils import rgb2idx

class C_P_Dataset(Dataset):
    
    def __init__(self, X, transform, classes = ['pothole', 'expand', 'crack']):
        self.X = X
        self.transform = transform
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.classes = classes
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.X[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.X[idx].replace('jpg', 'png'))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        label = rgb2idx(label, classes=self.classes)
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