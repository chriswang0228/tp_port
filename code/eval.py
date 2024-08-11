import numpy as np
import torch
from torchvision import transforms as T
import cv2, glob, json, configs
import segmentation_models_pytorch_v2 as smp
from utils import rgb2idx

class Eval:
    def __init__(self, image_path, save_path, ckpt, classes = ['pothole', 'expand', 'crack'], device = 'cuda'):
        self.image_path = image_path
        self.save_path = save_path
        self.ckpt = ckpt
        self.classes = classes
        self.n_classes = len(classes)+1
        self.device = device
        self.transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
      
    def predict(self, img):
        with torch.no_grad():
            image = self.transform(img)
            image=image.to(self.device)
            image = image.unsqueeze(0)     
            output = self.model(image)
            result=torch.argmax(output, dim=1).squeeze(0).squeeze(0).cpu().numpy()
        return result
                
    
    def run(self):
        self.model = smp.create_model(arch='UnetPlusPlus',encoder_name='efficientnet-b6', encoder_weights='imagenet',
                                              classes=self.n_classes, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16]).to(self.device)
        self.model.load_state_dict(torch.load(self.ckpt))
        self.model.eval()
        y = []
        y_pred = []
        for img_path in glob.glob(self.image_path):
            mask_path = img_path.replace('jpg', 'png')
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, [480, 480])
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = rgb2idx(mask, classes=self.classes)
            mask = cv2.resize(mask, [480, 480], cv2.INTER_NEAREST)
            pred = self.predict(img)
            y.append(mask)
            y_pred.append(pred)
        y = np.array(y).reshape(-1)
        y_pred = np.array(y_pred).reshape(-1)
        iou_per_class_dict = {}
        for i in range(len(self.classes)): #loop per pixel class
            true_class = y_pred == i+1
            true_label = y == i+1
            intersect = np.logical_and(true_class, true_label).sum()
            union = np.logical_or(true_class, true_label).sum()
            iou = intersect/union
            iou_per_class_dict[self.classes[i]]=iou
        with open(self.save_path, 'w') as f:
            json.dump(iou_per_class_dict, f)

        
def main():
    cfg = configs.pothole.eval
    eval = Eval(
        image_path = cfg.image_pathimage_path,
        save_path = cfg.save_path,
        ckpt = cfg.ckpt,
        classes = cfg.classes
        )
    eval.run()

    
if __name__ == '__main__':
    main()