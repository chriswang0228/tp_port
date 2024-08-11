import torch
from torchvision import transforms as T
import cv2, os, glob, configs
import segmentation_models_pytorch_v2 as smp
from utils import pred_alg, visualize

class Predictor:
    def __init__(self, image_path, save_path, ckpt, classes = ['pothole', 'expand', 'crack'], device = 'cuda'):
        self.image_path = image_path
        self.save_path = save_path
        self.ckpt = ckpt
        self.classes = classes
        self.n_classes = len(classes)+1
        self.device = device
        self.transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

                
    
    def run(self):
        self.model = smp.create_model(arch='UnetPlusPlus',encoder_name='efficientnet-b6', encoder_weights='imagenet',
                                              classes=self.n_classes, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16]).to(self.device)
        self.model.load_state_dict(torch.load(self.ckpt))
        self.model.eval()
        for img_path in glob.glob(self.image_path):
            img_name = img_path.split('/')[-1].replace('JPG', 'png')
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = pred_alg(img, self.model)
            pred = visualize(pred, classes=self.classes)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
            
            cv2.imwrite(os.path.join(self.save_path, img_name), pred)
        
def main():
    cfg = configs.pothole.prediction
    predictor = Predictor(
        image_path = cfg.image_path,
        save_path = cfg.save_path,
        ckpt = cfg.ckpt,
        classes = cfg.classes
        )
    predictor.run()

    
if __name__ == '__main__':
    main()