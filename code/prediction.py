import torch
from torchvision import transforms as T
import cv2, os, glob, argparse
import segmentation_models_pytorch_v2 as smp
from utils import pred_alg, visualize

class Predictor:
    def __init__(self, image_path, save_path, ckpt, classes = ['pothole', 'expand', 'crack']):
        self.image_path = image_path
        self.save_path = save_path
        self.ckpt = ckpt
        self.classes = classes
        self.n_classes = len(classes)+1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

                
    
    def run(self):
        self.model = smp.create_model(arch='UnetPlusPlus',encoder_name='efficientnet-b6', encoder_weights='imagenet',
                                              classes=self.n_classes, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16]).to(self.device)
        self.model.load_state_dict(torch.load(self.ckpt))
        self.model.eval()
        if os.path.isdir(self.save_path)==False:
            os.mkdir(self.save_path)
        for img_path in glob.glob(self.image_path):
            img_name = img_path.split('/')[-1].replace('JPG', 'png')
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = pred_alg(img, self.model)
            pred = visualize(pred, classes=self.classes)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
            
            cv2.imwrite(os.path.join(self.save_path, img_name), pred)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, default='../dataset/road/20230530/100FTASK/*.JPG', help='prediction image file path')
    parser.add_argument('save_path', type=str, default='../output/prediction_c_p/', help='mask save path')
    parser.add_argument('ckpt', type=str, default='../ckpt/ph_ckpt.pth', help='model checkpoint')
    parser.add_argument('classes', type=list, default=['pothole', 'expand', 'crack'], help='classes')
    args = parser.parse_args()
    predictor = Predictor(
        image_path = args.image_path,
        save_path = args.save_path,
        ckpt = args.ckpt,
        classes = args.classes
        )
    predictor.run()

    
if __name__ == '__main__':
    main()