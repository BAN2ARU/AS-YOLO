from ultralytics import YOLO
import os

data = os.path.join(os.getcwd(),'datasets','AppleStem-Segmentation dataset.yaml')
# Enter the path to the trained model (.pt file).
model = os.path.join(os.getcwd(),'weights', 'best.pt')

if __name__ == '__main__' :
    model = YOLO(model)
    model.info()
    metrics = model.val(data=data, batch=1, save_json=True, half=False, plots=True, device=0, verbose=True, split='test')