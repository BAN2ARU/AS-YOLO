from ultralytics import YOLO
import os

data = os.path.join(os.getcwd(), 'ultralytics', 'cfg', 'datasets','AppleStem-Segmentationdataset.yaml')
cfg = os.path.join(os.getcwd(), 'ultralytics', 'cfg', 'models', 'v8', 'asyolov8l-seg.yaml')

model = YOLO(cfg, task='segment')

if __name__ == '__main__' :
    model.train(data=data, epochs=500, batch=-1, save_period=20, device=0, project='asyolov8', pretrained=False, verbose=True, dropout=0.2, plots=True, mosaic=0.5)
    metrics = model.val()