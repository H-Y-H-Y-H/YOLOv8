import os
import sys

if __name__ == '__main__':
    # train()
    CUDA_LAUNCH_BLOCKING = "1"
    from ultralytics import YOLO

    # Load a model
    model = YOLO('yolov8-knolling.yaml')  # build from YAML and transfer weights
    model = model.load('/home/zhizhuo/ADDdisk/Create Machine Lab/YOLOv8/runs/pose/train_standard_518_3/weights/last.pt')
    # model = model.load('yolov8n-pose.pt')
    # Train the model
    model.train(data='knolling_grasp.yaml', epochs=150, imgsz=640, patience=300, name='train_pile_grasp_624')

