

if __name__ == '__main__':
    # train()

    from ultralytics import YOLO

    # Load a model
    model = YOLO('yolov8n-knolling.yaml')  # build from YAML and transfer weights
    model = model.load('yolov8n-pose.pt')
    # Train the model
    model.train(data='knolling.yaml', epochs=100, imgsz=640, patience=300, name='train_standard_1000')

