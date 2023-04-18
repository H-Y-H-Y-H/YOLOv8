# Ultralytics YOLO 🚀, GPL-3.0 license
import numpy as np

from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor


class PosePredictor(DetectionPredictor):

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes,
                                        nc=len(self.model.names))

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, shape)
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(
                Results(orig_img=orig_img,
                        path=img_path,
                        names=self.model.names,
                        boxes=pred[:, :6],
                        keypoints=pred_kpts))
        return results

import torch
def predict(cfg=DEFAULT_CFG, use_python=False):
    # model = cfg.model or 'yolov8n-pose.pt'
    # model = "/home/ubuntu/Desktop/YOLOv8/runs/pose/train4/weights/last.pt"
    model = "/home/zhizhuo/ADDdisk/Create Machine Lab/YOLOv8/runs/pose/train4/weights/last.pt"
    # source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
    #     else 'https://ultralytics.com/images/bus.jpg'
    # source_pth = '/home/ubuntu/Desktop/datasets/knolling_data/images/val'
    source_pth = '/home/zhizhuo/ADDdisk/Create Machine Lab/YOLOv8/datasets/knolling_data/images/train'
    args = dict(model=model, source=source_pth, save=True, save_txt=True)
    use_python = True
    if use_python:
        from ultralytics import YOLO
        img = YOLO(model)(**args)
    else:
        predictor = PosePredictor(overrides=args)
        predictor.predict_cli()
    device = 'cuda'


    for i in range(len(img)):
        keypoints = img[i].keypoints
        cos_sin = keypoints[:, :, :2] / torch.tensor((640, 480)).to(device)
        cos_sin = cos_sin.cpu().detach().numpy().reshape(len(cos_sin), 2)
        box = img[i].boxes
        xylw = box.xywhn.cpu().detach().numpy()
        pred = np.concatenate((np.zeros((len(cos_sin), 1)), xylw, cos_sin), axis=1)
        pred_order = np.lexsort((pred[:, 2], pred[:, 1]))
        pred = pred[pred_order]

        target = np.loadtxt('/home/zhizhuo/ADDdisk/Create Machine Lab/YOLOv8/datasets/knolling_data/labels/train/%012d.txt' % i)
        target_order = np.lexsort((target[:, 2], target[:, 1]))
        target = target[target_order]

        loss_mean = np.mean((target - pred) ** 2)
        loss_std = np.mean((target - pred), dtype=np.float64)
        print('this is pred\n', pred)
        print('this is target\n',target)
        print('this is mean error', loss_mean)
        print('this is std error', loss_std)

    print('this is key point')


if __name__ == '__main__':
    predict()
    # from ultralytics import YOLO
    #
    # model_path = "/home/ubuntu/Desktop/YOLOv8/runs/pose/train4/weights/"
    #
    # # Load a model
    # model = YOLO('yolov8n-knolling.yaml')  # build from YAML and transfer weights
    # model.load(model_path+'last.pt')
    #
    # source_pth = '/home/ubuntu/Desktop/datasets/knolling_data/images/val'
    #
    #
    # result = model(source=source_pth,conf = 0.5,save=True)
    # print(result)
