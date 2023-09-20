# Ultralytics YOLO 🚀, GPL-3.0 license
import os

import numpy as np

from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

import torch
import cv2

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


def plot_and_transform(im, box, label='', color=(0, 0, 0), txt_color=(255, 255, 255), index=None, scaled_xylw=None, keypoints=None,
                       cls=None, conf=None, use_xylw=True, truth_flag=None):
    # Add one xyxy box to image with label

    ############### zzz plot parameters ###############
    zzz_lw = 1
    tf = 1 # font thickness
    mm2px = 530 / 0.34
    # x_mm_center = scaled_xylw[1] * 0.3
    # y_mm_center = scaled_xylw[0] * 0.4 - 0.2
    # x_px_center = x_mm_center * mm2px + 6
    # y_px_center = y_mm_center * mm2px + 320
    x_px_center = scaled_xylw[1] * 480
    y_px_center = scaled_xylw[0] * 640

    # this is the knolling sequence, not opencv!!!!
    keypoints_x = ((keypoints[:, 1] * 480 - 6) / mm2px).reshape(-1, 1)
    keypoints_y = ((keypoints[:, 0] * 640 - 320) / mm2px).reshape(-1, 1)
    keypoints_mm = np.concatenate((keypoints_x, keypoints_y), axis=1)
    keypoints_center = np.average(keypoints_mm, axis=0)

    length = max(np.linalg.norm(keypoints_mm[0] - keypoints_mm[-1]),
                 np.linalg.norm(keypoints_mm[1] - keypoints_mm[2]))
    width = min(np.linalg.norm(keypoints_mm[0] - keypoints_mm[-1]),
                np.linalg.norm(keypoints_mm[1] - keypoints_mm[2]))
    c1 = np.array([length / (2), width / (2)])
    c2 = np.array([length / (2), -width / (2)])
    c3 = np.array([-length / (2), width / (2)])
    c4 = np.array([-length / (2), -width / (2)])
    if use_xylw == True:
        # length = scaled_xylw[2] / 3
        # width = scaled_xylw[3] / 3
        # c1 = np.array([length / (2), width / (2)])
        # c2 = np.array([length / (2), -width / (2)])
        # c3 = np.array([-length / (2), width / (2)])
        # c4 = np.array([-length / (2), -width / (2)])
        box_center = np.array([scaled_xylw[0], scaled_xylw[1]])
    else:
        box_center = keypoints_center

    all_distance = np.linalg.norm((keypoints_mm - keypoints_center), axis=1)
    k = 2
    max_index = all_distance.argsort()[-k:]
    lucky_keypoint_index = np.argmax([keypoints_mm[max_index[0], 1], keypoints_mm[max_index[1], 1]])
    lucky_keypoint = keypoints_mm[max_index[lucky_keypoint_index]]
    # print('the ori keypoint is ', keypoints_mm[max_index[lucky_keypoint_index]])
    my_ori = np.arctan2(lucky_keypoint[1] - keypoints_center[1], lucky_keypoint[0] - keypoints_center[0])
    # In order to grasp, this ori is based on the longest side of the box, not the label ori!

    if length < width:
        if my_ori > np.pi / 2:
            my_ori_plot = my_ori - np.pi / 2
        else:
            my_ori_plot = my_ori + np.pi / 2
    else:
        my_ori_plot = my_ori

    rot_z = [[np.cos(my_ori_plot), -np.sin(my_ori_plot)],
             [np.sin(my_ori_plot), np.cos(my_ori_plot)]]
    corn1 = (np.dot(rot_z, c1)) * mm2px
    corn2 = (np.dot(rot_z, c2)) * mm2px
    corn3 = (np.dot(rot_z, c3)) * mm2px
    corn4 = (np.dot(rot_z, c4)) * mm2px

    corn1 = [corn1[0] + x_px_center, corn1[1] + y_px_center]
    corn2 = [corn2[0] + x_px_center, corn2[1] + y_px_center]
    corn3 = [corn3[0] + x_px_center, corn3[1] + y_px_center]
    corn4 = [corn4[0] + x_px_center, corn4[1] + y_px_center]
    ############### zzz plot parameters ###############


    ############### zzz plot the box ###############
    if isinstance(box, torch.Tensor):
        box = box.cpu().detach().numpy()
    # print(box)
    p1 = np.array([int(box[0] * 640), int(box[1] * 480)])
    # print('this is p1 and p2', p1, p2)

    # cv2.rectangle(self.im, p1, p2, color, thickness=zzz_lw, lineType=cv2.LINE_AA)
    im = cv2.line(im, (int(corn1[1]), int(corn1[0])), (int(corn2[1]), int(corn2[0])), color, 1)
    im = cv2.line(im, (int(corn2[1]), int(corn2[0])), (int(corn4[1]), int(corn4[0])), color, 1)
    im = cv2.line(im, (int(corn4[1]), int(corn4[0])), (int(corn3[1]), int(corn3[0])), color, 1)
    im = cv2.line(im, (int(corn3[1]), int(corn3[0])), (int(corn1[1]), int(corn1[0])), color, 1)
    plot_x = np.copy((scaled_xylw[1] * 480 - 6) / mm2px)
    plot_y = np.copy((scaled_xylw[0] * 640 - 320) / mm2px)
    plot_l = np.copy(length)
    plot_w = np.copy(width)
    label1 = 'cls: %d, conf: %.4f' % (cls, conf)
    label2 = 'index: %d, x: %.4f, y: %.4f' % (index, plot_x, plot_y)
    label3 = 'l: %.4f, w: %.4f, ori: %.4f' % (plot_l, plot_w, my_ori)
    if label:
        w, h = cv2.getTextSize(label, 0, fontScale=zzz_lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        # cv2.rectangle(self.im, p1, p2, color, 0, cv2.LINE_AA)  # filled
        if truth_flag == True:
            txt_color = (0, 0, 255)
            # im = cv2.putText(im, label1, (p1[0] - 50, p1[1] - 32 if outside else p1[1] + h + 2),
            #                  0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
            # im = cv2.putText(im, label2, (p1[0] - 50, p1[1] - 22 if outside else p1[1] + h + 12),
            #                  0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
        else:
            im = cv2.putText(im, label1, (p1[0] - 50, p1[1] + 22 if outside else p1[1] + h + 2),
                             0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
            im = cv2.putText(im, label2, (p1[0] - 50, p1[1] + 32 if outside else p1[1] + h + 12),
                             0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
            # m = cv2.putText(im, label3, (p1[0] - 50, p1[1] + 42 if outside else p1[1] + h + 22),
            #                 0, zzz_lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
        # im = cv2.putText(im, label1, (c1[0] - 70, c1[1] - 35), 0, tl / 3, color, thickness=tf, lineType=cv2.LINE_AA)
    ############### zzz plot the box ###############

    ############### zzz plot the keypoints ###############
    shape = (640, 640)
    radius = 1
    for i, k in enumerate(keypoints):
        if truth_flag == False:
            if i == 0:
                color_k = (255, 0, 0)
            else:
                color_k = (0, 0, 0)
        elif truth_flag == True:
            if i == 0:
                color_k = (0, 0, 255)
            elif i == 3:
                color_k = (255, 255, 0)
        x_coord, y_coord = k[0] * 640, k[1] * 480
        # if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
        #     if len(k) == 3:
        #         conf = k[2]
        #         if conf < 0.5:
        #             continue
        im = cv2.circle(im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)
    ############### zzz plot the keypoints ###############

    result = np.concatenate((box_center, [round(length, 3)], [round(width, 3)], [my_ori]))

    return im, result

def change_sequence(pos_before):

    origin_point = np.array([0, -0.2])
    delete_index = np.where(pos_before == 0)[0]
    distance = np.linalg.norm(pos_before[:, :2] - origin_point, axis=1)
    order = np.argsort(distance)
    return order

def custom_predict(cfg=DEFAULT_CFG, use_python=False, data_path=None, model_path=None, test_name=None, val_start_num=None): # this is for pile grasp
    model = model_path
    os.makedirs(data_path + 'pred_2/', exist_ok=True)

    source_pth = data_path + 'images/val/'
    output_pth = data_path + 'pred_2/'
    args = dict(model=model, source=source_pth, conf=0.6, iou=0.8)
    use_python = True
    if use_python:
        from ultralytics import YOLO
        images = YOLO(model)(**args)
    else:
        predictor = PosePredictor(overrides=args)
        predictor.predict_cli()

    use_xylw = False  # use lw or keypoints to export length and width

    for i in range(50):
        origin_img = cv2.imread(source_pth + '%012d.png' % i)
        one_img = images[i]

        pred_result = []
        pred_xylws = one_img.boxes.xywhn.cpu().detach().numpy()
        if len(pred_xylws) == 0:
            return [], []
        else:
            pred_cls = one_img.boxes.cls.cpu().detach().numpy()
            pred_conf = one_img.boxes.conf.cpu().detach().numpy()
            pred_keypoints = one_img.keypoints.cpu().detach().numpy()
            pred_keypoints[:, :, :2] = pred_keypoints[:, :, :2] / np.array([640, 480])
            pred_keypoints = pred_keypoints.reshape(len(pred_xylws), -1)
            pred = np.concatenate((np.zeros((len(pred_xylws), 1)), pred_xylws, pred_keypoints), axis=1)

        ######## order based on distance to draw it on the image while deploying the model ########
        mm2px = 530 / 0.34
        x_px_center = pred_xylws[:, 1] * 480
        y_px_center = pred_xylws[:, 0] * 640
        mm_center = np.concatenate(
            (((x_px_center - 6) / mm2px).reshape(-1, 1), ((y_px_center - 320) / mm2px).reshape(-1, 1)), axis=1)
        pred_order = change_sequence(mm_center)

        pred = pred[pred_order]
        pred_xylws = pred_xylws[pred_order]
        pred_keypoints = pred_keypoints[pred_order]
        pred_cls = pred_cls[pred_order]
        pred_conf = pred_conf[pred_order]
        print('this is the pred order', pred_order)

        for j in range(len(pred_xylws)):
            pred_keypoint = pred_keypoints[j].reshape(-1, 3)
            pred_xylw = pred_xylws[j]

            # print('this is pred xylw', pred_xylw)
            origin_img, result = plot_and_transform(im=origin_img, box=pred_xylw, label='0:, predic', color=(0, 0, 0),
                                                         txt_color=(255, 255, 255),
                                                         index=j, scaled_xylw=pred_xylw, keypoints=pred_keypoint,
                                                         cls=pred_cls[j], conf=pred_conf[j],
                                                         use_xylw=use_xylw, truth_flag=False)
            pred_result.append(result)

        # cv2.namedWindow('zzz', 0)
        # cv2.resizeWindow('zzz', 1280, 960)
        # cv2.imshow('zzz', origin_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img_path_output = output_pth + '%012d.png' % i
        cv2.imwrite(img_path_output, origin_img)

def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO object detection on an image or video source."""
    model = '../../../../../YOLOv8/runs/pose/830_pile_real_box/weights/best.pt'
    # source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
    #     else 'https://ultralytics.com/images/bus.jpg'
    source = '/home/zhizhuo/Creative_Machines_Lab/knolling_dataset/yolo_pile_830_real_box/images/val/'

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = PosePredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':

    # val_start_num = 0
    # test_name = 'pile_model_pile_dataset/'
    # data_path = '../../../../../knolling_dataset/yolo_segmentation_820_real_box/'
    # model_path = '../../../../../Knolling_bot_2/828_pile_pose_real_sundry/weights/best.pt'
    # custom_predict(data_path=data_path, model_path=model_path, test_name=test_name, val_start_num=val_start_num)

    predict()