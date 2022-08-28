import os
import numpy as np
import torch
import albumentations
from torch.utils.data import Dataset
# from det3d.datasets import build_dataloader, build_dataset
from mmdet3d.datasets import build_dataloader, build_dataset
# from det3d.torchie import Config
from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.utils import recursive_eval

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
import cv2
# from tools.demo_utils import Box,_second_det_to_nusc_box

# def to_map(detects,maps):
#     ## given detections and map,
#     ## merge detection results into map to form binary results for every class
#     de_map = np.zeros((2,512,512))
#     boxes = _second_det_to_nusc_box(detects)

#     for box in boxes:
#         poly = (box.bottom_corners()[:2,:].T + 51.2) * 5
#         poly = np.round(poly).astype(np.int32)

#         if box.label == 0 and box.score > 0.2:  
#             de_map[0] = cv2.fillPoly(de_map[0],[poly],1)
#         elif box.label == 8 and box.score> 0.3:
#             de_map[1] = cv2.fillPoly(de_map[1],[poly],1)
        
#     # de_map = de_map[:,:,:].copy()
#     return np.concatenate((de_map,maps),axis=0)

# def convert_box(info):
#     boxes =  info["gt_boxes"].astype(np.float32)
#     names = info["gt_names"]

#     assert len(boxes) == len(names)

#     detection = {}

#     detection['box3d_lidar'] = boxes

#     # dummy value 
#     # 2 is not used value
#     detection['label_preds'] = np.zeros(len(boxes)) + 2
#     for i in range(len(names)):
#         if names[i] == 'car':
#             detection['label_preds'][i] = 0
#         elif names[i] == 'pedestrian':
#             detection['label_preds'][i] = 8

#     detection['scores'] = np.ones(len(boxes))

#     return detection 


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        print(example)
        print("keys!!",example.keys())        
        # info = self.data._nusc_infos[i]
        # print(info)
        # cur_annos = convert_box(info)
        
        # gt_map = to_map(cur_annos,example['bin_map'].cpu().numpy())
        
        # pred_map = to_map(outputs[0],seg_outputs[0].cpu().numpy())
        # cal_iou(gt_map,pred_map)
        # print(np.shape(gt_map))

        # res_example  = {}
        # res_example['image'] = gt_map

        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()

        # cfg = Config.fromfile("bev_data.py")
        
        # print(cfg)
        config_name = '/home/xiyuez2/xiyue/bev-taming-transformers/bev_lib/configs/nuscenes/seg/vq_image.yaml'
        configs.load(config_name, recursive=True)
        cfg = Config(recursive_eval(configs), filename=config_name)
        dataset = build_dataset(cfg.data.train)
        self.data = dataset


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        # cfg = Config.fromfile("bev_data.py")
        config_name = '/home/xiyuez2/xiyue/bev-taming-transformers/bev_lib/configs/nuscenes/seg/vq_image.yaml'
        configs.load(config_name, recursive=True)
        cfg = Config(recursive_eval(configs), filename=config_name)
        dataset = build_dataset(cfg.data.test)
        self.data = dataset


        # dataset = build_dataset(cfg.data.val)
        
        

