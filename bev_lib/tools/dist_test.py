import argparse
import copy
import json
import os
import sys
import cv2


import numpy as np
import torch
import yaml
from det3d import torchie
from det3d.datasets import build_dataset
from det3d.torchie import Config

from det3d.torchie.parallel import collate_kitti
from torch.utils.data import DataLoader

import pickle 
import time 
from tools.demo_utils import Box,_second_det_to_nusc_box
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def compute_IOU_each_laryer(pred,gt,thrs = (0.3, 0.4, 0.5, 0.5, 0.7, 0.6, 0.4, 0.3, 0.9)):
    # shape of pred,gt: (layers, w, h) 
    # return I and U , shape (layers,2)
    IOUs = []
    for i in range(len(pred)):
        cur_pred = pred[i]
        cur_gt = gt[i].astype(bool)
        bin_pred = cur_pred > thrs[i]
        iou = np.array((np.sum(bin_pred & cur_gt),np.sum(bin_pred | cur_gt)))
        IOUs.append(iou)
    # print(IOUs)
    return IOUs

def vis_layer(pred,fig_name,thrs = [0.3, 0.4, 0.5, 0.5, 0.7, 0.6, 0.4, 0.3, 0.9]):
    # shape of pred,gt: layers, w, h 
    color = np.array([(0, 0, 230),(255, 158, 0),(31, 120, 180),(51, 160, 44),(166, 206, 227),(251, 154, 153),(227, 26, 28),(253, 191, 111),(255, 127, 0)])
    # color[:,0],color[:,2] = color[:,2],color[:,0]
    
    # bg_color = np.array((255,255,255))
    array_to_print = np.zeros((np.shape(pred)[1],np.shape(pred)[2],3)) + 255
    # full_mask = np.zeros((np.shape(pred)[1],np.shape(pred)[2])).astype(bool)
    
    # thrs = [0.3, 0.4, 0.5, 0.5, 0.7, 0.6, 0.4, 0.3, 0.9]#[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
    name = ['ped','car','road_segment', 'lane','drivable_area','ped_crossing','walkway','stop_line','carpark_area']
    order = [4,2,3,5,6,7,8,0,1]#,3,5,6,7,8]
    for layer_idx in order:
        ## init varibles
        # print(np.mean(full_mask))
        cur_pred = pred[layer_idx]
        cur_color = color[layer_idx]
        cur_thr = thrs[layer_idx]
        bin_pred = cur_pred > cur_thr
        # clear previous layers if necessary
        
        # clear_mask = full_mask & bin_pred
        
        # next_full_mask = full_mask | bin_pred

        clear_mask = (bin_pred == False).astype(int)
        
        cleared_array = array_to_print * np.array([clear_mask,clear_mask,clear_mask]).transpose(1,2,0)
        # add in newthings
        bin_pred = np.array([bin_pred,bin_pred,bin_pred]).transpose(1,2,0)
        array_to_print = cleared_array + bin_pred * cur_color
        # print(np.mean(bin_pred.astype(int)))
        # full_mask = next_full_mask
    # print(array_to_print)
    plt.figure()
    # plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.,)
    array_to_print[512//2-10:512//2+10,512//2-5:512//2+5,:] = np.array([50,50,100])
    
    t = 1 ## alpha value
    cmap = {}
    for i in range(len(color)):
        curcolor = color[i]/255.
        cmap[i] = [curcolor[0],curcolor[1],curcolor[2],1]
    
    labels = {}
    for i in range(len(name)):
        labels[i] = name[i]
    
    patches =[mpatches.Patch(color=cmap[i],label=labels[i]) for i in cmap]
    plt.imshow(array_to_print.astype(int))
    # get rid of axis etc
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    # plt.legend(handles=patches, loc=2, borderaxespad=0., bbox_to_anchor=(1.05,1.0))
    plt.savefig(fig_name)



def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)

def convert_box(info):
    boxes =  info["gt_boxes"].astype(np.float32)
    names = info["gt_names"]

    assert len(boxes) == len(names)

    detection = {}

    detection['box3d_lidar'] = boxes

    # dummy value 
    # 2 is not used value
    detection['label_preds'] = np.zeros(len(boxes)) + 2
    for i in range(len(names)):
        if names[i] == 'car':
            detection['label_preds'][i] = 0
        elif names[i] == 'pedestrian':
            detection['label_preds'][i] = 8

    detection['scores'] = np.ones(len(boxes))

    return detection 

def to_map(detects,maps):
    ## given detections and map,
    ## merge detection results into map to form binary results for every class
    de_map = np.zeros((2,512,512))
    boxes = _second_det_to_nusc_box(detects)
    # print('detect:',detects)
    # print('box:',boxes)
    # print('!!!!!!!!debug!!!!!!!!!',len(boxes))
    # print(boxes[0])
    for box in boxes:
        # print('before process:',box.poly())
        poly = (box.bottom_corners()[:2,:].T + 51.2) * 5
        poly = np.round(poly).astype(np.int32)
        # print('to map poly:',poly)
        # poly[:, [1, 0]] = poly[:, [0, 1]]

        if box.label == 0 and box.score > 0.2:  
            de_map[0] = cv2.fillPoly(de_map[0],[poly],1)
        elif box.label == 8 and box.score> 0.3:
            de_map[1] = cv2.fillPoly(de_map[1],[poly],1)
        
    # de_map = de_map[:,:,:].copy()
    return np.concatenate((de_map,maps),axis=0)




def main():

    cfg = Config.fromfile("bev_data.py")  

    dataset = build_dataset(cfg.data.val)
    print('ini done')
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=None,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_kitti,
        pin_memory=False,
    )
   
    print('loader_done')

    for i, data_batch in enumerate(data_loader):
        print(i)
        temp = data_batch['bin_map']
        data_batch['bin_map'] = torch.tensor(temp[:,:,:,::-1].copy())
        del temp

        print(data_batch.keys())
        info = dataset._nusc_infos[i]
        # print(info)
        cur_annos = convert_box(info)
        

        gt_map = to_map(cur_annos,data_batch['bin_map'][0].cpu().numpy())
        # pred_map = to_map(outputs[0],seg_outputs[0].cpu().numpy())
        # cal_iou(gt_map,pred_map)
        print(np.shape(gt_map))

        # if i < 200:
        #     vis_layer(gt_map,'./vis/gt_'+str(i))


if __name__ == "__main__":
    main()
