import argparse
import copy
import json
import os
import sys
import cv2
try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.parallel import collate, collate_kitti
from torch.utils.data import DataLoader

from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
import pickle 
import time 
from tools.demo_utils import Box,_second_det_to_nusc_box
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def compute_IOU_each_laryer(pred,gt,thrs = (0.3, 0.4, 0.5, 0.5, 0.7, 0.6, 0.4, 0.3, 0.9)):
    # shape of pred,gt: (layers, w, h) 
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", required=True, help="the dir to save logs and models")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument(
        "--txt_result",
        type=bool,
        default=False,
        help="whether to save results to standard KITTI format of txt type",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--speed_test", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--testset", action="store_true")

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():

    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed testing: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    if args.testset:
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test)
    else:
        print("Use Val Set")
        dataset = build_dataset(cfg.data.val)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=None,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_kitti,
        pin_memory=False,
    )
    # build_dataloader(
    #     dataset,
    #     batch_size=1,#cfg.data.samples_per_gpu if not args.speed_test else 1,
    #     workers_per_gpu=1, #cfg.data.workers_per_gpu,
    #     dist=distributed,
    #     shuffle=False,
    # )

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    # put model on gpus
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)#apex.parallel.convert_syncbn_model(model)
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            # broadcast_buffers=False,
            find_unused_parameters=True,
        )
    else:
        # model = fuse_bn_recursively(model)
        model = model.cuda()

    model.eval()
    mode = "val"

    logger.info(f"work dir: {args.work_dir}")
    if cfg.local_rank == 0:
        prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)

    detections = {}
    seg_dects = []
    # gt_annos = []
    cpu_device = torch.device("cpu")

    start = time.time()

    start = int(len(dataset) / 3)
    end = int(len(dataset) * 2 /3)

    time_start = 0 
    time_end = 0 
    IOUs = [] # batches * layers * 2
    for i, data_batch in enumerate(data_loader):
        # print(data_batch)
        temp = data_batch['bin_map']
        data_batch['bin_map'] = torch.tensor(temp[:,:,:,::-1].copy())
        del temp
        # print(data_batch)
        info = dataset._nusc_infos[i]
        # print(info)
        cur_annos = convert_box(info)
        # print("!!!!!!",i,)
        # print(cur_annos)
        # print(len(data_batch))
        # gt_annos.append((cur_annos,data_batch['bin_map'][0]))

        if i == start:
            torch.cuda.synchronize()
            time_start = time.time()

        if i == end:
            torch.cuda.synchronize()
            time_end = time.time()

        # for quick test, will have error in eval :
        # if i > 400:
        #      break

        with torch.no_grad():
            outputs,seg_outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=args.local_rank,
            )
        # cur_detections = []
        for j in range(len(outputs)):
            output = outputs[j]
            seg_out = seg_outputs[j]
            
            token = output["metadata"]["token"]
            for k, v in output.items():
                if k not in [
                    "metadata",
                ]:
                    output[k] = v.to(cpu_device)
            detections.update(
                {token: output,}
            )
            # cur_detections.append(output)
            # seg_dects.append(seg_out)
            
            if args.local_rank == 0:
                prog_bar.update()
        
        gt_map = to_map(cur_annos,data_batch['bin_map'][0].cpu().numpy())
        pred_map = to_map(outputs[0],seg_outputs[0].cpu().numpy())
        # cal_iou(gt_map,pred_map)
        print(np.shape(gt_map),np.shape(pred_map))
        iou = compute_IOU_each_laryer(pred_map,gt_map) # layers * 2
        IOUs.append(iou)
        if i < 200:
            vis_layer(gt_map,'./vis/gt_'+str(i))
            vis_layer(pred_map,'./vis/pre_'+str(i))

    IOUs = np.array(IOUs)
    print("============IOU scores==============")
    for i in range(9):
        cur_iou = np.sum(IOUs[:,i,0])/np.sum(IOUs[:,i,1])
        print(cur_iou)

    synchronize()

    all_predictions = all_gather(detections)
    
    print("\n Total time per frame: ", (time_end -  time_start) / (end - start))

    if args.local_rank != 0:
        return

    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    # save_pred(predictions, args.work_dir)
    # print("!!! for debug !!!")
    # print(predictions)
    result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=args.work_dir, testset=args.testset)

    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print(f"Evaluation {k}: {v}")

    if args.txt_result:
        assert False, "No longer support kitti"

if __name__ == "__main__":
    main()
