import argparse
import json
import os
import sys

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

import numpy as np
import torch
import yaml
from det3d.datasets import build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
import torch.distributed as dist
import subprocess
from det3d.datasets import DATASETS, build_dataloader
from torch.utils.data import DataLoader
from det3d.torchie.parallel import collate, collate_kitti


# def parse_args():
#     parser = argparse.ArgumentParser(description="Train a detector")
#     parser.add_argument("config", help="train config file path")
#     parser.add_argument("--work_dir", help="the dir to save logs and models")
#     parser.add_argument("--resume_from", help="the checkpoint file to resume from")
#     parser.add_argument(
#         "--validate",
#         action="store_true",
#         help="whether to evaluate the checkpoint during training",
#     )
#     parser.add_argument(
#         "--gpus",
#         type=int,
#         default=1,
#         help="number of gpus to use " "(only applicable to non-distributed training)",
#     )
#     parser.add_argument("--seed", type=int, default=None, help="random seed")
#     parser.add_argument(
#         "--launcher",
#         choices=["pytorch", "slurm"],
#         default="pytorch",
#         help="job launcher",
#     )
#     parser.add_argument("--local_rank", type=int, default=0)
#     parser.add_argument(
#         "--autoscale-lr",
#         action="store_true",
#         help="automatically scale lr with the number of gpus",
#     )
#     args = parser.parse_args()
#     if "LOCAL_RANK" not in os.environ:
#         os.environ["LOCAL_RANK"] = str(args.local_rank)

#     return args

def build_bev_dataset_oritrain(cfg_path):
    cfg = Config.fromfile(cfg_path)
    # dataset = None
    dataset = [build_dataset(cfg.data.train)]
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds, cfg.data.samples_per_gpu, cfg.data.workers_per_gpu, dist=False
        )
        for ds in dataset
    ]

    return data_loaders[0]

def build_bev_dataset_oritest(cfg_path):
    cfg = Config.fromfile(cfg_path)
    # dataset = None

    dataset = build_dataset(cfg.data.train)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=None,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_kitti,
        pin_memory=False,
    )

    return data_loader


if __name__ == "__main__":
    cfg_path = './dataconfig.py'
    data_loader = build_bev_dataset_oritest(cfg_path)

    for i, data_batch in enumerate(data_loader):
        # know about the input and output
        # print(data_batch)
        for key in data_batch:
            print(key)
            try:
                print(np.shape(data_batch[key]))
            except:
                print("no shape")
