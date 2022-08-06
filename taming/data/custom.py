import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
from det3d.datasets import build_dataloader, build_dataset
from det3d.torchie import Config
from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        # print(example.keys())
        # print(np.shape(example['bin_map']))
        return example['bin_map']



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()

        cfg = Config.fromfile("bev_data.py")
        dataset = build_dataset(cfg.data.train)
        self.data = dataset


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        cfg = Config.fromfile("bev_data.py")
        dataset = build_dataset(cfg.data.val)
        self.data = dataset

