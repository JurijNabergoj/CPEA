import os
import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

import random
import math
import torch

IMAGE_PATH = 'C:\\Users\\Jurij\\PycharmProjects\\CPEA\\CPEA\\datasets\\tiered'
SPLIT_PATH = 'C:\\Users\\Jurij\\PycharmProjects\\CPEA\\tiered-imagenet-tools\\tiered_imagenet_split'


class TieredImagenet(Dataset):
    """ Usage:
    """

    def __init__(self, setname, args):
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        # import pdb
        # pdb.set_trace()
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(osp.join(IMAGE_PATH, setname), name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.num_class = len(set(label))
        self.setname = setname

        image_size = 224  # 112, 128; 144, 168
        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                 np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
        ])
        self.transform_val_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                 np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        if label == 0:
            label = 1
        num_zeros = max(0, 8 - len(str(label)))
        zeros = '0' * num_zeros
        img_path = os.path.split(path)[-1] + zeros + str(label) + '.jpg'
        final_path = path + "\\" + img_path
        # print(final_path)
        if self.setname == 'train':
            image = self.transform_train(Image.open(final_path).convert('RGB'))
        else:
            image = self.transform_val_test(Image.open(final_path).convert('RGB'))
        return image, label
