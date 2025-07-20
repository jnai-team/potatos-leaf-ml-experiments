#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===============================================================================
#
# Copyright (c) 2024 <> All Rights Reserved
#
#
# File: /d/git/Sports-Image-Classification-YOLO-ResNet/src/preprocess_data.py
# Author: Hai Liang Wang
# Date: 2024-12-18:12:11:47
#
# ===============================================================================

"""

"""
__copyright__ = "Copyright (c) 2020 . All Rights Reserved"
__author__ = "Hai Liang Wang"
__date__ = "2024-12-18:12:11:47"

import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

if sys.version_info[0] < 3:
    raise RuntimeError("Must be using Python 3")
else:
    unicode = str

import os
import pathlib
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transforms import read_image


# Raw dataset
class ImageDataset(Dataset):
    def __init__(self, root_dir, class_labels=None):
        """
        Initializes the ImageDataset object.

        Parameters:
            root_dir (str): The root directory path containing sub-directories, each representing a class.
        """

        self.root_dir = root_dir
        self.images = []
        self.labels = []
        # a mapping of class labels to integers: labels to num
        self.class_labels = class_labels

        if self.class_labels is None:
            raise "Invalid class_labels data, should not be None"

        # Iterate over sub-directories
        for class_dir in os.listdir(self.root_dir):
            class_dir_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_dir_path):
                # Iterate over images in the sub - directory
                for img_filename in os.listdir(class_dir_path):
                    if img_filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        img_path = os.path.join(class_dir_path, img_filename)
                        self.images.append(img_path)
                        if class_dir in self.class_labels:
                            self.labels.append(self.class_labels[class_dir])
                        else:
                            raise "%s not found in class_labels %s" % (
                                class_dir, self.class_labels)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """

        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Parameters:
            idx (int): The index of the sample to retrieve.

        Returns:
            PIL.Image.Image: The image sample.
            int: The label associated with the image.
        """

        image = read_image(self.images[idx])
        label = self.labels[idx]

        return image, label


def load_dev_data(label2num, data_root_dir):
    """
    Returns three PyTorch DataLoaders for training, validation.

    Parameters:
        label2num: label to number data
        split_data_name: dataset after splitted

    Returns:
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
    """
    if data_root_dir is None:
        raise ValueError("data_root_dir variable is not set.")
    if not os.path.exists(data_root_dir):
        raise FileNotFoundError(f"Data folder {data_root_dir} does not exist.")
    
    train_folder = os.path.join(data_root_dir, "train")
    test_folder = os.path.join(data_root_dir, "test")
    for folder in [train_folder, test_folder]:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder {folder} does not exist.")

    # pytorch dataset
    train_dataset = ImageDataset(train_folder, class_labels=label2num)
    test_dataset = ImageDataset(test_folder, class_labels=label2num)

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Check data source and path.")
    
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty. Check data source and path.")

    # pytorch dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, test_dataloader


def parse_image_targets(env_val):
    '''
    Parse targets as label and images' dir
    params:
        * env_val: string as `folder1#label1,folder2#label2,...`
    '''
    targets = dict()
    splits = env_val.split(",")
    for sp in splits:
        label = None
        sp = sp.rstrip()
        if "#" in sp:
            parts = sp.split("#")
            targets_dir = parts[0].rstrip()
            if not os.path.exists(targets_dir):
                print("Predict targets dir not exist: %s" % targets_dir)
                raise BaseException("Folder not found")

            label = parts[1].rstrip()
            targets[targets_dir] = label
        else:
            if not os.path.exists(sp):
                print("Predict targets dir not exist: %s" % targets_dir)
                raise BaseException("Folder not found")

            label = pathlib.PurePath(sp).name
            targets[sp] = label

    return targets


def load_test_data(targets_env_val):
    '''
    Load dataset for testing
    Return:
        * images: dict[image path] = fact label
    '''
    targets = parse_image_targets(targets_env_val)
    data = dict()

    for target_folder in targets.keys():
        for _, _, files in os.walk(target_folder):
            for x in files:
                # only check png, jpg, jpeg.
                if x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg"):
                    image_path = os.path.join(target_folder, x)
                    data[image_path] = targets[target_folder]

    return data


def get_1st_tensor(dataloader):
    '''
    Get first tensor as input sample
    '''

    inputs, classes = next(iter(dataloader))
    return inputs[0]
