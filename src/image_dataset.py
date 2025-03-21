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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image

# Get ENV
from env import ENV

ROOT_DIR = os.path.join(curdir, os.pardir)
DATA_ROOT_DIR = ENV.str("DATA_ROOT_DIR", None)


# Raw dataset
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_labels=None):

        """
        Initializes the ImageDataset object.

        Parameters:
            root_dir (str): The root directory path containing sub-directories, each representing a class.
            transform (optional, callable): A callable function to transform the images (e.g., resizing, normalization).
        """

        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        # a mapping of class labels to integers: labels to num
        self.class_labels = class_labels

        if self.class_labels == None:
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
                            raise "%s not found in class_labels %s" % (class_dir, self.class_labels)
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

        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]

        if image.mode == "L":
            image = Image.merge("RGB", (image, image, image))
        if self.transform:
            image = self.transform(image)

        return image, label


# pytorch dataloader
def model_dataloder(weights, transform, label2num, split_data_name="sample4_pp_1"):
    """
    Returns three PyTorch DataLoaders for training, validation, and testing.
    
    Parameters:
        weights (list): A list of weights used for data sampling in DataLoader (optional).
        transform (torchvision.transforms): Image transformation to be applied to the datasets.
        
    Returns:
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        test_dataloader (DataLoader): DataLoader for the test dataset.
    """
    weights = weights
    if DATA_ROOT_DIR is None:
        raise ValueError("DATA_ROOT_DIR environment variable is not set.")
    data_folder = os.path.join(DATA_ROOT_DIR, split_data_name)
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder {data_folder} does not exist.")
    train_folder = os.path.join(data_folder, "train")
    val_folder = os.path.join(data_folder, "valid")
    test_folder = os.path.join(data_folder, "test")
    for folder in [train_folder, val_folder, test_folder]:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder {folder} does not exist.")
        
    data_folder = os.path.join(DATA_ROOT_DIR, split_data_name)

    train_folder = data_folder + "/train"
    val_folder = data_folder + "/valid"
    test_folder = data_folder + "/test"

    # pytorch dataset
    train_dataset = ImageDataset(train_folder, transform=transform, class_labels=label2num)
    val_dataset = ImageDataset(val_folder, transform=transform, class_labels=label2num)
    test_dataset = ImageDataset(test_folder, transform=transform, class_labels=label2num)

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Check data source and path.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty. Check data source and path.")
    if len(test_dataset) == 0:
        raise ValueError("Testing dataset is empty. Check data source and path.")

    # pytorch dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader
