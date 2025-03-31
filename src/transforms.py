#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchvision import transforms
from PIL import Image

# transform in number 1
# A callable function to transform the images (e.g., resizing, normalization).
# mean and std params are generated with ImageNet dataset.
transform1 = transforms.Compose([
    # transforms.CenterCrop(size=(232, 232)),
    transforms.Resize(size=(232, 232)),
    # transforms.ColorJitter(brightness=(0.8, 1.2)),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def read_image(image_path):
    '''
    Read image as input
    '''
    image = Image.open(image_path).convert('RGB')

    if image.mode == "L":
        image = Image.merge("RGB", (image, image, image))

    image = transform1(image)
    return image


