#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchvision import transforms

resnet_transform = transforms.Compose([
    transforms.Resize(size=(232, 232)),
    transforms.ColorJitter(brightness=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])