#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===============================================================================
#
# Copyright (c) 2024 <> All Rights Reserved
#
#
# File: /c/Users/Administrator/courses/CV/Sports-Image-Classification-YOLO-ResNet/src/run_resnet.py
# Author: Hai Liang Wang
# Date: 2024-12-18:15:48:28
#
# ===============================================================================

"""
   
"""
__copyright__ = "Copyright (c) 2020 . All Rights Reserved"
__author__ = "Hai Liang Wang"
__date__ = "2024-12-18:15:48:28"

import os
import sys

if sys.version_info[0] < 3:
    raise RuntimeError("Must be using Python 3")
else:
    unicode = str

import torch
from sklearn.metrics import accuracy_score
import torchvision
from torchinfo import summary
import pandas as pd

# Get ENV
ENVIRON = os.environ.copy()
# This task has 7 classes
NUM_CLASSES = 3

import trainer
from torchvision import transforms
from image_dataset import model_dataloder


def get_resnet_dataloaders(model_weights):
    # resnet_weight.transforms()
    resnet_transform = transforms.Compose([
        transforms.Resize(size=(232, 232)),
        transforms.ColorJitter(brightness=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    resnet_train_dataloader, resnet_val_dataloader, resnet_test_dataloader = model_dataloder(weights=model_weights,
                                                                                             transform=resnet_transform)

    return resnet_train_dataloader, resnet_val_dataloader, resnet_test_dataloader


def test_accuracy_resnet(model, dataloader, device):
    # empty list store labels
    predict_label_list = []
    actual_label_list = []

    # eval mode
    model.eval()

    for images, labels in dataloader:

        for label in labels:
            label = label.item()
            actual_label_list.append(label)

        for image in images:
            with torch.inference_mode():
                image = image.to(device)
                # add batch_size and device
                image = image.unsqueeze(dim=0)
                # logits
                logits = model(image)
                # lables
                label = torch.argmax(logits).item()
                predict_label_list.append(label)

    accuracy = accuracy_score(actual_label_list, predict_label_list)
    return accuracy * 100


def main():
    target_dir = '../assets/models_results'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # 原main函数中的其他代码
    training_times = pd.DataFrame(columns=['Model', 'Testing Accuracy', 'Training_Time(Minutes)'])
    training_times.to_csv('../assets/models_results/training_time.csv', index=False)
    #...
    curdir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(curdir)

    resnet_weight_50 = torchvision.models.ResNet50_Weights.DEFAULT
    resnet_model_50 = torchvision.models.resnet50(weights=resnet_weight_50)
    resnet_model_50 = trainer.add_custom_layers(resnet_model_50, NUM_CLASSES)
    summary(resnet_model_50)

    resnet_train_dataloader, resnet_val_dataloader, resnet_test_dataloader = get_resnet_dataloaders(resnet_weight_50)

    # Actual training ResNet model
    resnet_results, training_time = trainer.training_loop(model=resnet_model_50,
                                                          train_dataloader=resnet_train_dataloader,
                                                          val_dataloader=resnet_val_dataloader,
                                                          device=trainer.device,
                                                          epochs=10,
                                                          patience=5)

    resnet_results.to_csv('../assets/models_results/resnet_model_50.csv', index=False)

    test_accuracy = test_accuracy_resnet(resnet_model_50, resnet_test_dataloader, trainer.device)
    print(f"Testing Accuracy is {test_accuracy}%")
    print(f'Model Training Time {round(training_time / 60, 4)} Minitues')

    training_times = pd.read_csv('../assets/models_results/training_time.csv')
    row = {'Model': 'ResNet50', 'Testing Accuracy': test_accuracy,
           'Training_Time(Minutes)': round(training_time / 60, 4)}

    # Use the loc method to add the new row to the DataFrame
    training_times.loc[len(training_times)] = row
    training_times.to_csv('../assets/models_results/training_time.csv', index=False)

    torch.save(resnet_model_50, '../assets/trained_models/resnet_model_50.pth')


if __name__ == '__main__':
    main()
