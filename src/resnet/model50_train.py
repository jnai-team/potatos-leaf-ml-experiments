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

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(curdir, os.pardir))

import json
import torch
from sklearn.metrics import accuracy_score
import torchvision
from torchinfo import summary
import pandas as pd

# Get ENV
from env import ENV
import trainer
import visual
from torchvision import transforms
from image_dataset import model_dataloder
from common.utils import get_humanreadable_timestamp
from common.logger import FileLogger

ROOT_DIR = os.path.join(curdir, os.pardir, os.pardir)
DATA_ROOT_DIR = ENV.str("DATA_ROOT_DIR", None)
DATASET_NAME = "sample4"
MODEL_NAME = "resnet_model50"
MODEL_ID = get_humanreadable_timestamp()
RESULT_DIR = os.path.join(ROOT_DIR, "results", MODEL_NAME, MODEL_ID)
HYPER_PARAMS_FILE = os.path.join(RESULT_DIR, "hyper_params.json") # 超参数
HYPER_PARAMS = dict()
LOG_FILE = os.path.join(RESULT_DIR, "train.log")
logger = FileLogger(LOG_FILE)

def get_resnet_dataloaders(model_weights, label2num, split_data_name):
    # resnet_weight.transforms()
    from resnet.transforms import resnet_transform
    resnet_train_dataloader, resnet_val_dataloader, resnet_test_dataloader = model_dataloder(weights=model_weights,
                                                                                             transform=resnet_transform,
                                                                                             label2num=label2num,
                                                                                             split_data_name=split_data_name)

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
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    print("Save result into dir %s" % RESULT_DIR)
    print("Log file %s" % LOG_FILE)
    training_times = pd.DataFrame(columns=['Model', 'Testing Accuracy', 'Training_Time(Minutes)'])
    training_times.to_csv(os.path.join(RESULT_DIR, "training_time.csv"), index=False)
    
    '''
    Load weights
    '''
    resnet_weight_50 = torchvision.models.ResNet50_Weights.DEFAULT

    '''
    Load dataloader and class labels metadata
    '''
    split_data_name = "%s_pp_1" % DATASET_NAME
    label2num_file = os.path.join(DATA_ROOT_DIR, "%s.labels.label2num.json" % DATASET_NAME)
    if not os.path.exists(label2num_file):
        raise "Not exist %s" % label2num_file

    label2num = None
    with open(label2num_file, "r") as fin:
        label2num = json.load(fin)


    NUM_CLASSES = len(label2num.keys())
    resnet_train_dataloader, resnet_val_dataloader, resnet_test_dataloader = get_resnet_dataloaders(resnet_weight_50, label2num, split_data_name)


    '''
    Load model
    '''
    resnet_model_50 = torchvision.models.resnet50(weights=resnet_weight_50)
    resnet_model_50 = trainer.add_custom_layers(resnet_model_50, NUM_CLASSES)
    logger.info(summary(resnet_model_50))

    '''
    Training
    '''
    # Actual training ResNet model
    HYPER_PARAMS["lr"] = 0.0005
    HYPER_PARAMS["epochs"] = 10
    HYPER_PARAMS["patience"] = 5
    with open(HYPER_PARAMS_FILE, "w") as fout:
        json.dump(HYPER_PARAMS, fout, ensure_ascii=False, indent=4)
        
    dic_results, training_time = trainer.training_loop(model=resnet_model_50,
                                                          train_dataloader=resnet_train_dataloader,
                                                          val_dataloader=resnet_val_dataloader,
                                                          device=trainer.device,
                                                          epochs=HYPER_PARAMS["epochs"],
                                                          lr=HYPER_PARAMS["lr"],
                                                          patience=HYPER_PARAMS["patience"],
                                                          logger=logger)

    
    '''
    Paint figures
    '''
    # Add the 'Train loss' and 'Val loss' traces as lines
    fig = visual.go_figure(title="Loss over Epochs", xaxis_title="Epochs", yaxis_title="Loss", data=[dict({
        "name":"Train loss",
        "numbers": dic_results["Train_loss"],
        "mode": "lines"
    }), dict({
        "name": "Val loss",
        "numbers": dic_results["Validation_loss"],
        "mode": "lines"
    })], is_show=True)
    # program hangs as https://community.plotly.com/t/plotly-write-image-doesnt-run/63972/3
    # visual.save_figure2img(fig, os.path.join(RESULT_DIR, "loss_graph.jpg"))

    # Create the figure for the chart
    fig = visual.go_figure(title="Accuracy over Epochs", xaxis_title="Epochs", yaxis_title="Accuracy", data=[dict({
        "name": "Train Accuracy",
        "numbers": dic_results["Train_Accuracy"],
        "mode": "lines"
    }), dict({
        "name": "Val Accuracy",
        "numbers": dic_results["Validation_Accuracy"],
        "mode": "lines"
    })], is_show=True)
    # visual.save_figure2img(fig, os.path.join(RESULT_DIR, "accuracy_graph.jpg"))

    '''
    Save metrcis
    '''
    resnet_results = pd.DataFrame(dic_results)
    resnet_results.to_csv(os.path.join(RESULT_DIR, 'train.csv'), index=False)

    test_accuracy = test_accuracy_resnet(resnet_model_50, resnet_test_dataloader, trainer.device)
    logger.info(f"Testing Accuracy is {test_accuracy}%")
    logger.info(f'Model Training Time {round(training_time / 60, 4)} Minitues')

    training_times = pd.read_csv(os.path.join(RESULT_DIR, 'training_time.csv'))
    row = {'Model': 'ResNet50', 'Testing Accuracy': test_accuracy,
           'Training_Time(Minutes)': round(training_time / 60, 4)}

    # Use the loc method to add the new row to the DataFrame
    training_times.loc[len(training_times)] = row
    training_times.to_csv(os.path.join(RESULT_DIR, 'training_time.csv'), index=False)

    torch.save(resnet_model_50, os.path.join(RESULT_DIR, 'model.pth'))


if __name__ == '__main__':
    main()
