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
import torchvision
from torchinfo import summary
import pandas as pd
import shutil
from torch import nn as nn

# Get ENV
from env import ENV, ENV_LOCAL_RC
import trainer
import visual
from image_dataset import load_dev_data
from common.utils import get_humanreadable_timestamp
from common.logger import FileLogger

ROOT_DIR = os.path.join(curdir, os.pardir, os.pardir)
DATA_ROOT_DIR = ENV.str("DATA_ROOT_DIR", None)
# DATASET_NAME = "sample4"
DATASET_NAME = ENV.str("DATASET_NAME", None)
MODEL_NAME = "resnet_model50"
MODEL_ID = get_humanreadable_timestamp()
RESULT_DIR = os.path.join(ROOT_DIR, "results", MODEL_NAME, MODEL_ID)
HYPER_PARAMS_FILE = os.path.join(RESULT_DIR, "hyper_params.json") # 超参数
HYPER_PARAMS = dict()
LOG_FILE = os.path.join(RESULT_DIR, "train.log")
logger = FileLogger(LOG_FILE)


if DATASET_NAME is None:
    raise "Error, DATASET_NAME is None"

logger.info("Train with dataset name %s" % DATASET_NAME)

def add_custom_layers(model, last_layer_num):
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features

    # Define your custom layers (you can modify this part as per your requirements)
    custom_layers = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, last_layer_num)  # Assuming you have defined num_classes for your specific task
    )

    # Replace the last layer of the ResNet-50 model with the custom layers
    model.fc = custom_layers

    return model

def get_resnet_dataloaders(label2num, split_data_name):
    resnet_train_dataloader, resnet_val_dataloader = load_dev_data(label2num=label2num,
                                                                                             split_data_name=split_data_name)

    return resnet_train_dataloader, resnet_val_dataloader


# def test_accuracy_resnet(model, dataloader, device):
#     # empty list store labels
#     predict_label_list = []
#     actual_label_list = []

#     # eval mode
#     model.eval()

#     for images, labels in dataloader:
#         for label in labels:
#             label = label.item()
#             actual_label_list.append(label)

#         for image in images:
#             with torch.inference_mode():
#                 image = image.to(device)
#                 # add batch_size and device
#                 image = image.unsqueeze(dim=0)
#                 # logits
#                 logits = model(image)
#                 # lables
#                 label = torch.argmax(logits).item()
#                 predict_label_list.append(label)

#     accuracy = visual.accuracy_score(actual_label_list, predict_label_list)
#     return accuracy * 100


def main():
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    print("Save result into dir %s" % RESULT_DIR)
    print("Log file %s" % LOG_FILE)
    training_times = pd.DataFrame(columns=['Model', 'Training_Time(Minutes)'])
    training_times.to_csv(os.path.join(RESULT_DIR, "training_time.csv"), index=False)
    
    # copy .env as params
    shutil.copyfile(ENV_LOCAL_RC, os.path.join(RESULT_DIR, ".env"))

    '''
    Load weights
    '''
    resnet_weight_50 = torchvision.models.ResNet50_Weights.DEFAULT

    '''
    Load dataloader and class labels metadata
    '''
    split_data_name = "%s_pp_1" % DATASET_NAME
    logger.info(">> Used dataset splitted data %s" % os.path.join(DATA_ROOT_DIR, split_data_name))
    label2num_file = os.path.join(DATA_ROOT_DIR, "%s.labels.label2num.json" % DATASET_NAME)
    if not os.path.exists(label2num_file):
        raise "Not exist %s" % label2num_file
    
    label2num = None
    with open(label2num_file, "r") as fin:
        label2num = json.load(fin)


    NUM_CLASSES = len(label2num.keys())
    resnet_train_dataloader, resnet_val_dataloader = get_resnet_dataloaders(label2num, split_data_name)

    '''
    Load model
    '''
    resnet_model_50 = torchvision.models.resnet50(weights=resnet_weight_50)
    resnet_model_50 = add_custom_layers(resnet_model_50, NUM_CLASSES)
    logger.info(summary(resnet_model_50, verbose=0))

    '''
    Export graph file
    '''
    input_sample = torch.randn(1, 3, 224, 224)
    # model_onnx_file = os.path.join(RESULT_DIR, 'model.onnx')
    # visual.export_onnx_archive(model=resnet_model_50, filepath=model_onnx_file, input_sample=input_sample)
    # logger.info("onnx file --> %s" % model_onnx_file)

    model_graph_file = visual.export_model_graph(model=resnet_model_50, input_sample=input_sample, directory = RESULT_DIR)
    logger.info("model graph file saved --> %s" % model_graph_file)

    '''
    Training
    '''
    # Actual training ResNet model
    resnet_model_50.train()
    HYPER_PARAMS["lr"] = ENV.float("HYPER_PARAMS_LR", 0.0005)
    HYPER_PARAMS["epochs"] = ENV.int("HYPER_PARAMS_EPOCHS", 10)
    HYPER_PARAMS["patience"] = ENV.int("HYPER_PARAMS_PATIENCE", 5)
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
    visual.save_figure2jpg(fig, os.path.join(RESULT_DIR, "loss_graph.jpg"))

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
    visual.save_figure2jpg(fig, os.path.join(RESULT_DIR, "accuracy_graph.jpg"))

    '''
    Save metrcis
    '''
    resnet_results = pd.DataFrame(dic_results)
    resnet_results.to_csv(os.path.join(RESULT_DIR, 'train.csv'), index=False)

    # test_accuracy = test_accuracy_resnet(resnet_model_50, resnet_test_dataloader, trainer.device)
    # logger.info(f"Testing Accuracy is {test_accuracy}%")
    logger.info(f'Model Training Time {round(training_time / 60, 4)} Minitues')

    training_times = pd.read_csv(os.path.join(RESULT_DIR, 'training_time.csv'))
    row = {'Model': 'ResNet50', 'Training_Time(Minutes)': round(training_time / 60, 4)}

    # Use the loc method to add the new row to the DataFrame
    training_times.loc[len(training_times)] = row
    training_times.to_csv(os.path.join(RESULT_DIR, 'training_time.csv'), index=False)

    model_saved_path = os.path.join(RESULT_DIR, 'model.pth')
    torch.save(resnet_model_50, model_saved_path)
    logger.info("Saved model %s" % model_saved_path)

if __name__ == '__main__':
    main()
