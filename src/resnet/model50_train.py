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

'''
Get ENV
'''
ENV_LOCAL_RC = os.path.join(curdir, os.pardir, os.pardir, ".env")
from env3 import load_env
ENV = load_env(dotenv_file=ENV_LOCAL_RC)

'''
Constants
'''
from common.utils import get_humanreadable_timestamp, read_dataset_name
ROOT_DIR = os.path.join(curdir, os.pardir, os.pardir)
MODEL_NAME = "resnet_model50"
MODEL_ID = get_humanreadable_timestamp()
RESULT_DIR = os.path.join(ROOT_DIR, "results", MODEL_NAME, MODEL_ID)
DATA_ROOT_DIR = ENV.get("DATA_ROOT_DIR", os.path.join(ROOT_DIR, "data"))
DATASET_NAME = read_dataset_name(DATA_ROOT_DIR)
HYPER_PARAMS_FILE = os.path.join(RESULT_DIR, "hyper_params.json")  # 超参数
HYPER_PARAMS = dict()

'''
Init logger
'''
os.environ['LOG_FILE'] = os.path.join(RESULT_DIR,
                                      "train.log")  # set logger properties before import log5
import log5
logger = log5.get_logger(logger_name=log5.LN(__name__), output_mode=log5.OUTPUT_BOTH)

'''
Actural Work
'''
import trainer
import visual
from image_dataset import load_dev_data


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
        # Assuming you have defined num_classes for your specific task
        nn.Linear(512, last_layer_num)
    )

    # Replace the last layer of the ResNet-50 model with the custom layers
    model.fc = custom_layers

    return model


def get_resnet_dataloaders(label2num):
    resnet_train_dataloader, resnet_val_dataloader = load_dev_data(label2num=label2num,
                                                                   data_root_dir=DATA_ROOT_DIR)

    return resnet_train_dataloader, resnet_val_dataloader


def main():
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    print("Save result into dir %s" % RESULT_DIR)
    print("Log file %s" % os.environ['LOG_FILE'])
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
    label2num_file = os.path.join(DATA_ROOT_DIR, "labels", "label2num.json")
    if not os.path.exists(label2num_file):
        raise "Not exist %s" % label2num_file

    label2num = None
    with open(label2num_file, "r") as fin:
        label2num = json.load(fin)

    NUM_CLASSES = len(label2num.keys())
    resnet_train_dataloader, resnet_test_dataloader = get_resnet_dataloaders(
        label2num)

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

    model_graph_file = visual.export_model_graph(
        model=resnet_model_50,
        input_sample=input_sample,
        directory=RESULT_DIR)
    logger.info("model graph file saved --> %s" % model_graph_file)

    '''
    Training
    '''
    # Actual training ResNet model
    resnet_model_50.train()
    HYPER_PARAMS["lr"] = float(ENV.get("HYPER_PARAMS_LR", 0.0005))
    HYPER_PARAMS["epochs"] = int(ENV.get("HYPER_PARAMS_EPOCHS", 10))
    HYPER_PARAMS["patience"] = int(ENV.get("HYPER_PARAMS_PATIENCE", 5))
    with open(HYPER_PARAMS_FILE, "w") as fout:
        json.dump(HYPER_PARAMS, fout, ensure_ascii=False, indent=4)

    dic_results, training_time = trainer.training_loop(model=resnet_model_50,
                                                       train_dataloader=resnet_train_dataloader,
                                                       val_dataloader=resnet_test_dataloader,
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
        "name": "Train loss",
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
    fig = visual.go_figure(title="Accuracy over Epochs",
                           xaxis_title="Epochs",
                           yaxis_title="Accuracy",
                           data=[dict({"name": "Train Accuracy",
                                       "numbers": dic_results["Train_Accuracy"],
                                       "mode": "lines"}),
                                 dict({"name": "Val Accuracy",
                                       "numbers": dic_results["Validation_Accuracy"],
                                       "mode": "lines"})],
                           is_show=True)
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
