#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===============================================================================
#
# Copyright (c) 2024 <> All Rights Reserved
#
#
# File: /c/Users/Administrator/courses/CV/Sports-Image-Classification-YOLO-ResNet/src/run_train.py
# Author: Hai Liang Wang
# Date: 2024-12-18:15:41:45
#
# ===============================================================================

"""
   
"""
__copyright__ = "Copyright (c) 2020 . All Rights Reserved"
__author__ = "Hai Liang Wang"
__date__ = "2024-12-18:15:41:45"

import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

if sys.version_info[0] < 3:
    raise RuntimeError("Must be using Python 3")
else:
    unicode = str

import time
import numpy as np
import torch
from torch import nn as nn


# Get ENV
ENVIRON = os.environ.copy()

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"


# hardcode :
# loss_fn -> CrossEntropyLoss
# optimizer -> Adam(lr = 0.0005)

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


# Train -> train_loss, train_acc
def train(model, dataloader, loss_fn, optimizer, device):
    train_loss, train_acc = 0, 0

    model.to(device)
    model.train()

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        train_pred = model(x)

        loss = loss_fn(train_pred, y)
        train_loss = train_loss + loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_pred_label = torch.argmax(torch.softmax(train_pred, dim=1), dim=1)
        train_acc = train_acc + (train_pred_label == y).sum().item() / len(train_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc


# Val -> val_loss, val_acc
def val(model, dataloader, loss_fn, device):
    val_loss, val_acc = 0, 0

    model.to(device)
    model.eval()

    with torch.inference_mode():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            val_pred = model(x)

            loss = loss_fn(val_pred, y)
            val_loss = val_loss + loss.item()

            val_pred_label = torch.argmax(torch.softmax(val_pred, dim=1), dim=1)
            val_acc = val_acc + (val_pred_label == y).sum().item() / len(val_pred)

        val_loss = val_loss / len(dataloader)
        val_acc = val_acc / len(dataloader)

        return val_loss, val_acc


def classify_custom_images(model, dataloader, device, df):
    pred_labels = []
    # eval mode
    model.eval()

    for images, labels in dataloader:
        for image in images:
            with torch.inference_mode():
                image = image.to(device)
                # add batch_size and device
                image = image.unsqueeze(dim=0)
                # logits
                logits = model(image)
                # lables
                label = torch.argmax(logits).item()
                text_label = df[df['class id'] == label]['labels'].iloc[0]
                pred_labels.append(text_label)
    
    return pred_labels


def training_loop(model, 
                  train_dataloader, 
                  val_dataloader, 
                  device, 
                  epochs, 
                  lr,
                  patience, 
                  logger):
    # empty dict for restore results
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # hardcode loss_fn and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # variable to hold the training time
    training_time = 0.0
    epoch_run_time = []
    # loop through epochs
    for epoch in range(epochs):

        # record the start time for each epoch
        epoch_start_time = time.time()

        train_loss, train_acc = train(model=model,
                                      dataloader=train_dataloader,
                                      loss_fn=loss_fn,
                                      optimizer=optimizer,
                                      device=device)

        val_loss, val_acc = val(model=model,
                                dataloader=val_dataloader,
                                loss_fn=loss_fn,
                                device=device)

        # record the end time for each epoch
        epoch_end_time = time.time()

        # calculate the time taken for this epoch
        epoch_time = epoch_end_time - epoch_start_time
        epoch_run_time.append(epoch_time)
        training_time += epoch_time

        # print results for each epoch
        logger.info(f"Epoch: {epoch + 1}\n"
              f"Train loss: {train_loss:.4f} | Train accuracy: {(train_acc * 100):.3f}%\n"
              f"Val loss: {val_loss:.4f} | Val accuracy: {(val_acc * 100):.3f}%\n"
              f"| Epoch time: {epoch_time:.2f} seconds")

        # record results for each epoch
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        # calculate average "val_loss" for early_stopping
        mean_val_loss = np.mean(results["val_loss"])
        best_val_loss = float("inf")
        num_no_improvement = 0
        if np.mean(mean_val_loss > best_val_loss):
            best_val_loss = mean_val_loss

            model_state_dict = model.state_dict()
            # best_model.load_state_dict(model_state_dict)
        else:
            num_no_improvement += 1

        if num_no_improvement == patience:
            break

    # Saving Results for model
    dic_results = {
        "epochs": list(range(1, epochs + 1)),
        'Train_loss': results["train_loss"],
        'Train_Accuracy': results['train_acc'],
        'Validation_loss': results["val_loss"],
        'Validation_Accuracy': results['val_acc'],
        'Time_Taken': epoch_run_time

    }

    return dic_results, training_time