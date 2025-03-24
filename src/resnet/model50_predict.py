#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2025 <> All Rights Reserved
#
#
# File: /home/Administrator/projects/2025_03_01_zhangphd_paper/experiments/src/resnet/model50_predict.py
# Author: Hai Liang Wang
# Date: 2025-03-22:16:56:58
#
#===============================================================================

"""
   
"""
__copyright__ = "Copyright (c) 2020 . All Rights Reserved"
__author__ = "Hai Liang Wang"
__date__ = "2025-03-22:16:56:58"

import os, sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(curdir, os.pardir))

if sys.version_info[0] < 3:
    raise RuntimeError("Must be using Python 3")
else:
    unicode = str

import json
import torch
from PIL import Image
import pathlib

# Get ENV
from env import ENV
from common.logger import FileLogger
from resnet.transforms import resnet_transform

ROOT_DIR = os.path.join(curdir, os.pardir, os.pardir)
DATA_ROOT_DIR = ENV.str("DATA_ROOT_DIR", None)
DATASET_NAME = "sample4"
MODEL_NAME = "resnet_model50"
MODEL_ID = ENV.str("MODEL_ID", None)
RESULT_DIR = os.path.join(ROOT_DIR, "results", MODEL_NAME, MODEL_ID)
PREDICT_TARGETS_DIR = ENV.str("PREDICT_TARGETS_DIR", None)
PREDICT_RESULT = os.path.join(RESULT_DIR, "predict_result.txt")
PREDICT_LOG = os.path.join(RESULT_DIR, "predict.log")

logger = FileLogger(PREDICT_LOG)
device = "cuda" if torch.cuda.is_available() else "cpu"

'''
Parse predict data labels
'''
PREDICT_TARGETS_LABEL=None
if "#" in PREDICT_TARGETS_DIR:
    splits = PREDICT_TARGETS_DIR.split("#")
    PREDICT_TARGETS_DIR = splits[0].rstrip()
    PREDICT_TARGETS_LABEL = splits[1].rstrip()
else:
    PREDICT_TARGETS_LABEL=pathlib.PurePath(PREDICT_TARGETS_DIR).name

'''
Save png,jpg,jpeg images as predict targets in dir PREDICT_TARGETS_DIR
'''
if (not PREDICT_TARGETS_DIR) or (not os.path.exists(PREDICT_TARGETS_DIR)):
    raise "Error %s not found" % PREDICT_TARGETS_DIR

if not os.path.exists(RESULT_DIR):
    raise "Error, not found %s" % RESULT_DIR


def read_image(image_path):
    '''
    Read image as input
    '''
    image = Image.open(image_path).convert('RGB')

    if image.mode == "L":
        image = Image.merge("RGB", (image, image, image))

    image = resnet_transform(image)
    return image

def predict(model, image_path, num2label):
    '''
    Predict image
    '''
    logger.info("predict image: %s" % image_path)
    
    image = read_image(image_path=image_path)
    label = None

    with torch.inference_mode():
        image = image.to(device)
        image = image.unsqueeze(dim=0)
        # logits
        logits = model(image)
        # lables
        num = torch.argmax(logits).item()
        label = num2label[str(num)]
        logger.info("%s label --> %s" % (image_path, label))
        return label
    
    return label


def main():
    '''
    Run predict logic
    '''
    # load model
    model = torch.load(os.path.join(RESULT_DIR, "model.pth"), weights_only=False)
    model.to(device)
    model.eval()

    '''
    Load dataloader and class labels metadata
    '''
    label2num_file = os.path.join(DATA_ROOT_DIR, "%s.labels.label2num.json" % DATASET_NAME)
    if not os.path.exists(label2num_file):
        raise "Not exist %s" % label2num_file

    label2num = None
    with open(label2num_file, "r") as fin:
        label2num = json.load(fin)

    num2label_file = os.path.join(DATA_ROOT_DIR, "%s.labels.num2label.json" % DATASET_NAME)
    if not os.path.exists(num2label_file):
        raise "Not exist %s" % num2label_file

    num2label = None
    with open(num2label_file, "r") as fin:
        num2label = json.load(fin)

    '''
    load all images as targets
    '''
    output_lines = ["image,desired,predicted\n"]
    for _, _, images in os.walk(PREDICT_TARGETS_DIR):
        total_files = len(images)
        logger.info("%s has %s files" % (PREDICT_TARGETS_DIR, total_files))
        corrected = 0
        total = 0

        for x in images:
            if x.endswith(".png") or x.endswith(".jpg"):
                image_path = os.path.join(PREDICT_TARGETS_DIR, x)
                label = predict(model, image_path, num2label)
                output_lines.append("%s,%s,%s\n" % (image_path, PREDICT_TARGETS_LABEL, label))

                if PREDICT_TARGETS_LABEL == label:
                    corrected += 1

                total += 1

    with open(PREDICT_RESULT, "w", encoding="utf-8") as fout:
        fout.writelines(output_lines)


    logger.info("Precision %s/%s" % (corrected, total))

main()

