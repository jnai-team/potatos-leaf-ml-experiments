#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===============================================================================
#
# Copyright (c) 2025 <> All Rights Reserved
#
#
# File: /home/Administrator/projects/2025_03_01_zhangphd_paper/experiments/src/resnet/model50_predict.py
# Author: Hai Liang Wang
# Date: 2025-03-22:16:56:58
#
# ===============================================================================

"""

"""
__copyright__ = "Copyright (c) 2020 . All Rights Reserved"
__author__ = "Hai Liang Wang"
__date__ = "2025-03-22:16:56:58"

import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(curdir, os.pardir))

if sys.version_info[0] < 3:
    raise RuntimeError("Must be using Python 3")
else:
    unicode = str

import json
import torch

# Get ENV
from env import ENV
from common.logger import FileLogger
from transforms import read_image
from image_dataset import load_test_data
import visual

ROOT_DIR = os.path.join(curdir, os.pardir, os.pardir)
DATA_ROOT_DIR = ENV.str("DATA_ROOT_DIR", None)
DATASET_NAME = ENV.str("DATASET_NAME", None)
MODEL_NAME = "resnet_model50"
MODEL_ID = ENV.str("MODEL_ID", None)
RESULT_DIR = os.path.join(ROOT_DIR, "results", MODEL_NAME, MODEL_ID)
PREDICT_TARGETS = ENV.str("PREDICT_TARGETS", None)
PREDICT_RESULT = os.path.join(RESULT_DIR, "predict_result.csv")
PREDICT_LOG = os.path.join(RESULT_DIR, "predict.log")

logger = FileLogger(PREDICT_LOG)
device = "cuda" if torch.cuda.is_available() else "cpu"

'''
Parse predict data labels
'''
if PREDICT_TARGETS is None:
    raise BaseException("ERROR, PREDICT_TARGETS should not be None.")

if not os.path.exists(RESULT_DIR):
    raise BaseException("Error, not found %s" % RESULT_DIR)


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
    logger.info("Predict with model %s" % RESULT_DIR)
    model = torch.load(os.path.join(RESULT_DIR, "model.pth"), weights_only=False)
    model.to(device)
    model.eval()

    '''
    Load dataloader and class labels metadata
    '''
    label2num_file = os.path.join(DATA_ROOT_DIR, "%s.labels.label2num.json" % DATASET_NAME)
    if not os.path.exists(label2num_file):
        raise "Not exist %s" % label2num_file

    num2label_file = os.path.join(DATA_ROOT_DIR, "%s.labels.num2label.json" % DATASET_NAME)
    if not os.path.exists(num2label_file):
        raise "Not exist %s" % num2label_file

    num2label = None
    with open(num2label_file, "r") as fin:
        num2label = json.load(fin)

    '''
    parse targets
    '''
    targets = load_test_data(PREDICT_TARGETS)

    '''
    load all images as targets
    '''
    output_lines = ["image,actually,predicted\n"]
    predicted_labels = []
    actual_labels = []
    for image_path in targets.keys():
        predicted_label = predict(model, image_path, num2label)
        actual_label = targets[image_path]

        predicted_labels.append(predicted_label)
        actual_labels.append(actual_label)

        output_lines.append("%s,%s,%s\n" % (image_path, actual_label, predicted_label))

    # With average='micro', the three values would be the same, for multi classes, the average should be set as macro
    # https://stackoverflow.com/questions/71799168/can-the-f1-precision-accuracy-and-recall-all-have-the-same-values
    # https://blog.csdn.net/qq_45041871/article/details/128385945
    accuracy_score = visual.accuracy_score(actual_labels, predicted_labels)
    f1_score = visual.f1_score(actual_labels, predicted_labels, average='macro')
    recall_score = visual.recall_score(actual_labels, predicted_labels, average='macro')
    logger.info("Precision in predicting %s total %s" %
                (("%.2f" % (accuracy_score * 100)) + "%", len(actual_labels)))
    logger.info("F1 Score in predicting %s" % f1_score)
    logger.info("Recall Score in predicting %s" % recall_score)

    # TODO plot ROC Cure
    # https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python

    with open(PREDICT_RESULT, "w", encoding="utf-8") as fout:
        fout.writelines(output_lines)

    logger.info("Predict result for every image saved in %s" % PREDICT_RESULT)


main()
