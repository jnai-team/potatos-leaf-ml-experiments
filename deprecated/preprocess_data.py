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

import shutil
import math
import csv
import json

from pathlib import Path
from sklearn.utils import shuffle

# Get ENV
from env3 import load_env
from log5 import LN, get_logger

ENV = load_env(dotenv_file=os.path.join(curdir, os.pardir, ".env"))
logger = get_logger(LN(__name__))

ROOT_DIR = os.path.join(curdir, os.pardir)
DATA_ROOT_DIR = ENV.get("DATA_ROOT_DIR", None)
DATA_TRAIN_RECORDS_RATIO = float(ENV.get("DATA_TRAIN_RECORDS_RATIO", "0.9"))
DATA_TEST_RECORDS_SPLIT = ENV.get("DATA_TEST_RECORDS_SPLIT", "false")
DATA_TEST_RECORDS_RATIO = float(ENV.get("DATA_TEST_RECORDS_RATIO", "0.1"))


def move_images_to_train_and_valid(total_data, labels, train_img_dir, val_img_dir):
    '''
    Move images into train, validate, test dirs
    '''

    if not os.path.exists(train_img_dir):
        os.mkdir(train_img_dir)

    if not os.path.exists(val_img_dir):
        os.mkdir(val_img_dir)

    for label in labels:
        class_data = total_data[label]
        logger.info(">> Get records %d for label %s" % (len(class_data), label))
        train_num = math.floor(len(class_data) * DATA_TRAIN_RECORDS_RATIO)
        valid_num = len(class_data) - train_num
        logger.info(">> train_num %d, valid_num %d" % (train_num, valid_num))
        shuffled = shuffle(class_data)
        for x in range(train_num):
            shutil.copy(os.path.join(DATA_ROOT_DIR, shuffled[x]), train_img_dir)
        for x in range(valid_num):
            shutil.copy(os.path.join(DATA_ROOT_DIR, shuffled[train_num + x]), val_img_dir)


def splitdata_from_validate_to_test(valid_dir, test_dir, ratio):
    '''
    Make a subset of test images from validate images
    '''
    logger.info("splitdata_from_validate_to_test %s --> %s: %s" % (valid_dir, test_dir, ratio))

    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # Get all files in validate dir
    for _, _, images in os.walk(valid_dir):
        total_files = len(images)
        shuffled = shuffle(images)
        logger.info("%s has %s files" % (valid_dir, total_files))
        test_num = math.floor(total_files * ratio)

        for x in range(test_num):
            shutil.move(os.path.join(valid_dir, shuffled[x]), test_dir)


def split_dataset(dataset_name):
    if not DATA_ROOT_DIR:
        print("DATA_ROOT_DIR not def in ENV")
        raise "Env not found error."

    logger.info("Processing data in %s" % DATA_ROOT_DIR)

    if not os.path.exists(DATA_ROOT_DIR):
        logger.info("DATA_ROOT_DIR not exist on filesystem %s" % DATA_ROOT_DIR)

    logger.info(">> handle data on %s" % DATA_ROOT_DIR)

    DATASET_NAME = dataset_name
    DATASET_DIR = os.path.join(DATA_ROOT_DIR, DATASET_NAME)
    if not os.path.exists(DATASET_DIR):
        logger.info("DATASET_DIR not exist on filesystem %s" % DATASET_DIR)

    # generate class labels csv file
    label_class_csv = os.path.join(DATA_ROOT_DIR, "%s.labels.autogen.csv" % DATASET_NAME)

    with open(label_class_csv, "w") as fout:
        fout.writelines(["filepath, label\n"])

    # https://docs.python.org/3/library/pathlib.html#general-properties
    # DATA_PATH_OBJ = Path(DATASET_DIR)
    # subclassfolder = [f.parts[-1] for f in DATA_PATH_OBJ.iterdir() if f.is_dir()]
    subclassfolder = ["train"]

    output_lines = []
    for x in subclassfolder:
        target_images_folder = os.path.join(DATASET_DIR, x)
        for root, dirs, images in os.walk(target_images_folder):
            print("root", root)
            print("dirs", dirs)
            for y in images:
                print("dirname", os.path.dirname(root))
                print("y", y)
                sys.exit()
                output_lines.append("%s/%s/%s, %s\n" % (DATASET_NAME, x, y, x))

    with open(label_class_csv, "a") as fout:
        fout.writelines(output_lines)

    SPLIITED_DATA = os.path.join(DATA_ROOT_DIR, "%s_pp_1" % DATASET_NAME)  # pp is pre-process
    SPLIITED_DATA_TRAIN = os.path.join(SPLIITED_DATA, "train")
    SPLIITED_DATA_VAL = os.path.join(SPLIITED_DATA, "valid")
    SPLIITED_DATA_TEST = os.path.join(SPLIITED_DATA, "test")
    NUM2LABEL_JSON = os.path.join(DATA_ROOT_DIR, "%s.labels.num2label.json" % DATASET_NAME)
    LABEL2NUM_JSON = os.path.join(DATA_ROOT_DIR, "%s.labels.label2num.json" % DATASET_NAME)

    total_data = {}
    total_images_counter = 0
    class_labels = set()
    missing_files = []

    with open(label_class_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                img_filename = row[0].strip()
                img_label = row[1].strip()
                class_labels.add(img_label)
                image_file_abs = os.path.join(DATA_ROOT_DIR, img_filename)
                if not os.path.exists(image_file_abs):
                    logger.info(f"[WARN] file not exist {image_file_abs}")
                    missing_files.append(image_file_abs)
                    continue
                total_images_counter += 1
                if img_label in total_data:
                    total_data[img_label].append(image_file_abs)
                else:
                    total_data[img_label] = [image_file_abs]
            else:
                logger.info(f"[WARN] Insufficient fields in row: {row}")

    logger.info(f"Get all labels {class_labels}")
    logger.info(f"Get images {total_images_counter}")

    #####################
    # Dump lables metadata
    #####################
    with open(NUM2LABEL_JSON, 'w') as f:
        data = {}
        num = 0
        for x in list(class_labels):
            data[num] = x
            num += 1

        json.dump(data, f, ensure_ascii=False, indent=4)

    with open(LABEL2NUM_JSON, 'w') as f:
        data = {}
        num = 0
        for x in list(class_labels):
            data[x] = num
            num += 1

        json.dump(data, f, ensure_ascii=False, indent=4)

    if os.path.exists(SPLIITED_DATA):
        shutil.rmtree(SPLIITED_DATA)

    os.mkdir(SPLIITED_DATA)
    os.mkdir(SPLIITED_DATA_TRAIN)
    os.mkdir(SPLIITED_DATA_VAL)

    for x in class_labels:
        move_images_to_train_and_valid(
            total_data, [x], os.path.join(
                SPLIITED_DATA_TRAIN, x), os.path.join(
                SPLIITED_DATA_VAL, x))

    logger.info("DATA_TEST_RECORDS_SPLIT is %s" % DATA_TEST_RECORDS_SPLIT)
    if DATA_TEST_RECORDS_SPLIT == "true":
        os.mkdir(SPLIITED_DATA_TEST)
        for x in class_labels:
            splitdata_from_validate_to_test(
                os.path.join(
                    SPLIITED_DATA_VAL, x), os.path.join(
                    SPLIITED_DATA_TEST, x), DATA_TEST_RECORDS_RATIO)
    elif os.path.exists(os.path.join(DATASET_DIR, "test")):
        if os.path.exists(SPLIITED_DATA_TEST):
            os.removedirs(os.mkdir(SPLIITED_DATA_TEST))
        shutil.copytree(os.path.join(DATASET_DIR, "test"), SPLIITED_DATA_TEST)


##########################################################################
# Testcases
##########################################################################
import unittest


# run testcase: python
# /d/git/Sports-Image-Classification-YOLO-ResNet/src/preprocess_data.py
# Test.testExample
class Test(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_splitdata(self):
        print("test_splitdata")
        dataset_name = ENV.get("DATASET_NAME", None)

        if not dataset_name:
            raise BaseException("ENV DATASET_NAME not exist")

        dataset_path = os.path.join(DATA_ROOT_DIR, dataset_name)

        if not os.path.exists(dataset_path):
            raise BaseException(f"${dataset_path} not found")

        split_dataset(dataset_name)

    def splitdata_sample4(self):
        print("splitdata_sample4")
        split_dataset("sample4")

    def splitdata_sample7(self):
        print("splitdata_sample7")
        split_dataset("sample7")

    def splitdata_plantvillage(self):
        print("splitdata_plantvillage")
        split_dataset("PlantVillage")


def test():
    unittest.main()


def main():
    test()


if __name__ == '__main__':
    main()
