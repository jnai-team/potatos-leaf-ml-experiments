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
import random
import csv
# Get ENV
from env import ENV
from common.utils import console_log

ROOT_DIR = os.path.join(curdir, os.pardir)
DATA_ROOT_DIR = ENV.str("DATA_ROOT_DIR", None)
DATA_TRAIN_RECORDS_RATIO = float(ENV.str("DATA_TRAIN_RECORDS_RATIO", "0.9"))


def move_images_1(total_data_1,labels, train_img_dir, val_img_dir):
    for label in labels:
        class_data_1 = total_data_1[label]
        console_log(">> Get records %d for label %s" % (len(class_data_1), label))
        train_num = math.floor(len(class_data_1) * DATA_TRAIN_RECORDS_RATIO)
        valid_num = len(class_data_1) - train_num
        console_log(">> train_num %d, valid_num %d" % (train_num, valid_num))
        random.shuffle(class_data_1)
        for x in range(train_num):
            shutil.copy(os.path.join(DATA_ROOT_DIR, class_data_1[x]), train_img_dir)
        for x in range(valid_num):
            shutil.copy(os.path.join(DATA_ROOT_DIR, class_data_1[train_num + x]), val_img_dir)

            
def splitdata_1():
    if not DATA_ROOT_DIR:
        print("DATA_ROOT_DIR not def in ENV")
        raise "Env not found error."

    console_log("Processing data in %s" % DATA_ROOT_DIR)

    if not os.path.exists(DATA_ROOT_DIR):
        console_log("DATA_ROOT_DIR not exist on filesystem %s" % DATA_ROOT_DIR)

    console_log(">> handle data on %s" % DATA_ROOT_DIR)

    CLASS_LABELS_1 = os.path.join(DATA_ROOT_DIR, "train.csv")
    SPLIITED_DATA1 = os.path.join(DATA_ROOT_DIR, "splitted_data_1")
    SPLIITED_DATA1_TRAIN = os.path.join(SPLIITED_DATA1, "train")
    SPLIITED_DATA1_VAL = os.path.join(SPLIITED_DATA1, "valid")
    SPLIITED_DATA1_TEST = os.path.join(SPLIITED_DATA1, "test")

    # 处理splitted_data_1
    total_data_1 = {}
    total_images_counter_1 = 0
    class_labels_1 = set()
    missing_files_1 = []

    with open(CLASS_LABELS_1, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                img_filename = row[0].strip()
                img_label = row[1].strip()
                class_labels_1.add(img_label)
                image_file_abs = os.path.join(DATA_ROOT_DIR, img_filename)
                if not os.path.exists(image_file_abs):
                    console_log(f"[WARN] file not exist {image_file_abs}")
                    missing_files_1.append(image_file_abs)
                    continue
                total_images_counter_1 += 1
                if img_label in total_data_1:
                    total_data_1[img_label].append(img_filename)
                else:
                    total_data_1[img_label] = [img_filename]
            else:
                console_log(f"[WARN] Insufficient fields in row: {row}")

    console_log(f"Get all labels {class_labels_1}")
    console_log(f"Get images {total_images_counter_1}")

    if os.path.exists(SPLIITED_DATA1):
        shutil.rmtree(SPLIITED_DATA1)

    os.mkdir(SPLIITED_DATA1)
    os.mkdir(SPLIITED_DATA1_TRAIN)
    os.mkdir(SPLIITED_DATA1_VAL)
    os.mkdir(SPLIITED_DATA1_TEST)

    for x in class_labels_1:
        os.mkdir(os.path.join(SPLIITED_DATA1_TRAIN, x))
        os.mkdir(os.path.join(SPLIITED_DATA1_VAL, x))

    for x in class_labels_1:
        move_images_1(total_data_1, [x], os.path.join(SPLIITED_DATA1_TRAIN, x), os.path.join(SPLIITED_DATA1_VAL, x))


def move_images_2(total_data_2, labels, train_img_dir, val_img_dir):
    for label in labels:
        class_data_2 = total_data_2[label]
        train_num = math.floor(len(class_data_2) * DATA_TRAIN_RECORDS_RATIO)
        valid_num = len(class_data_2) - train_num
        console_log(">> Get records %d for label %s" % (len(class_data_2), label))
        console_log(">> train_num %d, valid_num %d" % (train_num, valid_num))
        import random
        random.shuffle(class_data_2)

        for x in range(train_num):
            file_path = class_data_2[x]
            print(f"当前处理的文件路径: {file_path}")
            # 构建完整的训练集目标路径，包含类别和百分比子文件夹
            full_train_path = os.path.join(train_img_dir, label, os.path.basename(os.path.dirname(file_path)))
            if not os.path.exists(full_train_path):
                os.makedirs(full_train_path)
            shutil.copy(file_path, full_train_path)

        for x in range(valid_num):
            file_path = class_data_2[valid_num + x]
            print(f"当前处理的文件路径: {file_path}")
            # 构建完整的验证集目标路径，包含类别和百分比子文件夹
            full_val_path = os.path.join(val_img_dir, label, os.path.basename(os.path.dirname(file_path)))
            if not os.path.exists(full_val_path):
                os.makedirs(full_val_path)
            shutil.copy(file_path, full_val_path)


def splitdata_2():
    if not DATA_ROOT_DIR:
        print("DATA_ROOT_DIR not def in ENV")
        raise "Env not found error."

    console_log("Processing data in %s" % DATA_ROOT_DIR)

    if not os.path.exists(DATA_ROOT_DIR):
        console_log("DATA_ROOT_DIR not exist on filesystem %s" % DATA_ROOT_DIR)

    console_log(">> handle data on %s" % DATA_ROOT_DIR)

    CLASS_LABELS_2 = os.path.join(DATA_ROOT_DIR, "train_1.csv")
    SPLIITED_DATA2 = os.path.join(DATA_ROOT_DIR, "splitted_data_2")
    SPLIITED_DATA2_TRAIN = os.path.join(SPLIITED_DATA2, "train")
    SPLIITED_DATA2_VAL = os.path.join(SPLIITED_DATA2, "valid")
    SPLIITED_DATA2_TEST = os.path.join(SPLIITED_DATA2, "test")

    total_data_2 = {}
    total_images_counter_2 = 0
    class_labels_2 = set()
    missing_files_2 = []

    with open(CLASS_LABELS_2, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                img_filename = row[0].strip()
                img_label = row[1].strip()
                class_labels_2.add(img_label)
                image_file_abs = os.path.join(DATA_ROOT_DIR, img_filename)
                if not os.path.exists(image_file_abs):
                    console_log(f"[WARN] file not exist {image_file_abs}")
                    missing_files_2.append(image_file_abs)
                    continue
                total_images_counter_2 += 1
                if img_label in total_data_2:
                    total_data_2[img_label].append(image_file_abs)
                else:
                    total_data_2[img_label] = [image_file_abs]
            else:
                console_log(f"[WARN] Insufficient fields in row: {row}")

    console_log(f"Get all labels {class_labels_2}")
    console_log(f"Get images {total_images_counter_2}")

    if os.path.exists(SPLIITED_DATA2):
        shutil.rmtree(SPLIITED_DATA2)

    os.mkdir(SPLIITED_DATA2)
    os.mkdir(SPLIITED_DATA2_TRAIN)
    os.mkdir(SPLIITED_DATA2_VAL)
    os.mkdir(SPLIITED_DATA2_TEST)

    for x in class_labels_2:
        for subfolder in ['10%', '20%', '30%', '40%', '50%', '70%','more_than_70%']:
            # 这里不再单独创建中间层次的文件夹，而是在移动时按需创建
            pass

    for x in class_labels_2:
        class_data_2 = total_data_2[x]
        move_images_2(total_data_2, [x], SPLIITED_DATA2_TRAIN, SPLIITED_DATA2_VAL)
##########################################################################
# Testcases
##########################################################################
import unittest


# run testcase: python /d/git/Sports-Image-Classification-YOLO-ResNet/src/preprocess_data.py Test.testExample
class Test(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_splitdata(self):
        print("test_splitdata")
        splitdata_1()

    def test_splitdata2(self):
        print("test_splitdata2")
        splitdata_2()


def test():
    unittest.main()


def main():
    test()


if __name__ == '__main__':
    main()