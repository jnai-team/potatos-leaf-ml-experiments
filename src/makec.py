import os,glob
import csv

# 假设 DATA_ROOT_DIR 为数据根目录，与原代码保持一致
DATA_ROOT_DIR = 'data'

class_to_num = {}
num_to_class = {}

class_name_list = os.listdir(os.path.join(DATA_ROOT_DIR, 'train/sample4'))
for index,class_name in enumerate(class_name_list):
    class_to_num[class_name] = index
    num_to_class[index] = class_name

image_dir = []
for class_name in class_name_list:
    image_dir += glob.glob(os.path.join(DATA_ROOT_DIR, 'train/sample4',class_name,'*.png'))

import random
random.shuffle(image_dir)

with open ('train.csv',mode='w',newline='')as f:
    writer = csv.writer(f)
    for image in image_dir:
        # 这里将相对路径转换为相对于 DATA_ROOT_DIR 的路径
        relative_path = os.path.relpath(image, DATA_ROOT_DIR)
        print("relative_path:",relative_path)
        parts = relative_path.split(os.sep)
        print("split parts:",parts)
        class_name = parts[2]

        label =  num_to_class[class_to_num[class_name]]
        writer.writerow([relative_path, label])
        
print('write finished') 