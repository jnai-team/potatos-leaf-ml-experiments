import os,glob
import csv

DATA_ROOT_DIR = 'data'

class_to_num = {}
num_to_class = {}

class_name_list = os.listdir(os.path.join(DATA_ROOT_DIR, 'sample6'))
for index,class_name in enumerate(class_name_list):
    class_to_num[class_name] = index
    num_to_class[index] = class_name

image_dir = []
for class_name in class_name_list:
    image_dir += glob.glob(os.path.join(DATA_ROOT_DIR, 'sample6',class_name,'*.JPG'))
    subfolders = ['10%', '20%', '30%','40%','50%','60%','70%','more_than_70%']
    for subfolder in subfolders:
        subfolder_path = os.path.join(DATA_ROOT_DIR, 'train/sample6', class_name, subfolder)
        if os.path.exists(subfolder_path):
            image_dir += glob.glob(os.path.join(subfolder_path, '*.JPG'))
import random
random.shuffle(image_dir)

with open ('train_1.csv',mode='w',newline='')as f:
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