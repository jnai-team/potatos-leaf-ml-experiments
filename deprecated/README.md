# Potatos Leaf ML Experiments
马铃薯病害叶片分类 实验

实验项目地址：[GitHub](https://github.com/jnai-team/potatos-leaf-ml-experiments) | [Gitee](https://gitee.com/jnai/potatos-leaf-ml-experiments)

实验用数据集：[GitHub](https://github.com/jnai-team/potato-diseases-dataset) | [Gitee](https://gitee.com/jnai/potato-diseases-dataset)

## Install deps

前提条件：

* Python ~3.11 (3.9 - 3.11；不要使用最新的 Python 3.13, 较多兼容性错误)
* Pip

## Clone code

将代码下载到 `ROOT_DIR` （ROOT_DIR 是自定义的路径，比如 C:\git）目录，在命令行[^20250712112234]中执行下面命令：

```bash
# 在 Git Bash 中执行
cd $ROOT_DIR
git clone git@gitee.com:jnai/potatos-leaf-ml-experiments.git
```

继续安装 pip 依赖：

```bash
# 在 Git Bash 中执行
cd $ROOT_DIR/potatos-leaf-ml-experiments # 进入刚刚下载好的代码地址
./bin/000.install_deps.sh   # 安装 Python 依赖
```

# Data

## Download data

下载数据集 Sample7 到根目录 `DATA_ROOT_DIR`（DATA_ROOT_DIR 是自定义的路径）：

```
cd $DATA_ROOT_DIR
git clone git@gitee.com:jnai/potato-diseases-dataset.git
```

所以，数据集的路径就是：
* Development Data - $DATA_ROOT_DIR/potato-diseases-dataset/train
* Test data - $DATA_ROOT_DIR/potato-diseases-dataset/test

## Pre-process image data for development

数据预处理，因为现在都是做图片分类。用哪个数据集，数据集位置，超参数等，使用配置文件 `.env` 进行配置，示例 sample.env

复制配置文件。

```
cd $ROOT_DIR
cp sample.env .env
```

然后使用 VSCode 或 Notepad 打开 .env 文件，编辑文件，设置：`DATA_ROOT_DIR` 和 `DATASET_NAME`，比如：

```
DATA_ROOT_DIR=D:\packages\potato-datasets
DATASET_NAME=potato-diseases-dataset
DATA_TRAIN_RECORDS_RATIO=0.9
```

那么，程序就会认为：

* Development Data - D:\packages\potato-datasets\potato-diseases-dataset\train
* Test Data - D:\packages\potato-datasets\potato-diseases-dataset\test

Run script to split data.

```
./bin/001.preprocess_data_sample7.sh
```

After that, a new folder is generated at

```
$DATA_ROOT_DIR/sample7_pp_1 ## pp_1 means pre-process phase 1
```

# Train and predict

Config model trainer in `.env`, by default they are 

```
MODEL_TRAIN_SCRIPT=resnet/model50_train.py
MODEL_PREDICT_SCRIPT=resnet/model50_predict.py
```

Scripts are all placed under `src`.

## Train model

Run script to train model.
```
bin/002.train_model.sh
```

After the training, model files are saved into a dir in `DATA_ROOT_DIR`, e.g.

![alt text](./assets/media/1742894625155.png)

The dirname `2025_03_25_170538` is `MODEL_ID` which is used later.

## Predict model

First, edit `.env` file again, add following info.

```
MODEL_ID=2025_03_24_170324
PREDICT_TARGETS=C:\experiments\data\potato-datasets\potato-diseases-dataset\test\Potato___Early_blight,C:\experiments\data\potato-datasets\potato-diseases-dataset\test\Potato___healthy,C:\experiments\data\potato-datasets\potato-diseases-dataset\test\Potato___Late_blight
```

`PREDICT_TARGETS` stores folders concatenating with comma, each folder contains images, and the images are assumed labeled as their folder name, another sytanx of `PREDICT_TARGETS` is also supported -

```
PREDICT_TARGETS=FolderA#LabelA,FolderB#LabelB[...]
```

After setting `PREDICT_TARGETS` in `.env`, run the following script to get predict result -

```
bin/003.predict_images.sh
```

Check out the console log for predict details.


![alt text](./assets/media/1742895300964.png)

# Licesne
[LICENSE](./LICENSE)

# Refs

[^20250712112234]: Windows 比如 Git Bash，Linux Bash Shell.