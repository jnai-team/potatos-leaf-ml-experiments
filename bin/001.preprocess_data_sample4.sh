#! /bin/bash 
###########################################
#
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
cwdDir=$PWD
export PYTHONUNBUFFERED=1
export PATH=/opt/miniconda3/envs/venv-py3/bin:$PATH
export TS=$(date +%Y%m%d%H%M%S)
export DATE=`date "+%Y%m%d"`
export DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"` #add %3N as we want millisecond too

# functions

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
source $baseDir/../.env

cd $baseDir/../src
python preprocess_data.py Test.splitdata_sample4

# if [ ! -d $DATA_ROOT_DIR/splitted_data_1/test/images ]; then
#   mkdir -p $DATA_ROOT_DIR/splitted_data_1/test/images
# fi

# cp -rf $baseDir/../assets/testing_images/images/*.jpg $DATA_ROOT_DIR/splitted_data_1/test/images