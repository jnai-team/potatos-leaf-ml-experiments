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
cd $baseDir/..

if [ ! -f .env ]; then
    echo `pwd`"/.env file not found"
    exit 1
fi

source .env

if [ -z ${MODEL_PREDICT_SCRIPT+x} ]; then echo "ERROR, MODEL_PREDICT_SCRIPT is not defined"; exit 2; else echo "MODEL_PREDICT_SCRIPT is set to '$MODEL_PREDICT_SCRIPT'"; fi

if [ -z ${MODEL_PREDICT_SCRIPT} ]; then echo "ERROR, MODEL_PREDICT_SCRIPT is not defined"; exit 2; fi

if [ ! -d tmp ]; then
    mkdir tmp
fi

cd $baseDir/../src

if [ ! -f $MODEL_PREDICT_SCRIPT ]; then
    echo "$MODEL_PREDICT_SCRIPT not found"
    exit 3
fi

echo ">> Predicting is started"
python $MODEL_PREDICT_SCRIPT
echo "<< DONE"