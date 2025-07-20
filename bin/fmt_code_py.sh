#! /bin/bash 
###########################################
# pip install autopep8==1.5
###########################################

# constants
baseDir=$(cd -P `dirname "$0"`;pwd)
cwdDir=$PWD
export PYTHONUNBUFFERED=1
export TS=$(date +%Y%m%d%H%M%S)
export DATE=`date "+%Y%m%d"`
export DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"` #add %3N as we want millisecond too
# export AUTOPEP8_CMD=/c/devel/Python/Python37/Scripts/autopep8
export AUTOPEP8_CMD=autopep8
export AUTOPEP8_OPTS="--in-place --aggressive --aggressive -r -v --ignore E226,E24,W50,W690,E402 --max-line-length 99"

# functions
function printUsage(){
    echo "Usage $0 PATH_DIR_OR_FILE"
    echo "default $0 `PWD`"
}

function formatDir(){
    cd $1
    for x in `find . -name "*.py" -not -path "./venv/*" -not -path "./tmp/*" -not -path "*/__pycache__/*"`; do 
    echo "Python script -->" $x
    $AUTOPEP8_CMD $AUTOPEP8_OPTS $x
    echo ""
    echo ""
done
}

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
cd $cwdDir
echo ">> Working in" `pwd`
printUsage
sleep 3

if [ $# -gt 0 ]; then
    if [ -f $1 ]; then
        $AUTOPEP8_CMD $AUTOPEP8_OPTS $1
    elif [ -e $1 ]; then
        formatDir $1
    else
        printUsage
        exit 1
    fi
else
    formatDir $cwdDir
fi