#!/bin/bash

if [ "$1" = "" ];
then
    echo "Invalid argument"
    exit 3
elif [ "$2" = "" ];
then
    echo "Invalid argument"
    exit 3
fi

dir=$1
path=$2

mkdir $dir
python scripts/misc/prepare_datasets_recursive.py --input-folder-path=$path --output-folder-path=output1234
cd output1234
for f in *
  do
    python ../scripts/data_preparation/prepare_data.py --input-folder-path=$f/ --output-folder-path=../$dir/$f/ --dataset-name=$f --non-interactive-mode
done
cd ..
rm -r output1234
