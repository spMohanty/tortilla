#!/bin/bash

mkdir output
mkdir my_food_repo
python scripts/misc/prepare_datasets_recursive.py --input-folder-path=/mount/SDE/instagram/myfoodrepo-images/myfoodrepo-images/images --output-folder-path=output
cd output
for f in *
  do
    python ../scripts/data_preparation/prepare_data.py --input-folder-path=$f/ --output-folder-path=../my_food_repo/$f/ --dataset-name=$f --non-interactive-mode --min-images-per-class=100
done
