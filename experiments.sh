#!/bin/bash


for i in "datasets"/*; do

	export i=$(echo $i | sed 's/datasets\///g')
	echo $i
	if [ -d datasets/$i ]; then

		python ~/tortilla/tortilla-train.py \
		--dataset-dir "datasets/$i" \
		--experiments-dir "/home/harsh/experiments" \
		--experiment-name "$i"  \
		--no-render-images   \
		--plot-platform visdom

fi
done
