#!/bin/bash



for i in "DatasetStudentFiltration"/*; do

	export i=$(echo $i | sed "s/DatasetStudentFiltration\///g")
	if [ -d DatasetStudentFiltration/$i ]; then
		echo $i
		python ~/tortilla/tortilla-train.py \
		--dataset-dir "DatasetStudentFiltration/$i" \
		--experiments-dir "/home/harsh/experiments" \
		--experiment-name "sf$i"  \
		--no-render-images   \
		--plot-platform visdom \
		--checkpoint-frequency 3\

		#python ~/tortilla/tortilla-train.py \
		#--dataset-dir "DatasetStudentFiltration/$i" \
		#--experiments-dir "/home/harsh/experiments" \
		#--experiment-name "sfwl$i"  \
		#--no-render-images   \
		#--plot-platform visdom \
		#--wloss \
		#--checkpoint-frequency 3 
	

fi
done
