#!/usr/bin/env python
"""
Converts dataset from folder-subfolder format (one subfolder per class) to
tortilla's data format
"""

import glob
import os
from PIL import Image
import shutil
from utils import *
import numpy as np
import uuid
import json
import random


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
			description="Converts dataset from folder-subfolder format \
						(one sub folder per class) to tortilla's data format")

	parser.add_argument('--input-folder-path', action='store', dest='input_folder_path',
						required=True,
						help='Path to input folder containing images')
	parser.add_argument('--output-folder-path', action='store', dest='output_folder_path',
						required=True,
						help='Path to output folder to write images')
	parser.add_argument('--min-images-per-class', action='store', dest='min_images_per_class',
						default=500,
						help='Minimum number of images required per class')
	parser.add_argument('--train-percent', action='store', dest='train_percent',
						default=0.8,
						type=float,
						help='Percentage of all images to use as training data')
	parser.add_argument('--dataset-name', action='store', dest='dataset_name',
						required=True,
						help='Name of the Dataset')
	parser.add_argument('--img-size', action='store', dest='img_size',
						default="256x256",
						help='Size of the target images')
	parser.add_argument('--absolute_path', dest='absolute_path', action='store_true')

	args = parser.parse_args()

	input_folder_path = args.input_folder_path
	output_folder_path = args.output_folder_path
	min_images_per_class = args.min_images_per_class
	train_percent = args.train_percent
	dataset_name = args.dataset_name
	img_size = (int(args.img_size.split("x")[0]), int(args.img_size.split("x")[1]))
	absolute_path = args.absolute_path

	"""
	Validation Input and Output Folder
	"""
	classes = get_classes_from_input_folder(input_folder_path,non_interactive_mode=False)
	output_folder_path_validation(output_folder_path, classes, non_interactive_mode =False)


	_message = """
		Input Folder Path : {}
		Output Folder Path : {}
		Minimum Images per Class : {}
		Train Percentage : {}
		Dataset Name : {}
		Target Image Size : {}
		Number of Classes : {}

		Proceed with these parameters ?
	""".format(
		input_folder_path,
		output_folder_path,
		min_images_per_class,
		train_percent,
		dataset_name,
		img_size,
		len(classes)
	)

	response = query_yes_no(_message, default="yes")
	if not response:
		exit(0)

	"""
	Actual Preprocessing starts
	"""
	train_class_frequency = {}
	val_class_frequency = {}
	for _class in classes:
		train_class_frequency[_class] = 0
		val_class_frequency[_class] = 0

	rough_class_frequency = quick_compute_class_frequency_from_folder(
				input_folder_path, classes
	)
	train_list = []
	val_list = []
	error_list = []

	files = glob.glob(input_folder_path+"/*/*")
	random.shuffle(files)

	for _idx, _file in enumerate(files):
		print("Processing {}/{} :: {}".format(str(_idx), str(len(files)), _file))
		_class = _file.split("/")[-2]

		# Open, Preprocess and write file to output_folder_path
		try:
			# TODO: Make this opening of the file optional
			im = Image.open(_file)
			im = im.resize(img_size)
		except Exception as e:
			error_list.append((_file, str(_class), str(e)))
			continue

		is_train = np.random.rand() <= train_percent

		target_file_name = "{}_{}".format(
			str(uuid.uuid4()),
			"_".join(_file.split("/")[-1].split())
			)
		# Absolute Path
		target_file_path = os.path.abspath(os.path.join(
			output_folder_path,
			"images",
			_class,
			target_file_name
		))
		# File path relative to the images root
		target_file_path_rel = os.path.join(
			"images",
			_class,
			target_file_name
		)

		im.save(target_file_path)

		# Conditionally save absolute paths to the file
		# Useful when designing multiple experiments on the same dataset
		if not absolute_path:
			target_file_path = target_file_path_rel

		if is_train:
			train_list.append((target_file_path, str(classes.index(_class))))
			train_class_frequency[_class] += 1
		else:
			val_list.append((target_file_path, str(classes.index(_class))))
			val_class_frequency[_class] += 1

	"""
	Aggregate results
	"""
	_meta = {}
	_meta["dataset_name"] = dataset_name
	_meta["train_percent"] = train_percent
	_meta["input_folder_path"] = os.path.abspath(input_folder_path)
	_meta["output_folder_path"] = os.path.abspath(output_folder_path)
	_meta["min_images_per_class"] = min_images_per_class
	_meta["img_size"] = img_size
	_meta["total_images"] = len(train_list) + len(val_list)
	_meta["errors"] = len(error_list)
	_meta["train_class_frequency"] = train_class_frequency
	_meta["val_class_frequency"] = val_class_frequency
	_meta["absolute_path"] = absolute_path

	# Write meta file
	f = open(os.path.join(
		output_folder_path,
		"meta.json"
	), "w")
	f.write(json.dumps(
					_meta,
					sort_keys=True,
					indent=4,
					separators=(',', ': ')
					))

	# Write classes.txt
	f = open(os.path.join(
		output_folder_path,
		"classes.txt"
	), "w")
	f.write("\n".join(classes))

	# Write train.json
	f = open(os.path.join(
		output_folder_path,
		"train.json"
	), "w")
	_train = {}
	_train = {item[0]: item[1] for item in train_list}
	f.write(json.dumps(
					_meta,
					sort_keys=True,
					indent=4,
					separators=(',', ': ')
					))

	# Write val.json
	f = open(os.path.join(
		output_folder_path,
		"val.json"
	), "w")
	_val = {}
	_val = {item[0]: item[1] for item in val_list}
	f.write(json.dumps(
					_val,
					sort_keys=True,
					indent=4,
					separators=(',', ': ')
					))

	# Write errors.txt
	f = open(os.path.join(
		output_folder_path,
		"error.txt"
	), "w")
	error_list = ["\t".join(x) for x in error_list]
	f.write("\n".join(error_list))
