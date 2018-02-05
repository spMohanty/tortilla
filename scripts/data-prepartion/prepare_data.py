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

    parser = argparse.ArgumentParser(
            description="Converts dataset from folder-subfolder format \
                        (one sub folder per class) to tortilla's data format")
    parser.add_argument('--input_folder_path',
                        metavar="input_folder_path",
                        type=str,
                        nargs = '+',
                        help='Path to input folder')
    parser.add_argument('--output_folder_path',
                        metavar="output_folder_path",
                        type=str,
                        nargs = '+',
                        help='Path to input folder')
    parser.add_argument('--min_images_per_class',
                        metavar='min_images_per_class',
                        nargs = 1,
                        default=500,
                        type=int,
                        help='Minimum number of images required per class')
    parser.add_argument('--train_percent',
                        metavar='train_percent',
                        nargs = 1,
                        default=0.8,
                        type=float,
                        help='Percentage of all images to use as training data')
    parser.add_argument('--dataset_name',
                        metavar='dataset_name',
                        nargs = 1,
                        required=True,
                        type=str,
                        help='Name of the Dataset')
    parser.add_argument('--img_size',
                        metavar='img_size',
                        nargs = 1,
                        default="256x256",
                        type=str,
                        help='Size of the target images')

    args = parser.parse_args()

    input_folder_path = args.input_folder_path[0]
    output_folder_path = args.output_folder_path[0]
    min_images_per_class = args.min_images_per_class
    train_percent = args.train_percent
    dataset_name = args.dataset_name
    img_size = (int(args.img_size.split("x")[0]), int(args.img_size.split("x")[1]))

    classes = os.listdir(input_folder_path)
    """
    Validation
    """
    input_folder_path_validation(input_folder_path)
    output_folder_path_validation(output_folder_path, classes)

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
        is_train = np.random.rand() <= train_percent

        target_file_name = "{}_{}".format(
            str(uuid.uuid4()),
            _file.split("/")[-1]
            )
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

        # Open, Preprocess and write file to output_folder_path
        try:
            # TODO: Make this opening of the file optional
            im = Image.open(_file)
            im = im.resize(img_size)
        except Exception as e:
            error_list.append((_file, str(_class), str(e)))

        im.save(target_file_path)

        if is_train:
            train_list.append((target_file_path_rel, str(classes.index(_class))))
            train_class_frequency[_class] += 1
        else:
            val_list.append((target_file_path_rel, str(classes.index(_class))))
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

    # Write meta file
    f = open(os.path.join(
        output_folder_path,
        "meta.json"
    ), "w")
    f.write(json.dumps(_meta))

    # Write classes.txt
    f = open(os.path.join(
        output_folder_path,
        "classes.txt"
    ), "w")
    f.write("\n".join(classes))

    # Write train.txt
    f = open(os.path.join(
        output_folder_path,
        "train.txt"
    ), "w")
    train_list = ["\t".join(x) for x in train_list]
    f.write("\n".join(train_list))

    # Write val.txt
    f = open(os.path.join(
        output_folder_path,
        "val.txt"
    ), "w")
    val_list = ["\t".join(x) for x in val_list]
    f.write("\n".join(val_list))

    # Write errors.txt
    f = open(os.path.join(
        output_folder_path,
        "error.txt"
    ), "w")
    error_list = ["\t".join(x) for x in error_list]
    f.write("\n".join(error_list))
