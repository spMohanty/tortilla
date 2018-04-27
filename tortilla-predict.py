#!/usr/bin/env python

import glob
import os
import json
import argparse
import multiprocessing
from PIL import Image
from tqdm import tqdm
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

from models import TortillaModel


def preprocess(im, transf):
    if transf:
        preprocessing = transf
    else:
        preprocessing =transforms.Compose([
            transforms.ToTensor()])
    im_tensor = preprocessing(im)
    return im_tensor



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path', action='store', dest='model_path',
                        required=True, help='Path of the saved model')
    parser.add_argument('--prediction-dir', action='store', dest='prediction_dir',
                        required=True, help='Directory of the images for prediction')

    args = parser.parse_args()
    model_path = args.model_path
    prediction_dir = args.prediction_dir

    """
    Load Model
    """
    state_dict = torch.load(model_path)

    if state_dict["config"].use_cpu:
        use_gpu = False
    else:
        use_gpu = torch.cuda.is_available()

    model_type = state_dict["model"]
    experiments_dir = state_dict["exp_dir_name"]
    classes = state_dict["classes"]
    transf = state_dict["transforms"]

    model = TortillaModel(model_type, classes)
    if use_gpu:
        net = torch.nn.DataParallel(model.net)
        net.load_state_dict(state_dict["model_state_dict"])
        net.cuda()
    else:
        model_state_dict = OrderedDict()
        for k, v in state_dict["model_state_dict"].items():
            name = k[7:] # remove module.
            model_state_dict[name] = v
        net = model.net
        net.load_state_dict(model_state_dict)

    net.avgpool = nn.AdaptiveAvgPool2d(1)
    net.eval()

    """
    Predict
    """
    prediction ={}

    images= glob.glob(os.path.join(prediction_dir,"*"))
    for _idx, _image in enumerate(tqdm(images)):
        im = Image.open(_image)
        im_tensor = preprocess(im, transf)
        im_tensor.unsqueeze_(0)
        image = Variable(im_tensor)
        if use_gpu:
            image = image.cuda()
        outputs= net(image)
        _, predicted = torch.max(outputs.data, 1)
        prediction[_image]=classes[int(predicted)]


    # Write prediction file
    path = os.path.join(experiments_dir,"prediction.json")
    print("Writing predictions at : ", path)
    f = open(path,"w")
    f.write(json.dumps(
					prediction,
					sort_keys=True,
					indent=4,
					separators=(',', ': ')
					))
