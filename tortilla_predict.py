#!/usr/bin/env python

import glob
import os
import json
import argparse
import multiprocessing
from PIL import Image
from tqdm import tqdm
from utils import *
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

from models import TortillaModel


def check_args(model_path, pred_dir):

    if not model_path.endswith(".net"):
        exit('Model path does not correspond to a model.')

    images= glob.glob(os.path.join(pred_dir,"*"))
    stop = False;
    while not stop:
        for _idx, _image in enumerate(images):
            try:
                im=Image.open(_image)
                stop = True;
                break
            except:
                if _idx == len(images)-1:
                    exit('Prediction directory does not contain valid images.')
                else:
                    continue

def preprocess(im, transf):
    if transf:
        preprocessing = transf
    else:
        preprocessing =transforms.Compose([
            transforms.ToTensor()])
    im_tensor = preprocessing(im)
    return im_tensor

def predict(model_path,prediction_dir):
    
    """
    Check arguments
    """
    check_args(model_path, prediction_dir)
    print(prediction_dir)	
    """
    Load Model
    """
    state_dict = torch.load(model_path)

    if state_dict["use_cpu"]:
        use_gpu = False
    else:
        use_gpu = torch.cuda.is_available()

    model_type = state_dict["model"]
    experiments_dir = state_dict["exp_dir_name"]
    classes = state_dict["classes"]
    transf = state_dict["transforms"]

    model = TortillaModel(model_type, classes)

    if use_gpu:
        # use GPU for both training and prediction
        net = torch.nn.DataParallel(model.net)
        net.load_state_dict(state_dict["model_state_dict"])
        net.cuda()
    elif state_dict["use_cpu"]:
        # use CPU for both training and prediction
        net = model.net
        net.load_state_dict(state_dict["model_state_dict"])
    else:
        # use GPU for training but CPU for prediction
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
    prediction = {}
    error_list = []

    images= glob.glob(os.path.join(prediction_dir,"*"))

    for _idx, _image in enumerate(tqdm(images)):
        try:
            im = Image.open(_image)
            im_tensor = preprocess(im, transf)
            im_tensor.unsqueeze_(0)
            image = Variable(im_tensor)
            if use_gpu:
                image = image.cuda()
            outputs= net(image)
            _, predicted = torch.max(outputs.data, 1)
            prediction[_image]=classes[int(predicted)]
        except Exception as e:
            error_list.append((_image, str(e)))

    """
    Create Prediction Folder and write predictions
    """

    path = os.path.join(experiments_dir,"predictions")
    if os.path.exists(path):
    #    response = query_yes_no(
    #                "Predictions Folder seems to exist, do you want to overwrite ?",
    #               default='no')
        shutil.rmtree(path)
    #   if response:
    #        shutil.rmtree(path)
    #    else:
    #        print("Exiting, because prediction path exists and cannot be deleted.")
    #        exit('No deletion of Predictions Folder')
    
    os.mkdir(path)

    # Write prediction file
    f = open(os.path.join(path,"prediction.json"),"w")
    f.write(json.dumps(
					prediction,
					sort_keys=True,
					indent=4,
					separators=(',', ': ')
					))

	# Write errors.txt
    f = open(os.path.join(path,"error.txt"), "w")
    error_list = ["\t".join(x) for x in error_list]
    f.write("\n".join(error_list))

    print("Finished! Find your predictions at : ", os.path.join(path,"prediction.json"))
	
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path', action='store', dest='model_path',
                        required=True, help='Path of the saved model')
    parser.add_argument('--prediction-dir', action='store', dest='prediction_dir',
                        required=True, help='Directory of the images for prediction')
    args = parser.parse_args()
    model_path = args.model_path
    prediction_dir = args.prediction_dir

	
    predict(model_path,prediction_dir)
