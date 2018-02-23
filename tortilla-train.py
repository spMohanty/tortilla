#!/usr/bin/env python

from config import Config as config
from data_loaders import TortillaDataset
from trainer import TortillaTrainer
from models import TortillaModel

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler

from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.nn import CrossEntropyLoss

from monitor import TortillaMonitor
import utils
import os, shutil
import pickle
import tqdm
tqdm.monitor_interval = 0

import argparse

def main(config):
	if config.use_cpu:
		use_gpu = False
	else:
		use_gpu = torch.cuda.is_available()

	utils.create_directory_structure(config.experiment_dir_name)
	"""
	Initialize Dataset
	"""
	dataset = TortillaDataset(	config.dataset_dir,
								batch_size=config.batch_size,
								num_cpu_workers=config.num_cpu_workers,
								debug=config.debug
								)
	"""
	Initialize Model
	"""
	model = TortillaModel("resnet50", dataset.classes)
	net = model.net

	# Make net use parallel gpu
	if use_gpu:
		net = nn.DataParallel(net).cuda()

	"""
	Initialize Optimizers, Loss, Loss Schedulers
	"""
	optimizer_ft = optim.Adam(net.parameters(), lr=config.learning_rate)
	exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
	criterion = CrossEntropyLoss()

	monitor = TortillaMonitor(	experiment_name=config.experiment_name,
								topk=config.topk,
								classes=dataset.classes,
								use_gpu = use_gpu,
								plot=True,
								config=config
								)
	"""
	Train
	"""
	trainer = TortillaTrainer(
				dataset = dataset,
				model = net,
				loss = criterion,
				optimizer = optimizer_ft,
				monitor = monitor,
				config=config
				)

	def _run_one_epoch(epoch, train=True):
		print("\n" + "+"*80)
		pbar = tqdm.tqdm(total=100)
		pbar.set_description("Epoch : {} ; {}".format(epoch, "Training" if train else "Validation"))
		end_of_epoch = False
		last_percentage = 0
		while not end_of_epoch:
			_loss, images, labels, \
			outputs, percent_complete, \
			end_of_epoch = trainer._step(use_gpu=use_gpu, train=train)

			pbar.update(percent_complete*100 - last_percentage)
			last_percentage = percent_complete*100
			if end_of_epoch:
				break
		pbar.close()

	def _save_checkpoint(model, optimizer, epoch):
		path = config.experiment_dir_name+"/checkpoints/snapshot_{}_{}.model".format(epoch, monitor.val_loss.get_last())
		latest_snapshot_path = config.experiment_dir_name+"/checkpoints/snapshot_latest.model"
		optimizer_snapshot_path = config.experiment_dir_name+"/checkpoints/optimizer_state.pickle"
		print("Checkpointing model at : ", path)
		torch.save(model, path)
		shutil.copy2(path, latest_snapshot_path)
		# pickle.dump(optimizer.state_dict(), open(optimizer_snapshot_path, "wb"))

	for epoch in range(config.epochs):
		for train in [False, True]:
			_run_one_epoch(epoch, train=train)
		_save_checkpoint(net, optimizer_ft, epoch)
	_run_one_epoch(epoch, train=False)
	print("Hurray !! Your network is trained ! Now you can use `tortilla-predict` to make predictions.")


def collect_args():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--experiment-name', action='store', dest='experiment_name',
						required=True,
	                    help='A unique name for the current experiment')

	parser.add_argument('--experiments-dir', action='store', dest='experiments_dir',
						default=config.experiments_dir,
	                    help='Directory where results of all experiments will be stored.')

	parser.add_argument('--dataset-dir', action='store', dest='dataset_dir',
						required=True,
	                    help='Dataset directory in the TortillaDataset format')

	parser.add_argument('--model', action='store', dest='model',
						default=config.model,
	                    help='Type of the pretrained network to train with. Options : {}'.format(TortillaModel.supported_models))

	parser.add_argument('--optimizer', action='store', dest='optimizer',
						default=config.optimizer,
	                    help='Type of the pretrained network to train with. Options : ["adam"]')

	parser.add_argument('--batch-size', action='store', dest='batch_size',
						default=config.batch_size,
	                    help='Batch Size.')

	parser.add_argument('--epochs', action='store', dest='epochs',
						default=config.epochs,
	                    help='Number of epochs.')
	parser.add_argument('--learning-rate', action='store', dest='learning_rate',
						default=config.learning_rate,
	                    help='Learning Rate.')

	parser.add_argument('--top-k', action='store', dest='top_k',
						default=",".join([str(x) for x in config.topk]),
	                    help='List of values to compute top-k accuracies during \
						train and val.')

	parser.add_argument('--num-cpu-workers', action='store', dest='num_cpu_workers',
						default=config.num_cpu_workers,
	                    help='Number of CPU workers to be used by data loaders.')

	parser.add_argument('--visdom-server', action='store', dest='visdom_server',
						default=config.visdom_server,
	                    help='Visdom server hostname.')

	parser.add_argument('--visdom-port', action='store', dest='visdom_port',
						default=config.visdom_port,
	                    help='Visdom server port.')

	parser.add_argument('--no-plots', action='store_true', default=config.no_plots,
	                    dest='no_plots',
	                    help='Disable plotting on the visdom server')

	parser.add_argument('--use-cpu', action='store_true', default=config.use_cpu,
	                    dest='use_cpu',
	                    help='Boolean Flag to forcibly use CPU (on servers which\
						have GPUs. If you do not have a GPU, tortilla will \
						automatically use just CPU)')

	parser.add_argument('--debug', action='store_true', default=config.debug,
	                    dest='debug',
	                    help='Run tortilla in debug mode')

	parser.add_argument('--version', action='version', version='tortilla '+str(config.version))

	args = parser.parse_args()

	config.experiment_name = args.experiment_name
	config.experiments_dir = args.experiments_dir
	config.experiment_dir_name = config.experiments_dir+"/"+config.experiment_name
	config.dataset_dir = args.dataset_dir
	config.batch_size = int(args.batch_size)
	config.epochs = int(args.epochs)
	config.learning_rate = float(args.learning_rate)
	config.topk = [int(x) for x in args.top_k.split(",")]
	config.num_cpu_workers = int(args.num_cpu_workers)
	config.visdom_server = args.visdom_server
	config.visdom_port = int(args.visdom_port)
	config.debug = args.debug
	config.use_cpu = args.use_cpu

	return config

if __name__ == "__main__":
	utils.logo()
	config = collect_args()
	main(config)
