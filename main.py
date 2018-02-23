	#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.nn import CrossEntropyLoss

from config import Config as config
from data_loaders import TortillaDataset
from trainer import TortillaTrainer

from monitor import TortillaMonitor
import utils
import os, shutil
import pickle

"""
Initliaze params
"""
use_gpu = torch.cuda.is_available()

def main():
	utils.logo()
	utils.create_directory_structure(config.experiment_dir_name)
	"""
	Initialize Dataset
	"""
	dataset = TortillaDataset(	"datasets/food-101",
								batch_size=config.batch_size,
								num_cpu_workers=10,
								debug=config.debug
								)

	"""
	Initialize Model
	"""
	net = models.resnet50(pretrained=True)
	num_ftrs = net.fc.in_features
	net.fc = nn.Linear(num_ftrs, len(dataset.classes))

	# Make net use parallel gpu
	if use_gpu:
		net = nn.DataParallel(net).cuda()

	"""
	Initialize Optimizers, Loss, Loss Schedulers
	"""
	optimizer_ft = optim.Adam(net.parameters(), lr=0.001)
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

	def _run_one_epoch(train=True):
		end_of_epoch = False
		while not end_of_epoch:
			_loss, images, labels, \
			outputs, end_of_epoch = trainer._step(use_gpu=use_gpu, train=train)
			if end_of_epoch:
				break
	def _save_checkpoint(model, optimizer, epoch):
		path = config.experiment_dir_name+"/checkpoints/snapshot_{}_{}.model".format(epoch, monitor.val_loss.get_last())
		latest_snapshot_path = config.experiment_dir_name+"/checkpoints/snapshot_latest.model"
		optimizer_snapshot_path = config.experiment_dir_name+"/checkpoints/optimizer_state.pickle"
		print("Checkpointing model at : ", path)
		torch.save(model, path)
		shutil.copy2(path, latest_snapshot_path)
		# pickle.dump(optimizer.state_dict(), open(optimizer_snapshot_path, "wb"))

	for epoch in range(config.epochs):
		print("Epoch : ", epoch)
		for train in [False, True]:
			_run_one_epoch(train=train)
		_save_checkpoint(net, optimizer_ft, epoch)
	_run_one_epoch(train=False)


if __name__ == "__main__":
	main()
