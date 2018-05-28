import torch
from torchvision import datasets, models, transforms
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np


import os
import os.path
import json

from utils import default_flist_reader, default_loader
from config import Config as config

class ImageFilelist(data.Dataset):
	def __init__(self, root, flist, classes, transform=None,
				target_transform=None, flist_reader=default_flist_reader,
				loader=default_loader, is_absolute_path=False, debug=False):
		self.root   = root
		self.flist = flist
		self.classes = classes
		self.imlist = flist_reader(self.flist, self.classes)
		self.total_images = len(self.imlist)
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader
		self.is_absolute_path = is_absolute_path
		self.debug = debug

		if self.debug:
			self.imlist = self.imlist[:3000]

	def __getitem__(self, index):
		impath, target = self.imlist[index]

		if not self.is_absolute_path:
			impath = os.path.join(self.root, impath)

		img = self.loader(impath)

		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	def __len__(self):
		return len(self.imlist)

class TortillaDataset:
	"""
	TortillaDataset is a high level wrapper over the dataset format of Tortilla
	"""
	def __init__(self, dataset_folder, data_transforms=None,
				shuffle=True, batch_size=32, num_cpu_workers=4,
				no_data_augmentation=False, debug=False,wrs=False):
		self.dataset_folder = dataset_folder
		self.data_transforms = data_transforms
		self.shuffle = shuffle
		self.batch_size = batch_size
		self.num_cpu_workers = num_cpu_workers
		self.no_data_augmentation = no_data_augmentation
		self.debug = debug
		self.wrs = wrs
	
		self.classes = open(os.path.join(self.dataset_folder,
										"classes.txt")).readlines()
		self.classes = [x.strip() for x in self.classes]
		self.meta = json.loads(open(os.path.join(self.dataset_folder,
										"meta.json")).read())
		self.weights = [self.meta['train_class_frequency'][i] for i in self.meta['train_class_frequency'].keys()]
		self.weights = (sum(self.weights)/torch.Tensor(self.weights))
		# Add default values if they dont exist in meta
		if "is_absolute_path" not in self.meta.keys():
			self.meta["is_absolute_path"] = False
		"""
		Define transforms
		"""
				
		if data_transforms == None:
			normalize = transforms.Normalize(config.normalize_params[:3],config.normalize_params[3:])
			trans = [transforms.ToTensor(),normalize]
			val =  [transforms.Resize(config.resize_shape),
				        transforms.CenterCrop(config.input_size)] + trans

			if not no_data_augmentation:
				train = [transforms.RandomResizedCrop(config.input_size),
				        transforms.RandomHorizontalFlip(),
				        transforms.RandomVerticalFlip(),
				        transforms.RandomRotation(180),
				        transforms.ColorJitter()] + trans
			else: 
				train = val 
			self.data_transforms = {'train' : transforms.Compose(train), 'val': transforms.Compose(val)}

		"""
			Define datasets from filelists
		"""
		train_filelist = os.path.join(self.dataset_folder, "train.json")
		val_filelist = os.path.join(self.dataset_folder, "val.json")
		self.train_dataset = ImageFilelist(
								self.dataset_folder,
								train_filelist,
								self.classes,
								transform=self.data_transforms["train"],
								is_absolute_path = self.meta["is_absolute_path"],
								debug=self.debug
								)
		self.val_dataset = ImageFilelist(
								self.dataset_folder,
								val_filelist,
								self.classes,
								transform=self.data_transforms["val"],
								is_absolute_path = self.meta["is_absolute_path"],
								debug=self.debug
								)

		self.reset_train_data_loaders()
		self.reset_val_data_loaders()
		self.describe()

	"""
	Describe datasets
	"""
	def describe(self):
		print("="*80)
		print("Dataset description \t ")
		print("-------------------")
		print("Dataset Path \t:\t {}".format(self.dataset_folder))
		print("Classes \t:\t {}".format(self.classes))
		print("Training Set Percent \t:\t {}".format(self.meta["train_percent"]))
		print("Images in training set \t:\t {}".format(self.train_dataset.total_images))
		print("Images in validation set \t:\t {}".format(self.val_dataset.total_images))
		print("="*80)
		print("Weights \t :\t {}".format(self.weights))
		print("wrs \t :\t {}".format(self.wrs))
	"""
	Define dataloaders
	"""
	
	def reset_train_data_loaders(self):
	
		if self.wrs ==True:
			train_targets = np.array([i[1] for i in self.train_dataset.imlist])
			sampler = torch.utils.data.sampler.WeightedRandomSampler(self.weights[train_targets],self.train_dataset.total_images)
			
			self.train_data_loader = torch.utils.data.DataLoader(
				dataset=self.train_dataset,
				batch_size=self.batch_size,
				shuffle=False,
				num_workers=self.num_cpu_workers,
				sampler=sampler
			)
		else:
			self.train_data_loader = torch.utils.data.DataLoader(
				dataset=self.train_dataset,
				batch_size=self.batch_size,
				shuffle=self.shuffle,
				num_workers=self.num_cpu_workers
			)
			
		self.train = self.train_data_loader
		self.train_iter = iter(self.train)
		self.train_iter_pointer = 0
		self.len_train_images = self.train_dataset.total_images

	def reset_val_data_loaders(self):
		self.val_data_loader = torch.utils.data.DataLoader(
				dataset=self.val_dataset,
				batch_size=self.batch_size,
				shuffle=self.shuffle,
				num_workers=self.num_cpu_workers
			)

		self.val = self.val_data_loader
		self.val_iter = iter(self.val)
		self.val_iter_pointer = 0
		self.len_val_images = self.val_dataset.total_images

	def percent_complete(self, train=True):
		"""
			Returns in percentage [0,1] the amount of data that has already
			been iterated
		"""
		if train:
			return float(self.train_iter_pointer*self.batch_size)/self.len_train_images
		else:
			return float(self.val_iter_pointer*self.batch_size)/self.len_val_images


	def get_next_batch(self, train=True, use_gpu=False):
		end_of_epoch = False
		if train:
			try:
				images, labels = next(self.train_iter)
				self.train_iter_pointer += 1
			except StopIteration:
				# return (images, labels, end_of_epoch?)
				end_of_epoch = True
				self.reset_train_data_loaders()
				return (False, False, end_of_epoch)
		else:
			try:
				images, labels = next(self.val_iter)
				self.val_iter_pointer += 1
			except StopIteration:
				# return (images, labels, end_of_epoch?)
				end_of_epoch = True
				self.reset_val_data_loaders()
				return (False, False, end_of_epoch)

		images = Variable(images)
		labels = Variable(labels)

		if use_gpu:
			images = images.cuda()
			labels = labels.cuda()

		# return (images, labels, end_of_epoch?)
		return (images, labels, end_of_epoch)

def main():
	dataset = TortillaDataset(	"datasets/601._apple",
								batch_size=128,
								num_cpu_workers=1,
								wrs=True	
								)

	# Example iteration using `.get_next_batch`
	_idx = 0
	while True:
		_idx += 1
		images, labels, end_of_epoch = dataset.get_next_batch(train=True)
		if end_of_epoch:
			print(_idx, end_of_epoch)
			break
		print(_idx, images.shape, labels.shape, end_of_epoch)

if __name__ == "__main__":
	main()
