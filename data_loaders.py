import torch
from torchvision import datasets, models, transforms
import torch.utils.data as data
from torch.autograd import Variable

import os
import os.path

from utils import default_flist_reader, default_loader

class ImageFilelist(data.Dataset):
	def __init__(self, root, flist, transform=None, target_transform=None,
			flist_reader=default_flist_reader, loader=default_loader):
		self.root   = root
		self.imlist = flist_reader(flist)
		self.total_images = len(self.imlist)
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader

	def __getitem__(self, index):
		impath, target = self.imlist[index]
		img = self.loader(os.path.join(self.root,impath))
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
				shuffle=True, batch_size=32, num_cpu_workers=4):
		self.dataset_folder = dataset_folder
		self.data_transforms = data_transforms
		self.shuffle = shuffle
		self.batch_size = batch_size
		self.num_cpu_workers = num_cpu_workers

		self.classes = open(os.path.join(self.dataset_folder,
										"classes.txt")).readlines()
		"""
		Define transforms
		"""
		if data_transforms == None:
			self.data_transforms = {
			    'train': transforms.Compose([
			        transforms.RandomSizedCrop(224),
			        transforms.RandomHorizontalFlip(),
			        transforms.RandomVerticalFlip(),
			        transforms.RandomRotation(180),
			        transforms.ColorJitter(),
			        transforms.ToTensor(),
			        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			    ]),
			    'val': transforms.Compose([
			        transforms.Scale(256),
			        transforms.CenterCrop(224),
			        transforms.ToTensor(),
			        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			    ]),
			}
		"""
			Define datasets from filelists
		"""
		train_filelist = os.path.join(self.dataset_folder, "train.txt")
		val_filelist = os.path.join(self.dataset_folder, "val.txt")
		self.train_dataset = ImageFilelist(
								self.dataset_folder,
								train_filelist,
								transform=self.data_transforms["train"]
								)
		self.val_dataset = ImageFilelist(
								self.dataset_folder,
								val_filelist,
								transform=self.data_transforms["val"]
								)

		"""
		Define dataloaders
		"""
		self.train_data_loader = torch.utils.data.DataLoader(
				dataset=self.train_dataset,
				batch_size=self.batch_size,
				shuffle=self.shuffle,
				num_workers=self.num_cpu_workers
		)
		self.train = self.train_data_loader
		self.train_iter = iter(self.train)
		self.len_train_images = self.train_dataset.total_images

		self.val_data_loader = torch.utils.data.DataLoader(
				dataset=self.val_dataset,
				batch_size=self.batch_size,
				shuffle=self.shuffle,
				num_workers=self.num_cpu_workers
		)
		self.val = self.val_data_loader
		self.val_iter = iter(self.val)
		self.len_val_images = self.val_dataset.total_images

	def get_next_batch(self, train=True, gpu=False):
		if train:
			try:
				images, labels = next(self.train_iter)
			except StopIteration:
				# return (images, labels, end_of_epoch?)
				return (False, False, True)
		else:
			try:
				images, labels = next(self.val_iter)
			except StopIteration:
				# return (images, labels, end_of_epoch?)
				return (False, False, True)

		images = Variable(images)
		labels = Variable(labels)

		if gpu:
			images = images.cuda()
			labels = labels.cuda()

		# return (images, labels, end_of_epoch?)
		return (images, labels, False)

if __name__ == "__main__":
	dataset = TortillaDataset(	"datasets/food-101",
								batch_size=128,
								num_cpu_workers=10
								)
	for (images, labels) in dataset.train:
		print(images.shape, labels.shape)
		exit(0)
