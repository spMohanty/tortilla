import torch
from torchvision import datasets, models, transforms

from config import Config as config

import torch.utils.data as data

from PIL import Image
import os
import os.path


def default_loader(path):
	return Image.open(path).convert('RGB')

def default_flist_reader(flist):
	"""
	flist format: impath label\nimpath label\n ...(same to caffe's filelist)
	"""
	imlist = []
	with open(flist, 'r') as rf:
		for line in rf.readlines():
			impath, imlabel = line.strip().split()
			imlist.append( (impath, int(imlabel)) )

	return imlist

class ImageFilelist(data.Dataset):
	def __init__(self, root, flist, transform=None, target_transform=None,
			flist_reader=default_flist_reader, loader=default_loader):
		self.root   = root
		self.imlist = flist_reader(flist)
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

		self.val_data_loader = torch.utils.data.DataLoader(
				dataset=self.val_dataset,
				batch_size=self.batch_size,
				shuffle=self.shuffle,
				num_workers=self.num_cpu_workers
		)
		self.val = self.val_data_loader

if __name__ == "__main__":
	dataset = TortillaDataset(	"datasets/food-101",
								batch_size=128,
								num_cpu_workers=10
								)
	for (images, labels) in dataset.train:
		print(images.shape, labels.shape)
		exit(0)
