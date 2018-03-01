from torchvision import datasets, models, transforms
import torch.nn as nn

class TortillaModel:
	supported_models = ['alexnet','densenet121','densenet161','densenet169','densenet201','inception_v3','resnet101','resnet152','resnet18','resnet34','resnet50','vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19','vgg19_bn', 'squeezenet1_0']
	def __init__(self, model_name, classes):
		self.model_name = model_name
		self.classes = classes
		if model_name not in self.supported_models:
			raise("Model not implemented Error !")
		else:
			if self.model_name=='alexnet':
				self.net=models.alexnet(pretrained=True)
				in_features = self.net.classifier[-1].in_features
				self.net.classifier[-1] = nn.Linear(in_features, len(self.classes))

			if self.model_name=='densenet121':
				self.net=models.densenet121(pretrained=True)
				in_features = self.net.classifier.in_features
				self.net.classifier = nn.Linear(in_features, len(self.classes))

			if self.model_name=='densenet161':
				self.net=models.densenet161(pretrained=True)
				in_features = self.net.classifier.in_features
				self.net.classifier = nn.Linear(in_features, len(self.classes))

			if self.model_name=='densenet169':
				self.net=models.densenet169(pretrained=True)
				in_features = self.net.classifier.in_features
				self.net.classifier = nn.Linear(in_features, len(self.classes))
			#
			if self.model_name=='densenet201':
				self.net=models.densenet201(pretrained=True)
				in_features = self.net.classifier.in_features
				self.net.classifier = nn.Linear(in_features, len(self.classes))

			if self.model_name=='inception_v3':
				self.net=models.inception_v3(pretrained=True)
				in_features = self.net.fc.in_features
				self.net.fc = nn.Linear(in_features, len(self.classes))
			#
			if self.model_name=='resnet101':
				self.net=models.resnet101(pretrained=True)
				num_ftrs = self.net.fc.in_features
				self.net.fc = nn.Linear(num_ftrs, len(self.classes))
			#
			if self.model_name=='resnet152':
				self.net=models.resnet152(pretrained=True)
				num_ftrs = self.net.fc.in_features
				self.net.fc = nn.Linear(num_ftrs, len(self.classes))

			if self.model_name=='resnet18':
				self.net=models.resnet18(pretrained=True)
				num_ftrs = self.net.fc.in_features
				self.net.fc = nn.Linear(num_ftrs, len(self.classes))

			if self.model_name=='resnet34':
				self.net=models.resnet34(pretrained=True)
				num_ftrs = self.net.fc.in_features
				self.net.fc = nn.Linear(num_ftrs, len(self.classes))

			if self.model_name=='resnet50':
				self.net=models.resnet50(pretrained=True)
				num_ftrs = self.net.fc.in_features
				self.net.fc = nn.Linear(num_ftrs, len(self.classes))

			if self.model_name=='vgg11':
				self.net=models.vgg11(pretrained=True)
				num_ftrs = self.net.classifier[-1].in_features
				self.net.classifier[-1] = nn.Linear(num_ftrs, len(self.classes))

			if self.model_name=='vgg11_bn':
				self.net=models.vgg11_bn(pretrained=True)
				num_ftrs = self.net.classifier[-1].in_features
				self.net.classifier[-1] = nn.Linear(num_ftrs, len(self.classes))

			if self.model_name=='vgg13':
				self.net=models.vgg13(pretrained=True)
				num_ftrs = self.net.classifier[-1].in_features
				self.net.classifier[-1] = nn.Linear(num_ftrs, len(self.classes))

			if self.model_name=='vgg13_bn':
				self.net=models.vgg13_bn(pretrained=True)
				num_ftrs = self.net.classifier[-1].in_features
				self.net.classifier[-1] = nn.Linear(num_ftrs, len(self.classes))

			if self.model_name=='vgg16':
				self.net=models.vgg16(pretrained=True)
				num_ftrs = self.net.classifier[-1].in_features
				self.net.classifier[-1] = nn.Linear(num_ftrs, len(self.classes))

			if self.model_name=='vgg16_bn':
				self.net=models.vgg16_bn(pretrained=True)
				num_ftrs = self.net.classifier[-1].in_features
				self.net.classifier[-1] = nn.Linear(num_ftrs, len(self.classes))

			if self.model_name=='vgg19':
				self.net=models.vgg19(pretrained=True)
				num_ftrs = self.net.classifier[-1].in_features
				self.net.classifier[-1] = nn.Linear(num_ftrs, len(self.classes))

			if self.model_name=='vgg19_bn':
				self.net=models.vgg19_bn(pretrained=True)
				num_ftrs = self.net.classifier[-1].in_features
				self.net.classifier[-1] = nn.Linear(num_ftrs, len(self.classes))

			if self.model_name=='squeezenet1_0':
				self.net = models.squeezenet1_0(pretrained=True)
				self.net.classifier[1] = nn.Conv2d(512, len(self.classes), kernel_size=(1,1), stride=(1,1))
