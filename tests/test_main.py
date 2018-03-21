import numpy as np
from data_loaders import *


class Test_Dataloaders:
    def test_image_list_length(self):
        classes = open("tests/plants/classes.txt").readlines()
        train_list = ImageFilelist('tests/data/prepared-plants', 'tests/data/prepared-plants/train.json', classes=classes)
        assert train_list.total_images == 664
        val_list = ImageFilelist('tests/data/prepared-plants', 'tests/data/prepared-plants/val.json', classes=classes)
        assert val_list.total_images == 173

    def test_tortilla_dataset_init(self):
        dataset_test = TortillaDataset('tests/data/prepared-plants')
        classes_test = np.array(['c_0', 'c_1', 'c_2', 'c_3', 'c_5'])
        assert set(dataset_test.classes) == set(classes_test)
        assert dataset_test.meta['dataset_name'] == 'plants'

    def test_image_transform(self):
        transform_test = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        'val' : transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        dataset_test1 = TortillaDataset('tests/data/prepared-plants', data_transforms=transform_test, batch_size = 4)
        image, label, end_of_epoch = dataset_test1.get_next_batch(train=True)
        assert image.shape == torch.Size([4, 3, 256, 256])
        dataset_test2 = TortillaDataset('tests/data/prepared-plants', batch_size = 4)
        image, label, end_of_epoch = dataset_test2.get_next_batch(train=True)
        assert image.shape == torch.Size([4, 3, 224, 224])
        assert dataset_test1.train_iter_pointer == 1
        dataset_test1.reset_train_data_loaders()
        assert dataset_test1.train_iter_pointer == 0
