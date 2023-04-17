import os
from .image_folder import make_dataset_with_labels, make_dataset
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms

class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def __getitem__(self, index):
        path = self.data_paths[index]
        img = Image.open(path).convert('RGB')
        if self.transform_1 is not None:
            # img_1 = random_color_aug(img)
            img_1 = self.transform_1(img)
        if self.transform_2 is not None:
            img_2 = self.transform_2(img)    
        label = self.data_labels[index] 

        return {'Path': path, 'Img_1': img_1, 'Img_2': img_2, 'Label': label}

    def initialize(self, root, transform_1=None, transform_2=None, **kwargs):
        self.root = root
        self.data_paths = []
        self.data_labels = []
        self.transform_1 = transform_1
        self.transform_2 = transform_2

    def __len__(self):
        return len(self.data_paths)

class SingleDataset(BaseDataset):
    def initialize(self, root, classnames, transform_1=None, transform_2=None,**kwargs):
        BaseDataset.initialize(self, root, transform_1=transform_1, transform_2=transform_2)
        self.data_paths, self.data_labels = make_dataset_with_labels(
				self.root, classnames)

        assert(len(self.data_paths) == len(self.data_labels)), \
            'The number of images (%d) should be equal to the number of labels (%d).' % \
            (len(self.data_paths), len(self.data_labels))

    def name(self):
        return 'SingleDataset'

class BaseDatasetWithoutLabel(Dataset):
    def __init__(self):
        super(BaseDatasetWithoutLabel, self).__init__()

    def name(self):
        return 'BaseDatasetWithoutLabel'

    def __getitem__(self, index):
        path = self.data_paths[index]
        img = Image.open(path).convert('RGB')
        if self.transform_1 is not None:
            # img_1 = random_color_aug(img)
            img_1 = self.transform_1(img)
        if self.transform_2 is not None:
            img_2 = self.transform_2(img)    

        return {'Path': path, 'Img_1': img_1, 'Img_2': img_2}

    def initialize(self, root, transform_1=None, transform_2=None, **kwargs):
        self.root = root
        self.data_paths = []
        self.transform_1 = transform_1
        self.transform_2 = transform_2

    def __len__(self):
        return len(self.data_paths)

class SingleDatasetWithoutLabel(BaseDatasetWithoutLabel):
    def initialize(self, root, transform_1=None, transform_2=None, **kwargs):
        BaseDatasetWithoutLabel.initialize(self, root, transform_1=transform_1, transform_2=transform_2)
        self.data_paths = make_dataset(self.root)

    def name(self):
        return 'SingleDatasetWithoutLabel'

# def random_color_aug(src):
#     brightness = transforms.ColorJitter(brightness=0.5)
#     contrast = transforms.ColorJitter(contrast=0.5)
#     saturation = transforms.ColorJitter(saturation=0.5)
#     hue = transforms.ColorJitter(hue=0.3)
#         # gray_augmentator = transforms.RandomGrayscale(p=0.5)

#     color_list = [ brightness, contrast, hue, saturation]
#     len_color = len(color_list)
#     len_color = [i for i in range(len_color)]
#         # 随机打乱
#     np.random.shuffle(len_color)
#     for j in len_color:
#             # if np.random.uniform(0, 1) >= 0.5:
#         src = color_list[j](src)
#     return src


