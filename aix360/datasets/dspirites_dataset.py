"""dataset.py"""

import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
#import os
#import torch.nn.functional as F
import torchvision.datasets as dset
#import torchvision.transforms as transforms

class DSpiritesDataset():
    """
    DSpirites Dataset
    """
    def __init__(self, batch_size=256, subset_size=5000, test_batch_size=256, dirpath=None):
        trans = transforms.Compose([transforms.ToTensor()])

        self._dirpath = dirpath
        if not self._dirpath:
            self._dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                '..', 'data','dsprites_data')


        dataset='dsprites'
        dset_dir=self._dirpath
        image_size=64
        num_workers=1

        #self.train_loader = self.return_data(dset_dir=self._dirpath,batch_size=batch_size,num_workers=4,image_size=64)
        #self.test_loader = self.return_data(dset_dir=self._dirpath,batch_size=batch_size,num_workers=4,image_size=64)
        #self.test_loader = self.return_data(args)
        name = dataset
        dset_dir = dset_dir
        batch_size = batch_size
        num_workers = num_workers
        image_size = image_size
        assert image_size == 64, 'currently only image size of 64 is supported'


        class CustomTensorDataset(Dataset):
            def __init__(self, data_tensor, transform=None):
                self.data_tensor = data_tensor
                self.transform = transform
                self.indices = range(len(self))

            def __getitem__(self, index1):
                index2 = random.choice(self.indices)

                img1 = self.data_tensor[index1]
                img2 = self.data_tensor[index2]
                if self.transform is not None:
                    img1 = self.transform(img1)
                    img2 = self.transform(img2)

                return img1, img2

            def __len__(self):
                return self.data_tensor.size(0)

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),])

        if name.lower() == 'celeba':
            root = os.path.join(dset_dir, 'CelebA')
            train_kwargs = {'root':root, 'transform':transform}
            dset = CustomImageFolder
        elif name.lower() == '3dchairs':
            root = os.path.join(dset_dir, '3DChairs')
            train_kwargs = {'root':root, 'transform':transform}
            dset = CustomImageFolder
        elif name.lower() == 'dsprites':
            root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
            data = np.load(root, encoding='latin1')
            data = torch.from_numpy(data['imgs']).float()
            print(data.size())
            train_kwargs = {'data_tensor':data}
            dset = CustomTensorDataset
        else:
            raise NotImplementedError


        train_data = dset(**train_kwargs)
        self.train_loader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=True)
        self.test_loader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=True)

        indices = torch.randperm(len(train_data))[:subset_size]
        train_set = torch.utils.data.Subset(train_data, indices)
        test_set=train_data
        self.name = "dsprites"
        self.data_dims = [64, 64, 1]
        self.train_size = len(self.train_loader)
        self.test_size = len(self.test_loader)
        self.range = [0.0, 1.0]
        self.batch_size = batch_size
        self.num_training_instances = len(train_set)
        self.num_test_instances = len(test_set)
        self.likelihood_type = 'gaussian'
        self.output_activation_type = 'sigmoid'
    

    def is_power_of_2(num):
        return ((num & (num - 1)) == 0) and num != 0


    class CustomImageFolder(ImageFolder):
        def __init__(self, root, transform=None):
            super(CustomImageFolder, self).__init__(root, transform)
            self.indices = range(len(self))

        def __getitem__(self, index1):
            index2 = random.choice(self.indices)

            path1 = self.imgs[index1][0]
            path2 = self.imgs[index2][0]
            img1 = self.loader(path1)
            img2 = self.loader(path2)
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img1, img2


    class CustomTensorDataset(Dataset):
        def __init__(self, data_tensor, transform=None):
            self.data_tensor = data_tensor
            self.transform = transform
            self.indices = range(len(self))

        def __getitem__(self, index1):
            index2 = random.choice(self.indices)

            img1 = self.data_tensor[index1]
            img2 = self.data_tensor[index2]
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img1, img2

        def __len__(self):
            return self.data_tensor.size(0)


    def return_data(dataset,dset_dir,batch_size,num_workers,image_size):
        name = dataset
        dset_dir = dset_dir
        batch_size = batch_size
        num_workers = num_workers
        image_size = image_size
        assert image_size == 64, 'currently only image size of 64 is supported'

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),])

        if name.lower() == 'celeba':
            root = os.path.join(dset_dir, 'CelebA')
            train_kwargs = {'root':root, 'transform':transform}
            dset = CustomImageFolder
        elif name.lower() == '3dchairs':
            root = os.path.join(dset_dir, '3DChairs')
            train_kwargs = {'root':root, 'transform':transform}
            dset = CustomImageFolder
        elif name.lower() == 'dsprites':
            root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
            data = np.load(root, encoding='latin1')
            #data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
            data = torch.from_numpy(data['imgs']).float()
            train_kwargs = {'data_tensor':data}
            dset = CustomTensorDataset
        else:
            raise NotImplementedError


        train_data = dset(**train_kwargs)
        train_loader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=True)

        data_loader = train_loader
        return data_loader

    def next_batch(self):
        for x, y in self.train_loader:
            x = np.reshape(x, (-1, 64*64))
            yield x, y

    def next_test_batch(self):
        for x, y in self.test_loader:
            x = np.reshape(x, (-1, 64*64))
            yield x, y
