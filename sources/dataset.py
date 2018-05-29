# -*- coding: utf-8 -*-
"""dataset.py - Define the class of MyDataset
"""
import random
import torch
from torch.utils.data import Dataset
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, filelist, input_transform=None, target_transform=None):
        """Customized Dataset

        Arguments:
            filelist {str} -- The path of filelists (wp1_cad.txt, wp1_real.txt, ...)

        Keyword Arguments:
            input_transform {Callable} -- The preprocessing applied to images (default: {None})
            target_transform {Callable} -- The preprocessing applied to labels (default: {None})

        Members:
            # Filelist, marking the label of each file.
            self.filelist = [
                (filename:str, label:int),
                (filename:str, label:int),
                ...
            ]
            # Real label list, marking all indexes of images within a specific real label.
            self.reallabellist = {
                0 : [idx1:int, idx2:int, ...],
                1 : [idx1:int, idx2:int, ...],
                ...
            }
            # Pseudo label list, marking all indexes of images within a specific pseudo label.
            self.pseudolabellist = {
                0 : [idx1:int, idx2:int, ...],
                1 : [idx1:int, idx2:int, ...],
                ...
            }

        Examples:
            >>> from torchvision.transforms import transforms
            >>> from preprocessing import Resize
            >>> input_transform = transforms.Compose([
                    Resize((300,300)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            >>> cadset = MyDataset(
                    filelist='../dataset/wp1_cad.txt',
                    input_transform=input_transform
                )
        """
        with open(filelist, 'r') as fin:
            self.filelist = [line.split() for line in fin.readlines()]
            self.filelist = [(path, int(label)) for path, label in self.filelist]
        self.reallabellist, self.pseudolabellist = {}, {}
        for idx, (_, label) in enumerate(self.filelist):
            if label not in self.reallabellist:
                self.reallabellist[label] = []
            self.reallabellist[label].append(idx)
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """Read the image and the label by index.

        Arguments:
            index {int} -- index in the filelist

        Returns:
            image By default, it is a PIL.Image.Image, but you may convert
                  it to Torch.FloatTensor by passing an input_transform.
            label By default, it is an int object, but you may convert it
                  to Torch.LongTensor by passing a target_transform.
        """
        filepath, label = self.filelist[index]
        image = Image.open(filepath).convert('RGB')
        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        """Get the size of the dataset.

        Returns:
            int -- The size of the dataset
        """
        return len(self.filelist)

    def random_choice(self, selected_labels, use_pseudo=False):
        """Randomly select some images according to selected_labels
        from self.reallabellist or self.pseudolabellist, and return
        the selected images and their corresponding real labels.

        Arguments:
            selected_labels {[int]} -- Selected labels

        Keyword Arguments:
            use_pseudo {bool} -- Whether selecting from pseudo labels  (default: {False})

        Returns:
            torch.FloatTensor -- Selected images
            torch.LongTensor -- Real labels of the selected images
                                (If use_pseudo == True, it will be identical to selected_labels)
        """
        # Randomly select some images according to selected_labels from self.reallabellist or self.pseudolabellist
        if use_pseudo:
            imgidxes = [random.choice(self.pseudolabellist[label]) for label in selected_labels]
        else:
            imgidxes = [random.choice(self.reallabellist[label]) for label in selected_labels]
        # Get the real label of the selected samples
        reallabels = torch.LongTensor([self.filelist[idx][1] for idx in imgidxes])
        # Read images
        images = [Image.open(self.filelist[idx][0]).convert('RGB') for idx in imgidxes]
        # Preprocess
        if self.input_transform:
            images = [self.input_transform(image) for image in images]
        # Add a dimension of batch
        images = [image.unsqueeze(0) for image in images]
        # Concatenate images into a torch.FloatTensor, and return
        return torch.cat(images, 0), reallabels

    def update_pseudo_labels(self, labels):
        """Update pseudo labels

        Arguments:
            labels {np.array or [int]} -- pseudo labels (num_samples)
        """
        self.pseudolabellist.clear()
        for i, label in enumerate(labels):
            if label not in self.pseudolabellist:
                self.pseudolabellist[label] = []
            self.pseudolabellist[label].append(i)
