# -*- coding: utf-8 -*-
import sys
sys.path.append('sources')
import torch
import random
import unittest
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from predict import predict
from network import Model
from score import AP, meanAP


class TestNetwork(unittest.TestCase):
    class RandomData(Dataset):
        def __init__(self, n_images, n_classes, input_transform=None):
            self.n_images = n_images
            self.n_classes = n_classes
            self.input_transform = input_transform

        def __getitem__(self, index):
            images = torch.rand(3, 300, 300)
            labels = random.randint(0, self.n_classes - 1)
            if self.input_transform:
                images = self.input_transform(images)
            return images, labels

        def __len__(self):
            return self.n_images

    def test_network(self):
        input_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.realset = TestNetwork.RandomData(n_images=10, n_classes=7, input_transform=input_transform)
        self.model = Model(n_classes=7)
        self.model.eval()
        all_features, all_outputs, all_preds, all_labels = predict(
            self.model, self.realset, batch_size=4, n_classes=7, GPUs=None)
        recall = np.sum(all_preds == all_labels) / float(len(self.realset))
        ap = AP(all_outputs, all_labels)
        mean_ap = meanAP(all_outputs, all_labels)
        self.assertGreaterEqual(mean_ap, 0)
        self.assertLessEqual(mean_ap, 1)


if __name__ == '__main__':
    unittest.main()
