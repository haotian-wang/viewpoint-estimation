# -*- coding: utf-8 -*-
import sys
sys.path.append('sources')
import torch
import unittest
from tqdm import trange
from torch.autograd import Variable
from torchvision import transforms
from predict import predict
from network import Model
from preprocessing import Resize


class NetTester(unittest.TestCase):
    def test_testwork(self):
        batchsize, n_classes = 4, 7
        model = Model(n_classes=n_classes)
        input_transform = transforms.Compose([
            Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        for _ in trange(32, ascii=True):
            input_cad = torch.rand(batchsize, 3, 300, 300)
            input_real = torch.rand(batchsize, 3, 300, 300)
            input_cad, input_real = Variable(input_cad), Variable(input_real)
            _, _, pred_cad, pred_real = model(input_cad, input_real)
            assert pred_cad.shape == pred_real.shape == (batchsize, n_classes)

            input_real = Variable(torch.rand(batchsize, 3, 300, 300))
            _, pred_real = model(input_real)
            assert pred_real.shape == (batchsize, n_classes)


if __name__ == '__main__':
    unittest.main()
