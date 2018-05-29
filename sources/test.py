# -*- coding: utf-8 -*-
"""test.py - Test all samples in testset and visualize the result
"""
from __future__ import absolute_import, print_function, division
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import torch
from torch import nn
from torchvision import transforms

from network import Model
from dataset import MyDataset
from score import AP, meanAP
from predict import predict
from preprocessing import Resize


if __name__ == '__main__':
    parameters = {
        'batch_size': 64,
        'n_classes': 7,
        # Whether to use GPU?
        # None      -- CPU only
        # 0 or (0,) -- Use GPU0
        # (0, 1)    -- Use GPU0 and GPU1
        'GPUs': 0
    }
    if isinstance(parameters['GPUs'], int):
        parameters['GPUs'] = (parameters['GPUs'], )

    testset = MyDataset(
        filelist='../dataset/wp1_real.txt',
        input_transform=transforms.Compose([
            Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )

    model = Model(parameters['n_classes'])
    model.load_state_dict(torch.load('wp1-cold.pth'))

    if parameters['GPUs']:
        model = model.cuda(parameters['GPUs'][0])
        if len(parameters['GPUs']) > 1:
            model = nn.DataParallel(model, device_ids=parameters['GPUs'])

    model.eval()

    all_features, all_outputs, all_preds, all_labels = predict(model, testset, **parameters)

    recall = np.sum(all_preds == all_labels) / float(len(testset))
    ap = AP(all_outputs, all_labels)
    mean_ap = meanAP(all_outputs, all_labels)

    print('Mean Recall: ', recall)
    print('AP of each class: ', ap)
    print('mean AP: ', mean_ap)

    tsne = TSNE(verbose=True)
    P = tsne.fit_transform(all_features)
    plt.scatter(P[:,0], P[:,1], c=all_labels, marker='x')
    plt.axis('off')
    plt.show()
