# -*- coding: utf-8 -*-
"""predict.py - Predict all samples in testset
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable


def predict(model, testset, **kwargs):
    """Predict all samples in testset

    Arguments:
        model {nn.Module/nn.DataParallel} -- The network to be tested
        testset {Dataset} -- The test set

    Keyword Arguments:
        batch_size {int} -- Batch size
        n_classes {int} -- Number of classes in your dataset

    Returns:
        all_features {numpy.array} -- All features of testset (num_samples × num_features)
        all_outputs {numpy.array} -- All output scores of testset (num_samples × num_classes)
        all_pred {numpy.array} -- All predicted labels of testset (num_samples)
        all_label {numpy.array} -- All real labels of testset (num_samples)
    """
    if isinstance(model, nn.DataParallel):
        all_features = np.zeros((len(testset), model.module.n_features), dtype=np.float)
    else:
        all_features = np.zeros((len(testset), model.n_features), dtype=np.float)
    all_outputs = np.zeros((len(testset), kwargs['n_classes']), dtype=np.float)
    all_preds = np.zeros(shape=len(testset), dtype=np.int)
    all_labels = np.zeros(shape=len(testset), dtype=np.int)
    model.eval()
    for batch, (images, labels) in enumerate(tqdm(DataLoader(testset, batch_size=kwargs['batch_size'], shuffle=False, num_workers=os.cpu_count()))):
        images = Variable(images)
        if kwargs['GPUs']:
            images = images.cuda(kwargs['GPUs'][0])
        features, outputs = model(images)
        features, outputs = features.data, outputs.data
        if features.is_cuda:
            features = features.cpu()
        if outputs.is_cuda:
            outputs = outputs.cpu()
        all_features[batch * kwargs['batch_size']: (batch + 1) * kwargs['batch_size']] = features.numpy()
        all_labels[batch * kwargs['batch_size']: (batch + 1) * kwargs['batch_size']] = labels.numpy()
        all_outputs[batch * kwargs['batch_size']: (batch + 1) * kwargs['batch_size']] = outputs.numpy()
        all_preds[batch * kwargs['batch_size']: (batch + 1) * kwargs['batch_size']] = torch.max(outputs, 1)[1].numpy()
    model.train()
    return all_features, all_outputs, all_preds, all_labels
