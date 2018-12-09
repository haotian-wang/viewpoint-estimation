# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import random
import logging
import traceback
from multiprocessing import cpu_count
from distutils.version import LooseVersion

import torch
from torch import nn
from torch.optim import SGD
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
try:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter()
except:
    writer = None
    print("Warning: TensorboardX is not installed, so we will not use Tensorboard.")

from network import Model
from dataset import MyDataset
from mmd import mix_rbf_mmd2
from score import AP, meanAP
from preprocessing import Resize
from predict import predict

if LooseVersion(torch.__version__) < LooseVersion('0.4.0'):
    Variable.item = lambda self: self.data[0]


def train(model, cadset, realset, optimizer, hot=False,
          summarywriter=None, savefilename=None, **kwargs):
    """Train the network using our method

    Arguments:
        model {nn.Module or nn.DataParallel} -- The network to be trained
        cadset {MyDataset} -- Dataset of synthetic images
        realset {MyDataset} -- Dataset of real images
        optimizer {torch.optim.Optimizer} -- Optimizer

    Keyword Arguments:
        hot {bool} -- Whether training in hot stage (default: {False})
        summarywriter {tensorboardX.SummaryWriter} -- Tensorboard writer (default: {None})
        savefilename {str} -- Filename of the saved model (default: {None})
        epoch {int} -- Number of epoches to train
        batch_size {int} -- Batch size
        n_classes {int} -- Number of classes of your dataset
        test_steps {int} -- Test intervals while training
        GPUs {None/int/(int)} -- CUDA device IDs

    Returns:
        max_recall {float} Maximum recall during training process
        max_ap {[float]} Maximum AP during training process
        max_mean_ap {[float]} Maximum mean AP during training process
    """

    if not hot:
        print('Traning in cold stage!')
    else:
        print('Training in hot stage!')
    if isinstance(model, nn.DataParallel):
        print('Warning: Your are using DataParallel. We will only save the state dict of the module, instead of the whole DataParallel object.')
    if summarywriter is None:
        print('Warning: summarywriter is None. The result will not be displayed on Tensorboard!')
    if savefilename is None:
        print('Warning: savefilename is None. The trained model will not be saved!')
    if 'epoch' not in kwargs:
        raise ValueError('Please specify the number of epoches by passing "epoch=YOUR_EPOCHES"!')
    if 'batch_size' not in kwargs:
        raise ValueError('Please specify the batch size by passing "batch_size=YOUR_BATCH_SIZE"!')
    if 'n_classes' not in kwargs:
        raise ValueError('Please specify the number of classes in your dataset by passing "n_classes=YOUR_CLASSES"!')
    if 'test_steps' not in kwargs:
        kwargs['test_steps'] = 50
        print('Warning: test_steps is not specified, we will use 50 by default.')

    max_recall, max_ap, max_mean_ap = 0, None, 0
    for epoch in range(kwargs['epoch']):
        cadloader = DataLoader(cadset, batch_size=kwargs['batch_size'], shuffle=True,
                               num_workers=cpu_count(), drop_last=True)
        for batch, (images_cad, labels_cad) in enumerate(cadloader):
            # Test accuracies
            model.eval()
            if (epoch * len(cadloader) + batch) % kwargs['test_steps'] == 0:
                _, all_output, all_pred, all_label = predict(model, realset, **kwargs)
                recall = np.sum(all_pred == all_label) / float(len(realset))
                ap = AP(all_output, all_label)
                mean_ap = meanAP(all_output, all_label)
                print('Mean Recall: ', recall)
                print('AP: ', ap)
                print('Mean AP: ', mean_ap)
                print('Previous Maximum Mean AP: ', max_mean_ap)
                print('Previous Maximum Accuracy: ', max_recall)
                if mean_ap >= max_mean_ap:
                    max_ap, max_mean_ap = ap, mean_ap
                if recall >= max_recall:
                    max_recall = recall
                    if hot:
                        print('Update pseudo labels!')
                        realset.update_pseudo_labels(all_pred)
                    if savefilename is not None:
                        if isinstance(model, nn.DataParallel):
                            torch.save(model.module.state_dict(), savefilename)
                        else:
                            torch.save(model.state_dict(), savefilename)

            # Read training samples
            if hot:
                images_real, labels_real = realset.random_choice(labels_cad, use_pseudo=True)
            else:
                images_real, labels_real = realset.random_choice([
                    random.randint(0, kwargs['n_classes'] - 1) for _ in range(kwargs['batch_size'])
                ])

            # Convert torch.Tensor to torch.autograd.Variable
            model.train()
            images_cad = Variable(images_cad)
            labels_cad = Variable(labels_cad)
            images_real = Variable(images_real)
            labels_real = Variable(labels_real)

            if kwargs['GPUs']:
                images_cad = images_cad.cuda(kwargs['GPUs'][0])
                labels_cad = labels_cad.cuda(kwargs['GPUs'][0])
                images_real = images_real.cuda(kwargs['GPUs'][0])
                labels_real = labels_real.cuda(kwargs['GPUs'][0])

            # Feed to our network
            mmd_cad, mmd_real, out_cad, out_real = model(images_cad, images_real)

            # Calculate the loss
            loss_class = F.cross_entropy(out_cad, labels_cad)
            loss_mmd = mix_rbf_mmd2(mmd_cad, mmd_real, [1, 2, 4, 8, 16])
            loss = loss_class + loss_mmd

            # Calculate the accuracy within this batch
            accuracy_cad = torch.sum(labels_cad == torch.max(out_cad, 1)[1]).item() / float(kwargs['batch_size'])
            accuracy_pseudo = torch.sum(labels_cad == torch.max(out_real, 1)[1]).item() / float(kwargs['batch_size'])
            accuracy_real = torch.sum(labels_real == torch.max(out_real, 1)[1]).item() / float(kwargs['batch_size'])

            # Print the loss and the accuracy
            if hot:
                print('epoch:%d, batch:%d, loss:%0.5f, loss_class:%0.5f, loss_mmd:%0.5f, accuracy of CAD:%0.5f, accuracy of pseudo:%0.5f, accuracy of real:%0.5f' % (
                    epoch, batch, loss.item(), loss_class.item(), loss_mmd.item(), accuracy_cad, accuracy_pseudo, accuracy_real
                ))
            else:
                print('epoch:%d, batch:%d, loss:%0.5f, loss_class:%0.5f, loss_mmd:%0.5f, accuracy of CAD:%0.5f, accuracy of real:%0.5f' % (
                    epoch, batch, loss.item(), loss_class.item(), loss_mmd.item(), accuracy_cad, accuracy_real
                ))

            # Print to Tensorboard
            if summarywriter:
                summarywriter.add_scalar('accuracy_of_cad', accuracy_cad, epoch * len(cadloader) + batch)
                summarywriter.add_scalar('accuracy_of_real', accuracy_real, epoch * len(cadloader) + batch)
                summarywriter.add_scalar('loss_of_classification', loss_class.item(), epoch * len(cadloader) + batch)
                summarywriter.add_scalar('loss_of_mmd', loss_mmd.item(), epoch * len(cadloader) + batch)
                summarywriter.add_scalar('loss', loss.item(), epoch * len(cadloader) + batch)

            # Optimize the network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return max_recall, max_ap, max_mean_ap


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        filename='output.log',
        filemode='a'
    )
    parameters = {
        'epoch': 60,
        'batch_size': 64,
        'n_classes': 7,
        'test_steps': 50,
        # Whether to use GPU?
        # None      -- CPU only
        # 0 or (0,) -- Use GPU0
        # (0, 1)    -- Use GPU0 and GPU1
        'GPUs': 0
    }
    if isinstance(parameters['GPUs'], int):
        parameters['GPUs'] = (parameters['GPUs'], )
    for wp in ('wp{}'.format(i) for i in range(1, 9)):  # Training from WP1 to WP8
        cadset = MyDataset(
            filelist='../dataset/{}_cad.txt'.format(wp),
            input_transform=transforms.Compose([
                Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )
        realset = MyDataset(
            filelist='../dataset/{}_real.txt'.format(wp),
            input_transform=transforms.Compose([
                Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )

        model = Model(parameters['n_classes'])
        # If you need to load your pretrained model, uncomment the following line.
        # model.load_state_dict(torch.load('{}-cold.pth'.format(wp)))

        if parameters['GPUs']:
            model = model.cuda(parameters['GPUs'][0])
            if len(parameters['GPUs']) > 1:
                model = nn.DataParallel(model, device_ids=parameters['GPUs'])

        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

        try:
            # Cold stage
            recall, ap, meanap = train(model, cadset, realset, optimizer, hot=False,
                                       summarywriter=writer, savefilename='{}-cold.pth'.format(wp), **parameters)
            logging.info('{}, Recall: {}, AP: {}, mean AP: {}'.format(wp, recall, ap, meanap))
            print('{}, Recall: {}, AP: {}, mean AP: {}'.format(wp, recall, ap, meanap))
            # Hot stage
            recall, ap, meanap = train(model, cadset, realset, optimizer, hot=True,
                                       summarywriter=writer, savefilename='{}-hot.pth'.format(wp), **parameters)
            logging.info('{}-hot, Recall: {}, AP: {}, mean AP: {}'.format(wp, recall, ap, meanap))
            print('{}-hot, Recall: {}, AP: {}, mean AP: {}'.format(wp, recall, ap, meanap))
        except Exception as e:
            logging.info('{} error!{}'.format(wp, e))
            print('{} error!{}'.format(wp, e))
            traceback.print_exc()
