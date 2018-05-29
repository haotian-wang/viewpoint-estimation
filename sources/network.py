# -*- coding: utf-8 -*-
"""network.py - The definition of our network
"""
from torch import nn
from torchvision.models.resnet import resnet34


class Model(nn.Module):
    def __init__(self, n_classes):
        """The definition of our network

        Arguments:
            n_classes {int} -- The number of classes in the dataset
        """
        super(Model, self).__init__()
        self.n_features = 512   # The dimension of features before the classification layer
        self.feature_size = 8   # The size of feature map extracted from self.features
        self.resnet = resnet34(pretrained=True)
        self.features = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
            # nn.AvgPool2d(7, stride=1)     # We replace this layer by nn.AdaptiveAvgPool2d
            nn.AdaptiveAvgPool2d((self.feature_size, self.feature_size))
        )
        # Adaptation layer
        self.fc = nn.Sequential(
            nn.Linear(self.n_features * self.feature_size**2, self.n_features),
            # nn.ReLU(True) # ReLU will lead to feature disappearance, while Sigmoid will not.
            nn.Sigmoid()
        )
        self.fc[0].weight.data.normal_(0, 0.5)
        # Classfication layer
        self.classifier = nn.Linear(self.n_features, n_classes)

    def forward(self, input_cad, input_real=None):
        """Forward propagation

        Arguments:
            input_cad {torch.autograd.Variable} -- a batch of synthetic images (batch × 3 × height × width)
            input_real {torch.autograd.Variable} -- a batch of real images (batch × 3 × height × width)

        Returns:
            if input_real is not None:
                [torch.autograd.Variable] -- The features of synthetic images before the classification layer
                [torch.autograd.Variable] -- The features of real images before the classification layer
                [torch.autograd.Variable] -- The outputs of synthetic images
                [torch.autograd.Variable] -- The outputs of real images
            else:
                [torch.autograd.Variable] -- The features of images before the classification layer
                [torch.autograd.Variable] -- The outputs of images
        """
        if input_real is not None:
            input_cad, input_real = self.features(input_cad), self.features(input_real)
            input_cad, input_real = input_cad.view(input_cad.size(0), -1), input_real.view(input_real.size(0), -1)
            input_cad, input_real = self.fc(input_cad), self.fc(input_real)
            return input_cad, input_real, self.classifier(input_cad), self.classifier(input_real)
        else:
            input = self.features(input_cad)
            input = self.fc(input.view(input.size(0), -1))
            return input, self.classifier(input)
