# -*- coding: utf-8 -*-
"""preprocessing.py
"""
from PIL import Image


class Scale(object):
    def __init__(self, scale, interpolation=Image.BILINEAR):
        """Resize the image to scale times, without changing the ratio of length and width

        Arguments:
            scale {float} -- The scale to resize

        Keyword Arguments:
            interpolation {int} -- interpolation method (default: {PIL.Image.BILINEAR})

        Examples:
            >>> scale = Scale(2)
            >>> image = Image.open('image.png')
            >>> image = scale(image)
        """
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, image):
        """Apply resizing to the image

        Arguments:
            image {Image.Image} -- The image to be resized

        Returns:
            Image.Image -- The resized image
        """
        return image.resize(
            (int(image.size[0] * self.scale), int(image.size[1] * self.scale)),
            resample=self.interpolation
        )


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        """Resize the image to the target size
        This callable object is used for the compatibility with the lower version of PyTorch

        Arguments:
            scale {{int,int}} -- Target size (width × height)

        Keyword Arguments:
            interpolation {int} -- interpolation method (default: {Image.BILINEAR})

        Examples:
            >>> resize = Resize(size=(800, 600))
            >>> image = Image.open('image.png')
            >>> image = resize(image)
        """
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image):
        return image.resize(self.size, resample=self.interpolation)
