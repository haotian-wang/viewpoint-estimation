# -*- coding: utf-8 -*-
"""generate.py - Generate the filelists of datasets.

Please edit the following parameters `LABEL` and `root` before running this code.
The directory structure should be like this:
root
├─ wp1
│    ├─ synthetic
│    │    ├─ 1.90_0
│    │    │    ├─ xxx.png
│    │    │    └─ ...
│    │    ├─ 3.45_90
│    │    │    ├─ xxx.png
│    │    │    └─ ...
│    │    └─ ...
│    └─ real
│         ├─ 1.90_0
│         │    ├─ xxx.png
│         │    └─ ...
│         ├─ 3.45_90
│         │    ├─ xxx.png
│         │    └─ ...
│         └─ ...
├─ wp2
│    ├─ ...
│    └─ ...
└─ ...
"""
import os
import random

# Please edit the root directory of your dataset here
root = '/home/nip/LabDatasets/WorkPieces'
# Please edit your label mapping here
LABEL = {
    '1.90_0': 0,
    '3.45_90': 1,
    '5.45_270': 2,
    '6.0_0': 3,
    '7.0_90': 4,
    '8.0_180': 5,
    '9.0_270': 6
}
# Please edit your workpieces names here
WP = ['wp{}'.format(i) for i in range(1, 9)]


def get_filelists(srcdir, shuffle=True):
    """Get filelists of the dataset

    Arguments:
        srcdir {str} -- Root directory of synthetic or real datasets

    Keyword Arguments:
        shuffle {bool} -- Whether to shuffle the filelist (default: {True})

    Returns:
        {[(str, int)]} -- A list, in which each element is a tuple.
                        The first element of the tuple is the file path, and the second is the corresponding label.
    """
    dataset = [
        (os.path.join(os.path.join(srcdir, dirname), filename), LABEL[dirname])
        for dirname in os.listdir(srcdir)
        for filename in os.listdir(os.path.join(srcdir, dirname))
    ]
    if shuffle:
        random.shuffle(dataset)
    return dataset


if __name__ == '__main__':
    for wp in WP:
        srcdircad = os.path.join(root, '{}/synthetic'.format(wp))
        srcdirreal = os.path.join(root, '{}/real'.format(wp))

        cadset = get_filelists(srcdircad, shuffle=True)
        realset = get_filelists(srcdirreal, shuffle=True)

        cadsettxt = '\n'.join(('{} {}'.format(filename, label) for filename, label in cadset))
        realsettxt = '\n'.join(('{} {}'.format(filename, label) for filename, label in realset))

        with open('{}_cad.txt'.format(wp), 'w') as fout:
            fout.write(cadsettxt)
        with open('{}_real.txt'.format(wp), 'w') as fout:
            fout.write(realsettxt)
