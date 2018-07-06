# Viewpoint Estimation for Workpieces with Deep Transfer Learning from Cold to Hot

[![Build Status][badge]][link]
[![Github All Releases](https://img.shields.io/github/downloads/haotian-wang/viewpoint-estimation/total.svg)](https://github.com/haotian-wang/viewpoint-estimation/releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository is the implementation of the paper *Viewpoint Estimation for Workpieces with Deep Transfer Learning from Cold to Hot* in *ICONIP 2018*. The code is implemented based on [PyTorch][pytorch].

## Requirements

- PyTorch and Torchvision

    We have tested the compatibility with different versions of [Python][python] and [PyTorch][pytorch] based on Travis-CI.

    |                |         Python 2.7         |         Python 3.5         |         Python 3.6         |
    |:--------------:|:--------------------------:|:--------------------------:|:--------------------------:|
    | PyTorch 0.1    | [![Travis][badge1]][link]  | [![Travis][badge13]][link] | [![Travis][badge25]][link] |
    | PyTorch 0.1.6  | [![Travis][badge2]][link]  | [![Travis][badge14]][link] | [![Travis][badge26]][link] |
    | PyTorch 0.1.7  | [![Travis][badge3]][link]  | [![Travis][badge15]][link] | [![Travis][badge27]][link] |
    | PyTorch 0.1.8  | [![Travis][badge4]][link]  | [![Travis][badge16]][link] | [![Travis][badge28]][link] |
    | PyTorch 0.1.9  | [![Travis][badge5]][link]  | [![Travis][badge17]][link] | [![Travis][badge29]][link] |
    | PyTorch 0.1.10 | [![Travis][badge6]][link]  | [![Travis][badge18]][link] | [![Travis][badge30]][link] |
    | PyTorch 0.1.11 | [![Travis][badge7]][link]  | [![Travis][badge19]][link] | [![Travis][badge31]][link] |
    | PyTorch 0.1.12 | [![Travis][badge8]][link]  | [![Travis][badge20]][link] | [![Travis][badge32]][link] |
    | PyTorch 0.2.0  | [![Travis][badge9]][link]  | [![Travis][badge21]][link] | [![Travis][badge33]][link] |
    | PyTorch 0.3.0  | [![Travis][badge10]][link] | [![Travis][badge22]][link] | [![Travis][badge34]][link] |
    | PyTorch 0.3.1  | [![Travis][badge11]][link] | [![Travis][badge23]][link] | [![Travis][badge35]][link] |
    | PyTorch 0.4.0  | [![Travis][badge12]][link] | [![Travis][badge24]][link] | [![Travis][badge36]][link] |

[python]:  https://python.org
[pytorch]: https://pytorch.org
[badge]:   https://travis-ci.org/haotian-wang/viewpoint-estimation.svg?branch=master
[badge1]:  https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/1
[badge2]:  https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/2
[badge3]:  https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/3
[badge4]:  https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/4
[badge5]:  https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/5
[badge6]:  https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/6
[badge7]:  https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/7
[badge8]:  https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/8
[badge9]:  https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/9
[badge10]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/10
[badge11]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/11
[badge12]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/12
[badge13]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/13
[badge14]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/14
[badge15]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/15
[badge16]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/16
[badge17]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/17
[badge18]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/18
[badge19]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/19
[badge20]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/20
[badge21]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/21
[badge22]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/22
[badge23]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/23
[badge24]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/24
[badge25]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/25
[badge26]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/26
[badge27]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/27
[badge28]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/28
[badge29]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/29
[badge30]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/30
[badge31]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/31
[badge32]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/32
[badge33]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/33
[badge34]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/34
[badge35]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/35
[badge36]: https://travis-matrix-badges.herokuapp.com/repos/haotian-wang/viewpoint-estimation/branches/master/36
[link]: https://travis-ci.org/haotian-wang/viewpoint-estimation

- TensorboardX and Tensorboard (Optional)

    To visualize the progress of training, we use TensorboardX to plot the loss and the accuracy in Tensorboard. The libaray TensorboardX can be installed by pip.
    ```bash
    pip install tensorboardX
    ```
    Also, remember to install [Tensorflow](https://www.tensorflow.org) and [Tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard). If you don't want to use Tensorboard, simply set `writer = None` in [train.py](sources/train.py).

- Other Dependencies

    Other dependencies are written in [requirements.txt](sources/requirements.txt), you can install them by pip.
    ```bash
    cd sources
    pip install -r requirements.txt
    ```

## Dataset Preparation

The data is read according to the filelists in `dataset` directory. Your may manually edit these files, or you can also generate the filelists by running our script ([dataset/generate.py](dataset/generate.py)).

Before running our script, you should prepare your dataset with synthetic images and real images. The directory structure should be like this:

```
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
```

Then replace the following code in [generate.py](dataset/generate.py) with your own choices:

```python
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
```

Finally, run the script, and you will get the filelists in `dataset` folder.

## Train

1. Set the parameters
    Our paremeters are written in [sources/train.py](sources/train.py). You can edit it manually.
    - Replace the filelists with your own filelists:

        ```python
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
        ```
    - Set your own parameters

        ```python
        parameters = {
            'epoch': 60,
            'batch_size': 128,
            'n_classes': 7,
            'test_steps': 50,
            # Whether to use GPU?
            # None      -- CPU only
            # 0 or (0,) -- Use GPU0
            # (0, 1)    -- Use GPU0 and GPU1
            'GPUs': 0
        }
        ```
    - Tensorboard Writer

        If you have installed Tensorboard and TensorboardX, by default, we will use TensorboardX to visualize the training process. Otherwise, we will skip it. The outputs will be written to `sources/runs`. If you don't want to use Tensorboard, you may set `writer = None` in your code.

2. Train your own model
    ```bash
    cd sources
    python train.py
    ```

## Test

Replace the parameters and the dataset in [sources/test.py](sources/test.py) with our own choice just like [Train](#train), and replace the name of the pre-trained model. Then run [test.py](sources/test.py) to test your model.

```python
model.load_state_dict(torch.load('model.pth'))
```
