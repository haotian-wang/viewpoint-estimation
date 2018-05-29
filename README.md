Viewpoint Estimation for Workpieces with Deep Transfer Learning from Cold to Hot
----------------------------------------

This repository is the implementation of the paper *Viewpoint Estimation for Workpieces with Deep Transfer Learning from Cold to Hot* in *ICONIP 2018*. The code is implemented based on [PyTorch](https://pytorch.org/).

----------------------------------------

### Dataset Preparation

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

### Train

1. Install the requirements 
    ```bash
    cd sources
    pip install -r Requirements.txt
    ```  
    **Note:** Currently, we support both `Python 2` and `Python 3`. If your `PyTorch` version is `0.4.0` or newer, you should replace all `.data[0]` to `.item()` in our codes.  
    If you want to use Tensorboard while training, you need to install Tensorboard and TensorboardX (a library to write summaries in Tensorboard for PyTorch).  
    ```bash
    pip install tensorflow tensorboard tensorboardX
    ```

2. Set the parameters  
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
            'test_steps': 50
        }
        ```  
    - Data Parallel  
    If you have multiple GPUs, you may use `torch.nn.DataParallel` to boost your performance. To use Data Parallel, you can uncomment the following code, and set `device_ids` to your available GPUs.  
        ```python
        model = nn.DataParallel(model, device_ids=(0, 1))
        ```  
    - Tensorboard Writer  
    If you have installed Tensorboard and TensorboardX, by default, we will use TensorboardX to visualize the training process. Otherwise, we will skip it. The outputs will be written to `sources/runs`. If you don't want to use Tensorboard, you may set `writer = None` in your code.

3. Train your own model  
    ```bash
    cd sources
    python train.py
    ```

### Test

Replace the parameters and the dataset in [sources/test.py](sources/test.py) with our own choice just like [Train](#train), and replace the name of the pre-trained model. Then run [test.py](sources/test.py) to test your model.  
```python
model.load_state_dict(torch.load('model.pth'))
```  