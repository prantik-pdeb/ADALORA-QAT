#### This folder "baselines" contains the python code for image segmentation models Deeplabv3+ and Segformer. The code to train the model, if needed and to perform inference.

To use the code, you could clone this repository and make changes file path in the code and run the code. The way to use the code files is given below.

#### How to use the code files?
##### Deeplabv3+:
1. To train the model, use [Deeplabv3+/train.py](./Deeplabv3+/train.py). The folder structure of the dataset is mentioned after this section. NOTE: The folder structure of the dataset must be the same as mentioned, else the script will result in error.
```
python train.py \
  --dataset_root /path/to/test/images_rgb/Final_dataset_split_Resized \
  --batch_size 32 \
  --lr 1e-4 \
  --max_epochs 100 \
  --num_workers 4
```
2. To perform inference of the trained model, use [Deeplabv3+/inference.py](./Deeplabv3+/inference.py).

```
python inference.py \
  --model_path /path/to/latest_checkpoint.pt \
  --val_images_dir /path/to/test/images_rgb \
  --val_masks_dir /path/to/test/masks_binary \
  --save_preds_dir /path/to/save/predictions \
  --batch_size 8 \
  --tolerance_mm 2.0
```

##### Segformer:
1. To train the model, use [Segformer/train.py](./Segformer/train.py). This file uses command line arguments, so we don't need to change any constants in the script. Below is an example of how to use the script,
```
python baselines/Segformer/train.py \
--data_root /path/to/data/
```
The only argument that is needed is the path to the dataset. Other arguments have defaults, you may look into the python code, if you want to change the values of these arguments.

2. For testing the trained model, use [Segformer/inference.py](./Segformer/inference.py). Below is an example of how to use the script,
```
python baselines/Segformer/inference.py \
  --model_path /path/to/best_model.ckpt \
  --val_images_dir /path/to/test/images_rgb \
  --val_masks_dir /path/to/test/masks_binary \
  --save_preds_dir ./predictions \
  --batch_size 4 \
  --tolerance_mm 2.0
```

##### nnUNet:
As for the code for nnUNet, we insist that you refer the original nnUNet repository. For training the model and performing inference of the model we followed the instructions given in the original repository, we also encourage you to do the same.
nnUNet repository link: [https://github.com/MIC-DKFZ/nnUNet.git](https://github.com/MIC-DKFZ/nnUNet.git)

#### Dataset folder structure:
```
├── test
│   ├── images
│   └── masks
├── train
│   ├── images
│   └── masks
└── val
    ├── images
    └── masks
```
**NOTE: It is important the images in the "images" folder must be in RGB format and the images in the "masks" folder must be in binary format i.e., the pixel values should be either 0 or 1, nothing else. All the scripts need the data to be in the above mentioned format.**