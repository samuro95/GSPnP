# Gradient Step Denoiser for convergent Plug-and-Play

Code for the paper "Gradient Step Denoiser for convergent Plug-and-Play" published at ICLR 2022. 

[[Paper](https://openreview.net/pdf?id=fPhKeld3Okz)]

[Samuel Hurault](https://www.math.u-bordeaux.fr/~shurault/), [Arthur Leclaire](https://www.math.u-bordeaux.fr/~aleclaire/), [Nicolas Papadakis](https://www.math.u-bordeaux.fr/~npapadak/). \
[Institut de Mathématiques de Bordeaux](https://www.math.u-bordeaux.fr/imb/spip.php), France.


## Prerequisites


The code was computed with Python 3.8.10, PyTorch Lightning 1.2.6, PyTorch 1.7.1

```
pip install -r requirements.txt
```

## Gradient Step Denoiser (GS-DRUNet)

The code relative to the Gradient Step Denoiser can be found in the ```GS_denoising``` directory.

### Training 

- Download training dataset from https://drive.google.com/file/d/1WVTgEBZgYyHNa2iVLUYwcrGWZ4LcN4--/view?usp=sharing and unzip ```DRUNET``` in the ```datasets``` folder
- Realize training
```
cd GS_denoising
python main_train.py --name experiment_name --log_folder logs
```
Checkpoints, tensorboard events and hyperparameters will be saved in the ```GS_denoising/logs/experiment_name``` subfolder. 

For denoising grayscale images, add the argument --grayscale

### Testing 

- Download pretrained checkpoint https://plmbox.math.cnrs.fr/f/ab6829cb933743848bef/?dl=1  for color denoising or https://plmbox.math.cnrs.fr/f/04318d36824443a6bf8d/?dl=1 for grayscale denoising and save it as ```GS_denoising/ckpts/GSDRUNet.ckpt```

- For denoising a single (clean) image IMAGE_PATH, taht will be noised with input Gaussian noise level NOISE_LEVEL (```int``` $\in [0,255]$) :
```
cd PnP_restoration
python denoise.py --image_path IMAGE_PATH --noise_level_img NOISE_LEVEL
```

- For denoising a set of (clean) images. Place your images in directory ```datasets/DATASET_NAME``` 
```
cd PnP_restoration
python denoise.py --dataset_name DATASET_NAME
```
Datasets CBSD68, CBSD10, set3c are already present in the directory. Default value is CBSD10. 


## Gradient Step PnP (GS-PnP)

### Deblurring

- Download pretrained checkpoint https://plmbox.math.cnrs.fr/f/ab6829cb933743848bef/?dl=1  for color denoising or https://plmbox.math.cnrs.fr/f/04318d36824443a6bf8d/?dl=1 for grayscale denoising and save it as ```GS_denoising/ckpts/GSDRUNet.ckpt```

- For deblurring an input (clean) image IMAGE_PATH, that we blur with kernel saved at KERNEL_PATH (saved as ```.npy```) and with input Gaussian noise level NOISE_LEVEL (```int``` $\in [0,255]$)

```
cd PnP_restoration
python deblur.py --image_path IMAGE_PATH --kernel_path KERNEL_PATH --noise_level_img NOISE_LEVEL
```

By default, without specifying ```--kernel_path ```, deblurring will be performed on the 10 kernels evaluated in the paper. You can specify  ```--kernel_index``` to choose a specific kernel in this list. 

You can also specify ```--dataset_name``` to treat a set of images places in in directory ```datasets/DATASET_NAME``` 

Add the argument ```--extract_curves``` the save convergence curves.


### Super-resolution

For performing super-resolution of the input image IMAGE_PATH, downscaled with scale ```sf```, Gaussian noise level NOISE_LEVEL (```int``` $\in [0,255]$)

```
cd PnP_restoration
python SR.py --noise_level_img NOISE_LEVEL --sf 2
```

By default, without specifying ```--kernel_path ```, deblurring will be performed on the 8 kernels evaluated in the paper. You can specify  ```--kernel_index``` to choose a specific kernel in this list. 

You can also specify ```--dataset_name``` to treat a set of images places in in directory ```datasets/DATASET_NAME``` 


Add the argument ```--extract_curves``` the save convergence curves.

### Inpainting
Inpainting with randomly masked pixels (with probability ```prop_mask = 0.5```) :
```
cd PnP_restoration
python inpaint.py --prop_mask 0.5
```

Add the argument ```--extract_curves``` the save convergence curves.

## Acknowledgments

This repo contains parts of code taken from : 
- Deep Plug-and-Play Image Restoration (DPIR) : https://github.com/cszn/DPIR 

## Citation 
```
@inproceedings{
hurault2022gradient,
title={Gradient Step Denoiser for convergent Plug-and-Play},
author={Samuel Hurault and Arthur Leclaire and Nicolas Papadakis},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=fPhKeld3Okz}
}

```
