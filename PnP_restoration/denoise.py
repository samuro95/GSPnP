import os
import numpy as np
from argparse import ArgumentParser

from GS_PnP_restoration import PnP_restoration
from utils.utils_restoration import single2uint, crop_center, imread_uint, imsave
from natsort import os_sorted
from utils.utils_restoration import psnr, array2tensor, tensor2array


def denoise():

    parser = ArgumentParser()
    parser.add_argument('--image_path', type=str)
    parser = PnP_restoration.add_specific_args(parser)
    hparams = parser.parse_args()

    # Denoising specific hyperparameters
    hparams.degradation_mode = 'denoising'

    # PnP_restoration class
    PnP_module = PnP_restoration(hparams)

    # Set input image paths
    if hparams.image_path is not None : # if a specific image path is given
        input_paths = [hparams.image_path]
        hparams.dataset_name = os.path.splitext(os.path.split(hparams.image_path)[-1])[0]
    else : # if not given, we aply on the whole dataset name given in argument 
        input_path = os.path.join(hparams.dataset_path,hparams.dataset_name)
        input_paths = os_sorted([os.path.join(input_path,p) for p in os.listdir(input_path)])

    # Output images and curves paths
    if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
        den_out_path = 'denoise'
        if not os.path.exists(den_out_path):
            os.mkdir(den_out_path)
        exp_out_path = os.path.join(den_out_path, hparams.dataset_name)
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
        exp_out_path = os.path.join(exp_out_path, str(hparams.noise_level_img))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)

    print('\n GS-DRUNET denoising with image sigma:{:.3f} \n'.format(hparams.noise_level_img))

    psnr_list = []

    for i in range(min(len(input_paths), hparams.n_images)): # For each image

        print('__ image__', i)

        # load image
        input_im_uint = imread_uint(input_paths[i])
        input_im = np.float32(input_im_uint / 255.)
        # Degrade image
        np.random.seed(seed=0)
        noise = np.random.normal(0, hparams.noise_level_img / 255., input_im.shape)
        noise_im = input_im + noise
        noise_im_tensor = array2tensor(noise_im).to(PnP_module.device)

        # Denoise
        Dx,g,Dg = PnP_module.denoise(noise_im_tensor, hparams.noise_level_img / 255)

        denoise_img = tensor2array(Dx.cpu())
        psnri = psnr(denoise_img, input_im)

        psnr_list.append(psnri)

        if hparams.extract_images:
            # Save images
            save_im_path = os.path.join(exp_out_path, 'images')
            if not os.path.exists(save_im_path):
                os.mkdir(save_im_path)
            imsave(os.path.join(save_im_path, 'img_' + str(i) + '_input.png'), input_im_uint)
            imsave(os.path.join(save_im_path, 'img_' + str(i) + '_noise.png'), single2uint(noise_im))
            imsave(os.path.join(save_im_path, 'img_' + str(i) + '_denoise.png'), single2uint(denoise_img))
            print('output image saved at ', os.path.join(save_im_path, 'img_' + str(i) + '_denoise.png'))

    avg_psnr = np.mean(np.array(psnr_list))
    print('avg RGB psnr for sigma={}: {:.2f}dB'.format(hparams.noise_level_img, avg_psnr))


if __name__ == '__main__':
    denoise()
