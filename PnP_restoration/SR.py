import os
import cv2
import numpy as np
import hdf5storage
from collections import OrderedDict
from argparse import ArgumentParser
from GS_PnP_restoration import PnP_restoration
from utils.utils_restoration import single2uint,crop_center, imread_uint, imsave, modcrop
from natsort import os_sorted
from utils.utils_sr import classical_degradation

def SR():

    parser = ArgumentParser()
    parser.add_argument('--sf', type=int, default=2)
    parser.add_argument('--kernel_path', type=str, default=os.path.join('kernels','kernels_12.mat'))
    parser = PnP_restoration.add_specific_args(parser)
    hparams = parser.parse_args()

    # SR specific hyperparameters
    hparams.degradation_mode = 'SR'
    hparams.classical_degradation = True
    hparams.relative_diff_F_min = 1e-6
    hparams.lamb = 0.065
    hparams.sigma_denoiser = 2*hparams.noise_level_img

    # PnP_restoration class
    PnP_module = PnP_restoration(hparams)

    # Set input image paths
    input_path = os.path.join(hparams.dataset_path,hparams.dataset_name)
    input_path = os.path.join(input_path,os.listdir(input_path)[0])
    input_paths = os_sorted([os.path.join(input_path,p) for p in os.listdir(input_path)])

    # Output images and curves paths
    den_out_path = 'SR'
    if not os.path.exists(den_out_path):
        os.mkdir(den_out_path)
    den_out_path = os.path.join('SR', hparams.denoiser_name)
    if not os.path.exists(den_out_path):
        os.mkdir(den_out_path)
    exp_out_path = os.path.join(den_out_path, hparams.PnP_algo)
    if not os.path.exists(exp_out_path):
        os.mkdir(exp_out_path)
    exp_out_path = os.path.join(exp_out_path, hparams.dataset_name)
    if not os.path.exists(exp_out_path):
        os.mkdir(exp_out_path)
    exp_out_path = os.path.join(exp_out_path, str(hparams.noise_level_img))
    if not os.path.exists(exp_out_path):
        os.mkdir(exp_out_path)

    psnr_list = []
    psnr_list_sr = []
    F_list = []

    # Load the 8 blur kernels
    kernels = hdf5storage.loadmat(hparams.kernel_path)['kernels']
    # Kernels follow the order given in the paper (Table 3)
    k_list = range(8)

    print('\n GS-DRUNET super-resolution with image sigma:{:.3f}, model sigma:{:.3f}, lamb:{:.3f} \n'.format(hparams.noise_level_img, hparams.sigma_denoiser, hparams.lamb))

    for k_index in k_list: # For each kernel

        psnr_k_list = []

        k = kernels[0, k_index].astype(np.float64)

        if hparams.extract_images or hparams.extract_curves:
            kout_path = os.path.join(exp_out_path, 'kernel_'+str(k_index))
            if not os.path.exists(kout_path):
                os.mkdir(kout_path)

        if hparams.extract_curves:
            PnP_module.initialize_curves()

        for i in range(min(len(input_paths),hparams.n_images)) : # For each image

            print('__ kernel__',k_index, '__ image__',i)

            # load image
            input_im_uint = imread_uint(input_paths[i])
            if hparams.patch_size < min(input_im_uint.shape[0], input_im_uint.shape[1]):
               input_im_uint = crop_center(input_im_uint, hparams.patch_size,hparams.patch_size)
            # Degrade image
            input_im_uint = modcrop(input_im_uint, hparams.sf)
            input_im = np.float32(input_im_uint / 255.)
            if classical_degradation:
                blur_im = classical_degradation(input_im, k, hparams.sf)
            else:
                print('not implemented yet')
            np.random.seed(seed=0)
            noise = np.random.normal(0, hparams.noise_level_img/255., blur_im.shape)
            blur_im += noise

            # PnP restoration
            if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
                deblur_im, output_psnr, output_psnrY, x_list, z_list, Dx_list, psnr_tab, Ds_list, s_list, F_list = PnP_module.restore(blur_im,input_im,k, extract_results=True)
            else :
                deblur_im, output_psnr, output_psnrY = PnP_module.restore(blur_im,input_im,k)

            print('PSNR: {:.2f}dB'.format(output_psnr))

            psnr_k_list.append(output_psnr)
            psnr_list.append(output_psnr)

            if hparams.extract_curves:
                # Create curves
                PnP_module.update_curves(x_list, z_list, Dx_list, psnr_tab, Ds_list, s_list, F_list)

            if hparams.extract_images:
                # Save images
                save_im_path = os.path.join(kout_path, 'images')
                if not os.path.exists(save_im_path):
                    os.mkdir(save_im_path)

                imsave(os.path.join(save_im_path, 'kernel_' + str(k_index) + '.png'), single2uint(deblur_im))
                imsave(os.path.join(save_im_path, 'img_' + str(i) + '_input.png'), input_im_uint)
                imsave(os.path.join(save_im_path, 'img_' + str(i) + '_HR.png'), single2uint(deblur_im))
                imsave(os.path.join(save_im_path, 'img_' + str(i) + '_GSPnP.png'), single2uint(blur_im))
                print('output image saved at ', os.path.join(save_im_path, 'img_' + str(i) + '_GSPnP.png'))

        if hparams.extract_curves:
            # Save curves
            save_curves_path = os.path.join(kout_path, 'curves')
            if not os.path.exists(save_curves_path):
                os.mkdir(save_curves_path)
            PnP_module.save_curves(save_curves_path)
            print('output curves saved at ', save_curves_path)

        avg_k_psnr = np.mean(np.array(psnr_k_list))
        print('avg RGB psnr on kernel {}: {:.2f}dB'.format(k_index, avg_k_psnr))

        psnr_list_sr.append(avg_k_psnr)

        if k_index == 3:
            print('------ avg RGB psnr on isotropic kernels : {:.2f}dB ------'.format(np.mean(np.array(psnr_list_sr))))
            psnr_list_sr = []
        if k_index == 7:
            print('------ avg RGB psnr on anisotropic kernel : {:.2f}dB ------'.format(np.mean(np.array(psnr_list_sr))))
            psnr_list_sr = []


if __name__ == '__main__':
    SR()