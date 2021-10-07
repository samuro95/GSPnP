import os
import numpy as np
import hdf5storage
from scipy import ndimage
from argparse import ArgumentParser
from utils.utils_restoration import single2uint,crop_center, matlab_style_gauss2D, imread_uint, imsave
from natsort import os_sorted
from GS_PnP_restoration import PnP_restoration



def deblur():

    parser = ArgumentParser()
    parser.add_argument('--kernel_path', type=str, default=os.path.join('kernels', 'Levin09.mat'))
    parser = PnP_restoration.add_specific_args(parser)
    hparams = parser.parse_args()

    # Deblurring specific hyperparameters
    hparams.relative_diff_F_min = 1e-5
    hparams.sigma_denoiser = 1.8 * hparams.noise_level_img
    hparams.degradation_mode = 'deblurring'

    # PnP_restoration class
    PnP_module = PnP_restoration(hparams)

    # Set input image paths
    input_path = os.path.join(hparams.dataset_path,hparams.dataset_name)
    input_path = os.path.join(input_path,os.listdir(input_path)[0])
    input_paths = os_sorted([os.path.join(input_path,p) for p in os.listdir(input_path)])

    # Output images and curves paths
    if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
        den_out_path = 'deblur'
        if not os.path.exists(den_out_path):
            os.mkdir(den_out_path)
        den_out_path = os.path.join('deblur', hparams.denoiser_name)
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
    F_list = []

    # Load the 8 motion blur kernels
    kernels = hdf5storage.loadmat(hparams.kernel_path)['kernels']

    # Kernels follow the order given in the paper (Table 2). The 8 first kernels are motion blur kernels, the 9th kernel is uniform and the 10th Gaussian.
    k_list = range(10)

    print('\n GS-DRUNET deblurring with image sigma:{:.3f}, model sigma:{:.3f}, lamb:{:.3f} \n'.format(hparams.noise_level_img, hparams.sigma_denoiser, hparams.lamb))

    for k_index in k_list: # For each kernel

        psnr_k_list = []
        psnrY_k_list = []

        if k_index == 8: # Uniform blur
            k = (1/81)*np.ones((9,9))
            hparams.lamb = 0.075
        elif k_index == 9:  # Gaussian blur
            k = matlab_style_gauss2D(shape=(25,25),sigma=1.6)
            hparams.lamb = 0.075
        else: # Motion blur
            k = kernels[0, k_index]
            hparams.lamb = 0.1

        kout_path = os.path.join(exp_out_path, 'kernel_'+str(k_index))
        if not os.path.exists(kout_path):
            os.mkdir(kout_path)

        if hparams.extract_curves:
            PnP_module.initialize_curves()

        for i in range(min(len(input_paths),hparams.n_images)): # For each image

            print('__ kernel__',k_index, '__ image__',i)

            # load image
            input_im_uint = imread_uint(input_paths[i])
            if hparams.patch_size < min(input_im_uint.shape[0], input_im_uint.shape[1]):
               input_im_uint = crop_center(input_im_uint, hparams.patch_size,hparams.patch_size)
            input_im = np.float32(input_im_uint / 255.)
            # Degrade image
            blur_im = ndimage.filters.convolve(input_im, np.expand_dims(k, axis=2), mode='wrap')
            np.random.seed(seed=0)
            noise = np.random.normal(0, hparams.noise_level_img/255., blur_im.shape)
            blur_im += noise

            # PnP restoration
            if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
                deblur_im, output_psnr, output_psnrY, x_list, z_list, Dx_list, psnr_tab, Ds_list, s_list, F_list = PnP_module.restore(blur_im,input_im,k, extract_results=True)
            else :
                deblur_im, output_psnr,output_psnrY = PnP_module.restore(blur_im,input_im,k)

            print('PSNR: {:.2f}dB'.format(output_psnr))

            psnr_k_list.append(output_psnr)
            psnrY_k_list.append(output_psnrY)
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
                imsave(os.path.join(save_im_path, 'img_'+str(i)+'_input.png'), input_im_uint)
                imsave(os.path.join(save_im_path, 'img_' + str(i) + '_deblur.png'), single2uint(deblur_im))
                imsave(os.path.join(save_im_path, 'img_'+str(i)+'_blur.png'), single2uint(blur_im))
                print('output image saved at ', os.path.join(save_im_path, 'img_' + str(i) + '_deblur.png'))

        if hparams.extract_curves:
            # Save curves
            save_curves_path = os.path.join(kout_path,'curves')
            if not os.path.exists(save_curves_path):
                os.mkdir(save_curves_path)
            PnP_module.save_curves(save_curves_path)
            print('output curves saved at ', save_curves_path)

        avg_k_psnr = np.mean(np.array(psnr_k_list))
        print('avg RGB psnr on kernel {}: {:.2f}dB'.format(k_index, avg_k_psnr))
        avg_k_psnrY = np.mean(np.array(psnrY_k_list))
        print('avg Y psnr on kernel {} : {:.2f}dB'.format(k_index, avg_k_psnrY))


if __name__ == '__main__':
    deblur()
