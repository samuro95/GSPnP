import os
import numpy as np
import hdf5storage
from scipy import ndimage
from argparse import ArgumentParser
from utils.utils_restoration import rescale, array2tensor, tensor2array, get_gaussian_noise_parameters, create_out_dir, single2uint,crop_center, matlab_style_gauss2D, imread_uint, imsave
from natsort import os_sorted
from GS_PnP_restoration import PnP_restoration
import wandb
import cv2

def deblur():

    parser = ArgumentParser()
    parser.add_argument('--kernel_path', type=str)
    parser.add_argument('--kernel_indexes', nargs='+', type=int)
    parser.add_argument('--image_path', type=str)
    parser = PnP_restoration.add_specific_args(parser)
    hparams = parser.parse_args()

    # Deblurring specific hyperparameters

    hparams.degradation_mode = 'deblurring'

    # PnP_restoration class
    PnP_module = PnP_restoration(hparams)

    # Set input image paths
    if hparams.image_path is not None : # if a specific image path is given
        input_paths = [hparams.image_path]
        hparams.dataset_name = os.path.splitext(os.path.split(hparams.image_path)[-1])[0]
    else : # if not given, we aply on the whole dataset name given in argument 
        input_path = os.path.join(hparams.dataset_path,hparams.dataset_name)
        input_paths = os_sorted([os.path.join(input_path,p) for p in os.listdir(input_path)])

    psnr_list = []
    F_list = []

    if hparams.kernel_path is not None : # if a specific kernel saved in hparams.kernel_path as np array is given 
        k_list = [np.load(hparams.kernel_path)]
        k_index_list = [0]
    else : 
        k_list = []
        # If no specific kernel is given, load the 8 motion blur kernels
        kernel_path = os.path.join('kernels', 'Levin09.mat')
        kernels = hdf5storage.loadmat(kernel_path)['kernels']
        # Kernels follow the order given in the paper (Table 2). The 8 first kernels are motion blur kernels, the 9th kernel is uniform and the 10th Gaussian.
        for k_index in range(10) :
            if k_index == 8: # Uniform blur
                k = (1/81)*np.ones((9,9))
            elif k_index == 9:  # Gaussian blur
                k = matlab_style_gauss2D(shape=(25,25),sigma=1.6)
            else : # Motion blur
                k = kernels[0, k_index]
            k_list.append(k)

        if hparams.kernel_indexes is not None : 
            k_index_list = hparams.kernel_indexes
        else :
            k_index_list = range(len(k_list))

    if hparams.use_wandb :
        wandb.init(project=hparams.degradation_mode)        
    data = []

    for k_index in k_index_list : # For each kernel

        psnr_k_list = []
        n_it_list = []

        k = k_list[k_index]

        if hparams.extract_curves:
            PnP_module.initialize_curves()

        PnP_module.hparams.lamb, PnP_module.hparams.sigma_denoiser, PnP_module.hparams.maxitr, PnP_module.hparams.thres_conv = get_gaussian_noise_parameters(
                                hparams.noise_level_img, k_index=k_index, degradation_mode='deblur')

        print('GS-DRUNET deblurring with image sigma:{:.3f}, model sigma:{:.3f}, lamb:{:.3f} \n'.format(hparams.noise_level_img, hparams.sigma_denoiser, hparams.lamb))

        if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
            exp_out_path = create_out_dir(hparams.degradation_mode, hparams.dataset_name)
            if len(k_index_list) > 1:
                exp_out_path = os.path.join(exp_out_path, "kernel_"+str(k_index))
                if not os.path.exists(exp_out_path):
                    os.mkdir(exp_out_path)

        for i in range(min(len(input_paths),hparams.n_images)): # For each image

            print('Deblurring of image {}, kernel index {}'.format(i,k_index))

            np.random.seed(seed=0)
            
            # load image
            input_im_uint = imread_uint(input_paths[i])
            input_im = np.float32(input_im_uint / 255.)
            if hparams.grayscale : 
                input_im = cv2.cvtColor(input_im, cv2.COLOR_BGR2GRAY)
                input_im = np.expand_dims(input_im, axis = 2)
            # Degrade image
            blur_im = ndimage.convolve(input_im, np.expand_dims(k, axis=2), mode='wrap')
            noise = np.random.normal(0, hparams.noise_level_img / 255., blur_im.shape)
            blur_im += noise
            blur_im =  np.float32(blur_im)
            init_im = blur_im

            # PnP restoration
            if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
                deblur_im, init_im, output_psnr, output_ssim, n_it, x_list, z_list, Dg_list, psnr_tab, ssim_tab, g_list, F_list, f_list = PnP_module.restore(blur_im.copy(),init_im.copy(),input_im.copy(),k, extract_results=True)
            else :
                deblur_im, init_im, output_psnr, output_ssim, n_it = PnP_module.restore(blur_im,init_im,input_im,k)

            print('PSNR: {:.2f}dB'.format(output_psnr))
            print('SSIM: {:.2f}'.format(output_ssim))
            print(f'N iterations: {n_it}')
            
            psnr_k_list.append(output_psnr)
            psnr_list.append(output_psnr)
            n_it_list.append(n_it)

            if hparams.extract_curves:
                # Create curves
                PnP_module.update_curves(x_list, psnr_tab, ssim_tab, Dg_list, g_list, F_list, f_list)

            if hparams.extract_images:
                # Save images
                save_im_path = os.path.join(exp_out_path, 'images')
                if not os.path.exists(save_im_path):
                    os.mkdir(save_im_path)
                imsave(os.path.join(save_im_path, 'img_'+str(i)+'_input.png'), input_im_uint)
                imsave(os.path.join(save_im_path, 'img_' + str(i) + "_deblur.png"), single2uint(rescale(deblur_im)))
                imsave(os.path.join(save_im_path, 'img_'+str(i)+'_blur.png'), single2uint(rescale(blur_im)))
                imsave(os.path.join(save_im_path, 'img_' + str(i) + '_init.png'), single2uint(rescale(init_im)))
                print('output image saved at ', os.path.join(save_im_path, 'img_' + str(i) + '_deblur.png'))

        if hparams.extract_curves:
            # Save curves
            save_curves_path = os.path.join(exp_out_path,'curves')
            if not os.path.exists(save_curves_path):
                os.mkdir(save_curves_path)
            PnP_module.save_curves(save_curves_path)
            print('output curves saved at ', save_curves_path)

        avg_k_psnr = np.mean(np.array(psnr_k_list))
        print('avg RGB psnr on kernel {}: {:.2f}dB'.format(k_index, avg_k_psnr))

        data.append([k_index, avg_k_psnr, np.mean(np.mean(n_it_list))])
    
    data = np.array(data)

    if hparams.use_wandb :
        table = wandb.Table(data=data, columns=['k', 'psnr', 'n_it'])
        for i, metric in enumerate(['psnr', 'n_it']):
            wandb.log({
                f'{metric}_plot': wandb.plot.scatter(
                    table, 'k', metric,
                    title=f'{metric} vs. k'),
                f'average_{metric}': np.mean(data[:,i+1])
            },
            step = 0)
    
    return np.mean(np.array(psnr_list))

if __name__ == '__main__':
    deblur()
