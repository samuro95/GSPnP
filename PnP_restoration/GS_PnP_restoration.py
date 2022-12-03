import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import utils_sr
import torch
from argparse import ArgumentParser
from utils.utils_restoration import rgb2y, psnr, array2tensor, tensor2array
import sys
from matplotlib.ticker import MaxNLocator


class PnP_restoration():

    def __init__(self, hparams):

        self.hparams = hparams
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialize_cuda_denoiser()

    def initialize_cuda_denoiser(self):
        '''
        Initialize the denoiser model with the given pretrained ckpt
        '''
        sys.path.append('../GS_denoising/')
        from lightning_GSDRUNet import GradMatch
        parser2 = ArgumentParser(prog='utils_restoration.py')
        parser2 = GradMatch.add_model_specific_args(parser2)
        parser2 = GradMatch.add_optim_specific_args(parser2)
        hparams = parser2.parse_known_args()[0]
        hparams.act_mode = self.hparams.act_mode_denoiser
        self.denoiser_model = GradMatch(hparams)
        checkpoint = torch.load(self.hparams.pretrained_checkpoint, map_location=self.device)
        self.denoiser_model.load_state_dict(checkpoint['state_dict'])
        self.denoiser_model.eval()
        for i, v in self.denoiser_model.named_parameters():
            v.requires_grad = False
        self.denoiser_model = self.denoiser_model.to(self.device)
        if self.hparams.precision == 'double' :
            if self.denoiser_model is not None:
                self.denoiser_model.double()

    def initialize_prox(self, img, degradation):
        '''
        calculus for future prox computatations
        :param img: degraded image
        :param degradation: 2D blur kernel for deblurring and SR, mask for inpainting
        '''
        if self.hparams.degradation_mode == 'deblurring':
            self.k = degradation
            self.k_tensor = array2tensor(np.expand_dims(self.k, 2)).double().to(self.device)
            self.FB, self.FBC, self.F2B, self.FBFy = utils_sr.pre_calculate(img, self.k_tensor, 1)
        elif self.hparams.degradation_mode == 'SR':
            self.k = degradation
            self.k_tensor = array2tensor(np.expand_dims(self.k, 2)).double().to(self.device)
            self.FB, self.FBC, self.F2B, self.FBFy = utils_sr.pre_calculate(img, self.k_tensor, self.hparams.sf)
        elif self.hparams.degradation_mode == 'inpainting':
            self.M = array2tensor(degradation).double().to(self.device)
            self.My = self.M*img
        else:
            print('degradation mode not treated')

    def calculate_prox(self, img):
        '''
        Calculation of the proximal mapping of the data term f
        :param img: input for the prox
        :return: prox_f(img)
        '''
        if self.hparams.degradation_mode == 'deblurring':
            rho = torch.tensor([1/self.tau]).double().repeat(1, 1, 1, 1).to(self.device)
            px = utils_sr.data_solution(img.double(), self.FB, self.FBC, self.F2B, self.FBFy, rho, 1)
        elif self.hparams.degradation_mode == 'SR':
            rho = torch.tensor([1 / self.tau]).double().repeat(1, 1, 1, 1).to(self.device)
            px = utils_sr.data_solution(img.double(), self.FB, self.FBC, self.F2B, self.FBFy, rho, self.hparams.sf)
        elif self.hparams.degradation_mode == 'inpainting':
            if self.hparams.noise_level_img > 1e-2:
                px = (self.tau*self.My + img)/(self.tau*self.M+1)
            else :
                px = self.My + (1-self.M)*img
        else:
            print('degradation mode not treated')
        return px

    def calculate_F(self,x,s,img):
        '''
        Calculation of the objective function value f + lamb*s
        :param x: Point where to evaluate F
        :param s: Precomputed regularization function value
        :param img: Degraded image
        :return: F(x)
        '''
        if self.hparams.degradation_mode == 'deblurring':
            deg_x = utils_sr.imfilter(x.double(),self.k_tensor[0].double().flip(1).flip(2).expand(3,-1,-1,-1))
            F = 0.5 * torch.norm(img - deg_x, p=2) ** 2 + self.hparams.lamb * s
        elif self.hparams.degradation_mode == 'SR':
            deg_x = utils_sr.imfilter(x.double(), self.k_tensor[0].double().flip(1).flip(2).expand(3, -1, -1, -1))
            deg_x = deg_x[...,0::self.hparams.sf, 0::self.hparams.sf]
            F = 0.5 * torch.norm(img - deg_x, p=2) ** 2 + self.hparams.lamb * s
        elif self.hparams.degradation_mode == 'inpainting':
            deg_x = self.M*x.double()
            F = 0.5*torch.norm(img - deg_x, p=2) ** 2 + self.hparams.lamb * s
        else :
            print('degradation not implemented')
        return F.item()

    def restore(self, img, clean_img, degradation,extract_results=False):
        '''
        Compute GS-PnP restoration algorithm
        :param img: Degraded image
        :param clean_img: ground-truth clean image
        :param degradation: 2D blur kernel for deblurring and SR, mask for inpainting
        :param extract_results: Extract information for subsequent image or curve saving
        '''

        if extract_results:
            z_list, x_list, Dx_list, psnr_tab, s_list, Ds_list, F_list = [], [], [], [], [], [], []

        # initalize parameters
        if self.hparams.tau is not None:
            self.tau = self.hparams.tau
        else:
            self.tau = 1 / self.hparams.lamb

        i = 0 # iteration counter

        img_tensor = array2tensor(img).to(self.device) # for GPU computations (if GPU available)
        self.initialize_prox(img_tensor, degradation) # prox calculus that can be done outside of the loop

        # Initialization of the algorithm
        if self.hparams.degradation_mode == 'SR' :
            x0 = cv2.resize(img, (img.shape[1] * self.hparams.sf, img.shape[0] * self.hparams.sf),interpolation=cv2.INTER_CUBIC)
            x0 = utils_sr.shift_pixel(x0, self.hparams.sf)
            x0 = array2tensor(x0).to(self.device)
        else:
            x0 = img_tensor
        x0 = self.calculate_prox(x0)

        if extract_results:  # extract np images and PSNR values
            out_x = tensor2array(x0.cpu())
            current_x_psnr = psnr(clean_img, out_x)
            if self.hparams.print_each_step:
                print('current x PSNR : ', current_x_psnr)
            psnr_tab.append(current_x_psnr)
            x_list.append(out_x)

        x = x0

        diff_F = 1
        F_old = 1
        self.relative_diff_F_min = self.hparams.relative_diff_F_min

        while i < self.hparams.maxitr and abs(diff_F)/F_old > self.relative_diff_F_min:

            if self.hparams.inpainting_init :
                if i < self.hparams.n_init:
                    self.sigma_denoiser = 50
                    self.relative_diff_F_min = 0
                else :
                    self.sigma_denoiser = self.hparams.sigma_denoiser
                    self.relative_diff_F_min = self.hparams.relative_diff_F_min
            else :
                self.sigma_denoiser = self.hparams.sigma_denoiser

            x_old = x

            #Denoising of x_old and calculation of F_old
            Ds, f = self.denoiser_model.calculate_grad(x_old, self.sigma_denoiser / 255.)
            Ds = Ds.detach()
            f = f.detach()
            Dx = x_old - self.denoiser_model.hparams.weight_Ds * Ds
            s_old = 0.5 * (torch.norm(x_old.double() - f.double(), p=2) ** 2)
            F_old = self.calculate_F(x_old, s_old, img_tensor)

            backtracking_check = False

            while not backtracking_check:

                # Gradient step
                z = (1 - self.hparams.lamb * self.tau) * x_old + self.hparams.lamb * self.tau * Dx

                # Proximal step
                x = self.calculate_prox(z)

                # Calculation of Fnew
                f = self.denoiser_model.calculate_grad(x, self.sigma_denoiser / 255.)[1]
                f = f.detach()
                s = 0.5 * (torch.norm(x.double() - f.double(), p=2) ** 2)
                F_new = self.calculate_F(x,s,img_tensor)

                # Backtracking
                diff_x = (torch.norm(x - x_old, p=2) ** 2).item()
                diff_F = F_old - F_new
                if self.hparams.degradation_mode == 'inpainting':
                    diff_F = 1
                    F_old = 1
                if self.hparams.use_backtracking and diff_F < (self.hparams.gamma / self.tau) * diff_x and abs(diff_F)/F_old > self.relative_diff_F_min:
                    backtracking_check = False
                    self.tau = self.hparams.eta_tau * self.tau
                    x = x_old
                else:
                    backtracking_check = True

            # Logging
            if extract_results:
                out_z = tensor2array(z.cpu())
                out_x = tensor2array(x.cpu())
                current_z_psnr = psnr(clean_img, out_z)
                current_x_psnr = psnr(clean_img, out_x)
                if self.hparams.print_each_step:
                    print('iteration : ', i)
                    print('current z PSNR : ', current_z_psnr)
                    print('current x PSNR : ', current_x_psnr)
                x_list.append(out_x)
                z_list.append(out_z)
                Dx_list.append(tensor2array(Dx.cpu()))
                Ds_list.append(torch.norm(Ds).cpu().item())
                s_list.append(s.cpu().item())
                F_list.append(F_new)
                psnr_tab.append(current_x_psnr)

            i += 1 # next iteration

        # post-processing gradient step
        if extract_results:
            Ds, f = self.denoiser_model.calculate_grad(x, self.sigma_denoiser / 255.)
            Ds = Ds.detach()
            f = f.detach()
            Dx = x - self.denoiser_model.hparams.weight_Ds * Ds.detach()
            s = 0.5 * (torch.norm(x.double() - f.double(), p=2) ** 2)
        else:
            Ds, _ = self.denoiser_model.calculate_grad(x, self.sigma_denoiser / 255.)
            Ds = Ds.detach()
            Dx = x - self.denoiser_model.hparams.weight_Ds * Ds

        z = (1 - self.hparams.lamb * self.tau) * x + self.hparams.lamb * self.tau * Dx

        if self.hparams.degradation_mode == 'inpainting':
            output_img = tensor2array(x.cpu())
        else :
            output_img = tensor2array(z.cpu())

        output_psnr = psnr(clean_img, output_img)
        output_psnrY = psnr(rgb2y(clean_img), rgb2y(output_img))

        if extract_results:
            if self.hparams.print_each_step:
                print('current z PSNR : ', output_psnr)
            z_list.append(tensor2array(z.cpu()))
            Dx_list.append(tensor2array(Dx.cpu()))
            Ds_list.append(torch.norm(Ds).cpu().item())
            s_list.append(s.cpu().item())
            return output_img, output_psnr, output_psnrY, np.array(x_list), np.array(z_list), np.array(Dx_list), np.array(psnr_tab), np.array(Ds_list), np.array(s_list), np.array(F_list)
        else:
            return output_img, output_psnr, output_psnrY

    def initialize_curves(self):

        self.rprox = []
        self.prox = []
        self.conv = []
        self.lip_algo = []
        self.lip_D = []
        self.PSNR = []
        self.s = []
        self.Ds = []
        self.F = []

    def update_curves(self, x_list, z_list, Dx_list, psnr_tab, Ds_list, s_list, F_list):

        prox_list = x_list
        self.F.append(F_list)
        self.s.append(s_list)
        self.Ds.append(Ds_list)
        self.prox.append(np.sqrt(np.array([np.sum(np.abs(prox_list[i + 1] - prox_list[i]) ** 2) for i in range(len(x_list[:-1]))]) / np.array([np.sum(np.abs(z_list[i + 1] - z_list[i]) ** 2) for i in range(len(z_list[:-1]))])))
        rprox_list = 2 * prox_list - z_list
        self.rprox.append(np.sqrt(np.array([np.sum(np.abs(rprox_list[i + 1] - rprox_list[i]) ** 2) for i in range(len(rprox_list[:-1]))]) / np.array([np.sum(np.abs(z_list[i + 1] - z_list[i]) ** 2) for i in range(len(rprox_list[:-1]))])))
        self.conv.append(np.array([np.sum(np.abs(x_list[k + 1] - x_list[k]) ** 2) for k in range(len(x_list) - 1)]) / np.sum(np.abs(x_list[0]) ** 2))
        self.lip_algo.append(np.sqrt(np.array([np.sum(np.abs(x_list[k + 1] - x_list[k]) ** 2) for k in range(1, len(x_list) - 1)]) / np.array([np.sum(np.abs(x_list[k] - x_list[k - 1]) ** 2) for k in range(1, len(x_list[:-1]))])))
        self.lip_D.append(np.sqrt(np.array([np.sum(np.abs(Dx_list[i + 1] - Dx_list[i]) ** 2) for i in range(len(Dx_list) - 1)]) / np.array([np.sum(np.abs(x_list[i + 1] - x_list[i]) ** 2) for i in range(len(x_list) - 1)])))
        self.PSNR.append(psnr_tab)

    def save_curves(self, save_path):

        import matplotlib
        matplotlib.rcParams.update({'font.size': 15})

        plt.figure(1)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.PSNR)):
            plt.plot(self.PSNR[i], '*', label='im_' + str(i))
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_path, 'PSNR.png'))

        plt.figure(2)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.F)):
            plt.plot(self.F[i], '-o', markersize=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'F.png'))

        plt.figure(3)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.conv)):
            plt.plot(self.conv[i], '-o', markersize=10)
            plt.semilogy()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'conv_log.png'), bbox_inches="tight")

        self.conv2 = [[np.min(self.conv[i][:k]) for k in range(1, len(self.conv[i]))] for i in range(len(self.conv))]
        conv_rate = [self.conv2[i][0]*np.array([(1/k) for k in range(1,len(self.conv2[i]))]) for i in range(len(self.conv2))]

        plt.figure(4)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.conv2)):
            plt.plot(self.conv2[i], '-o', markersize=10, label='GS-PnP')
            plt.plot(conv_rate[i], '--', color='red', label=r'$\mathcal{O}(\frac{1}{K})$')
            plt.semilogy()
        plt.legend()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'conv_log2.png'), bbox_inches="tight")

        plt.figure(5)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.lip_algo)):
            plt.plot(self.lip_algo[i], '-o', label='im_' + str(i))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid()
        plt.savefig(os.path.join(save_path, 'lip_algo.png'))

        plt.figure(6)
        for i in range(len(self.lip_D)):
            plt.plot(self.lip_D[i], '-o', label='im_' + str(i))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid()
        plt.savefig(os.path.join(save_path, 'lip_D.png'))


    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--denoiser_name', type=str, default='GS-DRUNet')
        parser.add_argument('--dataset_path', type=str, default='../datasets')
        parser.add_argument('--pretrained_checkpoint', type=str,default='../GS_denoising/ckpts/GSDRUNet.ckpt')
        parser.add_argument('--PnP_algo', type=str, default='HQS')
        parser.add_argument('--dataset_name', type=str, default='CBSD10')
        parser.add_argument('--sigma_denoiser', type=float)
        parser.add_argument('--noise_level_img', type=float, default=2.55)
        parser.add_argument('--maxitr', type=int, default=400)
        parser.add_argument('--lamb', type=float, default=0.1)
        parser.add_argument('--tau', type=float)
        parser.add_argument('--n_images', type=int, default=68)
        parser.add_argument('--weight_Ds', type=float, default=1.)
        parser.add_argument('--eta_tau', type=float, default=0.9)
        parser.add_argument('--gamma', type=float, default=0.1)
        parser.add_argument('--no_use_backtracking', dest='use_backtracking', action='store_false')
        parser.set_defaults(use_backtracking=True)
        parser.add_argument('--relative_diff_F_min', type=float, default=1e-6)
        parser.add_argument('--inpainting_init', dest='inpainting_init', action='store_true')
        parser.set_defaults(inpainting_init=False)
        parser.add_argument('--precision', type=str, default='simple')
        parser.add_argument('--n_init', type=int, default=10)
        parser.add_argument('--patch_size', type=int, default=256)
        parser.add_argument('--extract_curves', dest='extract_curves', action='store_true')
        parser.set_defaults(extract_curves=False)
        parser.add_argument('--no_extract_images', dest='extract_images', action='store_false')
        parser.set_defaults(extract_images=True)
        parser.add_argument('--print_each_step', dest='print_each_step', action='store_true')
        parser.set_defaults(print_each_step=False)
        parser.add_argument('--act_mode_denoiser', type=str, default='E')
        return parser
