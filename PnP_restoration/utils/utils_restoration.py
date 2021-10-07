import numpy as np
import random
from scipy.fftpack import dct, idct
import torch
import cv2

'''
Copyright (c) 2020 Kai Zhang (cskaizhang@gmail.com)
'''




def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


def randomCrop(img1,img2,width,height):
    assert img1.shape[0] >= height
    assert img1.shape[1] >= width
    x = random.randint(0, img1.shape[1] - width)
    y = random.randint(0, img1.shape[0] - height)
    img1 = img1[y:y+height, x:x+width]
    img2 = img2[y:y + height, x:x + width]
    return img1,img2

def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:int(H-H_r), :int(W-W_r), :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

def crop_center(img,cropx,cropy):
    y,x = img.shape[0],img.shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx,:]

def array2tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

def tensor2array(img):
    img = img.cpu()
    img = img.squeeze().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())

def imsave(img_path,img):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

def rgb2y(im):
    xform = np.array([.299, .587, .114])
    y = im.dot(xform.T)
    return y


def psnr(img1,img2) :
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean((img1 - img2)**2)
    return 20 * np.log10(1. / np.sqrt(mse))

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def dct2(img) :
    if img.shape[-1] == 1 :
        return dct(dct(img.T, norm='ortho').T, norm='ortho')
    else :
        out = np.zeros(img.shape)
        for i in range(img.shape[-1]):
            out[:,:,i] = dct(dct(img[:,:,i].T, norm='ortho').T, norm='ortho')
        return out

def idct2(freq) :
    if freq.shape[-1] == 1 :
        return idct(idct(freq.T, norm='ortho').T, norm='ortho')
    else :
        out = np.zeros(freq.shape)
        for i in range(freq.shape[-1]):
            out[:,:,i] = idct(idct(freq[:,:,i].T, norm='ortho').T, norm='ortho')
        return out

def extract_low_high_DCT_f_images(img,rho=2):
    w, h = img.shape[0], img.shape[1]
    freq = dct2(img)
    w_out = int(w / rho)
    h_out = int(h / rho)
    low_f = np.copy(freq)
    high_f = np.copy(freq)
    high_f[:w_out,:h_out] = 0
    low_f[w_out:, h_out:] = 0
    import matplotlib.pyplot as plt
    plt.imshow(np.abs(high_f))
    plt.show()
    plt.imshow(np.abs(low_f))
    plt.show()
    return idct2(low_f), idct2(high_f)

def extract_low_high_f_images(img,rho=2):
    w, h = img.shape[0], img.shape[1]
    freq = np.fft.fftshift(np.fft.fft2(img, axes=(0, 1)))
    mask = np.abs(np.ones_like(freq))
    mask[int(w/(2*rho)):int((2*rho-1)*w/(2*rho)),int(h/(2*rho)):int((2*rho-1)*h/(2*rho))] = 0
    high_f = np.fft.fftshift(freq*mask)
    low_f = np.fft.fftshift(freq*(1-mask))
    return np.real(np.fft.ifft2(low_f, axes=(0, 1))), np.real(np.fft.ifft2(high_f, axes=(0, 1)))

def decompose_DCT_pyramid(img,levels,rho,use_scaling=False, show_dyadic_DCT_pyramid=False):
    if show_dyadic_DCT_pyramid :
        show_dyadic_DCT_pyramid(img, levels, use_scaling)
    w,h = img.shape[0],img.shape[1]
    freq = dct2(img)
    pyramid = []
    for l in range(levels) :
        w_out = int(w/(rho**l))
        h_out = int(h/(rho**l))
        if use_scaling:
            scaling = np.sqrt((w_out * h_out) / (w * h))
        else:
            scaling = 1.
        out_freq = freq[:w_out,:h_out]*scaling
        pyramid.append(idct2(out_freq))
    return pyramid

def show_dyadic_DCT_pyramid(img,levels,use_scaling=False):
    w,h = img.shape[0],img.shape[1]
    freq = dct2(img)
    for l in range(levels) :
        w_out = int(w/(2**l))
        h_out = int(h/(2**l))
        if use_scaling:
            scaling = np.sqrt((w_out * h_out) / (w * h))
        else:
            scaling = 1.
        freq[:w_out,:h_out] = freq[:w_out,:h_out]*scaling
    import matplotlib.pyplot as plt
    im = 20*np.log(np.abs(freq)+1)
    plt.imshow(im)
    plt.show()

def merge_coarse(image,coarse,frec,use_scaling = False):
    freq = dct2(image)
    tmp = dct2(coarse)
    w, h = tmp.shape[0], tmp.shape[1]
    w_out, h_out = freq.shape[0], freq.shape[1]
    wrec, hrec = int(w * frec), int(h * frec)
    if use_scaling :
        scaling = np.sqrt((w_out * h_out) / (w * h))
    else :
        scaling = 1.
    freq[:wrec, :hrec] = tmp[:wrec, :hrec] * scaling
    out = idct2(freq)
    return out

def recompose_DCT_pyramid(pyramid,frec):
    img = pyramid[0]
    for l in range(1,len(pyramid)) :
        img = merge_coarse(img,pyramid[l],frec)
    return img


def get_DPIR_rho_sigma(sigma=2.55/255, iter_num=15, modelSigma1=49.0, modelSigma2=2.55, w=1.0, lamb = 0.23):
    '''
    One can change the sigma to implicitly change the trade-off parameter
    between fidelity term and prior term
    '''
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)
    modelSigmaS_lin = np.linspace(modelSigma1, modelSigma2, iter_num).astype(np.float32)
    sigmas = (modelSigmaS*w+modelSigmaS_lin*(1-w))/255.
    rhos = list(map(lambda x: lamb*(sigma**2)/(x**2), sigmas))
    return rhos, sigmas