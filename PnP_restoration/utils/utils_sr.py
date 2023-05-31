# -*- coding: utf-8 -*-
import numpy as np
from scipy import fftpack
import torch
from scipy import ndimage
from scipy.interpolate import interp2d
from scipy import signal
import scipy.stats as ss
import scipy.io as io
import scipy
import torch.nn.functional as F

'''
Modified from  Kai Zhang's code (github: https://github.com/cszn)
'''

'''
# =================
# pytorch
# =================
'''


def splits(a, sf):
    '''split a into sfxsf distinct blocks

    Args:
        a: NxCxWxHx2
        sf: split factor

    Returns:
        b: NxCx(W/sf)x(H/sf)x2x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=5)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=5)
    return b


def c2c(x):
    return torch.from_numpy(np.stack([np.float32(x.real), np.float32(x.imag)], axis=-1))


def r2c(x):
    # convert real to complex
    return torch.stack([x, torch.zeros_like(x)], -1)


def cdiv(x, y):
    # complex division
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]
    cd2 = c ** 2 + d ** 2
    return torch.stack([(a * c + b * d) / cd2, (b * c - a * d) / cd2], -1)


def crdiv(x, y):
    # complex/real division
    a, b = x[..., 0], x[..., 1]
    return torch.stack([a / y, b / y], -1)


def csum(x, y):
    # complex + real
    return torch.stack([x[..., 0] + y, x[..., 1]], -1)


def cabs(x):
    # modulus of a complex number
    return torch.pow(x[..., 0] ** 2 + x[..., 1] ** 2, 0.5)


def cabs2(x):
    return x[..., 0] ** 2 + x[..., 1] ** 2


def cmul(t1, t2):
    '''complex multiplication

    Args:
        t1: NxCxHxWx2, complex tensor
        t2: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)


def cconj(t, inplace=False):
    '''complex's conjugation

    Args:
        t: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c


def fft(t):
    # Real-to-complex Discrete Fourier Transform
    return torch.view_as_real(torch.fft.fft2(t))


def ifft(t):
    # Complex-to-real Inverse Discrete Fourier Transform
    return torch.real(torch.fft.ifft2(torch.view_as_complex(t)))


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis + 2)
    #otf = torch.rfft(otf, 2, onesided=False)
    otf = fft(otf)
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    otf[..., 1][torch.abs(otf[..., 1]) < n_ops * 2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def upsample(x, sf=3):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * sf, x.shape[3] * sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]

def G(x, k, sf=3):
    '''
    x: image, NxcxHxW
    k: kernel, hxw
    sf: scale factor
    center: the first one or the moddle one
    Matlab function:
    tmp = imfilter(x,h,'circular');
    y = downsample2(tmp,K);
    '''
    x = downsample(imfilter(x, k), sf=sf)
    return x


def Gt(x, k, sf=3):
    '''
    x: image, NxcxHxW
    k: kernel, hxw
    sf: scale factor
    center: the first one or the moddle one
    Matlab function:
    tmp = upsample2(x,K);
    y = imfilter(tmp,h,'circular');
    '''
    x = imfilter(upsample(x, sf=sf),k, transposed=True)
    return x


def pre_calculate_prox(x, k, sf):
    '''
    Args:
        x: NxCxHxW, LR input
        k: hxw
        sf: integer

    Returns:
        FB, FBC, F2B, FBFy
        will be reused during iterations
    '''
    w, h = x.shape[-2:]
    FB = p2o(k.repeat(1, 1, 1, 1), (w * sf, h * sf))
    FBC = cconj(FB, inplace=False)
    F2B = r2c(cabs2(FB))
    STy = upsample(x, sf=sf)
    FBFy = cmul(FBC, fft(STy))
    return FB, FBC, F2B, FBFy



def pre_calculate_grad(x, k, sf):
    '''
    Args:
        x: NxCxHxW, LR input
        k: Nx1xhxw
        sf: integer

    Returns:

        will be reused during iterations
    '''
    STx = upsample(x,sf=sf)
    return



def prox_solution_L2(x, FB, FBC, F2B, FBFy, stepsize, sf):
    alpha = torch.tensor([1/stepsize]).repeat(1, 1, 1, 1).to(x.device)
    FR = FBFy + fft(alpha * x)
    x1 = cmul(FB, FR)
    FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
    invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
    invWBR = cdiv(FBR, csum(invW, alpha))
    FCBinvWBR = cmul(FBC, invWBR.repeat(1, 1, sf, sf, 1))
    FX = (FR - FCBinvWBR) / alpha.unsqueeze(-1)
    Xest = ifft(FX)
    return Xest
    

def Wiener_filter(x, k, stepsize, sf):
    alpha = torch.tensor([1/stepsize]).repeat(1, 1, 1, 1).to(x.device)
    w, h = x.shape[-2:]
    FB = p2o(k.repeat(1, 1, 1, 1), (w * sf, h * sf))
    FBC = cconj(FB, inplace=False)
    F2B = r2c(cabs2(FB))
    STy = upsample(x, sf=sf)
    FBFy = cmul(FBC, fft(STy))
    FR = FBFy
    x1 = cmul(FB, FR)
    FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
    invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
    invWBR = cdiv(FBR, csum(invW, alpha))
    FCBinvWBR = cmul(FBC, invWBR.repeat(1, 1, sf, sf, 1))
    FX = (FR - FCBinvWBR)
    Xest = ifft(FX)
    return Xest


def grad_solution_L2_fft(x, FB, FBC, FBFy, sf):
    FBFx = cmul(FB, fft(x))
    kx = ifft(FBFx)
    AFx = downsample(ifft(FBFx),sf=sf)
    STAFx = upsample(AFx, sf=sf)
    ATAFx = cmul(FBC, fft(STAFx))
    FX = ATAFx - FBFy
    Xest = ifft(FX)
    return Xest


def grad_solution_L2(x, y, k, sf):
    '''
    Gradient of the L2 data fidelity term.

    x: image, NxcxHxW
    y: image, NxcxHxW
    k: kernel, hxw
    sf: scale factor
    '''
    I = G(x, k, sf=sf)-y
    return Gt(I, k, sf=sf)

def grad_solution_KL(x, y, k, sf, alpha):
    '''
    Gradient of the KL data fidelity term.

    x: image, NxcxHxW
    y: image, NxcxHxW
    k: kernel, hxw
    sf: scale factor
    '''
    I = alpha*torch.ones_like(x) - y/G(x, k, sf=sf)
    return Gt(I, k, sf=sf)




'''
# =================
PyTorch
# =================
'''


def real2complex(x):
    return torch.stack([x, torch.zeros_like(x)], -1)


def modcrop(img, sf):
    '''
    img: tensor image, NxCxWxH or CxWxH or WxH
    sf: scale factor
    '''
    w, h = img.shape[-2:]
    im = img.clone()
    return im[..., :w - w % sf, :h - h % sf]


def circular_pad(x, pad):
    '''
    # x[N, 1, W, H] -> x[N, 1, W + 2 pad, H + 2 pad] (pariodic padding)
    '''
    x = torch.cat([x, x[:, :, 0:pad, :]], dim=2)
    x = torch.cat([x, x[:, :, :, 0:pad]], dim=3)
    x = torch.cat([x[:, :, -2 * pad:-pad, :], x], dim=2)
    x = torch.cat([x[:, :, :, -2 * pad:-pad], x], dim=3)
    return x


def pad_circular(input, padding):
    # type: (Tensor, List[int]) -> Tensor
    """
    Arguments
    :param input: tensor of shape :math:`(N, C_{\text{in}}, H, [W, D]))`
    :param padding: (tuple): m-elem tuple where m is the degree of convolution
    Returns
    :return: tensor of shape :math:`(N, C_{\text{in}}, [D + 2 * padding[0],
                                     H + 2 * padding[1]], W + 2 * padding[2]))`
    """
    offset = 3
    for dimension in range(input.dim() - offset + 1):
        input = dim_pad_circular(input, padding[dimension], dimension + offset)
    return input


def dim_pad_circular(input, padding, dimension):
    # type: (Tensor, int, int) -> Tensor
    input = torch.cat([input, input[[slice(None)] * (dimension - 1) +
                                    [slice(0, padding)]]], dim=dimension - 1)
    input = torch.cat([input[[slice(None)] * (dimension - 1) +
                             [slice(-2 * padding, -padding)]], input], dim=dimension - 1)
    return input


def unpad_circular(input,padding):
    ph,pw = padding
    out = input[:, :, ph:-ph, pw:-pw]
    # sides
    out[:, :, :ph, :] += input[:, :, -ph:, pw:-pw]
    out[:, :, -ph:, :] += input[:, :, :ph, pw:-pw]
    out[:, :, :, :pw] += input[:, :, ph:-ph, -pw:]
    out[:, :, :, -pw:] += input[:, :, ph:-ph, :pw]
    # corners
    out[:, :, :ph, :pw] += input[:, :, -ph:, -pw:]
    out[:, :, -ph:, -pw:] += input[:, :, :ph, :pw]
    out[:, :, :ph, -pw:] += input[:, :, -ph:, :pw]
    out[:, :, -ph:, :pw] += input[:, :, :ph, -pw:]
    return out


def imfilter(x, k, transposed=False):
    '''
    Equivalent (verified) to scipy ndimage.convolve with mode='wrap'.
    x: image, NxcxHxW
    k: kernel, hxw
    '''
    k = k.repeat(3, 1, 1, 1)
    k = k.flip(-1).flip(-2) # flip kernel for convolution and not correlation !!!
    ph = (k.shape[-2] - 1)//2
    pw = (k.shape[-1] - 1)//2
    if not transposed:
        x = pad_circular(x, padding=(ph, pw))
        x = F.conv2d(x, k, groups=x.shape[1])
    else :
        x = F.conv_transpose2d(x, k, groups=x.shape[1])
        x = unpad_circular(x, padding=(ph, pw))
    return x


"""
# --------------------------------------------
# degradation models
# --------------------------------------------
"""


def bicubic_degradation(x, sf=3):
    '''
    Args:
        x: HxWxC image, [0, 1]
        sf: down-scale factor

    Return:
        bicubicly downsampled LR image
    '''
    x = util.imresize_np(x, scale=1 / sf)
    return x


def numpy_degradation(x, k, sf=3):
    ''' blur + downsampling

    Args:
        x: HxWxC image, [0, 1]/[0, 255]
        k: hxw, double, positive
        sf: down-scale factor

    Return:
        downsampled LR image
    '''
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    # x = filters.correlate(x, np.expand_dims(np.flip(k), axis=2))
    st = 0
    return x[st::sf, st::sf, ...]


def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf - 1) * 0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x
