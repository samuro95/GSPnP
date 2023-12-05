import torch

def normalize_min_max(A):
    AA = A.clone()
    AA = AA.view(A.size(0), -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    AA = AA.view(A.size())
    return AA

def psnr(img1,img2) :
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10( 1. / torch.sqrt(mse))