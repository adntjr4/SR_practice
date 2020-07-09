import math
import torch
import torchvision.transforms.functional as FT
import torchvision.transforms as tr


def tensor_to_pil_overcutting(img):
    # overflow cutting
    c, w, h = img.shape
    t_img = img.clone()
    for cc in range(c):
        for ww in range(w):
            for hh in range(h):
                if img[cc][ww][hh] > 1.0:
                    img[cc][ww][hh] = 1.0
                elif img[cc][ww][hh] < 0.0:
                    img[cc][ww][hh] = 0.0
    return FT.to_pil_image(t_img)
        

def img_transform(img, in_d_type, out_d_type, normalize=None):
    # d_type 0 : (0, 1)
    # d_type 1 : (-1, 1)
    # d_type 2 : normalized

    if normalize is not None:
        reverse_mean = (lambda x1, x2, x3: [-x1, -x2, -x3])(*normalize[0])
        reverse_std = (lambda x1, x2, x3: [1/x1, 1/x2, 1/x3])(*normalize[1])
        reverse_normalize = tr.transforms.Compose([ tr.Normalize(mean = [ 0., 0., 0. ], std = reverse_std),
                                      tr.Normalize(mean = reverse_mean, std = [ 1., 1., 1. ]) ])
        forward_normalize = tr.Normalize(*normalize)

    # to d_type
    if in_d_type == 0:
        tmp_img = img
    elif in_d_type == 1:
        tmp_img = (img + 1.0) / 2.0
    elif in_d_type == 2:
        tmp_img = reverse_normalize(img)
    
    if out_d_type == 0:
        tmp_img = tmp_img.clamp(min=0.0, max=1.0)
    elif out_d_type == 1:
        tmp_img = tmp_img * 2.0 - 1.0
    elif out_d_type == 2:
        tmp_img = forward_normalize(tmp_img)

    return tmp_img


def get_psnr(img1, img2, d_type1=0, d_type2=0, normalize=None):
    # image pixel range : [0, 1]
    assert img1.shape == img2.shape

    img1_t = img_transform(img1, d_type1, 0, normalize)
    img2_t = img_transform(img2, d_type2, 0, normalize)

    err = img1_t - img2_t
    ycbcr_mat = torch.tensor([65.481, 128.553, 24.966])
    
    err_ychcr = torch.matmul(err.permute(1,2,0), ycbcr_mat) / 255.
    mse = err_ychcr.pow(2).mean()

    return -10 * math.log10(mse)

def net_name(net):
    if net.__class__.__name__ == 'DataParallel':
        return net.module.__class__.__name__
    else:
        return net.__class__.__name__

