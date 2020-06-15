import math
import torch

def img_transform(img, in_d_type, out_d_type):
    # d_type 0 : (0, 1)
    # d_type 1 : (-1, 1)

    # to d_type
    if in_d_type == 0:
        pass
    elif in_d_type == 1:
        tmp_img = (img + 1.) / 2.
    
    if out_d_type == 0:
        pass
    elif out_d_type == 1:
        tmp_img = tmp_img * 2. - 1.

    return tmp_img


def get_psnr(img1, img2, d_type=0):
    # image pixel range : [0, 1]
    assert img1.shape == img2.shape

    if d_type == 0:
        err = (img1-img2)#[..., 6:-6, 6:-6]
    elif d_type == 1:
        err = (img1-img2) / 2
    ycbcr_mat = torch.tensor([65.481, 128.553, 24.966])

    
    err_ychcr = torch.matmul(err.permute(0,2,3,1), ycbcr_mat) / 255.
    mse = err_ychcr.pow(2).mean()

    return -10 * math.log10(mse)
