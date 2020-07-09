
import os

import torchvision.transforms.functional as FT
from PIL import Image

from util import get_psnr


def img_evaluation(HR_dir, SR_dir, appended_name):

    ''' make file list '''
    '''
    assert any(scope_l in SR_dir for scope_l in ['x2', 'x3', 'x4']), 'SR image directory string must contain scope folder. ex) x2, x3, x4'
    scope = 'x2'
    for scope_l in ['x3', 'x4']:
        scope = scope_l if scope_l in SR_dir else scope
    '''

    HR_img_list = os.listdir(HR_dir)
    
    tmp_name_parse_list = [img_file.split('.') for img_file in HR_img_list]
    for img_file in tmp_name_parse_list:
        img_file.pop(1)
        img_file.append(appended_name)
        img_file.append('.png')
    SR_img_list = [''.join(name_list) for name_list in tmp_name_parse_list]

    ''' get evaluation '''
    value = 0.0
    for idx, HR_img in enumerate(HR_img_list):
        assert os.path.isfile(HR_dir+'/'+HR_img), 'there is no hr file of such name : %s'%HR_img
        assert os.path.isfile(SR_dir+'/'+SR_img_list[idx]), 'thers is no sr image file of such name : %s'%SR_img_list[idx]

        HR_tensor = FT.to_tensor(Image.open(HR_dir+'/'+HR_img))
        SR_tensor = FT.to_tensor(Image.open(SR_dir+'/'+SR_img_list[idx]))

        psnr = get_psnr(HR_tensor, SR_tensor)
        value += psnr
        print(psnr)

    return value / len(HR_img_list)


if __name__ == '__main__':
    print(img_evaluation('./data/test/Set5/HR', './data/test/Set5/LR/x4', 'x4'))
