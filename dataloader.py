import os, random

import torchvision.transforms.functional as FT
import torch.utils.data as data
from PIL import Image

def tensor2pil(img):
    return FT.to_pil_image(img)

class SRtransform():
    def __init__(self, crop_size, upscale_factor, train, lr_upsize, d_type=0):
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.train = train
        self.lr_upsize = lr_upsize
        self.d_type = d_type # 0:(0, 1), 1:(-1, 1)

    def __call__(self, img):
        w, h = img.size

        if self.train:
            left = random.randint(1, w-self.crop_size)
            top = random.randint(1, h-self.crop_size)
            hr_img = img.crop((left, top, left+self.crop_size, top+self.crop_size))
        else:
            cut_w = w % self.upscale_factor
            cut_h = h % self.upscale_factor
            hr_img = img.crop((cut_w//2, cut_h//2, cut_w//2+w-cut_w, cut_h//2+h-cut_h))
        lr_img = hr_img.resize((hr_img.width//self.upscale_factor, hr_img.height//self.upscale_factor), Image.BICUBIC)

        if self.lr_upsize:
            lr_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)

        # To tensor
        lr_img = FT.to_tensor(lr_img)
        hr_img = FT.to_tensor(hr_img)

        # to range [-1, 1]
        if self.d_type == 1:
            lr_img = (lr_img * 2.) - 1.
            hr_img = (hr_img * 2.) - 1.

        return lr_img, hr_img

class SRDataSet(data.Dataset):
    def __init__(self, dir, crop_size=48, upscale_factor=2, train=True, lr_upsize=True, d_type=0): 
        super(SRDataSet, self).__init__()
        
        self.dir = dir
        self.img_list = os.listdir(dir)
        self.upscale_factor = upscale_factor
        self.train = train
        self.lr_upsize = lr_upsize

        self.transform = SRtransform(crop_size, upscale_factor, train, lr_upsize, d_type=d_type)

    def __getitem__(self, index):
        img = Image.open(self.dir + '/' + self.img_list[index])
        
        return self.transform(img)

    def __len__(self):
        return len(self.img_list)

if __name__ == '__main__':
    
    dir = './data/test/Set5'

    test_dataset = SRDataSet(dir, 96, 4, True, True)
    lr, hr = test_dataset[0]

    tensor2pil(lr).show()
    tensor2pil(hr).show()
