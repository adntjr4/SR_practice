import os, random

import torchvision.transforms.functional as FT
import torchvision.transforms as tr
import torch.utils.data as data
import torch
from PIL import Image

class SRtransform():
    def __init__(self, crop_size, upscale_factor, train, lr_upsize, in_d_type=0, out_d_type=0, normalize=None):
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.train = train
        self.lr_upsize = lr_upsize
        self.in_d_type = in_d_type # 0:(0, 1), 1:(-1, 1), 2:normalize
        self.out_d_type = out_d_type
        self.normalize = tr.Normalize(*normalize) if normalize is not None else None

        self.train_transform = tr.transforms.Compose([
                                                tr.transforms.RandomHorizontalFlip(),
                                                tr.transforms.RandomApply([tr.transforms.RandomRotation((90,90))])
                                                ])

    def __call__(self, img):
        w, h = img.size

        # random crop
        if self.train:
            left = random.randint(1, w-self.crop_size)
            top = random.randint(1, h-self.crop_size)
            hr_img = img.crop((left, top, left+self.crop_size, top+self.crop_size))
        else:
            cut_w = w % self.upscale_factor
            cut_h = h % self.upscale_factor
            hr_img = img.crop((cut_w//2, cut_h//2, cut_w//2+w-cut_w, cut_h//2+h-cut_h))

        # apply random horizontal flip, 90 rotation
        if self.train:
            hr_img = self.train_transform(hr_img)

        # make low resolution image
        lr_img = hr_img.resize((hr_img.width//self.upscale_factor, hr_img.height//self.upscale_factor), Image.BICUBIC)

        if self.lr_upsize:
            lr_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)

        # to tensor
        lr_img = FT.to_tensor(lr_img)
        hr_img = FT.to_tensor(hr_img)

        # data transform
        if self.in_d_type == 1:
            lr_img = (lr_img * 2.) - 1.
        elif self.in_d_type == 2:
            lr_img = self.normalize(lr_img)

        if self.out_d_type == 1:
            hr_img = (hr_img * 2.) - 1.
        elif self.out_d_type == 2:
            hr_img = self.normalize(hr_img)    

        return lr_img, hr_img

class SRDataSet(data.Dataset):
    def __init__(self, dir, crop_size=48, upscale_factor=2, train=True, lr_upsize=True, in_d_type=0, out_d_type=0, normalize=None): 
        super(SRDataSet, self).__init__()
        
        self.dir = dir
        self.img_list = os.listdir(dir)
        self.upscale_factor = upscale_factor
        self.train = train
        self.lr_upsize = lr_upsize

        self.transform = SRtransform(crop_size, upscale_factor, train, lr_upsize, in_d_type=in_d_type, out_d_type=out_d_type, normalize=normalize)

    def __getitem__(self, index):
        img = Image.open(self.dir + '/' + self.img_list[index])
        
        return self.transform(img.convert('RGB'))

    def __len__(self):
        return len(self.img_list)

def tt(img):
    img[0] = 1.0

if __name__ == '__main__':
    test_tensor = torch.zeros(3,10,10)
    tt(test_tensor)
    c, w, h = test_tensor.shape
    for cc in range(c):
        for ww in range(w):
            for hh in range(h):
                if test_tensor[cc][ww][hh] > 1.0:
                    test_tensor[cc][ww][hh] = 1.0
                elif test_tensor[cc][ww][hh] < 0.0:
                    test_tensor[cc][ww][hh] = 0.0

    img = FT.to_pil_image(test_tensor)
    img.save('test.jpg')

