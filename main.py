
import glob, os, time
from math import log10, ceil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader

from EDSR import EDSR, geometric_self_ensemble
from dataloader import SRDataSet
from util.progress_msg import ProgressMsg
from util.util import get_psnr, img_transform

''' Training Setting '''
train_dir = './data/train/DIV2K_train_HR'
test_dir = './data/test/Set5'
crop_size = 96
upscale_factor = 4
batch_size = 16
total_epoch = 2400
initial_learning_rate = 1e-5

momentum = 0.9
weight_decay = 1e-4

''' Device Setting '''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

in_data_type = 2 # sub mean
out_data_type = 2 # sub mean
normalize = [[0.4488, 0.4371, 0.4040], [1.0, 1.0, 1.0]]
checkpoint = './model/checkpoint/EDSR_checkpoint.pth'

def main():
    ''' Model Setting '''

    # load model
    if checkpoint is not None:
        load = torch.load(checkpoint)
        start_epoch = load['epoch']
        net = load['model']
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=initial_learning_rate, betas=(0.9, 0.999), eps=1e-8)
        print('continue training %s model (%d epoch~)...'%(net.__class__.__name__, load['epoch']))
    # make model
    else:
        net = EDSR(res_blocks=32, feature_ch=256, upscale_factor=upscale_factor)
        start_epoch = 0
        #optimizer = optim.SGD(net.parameters(), lr=initial_learning_rate, momentum=momentum, weight_decay=weight_decay)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=initial_learning_rate, betas=(0.9, 0.999), eps=1e-8)
    net.to(device)
    #net = torch.nn.DataParallel(net)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of parameters : %d' % num_params)

    criterion = nn.L1Loss().to(device)

    train_dataset = SRDataSet(train_dir, crop_size, upscale_factor, train=True, lr_upsize=False, in_d_type=in_data_type, out_d_type=out_data_type, normalize=normalize)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    train(net, train_loader, start_epoch, criterion, optimizer)

def train(net, train_loader, start_epoch, criterion, optimizer):
    progress_msg = ProgressMsg((total_epoch-start_epoch, len(train_loader)))

    #step_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(total_epoch*0.25), int(total_epoch*0.50), int(total_epoch*0.75)], gamma=0.1)

    for epoch in range(start_epoch, total_epoch):
        net.train()
        epoch_loss = 0

        #train_psnr_sum = 0

        for i, data in enumerate(train_loader):
            input, target = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            prediction = net(input)
            loss = criterion(prediction, target)
            epoch_loss += loss.item()
            #train_psnr_sum += get_psnr(prediction.cpu(), data[1], in_d_type=in_data_type, out_d_type=out_data_type, normalize=normalize)
            loss.backward()
            optimizer.step()

            progress_msg.print_prog_msg((epoch-start_epoch, i))

        print("[epoch : %d] loss : %.4f\t\t\t\t\t\t\t\t" % (epoch+1, epoch_loss/len(train_loader)))

        torch.save({'epoch': epoch+1,
                    'model': net,
                    'optimizer': optimizer,
                    'loss': loss},
                    './model/checkpoint/%s_checkpoint.pth'%net.__class__.__name__)

        #step_lr_scheduler.step()
    progress_msg.print_finish_msg()

def test():
    
    test_dataset = SRDataSet(test_dir, crop_size, upscale_factor, train=False, lr_upsize=False, in_d_type=in_data_type, out_d_type=out_data_type, normalize=normalize)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    load = torch.load('./model/checkpoint/EDSR_checkpoint.pth')

    net = load['model']
    print('model : %s'%net.__class__.__name__)
    print('epoch : %d'%load['epoch'])

    net.eval()
    psnr_sum = 0
    for i, data in enumerate(test_loader):
        input, target = data[0].to(device), data[1].to(device)

        prediction = net(input)
        psnr = get_psnr(prediction.cpu().squeeze(), data[1].squeeze(), d_type1=out_data_type, d_type2=out_data_type, normalize=normalize)
        psnr_sum += psnr
        print(psnr)

        FT.to_pil_image(img_transform(data[0].squeeze(), in_data_type, 0, normalize)).save('./data/sr/input%d.png'%i)
        FT.to_pil_image(img_transform(prediction.cpu().squeeze(), out_data_type, 0, normalize)).save('./data/sr/sr%d.png'%i)

    print("AVG psnr : %.2f dB \t\t\t\t\t\t\t\t" % (psnr_sum / len(test_loader)))

def bicubic_psnr():
    test_dataset = SRDataSet(test_dir, crop_size, upscale_factor, train=False, lr_upsize=True, in_d_type=in_data_type, out_d_type=out_data_type, normalize=normalize)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    psnr_sum = 0
    for i, data in enumerate(test_loader):
        psnr = get_psnr(data[0].squeeze(), data[1].squeeze(), d_type1=in_data_type, d_type2=out_data_type, normalize=normalize)
        psnr_sum += psnr
        print(psnr)

    print("AVG psnr : %.2f dB \t\t\t\t\t\t\t\t" % (psnr_sum / len(test_loader)))

def train_image():
    train_dataset = SRDataSet(train_dir, crop_size, upscale_factor, train=True, lr_upsize=False, in_d_type=in_data_type, out_d_type=out_data_type, normalize=normalize)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    for i, data in enumerate(train_loader):
        FT.to_pil_image(img_transform(data[0].squeeze(), in_data_type, 0, normalize)).show()
        FT.to_pil_image(img_transform(data[1].squeeze(), out_data_type, 0, normalize)).show()
        break

if __name__ == '__main__':
    main()
