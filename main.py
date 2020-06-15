
import glob, os, time
from math import log10, ceil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataloader import SRDataSet, tensor2pil
from progress_msg import ProgressMsg
from util import get_psnr, img_transform

''' Training Setting '''
train_dir = './data/train/DIV2K_train_HR'
test_dir = './data/test/Set5'
crop_size = 96
upscale_factor = 4
batch_size = 16
total_epoch = 600
initial_learning_rate = 1e-4
momentum = 0.9
weight_decay = 1e-4

''' Device Setting '''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

data_type = 1 # range (-1, 1)

def main():
    ''' Model Setting '''

    # load model
    if True:
        load = torch.load('SRResNet_checkpoint')
        start_epoch = load['epoch']
        net = load['model']
        optimizer = load['optimizer']
        print('continue training (%d epoch~)...'%load['epoch'])
    # make model
    else:
        #net = VDSR(20)
        start_epoch = 0
        net = SRResNet(scaling_factor=upscale_factor)
        #optimizer = optim.SGD(net.parameters(), lr=initial_learning_rate, momentum=momentum, weight_decay=weight_decay)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=initial_learning_rate, betas=(0.9, 0.999), eps=1e-8)
    net.to(device)
    net = torch.nn.DataParallel(net)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of parameters : %d' % num_params)

    criterion = nn.MSELoss().to(device)

    train_dataset = SRDataSet(train_dir, crop_size, upscale_factor, train=True, lr_upsize=False, d_type=data_type)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    

    train(net, train_loader, start_epoch, criterion, optimizer)

def train(net, train_loader, start_epoch, criterion, optimizer):
    progress_msg = ProgressMsg((total_epoch-start_epoch, len(train_loader)))

    #step_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(total_epoch*0.25), int(total_epoch*0.50), int(total_epoch*0.75)], gamma=0.1)

    for epoch in range(start_epoch, total_epoch):
        net.train()
        epoch_loss = 0

        train_psnr_sum = 0

        for i, data in enumerate(train_loader):
            input, target = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            prediction = net(input)
            loss = criterion(prediction, target)
            epoch_loss += loss.item()
            train_psnr_sum += get_psnr(prediction.cpu(), data[1], d_type=data_type)
            loss.backward()
            optimizer.step()

            progress_msg.print_prog_msg((epoch-start_epoch, i))

        print("[epoch : %d] loss : %.4f, train_psnr : %.2f dB \t\t\t\t\t\t\t\t" % (epoch+1, epoch_loss/len(train_loader), train_psnr_sum/len(train_loader)))

        torch.save({'epoch': epoch+1,
                    'model': net,
                    'optimizer': optimizer,
                    'loss': loss},
                    './model/checkpoint/%s_checkpoint.pth'%net.__class__.__name__)

        #step_lr_scheduler.step()
    progress_msg.print_finish_msg()

def test():
    
    test_dataset = SRDataSet(test_dir, crop_size, upscale_factor, train=False, lr_upsize=False, d_type=data_type)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    load = torch.load('SRResNet_checkpoint')

    net = load['model']
    print(load['epoch'])
    criterion = nn.MSELoss().to(device)

    #net.eval()
    psnr_sum = 0
    for i, data in enumerate(test_loader):
        input, target = data[0].to(device), data[1].to(device)

        prediction = net(input)
        mse = criterion(prediction, target)
        psnr = get_psnr(prediction.cpu(), data[1], d_type=data_type)
        psnr_sum += psnr
        print(psnr)

        tensor2pil(img_transform(data[0], 1, 0).squeeze()).save('./data/sr/input%d.png'%i)
        tensor2pil(img_transform(prediction, 1, 0).squeeze().cpu()).save('./data/sr/sr%d.png'%i)

    print("AVG psnr : %.2f dB \t\t\t\t\t\t\t\t" % (psnr_sum / len(test_loader)))

def bicubic_psnr():
    test_dataset = SRDataSet(test_dir, crop_size, upscale_factor, train=False, lr_upsize=True, d_type=data_type)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    criterion = nn.MSELoss()

    psnr_sum = 0
    for i, data in enumerate(test_loader):
        psnr = get_psnr(data[0], data[1], d_type=data_type)
        psnr_sum += psnr
        print(psnr)

    print("AVG psnr : %.2f dB \t\t\t\t\t\t\t\t" % (psnr_sum / len(test_loader)))

if __name__ == '__main__':
    test()
