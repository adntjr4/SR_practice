import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader

from EDSR import EDSR, geometric_self_ensemble
from dataloader import SRDataSet
from util.progress_msg import ProgressMsg
from util.util import *

''' Training Setting '''
train_dir = './data/train/DIV2K_train'
test_dir_list = ['./data/test/DIV2K_valid'] #['./data/test/Set5', './data/test/Set14', './data/test/B100', './data/test/DIV2K_valid']
crop_size = 96
upscale_factor = 2
batch_size = 16
total_epoch = 2400
initial_learning_rate = 0.5e-4

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
        print('continue training %s model (%d epoch~)...'%(net_name(net), load['epoch']))
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

    criterion = torch.nn.L1Loss().to(device)

    train_dataset = SRDataSet(train_dir+'/HR', crop_size, upscale_factor, train=True, lr_upsize=False, in_d_type=in_data_type, out_d_type=out_data_type, normalize=normalize)
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
                    './model/checkpoint/%s_checkpoint.pth'%net_name(net))

        #step_lr_scheduler.step()
    progress_msg.print_finish_msg()

def test(LR_save=False):

    load = torch.load('./model/checkpoint/EDSR_checkpoint.pth')

    net = load['model']

    print('model : %s'%net_name(net))
    print('epoch : %d'%load['epoch'])
    print('== Test ==')
    net.eval()

    for test_dir in test_dir_list:
        test_dataset = SRDataSet(test_dir+'/HR', crop_size, upscale_factor, train=False, lr_upsize=False, in_d_type=in_data_type, out_d_type=out_data_type, normalize=normalize)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

        psnr_sum = 0.0
        for i, data in enumerate(test_loader):
            input = data[0].to(device)

            with torch.no_grad():
                prediction = net(input)

            # save LR image
            if LR_save:
                LR_name = test_dir + '/LR/x%d/'%upscale_factor + test_dataset.get_name(i)[:-4] + 'x%d.png'%upscale_factor
                FT.to_pil_image(img_transform(data[0].squeeze(), in_data_type, 0, normalize)).save(LR_name)
            
            # save SR image
            SR_name = test_dir + '/SR/%s/x%d/'%(net_name(net), upscale_factor) + test_dataset.get_name(i)[:-4] + '_SRx%d.png'%upscale_factor
            FT.to_pil_image(img_transform(prediction.cpu().squeeze(), out_data_type, 0, normalize)).save(SR_name)

            psnr_sum += get_psnr(prediction.cpu().squeeze(), data[1].squeeze(), out_data_type, out_data_type, normalize)
            del input, prediction
        
        print('%s : %f'%(test_dir.split('/').pop(), psnr_sum/len(test_loader)))
        del test_dataset, test_loader

def bicubic_test():
    for test_dir in test_dir_list:
        test_dataset = SRDataSet(test_dir+'/HR', crop_size, upscale_factor, train=False, lr_upsize=True, in_d_type=in_data_type, out_d_type=out_data_type, normalize=normalize)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

        for i, data in enumerate(test_loader):
            # save bicubic image
            bicubic_name = test_dir + '/LR/x%d/'%upscale_factor + test_dataset.get_name(i)[:-4] + '_SRx%d.png'%upscale_factor
            FT.to_pil_image(img_transform(data[0].squeeze(), in_data_type, 0, normalize)).save(bicubic_name)

def train_image():
    train_dataset = SRDataSet(train_dir+'/HR', crop_size, upscale_factor, train=True, lr_upsize=False, in_d_type=in_data_type, out_d_type=out_data_type, normalize=normalize)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    for i, data in enumerate(train_loader):
        FT.to_pil_image(img_transform(data[0].squeeze(), in_data_type, 0, normalize)).show()
        FT.to_pil_image(img_transform(data[1].squeeze(), out_data_type, 0, normalize)).show()
        break

if __name__ == '__main__':
    main()
