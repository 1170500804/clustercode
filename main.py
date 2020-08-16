import argparse
import os
import sys
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

#import datasets
#import models

import math

from Datasets import First_Floor_Binary
from attention_utils.radam import RAdam
from lib.NCEAverage import NCEAverage
from lib.LinearAverage import LinearAverage
from lib.NCECriterion import NCECriterion
from lib.utils import AverageMeter
from models.residual_attention_network import ResidualAttentionModel_92_Small #ResidualAttentionModel_92
from test import NN, kNN

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--train-data',type=str,
                    help='path to dataset')
parser.add_argument('--val-data', type=str,
                    help='path to dataset')
# parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
#                     choices=model_names,
#                     help='model architecture: ' +
#                         ' | '.join(model_names) +
#                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--nce-k', default=4096, type=int,
                    metavar='K', help='number of negative samples for NCE')
parser.add_argument('--nce-t', default=0.07, type=float, 
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    help='momentum for non-parametric updates')
parser.add_argument('--iter_size', default=1, type=int,
                    help='caffe style iter size')
parser.add_argument('--finetune',action='store_true')
parser.add_argument('--name', type=str,default='unnamed')
parser.add_argument('--attribute-name',type=str)
parser.add_argument('--image-folder',type=str)
parser.add_argument('--regression',action='store_true')
parser.add_argument('--mask-buildings',action='store_true')
parser.add_argument('--softmask',action='store_true')

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True, finetune=args.finetune, low_dim= args.low_dim)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #
    #     model = models.__dict__[args.arch](low_dim=args.low_dim)

    # Data loading code


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # train_dataset = datasets.CombinedMaskDataset(
    #     other_data_path = '/home/saschaho/Simcenter/found_label_imgs',
    #     csv_root_folder='/home/saschaho/Simcenter/Floor_Elevation_Data/Streetview_Irma/Streetview_Irma/images',
    #     data_csv='/home/saschaho/Simcenter/Building_Information_Prediction/all_bims_train.csv',
    #     transform = transforms.Compose([
    #         transforms.RandomResizedCrop(224, scale=(0.2,1.)),
    #         transforms.RandomGrayscale(p=0.2),
    #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]),attribute = 'first_floor_elevation_ft', mask_images=True)

    # val_dataset = datasets.CombinedMaskDataset(
    #         csv_root_folder='/home/saschaho/Simcenter/Floor_Elevation_Data/Streetview_Irma/Streetview_Irma/images',
    #         data_csv='/home/saschaho/Simcenter/Building_Information_Prediction/all_bims_val.csv',
    #     transform=transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]),
        #attribute = 'first_floor_elevation_ft', mask_images=True)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.3, 1.)),
            transforms.RandomGrayscale(p=0.5),
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize])

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])

    train_dataset = First_Floor_Binary(args.attribute_name, args.train_data, args.image_folder, transform=train_transform, regression=args.regression, mask_buildings=args.mask_buildings, softmask=args.softmask)
    val_dataset = First_Floor_Binary(args.attribute_name, args.val_data, args.image_folder, transform=val_transform, regression=args.regression, mask_buildings=args.mask_buildings, softmask=args.softmask)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model = ResidualAttentionModel_92_Small(args.low_dim, dropout=False)
    model = torch.nn.DataParallel(model).cuda()

    print ('Train dataset instances: {}'.format(len(train_loader.dataset)))
    print ('Val dataset instances: {}'.format(len(val_loader.dataset)))
    # define lemniscate and loss function (criterion)
    ndata = train_dataset.__len__()
    if args.nce_k > 0:
        lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m).cuda()
        criterion = NCECriterion(ndata).cuda()
    else:
        lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m).cuda()
        criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)

    #optimizer = RAdam(model.parameters())

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']

            keyname = [keyname for keyname in model.state_dict().keys() if 'fc.weight' in keyname][0]
            lat_vec_len_model = model.state_dict()[keyname].shape[0]
            lat_vec_len_checkpoint = checkpoint['state_dict'][keyname].shape[0]

            low_dim_differ = False
            if lat_vec_len_model != lat_vec_len_checkpoint:
                low_dim_differ = True
                print('Warning: Latent vector sizes do not match. Assuming finetuning')
                print('Lemniscate will be trained from scratch with new optimizer.')
                del checkpoint['state_dict'][keyname]
                del checkpoint['state_dict'][keyname.replace('weight','bias')]

            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
            if len(missing_keys) or len(unexpected_keys):
                print('Warning: Missing or unexpected keys found.')
                print('Missing: {}'.format(missing_keys))
                print('Unexpected: {}'.format(unexpected_keys))

            if not low_dim_differ:
                # The memory bank will be trained from scratch if
                # the low dim is different. Maybe later repopulated
                lemniscate = checkpoint['lemniscate']
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        kNN(0, model, lemniscate, train_loader, val_loader, 200, args.nce_t)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, lemniscate, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = NN(epoch, model, lemniscate, train_loader, val_loader)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'lemniscate': lemniscate,
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.name)
    # evaluate KNN after last epoch
    kNN(0, model, lemniscate, train_loader, val_loader, 200, args.nce_t)


def train(train_loader, model, lemniscate, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()

    for i, (input, _, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        index = index.cuda(non_blocking=True)

        # compute output
        feature = model(input)
        output = lemniscate(feature, index)
        loss = criterion(output, index) / args.iter_size

        loss.backward()

        # measure accuracy and record loss
        losses.update(loss.item() * args.iter_size, input.size(0))

        if (i+1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))


def save_checkpoint(state, is_best, prefix, filename='checkpoint.pth.tar'):
    filename = prefix + '-' + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    lr = args.lr
    if epoch < 120:
        lr = args.lr
    elif epoch >= 120 and epoch < 160:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    #lr = args.lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
