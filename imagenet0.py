"""
Ref:https://github.com/bearpaw/pytorch-classification
Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M. X., et al. (2019). Searching for MobileNetV3.  IEEE/CVF International Conference on Computer Vision (ICCV),  Seoul, SOUTH KOREA, https://doi.org/10.1109/iccv.2019.00140.
"""


from __future__ import print_function

import argparse
import os

import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models
from models.mobilenetv3 import MobileNetV3_Large
import numpy as np
import matplotlib.pyplot as plt


from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

CUDA_VISIBLE_DEVICES = 0,1
# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets

parser.add_argument('-d', '--data', default='/home/qmz/Array optimization method/Best combination/6_point0', type=str)

parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=52, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=24, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')

parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint/temp0', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

parser.add_argument('--resume', default='/home/qmz/Array optimization method/checkpoint/temp0/model_best.pth.tar', type=str, metavar='/home/qmz/Array optimization method/model_best.ph.tar',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='mobilenet_v3_large',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
#Device options
parser.add_argument('--gpu-id', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args(args=[])
state = {k: v for k, v in args._get_kwargs()}


# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# print(os.environ['CUDA_VISIBLE_DEVICES'])
use_cuda = torch.cuda.is_available()
# print(torch.cuda.device_count())
# torch.cuda.set_device(0)
CUDA_LAUNCH_BLOCKING=1

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)



def main():
    all_test = []
    all_train = []
    abc = 0
    loss = 100
    best_train = 0
    best_val = 0
    epo = 0
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # args.data = args.data.to(device)
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=4,pretrained=True)
    elif args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    baseWidth=args.base_width,
                    cardinality=args.cardinality,
                )
    elif args.arch.startswith('mobileone'):
        print("=> creating model '{}'-0".format(args.arch))
        model = mobileone_s3(4)
    elif args.arch.startswith('mobilenetv3'):
        print("=> creating model '{}'-0".format(args.arch))
        model = MobileNetV3_Large(4)
    else:
        # print("=> creating model '{}'-0".format(args.arch))
        model = models.__dict__[args.arch](num_classes=4)

    # print(model)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()



    cudnn.benchmark = True
    # print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'ImageNet-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint0..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        # best_acc = checkpoint['best_acc']
        # train_best_acc = checkpoint['train_best_acc']
        # start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log1.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log1.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = tes(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val

    for epoch in range(start_epoch, args.epochs):
        #adjust_learning_rate(optimizer, epoch)


        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = tes(val_loader, model, criterion, epoch, use_cuda)
        print(f'\tTrain Loss0: {train_loss:.6f} | Train Acc0: {train_acc:.6f}%')
        print(f'\t Val. Loss0: {test_loss:.6f} |  Val. Acc0: {test_acc:.6f}%')
        if loss > train_loss*3.77/4.77 + test_loss*1/4.77:
            loss = train_loss*3.77/4.77 + test_loss*1/4.77
            best_train = train_acc
            best_val = test_acc
            epo = epoch+1

        # if test_acc.cpu().numpy()>70 and state['lr'] == 0.1:
        #     state['lr'] = 0.05
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = state['lr']
        # elif test_acc.cpu().numpy()>80 and state['lr'] == 0.05:
        #     state['lr'] = 0.01
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = state['lr']

        if epoch >= 0 and epoch < 10:
            state['lr'] = 0.01
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['lr']
        elif epoch >= 10 and epoch < 20:
            state['lr'] = 0.001
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['lr']
        elif epoch >= 20:
            state['lr'] = 0.0005
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['lr']
        
                
        

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model

        # save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'acc': test_acc,
        #         'optimizer': optimizer.state_dict(),
        #     }, all_train,all_test, train_acc,test_acc,checkpoint=args.checkpoint)
        all_test.append(test_acc)
        all_train.append(train_acc)

    logger.close()
    # logger.plot()
    # savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best train_acc: ',max(all_train))
    print('Best test_acc: ',max(all_test))
    print('Best train_acc_1: ', best_train)
    print('Best test_acc_1: ', best_val)
    print('loss/2: ', loss/2)
    print('Best epo', epo)

    return loss/2, best_train, best_val,abc

def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top3 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=False)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        # compute output
        outputs = model(inputs)
        # outputs = outputs.logits     #only used by GoogleNet
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec2, prec3 = accuracy(outputs.data, targets.data, topk=(1, 2, 3))
        losses.update(loss.item(), inputs.size(0))
        # print(prec1)
        top1.update(prec1.item(), inputs.size(0))
        top2.update(prec2.item(), inputs.size(0))
        top3.update(prec3.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top2: {top2: .4f} | top3: {top3: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top2=top2.avg,
                    top3=top3.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def tes(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    linshi = list()
    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec2, prec3 = accuracy(outputs.data, targets.data, topk=(1, 2, 3))
        losses.update(loss.item(), inputs.size(0))
        # print(prec1)
        top1.update(prec1.item(), inputs.size(0))
        top2.update(prec2.item(), inputs.size(0))
        top3.update(prec3.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top2: {top2: .4f} | top3: {top3: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top2=top2.avg,
                    top3=top3.avg,
                    )
        bar.next()

    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, all_train,all_test,train_acc,test_acc, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    # if state['epoch'] > 100 and ((train_acc >= max(all_train) and train_acc >=90) or (test_acc >= max(all_test) and test_acc >= 95)):
    #     shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best_%s.pth.tar'%state['epoch']))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image

def plot_images(images, labels, classes, normalize=True):
    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))
    fig = plt.figure(figsize=(10, 10))

    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        image = images[i]
        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax.set_title(classes[labels[i]])
        ax.axis('off')

if __name__ == '__main__':
    main()
