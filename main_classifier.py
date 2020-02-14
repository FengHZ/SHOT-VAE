import random

random.seed(1)
import numpy as np

np.random.seed(1)

import argparse
from classifier_model.wideresnet import get_wide_resnet
from lib.utils.avgmeter import AverageMeter
from lib.dataloader import cifar10_dataset, get_cifar10_ssl_sampler, cifar100_dataset, get_cifar100_ssl_sampler, \
    svhn_dataset, get_ssl_sampler
import os
from os import path
import time
import shutil
import ast
from itertools import cycle
from collections import defaultdict
import re
from lib.utils.utils import get_score_label_array_from_dict
from sklearn.metrics import roc_auc_score
import math


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v


parser = argparse.ArgumentParser(
    description='Pytorch Training Semi-Supervised Classifiers for Cifar10,Cifar100,SVHN Dataset')
# Dataset Parameters
parser.add_argument('-bp', '--base_path', default=".")
parser.add_argument('--dataset', default="Cifar10", type=str, help="name of dataset used")
parser.add_argument('-is', "--image-size", default=[32, 32], type=arg_as_list,
                    metavar='Image Size List', help='the size of h * w for image')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
# SSL Train Strategy Parameters
parser.add_argument('-t', '--train-time', default=1, type=int,
                    metavar='N', help='the x-th time of training')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--dp', '--data-parallel', action='store_false', help='Use Data Parallel')
parser.add_argument('--print-freq', '-p', default=3, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume-arg', action='store_false', help='if we not resume the argument')
parser.add_argument('--annotated-ratio', default=0.1, type=float, help='The ratio for semi-supervised annotation')
# Deep Learning Model Parameters
parser.add_argument('--net-name', default="wideresnet-28-2", type=str, help="the name for network to use")
parser.add_argument('-dr', '--drop-rate', default=0, type=float, help='drop rate for the network')
# Optimizer Parameters
parser.add_argument('-on', '--optimizer-name', default="SGD", type=str, metavar="Optimizer Name",
                    help="The name for the optimizer we used")
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay-ratio', default=0.2, type=float)
parser.add_argument('-b1', '--beta1', default=0.9, type=float, metavar='Beta1 In ADAM and SGD',
                    help='beta1 for adam as well as momentum for SGD')
parser.add_argument('-ad', "--adjust-lr", default=[300, 350,400], type=arg_as_list,
                    help="The milestone list for adjust learning rate")
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float)
# GPU Parameters
parser.add_argument("--gpu", default="0,1", type=str, metavar='GPU plans to use', help='The GPU id plans to use')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from torch.utils.data import DataLoader
from torchvision import utils
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

torch.manual_seed(1)
torch.cuda.manual_seed(1)


def main(args=args):
    if args.dataset == "Cifar10":
        dataset_base_path = path.join(args.base_path, "dataset", "cifar")
        train_dataset = cifar10_dataset(dataset_base_path)
        test_dataset = cifar10_dataset(dataset_base_path, train_flag=False)
        sampler_valid, sampler_train_l, sampler_train_u = get_cifar10_ssl_sampler(
            torch.tensor(train_dataset.targets, dtype=torch.int32), 500, round(4000 * args.annotated_ratio), 10)
        test_dloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        valid_dloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                                   sampler=sampler_valid)
        train_dloader_l = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                     pin_memory=True,
                                     sampler=sampler_train_l)
        train_dloader_u = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                     pin_memory=True,
                                     sampler=sampler_train_u)
        input_channels = 3
        num_classes = 10
        small_input = True
        criterion = torch.nn.CrossEntropyLoss()
    elif args.dataset == "Cifar100":
        dataset_base_path = path.join(args.base_path, "dataset", "cifar")
        train_dataset = cifar100_dataset(dataset_base_path)
        test_dataset = cifar100_dataset(dataset_base_path, train_flag=False)
        sampler_valid, sampler_train_l, sampler_train_u = get_cifar100_ssl_sampler(
            torch.tensor(train_dataset.targets, dtype=torch.int32), 50, round(400 * args.annotated_ratio), 100)
        test_dloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        valid_dloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                                   sampler=sampler_valid)
        train_dloader_l = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                     pin_memory=True,
                                     sampler=sampler_train_l)
        train_dloader_u = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                     pin_memory=True,
                                     sampler=sampler_train_u)
        input_channels = 3
        num_classes = 100
        small_input = True
        criterion = torch.nn.CrossEntropyLoss()
    elif args.dataset == "SVHN":
        dataset_base_path = path.join(args.base_path, "dataset", "svhn")
        train_dataset = svhn_dataset(dataset_base_path)
        test_dataset = svhn_dataset(dataset_base_path, train_flag=False)
        sampler_valid, sampler_train_l, sampler_train_u = get_ssl_sampler(
            torch.tensor(train_dataset.labels, dtype=torch.int32), 100, 100, 10)
        test_dloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        valid_dloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                                   sampler=sampler_valid)
        train_dloader_l = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                     pin_memory=True,
                                     sampler=sampler_train_l)
        train_dloader_u = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                     pin_memory=True,
                                     sampler=sampler_train_u)
        input_channels = 3
        num_classes = 10
        small_input = True
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("Dataset {} Not Implemented".format(args.dataset))
    model = get_wide_resnet(args.net_name, args.drop_rate, input_channels=input_channels, small_input=small_input,
                            data_parallel=args.dp, num_classes=num_classes)
    print("Begin the {} Time's Training Semi-Supervised Classifiers, Dataset {}".format(args.train_time, args.dataset))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.beta1, weight_decay=args.wd)
    scheduler = MultiStepLR(optimizer, milestones=args.adjust_lr, gamma=args.lr_decay_ratio)
    writer_log_dir = "{}/{}-SSL-Classifier/runs/train_time:{}".format(args.base_path, args.dataset,
                                                                      args.train_time)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args = checkpoint['args']
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise FileNotFoundError("Checkpoint Resume File {} Not Found".format(args.resume))
    else:
        if os.path.exists(writer_log_dir):
            flag = input("vae_train_time:{} will be removed, input yes to continue:".format(
                args.train_time))
            if flag == "yes":
                shutil.rmtree(writer_log_dir, ignore_errors=True)
    writer = SummaryWriter(log_dir=writer_log_dir)
    for epoch in range(args.start_epoch, args.epochs):
        if epoch == 0:
            # do warm up
            modify_lr_rate(opt=optimizer, lr=args.lr * 0.2)
        train(train_dloader_l, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch, writer=writer)
        test(valid_dloader, test_dloader, model=model, criterion=criterion, epoch=epoch, writer=writer,
             num_classes=num_classes)
        if epoch == 0:
            modify_lr_rate(opt=optimizer, lr=args.lr)
        scheduler.step(epoch)


def train(train_dloader, model, criterion, optimizer, epoch, writer):
    # some records
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    optimizer.zero_grad()
    for i, (image, label) in enumerate(train_dloader):
        data_time.update(time.time() - end)
        image = image.float().cuda()
        label = label.long().cuda()
        cls_result = model(image)
        loss = criterion(cls_result, label)
        loss.backward()
        losses.update(float(loss.item()), image.size(0))
        optimizer.step()
        optimizer.zero_grad()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            train_text = 'Epoch: [{0}][{1}/{2}]\t' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                         'Cls Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})'.format(
                epoch, i + 1, len(train_dloader), batch_time=batch_time, data_time=data_time,
                cls_loss=losses)
            print(train_text)
    writer.add_scalar(tag="Train/cls_loss", scalar_value=losses.avg, global_step=epoch + 1)
    return losses.avg


def test(valid_dloader, test_dloader, model, criterion, epoch, writer, num_classes):
    model.eval()
    # calculate index for valid dataset
    losses = AverageMeter()
    all_score = []
    all_label = []
    for i, (image, label) in enumerate(valid_dloader):
        image = image.float().cuda()
        label = label.long().cuda()
        with torch.no_grad():
            cls_result = model(image)
        label_onehot = torch.zeros(label.size(0), num_classes).cuda().scatter_(1, label.view(-1, 1), 1)
        loss = criterion(cls_result, label)
        losses.update(float(loss.item()), image.size(0))
        # here we add the all score and all label into one list
        all_score.append(torch.softmax(cls_result, dim=1))
        # turn label into one-hot code
        all_label.append(label_onehot)
    writer.add_scalar(tag="Valid/cls_loss", scalar_value=losses.avg, global_step=epoch + 1)
    all_score = torch.cat(all_score, dim=0).detach()
    all_label = torch.cat(all_label, dim=0).detach()
    _, y_true = torch.topk(all_label, k=1, dim=1)
    _, y_pred = torch.topk(all_score, k=5, dim=1)
    top_1_accuracy = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
    top_5_accuracy = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
    writer.add_scalar(tag="Valid/top 1 accuracy", scalar_value=top_1_accuracy, global_step=epoch + 1)
    if args.dataset == "Cifar100":
        writer.add_scalar(tag="Valid/top 5 accuracy", scalar_value=top_5_accuracy, global_step=epoch + 1)
    # calculate index for test dataset
    losses = AverageMeter()
    all_score = []
    all_label = []
    # don't use roc
    # roc_list = []
    for i, (image, label) in enumerate(test_dloader):
        image = image.float().cuda()
        label = label.long().cuda()
        with torch.no_grad():
            cls_result = model(image)
        label_onehot = torch.zeros(label.size(0), num_classes).cuda().scatter_(1, label.view(-1, 1), 1)
        loss = criterion(cls_result, label)
        losses.update(float(loss.item()), image.size(0))
        # here we add the all score and all label into one list
        all_score.append(torch.softmax(cls_result, dim=1))
        # turn label into one-hot code
        all_label.append(label_onehot)
    writer.add_scalar(tag="Test/cls_loss", scalar_value=losses.avg, global_step=epoch + 1)
    all_score = torch.cat(all_score, dim=0).detach()
    all_label = torch.cat(all_label, dim=0).detach()
    _, y_true = torch.topk(all_label, k=1, dim=1)
    _, y_pred = torch.topk(all_score, k=5, dim=1)
    # don't use roc auc
    # all_score = all_score.cpu().numpy()
    # all_label = all_label.cpu().numpy()
    # for i in range(num_classes):
    #     roc_list.append(roc_auc_score(all_label[:, i], all_score[:, i]))
    # ap_list.append(average_precision_score(all_label[:, i], all_score[:, i]))
    # calculate accuracy by hand
    top_1_accuracy = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
    top_5_accuracy = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
    writer.add_scalar(tag="Test/top 1 accuracy", scalar_value=top_1_accuracy, global_step=epoch + 1)
    if args.dataset == "Cifar100":
        writer.add_scalar(tag="Test/top 5 accuracy", scalar_value=top_5_accuracy, global_step=epoch + 1)
    # writer.add_scalar(tag="Test/mean RoC", scalar_value=mean(roc_list), global_step=epoch + 1)
    return top_1_accuracy


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    :param state: a dict including:{
                'epoch': epoch + 1,
                'args': args,
                "state_dict": model.state_dict(),
                'optimizer': optimizer.state_dict(),
        }
    :param filename: the filename for store
    :return:
    """
    filefolder = "{}/{}-SSL-Classifier/parameter/train_time:{}".format(args.base_path, args.dataset, args.train_time)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    torch.save(state, path.join(filefolder, filename))


def modify_lr_rate(opt, lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    main()
