import random

random.seed(1)
import numpy as np

np.random.seed(1)
import argparse
from shot_vae_model.vae import VariationalAutoEncoder
from lib.criterion import VAECriterion, ClsCriterion
from lib.utils.avgmeter import AverageMeter
from lib.utils.mixup import mixup_vae_data, label_smoothing
from lib.dataloader import cifar10_dataset, get_cifar10_ssl_sampler, cifar100_dataset, get_cifar100_ssl_sampler, \
    svhn_dataset, get_ssl_sampler
import os
from os import path
import time
import shutil
import ast
from itertools import cycle
import math


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v


parser = argparse.ArgumentParser(description='Pytorch Training Semi-Supervised VAE for Cifar10,Cifar100,SVHN Dataset')
# Dataset Parameters
parser.add_argument('-bp', '--base_path', default=".")
parser.add_argument('--dataset', default="Cifar10", type=str, help="name of dataset used")
parser.add_argument('-is', "--image-size", default=[32, 32], type=arg_as_list,
                    metavar='Image Size List', help='the size of h * w for image')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=768, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
# SSL VAE Train PreProcess Parameter
parser.add_argument('-t', '--train-time', default=1, type=int,
                    metavar='N', help='the x-th time of training')
parser.add_argument('--epochs', default=600, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--dp', '--data-parallel', action='store_false', help='Use Data Parallel')
parser.add_argument('--print-freq', '-p', default=3, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--reconstruct-freq', '-rf', default=20, type=int,
                    metavar='N', help='reconstruct frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume-arg', action='store_false', help='if we not resume the argument')
parser.add_argument('--annotated-ratio', default=0.1, type=float, help='The ratio for semi-supervised annotation')
# Deep VAE Model Parameters
parser.add_argument('--net-name', default="wideresnet-28-2", type=str, help="the name for network to use")
parser.add_argument('--temperature', default=0.67, type=float,
                    help='centeralization parameter')
parser.add_argument('-dr', '--drop-rate', default=0, type=float, help='drop rate for the network')
parser.add_argument("--br", "--bce-reconstruction", action='store_true', help='Do BCE Reconstruction')
parser.add_argument("-s", "--x-sigma", default=1, type=float,
                    help="The standard variance for reconstructed images, work as regularization")
# VAE parameters, notice we do not manually set the mutual information
parser.add_argument('--ldc', "--latent-dim-continuous", default=128, type=int,
                    metavar='Latent Dim For Continuous Variable',
                    help='feature dimension in latent space for continuous variable')
parser.add_argument('--cmi', "--continuous-mutual-info", default=0, type=float,
                    help='The mutual information bounding between x and the continuous variable z')
parser.add_argument('--dmi', "--discrete-mutual-info", default=0, type=float,
                    help='The mutual information bounding between x and the discrete variable z')
# VAE Loss Function Parameters
parser.add_argument("-ei", "--evaluate-inference", action='store_true',
                    help='Calculate the inference accuracy for unlabeled dataset')
parser.add_argument('--kbmc', '--kl-beta-max-continuous', default=1e-3, type=float, metavar='KL Beta',
                    help='the epoch to linear adjust kl beta')
parser.add_argument('--kbmd', '--kl-beta-max-discrete', default=1e-3, type=float, metavar='KL Beta',
                    help='the epoch to linear adjust kl beta')
parser.add_argument('--akb', '--adjust-kl-beta-epoch', default=200, type=int, metavar='KL Beta',
                    help='the max epoch to adjust kl beta')
parser.add_argument('--ewm', '--elbo-weight-max', default=1e-3, type=float, metavar='weight for elbo loss part')
parser.add_argument('--aew', '--adjust-elbo-weight', default=400, type=int,
                    metavar="the epoch to adjust elbo weight to max")
parser.add_argument('--wrd', default=1, type=float,
                    help="the max weight for the optimal transport estimation of discrete variable c")
parser.add_argument('--wmf', '--weight-modify-factor', default=0.4, type=float,
                    help="weight  will get wrz at amf * epochs")
parser.add_argument('--pwm', '--posterior-weight-max', default=1, type=float,
                    help="the max value for posterior weight")
parser.add_argument('--apw', '--adjust-posterior-weight', default=200, type=float,
                    help="adjust posterior weight")
# Optimizer Parameters
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-b1', '--beta1', default=0.9, type=float, metavar='Beta1 In ADAM and SGD',
                    help='beta1 for adam as well as momentum for SGD')
parser.add_argument('-ad', "--adjust-lr", default=[400, 500, 550], type=arg_as_list,
                    help="The milestone list for adjust learning rate")
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float)
# Optimizer Transport Estimation Parameters
parser.add_argument('--epsilon', default=0.1, type=float,
                    help="the label smoothing epsilon for labeled data")
parser.add_argument('--om', action='store_true', help="the optimal match for unlabeled data mixup")
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
        small_input = True
        discrete_latent_dim = 10
        args.dmi = 2.3
        elbo_criterion = VAECriterion(discrete_dim=discrete_latent_dim, x_sigma=args.x_sigma,
                                      bce_reconstruction=args.br).cuda()
        cls_criterion = ClsCriterion()
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
        small_input = True
        discrete_latent_dim = 100
        args.akb = 150
        args.apw = 400
        args.dmi = 4.6
        elbo_criterion = VAECriterion(discrete_dim=discrete_latent_dim, x_sigma=args.x_sigma,
                                      bce_reconstruction=args.br).cuda()
        cls_criterion = ClsCriterion()
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
        small_input = True
        discrete_latent_dim = 10
        args.dmi = 2.3
        elbo_criterion = VAECriterion(discrete_dim=discrete_latent_dim, x_sigma=args.x_sigma,
                                      bce_reconstruction=args.br).cuda()
        cls_criterion = ClsCriterion()
    else:
        raise NotImplementedError("Dataset {} not implemented".format(args.dataset))
    model = VariationalAutoEncoder(encoder_name=args.net_name, num_input_channels=input_channels,
                                   drop_rate=args.drop_rate, img_size=tuple(args.image_size), data_parallel=args.dp,
                                   continuous_latent_dim=args.ldc, disc_latent_dim=discrete_latent_dim,
                                   sample_temperature=args.temperature, small_input=small_input)
    model = model.cuda()

    print("Begin the {} Time's Training Semi-Supervised VAE, Dataset {}".format(args.train_time, args.dataset))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.beta1, weight_decay=args.wd)
    scheduler = MultiStepLR(optimizer, milestones=args.adjust_lr)
    writer_log_dir = "{}/{}-SHOT-VAE/runs/train_time:{}".format(args.base_path, args.dataset,
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
    best_valid_acc = 10
    for epoch in range(args.start_epoch, args.epochs):
        if epoch == 0:
            # do warm up
            modify_lr_rate(opt=optimizer, lr=args.lr * 0.2)
        train(train_dloader_u, train_dloader_l, model=model, elbo_criterion=elbo_criterion, cls_criterion=cls_criterion,
              optimizer=optimizer, epoch=epoch,
              writer=writer, discrete_latent_dim=discrete_latent_dim)
        elbo_valid_loss, *_ = valid(valid_dloader, model=model, elbo_criterion=elbo_criterion,
                                    epoch=epoch, writer=writer, discrete_latent_dim=discrete_latent_dim)
        if test_dloader is not None:
            test(test_dloader, model=model, elbo_criterion=elbo_criterion, epoch=epoch,
                 writer=writer, discrete_latent_dim=discrete_latent_dim)
        """
        Here we define the best point as the minimum average epoch loss
        """
        save_checkpoint({
            'epoch': epoch + 1,
            'args': args,
            "state_dict": model.state_dict(),
            'optimizer': optimizer.state_dict(),
        })
        if elbo_valid_loss < best_valid_acc:
            best_valid_acc = elbo_valid_loss
            if epoch >= args.adjust_lr[-1]:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'args': args,
                    "state_dict": model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, best_predict=True)
        scheduler.step(epoch)
        if epoch == 0:
            modify_lr_rate(opt=optimizer, lr=args.lr)
        if args.dataset == "Cifar10":
            if args.annotated_ratio >= 0.05:
                if epoch == args.adjust_lr[0]:
                    args.ewm = args.ewm * 5


def train(train_dloader_u, train_dloader_l, model, elbo_criterion, cls_criterion, optimizer, epoch, writer,
          discrete_latent_dim):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    kl_inferences = AverageMeter()
    model.train()
    end = time.time()
    optimizer.zero_grad()
    # mutual information
    cmi = alpha_schedule(epoch, args.akb, args.cmi)
    dmi = alpha_schedule(epoch, args.akb, args.dmi)
    # elbo part weight
    ew = alpha_schedule(epoch, args.aew, args.ewm)
    # mixup parameters
    kl_beta_c = alpha_schedule(epoch, args.akb, args.kbmc)
    kl_beta_d = alpha_schedule(epoch, args.akb, args.kbmd)
    pwm = alpha_schedule(epoch, args.apw, args.pwm)
    # unsupervised cls weight
    ucw = alpha_schedule(epoch, round(args.wmf * args.epochs), args.wrd)
    for i, ((image_l, label_l), (image_u, label_u)) in enumerate(zip(cycle(train_dloader_l), train_dloader_u)):
        batch_size_l = image_l.size(0)
        batch_size_u = image_u.size(0)
        data_time.update(time.time() - end)
        # for the labeled part, do classification and mixup
        image_l = image_l.float().cuda()
        label_l = label_l.long().cuda()
        label_onehot_l = torch.zeros(batch_size_l, discrete_latent_dim).cuda().scatter_(1, label_l.view(-1, 1), 1)
        reconstruction_l, norm_mean_l, norm_log_sigma_l, disc_log_alpha_l = model(image_l, disc_label=label_l)
        reconstruct_loss_l, continuous_prior_kl_loss_l, disc_prior_kl_loss_l = elbo_criterion(image_l, reconstruction_l,
                                                                                              norm_mean_l,
                                                                                              norm_log_sigma_l,
                                                                                              disc_log_alpha_l)
        prior_kl_loss_l = kl_beta_c * torch.abs(continuous_prior_kl_loss_l - cmi) + kl_beta_d * torch.abs(
            disc_prior_kl_loss_l - dmi)
        elbo_loss_l = reconstruct_loss_l + prior_kl_loss_l
        # do optimal transport estimation
        with torch.no_grad():
            smoothed_image_l, smoothed_z_mean_l, smoothed_z_sigma_l, smoothed_disc_alpha_l, smoothed_label_l, smoothed_lambda_l = \
                label_smoothing(
                    image_l,
                    norm_mean_l,
                    norm_log_sigma_l,
                    disc_log_alpha_l,
                    epsilon=args.epsilon,
                    disc_label=label_l)
            smoothed_label_onehot_l = torch.zeros(batch_size_l, discrete_latent_dim).cuda().scatter_(1,
                                                                                                     smoothed_label_l.view(
                                                                                                         -1,
                                                                                                         1),
                                                                                                     1)
        smoothed_reconstruction_l, smoothed_norm_mean_l, smoothed_norm_log_sigma_l, smoothed_disc_log_alpha_l, *_ = model(
            smoothed_image_l, True,
            label_l,
            smoothed_label_l,
            smoothed_lambda_l)
        disc_posterior_kl_loss_l = smoothed_lambda_l * cls_criterion(smoothed_disc_log_alpha_l, label_onehot_l) + (
                1 - smoothed_lambda_l) * cls_criterion(
            smoothed_disc_log_alpha_l, smoothed_label_onehot_l)
        continuous_posterior_kl_loss_l = (F.mse_loss(smoothed_norm_mean_l, smoothed_z_mean_l, reduction="sum") + \
                                          F.mse_loss(torch.exp(smoothed_norm_log_sigma_l), smoothed_z_sigma_l,
                                                     reduction="sum")) / batch_size_l
        elbo_loss_l = elbo_loss_l + kl_beta_c * pwm * continuous_posterior_kl_loss_l
        loss_supervised = ew * elbo_loss_l + disc_posterior_kl_loss_l
        loss_supervised.backward()

        # for the unlabeled part, do classification and mixup
        image_u = image_u.float().cuda()
        label_u = label_u.long().cuda()
        reconstruction_u, norm_mean_u, norm_log_sigma_u, disc_log_alpha_u = model(image_u)
        # calculate the KL(q_y_x|p_y_x)
        with torch.no_grad():
            label_smooth_u = torch.zeros(batch_size_u, discrete_latent_dim).cuda().scatter_(1, label_u.view(-1, 1),
                                                                                            1 - 0.001 - 0.001 / (
                                                                                                    discrete_latent_dim - 1))
            label_smooth_u = label_smooth_u + torch.ones(label_smooth_u.size()).cuda() * 0.001 / (
                    discrete_latent_dim - 1)
            disc_alpha_u = torch.exp(disc_log_alpha_u)
            inference_kl = disc_alpha_u * disc_log_alpha_u - disc_alpha_u * torch.log(label_smooth_u)
        kl_inferences.update(float(torch.sum(inference_kl) / batch_size_u), batch_size_u)
        reconstruct_loss_u, continuous_prior_kl_loss_u, disc_prior_kl_loss_u = elbo_criterion(image_u, reconstruction_u,
                                                                                              norm_mean_u,
                                                                                              norm_log_sigma_u,
                                                                                              disc_log_alpha_u)
        prior_kl_loss_u = kl_beta_c * torch.abs(continuous_prior_kl_loss_u - cmi) + kl_beta_d * torch.abs(
            disc_prior_kl_loss_u - dmi)
        elbo_loss_u = reconstruct_loss_u + prior_kl_loss_u
        # do mixup part
        with torch.no_grad():
            mixed_image_u, mixed_z_mean_u, mixed_z_sigma_u, mixed_disc_alpha_u, lam_u = \
                mixup_vae_data(
                    image_u,
                    norm_mean_u,
                    norm_log_sigma_u,
                    disc_log_alpha_u,
                    optimal_match=args.om)
        mixed_reconstruction_u, mixed_norm_mean_u, mixed_norm_log_sigma_u, mixed_disc_log_alpha_u, *_ = model(
            mixed_image_u)
        disc_posterior_kl_loss_u = cls_criterion(mixed_disc_log_alpha_u, mixed_disc_alpha_u)
        continuous_posterior_kl_loss_u = (F.mse_loss(mixed_norm_mean_u, mixed_z_mean_u, reduction="sum") + \
                                          F.mse_loss(torch.exp(mixed_norm_log_sigma_u), mixed_z_sigma_u,
                                                     reduction="sum")) / batch_size_u
        elbo_loss_u = elbo_loss_u + kl_beta_c * pwm * continuous_posterior_kl_loss_u
        loss_unsupervised = ew * elbo_loss_u + ucw * disc_posterior_kl_loss_u
        loss_unsupervised.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            train_text = 'Epoch: [{0}][{1}/{2}]\t' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                epoch, i + 1, len(train_dloader_u), batch_time=batch_time, data_time=data_time)
            print(train_text)
    # record unlabeled part loss
    writer.add_scalar(tag="Train/KL_Inference", scalar_value=kl_inferences.avg, global_step=epoch + 1)
    # after several epoch training, we add the image and reconstructed image into the image board, we just use 16 images
    if epoch % args.reconstruct_freq == 0:
        with torch.no_grad():
            image = utils.make_grid(image_u[:4, ...], nrow=2)
            reconstruct_image = utils.make_grid(torch.sigmoid(reconstruction_u[:4, ...]), nrow=2)
        writer.add_image(tag="Train/Raw_Image", img_tensor=image, global_step=epoch + 1)
        writer.add_image(tag="Train/Reconstruct_Image", img_tensor=reconstruct_image, global_step=epoch + 1)


def save_checkpoint(state, filename='checkpoint.pth.tar', best_predict=False):
    """
    :param state: a dict including:{
                'epoch': epoch + 1,
                'args': args,
                "state_dict": shot_vae_model.state_dict(),
                'optimizer': optimizer.state_dict(),
        }
    :param filename: the filename for store
    :param best_predict: the best predict flag
    :return:
    """
    filefolder = '{}/{}-SHOT-VAE/parameter/train_time_{}'.format(args.base_path, args.dataset,
                                                                 state["args"].train_time)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    if best_predict:
        filename = 'best.pth.tar'
        torch.save(state, path.join(filefolder, filename))
    else:
        torch.save(state, path.join(filefolder, filename))


def valid(valid_dloader, model, elbo_criterion, epoch, writer, discrete_latent_dim):
    continuous_kl_losses = AverageMeter()
    discrete_kl_losses = AverageMeter()
    mse_losses = AverageMeter()
    elbo_losses = AverageMeter()
    model.eval()
    all_score = []
    all_label = []

    for i, (image, label) in enumerate(valid_dloader):
        image = image.float().cuda()
        label = label.long().cuda()
        label_onehot = torch.zeros(label.size(0), discrete_latent_dim).cuda().scatter_(1, label.view(-1, 1), 1)
        batch_size = image.size(0)
        with torch.no_grad():
            reconstruction, norm_mean, norm_log_sigma, disc_log_alpha, *_ = model(image)
        reconstruct_loss, continuous_kl_loss, discrete_kl_loss = elbo_criterion(image, reconstruction, norm_mean,
                                                                                norm_log_sigma, disc_log_alpha)
        mse_loss = F.mse_loss(torch.sigmoid(reconstruction.detach()), image.detach(),
                              reduction="sum") / (
                           2 * image.size(0) * (args.x_sigma ** 2))
        mse_losses.update(float(mse_loss), image.size(0))
        all_score.append(torch.exp(disc_log_alpha))
        all_label.append(label_onehot)
        continuous_kl_losses.update(float(continuous_kl_loss.item()), batch_size)
        discrete_kl_losses.update(float(discrete_kl_loss.item()), batch_size)
        elbo_losses.update(float(mse_loss + 0.01 * (continuous_kl_loss + discrete_kl_loss)), image.size(0))

    writer.add_scalar(tag="Valid/KL(q(z|X)||p(z))", scalar_value=continuous_kl_losses.avg, global_step=epoch + 1)
    writer.add_scalar(tag="Valid/KL(q(y|X)||p(y))", scalar_value=discrete_kl_losses.avg, global_step=epoch + 1)
    writer.add_scalar(tag="Valid/log(p(X|z,y))", scalar_value=mse_losses.avg, global_step=epoch + 1)
    writer.add_scalar(tag="Valid/ELBO", scalar_value=elbo_losses.avg, global_step=epoch + 1)
    all_score = torch.cat(all_score, dim=0).detach()
    all_label = torch.cat(all_label, dim=0).detach()
    _, y_true = torch.topk(all_label, k=1, dim=1)
    _, y_pred = torch.topk(all_score, k=5, dim=1)
    # calculate accuracy by hand
    valid_top_1_accuracy = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
    valid_top_5_accuracy = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
    writer.add_scalar(tag="Valid/top1 accuracy", scalar_value=valid_top_1_accuracy, global_step=epoch + 1)
    if args.dataset == "Cifar100":
        writer.add_scalar(tag="Valid/top 5 accuracy", scalar_value=valid_top_5_accuracy, global_step=epoch + 1)
    if epoch % args.reconstruct_freq == 0:
        with torch.no_grad():
            image = utils.make_grid(image[:4, ...], nrow=2)
            reconstruct_image = utils.make_grid(torch.sigmoid(reconstruction[:4, ...]), nrow=2)
        writer.add_image(tag="Valid/Raw_Image", img_tensor=image, global_step=epoch + 1)
        writer.add_image(tag="Valid/Reconstruct_Image", img_tensor=reconstruct_image, global_step=epoch + 1)

    return valid_top_1_accuracy, valid_top_5_accuracy


def test(test_dloader, model, elbo_criterion, epoch, writer, discrete_latent_dim):
    continuous_kl_losses = AverageMeter()
    discrete_kl_losses = AverageMeter()
    mse_losses = AverageMeter()
    elbo_losses = AverageMeter()
    model.eval()
    all_score = []
    all_label = []

    for i, (image, label) in enumerate(test_dloader):
        image = image.float().cuda()
        label = label.long().cuda()
        label_onehot = torch.zeros(label.size(0), discrete_latent_dim).cuda().scatter_(1, label.view(-1, 1), 1)
        batch_size = image.size(0)
        with torch.no_grad():
            reconstruction, norm_mean, norm_log_sigma, disc_log_alpha, *_ = model(image)
        reconstruct_loss, continuous_kl_loss, discrete_kl_loss = elbo_criterion(image, reconstruction, norm_mean,
                                                                                norm_log_sigma, disc_log_alpha)
        mse_loss = F.mse_loss(torch.sigmoid(reconstruction.detach()), image.detach(),
                              reduction="sum") / (
                           2 * image.size(0) * (args.x_sigma ** 2))
        mse_losses.update(float(mse_loss), image.size(0))
        all_score.append(torch.exp(disc_log_alpha))
        all_label.append(label_onehot)
        continuous_kl_losses.update(float(continuous_kl_loss.item()), batch_size)
        discrete_kl_losses.update(float(discrete_kl_loss.item()), batch_size)
        elbo_losses.update(float(mse_loss + 0.01 * (continuous_kl_loss + discrete_kl_loss)), image.size(0))

    writer.add_scalar(tag="Test/KL(q(z|X)||p(z))", scalar_value=continuous_kl_losses.avg, global_step=epoch + 1)
    writer.add_scalar(tag="Test/KL(q(y|X)||p(y))", scalar_value=discrete_kl_losses.avg, global_step=epoch + 1)
    writer.add_scalar(tag="Test/log(p(X|z,y))", scalar_value=mse_losses.avg, global_step=epoch + 1)
    writer.add_scalar(tag="Test/ELBO", scalar_value=elbo_losses.avg, global_step=epoch + 1)
    all_score = torch.cat(all_score, dim=0).detach()
    all_label = torch.cat(all_label, dim=0).detach()
    _, y_true = torch.topk(all_label, k=1, dim=1)
    _, y_pred = torch.topk(all_score, k=5, dim=1)
    # calculate accuracy by hand
    test_top_1_accuracy = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
    test_top_5_accuracy = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
    writer.add_scalar(tag="Test/top1 accuracy", scalar_value=test_top_1_accuracy, global_step=epoch + 1)
    if args.dataset == "Cifar100":
        writer.add_scalar(tag="Test/top 5 accuracy", scalar_value=test_top_5_accuracy, global_step=epoch + 1)
    if epoch % args.reconstruct_freq == 0:
        with torch.no_grad():
            image = utils.make_grid(image[:4, ...], nrow=2)
            reconstruct_image = utils.make_grid(torch.sigmoid(reconstruction[:4, ...]), nrow=2)
        writer.add_image(tag="Test/Raw_Image", img_tensor=image, global_step=epoch + 1)
        writer.add_image(tag="Test/Reconstruct_Image", img_tensor=reconstruct_image, global_step=epoch + 1)

    return test_top_1_accuracy, test_top_5_accuracy


def modify_lr_rate(opt, lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr


def alpha_schedule(epoch, max_epoch, alpha_max):
    alpha = alpha_max * math.exp(-5 * (1 - min(1, epoch / max_epoch)) ** 2)
    return alpha


if __name__ == "__main__":
    main()
