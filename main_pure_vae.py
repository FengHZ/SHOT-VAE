import random

random.seed(1)
import numpy as np

np.random.seed(1)
import argparse
from ospot_vae_model.vae import VariationalAutoEncoder
from lib.criterion import VAECriterion, KLDiscCriterion, KLNormCriterion
from lib.utils.avgmeter import AverageMeter
from lib.utils.mixup import mixup_vae_data
from lib.dataloader import cifar10_dataset, get_cifar10_sl_sampler, cifar100_dataset, get_cifar100_sl_sampler
import os
from os import path
import time
import shutil
import ast
import math


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v


parser = argparse.ArgumentParser(description='Pytorch Training VAE for Cifar10,Cifar100,SVHN Dataset')
# Dataset Parameters
parser.add_argument('-bp', '--base_path', default="/data/fhz")
parser.add_argument('--dataset', default="Cifar10", type=str, help="name of dataset used")
parser.add_argument('-is', "--image-size", default=[32, 32], type=arg_as_list,
                    metavar='Image Size List', help='the size of h * w for image')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=250, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
# VAE Train Strategy Parameters
parser.add_argument('-t', '--train-time', default=1, type=int,
                    metavar='N', help='the x-th time of training')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--dp', '--data-parallel', action='store_false', help='Use Data Parallel')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--reconstruct-freq', '-rf', default=20, type=int,
                    metavar='N', help='reconstruct frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume-arg', action='store_false', help='if we not resume the argument')
# Deep Learning Model Parameters
parser.add_argument('--net-name', default="wideresnet-28-2", type=str, help="the name for network to use")
parser.add_argument('--temperature', default=0.67, type=float,
                    help='centeralization parameter')
parser.add_argument('-dr', '--drop-rate', default=0, type=float, help='drop rate for the network')
parser.add_argument("--br", "--bce-reconstruction", action='store_true', help='Do BCE Reconstruction')
parser.add_argument("-s", "--x-sigma", default=1, type=float,
                    help="The standard variance for reconstructed images, work as regularization")
parser.add_argument('--ldc', "--latent-dim-continuous", default=128, type=int,
                    metavar='Latent Dim For Continuous Variable',
                    help='feature dimension in latent space for continuous variable')
parser.add_argument('--ldd', "--latent-dim-discrete", default=10, type=int,
                    metavar='Latent Dim For Discrete Variable',
                    help='feature dimension in latent space for discrete variable')
parser.add_argument('--cmi', "--continuous-mutual-info", default=1280, type=float,
                    help='The mutual information bounding between x and the continuous variable z')
# dim usually set as -log(1/ldd)
parser.add_argument('--dmi', "--discrete-mutual-info", default=2.3, type=float,
                    help='The mutual information bounding between x and the discrete variable z')
parser.add_argument('--kbm', '--kl-beta-max', default=0.1, type=float, metavar='KL Beta',
                    help='the epoch to linear adjust kl beta')
parser.add_argument('--akb', '--adjust-kl-beta-epoch', default=200, type=int, metavar='KL Beta',
                    help='the max epoch to adjust kl beta')
# Optimizer Parameters
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-b1', '--beta1', default=0.9, type=float, metavar='Beta1 In ADAM', help='beta1 for adam')
parser.add_argument('-ad', "--adjust-lr", default=[200], type=arg_as_list,
                    help="The milestone list for adjust learning rate")
parser.add_argument('--wd', '--weight-decay', default=0, type=float)
# Mixup Strategy Parameters
parser.add_argument('--mixup', default=False, type=bool, help="use mixup method")
parser.add_argument('--ma', "--mixup-alpha", default=0.1, type=float,
                    help="the mixup alpha for data")
# parser.add_argument("--prior-weight", default=0.01, type=float, help="The weight for the prior kl divergency loss")
parser.add_argument("--apw", "--adjust-posterior-weight", default=200, type=int,
                    help="The adjust epoch for the posterior kl divergency loss")
parser.add_argument("--pwm", "--posterior-weight-max", default=1, type=float,
                    help="The max weight for the posterior kl divergency loss")
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
        sampler_valid, sampler_train = get_cifar10_sl_sampler(
            torch.tensor(train_dataset.targets, dtype=torch.int32), 500, 10)
        test_dloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        valid_dloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                                   sampler=sampler_valid)
        train_dloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                   pin_memory=True,
                                   sampler=sampler_train)
        input_channels = 3
        small_input = True
    elif args.dataset == "Cifar100":
        dataset_base_path = path.join(args.base_path, "dataset", "cifar")
        train_dataset = cifar100_dataset(dataset_base_path)
        test_dataset = cifar100_dataset(dataset_base_path, train_flag=False)
        sampler_valid, sampler_train = get_cifar100_sl_sampler(
            torch.tensor(train_dataset.targets, dtype=torch.int32), 50, 100)
        test_dloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        valid_dloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                                   sampler=sampler_valid)
        train_dloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                   pin_memory=True,
                                   sampler=sampler_train)
        input_channels = 3
        small_input = True
    else:
        raise NotImplementedError("Dataset {} not implemented".format(args.dataset))
    model = VariationalAutoEncoder(num_input_channels=input_channels, encoder_name=args.net_name,
                                   drop_rate=args.drop_rate, img_size=tuple(args.image_size),
                                   data_parallel=args.dp, continuous_latent_dim=args.ldc, disc_latent_dim=args.ldd,
                                   sample_temperature=args.temperature, small_input=small_input)
    model = model.cuda()
    input("Begin the {} time's training, Dataset {}".format(
        args.train_time, args.dataset))
    elbo_criterion = VAECriterion(discrete_dim=args.ldd,x_sigma=args.x_sigma, bce_reconstruction=args.br).cuda()
    if args.mixup:
        kl_disc_criterion = KLDiscCriterion().cuda()
        kl_norm_criterion = KLNormCriterion().cuda()
    else:
        kl_disc_criterion = None
        kl_norm_criterion = None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    scheduler = MultiStepLR(optimizer, milestones=args.adjust_lr)
    writer_log_dir = "{}/{}-VAE/runs/train_time:{}".format(args.base_path, args.dataset,
                                                           args.train_time)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.resume_arg:
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
    best_valid_loss = 1e10
    for epoch in range(args.start_epoch, args.epochs):

        train(train_dloader, model=model, elbo_criterion=elbo_criterion, optimizer=optimizer, epoch=epoch,
              writer=writer, kl_norm_criterion=kl_norm_criterion, kl_disc_criterion=kl_disc_criterion)
        elbo_valid_loss, *_ = valid(valid_dloader, model=model, elbo_criterion=elbo_criterion, epoch=epoch,
                                    writer=writer)
        if test_dloader is not None:
            test(test_dloader, model=model, elbo_criterion=elbo_criterion, epoch=epoch, writer=writer)

        save_checkpoint({
            'epoch': epoch + 1,
            'args': args,
            "state_dict": model.state_dict(),
            'optimizer': optimizer.state_dict(),
        })
        if elbo_valid_loss < best_valid_loss:
            best_valid_loss = elbo_valid_loss
            if epoch >= args.adjust_lr[-1]:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'args': args,
                    "state_dict": model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, best_predict=True)
        scheduler.step(epoch)


def train(train_dloader, model, elbo_criterion, optimizer, epoch, writer, kl_disc_criterion=None,
          kl_norm_criterion=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    reconstruct_losses = AverageMeter()
    kl_losses = AverageMeter()
    elbo_losses = AverageMeter()
    mse_losses = AverageMeter()
    model.train()
    end = time.time()
    optimizer.zero_grad()
    # mutual information
    cmi = args.cmi * alpha_schedule(epoch, args.akb, 1, strategy="exp")
    dmi = args.dmi * alpha_schedule(epoch, args.akb, 1, strategy="exp")
    kl_beta = alpha_schedule(epoch, args.akb, args.kbm)
    posterior_weight = alpha_schedule(epoch, args.apw, args.pwm)
    # mixup parameters:
    if args.mixup:
        mixup_posterior_kl_losses = AverageMeter()
        # mixup_prior_kl_losses = AverageMeter()
        # mixup_elbo_losses = AverageMeter()
        # mixup_reconstruct_losses = AverageMeter()
    print("Begin {} Epoch Training, CMI:{:.4f} DMI:{:.4f} KL-Beta{:.4f}".format(epoch + 1, cmi, dmi, kl_beta))
    for i, (image, *_) in enumerate(train_dloader):
        data_time.update(time.time() - end)
        image = image.float().cuda()
        batch_size = image.size(0)
        reconstruction, norm_mean, norm_log_sigma, disc_log_alpha, *_ = model(image)
        reconstruct_loss, continuous_kl_loss, disc_kl_loss = elbo_criterion(image, reconstruction, norm_mean,
                                                                            norm_log_sigma, disc_log_alpha)
        kl_loss = torch.abs(continuous_kl_loss - cmi) + torch.abs(disc_kl_loss - dmi)
        elbo_loss = reconstruct_loss + kl_beta * kl_loss
        elbo_loss.backward()
        if args.mixup:
            with torch.no_grad():
                mixed_image, mixed_z_mean, mixed_z_sigma, mixed_disc_alpha, lam = mixup_vae_data(image,
                                                                                                 norm_mean,
                                                                                                 norm_log_sigma,
                                                                                                 disc_log_alpha,
                                                                                                 alpha=args.ma)
            mixed_reconstruction, norm_mean, norm_log_sigma, disc_log_alpha, *_ = model(mixed_image)
            # continuous_kl_posterior_loss = kl_norm_criterion(norm_mean, norm_log_sigma, z_mean_gt=mixed_z_mean,
            #                                                  z_sigma_gt=mixed_z_sigma)
            disc_kl_posterior_loss = kl_disc_criterion(disc_log_alpha, mixed_disc_alpha)
            continuous_kl_posterior_loss = (F.mse_loss(norm_mean, mixed_z_mean, reduction="sum") + \
                                            F.mse_loss(torch.exp(norm_log_sigma), mixed_z_sigma,
                                                       reduction="sum")) / batch_size
            # disc_kl_posterior_loss = F.mse_loss(torch.exp(disc_log_alpha), mixed_disc_alpha,
            #                                     reduction="sum") / batch_size
            mixup_kl_posterior_loss = posterior_weight * (continuous_kl_posterior_loss + disc_kl_posterior_loss)
            mixup_kl_posterior_loss.backward()
            mixup_posterior_kl_losses.update(float(mixup_kl_posterior_loss), mixed_image.size(0))

        elbo_losses.update(float(elbo_loss), image.size(0))
        reconstruct_losses.update(float(reconstruct_loss), image.size(0))
        kl_losses.update(float(kl_loss), image.size(0))
        # calculate mse_losses if we use bce reconstruction loss
        if args.br:
            mse_loss = F.mse_loss(torch.sigmoid(reconstruction.detach()), image.detach(), reduction="sum") / (
                    2 * image.size(0) * (args.x_sigma ** 2))
            mse_losses.update(float(mse_loss), image.size(0))
        optimizer.step()
        optimizer.zero_grad()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            train_text = 'Epoch: [{0}][{1}/{2}]\t' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                         'ELBO Loss {elbo_loss.val:.4f} ({elbo_loss.avg:.4f})\t' \
                         'Reconstruct Loss {reconstruct_loss.val:.4f} ({reconstruct_loss.avg:.4f})\t' \
                         'KL Loss {kl_loss.val:.4f} ({kl_loss.avg:.4f})\t'.format(
                epoch, i + 1, len(train_dloader), batch_time=batch_time, data_time=data_time,
                elbo_loss=elbo_losses, reconstruct_loss=reconstruct_losses, kl_loss=kl_losses)
            print(train_text)

    if args.mixup:
        # writer.add_scalar(tag="Train/Mixup-ELBO", scalar_value=mixup_elbo_losses.avg, global_step=epoch + 1)
        # writer.add_scalar(tag="Train/Mixup-Reconstruct", scalar_value=mixup_reconstruct_losses.avg,
        #                   global_step=epoch + 1)
        # writer.add_scalar(tag="Train/Mixup-KL-Prior", scalar_value=mixup_prior_kl_losses.avg, global_step=epoch + 1)
        writer.add_scalar(tag="Train/Mixup-KL-Posterior", scalar_value=mixup_posterior_kl_losses.avg,
                          global_step=epoch + 1)
    writer.add_scalar(tag="Train/ELBO", scalar_value=elbo_losses.avg, global_step=epoch + 1)
    writer.add_scalar(tag="Train/Reconstruct", scalar_value=reconstruct_losses.avg,
                      global_step=epoch + 1)
    writer.add_scalar(tag="Train/KL", scalar_value=kl_losses.avg, global_step=epoch + 1)
    if args.br:
        writer.add_scalar(tag="Train/MSE", scalar_value=mse_losses.avg,
                          global_step=epoch)
    # after several epoch training, we add the image and reconstructed image into the image board, we just use 16 images
    if epoch % args.reconstruct_freq == 0:
        with torch.no_grad():
            image = utils.make_grid(image[:4, ...], nrow=2)
            reconstruct_image = utils.make_grid(torch.sigmoid(reconstruction[:4, ...]), nrow=2)
        writer.add_image(tag="Train/Raw_Image", img_tensor=image, global_step=epoch + 1)
        writer.add_image(tag="Train/Reconstruct_Image", img_tensor=reconstruct_image, global_step=epoch + 1)
    return elbo_losses.avg, reconstruct_losses.avg, kl_losses.avg


def save_checkpoint(state, filename='checkpoint.pth.tar', best_predict=False):
    """
    :param state: a dict including:{
                'epoch': epoch + 1,
                'args': args,
                "state_dict": ospot_vae_model.state_dict(),
                'optimizer': optimizer.state_dict(),
        }
    :param filename: the filename for store
    :param best_predict: the best predict flag
    :return:
    """
    filefolder = '{}/{}-VAE/parameter/train_time_{}'.format(args.base_path, args.dataset,
                                                            state["args"].train_time)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    if best_predict:
        filename = 'best.pth.tar'
        torch.save(state, path.join(filefolder, filename))
    else:
        torch.save(state, path.join(filefolder, filename))


def valid(valid_dloader, model, elbo_criterion, epoch, writer):
    reconstruct_losses = AverageMeter()
    kl_losses = AverageMeter()
    elbo_losses = AverageMeter()
    mse_losses = AverageMeter()
    model.eval()
    # mutual information
    cmi = args.cmi * min(1.0, float(epoch / args.adjust_lr[0]))
    dmi = args.dmi * min(1.0, float(epoch / args.adjust_lr[0]))

    for i, (image, *_) in enumerate(valid_dloader):
        image = image.float().cuda()
        with torch.no_grad():
            reconstruction, norm_mean, norm_log_sigma, disc_log_alpha, *_ = model(image)
            reconstruct_loss, continuous_kl_loss, disc_kl_loss = elbo_criterion(image, reconstruction, norm_mean,
                                                                                norm_log_sigma, disc_log_alpha)
            kl_loss = torch.abs(continuous_kl_loss - cmi) + torch.abs(disc_kl_loss - dmi)
            elbo_loss = reconstruct_loss + kl_loss
            if args.br:
                mse_loss = F.mse_loss(torch.sigmoid(reconstruction.detach()), image.detach(),
                                      reduction="sum") / (
                                   2 * image.size(0) * (args.x_sigma ** 2))
                mse_losses.update(float(mse_loss), image.size(0))
        elbo_losses.update(float(elbo_loss), image.size(0))
        reconstruct_losses.update(float(reconstruct_loss), image.size(0))
        kl_losses.update(float(kl_loss), image.size(0))
    writer.add_scalar(tag="Valid/ELBO", scalar_value=elbo_losses.avg, global_step=epoch + 1)
    writer.add_scalar(tag="Valid/Reconstruct", scalar_value=reconstruct_losses.avg,
                      global_step=epoch + 1)
    writer.add_scalar(tag="Valid/KL", scalar_value=kl_losses.avg, global_step=epoch + 1)
    if args.br:
        writer.add_scalar(tag="Valid/MSE", scalar_value=mse_losses.avg,
                          global_step=epoch)
    if epoch % args.reconstruct_freq == 0:
        with torch.no_grad():
            image = utils.make_grid(image[:4, ...], nrow=2)
            reconstruct_image = utils.make_grid(torch.sigmoid(reconstruction[:4, ...]), nrow=2)
        writer.add_image(tag="Valid/Raw_Image", img_tensor=image, global_step=epoch + 1)
        writer.add_image(tag="Valid/Reconstruct_Image", img_tensor=reconstruct_image, global_step=epoch + 1)
    return elbo_losses.avg, reconstruct_losses.avg, kl_losses.avg


# def inference(train_dloader, valid_dloader, ospot_vae_model):
#     ospot_vae_model.eval()
#     train_inference_dict = {}
#     for i, (image, index, image_name, *_) in enumerate(train_dloader):
#         image = image.float().cuda()
#         with torch.no_grad():
#             reconstruction, norm_mean, norm_log_sigma, disc_param_list, *_ = ospot_vae_model(image)
#             norm_sigma = torch.exp(norm_log_sigma)
#         for i, img_name in enumerate(image_name):
#             train_inference_dict[img_name] = {"mean": norm_mean[i, :].detach().cpu(),
#                                               "sigma": norm_sigma[i, :].detach().cpu()}
#     valid_inference_dict = {}
#     for i, (image, index, image_name, *_) in enumerate(valid_dloader):
#         image = image.float().cuda()
#         with torch.no_grad():
#             reconstruction, norm_mean, norm_log_sigma, disc_param_list, *_ = ospot_vae_model(image)
#             norm_sigma = torch.exp(norm_log_sigma)
#         for i, img_name in enumerate(image_name):
#             valid_inference_dict[img_name] = {"mean": norm_mean[i, :].detach().cpu(),
#                                               "sigma": norm_sigma[i, :].detach().cpu()}
#     return train_inference_dict, valid_inference_dict


def test(test_dloader, model, elbo_criterion, epoch, writer):
    reconstruct_losses = AverageMeter()
    kl_losses = AverageMeter()
    elbo_losses = AverageMeter()
    mse_losses = AverageMeter()
    model.eval()
    # mutual information
    cmi = args.cmi * min(1.0, float(epoch / args.adjust_lr[0]))
    dmi = args.dmi * min(1.0, float(epoch / args.adjust_lr[0]))

    for i, (image, *_) in enumerate(test_dloader):
        image = image.float().cuda()
        with torch.no_grad():
            reconstruction, norm_mean, norm_log_sigma, disc_log_alpha, *_ = model(image)
            reconstruct_loss, continuous_kl_loss, disc_kl_loss = elbo_criterion(image, reconstruction, norm_mean,
                                                                                norm_log_sigma, disc_log_alpha)
            kl_loss = torch.abs(continuous_kl_loss - cmi) + torch.abs(disc_kl_loss - dmi)
            elbo_loss = reconstruct_loss + kl_loss
            if args.br:
                mse_loss = F.mse_loss(torch.sigmoid(reconstruction.detach()), image.detach(),
                                      reduction="sum") / (
                                   2 * image.size(0) * (args.x_sigma ** 2))
                mse_losses.update(float(mse_loss), image.size(0))
        elbo_losses.update(float(elbo_loss), image.size(0))
        reconstruct_losses.update(float(reconstruct_loss), image.size(0))
        kl_losses.update(float(kl_loss), image.size(0))
    writer.add_scalar(tag="Test/ELBO", scalar_value=elbo_losses.avg, global_step=epoch + 1)
    writer.add_scalar(tag="Test/Reconstruct", scalar_value=reconstruct_losses.avg,
                      global_step=epoch + 1)
    writer.add_scalar(tag="Test/KL", scalar_value=kl_losses.avg, global_step=epoch + 1)
    if args.br:
        writer.add_scalar(tag="Test/MSE", scalar_value=mse_losses.avg,
                          global_step=epoch)
    if epoch % args.reconstruct_freq == 0:
        with torch.no_grad():
            image = utils.make_grid(image[:4, ...], nrow=2)
            reconstruct_image = utils.make_grid(torch.sigmoid(reconstruction[:4, ...]), nrow=2)
        writer.add_image(tag="Test/Raw_Image", img_tensor=image, global_step=epoch + 1)
        writer.add_image(tag="Test/Reconstruct_Image", img_tensor=reconstruct_image, global_step=epoch + 1)
    return elbo_losses.avg, reconstruct_losses.avg, kl_losses.avg


def alpha_schedule(epoch, max_epoch, alpha_max, strategy="exp"):
    if strategy == "linear":
        alpha = alpha_max * min(1, epoch / max_epoch)
    elif strategy == "exp":
        alpha = alpha_max * math.exp(-5 * (1 - min(1, epoch / max_epoch)) ** 2)
    else:
        raise NotImplementedError("Strategy {} not implemented".format(strategy))
    return alpha


if __name__ == "__main__":
    main()
