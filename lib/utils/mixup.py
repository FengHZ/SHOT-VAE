import numpy as np
import torch


def mixup_vae_data(image, z_mean, z_log_sigma, disc_log_alpha, alpha=1.0, disc_label=None):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = image.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_image = lam * image + (1 - lam) * image[index, :]
    mixed_z_mean = lam * z_mean + (1 - lam) * z_mean[index]
    mixed_z_sigma = lam * torch.exp(z_log_sigma) + (1 - lam) * torch.exp(z_log_sigma[index])
    mixed_disc_alpha = lam * torch.exp(disc_log_alpha) + (1 - lam) * torch.exp(disc_log_alpha[index])
    if disc_label is not None:
        disc_label_mixup = disc_label[index]
        return mixed_image, mixed_z_mean, mixed_z_sigma, mixed_disc_alpha, disc_label_mixup, lam
    else:
        return mixed_image, mixed_z_mean, mixed_z_sigma, mixed_disc_alpha, lam


def mixup_raw_labeled_data(image, label, label_weight, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = image.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_image = lam * image + (1 - lam) * image[index, :]
    label_a, label_b = label, label[index]
    label_weight_a, label_weight_b = label_weight, label_weight[index]
    return mixed_image, label_a, label_b, label_weight_a, label_weight_b, lam


def mixup_criterion(criterion, prediction, label_a, label_b, lam):
    """
    :param criterion: the cross entropy criterion
    :param prediction: y_pred
    :param label_a: label = lam * label_a + (1-lam)* label_b
    :param label_b: label = lam * label_a + (1-lam)* label_b
    :param lam: label = lam * label_a + (1-lam)* label_b
    :return:  cross_entropy(pred,label)
    """
    return lam * criterion(label_a, prediction) + (1 - lam) * criterion(label_b, prediction)


def mixup_data(image, label, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = image.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_image = lam * image + (1 - lam) * image[index, :]
    label_a, label_b = label, label[index]
    return mixed_image, label_a, label_b, lam
