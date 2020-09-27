import numpy as np
import torch


def mixup_vae_data(image, z_mean, z_log_sigma, disc_log_alpha, optimal_match=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(2.0, 2.0)
    batch_size = image.size()[0]
    if optimal_match:
        # use the optimal match to provide match index
        with torch.no_grad():
            kl_metric = torch.zeros(batch_size, batch_size)
            for i in range(batch_size):
                for j in range(batch_size):
                    kl_metric[i, j] = gaussian_kl_divergence_calculation(z_mean[i, ...], z_log_sigma[i, ...],
                                                                         z_mean[j, ...], z_log_sigma[j, ...])
        index = torch.argmin(kl_metric, dim=1)
    else:
        # use random permutation to provide match index
        index = torch.randperm(batch_size).cuda()
    mixed_image = lam * image + (1 - lam) * image[index, :]
    mixed_z_mean = lam * z_mean + (1 - lam) * z_mean[index]
    mixed_z_sigma = lam * torch.exp(z_log_sigma) + (1 - lam) * torch.exp(z_log_sigma[index])
    mixed_disc_alpha = lam * torch.exp(disc_log_alpha) + (1 - lam) * torch.exp(disc_log_alpha[index])
    return mixed_image, mixed_z_mean, mixed_z_sigma, mixed_disc_alpha, lam


def label_smoothing(image, z_mean, z_log_sigma, disc_log_alpha, epsilon=0.1, disc_label=None):
    if epsilon > 0:
        lam = np.random.beta(epsilon, epsilon)
    else:
        lam = 1
    batch_size = image.size()[0]
    index = torch.randperm(batch_size).cuda()
    smoothed_image = lam * image + (1 - lam) * image[index, :]
    smoothed_z_mean = lam * z_mean + (1 - lam) * z_mean[index]
    smoothed_z_sigma = lam * torch.exp(z_log_sigma) + (1 - lam) * torch.exp(z_log_sigma[index])
    smoothed_disc_alpha = lam * torch.exp(disc_log_alpha) + (1 - lam) * torch.exp(disc_log_alpha[index])
    smoothed_disc_label = disc_label[index]
    return smoothed_image, smoothed_z_mean, smoothed_z_sigma, smoothed_disc_alpha, smoothed_disc_label, lam


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


def gaussian_kl_divergence_calculation(z_mean_1, z_log_sigma_1, z_mean_2, z_log_sigma_2):
    dim = z_mean_1.size(0)
    z_sigma_1 = torch.exp(z_log_sigma_1)
    z_sigma_2 = torch.exp(z_log_sigma_2)
    kl_1_2 = torch.sum(z_log_sigma_2 - z_log_sigma_1) + 0.5 * torch.sum(
        z_sigma_1 ** 2 / z_sigma_2 ** 2) + 0.5 * torch.sum((z_mean_1 - z_mean_2) ** 2 / (z_sigma_2 ** 2)) - 0.5 * dim
    return kl_1_2
