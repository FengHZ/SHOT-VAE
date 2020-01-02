import torch
from torch import nn
import torch.nn.functional as F

eps = 1e-7


class VAECriterion(nn.Module):
    """
    Here we calculate the VAE loss
    VAE loss's math formulation is :
    E_{z~Q}[log(P(X|z))]-D[Q(z|X)||P(z)]
    which can be transformed into:
    ||X-X_{reconstructed}||^2/(\sigma)^2 - [<L2norm(u)>^2+<L2norm(diag(\Sigma))>^2
    -<L2norm(diag(ln(\Sigma)))>^2-1]
    Our input is :
    x_sigma,x_reconstructed,x,z_mean,z_Sigma
    """

    def __init__(self, discrete_dim=10, x_sigma=1, bce_reconstruction=True):
        """
        :param discrete_dim : the dim for discrete latent variables
        :param x_sigma:
        :param bce_reconstruction:
        """
        super(VAECriterion, self).__init__()
        self.x_sigma = x_sigma
        self.bce_reconstruction = bce_reconstruction
        self.disc_log_prior_param = torch.log(
            torch.tensor([1 / discrete_dim for i in range(discrete_dim)]).view(1, -1).float().cuda())

    def forward(self, x, x_reconstructed, z_mean, z_log_sigma, disc_log_alpha):
        """
        :param x: input & ground truth
        :param x_reconstructed: the reconstructed output by VAE
        :param z_mean: the mean of the continuous latent variable
        :param z_log_sigma : the log std of the continuous latent variable
        :param disc_log_alpha : the param list for the disc param
        :return: reconstruct_loss, continuous_kl_loss, disc_kl_loss_tensor
        """
        batch_size = x.size(0)
        # calculate reconstruct loss, sum in instance, mean in batch
        # we use the Binary Cross Entropy loss to do calculation
        if self.bce_reconstruction:
            reconstruct_loss = F.binary_cross_entropy_with_logits(x_reconstructed, x, reduction="sum") / (batch_size)
        else:
            reconstruct_loss = F.mse_loss(torch.sigmoid(x_reconstructed), x, reduction="sum") / (
                    2 * batch_size * (self.x_sigma ** 2))
        # calculate latent space KL divergence
        z_mean_sq = z_mean * z_mean
        z_log_sigma_sq = 2 * z_log_sigma
        z_sigma_sq = torch.exp(z_log_sigma_sq)
        continuous_kl_loss = 0.5 * torch.sum(z_mean_sq + z_sigma_sq - z_log_sigma_sq - 1) / batch_size
        # notice here we duplicate the 0.5 by each part
        # disc param : log(a1),...,log(an) type
        disc_kl_loss = torch.sum(torch.exp(disc_log_alpha) * (disc_log_alpha - self.disc_log_prior_param)) / batch_size
        return reconstruct_loss, continuous_kl_loss, disc_kl_loss


class ClsCriterion(nn.Module):
    def __init__(self):
        super(ClsCriterion, self).__init__()

    def forward(self, predict, label, batch_weight=None):
        """
        :param predict: B*C log_softmax result
        :param label: B*C one-hot label
        :param batch_weight: B*1 0-1 weight for each item in a batch
        :return: cross entropy loss
        """
        if batch_weight is None:
            cls_loss = -1 * torch.mean(torch.sum(predict * label, dim=1))
        else:
            cls_loss = -1 * torch.mean(torch.sum(predict * label, dim=1) * batch_weight)
        return cls_loss


class ReconstructionCriterion(nn.Module):
    """
    Here we calculate the criterion for -log p(x|z), we list two forms, the binary cross entropy form
    as well as the mse loss form
    """

    def __init__(self, x_sigma=1, bce_reconstruction=True):
        super(ReconstructionCriterion, self).__init__()
        self.x_sigma = x_sigma
        self.bce_reconstruction = bce_reconstruction

    def forward(self, x, x_reconstructed):
        batch_size = x.size(0)
        # calculate reconstruct loss, sum in instance, mean in batch
        # we use the Binary Cross Entropy loss to do calculation
        if self.bce_reconstruction:
            reconstruct_loss = F.binary_cross_entropy_with_logits(x_reconstructed, x, reduction="sum") / (batch_size)
        else:
            reconstruct_loss = F.mse_loss(torch.sigmoid(x_reconstructed), x, reduction="sum") / (
                    2 * batch_size * (self.x_sigma ** 2))
        return reconstruct_loss


class KLNormCriterion(nn.Module):
    def __init__(self):
        super(KLNormCriterion, self).__init__()

    def forward(self, z_mean_pre, z_log_sigma_pre, z_mean_gt=None, z_sigma_gt=None):
        batch_size = z_mean_pre.size(0)
        if z_mean_gt is None or z_sigma_gt is None:
            """
            KL[N(z_mean_pre,z_sigma_pre)||N(0,I)]
            """
            z_mean_sq = z_mean_pre * z_mean_pre
            z_log_sigma_sq = 2 * z_log_sigma_pre
            z_sigma_sq = torch.exp(z_log_sigma_sq)
            kl_loss = 0.5 * torch.sum(z_mean_sq + z_sigma_sq - z_log_sigma_sq - 1) / batch_size
        else:
            """
            KL[N(z_mean_pre,z_sigma_pre)||N(z_mean_gt,z_sigma_gt)]
            """
            z_log_sigma_sq_pre = 2 * z_log_sigma_pre
            z_sigma_sq_pre = torch.exp(z_log_sigma_sq_pre)
            z_log_sigma_sq_gt = 2 * torch.log(z_sigma_gt + 1e-4)
            z_sigma_sq_gt = z_sigma_gt ** 2
            kl_loss = 0.5 * torch.sum(z_log_sigma_sq_gt - z_log_sigma_sq_pre + z_sigma_sq_pre / z_sigma_sq_gt + (
                    z_mean_pre - z_mean_gt) ** 2 / z_sigma_sq_gt - 1) / batch_size
        return kl_loss


class KLDiscCriterion(nn.Module):
    """
    calculate
    sum (j=1,...,K) D_KL[q(c_j|x)||p(c_j|x)]
    """

    def __init__(self):
        super(KLDiscCriterion, self).__init__()

    def forward(self, disc_log_pre, disc_gt, qp_order=True):
        batch_size = disc_log_pre.size(0)
        disc_log_gt = torch.log(disc_gt + 1e-4)
        if qp_order:
            loss = torch.sum(torch.exp(disc_log_pre) * (disc_log_pre - disc_log_gt)) / batch_size
        else:
            loss = torch.sum(disc_gt * (disc_log_gt - disc_log_pre)) / batch_size
        return loss
