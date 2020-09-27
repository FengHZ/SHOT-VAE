import torch
from torch import nn
from densenet import get_densenet
from preactresnet import get_preact_resnet
from wideresnet import get_wide_resnet
from decoder import Decoder
import numpy as np


class _Inference(nn.Sequential):
    def __init__(self, num_input_channels, latent_dim, disc_variable=True):
        super(_Inference, self).__init__()
        self.add_module('fc', nn.Linear(num_input_channels, latent_dim))
        if disc_variable:
            self.add_module('log_softmax', nn.LogSoftmax(dim=1))


class Sample(nn.Module):
    def __init__(self, temperature):
        super(Sample, self).__init__()
        self._temperature = temperature

    def forward(self, norm_mean, norm_log_sigma, disc_log_alpha, disc_label=None, mixup=False, disc_label_mixup=None,
                mixup_lam=None):
        """
        :param norm_mean: mean parameter of continuous norm variable
        :param norm_log_sigma: log sigma parameter of continuous norm variable
        :param disc_log_alpha: log alpha parameter of discrete multinomial variable
        :param disc_label: the ground truth label of discrete variable (not one-hot label)
        :param mixup: if we do mixup
        :param disc_label_mixup: the mixup target label
        :param mixup_lam: the mixup lambda
        :return: sampled latent variable
        """
        batch_size = norm_mean.size(0)
        latent_sample = list([])
        latent_sample.append(self._sample_norm(norm_mean, norm_log_sigma))
        if disc_label is not None:
            # it means we have the real label, then we can use real label instead of sampling
            # c: N*c onehot
            if mixup:
                c_a = torch.zeros(disc_log_alpha.size()).cuda()
                c_a = c_a.scatter(1, disc_label.view(-1, 1), 1)
                c_b = torch.zeros(disc_log_alpha.size()).cuda()
                c_b = c_b.scatter(1, disc_label_mixup.view(-1, 1), 1)
                c = mixup_lam * c_a + (1 - mixup_lam) * c_b
            else:
                c = torch.zeros(disc_log_alpha.size()).cuda()
                c = c.scatter(1, disc_label.view(-1, 1), 1)
            latent_sample.append(c)
        else:
            latent_sample.append(self._sample_gumbel_softmax(disc_log_alpha))
        latent_sample = torch.cat(latent_sample, dim=1)
        dim_size = latent_sample.size(1)
        latent_sample = latent_sample.view(batch_size, dim_size, 1, 1)
        return latent_sample

    def _sample_gumbel_softmax(self, log_alpha):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.

        Parameters
        ----------
        log_alpha : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D)
        """
        EPS = 1e-12
        unif = torch.rand(log_alpha.size()).cuda()
        gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
        # Reparameterize to create gumbel softmax sample
        logit = (log_alpha + gumbel) / self._temperature
        return torch.softmax(logit, dim=1)

    @staticmethod
    def _sample_norm(mu, log_sigma):
        """
        :param mu: the mu for sampling with N*D
        :param log_sigma: the log_sigma for sampling with N*D
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        std_z = torch.randn(mu.size())
        if mu.is_cuda:
            std_z = std_z.cuda()

        return mu + torch.exp(log_sigma) * std_z


class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder_name, num_input_channels=1, drop_rate=0, img_size=(160, 160), data_parallel=True,
                 continuous_latent_dim=100, disc_latent_dim=10, sample_temperature=0.67, small_input=False):
        super(VariationalAutoEncoder, self).__init__()
        if "densenet" in encoder_name:
            self.feature_extractor = get_densenet(encoder_name, drop_rate, input_channels=num_input_channels,
                                                  small_input=small_input,
                                                  data_parallel=data_parallel)
        elif "wideresnet" in encoder_name:
            self.feature_extractor = get_wide_resnet(encoder_name, drop_rate, input_channels=num_input_channels,
                                                     small_input=small_input,
                                                     data_parallel=data_parallel)
        elif "preactresnet" in encoder_name:
            self.feature_extractor = get_preact_resnet(encoder_name, drop_rate, input_channels=num_input_channels,
                                                       small_input=small_input,
                                                       data_parallel=data_parallel)
        else:
            raise NotImplementedError("{} not implemented".format(encoder_name))
        global_avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if data_parallel:
            global_avg = nn.DataParallel(global_avg)
        self.global_avg = global_avg
        self.continuous_inference = nn.Sequential()
        self.disc_latent_inference = nn.Sequential()
        conti_mean_inf_module = _Inference(num_input_channels=self.feature_extractor.num_feature_channel,
                                           latent_dim=continuous_latent_dim,
                                           disc_variable=False)
        conti_logsigma_inf_module = _Inference(num_input_channels=self.feature_extractor.num_feature_channel,
                                               latent_dim=continuous_latent_dim,
                                               disc_variable=False)
        if data_parallel:
            conti_mean_inf_module = nn.DataParallel(conti_mean_inf_module)
            conti_logsigma_inf_module = nn.DataParallel(conti_logsigma_inf_module)
        self.continuous_inference.add_module("mean", conti_mean_inf_module)
        self.continuous_inference.add_module("log_sigma", conti_logsigma_inf_module)
        self._disc_latent_dim = disc_latent_dim
        dic_inf = _Inference(num_input_channels=self.feature_extractor.num_feature_channel, latent_dim=disc_latent_dim,
                             disc_variable=True)
        if data_parallel:
            dic_inf = nn.DataParallel(dic_inf)
        self.disc_latent_inference = dic_inf
        sample = Sample(temperature=sample_temperature)
        if data_parallel:
            sample = nn.DataParallel(sample)
        self.sample = sample
        decoder_kernel_size = tuple([int(s / 32) for s in img_size])
        self.feature_reconstructor = Decoder(num_channel=num_input_channels,
                                             latent_dim=int(continuous_latent_dim + np.sum(disc_latent_dim)),
                                             data_parallel=data_parallel,
                                             kernel_size=decoder_kernel_size)

    def forward(self, input_img, mixup=False, disc_label=None, disc_pseudo_label=None, mixup_lam=None):
        batch_size = input_img.size(0)
        features = self.feature_extractor(input_img)
        avg_features = self.global_avg(features).view(batch_size, -1)
        norm_mean = self.continuous_inference.mean(avg_features)
        norm_log_sigma = self.continuous_inference.log_sigma(avg_features)
        disc_log_alpha = self.disc_latent_inference(avg_features)
        latent_sample = self.sample(norm_mean=norm_mean, norm_log_sigma=norm_log_sigma, disc_log_alpha=disc_log_alpha,
                                    disc_label=disc_label, mixup=mixup, disc_label_mixup=disc_pseudo_label,
                                    mixup_lam=mixup_lam)
        reconstruction = self.feature_reconstructor(latent_sample)
        return reconstruction, norm_mean, norm_log_sigma, disc_log_alpha


if __name__ == "__main__":
    model = VariationalAutoEncoder(encoder_name="densenet121",
                                   img_size=(320, 448),
                                   data_parallel=True,
                                   continuous_latent_dim=16,
                                   disc_latent_dim=10,
                                   sample_temperature=0.67,
                                   small_input=False).cuda()
    a = torch.randn(1, 1, 320, 448).cuda()
    print(model(a))
