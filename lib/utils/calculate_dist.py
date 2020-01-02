import numpy as np
import torch


def gaussian_wd_calculation(u1, u2, log_sigma1, log_sigma2, diagflag=True):
    if diagflag:
        wd = np.sum((u1 - u2) ** 2) + np.sum((np.exp(log_sigma1) - np.exp(log_sigma2)) ** 2)
        return wd
    else:
        raise NotImplementedError("No diag covariance matrix not implemented")


def gaussian_kl_calculation(u1, u2, log_sigma1, log_sigma2, diag1flag=True, diag2flag=True):
    """
    The calculation method is follow https://fenghz.github.io/KL-Divergency-Description/
    :param u1: the mean of variable1, numpy vector length m
    :param u2: the mean of variable2, numpy vector length m
    :param log_sigma1: the cov matrix of variable1, size m*m
    :param log_sigma2: the cov matrix of variable2, size m*m
    :param diag1flag: if true them sigma 1 is a numpy vector with length m, storing diag elements
    :param diag2flag: if true then sigma 2 is a numpy vector with length m, storing diag elements
    :return:D[p1||p2]
    """
    sigma1sq = np.exp(log_sigma1) ** 2
    sigma2sq = np.exp(log_sigma2) ** 2
    dim = u1.shape[0]
    if diag1flag == True and diag2flag == True:
        kl = (1 / 2) * (np.sum(np.log(sigma2sq)) - np.sum(np.log(sigma1sq)) + np.sum(sigma1sq / sigma2sq) + np.sum(
            (u1 - u2) ** 2 / sigma2sq) - dim)
        return kl
    else:
        raise ValueError("Undefined value for diag1flag and diag2flag")


def gaussian_kl_calculation_vec(u, log_sigma, GPU_flag=False):
    """
    GPU part finished
    :param u: n*m-dim vector
    :param log_sigma: n*m-dim vector
    :return: n*n KL divergency distance matrix,[i,j] means KL[i,j]
    """
    if GPU_flag:
        u = torch.from_numpy(u).cuda()
        log_sigma = torch.from_numpy(log_sigma).cuda()
        sigma_sq = torch.exp(log_sigma) ** 2
        n, d = sigma_sq.size()
        s_pairwise = torch.unsqueeze(sigma_sq, 1) / torch.unsqueeze(sigma_sq, 0)
        u_pairwise = (torch.unsqueeze(u, 1) - torch.unsqueeze(u, 0)) ** 2 / torch.unsqueeze(sigma_sq, 0)
        kl = (1 / 2) * (torch.sum(torch.log(s_pairwise.permute(1, 0, 2)), dim=2) +
                        torch.sum(s_pairwise, dim=2) + torch.sum(u_pairwise, dim=2) - d)
    else:
        sigma_sq = np.exp(log_sigma) ** 2
        n, dim = sigma_sq.shape
        s_pairwise = sigma_sq.reshape(n, 1, dim) / sigma_sq.reshape(1, n, dim)
        u_pairwise = (u.reshape(n, 1, dim) - u.reshape(1, n, dim)) ** 2 / sigma_sq.reshape(1, n, dim)
        kl = (1 / 2) * (np.sum(np.log(np.swapaxes(s_pairwise, 0, 1)), axis=2) + np.sum(s_pairwise, axis=2) +
                        np.sum(u_pairwise, axis=2) - dim)
    return kl


def gaussian_kl_calculation_vec_pairwise(u1, log_sigma1, u2, log_sigma2, GPU_flag=False):
    """
    GPU part finished
    :param u1: n1*m-d vector
    :param log_sigma1: n1*m-d vector
    :param u2: n2*m-d vector
    :param log_sigma2: n2*m-d vector
    :return: n1*n2 KL divergency distance matrix,[i,j] means KL[i,j]
    """
    if GPU_flag:
        u1 = torch.from_numpy(u1).cuda()
        log_sigma1 = torch.from_numpy(log_sigma1).cuda()
        sigma_sq1 = torch.exp(log_sigma1) ** 2
        u2 = torch.from_numpy(u2).cuda()
        log_sigma2 = torch.from_numpy(log_sigma2).cuda()
        sigma_sq2 = torch.exp(log_sigma2) ** 2
        _, d = sigma_sq1.size()
        s_pairwise = torch.unsqueeze(sigma_sq1, 1) / torch.unsqueeze(sigma_sq2, 0)
        u_pairwise = (torch.unsqueeze(u1, 1) - torch.unsqueeze(u2, 0)) ** 2 / torch.unsqueeze(sigma_sq2, 0)
        kl = (1 / 2) * (-1 * torch.sum(torch.log(s_pairwise), dim=2) +
                        torch.sum(s_pairwise, dim=2) + torch.sum(u_pairwise, dim=2) - d)
    else:
        sigma_sq1 = np.exp(log_sigma1) ** 2
        sigma_sq2 = np.exp(log_sigma2) ** 2
        n1, d = sigma_sq1.shape
        n2, d = sigma_sq2.shape
        s_pairwise = sigma_sq1.reshape(n1, 1, d) / sigma_sq2.reshape(1, n2, d)
        u_pairwise = (u1.reshape(n1, 1, d) - u2.reshape(1, n2, d)) ** 2 / sigma_sq2.reshape(1, n2, d)
        kl = (1 / 2) * (-1 * np.sum(np.log(s_pairwise), axis=2) + np.sum(s_pairwise, axis=2) +
                        np.sum(u_pairwise, axis=2) - d)
    return kl


def pairwise_norm_kl_dist_gpu(u1, log_sigma1, u2, log_sigma2):
    """
    :param u1: torch tensor with n1*k-dim
    :param log_sigma1: torch tensor with n1*k-dim
    :param u2: torch tensor with n2*k-dim
    :param log_sigma2: torch tensor with n2*k-dim
    :return: n1*n2 kl dist , [i,j] = kl[(u1[i,:],log_sigma1[i,:]||u2[j,:],(log_sigma[j,:])]
    """
    d = log_sigma1.size(1)
    s_pairwise = torch.unsqueeze(torch.exp(log_sigma1) ** 2, 1) / torch.unsqueeze(torch.exp(log_sigma2) ** 2, 0)
    u_pairwise = (torch.unsqueeze(u1, 1) - torch.unsqueeze(u2, 0)) ** 2 / torch.unsqueeze(torch.exp(log_sigma2) ** 2, 0)
    kl = (1 / 2) * (-1 * torch.sum(torch.log(s_pairwise), dim=2) +
                    torch.sum(s_pairwise, dim=2) + torch.sum(u_pairwise, dim=2) - d)
    return kl


def pairwise_square_euclidean_gpu(v1, v2):
    """
    :param v1: n1*m size vector
    :param v2: n2*m size vector
    :return: n1*n2 matrix, [i,j] represents euclidean distance (v1[i,:],v2[j,:])
    """
    euclidean_dist = torch.sum((v1.unsqueeze(1) - v2.unsqueeze(0)) ** 2, dim=2)
    return euclidean_dist


def pairwise_norm_wasserstein_dist_gpu(u1, log_sigma1, u2, log_sigma2):
    """
    :param u1: n1*m size vector
    :param log_sigma1: n1*m size vector
    :param u2: n2*m size vector
    :param log_sigma2: n2*m size vector
    :return: n1*n2 matrix, [i,j] represents the wassterstein disctance for (u1[i:,],sigma1[i,:]), (u2[j,:],sigma2[j,:])
    """
    wd = pairwise_square_euclidean_gpu(u1, u2) + pairwise_square_euclidean_gpu(torch.exp(log_sigma1),
                                                                               torch.exp(log_sigma2))
    return wd


def calculate_mean_dist_pairwise(u1, u2, GPU_flag=False, distance="euclidean"):
    n1, d = u1.shape
    n2, d = u2.shape
    if GPU_flag:
        if type(u1) == np.ndarray:
            u1 = torch.from_numpy(u1).float()
        if type(u2) == np.ndarray:
            u2 = torch.from_numpy(u2).float()
        u1 = u1.cuda()
        u2 = u2.cuda()
        if distance == "euclidean":
            dist = torch.sum((u1.unsqueeze(1) - u2.unsqueeze(0)) ** 2, dim=2)
        elif distance == "cosine":
            u1_norm = torch.sum(u1 ** 2, dim=1)
            u2_norm = torch.sum(u2 ** 2, dim=1)
            dist = torch.mm(u1, u2.t()) / (u1_norm.view(-1, 1) * u2_norm.view(1, -1))
        else:
            raise NotImplementedError("distance {} not implemented".format(distance))
    else:
        if distance == "euclidean":
            dist = np.sum((u1.reshape(n1, 1, d) - u2.reshape(1, n2, d)) ** 2, axis=2)
        elif distance == "cosine":
            u1_norm = np.sum(u1 ** 2, axis=1)
            u2_norm = np.sum(u2 ** 2, axis=1)
            dist = np.dot(u1, u2.T) / (u1_norm.reshape(-1, 1) * u2_norm.reshape(1, -1))
        else:
            raise NotImplementedError("distance {} not implemented".format(distance))
    return dist
