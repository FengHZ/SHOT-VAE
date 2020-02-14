from smooth_vae_model.mnist_vae import mnist_VAE
from lib.dataloader_one_stage_vae import get_mnist_dataloaders

import argparse
import os
import numpy as np

import torch
from torch import optim
from torch.nn import functional as F
from torch import nn

EPS = 1e-12

parser = argparse.ArgumentParser(description='Pytorch Training Semi-Supervised one-stage VAE for MNIST Dataset')
parser.add_argument('-bp', '--base_path', default=".")
parser.add_argument('--latent-spec', default={'cont': 10, 'disc': [10]}, type=set, help='vector length for latent variables')
parser.add_argument('--disc-capacity', default=[0.0, 17.0, 25000, 30], type=list, help='(min_capacity, max_capacity, num_iters, gamma_c)')
parser.add_argument('--cont-capacity', default=[0.0, 17.5, 25000, 30], type=list, help='(min_capacity, max_capacity, num_iters, gamma_z)')
parser.add_argument('--learning-rate', default=5e-4, type=float, help='learning rate')
parser.add_argument('--alpha', default=50, type=float)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--size-labeled-data', default=100, type=int)
parser.add_argument('--labeled-batch-size', default=4, type=int)
parser.add_argument('--unlabeled-batch-size', default=128, type=int)
parser.add_argument('--test-batch-size', default=1000, type=int)
parser.add_argument('--path-to-data', type=str, help='path to raw data') ###################################################
parser.add_argument('--gpu', type=str)
parser.add_argument('--train-time', default=1, type=int, help='the x-th time of training')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.path_to_data = os.path.join(args.base_path, "dataset", "mnist")
print(args)

class Trainer():
    def __init__(self, model, optimizer, cont_capacity=None,
                 disc_capacity=None, print_loss_every=50, record_loss_every=5,
                 use_cuda=False, alpha=1):
        """
        Class to handle training of model.

        Parameters
        ----------
        model : jointvae.models.VAE instance

        optimizer : torch.optim.Optimizer instance

        cont_capacity : tuple (float, float, int, float) or None
            Tuple containing (min_capacity, max_capacity, num_iters, gamma_z).
            Parameters to control the capacity of the continuous latent
            channels. Cannot be None if model.is_continuous is True.

        disc_capacity : tuple (float, float, int, float) or None
            Tuple containing (min_capacity, max_capacity, num_iters, gamma_c).
            Parameters to control the capacity of the discrete latent channels.
            Cannot be None if model.is_discrete is True.

        print_loss_every : int
            Frequency with which loss is printed during training.

        record_loss_every : int
            Frequency with which loss is recorded during training.

        use_cuda : bool
            If True moves model and training to GPU.
        """
        self.model = model
        self.optimizer = optimizer
        self.cont_capacity = cont_capacity
        self.disc_capacity = disc_capacity
        self.print_loss_every = print_loss_every
        self.record_loss_every = record_loss_every
        self.use_cuda = use_cuda
        self.alpha = alpha

        if self.model.is_continuous and self.cont_capacity is None:
            raise RuntimeError("Model is continuous but cont_capacity not provided.")

        if self.model.is_discrete and self.disc_capacity is None:
            raise RuntimeError("Model is discrete but disc_capacity not provided.")

        if self.use_cuda:
            self.model.cuda()

        # Initialize attributes
        self.num_steps = 0
        self.losses = {'loss': [],
                       'recon_loss': [],
                       'kl_loss': []}

        # Keep track of divergence values for each latent variable
        if self.model.is_continuous:
            self.losses['kl_loss_cont'] = []
            # For every dimension of continuous latent variables
            for i in range(self.model.latent_spec['cont']):
                self.losses['kl_loss_cont_' + str(i)] = []

        if self.model.is_discrete:
            self.losses['kl_loss_disc'] = []
            # For every discrete latent variable
            for i in range(len(self.model.latent_spec['disc'])):
                self.losses['kl_loss_disc_' + str(i)] = []


    def train(self, data_loaders, log_path, epochs=10, save_training_gif=None):

        f = open(log_path, 'w')

        self.labeled_batch_size = data_loaders[0].batch_size
        self.unlabeled_batch_size = data_loaders[1].batch_size
        self.model.train()

        for epoch in range(epochs):
            mean_epoch_loss, test_ac, u_split, l_split = self._train_epoch(data_loaders)

            tmp = 'Epoch: {} Average loss: {:.2f} Test Accuracy: {}\n'.format(epoch, mean_epoch_loss, test_ac)
            tmp += 'u_recon_loss: {:.2f}, u_cont: {:.2f}, u_disc: {:.2f}\n'.format(u_split[0],u_split[1],u_split[2])
            tmp += 'l_recon_loss: {:.2f}, l_cont: {:.2f}, l_disc: {:.2f}, class: {:.2f}\n'.format(l_split[0],l_split[1],l_split[2],l_split[3])
            print(tmp)
            f.write(tmp+'\n')

        f.close()

    def _train_epoch(self, data_loaders):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
        """
        labeled_loader, unlabeled_loader, test_loader = data_loaders

        epoch_loss = 0.
        print_every_loss = 0.  # Keeps track of loss to print every
                               # self.print_loss_every
        epoch_u_recon_loss, epoch_u_cont_capacity_loss, epoch_u_disc_capacity_loss = 0,0,0
        epoch_l_recon_loss, epoch_l_cont_capacity_loss, epoch_l_disc_capacity_loss, epoch_classification_loss = 0,0,0,0

        for batch_idx, (unlabeled_data, _) in enumerate(unlabeled_loader.get_iter()):

            self.num_steps += 1

            unlabeled_data = torch.stack(unlabeled_data).cuda()
            labeled_data, label = labeled_loader.next()
            labeled_data = torch.stack(labeled_data).cuda()
            label = torch.tensor(label).cuda()


            self.optimizer.zero_grad()
            unlabeled_recon_batch, unlabeled_latent_dist, _, _ = self.model(unlabeled_data)
            unlabeled_loss, unlabeled_loss_split = self._loss_function(data=unlabeled_data,
                                                 recon_data=unlabeled_recon_batch,
                                                 latent_dist=unlabeled_latent_dist)


            labeled_recon_batch, labeled_latent_dist ,_ ,labeled_disc_sample = self.model(labeled_data, label)
            labeled_loss, labeled_loss_split = self._loss_function(data=labeled_data,
                                               recon_data=labeled_recon_batch,
                                               latent_dist=labeled_latent_dist,
                                               disc_sample=labeled_disc_sample,
                                               label=label)


            loss = unlabeled_loss + labeled_loss


            loss.backward()
            self.optimizer.step()

            train_loss = loss.item()

            u_recon_loss, u_cont_capacity_loss, u_disc_capacity_loss = \
                unlabeled_loss_split[0].item(), unlabeled_loss_split[1].item(),unlabeled_loss_split[2].item()
            l_recon_loss, l_cont_capacity_loss, l_disc_capacity_loss, classification_loss = \
                labeled_loss_split[0].item(), labeled_loss_split[1].item(),labeled_loss_split[2].item(), labeled_loss_split[3].item()


            epoch_loss += train_loss

            epoch_u_recon_loss += u_recon_loss
            epoch_u_cont_capacity_loss += u_cont_capacity_loss
            epoch_u_disc_capacity_loss += u_disc_capacity_loss

            epoch_l_recon_loss += l_recon_loss
            epoch_l_cont_capacity_loss += l_cont_capacity_loss
            epoch_l_disc_capacity_loss += l_disc_capacity_loss
            epoch_classification_loss += classification_loss


            print_every_loss += train_loss
            # Print loss info every self.print_loss_every iteration
            if batch_idx % self.print_loss_every == 0:
                if batch_idx == 0:
                    mean_loss = print_every_loss
                else:
                    mean_loss = print_every_loss / self.print_loss_every
                print('{}/{}\tLoss: {:.3f}'.format(batch_idx * len(unlabeled_data),
                                                  len(unlabeled_loader),
                                                  mean_loss))
                print_every_loss = 0.


        # Return mean epoch loss
        return epoch_loss/batch_idx, self.eval(test_loader), \
               (epoch_u_recon_loss/batch_idx, epoch_u_cont_capacity_loss/batch_idx, epoch_u_disc_capacity_loss/batch_idx), \
               (epoch_l_recon_loss/batch_idx, epoch_l_cont_capacity_loss/batch_idx, epoch_l_disc_capacity_loss/batch_idx, epoch_classification_loss/batch_idx)

        # return epoch_loss / batch_idx, 0


    def eval(self, test_loader):
        count = 0
        for batch_idx, (test_data, test_label) in enumerate(test_loader.get_iter()) :
            test_data = torch.stack(test_data).cuda()
            # test_data = test_data.cuda()
            test_label = torch.tensor(test_label)
            _, test_latent_dist, _, _ = self.model(test_data)
            pre_label = torch.max(test_latent_dist['disc'][0], 1)[1].cpu().numpy()
            test_label = test_label.cpu().numpy()
            count += np.sum(pre_label==test_label)
            # print(a.shape)
            # print(test_label.shape)
        return count/len(test_loader)

    def _loss_function(self, data, recon_data, latent_dist, label=None, disc_sample=None):
        """
        Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Should have shape (N, C, H, W)

        recon_data : torch.Tensor
            Reconstructed data. Should have shape (N, C, H, W)

        latent_dist : dict
            Dict with keys 'cont' or 'disc' or both containing the parameters
            of the latent distributions as values.
        """
        # Reconstruction loss is pixel wise cross-entropy

        recon_loss = F.mse_loss(recon_data.view(-1, self.model.num_pixels),data.view(-1, self.model.num_pixels))

        # F.binary_cross_entropy takes mean over pixels, so unnormalise this
        recon_loss *= self.model.num_pixels

        # Calculate KL divergences
        kl_cont_loss = 0  # Used to compute capacity loss (but not a loss in itself)
        kl_disc_loss = 0  # Used to compute capacity loss (but not a loss in itself)
        cont_capacity_loss = 0
        disc_capacity_loss = 0
        classfication_loss = 0

        if self.model.is_continuous:
            # Calculate KL divergence
            mean, logvar = latent_dist['cont']
            kl_cont_loss = self._kl_normal_loss(mean, logvar)
            # Linearly increase capacity of continuous channels
            cont_min, cont_max, cont_num_iters, cont_gamma = self.cont_capacity
            # Increase continuous capacity without exceeding cont_max
            cont_cap_current = (cont_max - cont_min) * self.num_steps / float(cont_num_iters) + cont_min
            cont_cap_current = min(cont_cap_current, cont_max)
            # Calculate continuous capacity loss
            cont_capacity_loss = cont_gamma * torch.abs(cont_cap_current - kl_cont_loss)

        if self.model.is_discrete:
            # Calculate KL divergence
            kl_disc_loss = self._kl_multiple_discrete_loss(latent_dist['disc'], label)
            # Linearly increase capacity of discrete channels
            disc_min, disc_max, disc_num_iters, disc_gamma = self.disc_capacity
            # Increase discrete capacity without exceeding disc_max or theoretical
            # maximum (i.e. sum of log of dimension of each discrete variable)
            disc_cap_current = (disc_max - disc_min) * self.num_steps / float(disc_num_iters) + disc_min
            disc_cap_current = min(disc_cap_current, disc_max)
            # Require float conversion here to not end up with numpy float
            disc_theoretical_max = sum([float(np.log(disc_dim)) for disc_dim in self.model.latent_spec['disc']])
            disc_cap_current = min(disc_cap_current, disc_theoretical_max)
            # Calculate discrete capacity loss
            disc_capacity_loss = disc_gamma * torch.abs(disc_cap_current - kl_disc_loss)

        # Calculate total kl value to record it
        kl_loss = kl_cont_loss + kl_disc_loss


        if label is not None :
            one_hot = torch.Tensor(np.eye(10)[label.cpu()]).cuda()
            classfication_loss = self.alpha * F.binary_cross_entropy(latent_dist['disc'][0], one_hot)

        total_loss = recon_loss + cont_capacity_loss + disc_capacity_loss + classfication_loss


        # Record losses
        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['recon_loss'].append(recon_loss.item())
            self.losses['kl_loss'].append(kl_loss.item())
            self.losses['loss'].append(total_loss.item())

        return total_loss, (recon_loss ,cont_capacity_loss ,disc_capacity_loss, classfication_loss)

    def _kl_normal_loss(self, mean, logvar):
        """
        Calculates the KL divergence between a normal distribution with
        diagonal covariance and a unit normal distribution.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        """
        # Calculate KL divergence
        kl_values = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        # Mean KL divergence across batch for each latent variable
        kl_means = torch.mean(kl_values, dim=0)
        # KL loss is sum of mean KL of each latent variable
        kl_loss = torch.sum(kl_means)

        # Record losses
        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['kl_loss_cont'].append(kl_loss.item())
            for i in range(self.model.latent_spec['cont']):
                self.losses['kl_loss_cont_' + str(i)].append(kl_means[i].item())

        return kl_loss

    def _kl_multiple_discrete_loss(self, alphas, label=None):
        """
        Calculates the KL divergence between a set of categorical distributions
        and a set of uniform categorical distributions.

        Parameters
        ----------
        alphas : list
            List of the alpha parameters of a categorical (or gumbel-softmax)
            distribution. For example, if the categorical atent distribution of
            the model has dimensions [2, 5, 10] then alphas will contain 3
            torch.Tensor instances with the parameters for each of
            the distributions. Each of these will have shape (N, D).
        """
        # Calculate kl losses for each discrete latent

        kl_losses = [self._kl_discrete_loss(alpha, label) for alpha in alphas]

        # Total loss is sum of kl loss for each discrete latent
        kl_loss = torch.sum(torch.cat(kl_losses))

        # Record losses
        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['kl_loss_disc'].append(kl_loss.item())
            for i in range(len(alphas)):
                self.losses['kl_loss_disc_' + str(i)].append(kl_losses[i].item())

        return kl_loss

    def _kl_discrete_loss(self, alpha, label=None):
        """
        Calculates the KL divergence between a categorical distribution and a
        uniform categorical distribution.

        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the categorical or gumbel-softmax distribution.
            Shape (N, D)
        """

        disc_dim = int(alpha.size()[-1])
        log_dim = torch.Tensor([np.log(disc_dim)])
        if self.use_cuda:
            log_dim = log_dim.cuda()
        # Calculate negative entropy of each row
        neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
        # Take mean of negative entropy across batch
        mean_neg_entropy = torch.mean(neg_entropy, dim=0)
        # KL loss of alpha with uniform categorical variable
        kl_loss = log_dim + mean_neg_entropy


        return kl_loss

#############################################################

if __name__ == '__main__':

    labeled_batch_size = args.labeled_batch_size
    unlabeled_batch_size = args.unlabeled_batch_size
    size_labeled_data = args.size_labeled_data
    lr = args.learning_rate
    epochs = args.epochs
    latent_spec = args.latent_spec
    cont_capacity = args.cont_capacity
    disc_capacity = args.disc_capacity
    save_dir = os.path.join(args.base_path,"MNIST-One-Stage-VAE")
    dataset = 'mnist'
    alpha = args.alpha
    img_size = (1, 32, 32)
    train_time = args.train_time

    use_cuda = torch.cuda.is_available()
    setattr(args, 'use_cuda', use_cuda)

    # Save trained model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_name = 'MNIST-One-Stage-VAE.txt'
    model_name = 'MNIST-One-Stage-VAE.pt'

    # Load data
    labeled_loader, unlabeled_loader, test_loader = get_mnist_dataloaders(args)

    # Define latent spec and model
    model = mnist_VAE(img_size=img_size, latent_spec=latent_spec, use_cuda=use_cuda)
    if use_cuda:
        model.cuda()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Define trainer
    trainer = Trainer(model, optimizer,
                      cont_capacity=cont_capacity,
                      disc_capacity=disc_capacity,
                      use_cuda=use_cuda,
                      alpha=alpha)


    trainer.train([labeled_loader, unlabeled_loader, test_loader], os.path.join(save_dir, log_name), epochs)
    torch.save(trainer.model.state_dict(), os.path.join(save_dir, model_name))

    