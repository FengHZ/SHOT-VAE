import sys

sys.path.append('..')

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class DataLoader(object):

    def __init__(self, dataset, indices, batch_size):
        self.images, self.labels = [], []
        for i in indices:
            image, label = dataset[i]
            self.images.append(image)
            self.labels.append(label)

        self.batch_size = batch_size
        self.len = len(indices)
        self.unlimited_gen = self.generator(True)

    def __len__(self):
        return self.len

    def next(self):
        return next(self.unlimited_gen)

    def get_iter(self):
        return self.generator()

    def generator(self, unlimited=False):
        while True:
            for start in range(0, self.len, self.batch_size):
                end = min(start + self.batch_size, self.len)
                ret_images, ret_labels = self.images[start:end], self.labels[start:end]
                yield ret_images, ret_labels
            if not unlimited: break


def get_svhn_dataloaders(config):
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_data = datasets.SVHN(config.path_to_data, split='train', download=True, transform=all_transforms)
    test_data = datasets.SVHN(config.path_to_data, split='test', download=True, transform=all_transforms)

    def preprocess(data_set):
        for i in range(len(data_set.data)):
            if data_set.labels[i] == 10:
                data_set.labels[i] = 0

    preprocess(train_data)
    preprocess(test_data)

    indices = np.arange(len(train_data))
    np.random.shuffle(indices)
    labels = np.array([train_data.labels[i] for i in indices])
    mask = np.zeros(indices.shape[0], dtype=np.bool)

    for i in range(10):
        mask[indices[labels == i][:int(config.size_labeled_data / 10)]] = True

    new_indices = np.arange(len(train_data))
    labeled_indices, unlabeled_indices = new_indices[mask], new_indices
    print('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size',
          len(test_data))

    labeled_loader = DataLoader(train_data, labeled_indices, config.labeled_batch_size)
    unlabeled_loader = DataLoader(train_data, unlabeled_indices, config.unlabeled_batch_size)
    test_loader = DataLoader(test_data, np.arange(len(test_data)), config.test_batch_size)

    return labeled_loader, unlabeled_loader, test_loader


def get_mnist_dataloaders(config):
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_data = datasets.MNIST(config.path_to_data, train=True, transform=all_transforms, download=True)
    test_data = datasets.MNIST(config.path_to_data, train=False, transform=all_transforms, download=True)

    indices = np.arange(len(train_data))
    np.random.shuffle(indices)
    labels = np.array([train_data[i][1] for i in indices])
    mask = np.zeros(indices.shape[0], dtype=np.bool)

    for i in range(10):
        mask[indices[labels == i][:int(config.size_labeled_data / 10)]] = True

    new_indices = np.arange(len(train_data))
    labeled_indices, unlabeled_indices = new_indices[mask], new_indices
    print('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size',
          len(test_data))

    labeled_loader = DataLoader(train_data, labeled_indices, config.labeled_batch_size)
    unlabeled_loader = DataLoader(train_data, unlabeled_indices, config.unlabeled_batch_size)
    test_loader = DataLoader(test_data, np.arange(len(test_data)), config.test_batch_size)

    return labeled_loader, unlabeled_loader, test_loader
