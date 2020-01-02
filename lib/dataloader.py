import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, datasets


def mnist_dataset(dataset_base_path, train_flag=True):
    transform = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomCrop(32),
        transforms.ToTensor()
    ])
    if train_flag:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transform
        ])
    if train_flag:
        dataset = datasets.MNIST(root=dataset_base_path, train=True, transform=transform, download=True)
    else:
        dataset = datasets.MNIST(root=dataset_base_path, train=False, transform=transform, download=True)
    return dataset


def svhn_dataset(dataset_base_path, train_flag=True):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    if train_flag:
        transform = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transform
        ])
    if train_flag:
        dataset = datasets.SVHN(root=dataset_base_path, split='train', transform=transform, download=True)
    else:
        dataset = datasets.SVHN(root=dataset_base_path, split='test', transform=transform, download=True)
    return dataset


def cifar100_dataset(dataset_base_path, train_flag=True):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    if train_flag:
        transform = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transform
        ])
    dataset = datasets.CIFAR100(root=dataset_base_path, train=train_flag,
                                download=True, transform=transform)
    return dataset


def cifar10_dataset(dataset_base_path, train_flag=True):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    if train_flag:
        transform = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transform
        ])
    dataset = datasets.CIFAR10(root=dataset_base_path, train=train_flag, download=True, transform=transform)
    return dataset


def get_cifar10_sl_sampler(labels, valid_num_per_class, num_classes):
    """
    :param labels: torch.array(int tensor)
    :param valid_num_per_class: the number of validation for each class
    :param num_classes: the total number of classes
    :return: sampler_l,sampler_u
    """
    sampler_valid = []
    sampler_train = []
    for i in range(num_classes):
        loc = torch.nonzero(labels == i)
        loc = loc.view(-1)
        # do random perm to make sure uniform sample
        loc = loc[torch.randperm(loc.size(0))]
        sampler_valid.extend(loc[:valid_num_per_class].tolist())
        sampler_train.extend(loc[valid_num_per_class:].tolist())
    sampler_valid = SubsetRandomSampler(sampler_valid)
    sampler_train = SubsetRandomSampler(sampler_train)
    return sampler_valid, sampler_train


def get_cifar100_sl_sampler(labels, valid_num_per_class, num_classes=100):
    """
    :param labels: torch.array(int tensor)
    :param valid_num_per_class: the number of validation for each class
    :param num_classes: the total number of classes
    :return: sampler_l,sampler_u
    """
    sampler_valid = []
    sampler_train = []
    for i in range(num_classes):
        loc = torch.nonzero(labels == i)
        loc = loc.view(-1)
        # do random perm to make sure uniform sample
        loc = loc[torch.randperm(loc.size(0))]
        sampler_valid.extend(loc[:valid_num_per_class].tolist())
        sampler_train.extend(loc[valid_num_per_class:].tolist())
    sampler_valid = SubsetRandomSampler(sampler_valid)
    sampler_train = SubsetRandomSampler(sampler_train)
    return sampler_valid, sampler_train


def get_ssl_sampler(labels, valid_num_per_class, annotated_num_per_class, num_classes):
    """
    :param labels: torch.array(int tensor)
    :param valid_num_per_class: the number of validation for each class
    :param annotated_num_per_class: the number of annotation we use for each classes
    :param num_classes: the total number of classes
    :return: sampler_l,sampler_u
    """
    sampler_valid = []
    sampler_train_l = []
    sampler_train_u = []
    for i in range(num_classes):
        loc = torch.nonzero(labels == i)
        loc = loc.view(-1)
        # do random perm to make sure uniform sample
        loc = loc[torch.randperm(loc.size(0))]
        sampler_valid.extend(loc[:valid_num_per_class].tolist())
        sampler_train_l.extend(loc[valid_num_per_class:valid_num_per_class + annotated_num_per_class].tolist())
        # sampler_train_u.extend(loc[num_valid + annotated_num_per_class:].tolist())
        # here the unsampled part also include the train_l part
        sampler_train_u.extend(loc[valid_num_per_class:].tolist())
    sampler_valid = SubsetRandomSampler(sampler_valid)
    sampler_train_l = SubsetRandomSampler(sampler_train_l)
    sampler_train_u = SubsetRandomSampler(sampler_train_u)
    return sampler_valid, sampler_train_l, sampler_train_u


def get_cifar10_ssl_sampler(labels, valid_num_per_class, annotated_num_per_class, num_classes):
    """
    :param labels: torch.array(int tensor)
    :param valid_num_per_class: the number of validation for each class
    :param annotated_num_per_class: the number of annotation we use for each classes
    :param num_classes: the total number of classes
    :return: sampler_l,sampler_u
    """
    sampler_valid = []
    sampler_train_l = []
    sampler_train_u = []
    for i in range(num_classes):
        loc = torch.nonzero(labels == i)
        loc = loc.view(-1)
        # do random perm to make sure uniform sample
        loc = loc[torch.randperm(loc.size(0))]
        sampler_valid.extend(loc[:valid_num_per_class].tolist())
        sampler_train_l.extend(loc[valid_num_per_class:valid_num_per_class + annotated_num_per_class].tolist())
        # sampler_train_u.extend(loc[num_valid + annotated_num_per_class:].tolist())
        # here the unsampled part also include the train_l part
        sampler_train_u.extend(loc[valid_num_per_class:].tolist())
    sampler_valid = SubsetRandomSampler(sampler_valid)
    sampler_train_l = SubsetRandomSampler(sampler_train_l)
    sampler_train_u = SubsetRandomSampler(sampler_train_u)
    return sampler_valid, sampler_train_l, sampler_train_u


def get_cifar100_ssl_sampler(labels, valid_num_per_class, annotated_num_per_class, num_classes=100):
    """
    :param labels: torch.array(int tensor)
    :param valid_num_per_class: the number of validation for each class
    :param annotated_num_per_class: the number of annotation we use for each classes
    :param num_classes: the total number of classes
    :return: sampler_l,sampler_u
    """
    sampler_valid = []
    sampler_train_l = []
    sampler_train_u = []
    for i in range(num_classes):
        loc = torch.nonzero(labels == i)
        loc = loc.view(-1)
        # do random perm to make sure uniform sample
        loc = loc[torch.randperm(loc.size(0))]
        sampler_valid.extend(loc[:valid_num_per_class].tolist())
        sampler_train_l.extend(loc[valid_num_per_class:valid_num_per_class + annotated_num_per_class].tolist())
        # sampler_train_u.extend(loc[num_valid + annotated_num_per_class:].tolist())
        # here the unsampled part also include the train_l part
        sampler_train_u.extend(loc[valid_num_per_class:].tolist())
    sampler_valid = SubsetRandomSampler(sampler_valid)
    sampler_train_l = SubsetRandomSampler(sampler_train_l)
    sampler_train_u = SubsetRandomSampler(sampler_train_u)
    return sampler_valid, sampler_train_l, sampler_train_u


if __name__ == "__main__":
    from torch.utils.data import DataLoader
