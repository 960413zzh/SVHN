import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from dataset.randaugment import RandAugmentMC

logger = logging.getLogger(__name__)


def get_cifar10(root):
    transform_augment = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=False)

    train_labeled_idxs = x_u_split(base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_augment)

    # train_unlabeled_dataset = CIFAR10SSL(
    #     root, train_unlabeled_idxs, train=True,
    #     transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform, download=False)

    return train_labeled_dataset, test_dataset


def get_SVHN(root):
    transform_augment = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    base_dataset = datasets.SVHN(root, split='train', download=False)

    train_labeled_idxs = x_u_split(base_dataset.labels)

    train_labeled_dataset = SVHNSSL(
        root, train_labeled_idxs,
        transform=transform_augment)

    # train_unlabeled_dataset = CIFAR10SSL(
    #     root, train_unlabeled_idxs, train=True,
    #     transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.SVHN(
        root, split='test', transform=transform, download=False)

    return train_labeled_dataset, test_dataset


def x_u_split(labels):
    label_per_class = 7360 // 10
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    # unlabeled_idx = np.array(range(len(labels)))
    for i in range(10):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == 7360

    # if args.expand_labels or args.num_labeled < args.batch_size:
    #     num_expand_x = math.ceil(
    #         args.batch_size * args.eval_step / args.num_labeled)
    #     labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class SVHNSSL(datasets.SVHN):
    def __init__(self, root, indexs,
                 transform=None, target_transform=None,
                 download=True):
        super().__init__(root,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]

    def __getitem__(self, index):
        img, labels = self.data[index], self.labels[index]
        img = img.transpose(1, 2, 0)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return img, labels





DATASET_GETTERS = {'cifar10': get_cifar10,
                   'SVHN': get_SVHN}
