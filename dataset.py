import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from collections import Counter

import config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(config.IMG_SIZE * 1.1)),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, val_transform


def _filter_dataset(dataset, min_images):
    counts = Counter(cls for _, cls in dataset.samples)
    keep_classes = sorted(cls for cls, count in counts.items() if count >= min_images)
    class_map = {old: new for new, old in enumerate(keep_classes)}
    dataset.samples = [(p, class_map[c]) for p, c in dataset.samples if c in class_map]
    dataset.classes = [dataset.classes[i] for i in keep_classes]
    dataset.class_to_idx = {name: i for i, name in enumerate(dataset.classes)}
    return dataset


def get_dataloaders():
    train_transform, val_transform = get_transforms()

    full_dataset = datasets.ImageFolder(root=config.DATA_DIR, transform=train_transform)
    full_dataset = _filter_dataset(full_dataset, config.MIN_IMAGES)
    class_to_idx = full_dataset.class_to_idx

    total = len(full_dataset)
    train_size = int(config.TRAIN_SPLIT * total)
    val_size = total - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_full = datasets.ImageFolder(root=config.DATA_DIR, transform=val_transform)
    val_full = _filter_dataset(val_full, config.MIN_IMAGES)
    val_dataset.dataset = val_full

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, class_to_idx


def get_full_dataloader():
    _, val_transform = get_transforms()
    full_dataset = datasets.ImageFolder(root=config.DATA_DIR, transform=val_transform)
    full_dataset = _filter_dataset(full_dataset, config.MIN_IMAGES)
    loader = DataLoader(
        full_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return loader, full_dataset.class_to_idx
