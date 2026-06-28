import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from collections import Counter

import config

imagenetMean = [0.485, 0.456, 0.406]
imagenetStd = [0.229, 0.224, 0.225]


class TransformSubset(Dataset):
    """Wraps a random_split subset so train/val get different transforms from the same dataset."""

    def __init__(self, baseDataset, indices, transform):
        self.samples = baseDataset.samples
        self.indices = indices
        self.transform = transform
        self.classes = baseDataset.classes
        self.classToIdx = baseDataset.classToIdx

    def __getitem__(self, idx):
        path, label = self.samples[self.indices[idx]]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.indices)


def getTransforms():
    trainTransform = transforms.Compose([
        transforms.RandomResizedCrop(config.imgSize, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(imagenetMean, imagenetStd),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

    valTransform = transforms.Compose([
        transforms.Resize(int(config.imgSize * 1.1)),
        transforms.CenterCrop(config.imgSize),
        transforms.ToTensor(),
        transforms.Normalize(imagenetMean, imagenetStd),
    ])

    return trainTransform, valTransform


def filterDataset(dataset, minImages):
    counts = Counter(cls for _, cls in dataset.samples)
    keepClasses = sorted(cls for cls, count in counts.items() if count >= minImages)
    classMap = {old: new for new, old in enumerate(keepClasses)}
    dataset.samples = [(p, classMap[c]) for p, c in dataset.samples if c in classMap]
    dataset.classes = [dataset.classes[i] for i in keepClasses]
    dataset.classToIdx = {name: i for i, name in enumerate(dataset.classes)}
    return dataset


def getDataloaders():
    trainTransform, valTransform = getTransforms()

    fullDataset = datasets.ImageFolder(root=config.dataDir, transform=None)
    fullDataset = filterDataset(fullDataset, config.minImages)
    classToIdx = fullDataset.classToIdx

    total = len(fullDataset)
    trainSize = int(config.trainSplit * total)
    valSize = total - trainSize

    indices = torch.randperm(total).tolist()
    trainIndices = indices[:trainSize]
    valIndices = indices[trainSize:]

    trainDataset = TransformSubset(fullDataset, trainIndices, trainTransform)
    valDataset = TransformSubset(fullDataset, valIndices, valTransform)

    trainLoader = DataLoader(
        trainDataset,
        batch_size=config.batchSize,
        shuffle=True,
        num_workers=config.numWorkers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
    )
    valLoader = DataLoader(
        valDataset,
        batch_size=config.batchSize,
        shuffle=False,
        num_workers=config.numWorkers,
        pin_memory=True,
        persistent_workers=True,
    )

    return trainLoader, valLoader, classToIdx


def getFullDataloader():
    _, valTransform = getTransforms()
    fullDataset = datasets.ImageFolder(root=config.dataDir, transform=None)
    fullDataset = filterDataset(fullDataset, config.minImages)
    loader = DataLoader(
        fullDataset,
        batch_size=config.batchSize,
        shuffle=False,
        num_workers=config.numWorkers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    return loader, fullDataset.classToIdx
