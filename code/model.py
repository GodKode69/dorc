import torch.nn as nn
from torchvision import models

import config


def buildModel(numClasses=None):
    if numClasses is None:
        numClasses = config.numClasses
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.features[-3:].parameters():
        param.requires_grad = True

    inFeatures = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(inFeatures, numClasses),
    )

    return model


def loadBackboneWeights(model, checkpointPath):
    import torch
    checkpoint = torch.load(checkpointPath, map_location="cpu", weights_only=False)
    oldState = checkpoint["model_state_dict"]

    backboneState = {}
    skipped = []
    for k, v in oldState.items():
        if k.startswith("classifier"):
            skipped.append(k)
            continue
        if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
            backboneState[k] = v

    model.load_state_dict(backboneState, strict=False)
    print(f"Loaded {len(backboneState)} backbone layers from {checkpointPath}")
    if skipped:
        print(f"Skipped {len(skipped)} classifier layers (output size changed):")
        for k in skipped:
            print(f"  - {k}")
    return model
