import torch.nn as nn
from torchvision import models

import config


def build_model(num_classes=None):
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.features[-3:].parameters():
        param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    return model


def load_backbone_weights(model, checkpoint_path):
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    old_state = checkpoint["model_state_dict"]

    backbone_state = {}
    skipped = []
    for k, v in old_state.items():
        if k.startswith("classifier"):
            skipped.append(k)
            continue
        if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
            backbone_state[k] = v

    model.load_state_dict(backbone_state, strict=False)
    print(f"Loaded {len(backbone_state)} backbone layers from {checkpoint_path}")
    print(f"Skipped {len(skipped)} classifier layers (output size changed)")
    return model
