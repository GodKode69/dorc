import sys
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

import config
from model import buildModel


def predict(imagePath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(config.modelSavePath, map_location=device, weights_only=False)
    classToIdx = checkpoint["class_to_idx"]
    idxToClass = {v: k for k, v in classToIdx.items()}

    stateDict = checkpoint["model_state_dict"]
    stateDict = {k.replace("_orig_mod.", ""): v for k, v in stateDict.items()}

    model = buildModel(numClasses=len(classToIdx)).to(device)
    model.load_state_dict(stateDict)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((config.imgSize, config.imgSize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = Image.open(imagePath).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)

    predictedClass = idxToClass[predicted.item()]
    confidencePct = confidence.item() * 100

    print(f"Prediction: {predictedClass} ({confidencePct:.2f}%)")
    return predictedClass, confidencePct


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <image_path>")
        sys.exit(1)

    predict(sys.argv[1])
