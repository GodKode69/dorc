import sys
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

import config
from model import buildModel

sampleSize = 50
testExtensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def loadModel(device):
    checkpoint = torch.load(config.modelSavePath, map_location=device, weights_only=False)
    classToIdx = checkpoint["class_to_idx"]
    idxToClass = {v: k for k, v in classToIdx.items()}
    stateDict = checkpoint["model_state_dict"]
    stateDict = {k.replace("_orig_mod.", ""): v for k, v in stateDict.items()}
    model = buildModel(numClasses=len(classToIdx)).to(device)
    model.load_state_dict(stateDict)
    model.eval()
    return model, classToIdx, idxToClass


def getTransform():
    return transforms.Compose([
        transforms.Resize((config.imgSize, config.imgSize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def getClassCounts():
    counts = {}
    dataDir = Path(config.dataDir)
    for clsDir in sorted(dataDir.iterdir()):
        if clsDir.is_dir():
            n = len([f for f in clsDir.iterdir() if f.suffix.lower() in testExtensions])
            if n >= config.minImages:
                counts[clsDir.name] = n
    return counts


def testClass(model, className, idxToClass, transform, device, sampleSize, useBf16):
    classDir = Path(config.dataDir) / className
    images = [f for f in classDir.iterdir() if f.suffix.lower() in testExtensions]
    if not images:
        return []

    sample = random.sample(images, min(sampleSize, len(images)))
    correct = 0
    results = []
    for imgPath in sample:
        image = Image.open(imgPath).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device, non_blocking=True)
        if tensor.dim() == 4:
            tensor = tensor.to(memory_format=torch.channels_last)
        with torch.inference_mode():
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=useBf16):
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = probs.max(1)
        predClass = idxToClass[predicted.item()]
        isCorrect = predClass == className
        if isCorrect:
            correct += 1
        results.append((imgPath.name, predClass, confidence.item() * 100, isCorrect))
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    useBf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    model, classToIdx, idxToClass = loadModel(device)
    transform = getTransform()
    counts = getClassCounts()

    sortedClasses = sorted(counts.items(), key=lambda x: x[1])
    highest = sortedClasses[-1]
    lowest = sortedClasses[0]
    midIdx = len(sortedClasses) // 2
    middle = sortedClasses[midIdx]

    testClasses = [
        (highest[0], highest[1], f"Highest ({highest[1]} imgs)"),
        (middle[0], middle[1], f"Middle ({middle[1]} imgs)"),
        (lowest[0], lowest[1], f"Lowest ({lowest[1]} imgs)"),
    ]

    totalCorrect = 0
    totalTested = 0

    for clsName, clsCount, label in testClasses:
        print(f"\n{'='*60}")
        print(f"Testing: {clsName} — {label}")
        print(f"{'='*60}")

        results = testClass(model, clsName, idxToClass, transform, device, sampleSize, useBf16)
        correct = sum(1 for _, _, _, c in results if c)
        totalCorrect += correct
        totalTested += len(results)

        for fname, pred, conf, isCorrect in results:
            mark = "OK" if isCorrect else "WRONG"
            print(f"  [{mark:5s}] {fname:30s} -> {pred:20s} ({conf:5.1f}%)")

        acc = 100 * correct / len(results) if results else 0
        print(f"\n  Result: {correct}/{len(results)} correct ({acc:.1f}%)")

    overallAcc = 100 * totalCorrect / totalTested if totalTested else 0
    print(f"\n{'='*60}")
    print(f"OVERALL: {totalCorrect}/{totalTested} correct ({overallAcc:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
