import csv
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import config
from model import buildModel

imageExtensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def loadClipLabels():
    inputDir = config.basePath / "input"
    clipLabels = {}

    for csvPath in sorted(inputDir.glob("*.csv")):
        with open(csvPath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clipLabels[row["path"]] = {
                    "clip_label": row["clip_label"],
                    "clip_confidence": float(row["clip_confidence"]),
                }
    return clipLabels


def loadModel(device):
    checkpoint = torch.load(config.modelSavePath, map_location=device, weights_only=False)
    classToIdx = checkpoint["class_to_idx"]
    idxToClass = {v: k for k, v in classToIdx.items()}
    stateDict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model_state_dict"].items()}

    model = buildModel(numClasses=len(classToIdx)).to(device)
    model.load_state_dict(stateDict)
    model.eval()
    return model, classToIdx, idxToClass


def verify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    useBf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"Using device: {device}")

    model, classToIdx, idxToClass = loadModel(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    clipLabels = loadClipLabels()
    print(f"Loaded {len(clipLabels)} CLIP labels from input/")

    transform = transforms.Compose([
        transforms.Resize(int(config.imgSize * 1.1)),
        transforms.CenterCrop(config.imgSize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    reportsDir = config.basePath / "reports"
    reportsDir.mkdir(exist_ok=True)
    mismatchesPath = reportsDir / "mismatches.csv"
    agreementsPath = reportsDir / "agreements.csv"

    totalImages = 0
    totalMatch = 0
    totalMismatch = 0
    totalNoClip = 0
    classStats = {}
    startTime = time.time()

    with open(mismatchesPath, "w", newline="") as mF, \
         open(agreementsPath, "w", newline="") as aF:

        mismatchWriter = csv.writer(mF)
        agreementWriter = csv.writer(aF)
        mismatchWriter.writerow(["path", "ground_truth", "model_prediction", "clip_label", "model_confidence", "clip_confidence"])
        agreementWriter.writerow(["path", "ground_truth", "model_prediction", "clip_label", "model_confidence", "clip_confidence"])

        for className in sorted(classToIdx.values()):
            classDir = config.dataDir / className
            if not classDir.exists():
                continue

            imageFiles = sorted(
                f for f in classDir.iterdir()
                if f.suffix.lower() in imageExtensions
            )
            if not imageFiles:
                continue

            classTotal = 0
            classMatch = 0
            classMismatch = 0

            for imgPath in tqdm(imageFiles, desc=f"  {className}", leave=False):
                pathStr = str(imgPath.resolve())
                clipInfo = clipLabels.get(pathStr)
                if not clipInfo:
                    totalNoClip += 1
                    continue

                try:
                    image = Image.open(imgPath).convert("RGB")
                    tensor = transform(image).unsqueeze(0).to(device, non_blocking=True)
                    if tensor.dim() == 4:
                        tensor = tensor.to(memory_format=torch.channels_last)

                    with torch.inference_mode():
                        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=useBf16):
                            outputs = model(tensor)
                            probs = torch.softmax(outputs, dim=1)
                            confidence, predicted = probs.max(1)

                    modelPred = idxToClass[predicted.item()]
                    modelConf = confidence.item()
                    clipPred = clipInfo["clip_label"]
                    clipConf = clipInfo["clip_confidence"]

                    totalImages += 1
                    classTotal += 1

                    if modelPred == clipPred:
                        totalMatch += 1
                        classMatch += 1
                        agreementWriter.writerow([pathStr, className, modelPred, clipPred, f"{modelConf:.4f}", f"{clipConf:.4f}"])
                    else:
                        totalMismatch += 1
                        classMismatch += 1
                        mismatchWriter.writerow([pathStr, className, modelPred, clipPred, f"{modelConf:.4f}", f"{clipConf:.4f}"])

                except Exception as e:
                    pass

            if classTotal > 0:
                classStats[className] = {
                    "total": classTotal,
                    "match": classMatch,
                    "mismatch": classMismatch,
                    "rate": classMatch / classTotal * 100,
                }

    totalTime = time.time() - startTime

    print(f"\nVerification complete in {int(totalTime // 60)}m {int(totalTime % 60)}s")
    print(f"Total: {totalImages} images verified")
    print(f"Agreement: {totalMatch} ({totalMatch / totalImages * 100:.1f}%)")
    print(f"Mismatch: {totalMismatch} ({totalMismatch / totalImages * 100:.1f}%)")
    if totalNoClip > 0:
        print(f"No CLIP label: {totalNoClip}")

    print(f"\nMismatches saved to {mismatchesPath}")
    print(f"Agreements saved to {agreementsPath}")

    print(f"\n--- Per-Class Stats ---")
    sortedClasses = sorted(classStats.items(), key=lambda x: x[1]["rate"])
    for name, stats in sortedClasses[:10]:
        print(f"  {name:20s}: {stats['match']}/{stats['total']} agree ({stats['rate']:.1f}%)")
    print(f"\n--- Best 5 Classes ---")
    for name, stats in sortedClasses[-5:]:
        print(f"  {name:20s}: {stats['match']}/{stats['total']} agree ({stats['rate']:.1f}%)")


if __name__ == "__main__":
    verify()
