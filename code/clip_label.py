import csv
import sys
import time
from pathlib import Path

import clip
import torch
from PIL import Image
from tqdm import tqdm

import config


def loadClipModel(device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def precomputeTextFeatures(labels, model, device):
    textTokens = clip.tokenize([f"a photo of a {label}" for label in labels]).to(device)
    with torch.inference_mode():
        textFeatures = model.encode_text(textTokens)
        textFeatures = textFeatures / textFeatures.norm(dim=-1, keepdim=True)
    return textFeatures


def classifyImage(imagePath, textFeatures, labels, model, preprocess, device):
    image = preprocess(Image.open(imagePath).convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        imageFeatures = model.encode_image(image)
        imageFeatures = imageFeatures / imageFeatures.norm(dim=-1, keepdim=True)
        similarities = (imageFeatures @ textFeatures.T).softmax(dim=-1)

    probs = similarities[0].cpu().numpy()
    bestIdx = probs.argmax()
    return labels[bestIdx], float(probs[bestIdx])


def loadExistingLabels(csvPath):
    labeled = set()
    if csvPath.exists():
        with open(csvPath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labeled.add(row["path"])
    return labeled


def main():
    inputDir = config.basePath / "input"
    inputDir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, preprocess, device = loadClipModel()
    labels = config.classes

    textFeatures = precomputeTextFeatures(labels, model, device)
    print(f"Pre-computed text features for {len(labels)} labels")

    imageExtensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    totalImages = 0
    totalLabeled = 0
    totalSkipped = 0
    startTime = time.time()

    for className in sorted(labels):
        classDir = config.dataDir / className
        if not classDir.exists():
            print(f"  Skipping {className} (directory not found)")
            continue

        imageFiles = sorted(
            f for f in classDir.iterdir()
            if f.suffix.lower() in imageExtensions
        )
        if not imageFiles:
            print(f"  Skipping {className} (no images)")
            continue

        csvPath = inputDir / f"{className}.csv"
        existing = loadExistingLabels(csvPath)
        toLabel = [f for f in imageFiles if str(f.resolve()) not in existing]

        totalImages += len(imageFiles)
        totalSkipped += len(existing)
        totalLabeled += len(toLabel)

        if not toLabel:
            print(f"  {className}: all {len(imageFiles)} images already labeled")
            continue

        print(f"  {className}: labeling {len(toLabel)} new ({len(existing)} existing)")

        writeMode = "a" if existing else "w"
        with open(csvPath, writeMode, newline="") as f:
            writer = csv.writer(f)
            if not existing:
                writer.writerow(["path", "clip_label", "clip_confidence"])

            for imgPath in tqdm(toLabel, desc=f"  {className}", leave=False):
                try:
                    clipLabel, confidence = classifyImage(
                        imgPath, textFeatures, labels, model, preprocess, device
                    )
                    writer.writerow([str(imgPath.resolve()), clipLabel, f"{confidence:.4f}"])
                except Exception as e:
                    writer.writerow([str(imgPath.resolve()), "ERROR", f"0.0000"])

    totalTime = time.time() - startTime
    print(f"\nDone in {int(totalTime // 60)}m {int(totalTime % 60)}s")
    print(f"Total: {totalImages} images, {totalLabeled} labeled, {totalSkipped} skipped")

    summaryPath = config.basePath / "reports" / "clip_label_summary.txt"
    summaryPath.parent.mkdir(exist_ok=True)
    with open(summaryPath, "w") as f:
        f.write(f"CLIP Label Summary\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Total images: {totalImages}\n")
        f.write(f"Labeled: {totalLabeled}\n")
        f.write(f"Skipped (existing): {totalSkipped}\n")
        f.write(f"Time: {int(totalTime // 60)}m {int(totalTime % 60)}s\n\n")

        for className in sorted(labels):
            csvPath = inputDir / f"{className}.csv"
            if not csvPath.exists():
                continue
            with open(csvPath, "r") as csvFile:
                reader = csv.DictReader(csvFile)
                rows = list(reader)
                total = len(rows)
                if total == 0:
                    continue
                agree = sum(1 for r in rows if r["clip_label"] == className)
                agreeRate = agree / total * 100
                f.write(f"  {className:20s}: {agree}/{total} agree ({agreeRate:.1f}%)\n")

    print(f"Summary saved to {summaryPath}")


if __name__ == "__main__":
    main()
