import argparse
import csv
import sys
from pathlib import Path

import clip
import torch
from PIL import Image
from tqdm import tqdm

import config


def load_clip_model(device="auto"):
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
    return labels[bestIdx], float(probs[bestIdx]), {label: float(p) for label, p in zip(labels, probs)}


def classifyDirectory(sourceDir, textFeatures, labels, model, preprocess, device):
    imageFiles = sorted(
        f for f in sourceDir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    )

    results = []
    for imgPath in tqdm(imageFiles, desc=f"Classifying {sourceDir.name}"):
        try:
            predicted, confidence, allProbs = classifyImage(imgPath, textFeatures, labels, model, preprocess, device)
            results.append({
                "file": imgPath.name,
                "predicted": predicted,
                "confidence": confidence,
                "all_probs": allProbs,
            })
        except Exception as e:
            results.append({
                "file": imgPath.name,
                "predicted": "ERROR",
                "confidence": 0.0,
                "all_probs": {},
                "error": str(e),
            })
    return results


def moveFiles(results, sourceDir, minConf=0.25):
    moved = 0
    for r in results:
        if r["predicted"] == "ERROR":
            continue
        if r["predicted"] == sourceDir.name:
            continue
        if r["confidence"] < minConf:
            continue

        src = sourceDir / r["file"]
        destDir = config.dataDir / r["predicted"]
        destDir.mkdir(exist_ok=True)

        dest = destDir / r["file"]
        if dest.exists():
            dest = destDir / f"{sourceDir.name}_{r['file']}"
        src.rename(dest)
        moved += 1
    return moved


def printSummary(results, labels):
    from collections import Counter
    counts = Counter(r["predicted"] for r in results if r["predicted"] != "ERROR")
    print(f"\n{'='*50}")
    print(f"Classification Results ({len(results)} images)")
    print(f"{'='*50}")
    for label in sorted(counts, key=counts.get, reverse=True):
        bar = "█" * (counts[label] * 30 // max(counts.values()))
        print(f"  {label:20s} {counts[label]:4d}  {bar}")
    errors = counts.get("ERROR", 0)
    if errors:
        print(f"\n  {errors} images failed to load")
    print()


def main():
    parser = argparse.ArgumentParser(description="Zero-shot species classifier using CLIP")
    parser.add_argument("--source", type=Path, required=True, help="Source directory of images")
    parser.add_argument("--labels", type=str, required=True, help="Comma-separated list of labels")
    parser.add_argument("--move", action="store_true", help="Move non-matching images to correct dirs")
    parser.add_argument("--min-conf", type=float, default=0.25, help="Min confidence to move (default: 0.25)")
    parser.add_argument("--csv", type=str, default=None, help="Save results to CSV file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be moved without moving")
    args = parser.parse_args()

    sourceDir = args.source if args.source.is_absolute() else config.dataDir / args.source
    if not sourceDir.exists():
        print(f"Error: {sourceDir} does not exist")
        sys.exit(1)

    labels = [l.strip() for l in args.labels.split(",")]
    print(f"Source: {sourceDir}")
    print(f"Labels: {labels}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    model, preprocess, device = load_clip_model()
    textFeatures = precomputeTextFeatures(labels, model, device)
    print(f"Pre-computed text features for {len(labels)} labels")
    results = classifyDirectory(sourceDir, textFeatures, labels, model, preprocess, device)
    printSummary(results, labels)

    if args.csv:
        csvPath = Path(args.csv)
        with open(csvPath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "predicted", "confidence"])
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in ["file", "predicted", "confidence"]})
        print(f"Results saved to {csvPath}")

    if args.move or args.dry_run:
        moved = moveFiles(results, sourceDir, args.min_conf)
        action = "Would move" if args.dry_run else "Moved"
        print(f"{action} {moved} images to other directories")
        if args.dry_run:
            print("(dry run — no files were actually moved)")


if __name__ == "__main__":
    main()
