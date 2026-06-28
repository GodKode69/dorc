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


def classify_image(image_path, labels, model, preprocess, device):
    text_tokens = clip.tokenize([f"a photo of a {label}" for label in labels]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ text_features.T).softmax(dim=-1)

    probs = similarities[0].cpu().numpy()
    best_idx = probs.argmax()
    return labels[best_idx], float(probs[best_idx]), {label: float(p) for label, p in zip(labels, probs)}


def classify_directory(source_dir, labels, model, preprocess, device):
    image_files = sorted(
        f for f in source_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    )

    results = []
    for img_path in tqdm(image_files, desc=f"Classifying {source_dir.name}"):
        try:
            predicted, confidence, all_probs = classify_image(img_path, labels, model, preprocess, device)
            results.append({
                "file": img_path.name,
                "predicted": predicted,
                "confidence": confidence,
                "all_probs": all_probs,
            })
        except Exception as e:
            results.append({
                "file": img_path.name,
                "predicted": "ERROR",
                "confidence": 0.0,
                "all_probs": {},
                "error": str(e),
            })
    return results


def move_files(results, source_dir, min_conf=0.25):
    moved = 0
    for r in results:
        if r["predicted"] == "ERROR":
            continue
        if r["predicted"] == source_dir.name:
            continue
        if r["confidence"] < min_conf:
            continue

        src = source_dir / r["file"]
        dest_dir = config.dataDir / r["predicted"]
        dest_dir.mkdir(exist_ok=True)

        dest = dest_dir / r["file"]
        if dest.exists():
            dest = dest_dir / f"{source_dir.name}_{r['file']}"
        src.rename(dest)
        moved += 1
    return moved


def print_summary(results, labels):
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

    source_dir = args.source if args.source.is_absolute() else config.dataDir / args.source
    if not source_dir.exists():
        print(f"Error: {source_dir} does not exist")
        sys.exit(1)

    labels = [l.strip() for l in args.labels.split(",")]
    print(f"Source: {source_dir}")
    print(f"Labels: {labels}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()

    model, preprocess, device = load_clip_model()
    results = classify_directory(source_dir, labels, model, preprocess, device)
    print_summary(results, labels)

    if args.csv:
        csv_path = Path(args.csv)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "predicted", "confidence"])
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in ["file", "predicted", "confidence"]})
        print(f"Results saved to {csv_path}")

    if args.move or args.dry_run:
        moved = move_files(results, source_dir, args.min_conf)
        action = "Would move" if args.dry_run else "Moved"
        print(f"{action} {moved} images to other directories")
        if args.dry_run:
            print("(dry run — no files were actually moved)")


if __name__ == "__main__":
    main()
