import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from icrawler.builtin import BingImageCrawler
from PIL import Image
from tqdm import tqdm

import config
from classify_species import classify_image, load_clip_model
from predict import predict as predict_single

TMP_ROOT = Path("/tmp/opencode")

MIN_IMAGE_SIZE = 10_000


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_existing_hashes(*dirs):
    hashes = set()
    for d in dirs:
        if not d.exists():
            continue
        for f in d.rglob("*"):
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                hashes.add(file_hash(f))
    return hashes


def dedup_images(image_paths, existing_hashes):
    unique = []
    skipped = 0
    for p in image_paths:
        h = file_hash(p)
        if h in existing_hashes:
            skipped += 1
            p.unlink()
        else:
            existing_hashes.add(h)
            unique.append(p)
    if skipped:
        print(f"  Dedup: skipped {skipped} duplicates")
    return unique


def scrape_images(species, count, tmp_dir):
    tmp_dir.mkdir(parents=True, exist_ok=True)

    crawler = BingImageCrawler(
        storage={"root_dir": str(tmp_dir)},
        downloader_threads=4,
    )
    crawler.downloader.session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    })
    crawler.crawl(
        keyword=f"{species} animal photo",
        max_num=count * 2,
        min_size=(100, 100),
        file_idx_offset="auto",
    )

    images = [
        f for f in tmp_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        and f.stat().st_size >= MIN_IMAGE_SIZE
    ]
    print(f"  Scraped {len(images)} valid images (target: {count})")
    return images


def next_filename(dest_dir, prefix):
    existing = list(dest_dir.glob(f"{prefix}_*.jpg")) + list(dest_dir.glob(f"{prefix}_*.jpeg")) + list(dest_dir.glob(f"{prefix}_*.png"))
    nums = []
    for f in existing:
        try:
            nums.append(int(f.stem.split("_")[-1]))
        except ValueError:
            pass
    n = max(nums, default=0) + 1
    return dest_dir / f"{prefix}_{n:04d}.jpg"


def run_pipeline(species, count):
    clip_model, preprocess, device = load_clip_model()
    labels = config.CLASSES

    confirmed = 0
    total_scraped = 0
    stats = {"species_match": 0, "reclassified": 0, "unconfirmed": 0, "skipped": 0, "deduped": 0}

    species_dir = config.NEW_DIR / species
    species_dir.mkdir(parents=True, exist_ok=True)

    existing_hashes = load_existing_hashes(config.NEW_DIR, config.UNCONFIRMED_DIR, config.DATA_DIR)
    print(f"  Loaded {len(existing_hashes)} existing image hashes for dedup")

    max_retries = 3
    retries = 0

    while confirmed < count:
        remaining = count - confirmed
        print(f"\n--- Need {remaining} more '{species}' images (confirmed so far: {confirmed}) ---")

        tmp_dir = TMP_ROOT / species
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        images = scrape_images(species, remaining, tmp_dir)
        if not images:
            retries += 1
            if retries >= max_retries:
                print(f"  Failed after {max_retries} retries. Stopping.")
                break
            print(f"  No images scraped. Retrying ({retries}/{max_retries})...")
            import time; time.sleep(5)
            continue

        retries = 0
        before_dedup = len(images)
        images = dedup_images(images, existing_hashes)
        stats["deduped"] += before_dedup - len(images)
        total_scraped += len(images)

        if not images:
            print("  All scraped images were duplicates, retrying...")
            continue

        for img_path in tqdm(images, desc="Classifying"):
            try:
                cl_label, cl_conf, _ = classify_image(img_path, labels, clip_model, preprocess, device)
            except Exception:
                stats["skipped"] += 1
                continue

            if cl_label == species:
                dest = next_filename(species_dir, species)
                shutil.move(str(img_path), str(dest))
                confirmed += 1
                stats["species_match"] += 1
                if confirmed >= count:
                    break
                continue

            try:
                pr_label, pr_conf = predict_single(str(img_path))
            except Exception:
                stats["skipped"] += 1
                continue

            if pr_label == cl_label:
                dest_dir = config.NEW_DIR / cl_label
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = next_filename(dest_dir, cl_label)
                shutil.move(str(img_path), str(dest))
                stats["reclassified"] += 1
            else:
                dest_dir = config.UNCONFIRMED_DIR / cl_label
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / img_path.name
                if dest.exists():
                    dest = dest_dir / f"{img_path.stem}_{img_path.suffix}"
                shutil.move(str(img_path), str(dest))
                stats["unconfirmed"] += 1

        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'='*50}")
    print(f"Pipeline complete: {species}")
    print(f"{'='*50}")
    print(f"  Confirmed {species}:  {stats['species_match']}")
    print(f"  Reclassified to new/: {stats['reclassified']}")
    print(f"  Unconfirmed:          {stats['unconfirmed']}")
    print(f"  Duplicates removed:   {stats['deduped']}")
    print(f"  Skipped (errors):     {stats['skipped']}")
    print(f"  Total scraped:        {total_scraped}")
    print(f"  Output: {species_dir}/")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape, classify, and sort animal images")
    parser.add_argument("count", type=int, help="Number of confirmed images to collect")
    parser.add_argument("species", type=str, help="Species to scrape (e.g. wolf)")
    args = parser.parse_args()

    if args.species not in config.CLASSES:
        print(f"Error: '{args.species}' not in CLASSES list")
        print(f"Available: {', '.join(config.CLASSES)}")
        sys.exit(1)

    run_pipeline(args.species, args.count)
