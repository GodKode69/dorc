# DorC -

DorC is an open source image recognition model. It is built with PyTorch transfer learning with EfficientNet-B0 backbone with CLIP zero-shot classification for data curation and image scraping.

## Classes

**108 trained classes** with 60,000+ images across mammals, birds, reptiles, insects, fish, and marine animals.
Categorized - General/genus-level

## Setup

```bash
python -m venv env
source env/bin/activate
pip install torch torchvision Pillow scikit-learn seaborn matplotlib icrawler
```

CLIP (for zero-shot classification):

```bash
pip install git+https://github.com/openai/CLIP.git
```

## Usage

### Train from scratch

```bash
python train.py
```

### Incremental training (add new classes/images)

```bash
python train.py --increment
```

Loads backbone weights from previous model, trains for 15 epochs at lower LR (0.0001).

### Prediction

```bash
python predict.py <image_path>
```

### Evaluation

```bash
python evaluate.py          # validation set
python evaluate.py --full   # full dataset
```

### Scrape new images

```bash
python scrape.py <count> <species>
# e.g. python scrape.py 100 wolf
```

Scrapes from Bing, classifies with CLIP, verifies with model, sorts into `new/` or `unconfirmed/`.

## Project Structure

```
model_1/
├── data/                  # Trained class images (108 classes)
├── new/                   # Scraped images confirmed by model
├── unconfirmed/           # Scraped images where CLIP and model disagree
├── untrained/             # Classes below MIN_IMAGES threshold (34 classes)
├── classify_species.py    # CLIP zero-shot classifier
├── config.py              # Classes, hyperparameters, paths
├── dataset.py             # Data loading, filtering, augmentation
├── evaluate.py            # Confusion matrix, classification report
├── model.pt               # Trained weights + class_to_idx
├── model.py               # EfficientNet-B0 build + weight loading
├── predict.py             # Single image inference
├── scrape.py              # Image scraping pipeline
└── train.py               # Training loop with early stopping
```

## Architecture

- **Backbone:** EfficientNet-B0 (ImageNet pretrained)
- **Classifier:** Dropout(0.4) → Linear(1280, num_classes)
- **Input:** Datasets from Kaggle (credits.txt) + Scraper Images + Images from API's []
- **Optimization:** AdamW, CosineAnnealingLR, Label Smoothing (0.1)
- **Augmentation:** RandomCrop, Flip, Rotation, ColorJitter, GaussianBlur, RandomErasing

## Training Details

| Parameter        | Value                        |
| ---------------- | ---------------------------- |
| Batch Size       | 64 (incremental: 32)         |
| Epochs           | 30 (incremental: 15)         |
| Learning Rate    | 0.0003 (incremental: 0.0001) |
| Weight Decay     | 0.01                         |
| Early Stopping   | 7 epochs patience            |
| Min Images/Class | 100                          |

## Data Pipeline

1. **Data** merged from multiple sources (Kaggle, API's, Scraper)
2. **CLIP** classies similar looking animals (e.g. wolf, hyena, jackal, fox)
3. **Scrape** adds images via Bing with automatic deduplication (MD5 hash)
4. **Filter** drops classes below 100 images to `untrained/`

## Results

- **Val Accuracy:** 97.35% (108 classes, 58K+ train / 14K+ val images)

## License

Open source. Use freely.
