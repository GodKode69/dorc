# DorC - v1

DorC is an open source image classification model. It is built with PyTorch transfer learning with EfficientNet-B0 backbone with CLIP zero-shot classification for data curation and image scraping.

### Preview

![Preview](reprots/preview.png)

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
cd code
python train.py
```

### Incremental training (add new classes/images)

```bash
cd code
python train.py --increment
```

Loads backbone weights from previous model, trains for 15 epochs at lower LR (0.0001).

### Prediction

```bash
cd code
python predict.py <image_path>
```

### Evaluation

```bash
cd code
python evaluate.py          # validation set
python evaluate.py --full   # full dataset
```

### Scrape new images

```bash
cd code
python scrape.py <count> <species>
# e.g. python scrape.py 100 wolf
```

Scrapes from Bing, classifies with CLIP, verifies with model, sorts into `img/new/` or `img/unconfirmed/`.

### Export to ONNX

```bash
cd code
python export_onnx.py
```

Exports model to ONNX format for web deployment.

## Project Structure

```
model_1/
├── code/                  # Python scripts
│   ├── classify_species.py  # CLIP zero-shot classifier
│   ├── config.py            # Classes, hyperparameters, paths
│   ├── dataset.py           # Data loading, filtering, augmentation
│   ├── evaluate.py          # Confusion matrix, classification report
│   ├── export_onnx.py       # ONNX export script
│   ├── model.py             # EfficientNet-B0 build + weight loading
│   ├── predict.py           # Single image inference
│   ├── scrape.py            # Image scraping pipeline
│   └── train.py             # Training loop with early stopping
├── data/                  # Trained class images (108 classes)
├── img/                   # Image organization
│   ├── new/                 # Scraped images confirmed by model
│   ├── unconfirmed/         # Scraped images where CLIP and model disagree
│   └── untrained/           # Classes below MIN_IMAGES threshold
├── reports/               # Generated evaluation reports
├── frontend/              # Next.js web interface (client-side ONNX inference)
│   ├── public/
│   │   ├── model.onnx       # ONNX model (~16MB)
│   │   ├── classes.json     # Class names array
│   │   ├── class_samples/   # One sample image per class
│   │   └── ort-wasm-*.wasm  # ONNX WASM runtime
│   ├── lib/useClassifier.ts # Client-side inference hook
│   ├── components/          # React components
│   └── app/                 # Pages and styles
├── env/                   # Python virtual environment
├── model.pt               # Trained weights + class_to_idx
├── preview.png            # UI preview screenshot
└── README.md
```

## Architecture

- **Backbone:** EfficientNet-B0 (ImageNet pretrained)
- **Classifier:** Dropout(0.4) → Linear(1280, num_classes)
- **Input:** Datasets from Kaggle (credits.txt) + Scraper Images + Images from APIs for initial training [dog.ceo, thecatapi.com, random-d.uk, randomfox.ca]
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

1. **Data** merged from multiple sources (Kaggle, APIs, Scraper)
2. **CLIP** classifies similar looking animals (e.g. wolf, hyena, jackal, fox)
3. **Scrape** adds images via Bing with automatic deduplication (MD5 hash)
4. **Filter** drops classes below 100 images to `img/untrained/`

## Website

Client-side inference using `onnxruntime-web` (WASM)

- **Frontend:** `Next.JS`
- **Model:** `frontend/public/model.onnx`

## Results

- **Overall accuracy:** 97.2%
- **Average confidence:** 69.4%

### Validation Evaluation

#### Top Confused Pairs (True -> Predicted: Count)

man -> woman: 161
woman -> man: 76
cat -> dog: 34
turtle -> goose: 21
dolphin -> whale: 19
dog -> cat: 16
rat -> mouse: 15
whale -> dolphin: 15
shark -> whale: 14
toad -> frog: 12
turtle -> whale: 12
fish -> corals: 11
shark -> dolphin: 11
fish -> puffers: 9
corals -> fish: 8
mouse -> rat: 8
puffers -> eel: 8
puffers -> fish: 8
fish -> shark: 7
frog -> toad: 7


## License

Open source. Use freely.
