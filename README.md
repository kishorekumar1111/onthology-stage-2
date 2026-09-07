# Stage02_DL — Precision Oncology Deep Learning Pipeline

A five-stage deep learning pipeline built around a single teaching case study:
**can an AI spot tumor progression before it spreads, using pathology tiles and
ctDNA biomarker trends?**

This project turns the Day-2 "AI Oncology Vision Team" lecture into working code.
Each script in this folder is one of the five DL roles from the lecture, run in order:

```
01_data_engineer.py        Raw, messy tiles  ->  one clean labeled dataset
02_eda_engineer.py         Is the data (and later the model) looking at the right thing?
03_dl_engineer.py          Train the CNN (vision) + LSTM (sequence) models
04_evaluation_engineer.py  Can we trust the model on tiles it has never seen?
05_integration_engineer.py Turn a raw prediction into a dashboard an oncologist can use
```

## ⚠️ Important disclaimer

**All data in this project is synthetically generated.** There is no real patient
data, no real whole-slide-image (WSI) archive, and no real biomarker data anywhere
in this pipeline. Tiles are procedurally drawn "nuclei on tissue background" images
and ctDNA sequences are simulated trend lines. This project exists purely to teach
the *shape* of a real precision-oncology DL pipeline — the five roles, the
handoffs between them, and the CNN/LSTM architectures — without needing access to
protected health information. Swap in real, IRB-approved data before ever using
patterns like this in a clinical setting.

## Why no TensorFlow / PyTorch?

The CNN and LSTM in `03_dl_engineer.py` are implemented **from scratch using only
NumPy and SciPy** (forward pass, backward pass / backprop, and the SGD training
loop are all hand-written). This is deliberate:

- It runs anywhere, with no GPU and no deep learning framework installed.
- It makes the "convolution → ReLU → pooling → fully connected" story from
  Slide 12 of the lecture literally the code you're reading, not a framework
  abstraction hiding it.
- Stages 04 and 05 reuse the same forward-pass math to run inference on the
  pickled weights — nothing about using the trained model requires a framework.

If you want to swap in a real framework, `vision_model.pkl` / `sequence_model.pkl`
are just plain dictionaries of NumPy arrays — the architecture notes in each
script tell you exactly what to rebuild in Keras/PyTorch.

## Folder structure

```
Stage02_DL/
├── data/
│   ├── images.npy          # (N, 32, 32) float32 — cleaned, labeled pathology tiles
│   ├── labels.npy          # (N,) int64 — 0=Benign 1=Atypical 2=Malignant
│   ├── X_test.npy          # held-out test tiles (never seen during training)
│   ├── y_test.npy          # held-out test labels
│   ├── vision_model.pkl    # trained CNN weights + architecture metadata
│   └── sequence_model.pkl  # trained LSTM weights + architecture metadata
├── 01_data_engineer.py
├── 02_eda_engineer.py
├── 03_dl_engineer.py
├── 04_evaluation_engineer.py
├── 05_integration_engineer.py
└── README.md
```

Running the scripts also produces these inspection artifacts alongside them:
`eda_class_distribution.png`, `eda_sample_tiles.png`, `eda_density_maps.png`,
`eval_confusion_matrix.png`, and `dashboard.html`.

## Setup

```bash
pip install numpy scipy scikit-learn matplotlib
```

(That's the entire dependency list — no GPU, no TensorFlow/PyTorch required.)

## Running the pipeline

Run the five stages **in order** — each one depends on files written by the
previous stage:

```bash
python 01_data_engineer.py         # ~5 seconds  -> data/images.npy, data/labels.npy
python 02_eda_engineer.py          # ~5 seconds  -> eda_*.png
python 03_dl_engineer.py           # ~1–2 minutes -> trains CNN + LSTM, saves both .pkl + test split
python 04_evaluation_engineer.py   # ~5 seconds  -> eval_confusion_matrix.png, trust verdict
python 05_integration_engineer.py  # ~1 second   -> console dashboard + dashboard.html
```

## Stage-by-stage details

### 1. Data Engineer — `01_data_engineer.py`
Simulates a realistic "raw lake" of pathology tiles complete with the exact
problems the lecture calls out on Slide 16: **duplicate rows** and
**corrupted/missing tiles**. It then cleans (drops corrupted + duplicate rows),
labels (Benign / Atypical / Malignant, driven by nuclei density and
irregularity), and augments (flips + rotations, simulating "different labs,
different staining protocols, scanner noise" from Slide 4) the dataset before
saving the final clean set.

### 2. EDA Engineer — `02_eda_engineer.py`
Before any model is trained, checks:
- **Class balance** — are Benign/Atypical/Malignant roughly evenly represented?
- **Image quality** — a Laplacian-sharpness score flags blurry/artifact tiles.
- **Structural signal** — since no model exists yet, approximates "where the
  visual signal is" using a local pixel-variance heatmap per class (a
  pre-model stand-in for the saliency-map check described on Slide 18), and
  verifies the signal strength increases monotonically from Benign → Atypical
  → Malignant — evidence the CNN has something real to learn.

### 3. DL Engineer — `03_dl_engineer.py`
Trains two models, matching the lecture's "CNN for images, LSTM for
sequences" split (Slide 9, Slide 11):

- **Vision model (CNN)** — `Conv(5×5×8) → ReLU → MaxPool(2×2) → Global
  Average Pool → FC(16) → ReLU → FC(3) → Softmax`. Global Average Pooling
  (the same trick used in ResNet/EfficientNet, Slide 12) is used instead of a
  giant flatten+FC layer specifically to avoid overfitting on a dataset this
  size — an early version of this pipeline with a naive flatten layer hit 95%
  train accuracy but only ~50% test accuracy; GAP fixed that gap.
- **Sequence model (LSTM)** — a single-layer LSTM that reads 5 past ctDNA
  readings and forecasts the 6th, mirroring Slide 10's "SEQUENCE OVER TIME →
  LSTM LEARNS THE TREND" example.

Also splits off a **held-out test set** of images that the CNN never trains
on, saved to `data/X_test.npy` / `data/y_test.npy` specifically for the next
stage.

### 4. Evaluation Engineer — `04_evaluation_engineer.py`
Loads the trained CNN and tests it **only** on the held-out tiles from step 3
— tiles it has never seen. Reports accuracy, a full classification report,
and a confusion matrix, then specifically hunts for the scariest failure
mode in a clinical setting: **overconfident mistakes** — wrong predictions
made with high softmax confidence (≥ 85%). This operationalizes the
lecture's "memorizing exam answers vs. actually understanding the subject"
analogy from Slide 20.

### 5. Integration Engineer — `05_integration_engineer.py`
Loads both trained models and simulates one new incoming case (a
faded-stain, dense-cell tile plus a sharply-rising ctDNA series — the same
setup as the lecture's live-case Slide 23). Combines both model outputs into
one recommendation and renders it two ways: a console dashboard, and a
standalone `dashboard.html` file — the "usable application" a Data/DL
pipeline needs before an oncologist can act on it (Slide 21).

## The five roles, one sentence each

| Role | One-line job |
|---|---|
| Data Engineer | Turn messy raw slides into one clean, labeled dataset. |
| EDA Engineer | Make sure the data (and later the model) is looking at the right thing. |
| DL Engineer | Train the CNN + LSTM on the labeled data. |
| Evaluation Engineer | Prove the model generalizes — don't trust training accuracy alone. |
| Integration Engineer | Turn a raw prediction into a decision someone can actually act on. |

## Extending this project with real data

- Replace `make_tile()` in `01_data_engineer.py` with a real WSI tiling
  pipeline (e.g. OpenSlide) reading `.svs`/`.tiff` files, and replace the
  synthetic labels with pathologist annotations.
- Replace `make_ctdna_sequence()` in `03_dl_engineer.py` with real
  longitudinal biomarker records (e.g. from a clinical data warehouse).
- Swap the from-scratch CNN/LSTM for a framework implementation
  (Keras/PyTorch) once you have enough real data to justify GPU training —
  the architecture notes in `03_dl_engineer.py` specify the exact layers to
  rebuild.
- Add proper stain normalization (e.g. Macenko/Reinhard) to the Data Engineer
  stage instead of the simplified brightness-jitter simulation used here.
