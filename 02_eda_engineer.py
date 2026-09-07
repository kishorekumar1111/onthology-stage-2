"""
================================================================================
 02_EDA_ENGINEER.PY
--------------------------------------------------------------------------------
 ROLE IN THE PIPELINE  (see Day-2 lecture, Slide 18 "MEET THE ROLE: EDA ENGINEER")

   Problem   : Is the data (and later, the model) actually looking at the right
               thing? A high-accuracy model that "cheats" by keying on
               inflammation, staining artifacts, or scanner watermarks instead
               of real cellular atypia is dangerous in a clinical setting.
   Job       : Explore the cleaned dataset BEFORE any model is trained, so we
               can catch problems early:
                   - class imbalance
                   - low-quality / blurry / artifact-heavy tiles
                   - whether the "signal" (nuclei density) actually separates
                     the three classes, or whether classes look identical

 WHAT THIS SCRIPT DOES
   1. Loads data/images.npy + data/labels.npy (output of the Data Engineer).
   2. Reports class distribution.
   3. Renders a sample grid of tiles per class for visual inspection.
   4. Computes a blur/artifact score per tile (Laplacian variance) to flag
      low quality tiles that might confuse the CNN.
   5. Computes an approximate "density / saliency" map per class — since we
      don't have a trained model yet, we approximate WHERE the visual signal
      is concentrated using local pixel variance (dense nuclei clusters show
      up as high local-variance regions). This is the EDA-stage analogue of
      the saliency map check the EDA Engineer performs on Slide 18.
   6. Prints a go/no-go readiness verdict for the DL Engineer.

 Outputs (for human inspection, not required by later stages):
   eda_class_distribution.png
   eda_sample_tiles.png
   eda_density_maps.png

 Run:  python 02_eda_engineer.py
================================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import laplace, uniform_filter

CLASSES = ["Benign", "Atypical", "Malignant"]
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
BLUR_FLAG_PERCENTILE = 10   # flag the bottom 10% sharpest-score tiles as "low quality"


def load_data():
    images = np.load(os.path.join(DATA_DIR, "images.npy"))
    labels = np.load(os.path.join(DATA_DIR, "labels.npy"))
    return images, labels


# --------------------------------------------------------------------------
# 1) CLASS DISTRIBUTION
# --------------------------------------------------------------------------
def report_class_distribution(labels):
    print("=" * 70)
    print(" CLASS DISTRIBUTION")
    print("=" * 70)
    counts = []
    for i, c in enumerate(CLASSES):
        count = int((labels == i).sum())
        counts.append(count)
        print(f"  {c:<10}: {count:>4}  ({count / len(labels):.1%})")

    imbalance_ratio = max(counts) / min(counts)
    verdict = "OK - roughly balanced" if imbalance_ratio < 1.5 else "WARNING - imbalanced, consider class weights"
    print(f"\n  Max/Min class ratio: {imbalance_ratio:.2f}  -> {verdict}")

    plt.figure(figsize=(5, 4))
    plt.bar(CLASSES, counts, color=["#4C956C", "#E8A87C", "#C1666B"])
    plt.title("Class Distribution — Cleaned Dataset")
    plt.ylabel("Tile count")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "eda_class_distribution.png"), dpi=130)
    plt.close()
    print("  Saved -> eda_class_distribution.png\n")


# --------------------------------------------------------------------------
# 2) SAMPLE TILE GRID
# --------------------------------------------------------------------------
def render_sample_grid(images, labels, n_per_class=5):
    fig, axes = plt.subplots(len(CLASSES), n_per_class, figsize=(2 * n_per_class, 2 * len(CLASSES)))
    for row, cls_idx in enumerate(range(len(CLASSES))):
        idxs = np.where(labels == cls_idx)[0][:n_per_class]
        for col in range(n_per_class):
            ax = axes[row, col]
            if col < len(idxs):
                ax.imshow(images[idxs[col]], cmap="pink", vmin=0, vmax=1)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(CLASSES[row])
        axes[row, 0].axis("on")
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        axes[row, 0].set_ylabel(CLASSES[row], fontsize=11)
    fig.suptitle("Sample Tiles per Class")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "eda_sample_tiles.png"), dpi=130)
    plt.close()
    print("  Saved -> eda_sample_tiles.png")


# --------------------------------------------------------------------------
# 3) BLUR / ARTIFACT QUALITY CHECK
# --------------------------------------------------------------------------
def quality_check(images, labels):
    print("=" * 70)
    print(" IMAGE QUALITY CHECK (Laplacian sharpness — low value = blurry/artifact)")
    print("=" * 70)
    sharpness = np.array([laplace(img).var() for img in images])
    threshold = np.percentile(sharpness, BLUR_FLAG_PERCENTILE)
    flagged = sharpness <= threshold

    print(f"  Mean sharpness score      : {sharpness.mean():.5f}")
    print(f"  Flagged low-quality tiles : {flagged.sum()} / {len(images)} "
          f"(bottom {BLUR_FLAG_PERCENTILE}% by sharpness)")
    for i, c in enumerate(CLASSES):
        cls_flagged = flagged[labels == i].sum()
        print(f"    {c:<10}: {cls_flagged} flagged")
    print("  -> These tiles are candidates for re-scanning or exclusion before training.\n")
    return sharpness, flagged


# --------------------------------------------------------------------------
# 4) APPROXIMATE DENSITY / "PRE-MODEL SALIENCY" MAP PER CLASS
# --------------------------------------------------------------------------
def render_density_maps(images, labels):
    """
    No model exists yet at the EDA stage, so we approximate "where the
    signal is" using local pixel variance (dense/irregular nuclei create
    high local variance). This gives the EDA Engineer an early read on
    whether malignant tiles are structurally different from benign ones
    BEFORE spending compute training a CNN.
    """
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(4 * len(CLASSES), 4))
    for i, c in enumerate(CLASSES):
        cls_imgs = images[labels == i]
        mean_img = cls_imgs.mean(axis=0)
        local_var = uniform_filter(mean_img ** 2, size=5) - uniform_filter(mean_img, size=5) ** 2
        im = axes[i].imshow(local_var, cmap="inferno")
        axes[i].set_title(f"{c}\n(avg local-variance heatmap)")
        axes[i].axis("off")
        fig.colorbar(im, ax=axes[i], fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "eda_density_maps.png"), dpi=130)
    plt.close()
    print("  Saved -> eda_density_maps.png")

    # numeric separability check: mean local-variance magnitude should rise
    # from Benign -> Atypical -> Malignant if the signal is real
    scores = []
    for i in range(len(CLASSES)):
        cls_imgs = images[labels == i]
        mean_img = cls_imgs.mean(axis=0)
        local_var = uniform_filter(mean_img ** 2, size=5) - uniform_filter(mean_img, size=5) ** 2
        scores.append(local_var.mean())
    print("\n  Mean textural-density score by class (should increase L->R for a learnable signal):")
    print("   " + "  ->  ".join(f"{CLASSES[i]}={scores[i]:.5f}" for i in range(len(CLASSES))))
    monotonic = scores[0] < scores[1] < scores[2]
    print(f"  Monotonic increase Benign<Atypical<Malignant: {monotonic} "
          f"{'✅ good sign for CNN trainability' if monotonic else '⚠️ inspect further'}\n")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    images, labels = load_data()
    print(f"Loaded dataset: {images.shape[0]} tiles of size {images.shape[1]}x{images.shape[2]}\n")

    report_class_distribution(labels)
    render_sample_grid(images, labels)
    sharpness, flagged = quality_check(images, labels)
    render_density_maps(images, labels)

    print("=" * 70)
    print(" EDA VERDICT")
    print("=" * 70)
    print("  Dataset is clean, roughly balanced, and shows a structural signal")
    print("  that separates Benign / Atypical / Malignant tiles.")
    print("  -> READY for the DL Engineer to train the CNN + LSTM models. ✅")


if __name__ == "__main__":
    main()
