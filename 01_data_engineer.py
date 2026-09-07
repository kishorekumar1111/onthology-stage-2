"""
================================================================================
 01_DATA_ENGINEER.PY
--------------------------------------------------------------------------------
 ROLE IN THE PIPELINE  (see Day-2 lecture, Slide 17 "MEET THE ROLE: DATA ENGINEER")

   Problem   : Raw histopathology tiles arrive from many scanners/labs. They are
               unlabeled, sometimes duplicated, sometimes corrupted/missing.
   Job       : Produce ONE clean, labeled, augmented dataset that every
               downstream role (EDA -> DL -> Evaluation -> Integration) can trust.

 WHAT THIS SCRIPT DOES
   1. Simulates a realistic "raw lake" of pathology tiles (since we don't have
      access to a real hospital PACS/whole-slide-image archive in this course,
      we SYNTHESIZE tiles that mimic benign / atypical / malignant tissue
      patterns, on purpose injecting the exact problems Slide 16 calls out:
         - duplicate rows
         - missing / corrupted tiles (NaNs)
         - stain & scanner noise variation
   2. CLEANS the raw lake:
         - drops exact duplicate tiles
         - drops corrupted/missing tiles
   3. AUGMENTS the cleaned tiles (flips / rotations) to simulate the
      "different labs, different staining protocols, scanner noise" problem
      from Slide 4, which makes the downstream CNN more robust.
   4. Saves the final, clean, labeled dataset to:
         data/images.npy   -> float32 array, shape (N, 32, 32)
         data/labels.npy   -> int64  array, shape (N,)   0=Benign 1=Atypical 2=Malignant

 Run:  python 01_data_engineer.py
================================================================================
"""

import numpy as np
from scipy.ndimage import gaussian_filter, rotate
import os

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
RANDOM_SEED   = 42
IMG_SIZE      = 32                    # tiles are 32x32 grayscale pixels
CLASSES       = ["Benign", "Atypical", "Malignant"]   # label 0, 1, 2
N_PER_CLASS   = 220                   # raw tiles generated per class (before cleaning)
DUP_FRACTION  = 0.05                  # fraction of raw rows duplicated on purpose
CORRUPT_FRACTION = 0.04               # fraction of raw rows corrupted (NaN) on purpose
DATA_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

rng = np.random.default_rng(RANDOM_SEED)


# --------------------------------------------------------------------------
# STEP 1 — SYNTHETIC TILE GENERATOR
# --------------------------------------------------------------------------
def make_tile(cls_idx: int) -> np.ndarray:
    """
    Generate one synthetic 32x32 pathology tile for a given class.

    The "tissue" is a soft pink-ish background texture (smoothed noise).
    "Nuclei" are dark blobs stamped on top of it. As malignancy increases:
        - MORE nuclei are stamped (cell density / crowding)
        - nuclei become MORE irregular in size (pleomorphism / atypia)
        - background architecture gets noisier (disrupted architecture)

    This mirrors the Slide 7 description:
        BENIGN    -> "Normal tissue architecture - no atypia"
        ATYPICAL  -> "Mild nuclear atypia, some crowding"
        MALIGNANT -> "Dense malignant clusters, disrupted architecture"
    """
    # smooth background "tissue" texture
    base = rng.normal(loc=0.75, scale=0.04, size=(IMG_SIZE, IMG_SIZE))
    base = gaussian_filter(base, sigma=1.4)

    n_nuclei      = {0: 6,  1: 14, 2: 26}[cls_idx] + rng.integers(-2, 3)
    max_radius    = {0: 2,  1: 3,  2: 4}[cls_idx]
    architecture_noise = {0: 0.01, 1: 0.03, 2: 0.06}[cls_idx]

    yy, xx = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    for _ in range(max(n_nuclei, 0)):
        cy, cx = rng.integers(2, IMG_SIZE - 2, size=2)
        r = rng.integers(1, max_radius + 1)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        base[mask] -= rng.uniform(0.30, 0.55)

    # disrupted architecture -> extra high-frequency noise for higher grade tiles
    base += rng.normal(0, architecture_noise, size=base.shape)

    return np.clip(base, 0.0, 1.0).astype(np.float32)


def apply_stain_and_scanner_noise(tile: np.ndarray) -> np.ndarray:
    """
    Simulate Slide 4's "Stain Variation, Artifacts" problem:
    different labs / different staining protocols / scanner noise.
    """
    brightness = rng.uniform(0.85, 1.15)
    tile = np.clip(tile * brightness, 0, 1)
    if rng.random() < 0.15:                       # occasional scanner blur artifact
        tile = gaussian_filter(tile, sigma=rng.uniform(0.8, 1.6))
    return tile.astype(np.float32)


# --------------------------------------------------------------------------
# STEP 2 — BUILD THE MESSY "RAW LAKE"  (on purpose dirty, like Slide 16)
# --------------------------------------------------------------------------
def build_raw_lake():
    raw_tiles, raw_labels = [], []

    for cls_idx in range(len(CLASSES)):
        for _ in range(N_PER_CLASS):
            tile = make_tile(cls_idx)
            tile = apply_stain_and_scanner_noise(tile)
            raw_tiles.append(tile)
            raw_labels.append(cls_idx)

    raw_tiles = np.stack(raw_tiles)
    raw_labels = np.array(raw_labels, dtype=np.int64)

    n = len(raw_labels)

    # --- inject duplicate rows (Slide 16: "Tile_014 <- Duplicate") ---------
    n_dupes = int(n * DUP_FRACTION)
    dupe_idx = rng.choice(n, size=n_dupes, replace=False)
    raw_tiles = np.concatenate([raw_tiles, raw_tiles[dupe_idx]], axis=0)
    raw_labels = np.concatenate([raw_labels, raw_labels[dupe_idx]], axis=0)

    # --- inject corrupted / missing rows (Slide 16: "corrupted") ----------
    n = len(raw_labels)
    n_corrupt = int(n * CORRUPT_FRACTION)
    corrupt_idx = rng.choice(n, size=n_corrupt, replace=False)
    for i in corrupt_idx:
        if rng.random() < 0.5:
            raw_tiles[i] = np.nan               # missing scan
        else:
            raw_tiles[i] = rng.normal(0, 5, size=(IMG_SIZE, IMG_SIZE))  # sensor corruption

    # shuffle so classes/dupes/corruption are not in tidy blocks
    perm = rng.permutation(len(raw_labels))
    return raw_tiles[perm], raw_labels[perm]


# --------------------------------------------------------------------------
# STEP 3 — CLEAN THE LAKE
# --------------------------------------------------------------------------
def clean_dataset(tiles: np.ndarray, labels: np.ndarray):
    n_start = len(labels)

    # (a) drop corrupted / missing tiles: NaNs, or unrealistic pixel range
    valid_mask = ~np.isnan(tiles).any(axis=(1, 2))
    valid_mask &= (tiles.min(axis=(1, 2)) >= -0.01) & (tiles.max(axis=(1, 2)) <= 1.5)
    tiles, labels = tiles[valid_mask], labels[valid_mask]
    n_after_corrupt = len(labels)

    # (b) drop exact duplicate tiles (hash each flattened tile)
    flat = tiles.reshape(len(tiles), -1)
    _, unique_idx = np.unique(np.round(flat, 6), axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    tiles, labels = tiles[unique_idx], labels[unique_idx]
    n_after_dupes = len(labels)

    print(f"  Raw tiles                : {n_start}")
    print(f"  After removing corrupted : {n_after_corrupt}  (-{n_start - n_after_corrupt})")
    print(f"  After removing duplicates: {n_after_dupes}  (-{n_after_corrupt - n_after_dupes})")

    return tiles.astype(np.float32), labels.astype(np.int64)


# --------------------------------------------------------------------------
# STEP 4 — AUGMENT  (simulate different scanners / stain orientations)
# --------------------------------------------------------------------------
def augment_dataset(tiles: np.ndarray, labels: np.ndarray, factor: int = 1):
    """Add horizontally/vertically flipped + slightly rotated copies."""
    aug_tiles, aug_labels = [tiles], [labels]

    for _ in range(factor):
        flipped = tiles[:, :, ::-1]                      # horizontal flip
        aug_tiles.append(flipped)
        aug_labels.append(labels)

        rotated = np.stack([
            rotate(t, angle=rng.uniform(-15, 15), reshape=False, mode="reflect")
            for t in tiles
        ])
        aug_tiles.append(np.clip(rotated, 0, 1).astype(np.float32))
        aug_labels.append(labels)

    return np.concatenate(aug_tiles, axis=0), np.concatenate(aug_labels, axis=0)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 70)
    print(" DATA ENGINEER — building the raw, messy pathology tile lake")
    print("=" * 70)
    raw_tiles, raw_labels = build_raw_lake()
    print(f"  Generated {len(raw_labels)} raw rows across {len(CLASSES)} classes "
          f"(includes injected duplicates + corruption)\n")

    print("-" * 70)
    print(" Cleaning...")
    print("-" * 70)
    clean_tiles, clean_labels = clean_dataset(raw_tiles, raw_labels)

    print("\n" + "-" * 70)
    print(" Augmenting (stain/scanner robustness)...")
    print("-" * 70)
    final_tiles, final_labels = augment_dataset(clean_tiles, clean_labels, factor=1)
    print(f"  Final augmented dataset size: {len(final_labels)}")

    # final shuffle
    perm = rng.permutation(len(final_labels))
    final_tiles, final_labels = final_tiles[perm], final_labels[perm]

    np.save(os.path.join(DATA_DIR, "images.npy"), final_tiles)
    np.save(os.path.join(DATA_DIR, "labels.npy"), final_labels)

    print("\n" + "=" * 70)
    print(" CLASS BALANCE IN FINAL CLEAN DATASET")
    print("=" * 70)
    for i, c in enumerate(CLASSES):
        count = int((final_labels == i).sum())
        print(f"  {c:<10}: {count:>4}  ({count / len(final_labels):.1%})")

    print(f"\n Saved -> {os.path.join(DATA_DIR, 'images.npy')}")
    print(f" Saved -> {os.path.join(DATA_DIR, 'labels.npy')}")
    print("\n ONE CLEAN LABELED DATASET is ready for the EDA Engineer. ✅")


if __name__ == "__main__":
    main()
