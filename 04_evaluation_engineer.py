"""
================================================================================
 04_EVALUATION_ENGINEER.PY
--------------------------------------------------------------------------------
 ROLE IN THE PIPELINE  (see Day-2 lecture, Slide 20 "MEET THE ROLE: EVALUATION ENGINEER")

   Problem   : The model reports 100% accuracy on the TRAINING slides. Should we
               trust it immediately? NO — it may have only memorized the
               training tiles instead of learning the real malignancy pattern
               (the "memorizing exam answers vs actually understanding the
               subject" analogy from the lecture notes).
   Job       : Test the model on UNSEEN slides (data/X_test.npy, data/y_test.npy
               — tiles the model never touched during training) and check for
               overconfidence: cases where the model is very sure of a WRONG
               prediction (e.g. mistaking inflammation for malignancy).

 WHAT THIS SCRIPT DOES
   1. Loads the trained vision model (data/vision_model.pkl).
   2. Loads the held-out test set (data/X_test.npy, data/y_test.npy) that the
      DL Engineer set aside and the CNN never saw during training.
   3. Runs inference (forward pass only — no training here).
   4. Reports accuracy, a full classification report, and a confusion matrix.
   5. Flags "overconfident mistakes": wrong predictions made with high
      softmax confidence — the clinical failure mode we care about most,
      since a confidently-wrong model is more dangerous than an unsure one.
   6. Saves a confusion matrix plot for the tumor board / stakeholders.

 Run:  python 04_evaluation_engineer.py
================================================================================
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CLASSES = ["Benign", "Atypical", "Malignant"]
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OVERCONFIDENCE_THRESHOLD = 0.85   # "sure" prediction that turned out wrong


# --------------------------------------------------------------------------
# Forward-only CNN inference (must mirror the architecture trained in
# 03_dl_engineer.py: Conv -> ReLU -> MaxPool -> GlobalAvgPool -> FC -> FC -> Softmax)
# --------------------------------------------------------------------------
def cnn_predict_proba(params, X_batch):
    B = X_batch.shape[0]
    n_filters, k = params["n_filters"], params["k"]
    pool_out = params["pool_out"]
    conv_out = params["img_size"] - k + 1

    conv = np.empty((B, n_filters, conv_out, conv_out), dtype=np.float32)
    for b in range(B):
        for f in range(n_filters):
            conv[b, f] = correlate2d(X_batch[b], params["conv_w"][f], mode="valid") + params["conv_b"][f]

    relu = np.maximum(conv, 0)
    pooled = relu.reshape(B, n_filters, pool_out, 2, pool_out, 2)
    pooled_max = pooled.max(axis=(3, 5))
    gap = pooled_max.mean(axis=(2, 3))

    fc1 = np.maximum(gap @ params["fc1_w"] + params["fc1_b"], 0)
    logits = fc1 @ params["fc2_w"] + params["fc2_b"]

    logits_shift = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits_shift)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs


# --------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" EVALUATION ENGINEER — testing the CNN on UNSEEN slides")
    print("=" * 70)

    with open(os.path.join(DATA_DIR, "vision_model.pkl"), "rb") as f:
        vision_model = pickle.load(f)

    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    print(f"Loaded {len(y_test)} held-out test tiles the model has NEVER seen.\n")

    probs = cnn_predict_proba(vision_model, X_test)
    y_pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    # ---- headline metrics ----
    acc = accuracy_score(y_test, y_pred)
    print("-" * 70)
    print(f" TEST ACCURACY (unseen tiles): {acc:.3f}")
    print("-" * 70)
    print("\n Classification report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES, digits=3))

    # ---- confusion matrix ----
    cm = confusion_matrix(y_test, y_pred)
    print(" Confusion matrix (rows = actual, cols = predicted):")
    header = "            " + "".join(f"{c:>11}" for c in CLASSES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {CLASSES[i]:<10}" + "".join(f"{v:>11}" for v in row))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — Test Accuracy {acc:.1%}")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "eval_confusion_matrix.png"), dpi=130)
    plt.close()
    print("\n Saved -> eval_confusion_matrix.png")

    # ---- overconfidence / "did it really learn, or memorize?" check ----
    print("\n" + "=" * 70)
    print(" CAN WE TRUST IT? — Overconfidence check")
    print("=" * 70)
    wrong_mask = y_pred != y_test
    overconfident_wrong = wrong_mask & (confidence >= OVERCONFIDENCE_THRESHOLD)

    print(f"  Total wrong predictions           : {wrong_mask.sum()} / {len(y_test)}")
    print(f"  ...of which CONFIDENTLY wrong (>= {OVERCONFIDENCE_THRESHOLD:.0%} conf): "
          f"{overconfident_wrong.sum()}")

    if overconfident_wrong.sum() > 0:
        print("\n  ⚠️  Example overconfident mistakes (index, actual -> predicted, confidence):")
        idxs = np.where(overconfident_wrong)[0][:5]
        for i in idxs:
            print(f"    tile[{i}]: actual={CLASSES[y_test[i]]:<9} "
                  f"predicted={CLASSES[y_pred[i]]:<9} confidence={confidence[i]:.2%}")
        print("\n  -> These are the highest-risk clinical failure mode: a model that is")
        print("     SURE of the wrong answer. Flag for the EDA Engineer to inspect")
        print("     saliency maps on these specific tiles before deployment.")
    else:
        print("\n  ✅ No high-confidence mistakes found — errors the model does make tend")
        print("     to be low-confidence, meaning it 'knows what it doesn't know'.")

    # ---- most-confused class pair ----
    off_diag = cm.copy().astype(float)
    np.fill_diagonal(off_diag, 0)
    if off_diag.max() > 0:
        i, j = np.unravel_index(off_diag.argmax(), off_diag.shape)
        print(f"\n  Most common confusion: true '{CLASSES[i]}' predicted as '{CLASSES[j]}' "
              f"({int(off_diag[i, j])} times)")

    print("\n" + "=" * 70)
    verdict = "TRUSTWORTHY ✅ (generalizes beyond training data)" if acc >= 0.70 else \
              "NEEDS MORE WORK ⚠️ (too close to chance / memorizing)"
    print(f" VERDICT: {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
