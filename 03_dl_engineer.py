"""
================================================================================
 03_DL_ENGINEER.PY
--------------------------------------------------------------------------------
 ROLE IN THE PIPELINE  (see Day-2 lecture, Slide 19 "MEET THE ROLE: DL ENGINEER")

   Problem   : Slides are labeled and verified. Now we need a model that can
               (a) classify a NEW tile as Benign / Atypical / Malignant, and
               (b) forecast the NEXT ctDNA biomarker reading from a patient's
                   recent trend.
   Job       : Train the CNN + LSTM/Transformer on labeled slides and
               biomarker sequences (Slide 9, Slide 11).

 WHAT THIS SCRIPT DOES
   1. Loads the clean dataset from the Data Engineer (data/images.npy/labels.npy).
   2. Synthesizes a matching ctDNA time-series per tile, whose trend is tied
      to the tile's label (Benign=flat, Atypical=mild rise, Malignant=sharp
      rise) — mirroring Slide 10's "SEQUENCE OVER TIME -> LSTM LEARNS THE TREND".
   3. Splits data into train/test.
        - The TEST split of the IMAGE data is saved to disk
          (data/X_test.npy, data/y_test.npy) so the Evaluation Engineer can
          test the model on tiles it has never seen (Slide 20).
   4. Builds and trains, FROM SCRATCH WITH NUMPY (no TensorFlow/PyTorch
      dependency, so the whole pipeline runs anywhere):
        (a) VISION MODEL — a small CNN implementing exactly the four core
            layers from Slide 12: Convolution -> ReLU -> Pooling -> Fully
            Connected -> Softmax.
        (b) SEQUENCE MODEL — a single-layer LSTM (Slide 13) that reads the
            last 5 ctDNA readings and forecasts the next one.
   5. Saves the trained models:
        data/vision_model.pkl    (CNN weights + architecture metadata)
        data/sequence_model.pkl  (LSTM weights + architecture metadata)

 Run:  python 03_dl_engineer.py
================================================================================
"""

import os
import pickle
import numpy as np
from scipy.signal import correlate2d, convolve2d
from sklearn.model_selection import train_test_split

CLASSES = ["Benign", "Atypical", "Malignant"]
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# =============================================================================
# PART A — SYNTHETIC ctDNA SEQUENCES (tied to the tile's label)
# =============================================================================
SEQ_LEN = 5   # 5 past readings -> predict the 6th ("next reading")


def make_ctdna_sequence(cls_idx: int) -> np.ndarray:
    """
    Build one ctDNA time-series of length SEQ_LEN + 1 (last value = target).
    Trend strength mirrors Slide 10: Benign flat, Atypical mild rise,
    Malignant sharp rise (like the R-4..Now example: 1.2 -> 1.8 -> 2.6 -> 3.5 -> 4.3).
    """
    base = rng.uniform(0.8, 1.6)
    slope = {0: rng.uniform(-0.02, 0.05),     # Benign: roughly flat
             1: rng.uniform(0.15, 0.35),      # Atypical: mild rise
             2: rng.uniform(0.55, 0.95)}[cls_idx]  # Malignant: sharp rise
    noise_scale = 0.08
    series = [base]
    for t in range(1, SEQ_LEN + 1):
        nxt = series[-1] + slope + rng.normal(0, noise_scale)
        series.append(max(nxt, 0.05))
    return np.array(series, dtype=np.float32)   # length SEQ_LEN+1


def build_sequence_dataset(labels: np.ndarray):
    seqs = np.stack([make_ctdna_sequence(c) for c in labels])
    X = seqs[:, :SEQ_LEN]          # (N, SEQ_LEN)   past readings
    y = seqs[:, SEQ_LEN]           # (N,)           next reading (target)
    return X.astype(np.float32), y.astype(np.float32)


# =============================================================================
# PART B — CNN FROM SCRATCH (numpy)
#   Conv(5x5x8, valid) -> ReLU -> MaxPool(2x2) -> Global-Average-Pool
#   -> FC(16) -> ReLU -> FC(3) -> Softmax
#
#   NOTE ON DESIGN: a naive "flatten everything into one big FC layer" head
#   (14*14*8 = 1568 inputs) has ~100k parameters — far more than our ~1500
#   training tiles, so it memorizes the training set instead of generalizing
#   (we saw this first-hand: 95% train accuracy but ~50% test accuracy).
#   Global Average Pooling (the same trick used in real architectures like
#   ResNet/EfficientNet from Slide 12) collapses each feature map to ONE
#   number — "how strongly did this filter fire, on average, over the whole
#   tile" — before the FC layers. This cuts the head down to a few hundred
#   parameters and forces the model to learn genuine texture/density filters
#   instead of memorizing pixel positions.
# =============================================================================
def init_cnn(img_size=32, n_filters=8, k=5, fc_hidden=16, n_classes=3):
    conv_out = img_size - k + 1               # valid conv, e.g. 32-5+1=28
    pool_out = conv_out // 2                   # 2x2 maxpool -> 14

    scale_conv = np.sqrt(2.0 / (k * k))
    scale_fc1 = np.sqrt(2.0 / n_filters)
    scale_fc2 = np.sqrt(2.0 / fc_hidden)

    params = {
        "conv_w": rng.normal(0, scale_conv, size=(n_filters, k, k)).astype(np.float32),
        "conv_b": np.zeros(n_filters, dtype=np.float32),
        "fc1_w": rng.normal(0, scale_fc1, size=(n_filters, fc_hidden)).astype(np.float32),
        "fc1_b": np.zeros(fc_hidden, dtype=np.float32),
        "fc2_w": rng.normal(0, scale_fc2, size=(fc_hidden, n_classes)).astype(np.float32),
        "fc2_b": np.zeros(n_classes, dtype=np.float32),
        "img_size": img_size, "n_filters": n_filters, "k": k,
        "pool_out": pool_out,
        "classes": CLASSES,
    }
    return params


def cnn_forward(params, X_batch, keep_cache=False):
    """X_batch: (B, H, W) float32 in [0,1]. Returns probs (B, n_classes) [+ cache]."""
    B = X_batch.shape[0]
    n_filters, k = params["n_filters"], params["k"]
    pool_out = params["pool_out"]

    conv_out = params["img_size"] - k + 1
    conv = np.empty((B, n_filters, conv_out, conv_out), dtype=np.float32)
    for b in range(B):
        for f in range(n_filters):
            conv[b, f] = correlate2d(X_batch[b], params["conv_w"][f], mode="valid") + params["conv_b"][f]

    relu = np.maximum(conv, 0)

    # 2x2 max pool, stride 2, with argmax tracking for backprop
    pooled = relu.reshape(B, n_filters, pool_out, 2, pool_out, 2)
    pooled_max = pooled.max(axis=(3, 5))                     # (B, F, pool_out, pool_out)

    gap = pooled_max.mean(axis=(2, 3))                        # (B, F) global average pool
    fc1 = gap @ params["fc1_w"] + params["fc1_b"]
    fc1_relu = np.maximum(fc1, 0)
    logits = fc1_relu @ params["fc2_w"] + params["fc2_b"]

    logits_shift = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits_shift)
    probs = exp / exp.sum(axis=1, keepdims=True)

    if not keep_cache:
        return probs

    cache = dict(X_batch=X_batch, conv=conv, relu=relu, pooled=pooled,
                 pooled_max=pooled_max, gap=gap, fc1=fc1, fc1_relu=fc1_relu, probs=probs)
    return probs, cache


def cnn_backward(params, cache, y_true, lr):
    """One SGD step of backprop given integer labels y_true (B,)."""
    B = cache["probs"].shape[0]
    n_filters, k = params["n_filters"], params["k"]
    pool_out = params["pool_out"]

    y_onehot = np.zeros_like(cache["probs"])
    y_onehot[np.arange(B), y_true] = 1
    d_logits = (cache["probs"] - y_onehot) / B              # softmax + CE grad

    d_fc2_w = cache["fc1_relu"].T @ d_logits
    d_fc2_b = d_logits.sum(axis=0)
    d_fc1_relu = d_logits @ params["fc2_w"].T

    d_fc1 = d_fc1_relu * (cache["fc1"] > 0)
    d_fc1_w = cache["gap"].T @ d_fc1
    d_fc1_b = d_fc1.sum(axis=0)
    d_gap = d_fc1 @ params["fc1_w"].T                        # (B, F)

    # global-average-pool backward: gradient spreads evenly over the
    # pool_out x pool_out map that was averaged
    d_pooled_max = np.broadcast_to(
        (d_gap / (pool_out * pool_out))[:, :, None, None],
        (B, n_filters, pool_out, pool_out)
    ).copy()

    # route pooling gradient back to the max location in each 2x2 window.
    # cache["pooled"] has axis order (B, F, ph, ih, pw, iw) — we must group
    # (ih, iw) together PER (ph, pw) cell, so transpose before flattening,
    # otherwise the argmax lands on the wrong pixel and training silently
    # learns the wrong function (this is a classic maxpool-backward bug).
    windows = cache["pooled"]                                   # (B,F,ph,ih,pw,iw)
    windows_grouped = windows.transpose(0, 1, 2, 4, 3, 5)        # (B,F,ph,pw,ih,iw)
    win_flat = windows_grouped.reshape(B, n_filters, pool_out, pool_out, 4)
    max_idx = win_flat.argmax(axis=-1)                           # (B,F,ph,pw)
    d_pool_flat = np.zeros_like(win_flat)
    it = np.indices(max_idx.shape)
    d_pool_flat[it[0], it[1], it[2], it[3], max_idx] = d_pooled_max
    d_relu_grouped = d_pool_flat.reshape(B, n_filters, pool_out, pool_out, 2, 2)  # (B,F,ph,pw,ih,iw)
    d_relu_windows = d_relu_grouped.transpose(0, 1, 2, 4, 3, 5)  # back to (B,F,ph,ih,pw,iw)
    d_relu = d_relu_windows.reshape(cache["relu"].shape)

    d_conv = d_relu * (cache["conv"] > 0)

    d_conv_w = np.zeros_like(params["conv_w"])
    d_conv_b = d_conv.sum(axis=(0, 2, 3))
    for b in range(B):
        for f in range(n_filters):
            d_conv_w[f] += correlate2d(cache["X_batch"][b], d_conv[b, f], mode="valid")

    # update
    params["conv_w"] -= lr * (d_conv_w / B)
    params["conv_b"] -= lr * (d_conv_b / B)
    params["fc1_w"]  -= lr * d_fc1_w
    params["fc1_b"]  -= lr * d_fc1_b
    params["fc2_w"]  -= lr * d_fc2_w
    params["fc2_b"]  -= lr * d_fc2_b


def train_cnn(params, X_train, y_train, epochs=35, batch_size=32, lr=0.08):
    n = len(y_train)
    print(f"  Training CNN — {n} tiles, {epochs} epochs, batch {batch_size}, lr {lr}")
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        X_train, y_train = X_train[perm], y_train[perm]
        losses, correct = [], 0
        for start in range(0, n, batch_size):
            xb = X_train[start:start + batch_size]
            yb = y_train[start:start + batch_size]
            probs, cache = cnn_forward(params, xb, keep_cache=True)
            eps = 1e-9
            loss = -np.log(probs[np.arange(len(yb)), yb] + eps).mean()
            losses.append(loss)
            correct += (probs.argmax(axis=1) == yb).sum()
            cnn_backward(params, cache, yb, lr)
        acc = correct / n
        if epoch == 1 or epoch % 3 == 0 or epoch == epochs:
            print(f"    epoch {epoch:>2}/{epochs}  loss={np.mean(losses):.4f}  train_acc={acc:.3f}")
    return params


# =============================================================================
# PART C — LSTM FROM SCRATCH (numpy) — ctDNA next-reading forecaster
# =============================================================================
def init_lstm(hidden_size=16):
    z = hidden_size + 1  # concat(h_prev, x_t) where x_t is a scalar reading
    scale = np.sqrt(2.0 / z)
    params = {}
    for gate in ["f", "i", "c", "o"]:
        params[f"W{gate}"] = rng.normal(0, scale, size=(z, hidden_size)).astype(np.float32)
        params[f"b{gate}"] = np.zeros(hidden_size, dtype=np.float32)
    params["Wy"] = rng.normal(0, np.sqrt(2.0 / hidden_size), size=(hidden_size, 1)).astype(np.float32)
    params["by"] = np.zeros(1, dtype=np.float32)
    params["hidden_size"] = hidden_size
    params["seq_len"] = SEQ_LEN
    return params


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def lstm_forward_one(params, x_seq, keep_cache=False):
    """x_seq: (SEQ_LEN,) one sequence. Returns scalar prediction [+ cache]."""
    H = params["hidden_size"]
    h = np.zeros(H, dtype=np.float32)
    c = np.zeros(H, dtype=np.float32)
    cache_steps = []
    for t in range(len(x_seq)):
        xt = np.array([x_seq[t]], dtype=np.float32)
        z = np.concatenate([h, xt])                     # (H+1,)
        f = sigmoid(z @ params["Wf"] + params["bf"])
        i = sigmoid(z @ params["Wi"] + params["bi"])
        c_hat = np.tanh(z @ params["Wc"] + params["bc"])
        o = sigmoid(z @ params["Wo"] + params["bo"])
        c_new = f * c + i * c_hat
        h_new = o * np.tanh(c_new)
        if keep_cache:
            cache_steps.append(dict(z=z, f=f, i=i, c_hat=c_hat, o=o, c_prev=c, c=c_new, h_prev=h, h=h_new))
        h, c = h_new, c_new
    y = h @ params["Wy"] + params["by"]
    if not keep_cache:
        return y[0]
    return y[0], cache_steps


def lstm_backward_one(params, cache_steps, d_y, grads, h_grad_next=None, c_grad_next=None):
    """Backprop-through-time for ONE sequence; accumulates into `grads` dict."""
    H = params["hidden_size"]
    last = cache_steps[-1]
    grads["Wy"] += np.outer(last["h"], d_y)
    grads["by"] += d_y

    dh_next = last["h"].reshape(1, -1).T @ d_y.reshape(1, -1)
    dh_next = (params["Wy"] @ d_y).astype(np.float32)      # (H,)
    dc_next = np.zeros(H, dtype=np.float32)

    for t in reversed(range(len(cache_steps))):
        s = cache_steps[t]
        dh = dh_next
        dc = dc_next + dh * s["o"] * (1 - np.tanh(s["c"]) ** 2)

        do = dh * np.tanh(s["c"])
        df = dc * s["c_prev"]
        di = dc * s["c_hat"]
        dc_hat = dc * s["i"]

        do_raw = do * s["o"] * (1 - s["o"])
        df_raw = df * s["f"] * (1 - s["f"])
        di_raw = di * s["i"] * (1 - s["i"])
        dc_hat_raw = dc_hat * (1 - s["c_hat"] ** 2)

        for gate, draw in zip(["f", "i", "c", "o"], [df_raw, di_raw, dc_hat_raw, do_raw]):
            grads[f"W{gate}"] += np.outer(s["z"], draw)
            grads[f"b{gate}"] += draw

        dz = (params["Wf"] @ df_raw + params["Wi"] @ di_raw +
              params["Wc"] @ dc_hat_raw + params["Wo"] @ do_raw)
        dh_next = dz[:H]
        dc_next = dc * s["f"]


def zero_grads_like(params):
    return {k: np.zeros_like(v) for k, v in params.items() if isinstance(v, np.ndarray)}


def train_lstm(params, X_train, y_train, epochs=40, lr=0.03, batch_size=16):
    n = len(y_train)
    print(f"  Training LSTM — {n} sequences, {epochs} epochs, batch {batch_size}, lr {lr}")
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        X_train, y_train = X_train[perm], y_train[perm]
        losses = []
        for start in range(0, n, batch_size):
            xb = X_train[start:start + batch_size]
            yb = y_train[start:start + batch_size]
            grads = zero_grads_like(params)
            batch_loss = 0.0
            for seq, target in zip(xb, yb):
                y_pred, cache_steps = lstm_forward_one(params, seq, keep_cache=True)
                err = y_pred - target
                batch_loss += 0.5 * err ** 2
                d_y = np.array([err], dtype=np.float32)
                lstm_backward_one(params, cache_steps, d_y, grads)
            for k in grads:
                params[k] -= lr * grads[k] / len(xb)
            losses.append(batch_loss / len(xb))
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"    epoch {epoch:>2}/{epochs}  mse={np.mean(losses):.4f}")
    return params


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print(" DL ENGINEER — training the CNN (vision) + LSTM (sequence) models")
    print("=" * 70)

    images = np.load(os.path.join(DATA_DIR, "images.npy"))
    labels = np.load(os.path.join(DATA_DIR, "labels.npy"))
    print(f"Loaded {len(labels)} labeled tiles.\n")

    # ---- train/test split (images) ----
    X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
        images, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=labels
    )
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test_img)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test_img)
    print(f"Train tiles: {len(y_train_img)}   Held-out test tiles: {len(y_test_img)} "
          f"(saved to data/X_test.npy, data/y_test.npy for the Evaluation Engineer)\n")

    # ---- Vision model: CNN ----
    print("-" * 70)
    print(" VISION MODEL (CNN)")
    print("-" * 70)
    cnn_params = init_cnn(img_size=images.shape[1])
    cnn_params = train_cnn(cnn_params, X_train_img, y_train_img, epochs=35, batch_size=32, lr=0.08)

    test_probs = cnn_forward(cnn_params, X_test_img)
    test_acc = (test_probs.argmax(axis=1) == y_test_img).mean()
    print(f"  >> CNN held-out test accuracy (quick check): {test_acc:.3f}\n")

    with open(os.path.join(DATA_DIR, "vision_model.pkl"), "wb") as f:
        pickle.dump(cnn_params, f)
    print("  Saved -> data/vision_model.pkl\n")

    # ---- Sequence model: LSTM ----
    print("-" * 70)
    print(" SEQUENCE MODEL (LSTM) — ctDNA next-reading forecaster")
    print("-" * 70)
    X_seq, y_seq = build_sequence_dataset(labels)
    X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=RANDOM_SEED
    )
    lstm_params = init_lstm(hidden_size=16)
    lstm_params = train_lstm(lstm_params, X_seq_train, y_seq_train, epochs=40, lr=0.03, batch_size=16)

    preds = np.array([lstm_forward_one(lstm_params, s) for s in X_seq_test])
    mae = np.abs(preds - y_seq_test).mean()
    print(f"  >> LSTM held-out test MAE (quick check): {mae:.3f} ng/mL\n")

    with open(os.path.join(DATA_DIR, "sequence_model.pkl"), "wb") as f:
        pickle.dump(lstm_params, f)
    print("  Saved -> data/sequence_model.pkl\n")

    print("=" * 70)
    print(" Both models trained and saved. Ready for the Evaluation Engineer. ✅")
    print("=" * 70)


if __name__ == "__main__":
    main()
