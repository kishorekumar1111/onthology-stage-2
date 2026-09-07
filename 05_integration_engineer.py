"""
================================================================================
 05_INTEGRATION_ENGINEER.PY
--------------------------------------------------------------------------------
 ROLE IN THE PIPELINE  (see Day-2 lecture, Slide 21 "MEET THE ROLE: INTEGRATION ENGINEER")

   Problem   : Can an oncologist open a Jupyter notebook and read raw CNN
               output during a tumor board meeting? NO.
   Job       : Connect the trained CNN + LSTM models to a usable
               application / dashboard that turns a raw prediction into a
               clear recommended action (Slide 21's dashboard mock-up).

 WHAT THIS SCRIPT DOES
   1. Loads BOTH trained models:
        data/vision_model.pkl    (CNN  -> tile classification)
        data/sequence_model.pkl  (LSTM -> ctDNA trend forecast)
   2. Simulates a NEW incoming case exactly like the lecture's live-case
      slide (Slide 23): a faded-stain tile with dense, overlapping cells,
      plus a ctDNA series that is sharply rising.
   3. Runs both models and COMBINES their outputs into one recommendation,
      the same way Slide 11 describes: "vision confirms the current tumor
      state, the sequence model predicts the trajectory."
   4. Renders the result as:
        (a) a clean text dashboard printed to the console, and
        (b) a small standalone HTML dashboard file (dashboard.html) that
            could be opened in a browser / embedded in a clinical tool —
            this is the "usable application" the Integration Engineer ships.

 Run:  python 05_integration_engineer.py
================================================================================
"""

import os
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d

CLASSES = ["Benign", "Atypical", "Malignant"]
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RANDOM_SEED = 7
rng = np.random.default_rng(RANDOM_SEED)

CONFIDENCE_FLAG_THRESHOLD = 0.70   # above this -> flag for oncologist review


# --------------------------------------------------------------------------
# Forward-only CNN inference (mirrors 03_dl_engineer.py's architecture)
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
    return exp / exp.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Forward-only LSTM inference (mirrors 03_dl_engineer.py's architecture)
# --------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def lstm_predict(params, x_seq):
    H = params["hidden_size"]
    h = np.zeros(H, dtype=np.float32)
    c = np.zeros(H, dtype=np.float32)
    for t in range(len(x_seq)):
        xt = np.array([x_seq[t]], dtype=np.float32)
        z = np.concatenate([h, xt])
        f = sigmoid(z @ params["Wf"] + params["bf"])
        i = sigmoid(z @ params["Wi"] + params["bi"])
        c_hat = np.tanh(z @ params["Wc"] + params["bc"])
        o = sigmoid(z @ params["Wo"] + params["bo"])
        c = f * c + i * c_hat
        h = o * np.tanh(c)
    y = h @ params["Wy"] + params["by"]
    return float(y[0])


# --------------------------------------------------------------------------
# Simulate ONE new incoming case (mirrors Slide 23's "Live Case")
# --------------------------------------------------------------------------
def make_live_case_tile(img_size=32):
    """Faded stain, dense/overlapping cells -> visually ambiguous-but-severe tile."""
    base = rng.normal(0.75, 0.04, size=(img_size, img_size))
    base = gaussian_filter(base, sigma=1.4)
    yy, xx = np.ogrid[:img_size, :img_size]
    for _ in range(24):                                # dense nuclei
        cy, cx = rng.integers(2, img_size - 2, size=2)
        r = rng.integers(2, 5)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        base[mask] -= rng.uniform(0.30, 0.55)
    base += rng.normal(0, 0.05, size=base.shape)
    base = gaussian_filter(base, sigma=1.2)             # faded / scanner blur
    return np.clip(base, 0, 1).astype(np.float32)


def make_live_case_sequence(seq_len=5):
    """Sharp ctDNA rise, like the R-4..Now example on Slide 10."""
    base = rng.uniform(1.0, 1.4)
    series = [base]
    for _ in range(seq_len - 1):
        series.append(series[-1] + rng.uniform(0.55, 0.9) + rng.normal(0, 0.08))
    return np.array(series, dtype=np.float32)


# --------------------------------------------------------------------------
# Recommendation logic (Slide 21's dashboard mock)
# --------------------------------------------------------------------------
def build_recommendation(predicted_class, confidence, forecast, last_reading):
    actions = []
    if predicted_class == "Malignant" and confidence >= CONFIDENCE_FLAG_THRESHOLD:
        actions.append("Flag case for oncologist review")
        actions.append("Recommend biopsy confirmation")
    elif predicted_class == "Atypical":
        actions.append("Flag for EDA Engineer saliency review (borderline case)")
        actions.append("Recommend short-interval re-scan")
    else:
        actions.append("Routine follow-up per standard protocol")

    if forecast > last_reading * 1.15:
        actions.append("ctDNA trend rising — schedule biomarker recheck")

    if confidence < CONFIDENCE_FLAG_THRESHOLD:
        actions.append(f"Low model confidence ({confidence:.0%}) — send to human pathologist review")

    return actions


# --------------------------------------------------------------------------
# Dashboard rendering
# --------------------------------------------------------------------------
def print_console_dashboard(predicted_class, confidence, probs, last_reading, forecast, actions):
    W = 62
    def line(txt=""):
        print("│ " + txt.ljust(W - 4) + " │")

    print("┌" + "─" * (W - 2) + "┐")
    line("LIVE ONCOLOGY COMMAND DASHBOARD")
    print("├" + "─" * (W - 2) + "┤")
    line(f"New Pathology Tile  ->  {predicted_class.upper()}  ({confidence:.0%} confidence)")
    line("Class probabilities:")
    for c, p in zip(CLASSES, probs):
        bar = "█" * int(p * 30)
        line(f"  {c:<10} {p:>6.1%}  {bar}")
    line()
    line(f"ctDNA — last reading   : {last_reading:.2f} ng/mL")
    line(f"ctDNA — forecast (next): {forecast:.2f} ng/mL "
         f"({'rising' if forecast > last_reading else 'stable/falling'})")
    print("├" + "─" * (W - 2) + "┤")
    line("RECOMMENDED ACTION:")
    for a in actions:
        line(f"  • {a}")
    print("└" + "─" * (W - 2) + "┘")


def write_html_dashboard(predicted_class, confidence, probs, last_reading, forecast, actions, path):
    color = {"Benign": "#4C956C", "Atypical": "#E8A87C", "Malignant": "#C1666B"}[predicted_class]
    bars = "".join(
        f'<div style="margin:4px 0"><span style="display:inline-block;width:90px">{c}</span>'
        f'<span style="display:inline-block;background:#eee;width:200px;height:14px;vertical-align:middle">'
        f'<span style="display:inline-block;background:{color};width:{p*200:.0f}px;height:14px"></span></span>'
        f' {p:.1%}</div>'
        for c, p in zip(CLASSES, probs)
    )
    actions_html = "".join(f"<li>{a}</li>" for a in actions)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Live Oncology Command Dashboard</title></head>
<body style="font-family: -apple-system, Arial, sans-serif; max-width: 560px; margin: 40px auto;
             padding: 24px; border: 1px solid #ddd; border-radius: 8px;">
  <h2>Live Oncology Command Dashboard</h2>
  <p style="font-size: 22px;">Prediction:
     <b style="color:{color}">{predicted_class}</b> ({confidence:.0%} confidence)</p>
  {bars}
  <hr>
  <p>ctDNA last reading: <b>{last_reading:.2f} ng/mL</b><br>
     ctDNA forecast (next): <b>{forecast:.2f} ng/mL</b>
     ({'rising' if forecast > last_reading else 'stable/falling'})</p>
  <hr>
  <h3>Recommended Action</h3>
  <ul>{actions_html}</ul>
  <p style="color:#888; font-size:12px;">Generated by the Integration Engineer stage of the
     Stage02_DL pipeline. Synthetic demo data — not a real clinical decision.</p>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" INTEGRATION ENGINEER — connecting the trained models to a live dashboard")
    print("=" * 70)

    with open(os.path.join(DATA_DIR, "vision_model.pkl"), "rb") as f:
        vision_model = pickle.load(f)
    with open(os.path.join(DATA_DIR, "sequence_model.pkl"), "rb") as f:
        sequence_model = pickle.load(f)
    print("Loaded vision_model.pkl and sequence_model.pkl.\n")

    tile = make_live_case_tile()
    seq = make_live_case_sequence()

    probs = cnn_predict_proba(vision_model, tile[None, ...])[0]
    pred_idx = int(probs.argmax())
    predicted_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    forecast = lstm_predict(sequence_model, seq)
    last_reading = float(seq[-1])

    actions = build_recommendation(predicted_class, confidence, forecast, last_reading)

    print_console_dashboard(predicted_class, confidence, probs, last_reading, forecast, actions)

    html_path = os.path.join(HERE, "dashboard.html")
    write_html_dashboard(predicted_class, confidence, probs, last_reading, forecast, actions, html_path)
    print(f"\nSaved -> {os.path.relpath(html_path, HERE)}  (open in a browser)")

    print("\n" + "=" * 70)
    print(" This is the payoff moment all five roles built toward:")
    print(" Data Engineer -> EDA Engineer -> DL Engineer -> Evaluation Engineer")
    print(" -> Integration Engineer -> a decision an oncologist can actually use.")
    print("=" * 70)


if __name__ == "__main__":
    main()
