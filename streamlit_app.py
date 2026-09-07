import importlib.util
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INTEGRATION_PATH = ROOT / "05_integration_engineer.py"

if not INTEGRATION_PATH.exists():
    raise FileNotFoundError("05_integration_engineer.py not found in the project root.")

spec = importlib.util.spec_from_file_location("integration_engineer", INTEGRATION_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

CLASSES = module.CLASSES
cnn_predict_proba = module.cnn_predict_proba
lstm_predict = module.lstm_predict
make_live_case_tile = module.make_live_case_tile
make_live_case_sequence = module.make_live_case_sequence
build_recommendation = module.build_recommendation


@st.cache_resource
def load_models():
    vision_model_path = DATA_DIR / "vision_model.pkl"
    sequence_model_path = DATA_DIR / "sequence_model.pkl"
    if not vision_model_path.exists() or not sequence_model_path.exists():
        st.error(
            "Model files are missing. Please run the training pipeline first: "
            "python 01_data_engineer.py -> python 02_eda_engineer.py -> python 03_dl_engineer.py"
        )
        st.stop()
    with open(vision_model_path, "rb") as f:
        vision_model = pickle.load(f)
    with open(sequence_model_path, "rb") as f:
        sequence_model = pickle.load(f)
    return vision_model, sequence_model


@st.cache_data
def generate_case():
    return make_live_case_tile(), make_live_case_sequence()


def preprocess_uploaded_image(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        img = Image.open(uploaded_file)
        img = ImageOps.grayscale(img)
        img = img.resize((32, 32), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim != 2 or arr.shape != (32, 32):
            raise ValueError("The uploaded image must resolve to a 32x32 grayscale tile.")
        return arr
    except Exception as exc:
        raise ValueError(f"Unable to read the uploaded image: {exc}") from exc


@st.cache_data
def predict_case(tile, seq, vision_model, sequence_model):
    probs = cnn_predict_proba(vision_model, tile[None, ...])[0]
    pred_idx = int(probs.argmax())
    predicted_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])
    forecast = lstm_predict(sequence_model, seq)
    last_reading = float(seq[-1])
    actions = build_recommendation(predicted_class, confidence, forecast, last_reading)
    return probs, predicted_class, confidence, forecast, last_reading, actions


def render_prediction_panel(tile, seq, vision_model, sequence_model):
    probs, predicted_class, confidence, forecast, last_reading, actions = predict_case(
        tile, seq, vision_model, sequence_model
    )

    col1, col2 = st.columns([1.0, 1.3])
    with col1:
        st.subheader("Pathology tile")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(tile, cmap="gray", vmin=0, vmax=1)
        ax.set_axis_off()
        st.pyplot(fig)

    with col2:
        st.subheader("Prediction")
        st.metric("Predicted class", predicted_class, f"{confidence:.1%} confidence")
        for label, prob in zip(CLASSES, probs):
            st.markdown(f"**{label}**")
            st.progress(float(prob), text=f"{prob:.1%}")

        st.subheader("ctDNA trend")
        x = np.arange(1, len(seq) + 1)
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(x, seq, marker="o", linewidth=2, color="#4C78A8")
        ax2.axvline(len(seq), color="#E45756", linestyle="--", linewidth=1.5)
        ax2.set_title("Recent ctDNA readings")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("ng/mL")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

        st.metric("Last reading", f"{last_reading:.2f} ng/mL")
        st.metric("Next forecast", f"{forecast:.2f} ng/mL")

    st.subheader("AI summary")
    st.write(f"- Image model: {predicted_class} ({confidence:.1%} confidence)")
    st.write(f"- Sequence model: projected next ctDNA reading {forecast:.2f} ng/mL")

    st.subheader("Recommended action")
    for action in actions:
        st.write(f"- {action}")

    st.caption("Educational prototype only. AI output is not a medical diagnosis and requires qualified clinical review.")


def main():
    st.set_page_config(page_title="Oncology DL Dashboard", page_icon="🧬", layout="wide")
    st.title("🧬 Precision Oncology AI Dashboard")
    st.caption("Synthetic pathology tile + ctDNA trajectory demo for the Stage02_DL teaching pipeline.")

    vision_model, sequence_model = load_models()

    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload pathology tile", type=["png", "jpg", "jpeg"])
        generate_new = st.button("Generate a new synthetic case")
        st.markdown("The educational prototype does not diagnose disease and requires expert review.")

    if uploaded_file is not None:
        try:
            uploaded_tile = preprocess_uploaded_image(uploaded_file)
            st.success("Uploaded image loaded successfully.")
            uploaded_seq = np.array([
                st.number_input("Reading 1", value=1.2, format="%.2f"),
                st.number_input("Reading 2", value=1.8, format="%.2f"),
                st.number_input("Reading 3", value=2.6, format="%.2f"),
                st.number_input("Reading 4", value=3.5, format="%.2f"),
                st.number_input("Reading 5", value=4.3, format="%.2f"),
            ], dtype=np.float32)
            render_prediction_panel(uploaded_tile, uploaded_seq, vision_model, sequence_model)
        except ValueError as exc:
            st.error(str(exc))
    else:
        if generate_new:
            tile, seq = generate_case()
            render_prediction_panel(tile, seq, vision_model, sequence_model)
        else:
            tile, seq = generate_case()
            render_prediction_panel(tile, seq, vision_model, sequence_model)


if __name__ == "__main__":
    main()
