# Final QA Report — Stage02_DL

## 1) Executive Summary

This project was audited as a working synthetic deep-learning teaching pipeline rather than a production clinical system. The evidence shows that the full five-stage workflow runs successfully in order, the CNN and LSTM train and save model artifacts, and the Streamlit dashboard loads and renders predictions in a browser.

Important limitation: the project is deliberately synthetic and educational. It is not a real pathology or biomarker pipeline, and it should not be interpreted as a clinical decision-support tool.

## 2) Project Inventory

Repository contents verified in the workspace:

- [01_data_engineer.py](01_data_engineer.py) — synthetic dataset generation, cleaning, augmentation, and save to data/
- [02_eda_engineer.py](02_eda_engineer.py) — image quality and class-structure checks
- [03_dl_engineer.py](03_dl_engineer.py) — custom CNN + custom LSTM training, plus held-out test export
- [04_evaluation_engineer.py](04_evaluation_engineer.py) — unseen-data evaluation and overconfidence check
- [05_integration_engineer.py](05_integration_engineer.py) — console + HTML dashboard generation
- [streamlit_app.py](streamlit_app.py) — Streamlit web dashboard
- [README.md](README.md) — project documentation and scientific disclaimer
- [dashboard.html](dashboard.html) — generated integration artifact
- [data/images.npy](data/images.npy), [data/labels.npy](data/labels.npy), [data/X_test.npy](data/X_test.npy), [data/y_test.npy](data/y_test.npy), [data/vision_model.pkl](data/vision_model.pkl), [data/sequence_model.pkl](data/sequence_model.pkl) — generated pipeline artifacts

## 3) Verified Runtime Evidence

I validated the pipeline by running the full sequence in order:

1. `python 01_data_engineer.py`
2. `python 02_eda_engineer.py`
3. `python 03_dl_engineer.py`
4. `python 04_evaluation_engineer.py`
5. `python 05_integration_engineer.py`

Fresh output confirmed:

- Final cleaned dataset size: 1902 tiles
- Class counts: Benign 630, Atypical 621, Malignant 651
- Training run produced CNN held-out test accuracy: 0.816
- Sequence model held-out MAE: 0.075 ng/mL
- Evaluation report saved to [eval_confusion_matrix.png](eval_confusion_matrix.png)
- Integration stage generated [dashboard.html](dashboard.html)

## 4) Data Audit

### 4.1 Synthetic labeling and dataset structure

The dataset was produced by synthetic generation, then cleaned and augmented. Verified shapes and types:

- `images.npy`: shape (1902, 32, 32), dtype float32
- `labels.npy`: shape (1902,), dtype int64
- `X_test.npy`: shape (381, 32, 32)
- `y_test.npy`: shape (381,)

Class balance check:

- Benign: 630 (33.1%)
- Atypical: 621 (32.6%)
- Malignant: 651 (34.2%)

This is close to balanced and acceptable for a teaching demo.

### 4.2 Data cleaning and augmentation

The cleaning step removes corrupted tiles and duplicates using a synthetic raw-lake workflow. This is consistent with the project purpose and the README requirement for a realistic but pre-cleaned dataset.

### 4.3 Data leakage review

No obvious leakage issue was identified in the core train/test split strategy:

- The image test set is explicitly saved and held out by [03_dl_engineer.py](03_dl_engineer.py)
- [04_evaluation_engineer.py](04_evaluation_engineer.py) tests only on `X_test.npy` and `y_test.npy`
- The model is not evaluated on training tiles

The synthetic generation process itself is not leakage in the legal/clinical sense; it is simply a simulated dataset with no real patient records.

## 5) Engineering Audit

### 5.1 Model architecture

The project implements the expected teaching architecture:

- CNN: convolution, ReLU, max pooling, global average pooling, fully connected head, softmax
- LSTM: single recurrent layer for next-reading forecasting

This matches the narrative in [README.md](README.md) and the code comments in [03_dl_engineer.py](03_dl_engineer.py).

### 5.2 Training quality

Verified training output:

- CNN final train accuracy: 0.909
- Quick hold-out test accuracy: 0.816

For a tiny synthetic dataset, this is a healthy result for a classroom project. It demonstrates the model learned a meaningful signal instead of pure memorization.

### 5.3 Saved model artifacts

The project saves valid pickled model dictionaries containing NumPy arrays. The behavior is consistent with the code, and the model files load correctly in both the integration and Streamlit scripts.

## 6) EDA Audit

The EDA stage checks the expected signals:

- class distribution is roughly balanced
- low-quality tiles are flagged by Laplacian sharpness
- density and texture metrics increase monotonically across Benign -> Atypical -> Malignant

Verified output from [02_eda_engineer.py](02_eda_engineer.py):

- Mean texture density score: Benign 0.00005, Atypical 0.00037, Malignant 0.00170
- Monotonic increase was reported as True

This is a valid sanity check for trainability in a synthetic setting.

## 7) Evaluation and Error Analysis

The hold-out evaluation in [04_evaluation_engineer.py](04_evaluation_engineer.py) reports:

- Accuracy: 0.816
- Precision / recall / F1 look sensible for the synthetic classes
- Most common confusion: true Benign predicted as Atypical
- Overconfident wrong predictions: 12 / 381

This is an important result: the code is doing the right thing by checking for false certainty. For a system intended to help with human review, this is the correct risk metric to monitor.

### 7.1 Overconfidence findings

The tool output confirms examples of confident misclassifications, including malignant tiles predicted as atypical with ~88-92% confidence. This is a genuine risk signal and should be treated as a warning, not as a hidden code defect.

This does not mean the pipeline is broken. It means the project explicitly surfaces the most dangerous failure mode in a clinical context.

## 8) Multimodal / Integration Status

The project is not a true multi-modal clinical fusion model in the strict sense. It contains:

- a vision branch for pathology tile classification
- a sequence branch for ctDNA forecasting

These are combined in the integration layer into a recommendation, but the model is still a synthetic educational simulation. It should be described as a paired-decision demo, not a validated clinical multimodal AI.

## 9) Streamlit / UI Audit

The Streamlit app in [streamlit_app.py](streamlit_app.py) was started and accessed successfully at http://localhost:8501.

Verified page state:

- Page title: Oncology DL Dashboard
- The main dashboard rendered successfully
- The generated synthetic case view displayed a pathology tile, prediction probabilities, ctDNA plot, and recommendation list
- The dashboard includes educational wording clarifying that outputs are not medical diagnoses

This confirms the app is functional and connected to the trained model artifacts.

## 10) Dependency and Environment Audit

The project is intentionally lightweight and relies mainly on:

- NumPy
- SciPy
- scikit-learn
- matplotlib
- Pillow
- Streamlit

This is consistent with the project documentation in [README.md](README.md). No substantial dependency issue was found during runtime validation.

## 11) Genuine Issues Found and Repairs

One issue was fixed in the integration pipeline:

- HTML dashboard writing needed UTF-8 encoding to avoid Windows encoding incompatibilities when writing generated content.
- The current code in [05_integration_engineer.py](05_integration_engineer.py) correctly opens the output with `encoding="utf-8"`.

No additional critical code defect was identified in the final project state after validation.

## 12) Remaining Risks / Non-Blocking Limitations

These are remaining scientific and product limitations, not code failures:

1. All data are synthetic; no real pathology or biomarker data are present.
2. The model is trained on synthetic patterns and should not be treated as a medical-grade classifier.
3. The project is educational and meant to demonstrate architecture and workflow, not deployable clinical logic.
4. The model's false-confidence problem is real and should be addressed with human review and stronger validation on real data before any real-world use.
5. The sequence forecast is a simple, synthetic trend model and not a clinically realistic biomarker predictive model.

## 13) Final Verdict

Status: PASS for the project objective as a synthetic educational pipeline.

Evidence-based conclusion:

- The full pipeline runs end-to-end
- Data artifacts are generated correctly
- Models train and save successfully
- Held-out evaluation passes at 0.816 accuracy
- Streamlit dashboard displays and functions correctly
- The project clearly documents its synthetic and educational nature

Status: FAIL if the expectation is a real clinical diagnostic system.

This project is not a production-grade medical AI; it is a working teaching prototype that successfully demonstrates the flow from synthetic data creation to model training, evaluation, and dashboard decision support.

## 14) Final QA Decision

- Quality gate: PASS for educational/synthetic demonstration objectives
- Clinical readiness: NOT READY
- Production readiness: NOT READY
- Recommended next step: replace synthetic data and scoring logic with IRB-approved real data, formal validation, and human-in-the-loop review before deployment
