# Knee Osteoarthritis Severity Detection

A deep learning web application that automates the diagnosis of knee osteoarthritis (KOA) severity from X-ray images — replacing a manual, error-prone process that typically requires a trained physician and significant time.

**EfficientNetB4 achieved 88% accuracy** across 5 KL-grade severity classes, outperforming Xception and InceptionResNetV3 in comparative evaluation.

---

## The Problem

Manual diagnosis of knee osteoarthritis involves a physician examining X-ray images and grading severity using the Kellgren–Lawrence (KL) system. This process requires specialist expertise, is time-consuming, and is prone to human error. This project automates that classification using deep learning.

---

## What the App Does

Upload a knee X-ray and the Streamlit app will:
- Predict the osteoarthritis severity grade across 5 classes
- Display confidence percentage for the prediction
- Generate a **Grad-CAM heatmap** highlighting which regions of the X-ray influenced the model's decision
- Show a full probability breakdown across all severity grades

---

## Severity Classes (Kellgren-Lawrence Grading)

| Grade | Label | Description |
|-------|-------|-------------|
| 0 | Healthy | No signs of osteoarthritis |
| 1 | Doubtful | Possible joint space narrowing |
| 2 | Minimal | Definite narrowing, possible osteophytes |
| 3 | Moderate | Multiple osteophytes, definite narrowing |
| 4 | Severe | Large osteophytes, severe narrowing |

---

## Dataset

- **Source:** Osteoarthritis Initiative (OAI) dataset via [Kaggle](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity)
- **Total images:** 8,000 pre-classified X-ray images
- **Classes:** 5 (Healthy, Doubtful, Minimal, Moderate, Severe)
- **Augmentation:** Brightness adjustment, width shift, horizontal flip, zoom range

---

## Models Compared

| Model | Result |
|-------|--------|
| **EfficientNetB4** | **88% accuracy ✅ Best** |
| InceptionResNetV3 | Lower accuracy |
| XceptionNet | Lower accuracy |

EfficientNetB4 was selected as the final model based on accuracy and evaluation metrics across all 5 classes.

---

## Tech Stack

- **Python 3.9**
- **TensorFlow 2.10**
- **Streamlit** — interactive web application
- **Grad-CAM** — visual explainability / heatmap overlay
- **Matplotlib, NumPy, Pandas, Scikit-learn**

---

## Setup & Run

**1. Clone the repo**
```bash
git clone https://github.com/Sphurthi06/Knee-Osteoarthritis-Diagnosis-using-Convolutional-Neural-Networks.git
cd Knee-Osteoarthritis-Diagnosis-using-Convolutional-Neural-Networks
```

**2. Create the conda environment**
```bash
conda env create -f environment.yml
conda activate "knee osteoarthritis diagnosis"
```

**3. Run the app**
```bash
streamlit run app.py
```

> **Note:** The trained model file (`efficientnetb4(1).hdf5`) is not included in the repo due to file size. You can retrain using `EfficientNetB4__final_.ipynb` or download the weights separately.

---

## Results

EfficientNetB4 achieved **88% test accuracy** on the OAI dataset, evaluated using accuracy, loss, balanced accuracy score, and per-class validation metrics.

The Grad-CAM visualisation confirms the model focuses on clinically relevant regions — joint space narrowing and osteophyte formation areas — consistent with how a physician would interpret the X-ray.
