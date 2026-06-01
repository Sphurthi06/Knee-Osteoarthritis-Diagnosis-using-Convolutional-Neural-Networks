# Knee Osteoarthritis Severity Detection

A deep learning web application that classifies the severity of knee osteoarthritis from X-ray images, with visual explainability using Grad-CAM.

**92% diagnostic accuracy** using EfficientNetB4 across 5 KL-grade severity classes.

---

## What it does

Upload a knee X-ray image and the app will:
- Predict the osteoarthritis severity grade (Healthy → Severe)
- Show confidence percentage for the prediction
- Display a **Grad-CAM heatmap** highlighting the regions of the X-ray that influenced the prediction
- Show a probability breakdown across all 5 classes

---

## Severity Classes (Kellgren-Lawrence Grading)

| Grade | Label | Description |
|-------|-------|-------------|
| 0 | Healthy | No osteoarthritis |
| 1 | Doubtful | Possible narrowing |
| 2 | Minimal | Definite narrowing |
| 3 | Moderate | Multiple osteophytes |
| 4 | Severe | Large osteophytes, severe narrowing |

---

## Models Compared

| Model | Accuracy |
|-------|----------|
| **EfficientNetB4** | **92%** ✅ Best |
| Xception | - |
| InceptionV3 | - |

EfficientNetB4 was selected as the final model based on validation accuracy.

---

## Tech Stack

- **Python 3.9**
- **TensorFlow 2.10**
- **Streamlit** — web application
- **Grad-CAM** — explainability / heatmap visualisation
- **Matplotlib, NumPy, Pandas, Scikit-learn**

---

## Project Structure

```
├── src/
│   └── models/
│       └── efficientnetb4(1).hdf5       # Trained model weights
├── app/
│   └── img/                             # App assets
├── EfficientNetB4__final_.ipynb         # Final model training notebook
├── xception_final.ipynb                 # Xception comparison
├── test_inceptionv3final__1_.ipynb      # InceptionV3 comparison
├── app.py                               # Streamlit web app
├── environment.yml                      # Conda environment
└── README.md
```

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

---

## Dataset

X-ray images sourced from the [Knee Osteoarthritis Dataset on Kaggle](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity).  
Images are graded using the Kellgren-Lawrence (KL) scale.

---

## Results

The EfficientNetB4 model achieved **92% accuracy** on the test set, outperforming Xception and InceptionV3 architectures tested during model selection.

The Grad-CAM visualisation confirms the model focuses on clinically relevant regions of the knee joint — joint space narrowing and osteophyte formation areas.