<<<<<<< HEAD
# Radiomics and Machine Learning based Automated Tuberculosis Detection from Chest X-Ray Images

## Project Overview

This project is an end-to-end Machine Learning pipeline designed for detecting Tuberculosis (TB) from Chest X-ray images. Built for a mini-project, it leverages **Radiomics** feature extraction (shape, texture, and statistical metrics) combined with a highly accurate classic **Machine Learning Classifier (XGBoost/Random Forest)**.

Unlike traditional "black-box" Deep Learning models, this project uses **Explainable AI (SHAP - SHapley Additive exPlanations)** to generate clear waterfall graphs showing exactly which physical textures and features of the lung X-ray led to the AI's diagnosis.

The application includes an interactive Streamlit-based web frontend for an optimal and simple user experience.

## Features

- **Radiomics Extraction (`pyradiomics`)**: Converts medical images into 100+ mathematical features describing complex textures (GLCM, GLRLM), intensity, and shapes.
- **Machine Learning Architecture**: Utilizes XGBoost/Decision Trees trained strictly on the extracted tabular mathematical features, preventing overfitting seen in small CNN deep-learning datasets.
- **Explainable AI (SHAP)**: Waterfall plots visualize the positive/negative impact of individual texture/shape features on the final diagnosis, offering high clinical interpretability.
- **Evaluation Utilities**: Automated generation of Confusion Matrices and Classification Reports (Precision, Recall, F1-Score).
- **Responsive Web UI**: A beautiful, minimalist Streamlit application designed for clean demonstration.

---

## Project Architecture

```text
TB_Detection_Project/
│
├── dataset/
│   ├── Normal/                  # Add your normal patient X-ray images here
│   ├── TB/                      # Add your TB patient X-ray images here
│   └── radiomics_features.csv   # Automatically generated dataset containing extracted math features
│
├── model/
│   └── tb_ml_model.pkl          # Your automatically saved trained Machine Learning Model & Preprocessor Scaler
│
├── notebooks/                   # (Optional) For EDA or Jupyter experimental workflows
│
├── utils/
│   ├── radiomics_extractor.py   # Uses PyRadiomics to get texture/shape values of X-Rays
│   └── explain_ml.py            # Generates the SHAP Explainable AI Waterfall plots
│
├── extract_features.py          # Step 1: Iterates dataset/ and extracts features to CSV file
├── train_model_improved.py      # Step 2: Trains XGBoost ML model on the CSV datasets
├── evaluate_model.py            # Step 3: Generates metric reports and saves a Confusion Matrix Image
├── app.py                       # Step 4: Streamlit Web Server application for the Visual Demo
├── requirements.txt             # Python Library Dependencies
└── README.md                    # You are reading this!
```

---

## 🛠️ Step-by-Step Execution Guide (VS Code)

### Step 1: Prepare the Environment

1. Open up VS Code and ensure you are in the project folder.
2. Open the terminal in VS Code (`Ctrl + ~` or Terminal > New Terminal).
3. **Activate your virtual environment** (Highly recommended):
   - **Windows:**
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - **Mac/Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
4. Install all the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```

### Step 2: Prepare Your Dataset

Before running code, setup the physical X-Ray images.

1. Download a "Chest X-Ray TB Dataset" (e.g., from Kaggle).
2. Move all your "Normal" X-ray images into `dataset/Normal/`.
3. Move all your "Tuberculosis" X-ray images into `dataset/TB/`.

### Step 3: Extract Radiomics Features

We must first mathematically extract all the textures and shapes from the images into a `CSV` file. Run:
_(This will take a few minutes as it iterates through all images)_

```bash
python extract_features.py
```

### Step 4: Train the ML Model (XGBoost)

Once `radiomics_features.csv` is created, train your XGBoost Classifier. It will automatically scale features and save the finalized model to `model/tb_ml_model.pkl`.

```bash
python train_model_improved.py
```

### Step 5: Evaluate the AI's Accuracy

Run the evaluation script to test the model against data, print a classification report (Accuracy, Precision, Recall), and generate a graphical map (`confusion_matrix.png`).

```bash
python evaluate_model.py
```

### Step 6: Launch the Web Application Demo

Run the user-friendly Streamlit web app. This starts a local server and automatically opens it in your default browser.

```bash
streamlit run app.py
```

_Upload an image via the Browser UI, click "**Analyze X-ray**", and observe the ML Prediction alongside the generated Explainable **SHAP** Importance graph!_
=======
# TB
>>>>>>> 4b770622534601dbcc7955bb286eb0129a2dc251
