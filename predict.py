import sys
import numpy as np
import tensorflow as tf
from utils.preprocessing import preprocess_single_image

MODEL_PATH = "model/tb_model.h5"

def predict_xray(image_path):
    """
    Simulates sending a single image to the AI to evaluate if it displays Tuberculosis.
    Returns: string 'TB' / 'Normal', float probability
    """
    # 1. Load the model
    # (Suppress tf warnings mostly to keep CLI output clean for beginners)
    print(f"Loading AI Model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure the model has finished training.")
        return

    # 2. Preprocess
    print("Formatting image size for Neural Network input...")
    try:
        img_array = preprocess_single_image(image_path)
    except Exception as e:
        print(f"Could not load image: {e}")
        return

    # 3. Predict Output
    print("Asking AI for prediction...")
    raw_pred = model.predict(img_array, verbose=0)[0]
    print(f"[DEBUG] Raw prediction array from model: {raw_pred}")
    
    # Handle both categorical (softmax) or binary (sigmoid) models
    if len(raw_pred) == 2:
        # Categorical: index 0 is Normal, index 1 is TB
        prob_tb = float(raw_pred[1])
        prediction_prob = prob_tb
    else:
        # Binary: 0 is Normal, 1 is TB
        prediction_prob = float(raw_pred[0])
        
    print(f"[DEBUG] Interpreted TB Class Probability: {prediction_prob:.4f}")
    
    # 4. Parse Results
    if prediction_prob >= 0.5:
        print("\n=== AI CLASSIFICATION RESULT ===")
        print("Prediction: Tuberculosis")
        print(f"Confidence Level: {prediction_prob * 100:.2f}%")
        return "TB", prediction_prob
    else:
        print("\n=== AI CLASSIFICATION RESULT ===")
        print("Prediction: Normal")
        print(f"Confidence Level: {(1 - prediction_prob) * 100:.2f}%")
        return "Normal", prediction_prob

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage error. Please provide path to X-Ray Image.")
        print("Example: python predict.py my_xray.png")
        sys.exit(1)
    
    img_path = sys.argv[1]
    predict_xray(img_path)
