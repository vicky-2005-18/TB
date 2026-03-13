import streamlit as st
st.set_page_config(page_title="AI TB Detection Dashboard", layout="wide", page_icon="🩺")
import os
import pandas as pd
import pickle
import xgboost as xgb
import glob
from utils.radiomics_extractor import extract_radiomics_features
from utils.explain_ml import generate_shap_plot

# App Configuration
MODEL_PATH = os.path.join("model", "tb_ml_model.pkl")
XGB_MODEL_PATH = os.path.join("model", "tb_xgboost_model.json")


# Load model pipeline
@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        p = pickle.load(f)
    
    # If model is not in pickle, load it natively
    if 'model' not in p:
        model = xgb.XGBClassifier()
        if os.path.exists(XGB_MODEL_PATH):
            model.load_model(XGB_MODEL_PATH)
            p['model'] = model
    return p

pipeline = load_pipeline()

# Custom CSS for beauty
st.markdown("""
<style>
    .result-box-positive {
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        background-color: rgba(255, 75, 75, 0.1);
        margin-top: 15px;
    }
    .result-box-negative {
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #09ab3b;
        background-color: rgba(9, 171, 59, 0.1);
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Build the Layout
st.sidebar.title("⚙️ System Instructions")

app_mode = st.sidebar.radio("Select Mode", ["Single Diagnosis", "Batch Evaluation (Metrics)"])

if app_mode == "Single Diagnosis":
    # Main Dashboard Title
    st.markdown("<h1 style='text-align: center;'>AI-Based Tuberculosis Detection from Chest X-rays</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1em; color: gray;'>A Radiomics and Machine Learning based diagnostic tool using XGBoost and SHAP</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.info("""
    **Steps:**
    1. Upload a clear High Resolution Chest X-ray.
    2. Click **'Analyze X-ray'**.
    3. View Prediction and Artificial Intelligence Heatmap (SHAP).
    """)

    col1, col2, col3 = st.columns([1, 1, 1], gap="large")

    with col1:
        st.write("### 📤 Image Upload")
        uploaded_file = st.file_uploader("Drop your medical image here", type=["jpg", "png", "jpeg"], help="Supported formats: JPG, PNG, JPEG")
        
        if uploaded_file is not None:
            # Save securely locally to prepare for PyRadiomics extraction 
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            img_path = os.path.join(temp_dir, uploaded_file.name)
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.image(uploaded_file, caption='🔍 Original Uploaded X-ray', use_container_width=True)

    with col2:
        st.write("### 📊 Prediction Result")
        analyze_placeholder = st.empty()
        analyze_placeholder.info("Awaiting image upload to proceed with AI analysis.")
        
        analyze_button = False
        if uploaded_file is not None:
            analyze_placeholder.empty()
            analyze_button = st.button("🔍 Analyze X-ray", type="primary", use_container_width=True)

    with col3:
        st.write("### 🧠 Explainable AI (SHAP)")
        shap_placeholder = st.empty()
        shap_placeholder.info("Upload an image and click **'Analyze X-ray'** to view the AI interpretability analysis here.")

    if uploaded_file is not None and analyze_button:
        with col2:
            with st.spinner('Extracting Radiomics and Processing ML...'):
                
                # 1. Extract Radiomic Features from user image
                features = extract_radiomics_features(img_path)
                
                # 2. Extract SHAP Explanation alongside Prediction
                cam_path = os.path.join(temp_dir, "shap_" + uploaded_file.name + ".png")
                prob = generate_shap_plot(features, MODEL_PATH, cam_path)
                
                # Formatted Prediction Result
                st.write("---")
                if prob >= 0.5:
                    st.markdown(f"""
                    <div class="result-box-positive">
                        <h3 style="color: #ff4b4b; margin:0px;">🚨 Tuberculosis Detected</h3>
                        <p style="margin:0px; font-size: 16px;"><b>Confidence: {prob * 100:.2f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box-negative">
                        <h3 style="color: #09ab3b; margin:0px;">✅ Normal Chest X-ray</h3>
                        <p style="margin:0px; font-size: 16px;"><b>Confidence: {(1 - prob) * 100:.2f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col3:
            shap_placeholder.empty()
            st.image(cam_path, caption='SHAP Feature Importance (Waterfall)', use_container_width=True)
            
            # Educational Box for user clarity
            st.info("""
            **App Interpretations:**
            - **Radiomics/GLCM Extraction**: 18 texture (GLCM) and statistical data points are measured from the image physically using Scikit-Image.
            - **XGBoost ML**: Uses decision trees mapped against thousands of previous cases to identify risk patterns.
            - **SHAP Explanation Chart**: Shows *why* the AI made the decision. Red indicates a high-value presence of a radiomic feature (like uneven rough textures) acting to push the decision towards "TB". Blue shows values pushing towards "Normal".
            """)

elif app_mode == "Batch Evaluation (Metrics)":
    # Main Dashboard Title
    st.markdown("<h1 style='text-align: center;'>AI-Based Tuberculosis Detection from Chest X-rays</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1em; color: gray;'>Batch Evaluation & Performance Metrics</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.info("""
    1. Enter a valid path to an image dataset folder (e.g., `dataset/test`).
    2. The folder MUST contain exactly two subfolders: `Normal` and `TB`.
    3. Click **'Evaluate Metrics'**.
    """)
    st.write("### 📊 Batch Evaluation")
    eval_dir = st.text_input("Enter local directory path to evaluate:", value="dataset/test", help="Must contain 'Normal' and 'TB' subfolders")
    
    if st.button("Evaluate Metrics", type="primary"):
        if not os.path.exists(eval_dir):
            st.error(f"Directory '{eval_dir}' does not exist.")
        elif not os.path.exists(os.path.join(eval_dir, "Normal")) or not os.path.exists(os.path.join(eval_dir, "TB")):
            st.error(f"Directory '{eval_dir}' must contain exactly 'Normal' and 'TB' subfolders with images inside.")
        else:
            with st.spinner("Processing Images and Evaluating Model... This might take a minute."):
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                import numpy as np
                
                # Fetch all images
                normal_imgs = glob.glob(os.path.join(eval_dir, "Normal", "*.*"))
                tb_imgs = glob.glob(os.path.join(eval_dir, "TB", "*.*"))
                
                # Filter true images
                valid_ext = ('.png', '.jpg', '.jpeg')
                normal_imgs = [img for img in normal_imgs if img.lower().endswith(valid_ext)]
                tb_imgs = [img for img in tb_imgs if img.lower().endswith(valid_ext)]
                
                all_imgs = normal_imgs + tb_imgs
                y_true = [0] * len(normal_imgs) + [1] * len(tb_imgs) # 0 for Normal, 1 for TB
                
                if len(all_imgs) == 0:
                    st.error("No images found in the specified directory.")
                else:
                    y_pred = []
                    
                    if pipeline is None:
                        st.error("Model pipeline not loaded. Please train the model first.")
                    else:
                        # Instead of `predict_proba` logic from generate_shap_plot, we need to extract features and predict
                        model_obj = pipeline['model']
                        scaler_obj = pipeline['scaler']
                        feature_names = pipeline['feature_names']
                        
                        progress_bar = st.progress(0)
                        for i, img_path in enumerate(all_imgs):
                            features = extract_radiomics_features(img_path)
                            # Create dataframe
                            df = pd.DataFrame([features])
                            # Ensure column order matches training
                            df = df[feature_names]
                            # Scale
                            scaled_features = scaler_obj.transform(df)
                            scaled_df = pd.DataFrame(scaled_features, columns=feature_names)
                            # Predict
                            pred = model_obj.predict(scaled_df)[0]
                            y_pred.append(pred)
                            
                            progress_bar.progress((i + 1) / len(all_imgs))
                        
                        st.success(f"Successfully processed {len(all_imgs)} images!")
                        
                        # Compute Metrics
                        accuracy = accuracy_score(y_true, y_pred)
                        precision = precision_score(y_true, y_pred, zero_division=0)
                        recall = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        
                        # Display Metrics natively using Streamlit columns
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Accuracy", f"{accuracy * 100:.2f}%")
                        m2.metric("Precision", f"{precision * 100:.2f}%")
                        m3.metric("Recall", f"{recall * 100:.2f}%")
                        m4.metric("F1 Score", f"{f1 * 100:.2f}%")
                        
                        # Generate Confusion Matrix Figure visually
                        st.write("---")
                        st.write("### Confusion Matrix")
                        
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        
                        cm = confusion_matrix(y_true, y_pred)
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                    xticklabels=['Predict Normal', 'Predict TB'], 
                                    yticklabels=['True Normal', 'True TB'], ax=ax)
                        ax.set_ylabel('Actual')
                        ax.set_xlabel('Predicted')
                        st.pyplot(fig)
