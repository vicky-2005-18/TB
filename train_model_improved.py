import os
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("Loading Radiomics Features...")
    data_path = "dataset/radiomics_features.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run extract_features.py first.")
        return
        
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples.")
    
    # Separate features and labels
    # Remove metadata columns: label, image_name
    X = df.drop(columns=['label', 'image_name'])
    y = df['label']
    feature_names = X.columns.tolist()
    
    # Split the dataset 80/20 train/test
    print("Splitting dataset into Training and Testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale Features
    print("Scaling Features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve feature names for XGBoost and SHAP later
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # Initialize XGBoost Classifier
    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("Evaluating Model...")
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nFinal Test Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=["Normal", "TB"]))
    
    # Save Model, Scaler, and Feature Names
    print("Saving completely trained ML pipeline...")
    os.makedirs("model", exist_ok=True)
    
    # Save XGBoost model natively
    xgb_model_path = os.path.join("model", "tb_xgboost_model.json")
    model.save_model(xgb_model_path)
    print(f"XGBoost model saved to {xgb_model_path}")
    
    # Save the rest of the pipeline
    pipeline_data = {
        'scaler': scaler,
        'feature_names': feature_names
    }
    
    model_path = os.path.join("model", "tb_ml_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline_data, f)
        
    print(f"Pipeline preprocessor saved to {model_path}")

if __name__ == "__main__":
    main()
