import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("TB Detection ML Model Evaluation Pipeline")
    
    # Check paths
    model_path = "model/tb_ml_model.pkl"
    data_path = "dataset/radiomics_features.csv"
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please train the model first.")
        return
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    # Load completely trained pipeline
    print("Loading ML Pipeline (Scaler & Features)...")
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
        
    scaler = pipeline['scaler']
    feature_names = pipeline['feature_names']

    print("Loading XGBoost Model...")
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(os.path.join("model", "tb_xgboost_model.json"))
    
    # Load entire dataset
    # You would ideally evaluate against a hold-out test set if created manually, 
    # but for script demonstration, we evaluate the entire dataset to show final metrics and the confusion matrix.
    print("Loading Radiomics Dataset...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['label', 'image_name'])
    y_true = df['label']
    
    # Preprocess test features
    print("Preprocessing Test Features...")
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    
    # Predict
    print("Generating Predictions...")
    y_pred = model.predict(X_scaled)
    
    # Evaluation Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Dataset Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_true, y_pred, target_names=["Normal", "TB"]))
    
    # Plot Confusion Matrix
    print("Plotting Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Normal", "TB"],
                yticklabels=["Normal", "TB"])
    plt.title('Confusion Matrix - TB ML Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save Image
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved as {cm_path}")

if __name__ == "__main__":
    main()
