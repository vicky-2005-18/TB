import os
import shap
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def generate_shap_plot(features_dict, pipeline_path, output_path):
    """
    Generates a SHAP force/waterfall plot explaining the prediction
    of a single instance.
    
    Args:
        features_dict (dict): The extracted radiomics features for a single image.
        pipeline_path (str): Path to the ML model .pkl file.
        output_path (str): Path where the explanation plot image will be saved.
        
    Returns:
        float: The raw prediction probability (0 to 1).
    """
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Model pipeline not found at {pipeline_path}")
        
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
        
    scaler = pipeline['scaler']
    feature_names = pipeline['feature_names']

    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(os.path.join("model", "tb_xgboost_model.json"))
    
    # Format single instance into DataFrame (to preserve columns and order)
    # Ensure all required features are present
    formatted_features = {}
    for fn in feature_names:
        formatted_features[fn] = features_dict.get(fn, 0.0) # default to 0.0 if missing safely
        
    df = pd.DataFrame([formatted_features])
    
    # Scale features
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=feature_names)
    
    # Model inference
    prob = model.predict_proba(df_scaled)[0][1] # Probability of Class 1 (TB)
    
    try:
        # Use XGBoost's native SHAP implementation to bypass SHAP TreeExplainer parsing bugs on Windows
        import xgboost as xgb
        dmatrix = xgb.DMatrix(df_scaled)
        contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
        
        # contribs has shape (1, n_features + 1), where the last column is the base value
        base_value = contribs[0, -1]
        shap_vals = contribs[0, :-1]
        
        # Manually assemble SHAP Explanation object
        explanation = shap.Explanation(
            values=shap_vals,
            base_values=base_value,
            data=df_scaled.iloc[0].values,
            feature_names=feature_names
        )
        
        # Plot single instance waterfall explain
        plt.figure(figsize=(10, 6))
        
        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.title(f"SHAP ML Feature Importance\nPrediction: {'TB' if prob >= 0.5 else 'Normal'} ({prob*100:.1f}%)")
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"SHAP visualization failed: {e}")
        # Create a blank/error image instead
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, f"SHAP Error: {e}", ha='center', va='center', wrap=True)
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()
        
    return prob
