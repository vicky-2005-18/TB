import pickle
import pandas as pd
import shap
import numpy as np
import xgboost as xgb

with open("model/tb_ml_model.pkl", "rb") as f:
    pipeline = pickle.load(f)

model = pipeline['model']
scaler = pipeline['scaler']
feature_names = pipeline['feature_names']

# Create dummy data
dummy_data = {fn: 0.5 for fn in feature_names}
df = pd.DataFrame([dummy_data])
df_scaled = scaler.transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=feature_names)

# Try XGBoost native SHAP
dmatrix = xgb.DMatrix(df_scaled)
contribs = model.get_booster().predict(dmatrix, pred_contribs=True)

print("contribs shape:", contribs.shape)
print("contribs:", contribs)

# Create an Explanation object
# contribs has shape (n_samples, n_features + 1), where the last column is the base value (expected value)
base_value = contribs[0, -1]
shap_vals = contribs[0, :-1]

explanation = shap.Explanation(
    values=shap_vals,
    base_values=base_value,
    data=df_scaled.iloc[0].values,
    feature_names=feature_names
)

try:
    shap.plots.waterfall(explanation, max_display=10, show=False)
    print("Plot succeeded with native XGBoost!")
except Exception as e:
    print(f"waterfall error: {e}")
