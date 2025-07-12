import torch
import pandas as pd
import joblib
from deep_model import BigNet


# Load preprocessed big dataset
df = pd.read_csv("big_dataset_preprocessed.csv")
X = df.drop("Label", axis=1).values
y_true = (df["Label"] == "Attack").astype(int).values

# Load scaler
scaler = joblib.load("scaler_big.save")
X_scaled = scaler.transform(X)

# Convert to tensor
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Load model
model = BigNet(input_size=X.shape[1])
model.load_state_dict(torch.load("big_traffic_model.pth"))
model.eval()

# Predict
with torch.no_grad():
    y_pred = model(X_tensor).round().numpy().flatten()

# Evaluate
accuracy = (y_pred == y_true).mean()
print(f"✅ Prediction Accuracy on full big dataset: {accuracy:.4f}")

# Optionally, save predictions
df["Predicted_Label"] = ["Attack" if p == 1 else "Normal" for p in y_pred]
df.to_csv("big_dataset_with_predictions.csv", index=False)
print("✅ Predictions saved to big_dataset_with_predictions.csv")
