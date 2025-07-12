import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from deep_model import BigNet
import joblib

df = pd.read_csv("big_dataset_preprocessed.csv")
X = df.drop("Label", axis=1).values
y = (df["Label"] == "Attack").astype(int).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler_big.save")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

model = BigNet(input_size=X.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 2 == 0:
        print(f"Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}")

with torch.no_grad():
    y_pred = model(X_test_tensor).round()
    acc = (y_pred.eq(y_test_tensor).sum() / float(y_test_tensor.shape[0])).item()
    print(f"✅ Test Accuracy: {acc:.4f}")

torch.save(model.state_dict(), "big_traffic_model.pth")
print("✅ Model saved to big_traffic_model.pth")
