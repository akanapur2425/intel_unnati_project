import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from deep_model import LiveNet  # ✅ Correct import

# Load preprocessed data
df = pd.read_csv("preprocessed.csv")

X = df.drop(columns=["Label"]).values
y = df["Label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

model = LiveNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 2 == 0:
        print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")

with torch.no_grad():
    preds = model(X_test_tensor)
    preds = (preds > 0.5).float()
    acc = (preds.eq(y_test_tensor)).sum() / float(y_test_tensor.shape[0])
    print(f"✅ Test Accuracy: {acc.item():.4f}")

torch.save(model.state_dict(), "live_traffic_model.pth")
print("✅ Live model saved to live_traffic_model.pth")
