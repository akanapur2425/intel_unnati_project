import pyshark
import torch
import joblib
from deep_model import LiveNet  

model = LiveNet()
model.load_state_dict(torch.load("live_traffic_model.pth"))
model.eval()

scaler = joblib.load("scaler_live.save")

def extract_features(packet):
    try:
        length = int(packet.length)
        protocol = 0  
        time_ = float(packet.sniff_timestamp)
        
        features = [length, protocol, time_] + [0]*(76-3)
        return features
    except:
        return None

cap = pyshark.LiveCapture(interface='en0')  

print("🚀 Starting live prediction (capture 20 packets)...")

for packet in cap.sniff_continuously(packet_count=20):
    features = extract_features(packet)
    if features:
        scaled = scaler.transform([features])
        x_tensor = torch.tensor(scaled, dtype=torch.float32)
        pred = model(x_tensor).item()
        if pred > 0.5:
            print("⚠️ Potential Threat Detected!")
        else:
            print("✅ Normal Packet")
