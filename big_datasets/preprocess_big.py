import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# ✅ Corrected path
df = pd.read_csv('../MachineLearningCSV/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

df.columns = df.columns.str.strip()

if 'Label' not in df.columns:
    print("❗ Column 'Label' not found after stripping. Columns are:")
    print(df.columns)
    exit()

labels = df['Label'].copy()

drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 
             'Protocol', 'Timestamp', 'Fwd Header Length.1', 'Label']
X = df.drop(columns=drop_cols, errors='ignore')

labels = labels.apply(lambda x: 'Normal' if x == 'BENIGN' else 'Attack')
le = LabelEncoder()
y = le.fit_transform(labels)

X = X.replace([np.inf, -np.inf], 0)
X = X.fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df['Label'] = y
processed_df.to_csv('big_dataset_preprocessed.csv', index=False)

joblib.dump(scaler, 'scaler.save')

print("✅ Preprocessing complete! Saved to big_dataset_preprocessed.csv")
