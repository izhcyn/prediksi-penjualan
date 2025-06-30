import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# Load data
df = pd.read_excel('cleans.xlsx')
df['WeekStart'] = pd.to_datetime(df['WeekStart'])

# Korelasi fitur
correlation_matrix = df.corr(numeric_only=True)
unit_terjual_correlation = correlation_matrix['UnitTerjual'].abs().sort_values(ascending=False)
selected_features = unit_terjual_correlation[unit_terjual_correlation >= 0.4].index.tolist()
if 'UnitTerjual' in selected_features:
    selected_features.remove('UnitTerjual')

# Simpan selected features
with open('models/selected_features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

# Buat folder models jika belum ada
if not os.path.exists('models'):
    os.makedirs('models')

# Scaling global
feature_scaler = MinMaxScaler()
feature_scaler.fit(df[selected_features])

target_scaler = MinMaxScaler()
target_scaler.fit(df['UnitTerjual'].values.reshape(-1, 1))

with open('models/feature_scaler.pkl', 'wb') as f:
    pickle.dump(feature_scaler, f)

with open('models/target_scaler.pkl', 'wb') as f:
    pickle.dump(target_scaler, f)

# Fungsi buat sequence
def create_sequences(data, target, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(target[i + seq_length])
    return np.array(xs), np.array(ys)

# Loop per produk
sequence_length = 10
for product_code in df['KodeProduk'].unique():
    df_product = df[df['KodeProduk'] == product_code].copy()
    if len(df_product) >= sequence_length + 1:
        X_scaled = feature_scaler.transform(df_product[selected_features])
        y = df_product['UnitTerjual'].values.reshape(-1, 1)
        y_scaled = target_scaler.transform(y)

        X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)

        model = Sequential()
        model.add(Bidirectional(LSTM(64), input_shape=(X_seq.shape[1], X_seq.shape[2])))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer=Adam(0.001), loss=MeanSquaredError())

        model.fit(X_seq, y_seq, epochs=1000, batch_size=32, verbose=0)

        model.save(f'models/model_{product_code}.h5')
        print(f"Model saved for {product_code}")
    else:
        print(f"Data tidak cukup untuk produk {product_code}")
