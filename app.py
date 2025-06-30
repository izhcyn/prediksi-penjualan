import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import timedelta
import os

# Setup
st.set_page_config(layout="wide")

# Load data & scalers
@st.cache_data
def load_data():
    df = pd.read_excel("cleans.xlsx")
    df['WeekStart'] = pd.to_datetime(df['WeekStart'])
    return df

@st.cache_resource
def load_scalers():
    with open("models/feature_scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)
    with open("models/target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)
    with open("models/selected_features.pkl", "rb") as f:
        selected_features = pickle.load(f)
    return feature_scaler, target_scaler, selected_features

df = load_data()
feature_scaler, target_scaler, selected_features = load_scalers()
df['LabelProduk'] = df.apply(lambda row: f"{row['KodeProduk']} - {row['NamaProduk']}", axis=1)
produk_map = dict(zip(df['LabelProduk'], df['KodeProduk']))

sequence_length = 10

def create_sequences(data, seq_len=10):
    return np.array([data[i:i + seq_len] for i in range(len(data) - seq_len)])

# Sidebar navigasi
if 'page' not in st.session_state:
    st.session_state.page = 'Homepage'

st.sidebar.title("Navigasi")
for menu in ["Homepage", "Korelasi", "Prediksi", "Forecasting", "Strategi"]:
    if st.sidebar.button(menu, use_container_width=True, 
        type="primary" if st.session_state.page == menu else "secondary"):
        st.session_state.page = menu
        st.rerun()

# === HOMEPAGE ===
if st.session_state.page == "Homepage":
    st.title("Homepage")
    st.subheader("Data Preview")
    st.write(df.head())
    st.subheader("Penjelasan")
    st.write("""
    Aplikasi ini menggunakan data historis penjualan untuk memprediksi dan memforecast unit terjual produk.
    - Korelasi: Menampilkan fitur paling berpengaruh terhadap UnitTerjual.
    - Prediksi: Model LSTM per produk untuk prediksi mingguan.
    - Forecasting: Perkiraan penjualan beberapa minggu ke depan.
    - Strategi: Rekomendasi produk yang perlu diprioritaskan stoknya.
    """)

# === KORELASI ===
elif st.session_state.page == "Korelasi":
    st.title("Korelasi antar Fitur")
    corr_matrix = df.corr(numeric_only=True)
    unit_terjual_corr = corr_matrix['UnitTerjual'].abs().sort_values(ascending=False)
    st.write(corr_matrix)
    st.info(f"Fitur terpilih (korelasi >= 0.4 dengan UnitTerjual): {selected_features}")

# === PREDIKSI ===
elif st.session_state.page == "Prediksi":
    st.title("Prediksi Penjualan Mingguan")

    selected_label = st.selectbox("Pilih Produk", list(produk_map.keys()))
    selected_code = produk_map[selected_label]

    df_product = df[df['KodeProduk'] == selected_code].copy()
    if len(df_product) >= 11:
        model_path = f"models/model_{selected_code}.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
            X = feature_scaler.transform(df_product[selected_features])
            X_seq = create_sequences(X, 10)
            y_true = df_product['UnitTerjual'].values[10:]
            y_pred = target_scaler.inverse_transform(model.predict(X_seq))

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(y_true, label='Aktual')
            ax.plot(y_pred, label='Prediksi')
            ax.set_title(f'Produk {selected_code}')
            ax.legend()
            st.pyplot(fig)
        else:
            st.error("Model tidak ditemukan.")
    else:
        st.warning("Data belum cukup panjang.")

# === FORECASTING ===
elif st.session_state.page == "Forecasting":
    st.title("Forecasting Mingguan ke Depan")

    selected_label = st.selectbox("Pilih Produk", list(produk_map.keys()), key="forecast")
    selected_code = produk_map[selected_label]
    steps = st.slider("Forecast berapa minggu ke depan?", 1, 12, 4)

    df_product = df[df['KodeProduk'] == selected_code].copy()
    nama_produk = df_product['NamaProduk'].iloc[0] if 'NamaProduk' in df_product.columns else "Nama Tidak Diketahui"
    st.write(f"**Nama Produk:** {nama_produk}")
    if len(df_product) >= 11:
        model_path = f"models/model_{selected_code}.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
            X = feature_scaler.transform(df_product[selected_features])
            current_seq = X[-10:]
            future = []
            date = df_product['WeekStart'].max()
            dates = []
            for _ in range(steps):
                y_pred_scaled = model.predict(np.expand_dims(current_seq, axis=0))
                y_pred = target_scaler.inverse_transform(y_pred_scaled)[0][0]
                future.append(int(round(y_pred)))
                current_seq = np.vstack((current_seq[1:], current_seq[-1]))
                date += timedelta(days=7)
                dates.append(date)
            df_result = pd.DataFrame({"Tanggal": dates, "Forecast": future})
            st.dataframe(df_result)
        else:
            st.error("Model tidak ditemukan.")
    else:
        st.warning("Data belum cukup panjang.")

# === STRATEGI ===
elif st.session_state.page == "Strategi":
    st.title("📦 Strategi Stok Minggu Depan")

    st.info("Berikut adalah 10 produk dengan prediksi penjualan tertinggi minggu depan. Disarankan untuk menyiapkan stok lebih banyak agar tidak kehabisan.")

    result = []

    for kode in df['KodeProduk'].unique():
        df_p = df[df['KodeProduk'] == kode].copy()
        if len(df_p) < 11:
            continue
        model_path = f"models/model_{kode}.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
            X = feature_scaler.transform(df_p[selected_features])
            input_seq = X[-10:]
            y_pred_scaled = model.predict(np.expand_dims(input_seq, axis=0))
            y = target_scaler.inverse_transform(y_pred_scaled)[0][0]
            # Ambil nama produk dari df utama
            nama_produk = df_p['NamaProduk'].iloc[0] if 'NamaProduk' in df_p.columns else "Nama Tidak Diketahui"
            result.append((kode, nama_produk, y))

    # Ambil 5 terbesar
    result = sorted(result, key=lambda x: x[2], reverse=True)[:10]

    for i, (kode, nama, val) in enumerate(result):
       st.markdown(
            f"""
            <div style='font-size:22px; font-weight:bold; margin-top:20px;'>
                {i+1}. {kode} - {nama}
            </div>
            <div style='font-size:18px; margin-bottom:10px;'>
                📈 <b>Prediksi Penjualan:</b> {int(round(val))} unit<br>
                🛒 <b>Rekomendasi:</b> Siapkan stok tambahan minggu depan.
            </div>
            """, 
            unsafe_allow_html=True
        )

