import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard Peramalan Kurs", layout="wide")
st.title("📊 Dashboard Peramalan Kurs Yuan & Dollar")

# Inisialisasi state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'predicted_df' not in st.session_state:
    st.session_state.predicted_df = None

# Navigasi atas dengan tabs
tabs = st.tabs(["📁 Dataset", "📈 Visualisasi Dataset", "🧠 Model", "🔮 Hasil Prediksi"])

# 1. Dataset Tab
with tabs[0]:
    st.subheader("Upload Dataset Kurs")
    uploaded_file = st.file_uploader(
        "Upload file Excel dengan kolom: NO, Nilai, Kurs Jual, Kurs Beli, Tanggal",
        type=["xlsx"]
    )

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)

            required_columns = {"NO", "Nilai", "Kurs Jual", "Kurs Beli", "Tanggal"}
            if not required_columns.issubset(df.columns):
                st.error(f"❌ Kolom tidak lengkap! Harus ada kolom: {', '.join(required_columns)}")
            else:
                df["Tanggal"] = pd.to_datetime(df["Tanggal"])
                df.sort_values("Tanggal", inplace=True)
                st.session_state.df = df
                st.success("✅ Dataset berhasil diupload!")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat membaca file: {e}")

    if st.session_state.df is not None:
        st.write("📄 Data Kurs:")
        st.dataframe(st.session_state.df)

# Tab 2: Preprocessing
with tabs[1]:
    st.subheader("Hasil Preprocessing Data Kurs")

    if 'df_raw' in st.session_state:
        df = st.session_state.df_raw.copy()

        # Preprocessing sesuai instruksi
        df.drop(columns=["NO", "Nilai"], inplace=True)
        df["Tanggal"] = pd.to_datetime(df["Tanggal"])
        df.sort_values("Tanggal", inplace=True)
        df.set_index("Tanggal", inplace=True)
        df.sort_index(inplace=True)

        df_kurs_jual = df[["Kurs Jual"]].copy()
        df_kurs_beli = df[["Kurs Beli"]].copy()

        st.session_state.df = df
        st.session_state.df_kurs_jual = df_kurs_jual
        st.session_state.df_kurs_beli = df_kurs_beli
        st.session_state.preprocessed = True

        st.write("✅ Data setelah preprocessing:")
        st.dataframe(df.tail())

        # Visualisasi hasil preprocessing
        st.subheader("📊 Visualisasi Kurs Jual dan Beli")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_kurs_jual, label="Kurs Jual", color="green")
        ax.plot(df_kurs_beli, label="Kurs Beli", color="blue")
        ax.set_title("Kurs Jual dan Beli setelah Preprocessing")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Nilai Kurs")
        ax.legend()
        st.pyplot(fig)

    else:
        st.warning("Mohon upload file terlebih dahulu di tab Dataset.")

# Tab 3: Visualisasi Dataset
with tabs[2]:
    st.subheader("Visualisasi Dataset Terpilih")
    if st.session_state.preprocessed:
        df = st.session_state.df
        selected_cols = st.multiselect("Pilih kolom kurs untuk divisualisasikan:", df.columns.tolist(), default=df.columns.tolist())

        if selected_cols:
            fig, ax = plt.subplots(figsize=(12, 5))
            for col in selected_cols:
                ax.plot(df.index, df[col], label=col)
            ax.set_title("Visualisasi Data Kurs")
            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Nilai Kurs")
            ax.legend()
            st.pyplot(fig)
    else:
        st.warning("Mohon lakukan preprocessing data terlebih dahulu.")

# Tab 4: Model Dummy
with tabs[3]:
    st.subheader("Model Peramalan (Moving Average Dummy)")
    if st.session_state.preprocessed:
        df = st.session_state.df.copy()

        window = st.slider("Pilih window size (moving average):", 2, 10, 3)
        for col in df.columns:
            df[f"Prediksi {col}"] = df[col].rolling(window=window).mean()

        st.session_state.df_prediksi = df
        st.success("✅ Model dummy berhasil dijalankan!")
        st.dataframe(df.tail())
    else:
        st.warning("Mohon lakukan preprocessing data terlebih dahulu.")

# Tab 5: Hasil Prediksi
with tabs[4]:
    st.subheader("Visualisasi Hasil Prediksi")
    if 'df_prediksi' in st.session_state:
        df = st.session_state.df_prediksi
        kolom = st.selectbox("Pilih kolom kurs:", ["Kurs Jual", "Kurs Beli"])

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df[kolom], label="Asli", color='blue')
        ax.plot(df.index, df[f"Prediksi {kolom}"], label="Prediksi", linestyle='--', color='orange')
        ax.set_title(f"Prediksi vs Aktual - {kolom}")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Nilai Kurs")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Silakan jalankan model terlebih dahulu di tab Model.")
