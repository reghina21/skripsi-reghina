import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math
from datetime import timedelta

st.set_page_config(page_title="Dashboard Peramalan Kurs", layout="wide")
st.title("📊 Dashboard Peramalan Kurs Yuan & Dollar")
...
# Navigasi atas dengan tabs
tabs = st.tabs([
    "🏠 Home",
    "📁 Dataset",
    "🩵 Preprocessing",
    "📈 Prediksi Kurs Jual",
    "📉 Prediksi Kurs Beli"
])

# Tab Home
with tabs[0]:
    st.header("Selamat Datang di Dashboard Peramalan Kurs")
    st.markdown("""
    Dashboard ini dirancang untuk membantu Anda melakukan:
    - Upload dan eksplorasi dataset kurs
    - Preprocessing data
    - Visualisasi kurs jual dan beli
    - Prediksi kurs jual dan beli di masa depan

    Silakan mulai dengan mengunggah dataset Anda di tab *📁 Dataset*.
    """)

# Tab 1: Upload Dataset
with tabs[1]:
    st.subheader("Upload Dataset Kurs")
    uploaded_file = st.file_uploader(
        "Upload file Excel dengan kolom: NO, Nilai, Kurs Jual, Kurs Beli, Tanggal",
        type=["xlsx"]
    )

    if uploaded_file:
        try:
            df_raw = pd.read_excel(uploaded_file)
            required_columns = {"NO", "Nilai", "Kurs Jual", "Kurs Beli", "Tanggal"}

            if not required_columns.issubset(df_raw.columns):
                st.error(f"❌ Kolom tidak lengkap! Harus ada kolom: {', '.join(required_columns)}")
            else:
                st.session_state.df_raw = df_raw
                st.session_state.preprocessed = False
                st.success("✅ Dataset berhasil diupload!")
                st.dataframe(df_raw)
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")
    elif 'df_raw' in st.session_state:
        st.info("📄 Dataset sebelumnya:")
        st.dataframe(st.session_state.df_raw)

# Tab 2: Preprocessing
with tabs[2]:
    st.subheader("Hasil Preprocessing Data Kurs")

    if 'df_raw' in st.session_state:
        df = st.session_state.df_raw.copy()
        df.drop(columns=["NO", "Nilai"], inplace=True)
        df["Tanggal"] = pd.to_datetime(df["Tanggal"])
        df.sort_values("Tanggal", inplace=True)
        df.set_index("Tanggal", inplace=True)

        df_kurs_jual = df[["Kurs Jual"]].copy()
        df_kurs_beli = df[["Kurs Beli"]].copy()

        st.session_state.df = df
        st.session_state.df_kurs_jual = df_kurs_jual
        st.session_state.df_kurs_beli = df_kurs_beli
        st.session_state.preprocessed = True

        st.success("✅ Data berhasil diproses!")

        st.markdown("### 📉 Kurs Jual")
        st.dataframe(df_kurs_jual.tail())

        fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(df_kurs_jual, label="Kurs Jual", color="green")
        ax1.set_title("Visualisasi Kurs Jual")
        ax1.set_xlabel("Tanggal")
        ax1.set_ylabel("Nilai Kurs")
        ax1.legend()
        st.pyplot(fig1)

        st.markdown("### 💰 Kurs Beli")
        st.dataframe(df_kurs_beli.tail())

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(df_kurs_beli, label="Kurs Beli", color="blue")
        ax2.set_title("Visualisasi Kurs Beli")
        ax2.set_xlabel("Tanggal")
        ax2.set_ylabel("Nilai Kurs")
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.warning("Mohon upload file terlebih dahulu di tab Dataset.")

# Tab 3: Prediksi Kurs Jual
with tabs[3]:
    st.subheader("📈 Prediksi Kurs Jual")

    # (Isi dari prediksi kurs jual tetap sama seperti sebelumnya...)

# Tab 4: Prediksi Kurs Beli
with tabs[4]:
    st.subheader("📉 Prediksi Kurs Beli")

    if st.session_state.get('preprocessed', False):
        df_kurs_beli = st.session_state.df_kurs_beli.copy()

        kolom_kurs = "Kurs Beli"
        n = len(df_kurs_beli)
        K = round(1 + 3.322 * math.log10(n))
        Dmax = df_kurs_beli[kolom_kurs].max()
        Dmin = df_kurs_beli[kolom_kurs].min()
        R = Dmax - Dmin
        I = R / K

        intervals = []
        for i in range(K):
            lower = Dmin + i * I
            upper = lower + I
            intervals.append((lower, upper))

        def fuzzy_label(value):
            for idx, (low, high) in enumerate(intervals):
                if low <= value <= high:
                    return f"A{idx + 1}"
            return None

        df_kurs_beli['Fuzzy Set'] = df_kurs_beli[kolom_kurs].apply(fuzzy_label)

        hasil_list = []
        for i in range(3, len(df_kurs_beli)):
            E_i = df_kurs_beli['Kurs Beli'].iloc[i - 1]
            E_i_1 = df_kurs_beli['Kurs Beli'].iloc[i - 2]
            E_i_2 = df_kurs_beli['Kurs Beli'].iloc[i - 3]

            D_i = abs(abs(E_i - E_i_1) - abs(E_i_1 - E_i_2))

            values_to_check = [
                E_i + D_i / 2, E_i - D_i / 2,
                E_i + D_i, E_i - D_i,
                E_i + D_i / 4, E_i - D_i / 4,
                E_i + 2 * D_i, E_i - 2 * D_i,
                E_i + D_i / 6, E_i - D_i / 6,
                E_i + 3 * D_i, E_i - 3 * D_i
            ]

            fuzzy_i1 = df_kurs_beli['Fuzzy Set'].iloc[i]
            interval_idx = int(fuzzy_i1[1:]) - 1 if fuzzy_i1 and fuzzy_i1[1:].isdigit() else -1

            if interval_idx < 0 or interval_idx >= len(intervals):
                df_kurs_beli.at[i, 'Prediksi'] = None
                continue

            low, high = intervals[interval_idx]
            mid = (low + high) / 2

            R = sum(val for val in values_to_check if low <= val <= high)
            S = sum(1 for val in values_to_check if low <= val <= high)

            F_j = (R + mid) / (S + 1) if S > 0 else mid
            df_kurs_beli.at[i, 'Prediksi'] = round(F_j, 2)

            hasil_list.append({
                'Tanggal': df_kurs_beli.index[i],
                'Aktual': df_kurs_beli['Kurs Beli'].iloc[i],
                'Prediksi': round(F_j, 2)
            })

        for j in range(3):
            hasil_list.insert(j, {
                'Tanggal': df_kurs_beli.index[j],
                'Aktual': df_kurs_beli['Kurs Beli'].iloc[j],
                'Prediksi': None
            })

        df_hasil_perhitungan_beli = pd.DataFrame(hasil_list)

        st.markdown("### 📋 Tabel Prediksi Kurs Beli")
        st.dataframe(df_hasil_perhitungan_beli[['Tanggal', 'Aktual', 'Prediksi']])

        st.markdown("### 📉 Grafik Prediksi Kurs Beli")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_hasil_perhitungan_beli['Tanggal'], df_hasil_perhitungan_beli['Aktual'], label='Aktual', marker='o')
        ax.plot(df_hasil_perhitungan_beli['Tanggal'], df_hasil_perhitungan_beli['Prediksi'], label='Prediksi', marker='x')
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Kurs Beli")
        ax.set_title("Prediksi Kurs Beli vs Aktual")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Prediksi 5 Periode ke Depan
        def ambil_nilai_terakhir(df, n=3):
            hasil = []
            rows = df.iloc[-n:]
            for idx, row in rows.iterrows():
                val = row['Kurs Beli'] if pd.notnull(row['Kurs Beli']) else row['Prediksi']
                hasil.append(val)
            return hasil

        gabungan_data = df_kurs_beli[['Kurs Beli', 'Fuzzy Set', 'Prediksi']].copy()
        data_awal = gabungan_data[gabungan_data['Prediksi'].notnull()].iloc[-3:].copy()

        n_prediksi = 5
        future_dates = pd.date_range(start=df_kurs_beli.index.max() + timedelta(days=1), periods=5)

        prediksi_ke_depan = []

        for step in range(n_prediksi):
            nilai_terakhir = ambil_nilai_terakhir(data_awal, 3)
            E_i_2, E_i_1, E_i = nilai_terakhir

            D_i = abs(abs(E_i - E_i_1) - abs(E_i_1 - E_i_2))

            values_to_check = [
                E_i + D_i / 2, E_i - D_i / 2,
                E_i + D_i, E_i - D_i,
                E_i + D_i / 4, E_i - D_i / 4,
                E_i + 2 * D_i, E_i - 2 * D_i,
                E_i + D_i / 6, E_i - D_i / 6,
                E_i + 3 * D_i, E_i - 3 * D_i,
            ]

            fuzzy_i1 = data_awal['Fuzzy Set'].iloc[-1]
            interval_idx = int(fuzzy_i1[1:]) - 1 if isinstance(fuzzy_i1, str) and fuzzy_i1[1:].isdigit() else None

            if interval_idx is None or interval_idx < 0 or interval_idx >= len(intervals):
                low, high = min(v[0] for v in intervals), max(v[1] for v in intervals)
            else:
                low, high = intervals[interval_idx]

            mid = (low + high) / 2
            R = sum(val for val in values_to_check if low <= val <= high)
            S = sum(1 for val in values_to_check if low <= val <= high)

            F_j = (R + mid) / (S + 1) if S > 0 else mid
            F_j = round(F_j, 2)
            fuzzy_new = fuzzy_label(F_j)

            prediksi_ke_depan.append({
                'Tanggal': future_dates[step],
                'Prediksi': F_j
            })

            row_baru = pd.DataFrame([{
                'Kurs Beli': None,
                'Fuzzy Set': fuzzy_new,
                'Prediksi': F_j
            }], index=[future_dates[step]])

            data_awal = pd.concat([data_awal, row_baru], ignore_index=False)

        df_prediksi_5 = pd.DataFrame(prediksi_ke_depan)

        st.markdown("### 🕒 Prediksi Kurs Beli 5 Periode ke Depan")
        st.dataframe(df_prediksi_5)

        fig_future, ax_future = plt.subplots(figsize=(10, 4))
        ax_future.plot(df_prediksi_5['Tanggal'], df_prediksi_5['Prediksi'], marker='o', color='orange')
        ax_future.set_title("Prediksi Kurs Beli di Masa Depan")
        ax_future.set_xlabel("Tanggal")
        ax_future.set_ylabel("Nilai Prediksi")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_future)

    else:
        st.warning("Mohon lakukan preprocessing data terlebih dahulu.")
