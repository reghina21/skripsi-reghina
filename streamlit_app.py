import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math

st.set_page_config(page_title="Dashboard Peramalan Kurs", layout="wide")
st.title("📊 Dashboard Peramalan Kurs Yuan & Dollar")

# Navigasi atas dengan tabs
tabs = st.tabs([
    "🏠 Home",
    "📁 Dataset",
    "🧹 Preprocessing",
    "📊 Hasil Prediksi",
    "⏭️ Prediksi Masa Depan"
])

# Tab Home
with tabs[0]:
    st.header("Selamat Datang di Dashboard Peramalan Kurs")
    st.markdown("""
    Dashboard ini dirancang untuk membantu Anda melakukan:
    - Upload dan eksplorasi dataset kurs
    - Preprocessing data
    - Visualisasi kurs jual dan beli
    - Prediksi kurs jual di masa depan

    Silakan mulai dengan mengunggah dataset Anda di tab **📁 Dataset**.
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

        st.markdown("### 💹 Kurs Jual")
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

# Tab 3: Hasil Prediksi
with tabs[3]:
    st.subheader("📊 Hasil Prediksi Kurs dengan Interval Fuzzy")

    if st.session_state.get('preprocessed', False):
        # Dropdown pemilihan jenis kurs
        tipe_kurs = st.selectbox("Pilih jenis kurs yang ingin diprediksi:", ["Kurs Jual", "Kurs Beli"])

        if tipe_kurs == "Kurs Jual":
            df_kurs = st.session_state.df_kurs_jual.copy()
            kolom_kurs = "Kurs Jual"
        else:
            df_kurs = st.session_state.df_kurs_beli.copy()
            kolom_kurs = "Kurs Beli"

        # Hitung jumlah kelas dengan Aturan Sturges
        n = len(df_kurs)
        K = round(1 + 3.322 * math.log10(n))

        Dmax = df_kurs[kolom_kurs].max()
        Dmin = df_kurs[kolom_kurs].min()
        D1, D2 = 0, 0  # Penyesuaian bawah dan atas

        R = (Dmax + D2) - (Dmin - D1)
        I = R / K

        # Buat interval fuzzy
        intervals = []
        for i in range(K):
            lower = Dmin - D1 + i * I
            upper = lower + I
            intervals.append((lower, upper))

        # Fuzzifikasi
        def fuzzy_label(value):
            for idx, (low, high) in enumerate(intervals):
                if low <= value <= high:
                    return f"A{idx + 1}"
            return None

        df_kurs["Fuzzy Set"] = df_kurs[kolom_kurs].apply(fuzzy_label)

        hasil_list = []

        for i in range(3, len(df_kurs)):
            E_i = df_kurs[kolom_kurs].iloc[i - 1]
            E_i_1 = df_kurs[kolom_kurs].iloc[i - 2]
            E_i_2 = df_kurs[kolom_kurs].iloc[i - 3]

            D_i = abs(abs(E_i - E_i_1) - abs(E_i_1 - E_i_2))

            values_to_check = [
                E_i + D_i / 2, E_i - D_i / 2,
                E_i + D_i, E_i - D_i,
                E_i + D_i / 4, E_i - D_i / 4,
                E_i + 2 * D_i, E_i - 2 * D_i,
                E_i + D_i / 6, E_i - D_i / 6,
                E_i + 3 * D_i, E_i - 3 * D_i,
            ]

            fuzzy_i1 = df_kurs['Fuzzy Set'].iloc[i]
            interval_idx = int(fuzzy_i1[1:]) - 1 if fuzzy_i1 and fuzzy_i1[1:].isdigit() else -1

            if interval_idx < 0 or interval_idx >= len(intervals):
                df_kurs.at[i, 'Prediksi'] = None
                continue

            low, high = intervals[interval_idx]
            mid = (low + high) / 2

            R_sum = sum(val for val in values_to_check if low <= val <= high)
            S = sum(1 for val in values_to_check if low <= val <= high)

            F_j = (R_sum + mid) / (S + 1) if S > 0 else mid
            df_kurs.at[i, 'Prediksi'] = round(F_j, 2)

            hasil_list.append({
                'i': i,
                'Tanggal': df_kurs.index[i],
                'Aktual': df_kurs[kolom_kurs].iloc[i],
                'Prediksi': round(F_j, 2)
            })

        # Tambahkan 3 baris awal tanpa prediksi
        for j in range(3):
            hasil_list.insert(j, {
                'i': j,
                'Tanggal': df_kurs.index[j],
                'Aktual': df_kurs[kolom_kurs].iloc[j],
                'Prediksi': None
            })

        df_hasil_perhitungan = pd.DataFrame(hasil_list)

        st.markdown(f"### 🔍 Tabel Hasil Prediksi {tipe_kurs}")
        st.dataframe(df_hasil_perhitungan[['Tanggal', 'Aktual', 'Prediksi']])

        # Tambahan: Grafik Matplotlib
        st.markdown(f"### 📊 Grafik {tipe_kurs} Aktual vs Prediksi")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_hasil_perhitungan['Tanggal'], df_hasil_perhitungan['Aktual'], label='Aktual', marker='o')
        ax.plot(df_hasil_perhitungan['Tanggal'], df_hasil_perhitungan['Prediksi'], label='Prediksi', marker='x')
        ax.set_xlabel("Tanggal")
        ax.set_ylabel(tipe_kurs)
        ax.set_title(f"Perbandingan {tipe_kurs} vs Prediksi")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.warning("Mohon lakukan preprocessing data terlebih dahulu.")

from datetime import timedelta
import pandas as pd

# Fungsi untuk menentukan fuzzy label berdasarkan interval
def fuzzy_label(value, intervals):
    for i, (low, high) in enumerate(intervals):
        if low <= value <= high:
            return f"A{i+1}"
    return None

# 🔧 Fungsi bantu: Ambil 3 nilai terakhir (gabungan Kurs Beli dan Prediksi)
def ambil_nilai_terakhir(df, n=3):
    hasil = []
    sumber = []
    rows = df.iloc[-n:]
    for idx, row in rows.iterrows():
        val = row['Kurs Beli'] if pd.notnull(row['Kurs Beli']) else row['Prediksi']
        sumber.append('Aktual' if pd.notnull(row['Kurs Beli']) else 'Prediksi')
        hasil.append(val)
    return hasil, sumber

# 🔁 Prediksi 5 Periode ke Depan
gabungan_data = df_kurs_beli[['Kurs Beli', 'Fuzzy Set', 'Prediksi']].copy()
data_awal = gabungan_data[gabungan_data['Prediksi'].notnull()].iloc[-3:].copy()

n_prediksi = 5
future_dates = pd.date_range(start='2025-01-13', periods=5)


prediksi_ke_depan = []

for step in range(n_prediksi):
    # Ambil E(i), E(i-1) E(i-2)
    nilai_terakhir, sumber = ambil_nilai_terakhir(data_awal, 3)
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

    # Gunakan fuzzy label dari baris terakhir
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
    fuzzy_new = fuzzy_label(F_j, intervals)

    # Simpan prediksi
    prediksi_ke_depan.append({
        'Tanggal': future_dates[step],
        'Prediksi': F_j
    })

    # Tambahkan ke data_awal
    row_baru = pd.DataFrame([{
        'Kurs Beli': None,
        'Fuzzy Set': fuzzy_new,
        'Prediksi': F_j
    }], index=[future_dates[step]])

    data_awal = pd.concat([data_awal, row_baru], ignore_index=False)

# 🚀 Tampilkan hasil akhir
df_prediksi_5 = pd.DataFrame(prediksi_ke_depan)
print("📊 Prediksi 5 Periode ke Depan (dengan fluktuasi):")
print(df_prediksi_5)

# 📈 Visualisasi (opsional)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(df_prediksi_5['Tanggal'], df_prediksi_5['Prediksi'], marker='o', color='blue')
plt.title('Prediksi 5 Periode Kedepan')
plt.xlabel('Tanggal')
plt.ylabel('Kurs Prediksi')
plt.grid(True)
plt.tight_layout()
plt.show()

