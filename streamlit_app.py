import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard Peramalan Kurs", layout="wide")
st.title("📊 Dashboard Peramalan Kurs Yuan & Dollar")

# Navigasi atas dengan tabs
tabs = st.tabs([
    "📁 Dataset",
    "🧹 Preprocessing",
    "📈 Visualisasi Dataset",
    "🧠 Model",
    "🔮 Hasil Prediksi"
])

# Tab 1: Upload Dataset
with tabs[0]:
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
with tabs[1]:
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

        # -- Kurs Jual --
        st.markdown("### 💹 Kurs Jual")
        st.dataframe(df_kurs_jual.tail())

        fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(df_kurs_jual, label="Kurs Jual", color="green")
        ax1.set_title("Visualisasi Kurs Jual")
        ax1.set_xlabel("Tanggal")
        ax1.set_ylabel("Nilai Kurs")
        ax1.legend()
        st.pyplot(fig1)

        # -- Kurs Beli --
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

# Tab 3: Visualisasi Dataset
with tabs[2]:
    st.subheader("Visualisasi Dataset Terpilih")
    if 'preprocessed' in st.session_state and st.session_state.preprocessed:
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

# Tab 4: Model Peramalan Fuzzy
with tabs[3]:
    st.subheader("Model Peramalan Fuzzy untuk Kurs Jual")

    if 'preprocessed' in st.session_state and st.session_state.preprocessed:
        df_kurs_jual = st.session_state.df_kurs_jual.copy()

        # --- Tetapkan jumlah interval tetap (misalnya 5 interval) ---
        jml_interval = 9  # Tentukan jumlah interval yang tetap
        min_val = df_kurs_jual["Kurs Jual"].min()
        max_val = df_kurs_jual["Kurs Jual"].max()
        interval_width = (max_val - min_val) / jml_interval

        intervals = []
        fuzzy_sets = []
        for i in range(jml_interval):
            low = min_val + i * interval_width
            high = low + interval_width
            intervals.append((low, high))
            fuzzy_sets.append(f"A{i+1}")

        # --- 2. Transformasi Fuzzy ---
        def fuzzify(value):
            for idx, (low, high) in enumerate(intervals):
                if low <= value <= high:
                    return f"A{idx + 1}"
            return None

        df_kurs_jual["Fuzzy_Set"] = df_kurs_jual["Kurs Jual"].apply(fuzzify)

        # --- 3. Proses Peramalan Fuzzy ---
        hasil_list = []

        for i in range(3, len(df_kurs_jual)):
            E_i = df_kurs_jual['Kurs Jual'].iloc[i - 1]
            E_i_1 = df_kurs_jual['Kurs Jual'].iloc[i - 2]
            E_i_2 = df_kurs_jual['Kurs Jual'].iloc[i - 3]

            D_i = abs(abs(E_i - E_i_1) - abs(E_i_1 - E_i_2))

            # Hitung semua variabel fuzzy
            X_i = E_i + D_i / 2
            XX_i = E_i - D_i / 2
            Y_i = E_i + D_i
            YY_i = E_i - D_i
            P_i = E_i + D_i / 4
            PP_i = E_i - D_i / 4
            Q_i = E_i + 2 * D_i
            QQ_i = E_i - 2 * D_i
            G_i = E_i + D_i / 6
            GG_i = E_i - D_i / 6
            H_i = E_i + 3 * D_i
            HH_i = E_i - 3 * D_i

            values_to_check = [X_i, XX_i, Y_i, YY_i, P_i, PP_i, Q_i, QQ_i, G_i, GG_i, H_i, HH_i]
            fuzzy_i1 = df_kurs_jual['Fuzzy_Set'].iloc[i]

            if not isinstance(fuzzy_i1, str) or not fuzzy_i1[1:].isdigit():
                df_kurs_jual.at[i, 'Prediksi'] = None
                continue

            interval_idx = int(fuzzy_i1[1:]) - 1
            if interval_idx < 0 or interval_idx >= len(intervals):
                df_kurs_jual.at[i, 'Prediksi'] = None
                continue

            low, high = intervals[interval_idx]
            mid = (low + high) / 2
            R = 0
            S = 0

            for val in values_to_check:
                if low <= val <= high:
                    R += val
                    S += 1

            F_j = (R + mid) / (S + 1) if S != 0 else mid
            df_kurs_jual.at[i, 'Prediksi'] = round(F_j, 2)

            hasil_list.append({
                'i': i,
                'Tanggal': df_kurs_jual.index[i],
                'Aktual': df_kurs_jual['Kurs Jual'].iloc[i],
                'Prediksi': round(F_j, 2),
                'Fuzzy_(i)': fuzzy_i1,
                'Midpoint': mid
            })

        # Tambahkan 3 baris awal tanpa prediksi
        for j in range(3):
            hasil_list.insert(j, {
                'i': j,
                'Tanggal': df_kurs_jual.index[j],
                'Aktual': df_kurs_jual['Kurs Jual'].iloc[j],
                'Prediksi': None,
                'Fuzzy_(i)': None,
                'Midpoint': None
            })

        # Buat DataFrame hasil perhitungan
        df_hasil_perhitungan = pd.DataFrame(hasil_list)
        st.session_state.df_hasil_prediksi = df_hasil_perhitungan

        st.success("✅ Model fuzzy berhasil dijalankan.")
        st.dataframe(df_hasil_perhitungan[['Tanggal', 'Aktual', 'Prediksi']].dropna())
    else:
        st.warning("Silakan lakukan preprocessing terlebih dahulu.")

# Tab 5: Hasil Prediksi dan Peramalan Masa Depan
with tabs[4]:
    st.subheader("🔮 Visualisasi Hasil Prediksi & Peramalan Masa Depan")

    if 'df_hasil_prediksi' in st.session_state:
        df_hasil = st.session_state.df_hasil_prediksi.dropna().copy()

        # Grafik Prediksi vs Aktual
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_hasil["Tanggal"], df_hasil["Aktual"], label="Aktual", color='blue')
        ax.plot(df_hasil["Tanggal"], df_hasil["Prediksi"], label="Prediksi", color='orange', linestyle='--')
        ax.set_title("Prediksi Kurs Jual vs Aktual")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Nilai Kurs")
        ax.legend()
        st.pyplot(fig)

        # --- Peramalan Masa Depan ---
        st.markdown("### 📆 Peramalan Masa Depan")

        n_forecast = st.number_input("Jumlah hari ke depan yang ingin diramal:", min_value=1, max_value=30, value=7, step=1)

        # Fungsi untuk melakukan peramalan fuzzy ke depan
        def forecast_future(df_kurs, label):
            df_temp = df_kurs.copy()
            df_temp = df_temp.dropna().copy()
            df_temp = df_temp.tail(3).reset_index()

            if len(df_temp) < 3:
                return pd.DataFrame()

            hasil = []
            last_date = df_temp["Tanggal"].iloc[-1]
            for i in range(n_forecast):
                E_i = df_temp[label].iloc[-1]
                E_i_1 = df_temp[label].iloc[-2]
                E_i_2 = df_temp[label].iloc[-3]
                D_i = abs(abs(E_i - E_i_1) - abs(E_i_1 - E_i_2))

                # Variabel fuzzy
                X_i = E_i + D_i / 2
                XX_i = E_i - D_i / 2
                Y_i = E_i + D_i
                YY_i = E_i - D_i
                P_i = E_i + D_i / 4
                PP_i = E_i - D_i / 4
                Q_i = E_i + 2 * D_i
                QQ_i = E_i - 2 * D_i
                G_i = E_i + D_i / 6
                GG_i = E_i - D_i / 6
                H_i = E_i + 3 * D_i
                HH_i = E_i - 3 * D_i

                values = [X_i, XX_i, Y_i, YY_i, P_i, PP_i, Q_i, QQ_i, G_i, GG_i, H_i, HH_i]

                # Tentukan interval yang sesuai
                fuzzy_set = None
                interval_idx = -1
                for idx, (low, high) in enumerate(intervals):
                    if low <= E_i <= high:
                        fuzzy_set = f"A{idx+1}"
                        interval_idx = idx
                        break

                if interval_idx == -1:
                    pred = E_i
                else:
                    low, high = intervals[interval_idx]
                    mid = (low + high) / 2
                    R = sum(v for v in values if low <= v <= high)
                    S = sum(1 for v in values if low <= v <= high)
                    pred = (R + mid) / (S + 1) if S != 0 else mid

                pred = round(pred, 2)
                pred_date = last_date + pd.Timedelta(days=i+1)
                hasil.append({
                    "Tanggal": pred_date,
                    f"Prediksi {label}": pred
                })

                df_temp = pd.concat([
                    df_temp,
                    pd.DataFrame([{
                        "Tanggal": pred_date,
                        label: pred
                    }])
                ]).tail(3).reset_index(drop=True)

            return pd.DataFrame(hasil)

        # Prediksi Masa Depan Kurs Jual
        st.markdown("#### 💹 Kurs Jual - Peramalan")
        df_forecast_jual = forecast_future(st.session_state.df_kurs_jual.reset_index(), "Kurs Jual")
        if not df_forecast_jual.empty:
            st.dataframe(df_forecast_jual)
        else:
            st.warning("Data tidak cukup untuk meramalkan Kurs Jual.")

        # Prediksi Masa Depan Kurs Beli
        st.markdown("#### 💰 Kurs Beli - Peramalan")
        df_forecast_beli = forecast_future(st.session_state.df_kurs_beli.reset_index(), "Kurs Beli")
        if not df_forecast_beli.empty:
            st.dataframe(df_forecast_beli)
        else:
            st.warning("Data tidak cukup untuk meramalkan Kurs Beli.")

    else:
        st.warning("Belum ada hasil prediksi. Jalankan model terlebih dahulu.")
