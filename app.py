import joblib
import streamlit as st
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Prediksi Penyakit", layout="wide")

# Fungsi untuk memuat model dengan aman
def load_model(model_path):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.warning(f"File model tidak ditemukan: {model_path}")
        return None

# Memuat model
model_paths = {
    "Kanker Paru": "Models/lung_cancer_model.pkl",
    "Penyakit Ginjal": "Models/kidney_model.pkl",
    "Diabetes": "Models/diabetes_model.pkl",
    "Penyakit Jantung": "Models/heart_disease_model.pkl",
    "Kesehatan Janin": "Models/fetal_health_rf_model.pkl",
    "Kanker Payudara": "Models/breast_cancer_model.pkl"
}

models = {penyakit: load_model(path) for penyakit, path in model_paths.items()}

st.sidebar.title("Navigasi")
penyakit = st.sidebar.selectbox("Pilih jenis penyakit untuk prediksi", list(models.keys()), key="disease_select")

st.sidebar.write("""
### Petunjuk:
- Pilih jenis penyakit dari dropdown.
- Masukkan nilai input yang dibutuhkan.
- Klik **Prediksi** untuk melihat hasil.
""")

# Menentukan input untuk setiap penyakit
disease_inputs = {
    "Kanker Paru": [
        "USIA", "MEROKOK", "JARI KEKUNINGAN", "KECEMASAN", "TEKANAN TEMAN", "PENYAKIT KRONIS", "LELAH", "ALERGI", "MENGI", "KONSUMSI ALKOHOL", "BATUK", "SESAK NAPAS", "SULIT MENELAN", "NYERI DADA"], 
    "Penyakit Ginjal": [
        "USIA", "BP", "SG", "AL", "SU", "RBC", "PC", "PCC", "BA", "BGR", "BU", "SC", "SOD", "POT", "HEMO", "PCV", "WC", "RC", "HTN", "DM", "CAD", "APPET", "PE", "ANE"],
    "Diabetes": [
        "KEHAMILAN", "GLUKOSA", "TEKANAN DARAH", "KETEBALAN KULIT", "INSULIN", "BMI", "FUNGSI SILSILAH DIABETES", "USIA"],
    "Penyakit Jantung": {
        "USIA": "number",
        "JENIS KELAMIN": {"Perempuan": 0, "Laki-laki": 1},
        "CP": {"Angina Tipikal": 0, "Angina Atypikal": 1, "Nyeri Non-Anginal": 2, "Asimptomatik": 3},
        "TRESTBPS": "number",
        "KOLESTEROL": "number",
        "FBS": {"Tidak": 0, "Ya": 1},
        "RESTECG": {"Normal": 0, "Abnormalitas Gelombang ST-T": 1, "Hipertrofi Ventrikel Kiri": 2},
        "THALACH": "number",
        "EXANG": {"Tidak": 0, "Ya": 1},
        "OLDPEAK": "number",
        "SLOPE": {"Meningkat": 0, "Datar": 1, "Menurun": 2},
        "CA": {"0": 0, "1": 1, "2": 2, "3": 3},
        "THAL": {"Normal": 0, "Cacat Tetap": 1, "Cacat Reversibel": 2}},  
    "Kesehatan Janin": [
        "nilai dasar", "percepatan", "gerakan janin", "kontraksi rahim", "deselerasi ringan", "deselerasi berat", "deselerasi berkepanjangan", "variabilitas jangka pendek abnormal", "nilai rata-rata variabilitas jangka pendek", "persentase waktu dengan variabilitas jangka panjang abnormal", "nilai rata-rata variabilitas jangka panjang", "lebar histogram", "histogram min", "histogram maks", "jumlah puncak histogram", "jumlah nol histogram", "modus histogram", "rata-rata histogram", "median histogram", "varian histogram", "kecenderungan histogram"],
    "Kanker Payudara": [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]
}

# Antarmuka Streamlit

st.title(f"Diagnosis {penyakit}")

with st.form("input_form"):
    inputs = []
    
    if penyakit in disease_inputs:
        fields = disease_inputs[penyakit]
        
        if isinstance(fields, dict):
            for key, val in fields.items():
                if isinstance(val, dict):
                    user_input = st.selectbox(key, list(val.keys()), key=key)
                    inputs.append(val[user_input])
                else:
                    inputs.append(st.number_input(key, value=0.0, key=key))
        else:
            for feature in fields:
                if penyakit == "Kanker Payudara":
                    inputs.append(st.number_input(feature, value=0.0, format="%.6f", key=feature))
                else:
                    inputs.append(st.number_input(feature, value=0.0, key=feature))

    submitted = st.form_submit_button("Prediksi")

    if submitted:
        model = models[penyakit]
        input_array = np.array([inputs])
        prediction = model.predict(input_array)

        prediction_label = "Ya" if prediction[0] == 1 else "Tidak"
    
        st.success(f"Hasil Prediksi: {prediction_label}")
