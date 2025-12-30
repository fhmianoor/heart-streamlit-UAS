import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Detection", page_icon="🫀")

model = joblib.load("model/heart_model.pkl")

st.title(" Deteksi Penyakit Jantung")
with st.expander("Penjelasan Istilah Fitur"):
    st.markdown("""
### Informasi Fitur yang Digunakan

| Nama Fitur | Istilah / Kode | Arti Sederhana |
|-----------|----------------|----------------|
| **Age** | Angka (tahun) | Umur pasien dalam satuan tahun |
| **Sex** | **M** | Laki-laki |
|  | **F** | Perempuan |
| **Chest Pain Type** | **TA** | Nyeri dada khas akibat jantung |
|  | **ATA** | Nyeri dada tidak khas |
|  | **NAP** | Nyeri dada bukan dari jantung |
|  | **ASY** | Tidak merasakan nyeri dada |
| **Resting Blood Pressure** | mmHg | Tekanan darah saat istirahat |
| **Cholesterol** | mg/dL | Kadar kolesterol total |
| **Fasting Blood Sugar > 120 mg/dl** | **0** | Gula darah puasa normal |
|  | **1** | Gula darah puasa tinggi |
| **Resting ECG** | **Normal** | Hasil ECG normal |
|  | **ST** | Kelainan segmen ST |
|  | **LVH** | Pembesaran bilik kiri jantung |
| **Max Heart Rate** | bpm | Denyut jantung maksimum |
| **Exercise Angina** | **Y** | Nyeri dada saat olahraga |
|  | **N** | Tidak ada nyeri saat olahraga |
| **Oldpeak** | Angka desimal | Penurunan ST saat olahraga |
| **ST Slope** | **Up** | ST naik (lebih normal) |
|  | **Flat** | ST datar (waspada) |
|  | **Down** | ST turun (berisiko tinggi) |
""")


data = {
    "Age": st.number_input("Age", 1, 120, 45),
    "Sex": st.selectbox("Sex", ["M", "F"]),
    "ChestPainType": st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"]),
    "RestingBP": st.number_input("Resting Blood Pressure", 80, 220, 120),
    "Cholesterol": st.number_input("Cholesterol", 100, 600, 200),
    "FastingBS": st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1]),
    "RestingECG": st.selectbox("Resting ECG", ["Normal", "ST", "LVH"]),
    "MaxHR": st.number_input("Max Heart Rate", 60, 220, 150),
    "ExerciseAngina": st.selectbox("Exercise Angina", ["Y", "N"]),
    "Oldpeak": st.number_input("Oldpeak", 0.0, 10.0, 1.0),
    "ST_Slope": st.selectbox("ST Slope", ["Up", "Flat", "Down"]),
}

input_df = pd.DataFrame([data])

if st.button("Predict"):
    proba = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.write(f"Probability of Heart Disease: **{proba:.2%}**")

    if pred == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")
