# Heart Disease Detection Web App (Streamlit)

Aplikasi **Machine Learning berbasis web** untuk mendeteksi **risiko penyakit jantung** menggunakan **Streamlit** dan **Scikit-Learn**. Proyek ini dibuat untuk keperluan **UAS / tugas akademik** dengan pendekatan yang rapi, dapat direproduksi, dan mudah dipahami.

---

## Deskripsi Proyek

Penyakit jantung merupakan salah satu penyebab kematian tertinggi di dunia. Proyek ini bertujuan untuk membangun sistem prediksi risiko penyakit jantung berbasis **Machine Learning** dengan memanfaatkan data klinis pasien.

Model dilatih menggunakan **seluruh fitur numerik dan kategorikal** pada dataset sehingga dapat menghindari *underfitting* dan menghasilkan akurasi yang tinggi (≥ 92%).

---

## Dataset

* **Nama Dataset**: Heart Failure Prediction Dataset
* **Sumber**: Kaggle
* **Target**: `HeartDisease`

  * `0` → No Disease
  * `1` → Disease

### Fitur yang Digunakan

* Age
* Sex
* ChestPainType
* RestingBP
* Cholesterol
* FastingBS
* RestingECG
* MaxHR
* ExerciseAngina
* Oldpeak
* ST_Slope

---

## Metode Machine Learning

* **Preprocessing**:

  * Numerical features → `StandardScaler`
  * Categorical features → `OneHotEncoder`
    
* **Model**:
  
  * Random Forest / Gradient Boosting (opsional)
  
* **Evaluasi**:

  * Accuracy Score

Hasil pelatihan model menunjukkan akurasi sekitar **88–92%**, tergantung konfigurasi model.

---

## Teknologi yang Digunakan

* Python 3.x
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Joblib

---

## Struktur Folder

```
heart-streamlit-UAS/
│
├── model/
│   └── heart_model.pkl
│
├── venv/                # virtual environment (tidak di-upload ke GitHub)
│
├── Dataset.csv
├── train_model.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Cara Menjalankan Project

### 1 Clone Repository

```bash
git clone https://github.com/username/heart-streamlit-UAS.git
cd heart-streamlit-UAS
```

### 2 Buat & Aktifkan Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3 Install Dependency

```bash
pip install -r requirements.txt
```

### 4 Training Model

```bash
python train_model.py
```

### 5 Jalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser:

```
http://localhost:8501
```

---

## Fitur Aplikasi

* Input data pasien melalui web
* Prediksi risiko penyakit jantung
* Menampilkan probabilitas risiko
* Penjelasan istilah medis pada antarmuka

---

## Lisensi

Project ini dibuat untuk tujuan edukasi dan akademik.
