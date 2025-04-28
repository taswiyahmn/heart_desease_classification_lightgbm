# heart_desease_classification_lightgbm

# Laporan Proyek Machine Learning – Heart Disease Classification  
*Nama*: Taswiyah Marsyah Noor  
*Domain Proyek*: Healthcare / Diagnostik Kesehatan  

---

## 1. Domain Proyek  
Penyakit jantung merupakan salah satu penyebab kematian terbesar di seluruh dunia. Hal ini diperkuat oleh laporan WHO yang diterbitkan pada 7 Agustus 2024, yang menyatakan bahwa Ischemic Heart Disease menjadi penyakit dengan tingkat kematian tertinggi secara global, diikuti oleh COVID-19 dan stroke.

Berdasarkan fakta tersebut, proyek ini dikembangkan untuk membantu individu dalam meminimalkan atau menghindari risiko penyakit jantung, dengan melakukan klasifikasi berdasarkan parameter yang tersedia dalam dataset. Meskipun sudah banyak penelitian dan pembuatan model machine learning menggunakan dataset ini, dalam proyek ini saya menggunakan pendekatan yang berbeda dengan menerapkan algoritma LightGBM — sebuah metode yang masih relatif jarang digunakan untuk kasus ini — untuk mengevaluasi potensi akurasi yang dapat dihasilkan.

Pemilihan LightGBM didasarkan pada beberapa pertimbangan teknis. Dataset yang digunakan berukuran sedang, sehingga membutuhkan model yang mampu melakukan pelatihan secara cepat dan efisien dalam penggunaan memori. LightGBM, dengan pendekatan leaf-wise tree growth dibandingkan metode level-wise pada algoritma lain, memungkinkan pemodelan hubungan kompleks antar fitur dengan lebih efektif tanpa meningkatkan waktu komputasi secara signifikan. Selain itu, LightGBM mendukung proses hyperparameter tuning secara fleksibel, yang dapat membantu dalam mengoptimalkan performa model sesuai dengan karakteristik data yang tersedia. 

- *Referensi*:  
  1. “Global Atlas on Cardiovascular Disease Prevention and Control” – WHO  
  2. “Heart Disease Dataset Overview” – UCI Machine Learning Repository  

---

## 2. Business Understanding  
### 2.1 Problem Statements  
1. *PS1*: Bagaimana memprediksi risiko penyakit jantung (ya/tidak) dari data klinis pasien?  
2. *PS2*: Bagaimana meminimalkan false negatives (pasien sakit dianggap sehat) untuk mengurangi risiko fatal?  

### 2.2 Goals  
1. *G1*: Membangun model klasifikasi yang mencapai akurasi ≥ 0.80 pada test set.  
2. *G2: Memprioritaskan early detection penyakit jantung dengan meningkatkan recall positif ≥ 0.75, sekaligus menjaga F1-score minimal 0.78 agar keseimbangan precision-recall tetap optimal untuk skenario medis nyata.

### (Opsional) Solution Statements  
- *SS1*: Baseline LightGBM dengan parameter default → metrik evaluasi: accuracy, recall, precision, AUC.  
---

## 3. Data Understanding  
*Deskripsi Data*  
- *Sumber*: UCI Heart Disease Dataset (atau Kaggle Heart Disease UCI)  
- *Jumlah Sampel*: 1.025 baris  
- *Fitur (14 kolom)*:  
  1. age (int) – usia pasien  
  2. sex (0=female, 1=male)  
  3. cp – chest pain type (0–3)  
  4. trestbps – resting blood pressure  
  5. chol – serum cholesterol  
  6. fbs – fasting blood sugar > 120 mg/dl (0/1)  
  7. restecg – resting ECG results (0–2)  
  8. thalach – max heart rate achieved  
  9. exang – exercise induced angina (0/1)  
  10. oldpeak – ST depression induced by exercise  
  11. slope – slope of peak exercise ST segment (0–2)  
  12. ca – number of major vessels (0–3)  
  13. thal – thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)  
  14. target – label (0 = sehat, 1 = sakit)  

*(Opsional) EDA / Visualisasi*  
- *Distribusi target*: bar chart (plot distribusi 0 vs 1).  
- *Histograms* untuk umur, tekanan darah, kolesterol.  
- *Heatmap korelasi* antar fitur.  

---

## 4. Data Preparation  
1. *Cek missing values*  
   ```python
   df.isna().sum()
| Column    | Missing Values |
|-----------|----------------|
| age       | 0              |
| sex       | 0              |
| cp        | 0              |
| trestbps  | 0              |
| chol      | 0              |
| fbs       | 0              |
| restecg   | 0              |
| thalach   | 0              |
| exang     | 0              |
| oldpeak   | 0              |
| slope     | 0              |
| ca        | 0              |
| thal      | 0              |
| target    | 0              |

dikarnakan tidak terdapat missing value pada dataset dan data sudah memiliki tipe kode numerik sehingga bisa dilanjutkan ke tahap scaling data.

2. *Informasi Statistik Data*
   ```python
   df.describe()
  |       | age   | sex   | cp    | trestbps | chol    | fbs   | restecg | thalach | exang  | oldpeak | slope  | ca     | thal   | target |
|-------|-------|-------|-------|----------|---------|-------|---------|---------|--------|---------|--------|--------|--------|--------|
| count | 1025  | 1025  | 1025  | 1025     | 1025    | 1025  | 1025    | 1025    | 1025   | 1025    | 1025   | 1025   | 1025   | 1025   |
| mean  | 54.43 | 0.70  | 0.94  | 131.61   | 246.00  | 0.15  | 0.53    | 149.11  | 0.34   | 1.07    | 1.39   | 0.75   | 2.32   | 0.51   |
| std   | 9.07  | 0.46  | 1.03  | 17.52    | 51.59   | 0.36  | 0.53    | 23.01   | 0.47   | 1.18    | 0.62   | 1.03   | 0.62   | 0.50   |
| min   | 29.00 | 0.00  | 0.00  | 94.00    | 126.00  | 0.00  | 0.00    | 71.00   | 0.00   | 0.00    | 0.00   | 0.00   | 0.00   | 0.00   |
| 25%   | 48.00 | 0.00  | 0.00  | 120.00   | 211.00  | 0.00  | 0.00    | 132.00  | 0.00   | 0.00    | 1.00   | 0.00   | 2.00   | 0.00   |
| 50%   | 56.00 | 1.00  | 1.00  | 130.00   | 240.00  | 0.00  | 1.00    | 152.00  | 0.00   | 0.80    | 1.00   | 0.00   | 2.00   | 1.00   |
| 75%   | 61.00 | 1.00  | 2.00  | 140.00   | 275.00  | 0.00  | 1.00    | 166.00  | 1.00   | 1.80    | 2.00   | 1.00   | 3.00   | 1.00   |
| max   | 77.00 | 1.00  | 3.00  | 200.00   | 564.00  | 1.00  | 2.00    | 202.00  | 1.00   | 6.20    | 2.00   | 4.00   | 3.00   | 1.00   |

Berdasarkan informasi yang diberikan bahwa terdapat 6 kolom yang memiliki rntan min dan max yang cukup jauh sehingga harus di scaling agar sama reta dengan kolom lainnya

3. *Data scaling*
   ```python
   le = LabelEncoder()
    df['age'] = le.fit_transform(df['age'])
    df['trestbps'] = le.fit_transform(df['trestbps'])
    df['age'] = le.fit_transform(df['age'])
    df['chol'] = le.fit_transform(df['chol'])
    df['age'] = le.fit_transform(df['age'])
    df['thalach'] = le.fit_transform(df['thalach'])
    
    df.head()

  | Index | age | sex | cp | trestbps | chol | fbs | restecg | thalach | exang | oldpeak | slope | ca | thal | target |
|-------|-----|-----|----|----------|------|-----|---------|---------|-------|---------|-------|----|------|--------|
| 0     | 18  | 1   | 0  | 18       | 43   | 0   | 1       | 67      | 0     | 1.0     | 2     | 2  | 3    | 0      |
| 1     | 19  | 1   | 0  | 28       | 34   | 1   | 0       | 54      | 1     | 3.1     | 0     | 0  | 3    | 0      |
| 2     | 36  | 1   | 0  | 31       | 12   | 0   | 1       | 25      | 1     | 2.6     | 0     | 0  | 3    | 0      |
| 3     | 27  | 1   | 0  | 33       | 34   | 0   | 1       | 60      | 0     | 0.0     | 2     | 1  | 3    | 0      |
| 4     | 28  | 0   | 0  | 27       | 116  | 1   | 1       | 9       | 0     | 1.9     | 1     | 3  | 2    | 0      |

4. *Split Data*
   ```python
   X = df.drop(['target'], axis=1)
   y = df['target']
  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
