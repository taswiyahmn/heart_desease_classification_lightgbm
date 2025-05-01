# Laporan Proyek Machine Learning – Heart Disease Classification  
*Nama*: Taswiyah Marsyah Noor  
*Domain Proyek*: Healthcare / Diagnostik Kesehatan  

---

# 1. Domain Proyek  
Penyakit jantung merupakan salah satu penyebab kematian terbesar di seluruh dunia. Hal ini diperkuat oleh laporan WHO yang diterbitkan pada 7 Agustus 2024, yang menyatakan bahwa Ischemic Heart Disease menjadi penyakit dengan tingkat kematian tertinggi secara global, diikuti oleh COVID-19 dan stroke.

Berdasarkan fakta tersebut, proyek ini dikembangkan untuk membantu individu dalam meminimalkan atau menghindari risiko penyakit jantung, dengan melakukan klasifikasi berdasarkan parameter yang tersedia dalam dataset. Meskipun sudah banyak penelitian dan pembuatan model machine learning menggunakan dataset ini, dalam proyek ini saya menggunakan pendekatan yang berbeda dengan menerapkan algoritma LightGBM sebuah metode yang masih relatif jarang digunakan untuk kasus ini untuk mengevaluasi potensi akurasi yang dapat dihasilkan.

Pemilihan LightGBM didasarkan pada beberapa pertimbangan teknis. Dataset yang digunakan berukuran sedang, sehingga membutuhkan model yang mampu melakukan pelatihan secara cepat dan efisien dalam penggunaan memori. LightGBM, dengan pendekatan leaf-wise tree growth dibandingkan metode level-wise pada algoritma lain, memungkinkan pemodelan hubungan kompleks antar fitur dengan lebih efektif tanpa meningkatkan waktu komputasi secara signifikan. Selain itu, LightGBM mendukung proses hyperparameter tuning secara fleksibel, yang dapat membantu dalam mengoptimalkan performa model sesuai dengan karakteristik data yang tersedia. 

- *Referensi*:  
  1. “Global Atlas on Cardiovascular Disease Prevention and Control” – WHO  
  2. “Heart Disease Dataset Overview” – UCI Machine Learning Repository  

---

# 2. Business Understanding  
### 2.1 Problem Statements  
1. *PS1*: Bagaimana memprediksi risiko penyakit jantung (ya/tidak) dari data klinis pasien?  
2. *PS2*: Bagaimana meminimalkan false negatives (pasien sakit dianggap sehat) untuk mengurangi risiko fatal?  

### 2.2 Goals  
1. *G1*: Membangun model klasifikasi yang mencapai akurasi ≥ 0.80 pada test set.  
2. *G2: Memprioritaskan early detection penyakit jantung dengan meningkatkan recall positif ≥ 0.75, sekaligus menjaga F1-score minimal 0.78 agar keseimbangan precision-recall tetap optimal untuk skenario medis nyata.

### (Opsional) Solution Statements  
- *SS1*: Baseline LightGBM dengan parameter default → metrik evaluasi: accuracy, recall, precision, AUC.  
---

# 3. Data Understanding  
*Deskripsi Data*  
- *Sumber*: UCI Heart Disease Dataset (atau Kaggle Heart Disease UCI)  
- *Jumlah Sampel*: 1.025 baris  
- *Fitur (14 kolom)*:  
    1. Fitur (14 kolom):
    2. age (int) – usia pasien
    3. sex – jenis kelamin pasien (0 = perempuan, 1 = laki-laki)
    4. cp – jenis nyeri dada (0–3)
    5. trestbps – tekanan darah saat istirahat (resting blood pressure)
    6. chol – kadar kolesterol dalam serum
    7. fbs – kadar gula darah puasa > 120 mg/dl (0 = tidak, 1 = ya)
    8. restecg – hasil elektrokardiogram saat istirahat (0–2)
    9. thalach – detak jantung maksimum yang dicapai
    10. exang – angina yang diinduksi oleh olahraga (0 = tidak, 1 = ya)
    11. oldpeak – tingkat depresi segmen ST akibat latihan
    12. slope – kemiringan segmen ST saat puncak latihan (0–2)
    13. ca – jumlah pembuluh darah utama yang terlihat (0–3)
    14. thal – kondisi thalassemia (1 = normal, 2 = cacat tetap, 3 = cacat reversibel)
    15. target – label kondisi pasien (0 = sehat, 1 = memiliki penyakit jantung) 

---
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

Hasil eksplorasi awal menunjukkan bahwa dataset tidak memiliki nilai kosong (missing value) atau data duplikat. Selain itu, semua fitur dalam dataset sudah menggunakan tipe data numerik. Hal ini disebabkan oleh fakta bahwa data yang tersedia di Kaggle telah melewati tahap preprocessing, sehingga tahapan preprocessing tidak diperlukan lagi dalam proyek ini.

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

Berdasarkan hasil analisis terhadap distribusi nilai pada masing-masing fitur numerik, ditemukan bahwa terdapat empat kolom yang memiliki rentang nilai (minimum dan maksimum) yang cukup jauh atau tidak seragam jika dibandingkan dengan kolom-kolom lainnya. Perbedaan rentang ini dapat menyebabkan algoritma pembelajaran mesin bekerja secara tidak optimal karena model cenderung memberi bobot lebih pada fitur dengan nilai yang lebih besar. Oleh karena itu, dilakukan proses normalisasi atau scaling untuk menyesuaikan skala keenam kolom tersebut agar berada dalam rentang yang setara dengan fitur lainnya, sehingga model dapat mempelajari data secara seimbang dan adil.

* Distribusi Taget (0 dan 1)

![image](https://github.com/user-attachments/assets/955d11ba-d569-4850-9841-14740efbf2e2)

* Heatmap correlation terhadap fitur

![image](https://github.com/user-attachments/assets/14327d50-667d-417e-a8ed-2d184df71899)

### Penjelasan Korelasi Fitur dan Target
1. **Korelasi Positif Signifikan**: Fitur seperti `cp`, `thalach`, dan `exang` menunjukkan korelasi positif dengan target, yang berarti semakin tinggi nilai pada fitur-fitur ini, semakin besar kemungkinan pasien mengalami penyakit jantung.
2. **Korelasi Negatif Signifikan**: Fitur seperti `oldpeak` dan `ca` memiliki korelasi negatif dengan target, di mana penurunan pada nilai fitur ini cenderung meningkatkan kemungkinan penyakit jantung.
3. **Korelasi Antar Fitur**: Korelasi negatif antara `slope` dan `oldpeak` serta antara `age` dan `thalach` menunjukkan adanya hubungan antar faktor-faktor risiko yang mempengaruhi hasil kesehatan.
4. **Korelasi Rendah**: Fitur seperti `trestbps`, `chol`, `fbs`, dan `sex` memiliki korelasi rendah dengan target, yang berarti mereka tidak secara langsung mempengaruhi risiko penyakit jantung, namun tetap relevan untuk analisis model.

# 4. Data Preparation  
1. *Data scaling*

| Index | age | sex | cp | trestbps | chol | fbs | restecg | thalach | exang | oldpeak | slope | ca | thal | target |
|-------|-----|-----|----|----------|------|-----|---------|---------|-------|---------|-------|----|------|--------|
| 0     | 18  | 1   | 0  | 18       | 43   | 0   | 1       | 67      | 0     | 1.0     | 2     | 2  | 3    | 0      |
| 1     | 19  | 1   | 0  | 28       | 34   | 1   | 0       | 54      | 1     | 3.1     | 0     | 0  | 3    | 0      |
| 2     | 36  | 1   | 0  | 31       | 12   | 0   | 1       | 25      | 1     | 2.6     | 0     | 0  | 3    | 0      |
| 3     | 27  | 1   | 0  | 33       | 34   | 0   | 1       | 60      | 0     | 0.0     | 2     | 1  | 3    | 0      |
| 4     | 28  | 0   | 0  | 27       | 116  | 1   | 1       | 9       | 0     | 1.9     | 1     | 3  | 2    | 0      |

Dalam proyek ini, proses scaling diterapkan menggunakan Standard Scaler, yaitu metode yang mengubah data sehingga memiliki distribusi dengan rata-rata nol dan standar deviasi satu. Teknik ini dipilih untuk menyamakan skala dan rentang nilai setiap fitur numerik, sehingga semua fitur berada dalam kisaran yang konsisten dan dapat meminimalkan perbedaan skala antar fitur yang dapat mempengaruhi performa model.

2. *Split Data*

Proses pembagian data (split) dilakukan untuk memisahkan data pelatihan dan data pengujian. Dalam proyek ini, data dibagi dengan proporsi 80:20, di mana 20% data digunakan sebagai data uji (test data) dan sisanya, yaitu 80%, digunakan sebagai data latih (training data). Pembagian ini dilakukan untuk memastikan bahwa model dapat dilatih dengan data yang cukup serta dievaluasi kinerjanya menggunakan data yang belum pernah dilihat sebelumnya.

# 5. Modelling
### Tahap Pembangunan Model Klasifikasi
Decision Tree dipilih sebagai model dasar atau baseline karena algoritma ini memiliki struktur yang sederhana, mudah dipahami, dan cepat dalam pelatihan awal. Setelah itu, LightGBM digunakan sebagai model utama yang diharapkan dapat memberikan performa yang lebih baik, mengingat algoritma ini merupakan salah satu model boosting yang unggul dalam kecepatan pelatihan dan akurasi prediksi.

## Model 1: LightGBM

### Cara Kerja
LightGBM adalah library machine learning yang digunakan untuk implementasi model Gradient Boosting Decision Tree (GBDT). LightGBM bekerja dengan cara membangun beberapa pohon keputusan secara bertahap dan memperbaiki kesalahan model sebelumnya.

### Parameter:
- `objective='binary'`: Karena ini adalah masalah klasifikasi biner.
- `metric='binary_error'`: Kita menggunakan error sebagai metric untuk evaluasi.
- `boosting_type='gbdt'`: Gradient Boosting Decision Tree (GBDT).
- `num_leaves=31`: Jumlah leaves dalam tree.
- `learning_rate=0.1`: Learning rate yang digunakan untuk memperbarui bobot model.
- `feature_fraction=0.9`: Persentase fitur yang dipilih untuk tiap iterasi.

**Kelebihan:**
- Cepat dan efisien dalam memproses data besar.
- Dapat menangani data dengan missing values.
- Mendukung fitur-fitur khusus untuk pengaturan model yang lebih lanjut.

**Kekurangan:**
- Memerlukan lebih banyak tuning parameter dibandingkan dengan model yang lebih sederhana seperti Decision Tree.


## Model 2: Decision Tree

### Cara Kerja
Decision Tree membagi data berdasarkan fitur yang memberikan informasi paling tinggi (dengan pengukuran seperti Gini atau Entropy). Setiap cabang merepresentasikan keputusan hingga mencapai leaf node (output prediksi).

### Parameter:
- `random_state=42`: Untuk menjaga hasil tetap konsisten.

### Kode:
**Kelebihan:**
- Mudah dipahami dan divisualisasikan.
- Tidak memerlukan normalisasi data.

**Kekurangan:**
- Rentan overfitting jika pohon terlalu dalam.

# 6. Evaluation
### Penjelasan Metrik:
1. **Precision**  
   Mengukur seberapa akurat prediksi positif model. Formula:  
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

2. **Recall**  
   Mengukur seberapa banyak prediksi positif model yang benar. Formula:  

$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

3. **F1-Score**  
   Rata-rata harmonis antara precision dan recall, memberikan gambaran kinerja model secara keseluruhan. Formula:  

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

4. **Accuracy**  

$$\text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Samples}}$$

5. **Macro Average**  
   Rata-rata nilai precision, recall, dan F1-score tanpa mempertimbangkan distribusi kelas.

6. **Weighted Average**  
   Rata-rata nilai precision, recall, dan F1-score yang mempertimbangkan distribusi kelas.

### Hasil:
- **Precision**, **Recall**, dan **F1-Score** untuk kedua kelas adalah 1.00, menunjukkan model sangat akurat.
- **Accuracy**, **Macro avg**, dan **Weighted avg** juga 1.00, mengindikasikan performa model yang sangat baik dan konsisten.

## Hasil Evaluasi Model:

#### 🟢 **Akurasi LightGBM**: 1.0000
#### 🟢 **Classification Report LightGBM**:
| Kelas | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00      | 1.00   | 1.00     | 100     |
| 1     | 1.00      | 1.00   | 1.00     | 105     |
| **Accuracy** |  |  | **1.00** | **205**  |
| Macro avg | 1.00 | 1.00 | 1.00 | 205 |
| Weighted avg | 1.00 | 1.00 | 1.00 | 205 |

#### 🟢 **Akurasi Decision Tree**: 0.9854
#### 🟢 **Classification Report Decision Tree**:
| Kelas | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.97      | 1.00   | 0.99     | 100     |
| 1     | 1.00      | 0.97   | 0.99     | 105     |
| **Accuracy** |  |  | **0.99** | **205**  |
| Macro avg | 0.99 | 0.99 | 0.99 | 205 |
| Weighted avg | 0.99 | 0.99 | 0.99 | 205 |

### Penjelasan ROC Curve:

ROC (Receiver Operating Characteristic) curve adalah alat evaluasi kinerja model klasifikasi yang digunakan untuk memvisualisasikan kemampuan model dalam membedakan antara kelas positif dan negatif.

1. **AUC (Area Under the Curve)**  
   AUC mengukur luas area di bawah ROC curve. Nilai AUC berkisar antara 0 dan 1. Semakin mendekati 1, semakin baik model dalam memisahkan kelas positif dan negatif.
   - **AUC = 1**: Model sempurna dalam membedakan kelas.
   - **AUC = 0.5**: Model tidak lebih baik dari tebakan acak.
   - **AUC < 0.5**: Model lebih buruk daripada tebakan acak.

2. **True Positive Rate (TPR) atau Recall**  
   TPR mengukur seberapa banyak prediksi positif yang benar dari seluruh data positif yang sebenarnya.

$$\text{TPR} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

3. **False Positive Rate (FPR)**  
   FPR mengukur seberapa banyak prediksi negatif yang salah dari seluruh data negatif yang sebenarnya.
   
$$\text{FPR} = \frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}}$$

5. **ROC Curve**  
   ROC curve adalah plot antara **TPR** (di sumbu Y) dan **FPR** (di sumbu X) untuk berbagai threshold. ROC curve membantu untuk memilih threshold terbaik yang menghasilkan keseimbangan antara sensitivitas dan spesifisitas.

### Interpretasi ROC Curve:
- **Kurva yang lebih dekat dengan sudut kiri atas** menunjukkan model yang lebih baik dalam membedakan kelas positif dan negatif.
- **Kurva yang mendekati garis diagonal** menunjukkan kinerja model yang buruk.

- Visualisasi ROC Lightgbm

![image](https://github.com/user-attachments/assets/6a93eb31-e02a-44ce-8531-071edee20138)

- Visualisasi ROC Decision tree

![image](https://github.com/user-attachments/assets/2d651b1e-4ec7-4b27-b184-69cae51edb10)

### Ringkasan ROC Curve

- LightGBM

AUC = 1.00
Kurva ROC menunjukkan hasil sempurna: model berhasil memisahkan kelas positif dan negatif tanpa kesalahan (TPR = 1, FPR = 0).
- Decision Tree

AUC ≈ 0.99
Kurva ROC hampir sempurna: hanya ada sedikit trade-off antara sensitivitas dan spesifisitas, dengan beberapa false positive/negative pada beberapa threshold.


# Kesimpulan
Proyek ini menunjukkan bahwa LightGBM dapat memberikan kinerja yang sangat baik dalam kasus klasifikasi penyakit jantung, dengan akurasi 100%. Sementara itu, model Decision Tree juga memberikan hasil yang sangat baik, tetapi dengan akurasi sedikit lebih rendah.
Model ini memiliki potensi untuk diterapkan dalam dunia medis, dengan kemampuan untuk memprediksi kemungkinan penyakit jantung pada pasien berdasarkan data klinis yang tersedia. Meskipun demikian, untuk penerapan dunia nyata, model ini harus diuji lebih lanjut pada dataset yang lebih besar dan lebih beragam untuk memastikan bahwa model ini dapat menggeneralisasi dengan baik pada data yang belum pernah dilihat sebelumnya.
