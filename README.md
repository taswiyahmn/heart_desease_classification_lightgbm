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
