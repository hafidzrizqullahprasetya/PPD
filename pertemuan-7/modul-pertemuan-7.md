# Modul Praktikum Penambangan Data – Teknologi Rekayasa Perangkat Lunak – 2025
## PERTEMUAN VII: IMBALANCED DATASET

### 7.1 TUJUAN PEMBELAJARAN
A. Mahasiswa memahami konsep *imbalanced dataset* serta permasalahan yang muncul akibat ketidakseimbangan kelas dalam model *machine learning*.  
B. Mahasiswa mampu menerapkan teknik *balancing data* seperti *Random Under-Sampling*, *Random Over-Sampling*, SMOTE, dan ADASYN untuk menyeimbangkan distribusi kelas pada dataset.  
C. Mahasiswa memahami konsep dan metode *feature selection* (*filter*, *wrapper*, *embedded*) untuk memilih fitur yang paling relevan.  

---

### 7.2 DASAR TEORI

**• Imbalanced Class**  
*Imbalance class* adalah kondisi di mana dalam suatu himpunan data terdapat satu kelas (kelas mayoritas) yang memiliki jumlah sampel jauh lebih besar dibandingkan kelas lainnya (kelas minoritas). Kondisi ini menyebabkan algoritma klasifikasi standar cenderung bias memprediksi kelas mayoritas dan gagal mengenali kelas minoritas dengan baik.  
Selain itu, *imbalance class* mengakibatkan nilai sensitivitas (*recall*) yang rendah pada kelas target yang sering kali justru merupakan kelas yang paling krusial untuk dideteksi (misalnya deteksi penyakit atau *fraud detection*). Ketidakseimbangan kelas dapat ditangani melalui manipulasi level data (*resampling*), modifikasi algoritma (*cost-sensitive learning*), atau kombinasi keduanya (Istiana & Mustafiril, 2023).

**• Confusion Matrix**  
*Confusion Matrix* adalah matriks evaluasi berbentuk tabel kontingensi yang membandingkan label aktual dengan hasil prediksi model *machine learning* (Normawati & Prayogi, 2021). Matriks ini memuat empat komponen fundamental:
- **True Positive (TP)**: Sampel positif yang berhasil diprediksi positif.
- **True Negative (TN)**: Sampel negatif yang berhasil diprediksi negatif.
- **False Positive (FP)**: Sampel negatif yang salah diprediksi sebagai positif (*Type I Error*).
- **False Negative (FN)**: Sampel positif yang salah diprediksi sebagai negatif (*Type II Error*).  
Berdasarkan nilai-nilai tersebut, dapat dihitung metrik evaluasi seperti *Accuracy*, *Precision*, *Recall*, dan *F1-Score* (Septiana, Susanto, & Tukiyat, 2021).

**• Feature Selection**  
*Feature Selection* merupakan proses identifikasi dan pemilihan subset fitur yang paling informatif dan relevan dari sekumpulan fitur awal untuk meningkatkan kinerja dan efisiensi algoritma (Pratama & Kusnawi, 2023). Ketika dataset memiliki dimensi fitur yang terlalu besar (*curse of dimensionality*), model rentan mengalami *overfitting* dan waktu pelatihan meningkat drastis. Seleksi fitur bertujuan mereduksi dimensi ruang data dengan membuang fitur redundan dan tidak relevan (Septiana, Susanto, & Tukiyat, 2021).

**• Filter Method**  
*Filter Method* merupakan pendekatan seleksi fitur independen yang mengevaluasi relevansi fitur berdasarkan sifat statistik dan korelasinya terhadap variabel target tanpa melibatkan model *machine learning* (Guyon & Elisseeff, 2003). Metrik yang umum digunakan meliputi korelasi Pearson/Spearman, uji Chi-Square ($\chi^2$), serta uji ANOVA (*Analysis of Variance*) untuk mengukur kemampuan fitur membedakan antar-kelompok kelas. Karena prosesnya tidak melatih model prediktif, *filter method* sangat cepat dan hemat komputasi.

**• Wrapper Method**  
*Wrapper Method* mengevaluasi kombinasi subset fitur dengan memanfaatkan algoritma *machine learning* secara langsung sebagai fungsi evaluasi (Kohavi & John, 1997). Metode ini secara iteratif mencoba berbagai kombinasi fitur—misalnya melalui teknik *Forward Selection*, *Backward Elimination*, atau *Recursive Feature Elimination* (RFE)—lalu memilih kombinasi yang menghasilkan performa model tertinggi. Pendekatan ini lebih akurat dibanding *filter method*, namun memerlukan komputasi yang jauh lebih berat karena melatih model secara berulang.

**• Embedded Method**  
*Embedded Method* memadukan keunggulan *filter* dan *wrapper* dengan melakukan proses seleksi fitur secara bersamaan (*built-in*) selama pelatihan model. Contoh paling umum adalah regresi LASSO (*L1 Regularization*) yang memaksa koefisien fitur yang tidak signifikan menjadi tepat nol (Tibshirani, 1996), serta model berbasis pohon keputusan (*Random Forest*, *Gradient Boosting*) yang menghitung *feature importance* berdasarkan penurunan *Gini impurity* atau *entropy*.

**• SMOTE**  
*Synthetic Minority Over-sampling Technique* (SMOTE) merupakan metode *oversampling* yang bekerja dengan membangkitkan sampel sintetis buatan pada kelas minoritas (bukan sekadar duplikasi data). SMOTE beroperasi di ruang fitur dengan mencari $k$-tetangga terdekat (*k-nearest neighbors*) dari setiap sampel kelas minoritas, lalu membuat titik sintetis baru di sepanjang garis linier yang menghubungkan sampel tersebut dengan salah satu tetangganya yang dipilih secara acak (Wijayanti, Kencana, & Sumarjaya, 2021).

**• ADASYN**  
*Adaptive Synthetic Sampling Approach* (ADASYN) adalah pengembangan dari SMOTE yang menggunakan pendekatan adaptif. ADASYN menghitung bobot distribusi kesulitan belajar bagi setiap sampel minoritas; sampel yang berada di daerah batas keputusan (*decision boundary*) dan lebih sulit dipelajari oleh model akan diberikan proporsi pembangkitan sampel sintetis yang lebih banyak dibanding sampel yang berada di area mudah (Ramadhanti, Santoso, & Widiharih, 2022).

---

### 7.3 ALAT DAN BAHAN

**A. Perangkat Keras**  
1. Laptop dengan minimal RAM 4 GB  
2. Koneksi internet stabil  

**B. Perangkat Lunak**  
1. Web Browser (Google Chrome, Mozilla Firefox, Microsoft Edge, dll.)  
2. Google Colab / Jupyter Notebook  
3. Pustaka Python: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn` (`imblearn`), `matplotlib`, `seaborn`  

**C. Dataset**  
1. *Stroke Prediction Dataset* (5.110 baris, 12 kolom) untuk percobaan penyeimbangan kelas: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset  
2. *Heart Attack Analysis and Prediction Dataset* (303 baris, 14 kolom) untuk percobaan seleksi fitur: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset  
3. *Wine Quality (red)* (1.599 baris, 12 kolom) untuk Tugas 2: https://archive.ics.uci.edu/dataset/186/wine+quality  

---

### 7.4 LANGKAH PERCOBAAN

#### 1. Persiapkan IDE
Mahasiswa membuka Google Colab sebagai lingkungan kerja utama. Google Colab telah menyediakan dukungan lingkungan Python interaktif beserta akses CPU/GPU gratis untuk pengolahan dataset dan komputasi penyeimbangan kelas secara efisien.

#### 2. Instalasi dan Impor Library
Menginstal pustaka `imbalanced-learn` (jika belum tersedia) dan mengimpor seluruh modul yang dibutuhkan:
- `pandas` dan `numpy` untuk manipulasi data tabular.
- `matplotlib.pyplot` dan `seaborn` untuk visualisasi sebaran dan distribusi kelas.
- `RandomUnderSampler`, `RandomOverSampler`, `SMOTE`, dan `ADASYN` dari `imblearn.over_sampling` / `imblearn.under_sampling`.
- Modul pemodelan dan evaluasi dari `sklearn`: `train_test_split`, `DecisionTreeClassifier`, `classification_report`, `confusion_matrix`.

![Gambar 7.4.1 Impor Pustaka](images/gambar-7-4-1-impor-pustaka.png)  
*Gambar 7.4.1 Impor Pustaka*

#### 3. Import dan Load Dataset Information
Memuat *Stroke Prediction Dataset* yang memuat informasi klinis dan demografis pasien (`age`, `gender`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`, serta label target `stroke`):

1. Membaca dataset dan menampilkan 5 baris awal:

![Gambar 7.4.2 Pembacaan Dataset](images/gambar-7-4-2-pembacaan-dataset.png)  
*Gambar 7.4.2 Pembacaan Dataset*

2. Memeriksa nilai kosong (*missing values*) dan tipe data tiap fitur:

![Gambar 7.4.3 Mengecek Null Value](images/gambar-7-4-3-mengecek-null-value.png)  
*Gambar 7.4.3 Mengecek Null Value*

#### 4. Exploratory Data Analysis (EDA)
Memvisualisasikan proporsi distribusi kelas target (`stroke`) menggunakan diagram lingkaran (*pie chart*) untuk memverifikasi tingkat ketidakseimbangan kelas (*class imbalance ratio*).

![Gambar 7.4.4 Distribusi data](images/gambar-7-4-4-distribusi-data.png)  
*Gambar 7.4.4 Distribusi data*

#### 5. Data Cleaning
Ditemukan 201 nilai hilang pada kolom `bmi`. Karena BMI bertipe data numerik kontinu dengan sebaran stabil, penanganan nilai hilang dilakukan dengan imputasi nilai rata-rata (*mean*):

![Gambar 7.4.5 Data Cleaning](images/gambar-7-4-5-data-cleaning.png)  
*Gambar 7.4.5 Data Cleaning*

Memeriksa kembali untuk memastikan tidak ada nilai null yang tersisa:

![Gambar 7.4.6 Mengecek jumlah data null](images/gambar-7-4-6-mengecek-jumlah-data-null.png)  
*Gambar 7.4.6 Mengecek jumlah data null*

#### 6. Random Under-Sampling
Memisahkan fitur ($X$) dan label target ($y$), melakukan *encoding* pada fitur-fitur kategorikal (`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`), serta membagi data menjadi set pelatihan dan pengujian dengan rasio 75:25.

Menerapkan `RandomUnderSampler` pada data pelatihan untuk mereduksi jumlah sampel kelas mayoritas hingga seimbang dengan kelas minoritas:

![Gambar 7.4.7 Random Sampling](images/gambar-7-4-7-random-sampling.png)  
*Gambar 7.4.7 Random Sampling*

Menampilkan data sampel hasil *resampling*:

![Gambar 7.4.8 Melihat data](images/gambar-7-4-8-melihat-data.png)  
*Gambar 7.4.8 Melihat data*

Perbandingan distribusi jumlah sampel sebelum vs sesudah *Random Under-Sampling*:

![Gambar 7.4.9 Perbandingan sebelum random sampling dan sesudah](images/gambar-7-4-9-perbandingan-sebelum-random-sampling-dan-sesudah.png)  
*Gambar 7.4.9 Perbandingan sebelum random sampling dan sesudah*

Visualisasi sebaran titik data (*scatter plot*) kelas mayoritas dan minoritas setelah *under-sampling*:

![Gambar 7.4.10 Visualisasi Scatter](images/gambar-7-4-10-visualisasi-scatter.png)  
*Gambar 7.4.10 Visualisasi Scatter*

#### 7. Random Over-Sampling
Menerapkan `RandomOverSampler` pada data pelatihan untuk menduplikasi sampel kelas minoritas secara acak hingga jumlahnya setara dengan kelas mayoritas:

![Gambar 7.4.11 Random over sampling](images/gambar-7-4-11-random-over-sampling.png)  
*Gambar 7.4.11 Random over sampling*

Perbandingan distribusi jumlah sampel sebelum vs sesudah *Random Over-Sampling*:

![Gambar 7.4.12 Sebelum vs Sesudah over sampling](images/gambar-7-4-12-sebelum-vs-sesudah-over-sampling.png)  
*Gambar 7.4.12 Sebelum vs Sesudah over sampling*

Visualisasi sebaran data setelah *Random Over-Sampling*:

![Gambar 7.4.13 Visualisasi scatter](images/gambar-7-4-13-visualisasi-scatter.png)  
*Gambar 7.4.13 Visualisasi scatter*

#### 8. SMOTE (Synthetic Minority Over-sampling Technique)
Menerapkan algoritma SMOTE untuk membangkitkan data sintetis baru bagi kelas minoritas di sepanjang garis penghubung tetangga terdekat:

![Gambar 7.4.14 SMOTE](images/gambar-7-4-14-smote.png)  
*Gambar 7.4.14 SMOTE*

Perbandingan distribusi jumlah sampel sebelum vs sesudah penerapan SMOTE:

![Gambar 7.4.15 Sebelum vs Sesudah SMOTE](images/gambar-7-4-15-sebelum-vs-sesudah-smote.png)  
*Gambar 7.4.15 Sebelum vs Sesudah SMOTE*

Visualisasi sebaran ruang fitur hasil penambahan data sintetis SMOTE:

![Gambar 7.4.16 Visualisasi Scatter](images/gambar-7-4-16-visualisasi-scatter.png)  
*Gambar 7.4.16 Visualisasi Scatter*

#### 9. ADASYN (Adaptive Synthetic Sampling)
Menerapkan ADASYN yang secara adaptif menghasilkan lebih banyak sampel sintetis pada titik-titik batas kelas minoritas yang sulit dipelajari oleh *classifier*:

![Gambar 7.4.17 ADASYN](images/gambar-7-4-17-adasyn.png)  
*Gambar 7.4.17 ADASYN*

Perbandingan distribusi jumlah sampel sebelum vs sesudah penerapan ADASYN:

![Gambar 7.4.18 Sebelum dan Sesudah Adasyn](images/gambar-7-4-18-sebelum-dan-sesudah-adasyn.png)  
*Gambar 7.4.18 Sebelum dan Sesudah Adasyn*

Visualisasi sebaran ruang fitur hasil pembentukan data sintetis ADASYN:

![Gambar 7.4.19 Visualisasi scatter](images/gambar-7-4-19-visualisasi-scatter.png)  
*Gambar 7.4.19 Visualisasi scatter*

#### 10. Memuat Dataset untuk Seleksi Fitur
Bagian seleksi fitur memakai *Heart Attack Analysis and Prediction Dataset* yang berisi 303 baris dan 14 kolom: 13 fitur prediktor (`age`, `sex`, `cp`, `trtbps`, `chol`, `fbs`, `restecg`, `thalachh`, `exng`, `oldpeak`, `slp`, `caa`, `thall`) dan satu label target `output`. Dataset dibaca langsung dari repositori daring, lalu distribusi labelnya diperiksa.

![Gambar 7.4.20 Pembacaan dataset heart.csv](images/ss-20-baca-heart.png)  
*Gambar 7.4.20 Pembacaan dataset heart.csv*

Label `output` terbagi menjadi 165 sampel kelas 1 dan 138 sampel kelas 0, sehingga dataset ini masih tergolong cukup seimbang dan cocok dipakai untuk menguji pengaruh seleksi fitur tanpa gangguan penyeimbangan kelas.

#### 11. Filter Method dengan ANOVA
*Filter method* menilai relevansi fitur dari sifat statistiknya, tanpa melatih model. Uji ANOVA (`f_classif`) menghitung nilai $F$ dan $p$ untuk setiap fitur terhadap label target. Nilai $F$ yang tinggi menandakan rata-rata fitur tersebut berbeda nyata antar-kelas.

![Gambar 7.4.21 Peringkat uji ANOVA](images/ss-21-anova-ranking.png)  
*Gambar 7.4.21 Peringkat uji ANOVA*

Hasil peringkat menunjukkan lima fitur teratas adalah `exng` ($F = 70{,}9524$), `cp` ($69{,}7723$), `oldpeak` ($68{,}5514$), `thalachh` ($65{,}1201$), dan `caa` ($54{,}5598$). Tiga fitur terbawah yaitu `restecg` ($5{,}7772$), `chol` ($2{,}2030$), dan `fbs` ($0{,}2369$) memiliki $p > 0{,}01$, artinya perbedaan rata-ratanya antar-kelas tidak cukup meyakinkan.

Delapan fitur dengan skor $F$ tertinggi dipilih sebagai subset hasil *filter method*:

![Gambar 7.4.22 Fitur terpilih hasil ANOVA](images/ss-22-fitur-anova.png)  
*Gambar 7.4.22 Fitur terpilih hasil ANOVA*

#### 12. Wrapper Method dengan RFE
*Wrapper method* memakai model *machine learning* sebagai penilai. *Recursive Feature Elimination* (RFE) melatih model, membuang fitur dengan bobot terkecil, lalu mengulang proses tersebut sampai jumlah fitur yang tersisa sesuai target. Pada percobaan ini estimator yang dipakai adalah `DecisionTreeClassifier` dengan target lima fitur.

![Gambar 7.4.23 Fitur terpilih hasil RFE](images/ss-23-fitur-rfe.png)  
*Gambar 7.4.23 Fitur terpilih hasil RFE*

Lima fitur yang bertahan adalah `age`, `cp`, `thalachh`, `caa`, dan `thall`. Susunan ini berbeda dari contoh pada slide perkuliahan (`sex`, `exng`, `slp`, `caa`, `thall`) karena urutan eliminasi bergantung pada pohon yang terbentuk di setiap iterasi. Kedua hasil sama-sama menyisakan lima fitur, hanya pilihan fiturnya bergeser.

#### 13. Embedded Method dengan ExtraTree
*Embedded method* menjalankan seleksi fitur menyatu dengan proses pelatihan model. `ExtraTreesClassifier` menghitung *feature importance* dari penurunan *impurity* saat pemisahan node, sehingga tidak perlu melatih model berulang kali seperti *wrapper*.

![Gambar 7.4.24 Feature importance ExtraTree](images/ss-24-fitur-extratree.png)  
*Gambar 7.4.24 Feature importance ExtraTree*

Enam fitur dengan kepentingan tertinggi adalah `caa` ($0{,}1231$), `cp` ($0{,}1191$), `exng` ($0{,}1030$), `thall` ($0{,}0992$), `oldpeak` ($0{,}0923$), dan `thalachh` ($0{,}0837$). Himpunan ini identik dengan hasil ANOVA, hanya urutannya berbeda.

#### 14. Komparasi Metode Seleksi Fitur dan Pemeriksaan Overfitting
Setiap subset fitur diuji dengan `DecisionTreeClassifier` pada pembagian data yang sama (80:20, `random_state=42`), sehingga perbedaan akurasi hanya berasal dari pilihan fiturnya.

![Gambar 7.4.25 Perbandingan akurasi tiga metode seleksi fitur](images/ss-25-komparasi-seleksi.png)  
*Gambar 7.4.25 Perbandingan akurasi tiga metode seleksi fitur*

Hasilnya: tanpa seleksi (13 fitur) akurasi uji $0{,}754$; *filter* ANOVA (8 fitur) $0{,}820$; *wrapper* RFE (5 fitur) $0{,}820$; *embedded* ExtraTree (6 fitur) $0{,}770$. Pemangkasan fitur menaikkan akurasi uji sampai 6,6 poin persen, dan RFE mencapai hasil tertinggi dengan jumlah fitur paling sedikit.

Akurasi latih yang menyentuh $1{,}000$ pada tiga skenario pertama menunjukkan pohon menghafal data latih. Perbandingan berikut memeriksa gejalanya secara langsung dan menguji pemangkasan kedalaman sebagai penanganannya:

![Gambar 7.4.26 Perbandingan akurasi latih dan uji sebelum dan sesudah pemangkasan](images/ss-26-overfitting-pangkas.png)  
*Gambar 7.4.26 Perbandingan akurasi latih dan uji sebelum dan sesudah pemangkasan*

Pohon penuh tumbuh sampai kedalaman 11 dengan 56 daun, menghasilkan akurasi latih $0{,}996$ tetapi akurasi uji hanya $0{,}770$. Setelah kedalaman dibatasi menjadi 3 (8 daun), akurasi latih turun ke $0{,}843$ dan akurasi uji naik ke $0{,}787$. Jarak antara akurasi latih dan uji menyempit dari $0{,}226$ menjadi $0{,}056$, tanda bahwa pohon yang dipangkas lebih mampu menggeneralisasi.

---

### 7.5 TUGAS DAN ANALISIS
1. **Tugas 1 (Eksplorasi Dataset):**
   - Unduh dataset *Stroke Prediction Dataset*: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
   - Unggah file dataset ke Google Drive atau repositori GitHub masing-masing.
   - Tampilkan dataset ke dalam DataFrame pandas.
   - Jelaskan tujuan penggunaan dataset ini dalam konteks penambangan data.
   - Jelaskan seluruh fitur yang terdapat pada dataset.
2. **Tugas 2 (Komparasi Balancing & Klasifikasi):**
   - Gunakan dataset klasifikasi lain yang bersifat *imbalanced*.
   - Kombinasikan teknik SMOTE atau ADASYN dengan algoritma pengklasifikasi (misalnya Decision Tree, Random Forest, atau SVM).
   - Tampilkan tabel performa model (*Accuracy*, *Precision*, *Recall*, *F1-Score*) sebelum dan sesudah penerapan *data balancing* pada data pelatihan (*training set*).
   - Pilihan dataset yang bisa dipakai antara lain *Wine Quality* (UCI), *Telco Customer Churn*, atau *Pima Indians Diabetes*.

---

### 7.6 REFERENSI
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547–553.
- Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3, 1157–1182.
- Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). Gene selection for cancer classification using support vector machines. *Machine Learning*, 46(1), 389–422.
- Istiana, A., & Mustafiril, M. (2023). Perbandingan Metode Klasifikasi pada Data dengan Imbalance Class dan Missing Value. *JURNAL INFORMATIKA*, 101-108.
- Kohavi, R., & John, G. H. (1997). Wrappers for feature subset selection. *Artificial Intelligence*, 97(1–2), 273–324.
- Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning. *Journal of Machine Learning Research*, 18(17), 1–5.
- Normawati, & Prayogi, T. (2021). Implementasi Naïve Bayes Classifier Dan Confusion Matrix Pada Analisis Sentimen Berbasis Teks Pada Twitter. *J-SAKTI (Jurnal Sains Komputer & Informatika)*, 697-711.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Pratama, R., & Kusnawi, K. (2023). Komparasi Algoritma Supervised Learning dan Feature Selection pada Klasifikasi Penyakit Gagal Jantung. *Indonesian Journal of Computer Science*, 3722-3733.
- Ramadhanti, A. P., Santoso, H. B., & Widiharih, T. (2022). Perbandingan SMOTE dan ADASYN pada Data Imbalance untuk Klasifikasi Rumah Tangga Miskin Kabupaten Temanggung dengan Algoritma K-Nearest Neighbor. *JURNAL GAUSSIAN*, 499-505.
- Septiana, R., Susanto, A., & Tukiyat. (2021). Analisis Sentimen Vaksinasi Covid-19 Pada Twitter Menggunakan Naive Bayes Classifier Dengan Feature Selection Chi-Squared Statistic Dan Particle Swarm Optimization. *Jurnal Sistem Komputer dan Kecerdasan Buatan*, 49-56.
- Tibshirani, R. (1996). Regression shrinkage and selection via the Lasso. *Journal of the Royal Statistical Society: Series B (Methodological)*, 58(1), 267–288.
- Wijayanti, N. L. P., Kencana, I. W., & Sumarjaya, I. M. (2021). SMOTE: Potensi dan Kekurangannya pada Survei. *E-Jurnal Matematika*, 235-240.
