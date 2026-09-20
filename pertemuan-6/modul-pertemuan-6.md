# Modul Praktikum Penambangan Data – Teknologi Rekayasa Perangkat Lunak – 2025
## PERTEMUAN VI: CLUSTERING

### 6.1 TUJUAN PEMBELAJARAN
A. Mahasiswa memahami konsep dasar *unsupervised learning* dan perannya dalam *machine learning*.  
B. Mahasiswa mampu menerapkan berbagai teknik *clustering* seperti K-Means, K-Medoids, dan DBSCAN untuk mengelompokkan data berdasarkan karakteristik tertentu.  
C. Mahasiswa dapat membandingkan kualitas hasil *clustering* menggunakan metrik evaluasi seperti Inertia dan *Silhouette Score*.  
D. Mahasiswa memahami konsep *Association Rule* dan mampu menerapkannya untuk menemukan pola hubungan antar-item dalam dataset.  

---

### 6.2 DASAR TEORI

**• Unsupervised Learning**  
*Unsupervised Learning* merupakan salah satu paradigma pembelajaran mesin di mana model harus menemukan pola tersembunyi dari data tanpa bantuan label eksternal. Pembelajaran ini mengaplikasikan algoritma untuk menganalisis dan menemukan pola dalam data tanpa intervensi atau bantuan manusia. Fokus utamanya adalah eksplorasi struktur data dan pengelompokan menggunakan algoritma seperti K-Means, *Artificial Neural Network* (ANN), dan *Gaussian Mixture Model* (GMM) yang menawarkan fleksibilitas lebih besar meskipun dengan akurasi yang biasanya lebih rendah (Nurhalizah & Ardianto, 2024).  
Algoritma ini banyak digunakan dalam pendeteksian pola dan pemodelan deskriptif untuk membentuk dasar pemahaman karakteristik data. Penggunaan utamanya mencakup *clustering* dan aturan asosiasi (*association rules*). Keuntungan *unsupervised learning* adalah kemampuannya yang fleksibel untuk mencari pola yang mungkin belum diketahui sebelumnya karena tidak bergantung pada anotasi label (Wijoyo, et al., 2024).

**• Clustering**  
*Clustering* atau klasterisasi merupakan suatu teknik atau metode untuk mengelompokkan data ke dalam kelompok-kelompok (*cluster*) tertentu sedemikian rupa sehingga objek-objek di dalam kelompok yang sama memiliki kesamaan yang tinggi, sedangkan objek pada kelompok berbeda memiliki perbedaan yang signifikan. *Clustering* adalah alat penting dalam eksplorasi dan analisis statistik. Saat ini analisis klaster telah banyak digunakan di berbagai bidang seperti ekonomi, psikologi, kesehatan, sosial masyarakat, dan kependudukan. Salah satu metode *clustering* yang paling populer dan banyak diterapkan adalah K-Means (Afidaha & Masrukan, 2023).

**• K-Means**  
Algoritma K-Means merupakan algoritma non-hirarki yang mempartisi dataset ke dalam $K$ buah klaster. Algoritma K-Means dimulai dengan pembentukan partisi awal (pemilihan titik centroid awal), kemudian secara iteratif posisi partisi/centroid ini diperbaiki hingga konvergen atau tidak lagi terjadi perubahan signifikan pada keanggotaan klaster.  
Prinsip utama teknik ini adalah menyusun $K$ buah titik pusat massa (*centroid* / mean) dari sekumpulan data. Tujuan optimasi pengelompokan ini adalah meminimalkan fungsi objektif (*Within-Cluster Sum of Squares* / WCSS), yaitu berusaha meminimalkan variasi di dalam suatu kelompok (*intra-cluster variation*) dan memaksimalkan variasi antar-kelompok (*inter-cluster variation*) (Sulistiyawati & Supriyanto, 2021).

**• K-Medoids**  
Algoritma K-Medoids (disebut juga *Partitioning Around Medoids* / PAM) berperan untuk menemukan *medoid* dalam suatu klaster sebagai titik pusat. Berbeda dengan K-Means yang menggunakan rata-rata (*mean*) titik-titik data (yang bisa berupa titik imajiner akibat sensitivitas terhadap *outlier*), K-Medoids memilih objek nyata dari dataset sebagai representasi pusat klaster (*medoid*).  
Penggunaan objek data aktual dan metrik jarak Manhattan atau Euclidean membuat K-Medoids jauh lebih tahan (*robust*) terhadap keberadaan *noise* dan data pencilan (*outliers*). Secara komputasi, algoritma K-Medoids lebih kompleks daripada K-Means karena memerlukan perhitungan dissimilaritas antar-objek pada setiap pembaruan medoid (Herviany, Delima, Nurhidayah, & Kasini, 2021).

**• DBSCAN**  
*Density-Based Spatial Clustering of Applications with Noise* (DBSCAN) adalah algoritma klasterisasi berbasis kerapatan (*density-based*) yang mengelompokkan titik-titik data berdasarkan kepadatan area di sekitarnya. Algoritma ini menggunakan dua parameter utama:
1. $\epsilon$ (*epsilon*): radius lingkungan pencarian di sekitar titik data.
2. *minPts*: jumlah minimum titik tetangga di dalam radius $\epsilon$ agar sebuah titik dianggap sebagai *core point*.  

DBSCAN unggul dalam mendeteksi klaster dengan bentuk arbitrer (tidak beraturan / non-sferis) serta mampu memisahkan *noise* atau *outlier* secara otomatis tanpa mengharuskan penentuan jumlah klaster $K$ di awal (Ester et al., 1996; Han, Kamber, & Pei, 2011).

**• Association Rule**  
*Association Rule Mining* merupakan teknik dalam *data mining* untuk menemukan pola asosiasi atau aturan keterkaitan antar-item dalam dataset transaksi. Aturan asosiasi umumnya direpresentasikan dalam bentuk implikasi $A \rightarrow B$, yang berarti bahwa kehadiran item $A$ cenderung disertai oleh kehadiran item $B$. Penilaian kualitas aturan didasarkan pada tiga metrik:
1. **Support**: frekuensi kemunculan kombinasi item dalam total transaksi.
2. **Confidence**: kepastian atau probabilitas kemunculan item $B$ ketika item $A$ muncul.
3. **Lift**: ukuran kekuatan aturan dibanding kejadian independen.  

Metode ini sangat umum diterapkan pada *Market Basket Analysis* untuk merancang strategi penempatan produk dan promosi (Agrawal, Imieliński, & Swami, 1993; Han, Kamber, & Pei, 2011).

**• Inertia**  
Inertia adalah metrik evaluasi internal pada algoritma K-Means yang mengukur kerapatan (*compactness*) dari sebuah klaster. Nilai inertia dihitung sebagai jumlah kuadrat jarak antara setiap titik data ke centroid klaster terdekatnya (*Within-Cluster Sum of Squares* / WCSS):
$$Inertia = \sum_{i=0}^{n} \min_{\mu_j \in C} (||x_i - \mu_j||^2)$$
Nilai inertia yang lebih rendah menunjukkan bahwa klaster lebih padat. Metrik ini dijadikan dasar dalam *Elbow Method* untuk menentukan jumlah klaster optimal (Bishop, 2006).

**• Silhouette Score**  
*Silhouette Score* merupakan metode evaluasi yang mengukur seberapa baik suatu objek ditempatkan di dalam klasternya dibandingkan dengan klaster lainnya. Untuk suatu sampel $i$, rumusnya:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
di mana $a(i)$ adalah rata-rata jarak ke semua titik dalam klaster yang sama (*intra-cluster distance*), dan $b(i)$ adalah rata-rata jarak ke titik dalam klaster terdekat lainnya (*nearest-cluster distance*).  
Nilai *silhouette score* berada pada rentang -1 hingga 1:
- Mendekati 1: Klaster terpisah sangat baik dan kompak.
- Mendekati 0: Terjadi tumpang tindih (*overlap*) antar-klaster.
- Negatif: Data kemungkinan ditempatkan pada klaster yang salah (Rousseeuw, 1987; Tan, Steinbach, & Kumar, 2019).

---

### 6.3 ALAT DAN BAHAN

**A. Perangkat Keras**  
1. Laptop dengan minimal RAM 4 GB  
2. Koneksi internet stabil  

**B. Perangkat Lunak**  
1. Browser web (Google Chrome, Microsoft Edge, Mozilla Firefox, dll.)  
2. Google Colab / Jupyter Notebook  
3. Pustaka Python: `pandas`, `numpy`, `scikit-learn`, `scikit-learn-extra`, `kneed`, `matplotlib`, `seaborn`  

---

### 6.4 LANGKAH PERCOBAAN

#### 1. Persiapkan IDE
Mahasiswa memulai percobaan dengan membuka Google Colab sebagai lingkungan pengembangan utama. Colab dipilih karena mendukung berbagai pustaka *machine learning* yang diperlukan, seperti `pandas`, `numpy`, `scikit-learn`, `matplotlib`, hingga `scikit-learn-extra` untuk K-Medoids. Selain itu, Colab menyediakan komputasi GPU/CPU gratis sehingga proses clustering dapat berjalan efisien.

#### 2. Instalasi dan Impor Library
Setelah lingkungan siap, mahasiswa mengimpor seluruh pustaka yang dibutuhkan:
1. `pandas` dan `numpy` untuk manipulasi dan pembersihan data tabular.
2. `StandardScaler` untuk standarisasi skala fitur.
3. `KMeans` dari `sklearn.cluster` dan `KMedoids` dari `sklearn_extra.cluster`.
4. `matplotlib.pyplot` dan `seaborn` untuk visualisasi sebaran dan pola klaster.
5. `silhouette_score` dari `sklearn.metrics` untuk evaluasi kualitas klaster.
6. `KneeLocator` dari pustaka `kneed` untuk mendeteksi titik siku (*elbow/knee point*) secara otomatis.

![Gambar 6.4.1 Impor Pustaka](images/gambar-6-4-1-impor-pustaka.png)  
*Gambar 6.4.1 Impor Pustaka*

#### 3. Read dan Load Dataset Information
Mahasiswa memuat dataset *Credit Card Dataset for Clustering* ke dalam DataFrame. Dataset ini memuat riwayat transaksi pemegang kartu kredit: saldo (*BALANCE*), pembelian (*PURCHASES*), penarikan tunai (*CASH_ADVANCE*), limit kredit (*CREDIT_LIMIT*), pembayaran minimum (*MINIMUM_PAYMENTS*), dll.

1. Mengimpor dataset dan menampilkan lima baris teratas:

![Gambar 6.4.2 Membaca dataset](images/gambar-6-4-2-membaca-dataset.png)  
*Gambar 6.4.2 Membaca dataset*

2. Melihat struktur dan tipe data dengan `.info()` serta ringkasan statistik dengan `.describe()`:

![Gambar 6.4.3 Melihat struktur Dataset](images/gambar-6-4-3-melihat-struktur-dataset.png)  
*Gambar 6.4.3 Melihat struktur Dataset*

3. Memeriksa korelasi antar-kolom numerik menggunakan matriks korelasi atau visualisasi *heatmap*:

![Gambar 6.4.4 Melihat korelasi antar kolom](images/gambar-6-4-4-melihat-korelasi-antar-kolom.png)  
*Gambar 6.4.4 Melihat korelasi antar kolom*

#### 4. Data Cleaning
1. Menghapus kolom identitas `CUST_ID` karena tidak memiliki makna analitik untuk proses klasterisasi:

![Gambar 6.4.5 Menghapus kolom CUST_ID](images/gambar-6-4-5-menghapus-kolom-cust-id.png)  
*Gambar 6.4.5 Menghapus kolom CUST_ID*

2. Mengidentifikasi jumlah nilai kosong (*null values*) pada setiap fitur:

![Gambar 6.4.6 Menghitung jumlah kolom null](images/gambar-6-4-6-menghitung-jumlah-kolom-null.png)  
*Gambar 6.4.6 Menghitung jumlah kolom null*

3. Melakukan imputasi nilai kosong pada kolom `MINIMUM_PAYMENTS` dan `CREDIT_LIMIT` menggunakan nilai median:

![Gambar 6.4.7 Imputasi Data](images/gambar-6-4-7-imputasi-data.png)  
*Gambar 6.4.7 Imputasi Data*

4. Memeriksa kembali dataset untuk memastikan sudah tidak ada nilai hilang yang tersisa:

![Gambar 6.4.8 Cek hasil Imputasi Data](images/gambar-6-4-8-cek-hasil-imputasi-data.png)  
*Gambar 6.4.8 Cek hasil Imputasi Data*

#### 5. Standarisasi Data
Melakukan standarisasi data menggunakan `StandardScaler` agar seluruh fitur memiliki nilai rata-rata (*mean*) = 0 dan varians = 1. Langkah ini krusial karena algoritma clustering berbasis jarak sangat rentan didominasi oleh fitur dengan rentang nilai besar.

![Gambar 6.4.9 Data Scaling](images/gambar-6-4-9-data-scaling.png)  
*Gambar 6.4.9 Data Scaling*

#### 6. Menentukan Jumlah Cluster (Elbow Method & Silhouette Score)

##### 1. Elbow Method
Menjalankan pemodelan dengan variasi nilai $K$ dari 1 hingga 10 lalu mencatat nilai *inertia* (WCSS). Titik belokan siku (*knee point*) menunjukkan penambahan klaster berikutnya tidak lagi memberikan reduksi inertia yang signifikan.

**a. K-Means:**

![Gambar 6.4.10 Elbow Method](images/gambar-6-4-10-elbow-method.png)  
*Gambar 6.4.10 Elbow Method*

Visualisasi grafik Elbow Method K-Means:

![Gambar 6.4.11 Visualisasi Elbow Method](images/gambar-6-4-11-visualisasi-elbow-method.png)  
*Gambar 6.4.11 Visualisasi Elbow Method*

Menentukan titik optimal (*knee point*) secara otomatis menggunakan `KneeLocator` dari pustaka `kneed`:

![Gambar 6.4.12 Menentukan titik optimal menggunakan kneed](images/gambar-6-4-12-menentukan-titik-optimal-menggunakan-kneed.png)  
*Gambar 6.4.12 Menentukan titik optimal menggunakan kneed*

Visualisasi grafik Knee Point pada K-Means:

![Gambar 6.4.13 Visualisasi Knee Point](images/gambar-6-4-13-visualisasi-knee-point.png)  
*Gambar 6.4.13 Visualisasi Knee Point*

**b. K-Medoids:**

Menghitung nilai *cost* / inertia untuk variasi nilai $K$ pada K-Medoids:

![Gambar 6.4.14 Mencari nilai k elbow method](images/gambar-6-4-14-mencari-nilai-k-elbow-method.png)  
*Gambar 6.4.14 Mencari nilai k elbow method*

Visualisasi kurva Elbow Method K-Medoids:

![Gambar 6.4.15 Visualisasi elbow method](images/gambar-6-4-15-visualisasi-elbow-method.png)  
*Gambar 6.4.15 Visualisasi elbow method*

Visualisasi titik Knee Point pada kurva K-Medoids:

![Gambar 6.4.16 Visualisasi Knee Point](images/gambar-6-4-16-visualisasi-knee-point.png)  
*Gambar 6.4.16 Visualisasi Knee Point*

##### 2. Silhouette Score
Menghitung koefisien *silhouette* untuk $K = 2$ hingga $10$. Nilai *silhouette score* tertinggi menunjukkan pemisahan klaster terbaik.

**a. K-Means:**

![Gambar 6.4.17 Menentukan silhouette score](images/gambar-6-4-17-menentukan-silhouette-score.png)  
*Gambar 6.4.17 Menentukan silhouette score*

Visualisasi nilai Silhouette Score pada K-Means:

![Gambar 6.4.18 Visualisasi Solhouette score](images/gambar-6-4-18-visualisasi-solhouette-score.png)  
*Gambar 6.4.18 Visualisasi Silhouette score*

**b. K-Medoids:**

Menghitung nilai *silhouette score* untuk model K-Medoids:

![Gambar 6.4.19 Kmedoid](images/gambar-6-4-19-kmedoid.png)  
*Gambar 6.4.19 Kmedoid*

Visualisasi grafik Silhouette Score pada K-Medoids:

![Gambar 6.4.20 Visualisasi Kmedoid](images/gambar-6-4-20-visualisasi-kmedoid.png)  
*Gambar 6.4.20 Visualisasi Kmedoid*

#### 7. Menerapkan Algoritma K-Means
Setelah jumlah klaster optimal dipilih:
1. Melatih model K-Means menggunakan nilai $K$ optimal:

![Gambar 6.4.21 Training K-Mean sesuai kluster optimal](images/gambar-6-4-21-training-k-mean-sesuai-kluster-optimal.png)  
*Gambar 6.4.21 Training K-Mean sesuai kluster optimal*

2. Menampilkan koordinat *centroids* dan menambahkan kolom label klaster ke dalam DataFrame:

![Gambar 6.4.22 Cek Centroids](images/gambar-6-4-22-cek-centroids.png)  
*Gambar 6.4.22 Cek Centroids*

3. Visualisasi sebaran klaster (*scatter plot*) antar-fitur:

![Gambar 6.4.23 Visualisasi K Means Clustering](images/gambar-6-4-23-visualisasi-k-means-clustering.png)  
*Gambar 6.4.23 Visualisasi K Means Clustering*

![Gambar 6.4.24 Visualisasi klustering](images/gambar-6-4-24-visualisasi-klustering.png)  
*Gambar 6.4.24 Visualisasi klustering*

![Gambar 6.4.25 Visualisasi Scatter](images/gambar-6-4-25-visualisasi-scatter.png)  
*Gambar 6.4.25 Visualisasi Scatter*

#### 8. Menerapkan Algoritma K-Medoids
1. Melakukan evaluasi jumlah klaster menggunakan inertia K-Medoids:

![Gambar 6.4.26 Evaluasi jumlah clluster menggunakan inertia](images/gambar-6-4-26-evaluasi-jumlah-clluster-menggunakan-inertia.png)  
*Gambar 6.4.26 Evaluasi jumlah klaster menggunakan inertia*

2. Melatih model K-Medoids dan menetapkan label klaster ke data:

![Gambar 6.4.27 Pelatihan model K-Medoids](images/gambar-6-4-27-pelatihan-model-k-medoids.png)  
*Gambar 6.4.27 Pelatihan model K-Medoids*

3. Visualisasi hasil klasterisasi K-Medoids:

![Gambar 6.4.28 Visualisasi K-Medoids clustering](images/gambar-6-4-28-visualisasi-k-medoids-clustering.png)  
*Gambar 6.4.28 Visualisasi K-Medoids clustering*

![Gambar 6.4.29 Visualisasi Kmedoid Scatter](images/gambar-6-4-29-visualisasi-kmedoid-scatter.png)  
*Gambar 6.4.29 Visualisasi Kmedoid Scatter*

#### 9. Analisis dan Interpretasi Cluster
Melakukan agregasi nilai rata-rata per klaster untuk memahami profil nasabah:
- Rata-rata saldo (`BALANCE`)
- Rata-rata total pembelian (`PURCHASES`)
- Total pembayaran (`PAYMENTS`)
- Frekuensi transaksi tunai (`CASH_ADVANCE_TRX`)

![Gambar 6.4.30 Interpretasi Cluster](images/gambar-6-4-30-interpretasi-cluster.png)  
*Gambar 6.4.30 Interpretasi Cluster*

Ringkasan segmentasi dan profil klaster:

![Gambar 6.4.31 Analisis Cluster](images/gambar-6-4-31-analisis-cluster.png)  
*Gambar 6.4.31 Analisis Cluster*

---

### 6.5 TUGAS DAN ANALISIS
1. **Dataset Preparation:**
   - Unduh dataset *Credit Card Dataset for Clustering*: https://www.kaggle.com/datasets/arjunbhasin2013/ccdata
   - Unggah ke Google Drive atau repositori GitHub masing-masing.
   - Jelaskan tujuan penggunaan dataset ini.
   - Jelaskan karakteristik fitur-fitur yang ada di dalam dataset.
2. **Eksperimen Dataset Tambahan:**
   - Gunakan dataset lain untuk menerapkan algoritma *clustering* K-Means dan K-Medoids.
   - Dataset referensi: *Air Traffic Passengers Statistics*: https://data.sfgov.org/Transportation/Air-Traffic-Passenger-Statistics/rkru-6vcg/about_data

---

### 6.6 REFERENSI
- Afidaha, M. W., & Masrukan, M. H. (2023). Penerapan Metode Clustering dengan Algoritma K-Means untuk Pengelompokkan Data Migrasi Penduduk Tiap Kecamatan di Kabupaten Rembang. *PRISMA, Prosiding Seminar Nasional Matematika* (pp. 729-738). Semarang: Jurusan Matematika, Universitas Negeri Semarang.
- Agrawal, R., Imieliński, T., & Swami, A. (1993). Mining association rules between sets of items in large databases. *Proceedings of the 1993 ACM SIGMOD International Conference on Management of Data*, 207–216.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD)*, 226–231.
- Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann.
- Herviany, N., Delima, R., Nurhidayah, S., & Kasini, A. (2021). Perbandingan Algoritma K-Means dan K-Medoids untuk Pengelompokkan Daerah Rawan Tanah Longsor di Provinsi Jawa Barat. *MALCOM: Indonesian Journal of Machine Learning and Computer Science*, 34-40.
- Nurhalizah, S., & Ardianto, D. (2024). Penerapan Unsupervised Learning dalam Analisis Pola Data. *Jurnal Sains Komputer dan Informatika*.
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.
- Sulistiyawati, A., & Supriyanto, E. (2021). Pengelompokan Data Menggunakan Metode K-Means untuk Analisis Cluster. *Jurnal Matematika dan Statistika*.
- Tan, P.-N., Steinbach, M., & Kumar, V. (2019). *Introduction to Data Mining* (2nd ed.). Pearson.
- Wijoyo, S., et al. (2024). Unsupervised Learning and Its Application in Pattern Discovery. *Jurnal Teknologi Informasi dan Sains Data*.
- scikit-learn. (2024). *Clustering Performance Evaluation – Inertia & Silhouette Score*. Diakses dari dokumentasi resmi scikit-learn.
