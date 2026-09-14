# Modul Praktikum Penambangan Data – Teknologi Rekayasa Perangkat Lunak – 2025
## PERTEMUAN V: FORECASTING

### 5.1 TUJUAN PEMBELAJARAN
A. Mahasiswa dapat menerapkan metode *forecasting* berbasis *machine learning*, meliputi Neural Network, K-Nearest Neighbors (KNN), Decision Tree, Support Vector Regression (SVR), dan Random Forest, untuk memprediksi nilai periode berikutnya berdasarkan data historis.
B. Mahasiswa dapat melakukan analisis dan perbandingan performa model menggunakan metrik evaluasi *Root Mean Squared Error* (RMSE) dan *Pearson Correlation Coefficient*.
C. Mahasiswa dapat menentukan model *forecasting* yang paling akurat berdasarkan hasil evaluasi menggunakan RMSE dan *Pearson Correlation Coefficient*.

---

### 5.2 DASAR TEORI

**• Supervised Learning**
*Supervised learning* merupakan salah satu teknik *machine learning* yang menggunakan *dataset* (data training) berlabel (*labeled data*) untuk melatih mesin, sehingga mampu mengidentifikasi label input berdasarkan fitur yang dimiliki dan melakukan prediksi atau klasifikasi. Algoritma yang termasuk dalam teknik *supervised learning* antara lain Decision Tree, K-Nearest Neighbor (KNN), Naïve Bayes, Regresi, dan Support Vector Machine (SVM) (Retnoningsih & Pramudita, 2020). Pada algoritma *Supervised Learning*, sistem diberikan *training dataset* berupa informasi masukan dan keluaran yang diinginkan, sehingga sistem akan mempelajari berdasarkan data yang telah ada. Sistem akan mencari pola dari *dataset*, kemudian pola itu akan dijadikan sebagai acuan untuk kumpulan data berikutnya (Santoso, Abijono, & Anggreini, 2021).

**• Forecasting**
*Forecasting* adalah proses memprediksi suatu keadaan di masa kini dan masa depan dengan menguji keadaan di masa lalu. Metode ini banyak digunakan untuk melakukan prediksi dengan berbagai algoritma (Vimala & Nugroho, 2022). Fungsi ramalan adalah membantu pengambilan keputusan berdasarkan pertimbangan terhadap apa yang akan terjadi saat keputusan tersebut dilaksanakan. Ramalan dapat bersifat kualitatif, yaitu tidak berbentuk angka, seperti perkiraan bahwa cuaca besok akan cerah. Selain itu, ramalan juga bisa bersifat kuantitatif, yang berarti dinyatakan dalam bentuk angka atau bilangan.
Peramalan merupakan bagian penting dalam setiap perusahaan atau organisasi bisnis karena berperan dalam pengambilan keputusan manajemen (Maysofa, Syaliman, & Sapriadi, 2023).

**• RMSE**
*Root Mean Squared Error* (RMSE) merupakan salah satu metrik evaluasi yang digunakan untuk mengukur tingkat kesalahan antara nilai aktual dan nilai hasil prediksi. RMSE dihitung dengan cara mengambil akar kuadrat dari rata-rata selisih kuadrat antara nilai aktual dan nilai prediksi. Semakin kecil nilai RMSE, semakin baik performa model karena menunjukkan bahwa model memiliki *error* yang rendah. RMSE banyak digunakan dalam peramalan (*forecasting*) karena sensitif terhadap *error* yang besar, sehingga mampu memberikan gambaran akurasi model secara lebih representatif (Chai & Draxler, 2014).

**• Pearson Correlation Coefficient**
*Pearson Correlation Coefficient* adalah ukuran statistik yang digunakan untuk menentukan tingkat hubungan linear antara dua variabel, yaitu nilai aktual dan nilai prediksi. Nilai koefisien berkisar antara –1 hingga 1, di mana nilai mendekati 1 menunjukkan korelasi positif yang sangat kuat, nilai mendekati –1 menunjukkan korelasi negatif yang kuat, dan nilai mendekati 0 menunjukkan tidak adanya hubungan linear.
Dalam konteks *forecasting*, *Pearson Correlation* digunakan untuk menilai seberapa baik pola prediksi mengikuti pola aktual. Semakin tinggi nilai korelasi, semakin baik model dalam menangkap pola data (Benesty et al., 2009).

---

### 5.3 ALAT DAN BAHAN
**A. Perangkat Keras**
1. Laptop dengan minimal RAM 4 GB
2. Koneksi internet stabil

**B. Perangkat Lunak**
1. Browser (Chrome, Edge, Firefox, dsb)
2. Google Colab / Jupyter Notebook
3. Library ML (pandas, numpy, scikit-learn, matplotlib, dsb)

---

### 5.4 LANGKAH PERCOBAAN

#### 1. Persiapan IDE
Mahasiswa menyiapkan lingkungan kerja dengan membuka Google Colab sebagai platform utama untuk menjalankan eksperimen *forecasting*. Colab dipilih karena mendukung seluruh library *machine learning* yang diperlukan dan mampu menangani komputasi tanpa membebani perangkat mahasiswa.

#### 2. Import Library
Setelah lingkungan siap, mahasiswa mengimpor seluruh library yang diperlukan untuk melakukan *preprocessing*, visualisasi, dan proses *forecasting*. Library utama yang digunakan meliputi:
1. `pandas` dan `numpy` untuk membaca, memanipulasi, dan mengelola data.
2. `matplotlib` dan `seaborn` untuk membuat visualisasi data, seperti tren deret waktu dan grafik hasil prediksi.
3. `MeanSquaredError` / RMSE dan *Pearson correlation* untuk evaluasi performa model.
4. `MLPRegressor`, `KNeighborsRegressor`, `DecisionTreeRegressor`, dan `RandomForestRegressor` untuk melakukan pemodelan *forecasting* dengan berbagai algoritma *supervised learning*.
5. `StandardScaler` atau library pendukung lain apabila dibutuhkan untuk normalisasi atau transformasi data.

> **[Deskripsi Gambar 5.4.1: Impor Pustaka]**
> *Gambar ini menampilkan potongan kode Python (code snippet) yang berisi serangkaian pernyataan `import`. Kode tersebut memuat impor pustaka `pandas` untuk manipulasi dataframe, `numpy` untuk operasi array, `matplotlib` dan `seaborn` untuk visualisasi data, serta berbagai modul dari `sklearn` seperti `MLPRegressor`, `KNeighborsRegressor`, `DecisionTreeRegressor`, `RandomForestRegressor`, `mean_squared_error`, dan `pearsonr` dari `scipy.stats`.*

#### 3. Import Dataset
Mahasiswa menggunakan dataset dari https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset yang dapat diakses melalui *Bike Sharing Dataset*. Dataset ini berisi data peminjaman sepeda per hari, termasuk tanggal, kondisi cuaca, dan total jumlah peminjaman. Pada tahap ini mahasiswa:

1. Mengimpor dataset dan menampilkan 5 baris pertama untuk memastikan dataset terbaca dengan benar.

> **[Deskripsi Gambar 5.4.2: Pembacaan Dataset]**
> *Gambar ini menampilkan output tabel dari perintah `df.head()`. Tabel menunjukkan 5 baris teratas dari dataset `bike-sharing` dengan kolom-kolom seperti `instant`, `dteday`, `season`, `yr`, `mnth`, `holiday`, `weekday`, `workingday`, `weathersit`, `temp`, `atemp`, `hum`, `windspeed`, `casual`, `registered`, dan `cnt`.*

2. Melakukan konversi kolom tanggal ke format datetime.

> **[Deskripsi Gambar 5.4.3: Konversi Tanggal ke Format Datetime]**
> *Gambar ini menampilkan blok kode Python yang menggunakan fungsi `pd.to_datetime()` untuk mengonversi kolom `dteday` (atau kolom tanggal lainnya) dari tipe data string/object menjadi tipe data datetime. Outputnya menunjukkan bahwa kolom tersebut kini memiliki format waktu yang valid (YYYY-MM-DD).*

#### 4. Membentuk Date Time Series
Data *time series* membutuhkan pembentukan input dan output secara berurutan. Oleh karena itu, mahasiswa membuat:
1. **Input (X):** nilai `cnt` selama 7 hari sebelumnya.
2. **Output (y):** jumlah peminjaman sepeda pada hari ke-8.

Pembentukan ini dilakukan dengan teknik *sliding window*, yaitu menggeser jendela data dari hari ke hari untuk menghasilkan pasangan input–output.

> **[Deskripsi Gambar 5.4.4: Fungsi Sliding Window]**
> *Gambar ini menampilkan blok kode Python yang mendefinisikan sebuah fungsi (misalnya `create_sequences`) untuk menerapkan teknik sliding window. Fungsi ini mengambil array data target (seperti `cnt`) dan ukuran window (misalnya 7), lalu melakukan perulangan untuk memotong data menjadi fitur (X) berupa 7 hari berturut-turut dan target (y) berupa hari ke-8.*

#### 5. Menambahkan Fitur Statistik
Agar model memiliki informasi tambahan, mahasiswa melakukan *feature engineering* dengan menambahkan fitur-fitur statistik pada setiap jendela data, seperti:
1. Nilai minimum
2. Nilai maksimum
3. Selisih antara max-min
4. Rata-rata
5. Standar deviasi
6. Median
7. Kurtosis
8. Skewness

> **[Deskripsi Gambar 5.4.5: Fungsi Statistik]**
> *Gambar ini menampilkan blok kode Python yang menghitung berbagai metrik statistik (min, max, max-min, mean, std, median, kurtosis, skewness) untuk setiap window data yang telah dibuat. Hasil perhitungan ini kemudian digabungkan menjadi kolom-kolom fitur tambahan ke dalam DataFrame X untuk memperkaya informasi yang akan dipelajari oleh model.*

#### 6. Pemrosesan Data
Pada tahap ini dilakukan proses pengolahan data agar dapat digunakan sebagai input bagi model-model *machine learning*. Seluruh proses *preprocessing* dilaksanakan dalam satu rangkaian, meliputi:
1. Pembentukan Data *Time Series* Menggunakan *Sliding Window*
2. Pemisahan Data Training dan Testing

> **[Deskripsi Gambar 5.4.6: Preprocessing Dataset]**
> *Gambar ini menampilkan kode untuk memanggil fungsi sliding window yang telah dibuat sebelumnya, menghasilkan array X dan y. Kemudian, data dibagi menjadi data training dan testing menggunakan `train_test_split` dengan rasio tertentu (misalnya 80:20) dan `shuffle=False` untuk mempertahankan urutan waktu.*

> **[Deskripsi Gambar 5.4.7: Preprocessing Dataset]**
> *Gambar ini menampilkan kode untuk melakukan normalisasi atau standardisasi data menggunakan `StandardScaler`. Scaler di-fit hanya pada data training (`X_train`) untuk mencegah data leakage, kemudian di-transformasikan ke `X_train` dan `X_test`.*

> **[Deskripsi Gambar 5.4.8: Preprocessing Dataset]**
> *Gambar ini menampilkan output atau kode yang menunjukkan bentuk akhir dari data training dan testing (seperti `X_train.shape`, `X_test.shape`) setelah melalui seluruh proses preprocessing, memastikan data siap untuk dimasukkan ke dalam model.*

> **[Deskripsi Gambar 5.4.9: Visualisasi Dataset]**
> *Gambar ini menampilkan plot line chart yang memvisualisasikan deret waktu (time series) dari variabel target (jumlah peminjaman sepeda / `cnt`). Sumbu X menunjukkan waktu (tanggal), dan sumbu Y menunjukkan jumlah peminjaman. Garis plot menunjukkan fluktuasi tren peminjaman dari waktu ke waktu.*

#### 7. Pelatihan Model Machine Learning
Mahasiswa melatih empat model *supervised learning* untuk *forecasting*:

1. **Multilayer Perceptron (MLP / Neural Network)**
> **[Deskripsi Gambar 5.4.10: Fungsi Multilayer Perceptron]**
> *Gambar ini menampilkan blok kode Python yang menginisialisasi model `MLPRegressor` dari `sklearn.neural_network`, mengatur parameter seperti jumlah hidden layers, activation function, dan max_iter. Kemudian, model dilatih menggunakan data training dengan method `.fit(X_train, y_train)`.*

2. **K-Nearest Neighbors (KNN)**
> **[Deskripsi Gambar 5.4.11: Fungsi KNN]**
> *Gambar ini menampilkan blok kode Python yang menginisialisasi model `KNeighborsRegressor` dengan parameter jumlah tetangga (n_neighbors). Model kemudian dilatih menggunakan data training dengan method `.fit(X_train, y_train)`.*

3. **Decision Tree (DT)**
> **[Deskripsi Gambar 5.4.12: Fungsi Decision Tree]**
> *Gambar ini menampilkan blok kode Python yang menginisialisasi model `DecisionTreeRegressor` dengan parameter seperti max_depth untuk mencegah overfitting. Model dilatih menggunakan data training dengan method `.fit(X_train, y_train)`.*

4. **Random Forest (RF)**
> **[Deskripsi Gambar 5.4.13: Fungsi Random Forest]**
> *Gambar ini menampilkan blok kode Python yang menginisialisasi model `RandomForestRegressor` dengan parameter default atau yang disesuaikan (seperti n_estimators). Model dilatih menggunakan data training dengan method `.fit(X_train, y_train)`.*

Setiap model dilatih menggunakan data training, kemudian menghasilkan prediksi berdasarkan data testing. Tujuan langkah ini adalah membandingkan performa antar model.

#### 8. Melakukan Prediksi
Setelah model dilatih, masing-masing model menghasilkan nilai prediksi jumlah peminjaman sepeda untuk periode berikutnya berdasarkan input data uji. Prediksi ini kemudian dibandingkan dengan data aktual.

> **[Deskripsi Gambar 5.4.14: Lakukan Prediksi]**
> *Gambar ini menampilkan blok kode Python yang menggunakan method `.predict(X_test)` pada masing-masing model (MLP, KNN, DT, RF) untuk menghasilkan array prediksi (`y_pred_mlp`, `y_pred_knn`, dll.). Hasil prediksi ini kemudian disimpan untuk tahap evaluasi dan visualisasi.*

#### 9. Visualisasi Hasil
Untuk melihat seberapa dekat pola prediksi dengan pola data asli, mahasiswa dapat membuat visualisasi perbandingan antara:
1. Nilai aktual (`y_test`), dan
2. Prediksi model (terutama MLP sebagai model utama).

> **[Deskripsi Gambar 5.4.15: Visualisasi Hasil Prediksi]**
> *Gambar ini menampilkan plot line chart yang membandingkan dua garis: garis pertama (misalnya berwarna biru) merepresentasikan nilai aktual (`y_test`), dan garis kedua (misalnya berwarna oranye) merepresentasikan nilai prediksi dari model (seperti MLP). Kedua garis digambar dalam satu grafik dengan sumbu X berupa indeks waktu dan sumbu Y berupa jumlah peminjaman, menunjukkan seberapa baik model menangkap pola fluktuasi data aktual.*

#### 10. Evaluasi Model
Evaluasi dilakukan menggunakan dua metrik utama:
1. **Root Mean Squared Error (RMSE)** - Mengukur seberapa besar kesalahan prediksi. Semakin kecil RMSE → model makin baik.
2. **Pearson Correlation Coefficient** - Mengukur hubungan linear antara nilai prediksi dan nilai aktual. Semakin mendekati 1 → prediksi semakin mengikuti pola aktual.

Mahasiswa diminta mencatat nilai RMSE dan Pearson dari seluruh model untuk dianalisis.

> **[Deskripsi Gambar 5.4.16: Model Evaluation]**
> *Gambar ini menampilkan blok kode Python dan outputnya yang menghitung metrik evaluasi untuk setiap model. Kode menggunakan `mean_squared_error` yang diakarkan untuk mendapatkan RMSE, dan `pearsonr` dari `scipy.stats` untuk mendapatkan nilai korelasi Pearson. Output menampilkan tabel atau teks yang berisi nilai RMSE dan Pearson Correlation untuk model MLP, KNN, Decision Tree, dan Random Forest.*

---

### 5.5 TUGAS DAN ANALISIS
1. Dengan menggunakan dataset ini: https://www.kaggle.com/datasets/timoboz/tesla-stock-data-from-2010-to-2020/data
2. Buatlah model *forecasting* untuk memprediksi target (`close`) 2 hari ke depan, dengan menggunakan data `close` 7 hari sebelumnya!
3. Buatlah model *forecasting* untuk memprediksi target (`close`) 2 hari ke depan, dengan menggunakan data `close` dan `open` 7 hari sebelumnya!

---

### 5.6 REFERENSI
* Benesty, J., Chen, J., Huang, Y., & Cohen, I. (2009). *Pearson Correlation Coefficient*. Dalam Noise Reduction in Speech Processing (pp. 1–4). Springer. https://doi.org/10.1007/978-3-642-00296-0_5
* Chai, T., & Draxler, R. R. (2014). Root mean square error (RMSE) or mean absolute error (MAE)? Arguments against avoiding RMSE in the literature. *Geoscientific Model Development*, 7(3), 1247–1250. https://doi.org/10.5194/gmd-7-1247-2014
* Maysofa, R., Syaliman, I., & Sapriadi, D. (2023). IMPLEMENTASI FORECASTING PADA PENJUALAN INAURA HAIR CARE DENGAN METODE SINGLE EXPONENTIAL SMOOTHING. *Jurnal Testing dan Implementasi Sistem Informasi*, 82-91.
* Retnoningsih, E., & Pramudita, A. (2020). Mengenal Machine Learning Dengan Teknik Supervised dan Unsupervised Learning Menggunakan Python. *BINA INSANI ICT JOURNAL*, 56-165.
* Santoso, A., Abijono, H., & Anggreini, D. (2021). ALGORITMA SUPERVISED LEARNING DAN UNSUPERVISED LEARNING DALAM PENGOLAHAN DATA. *Jurnal Teknologi Terapan*, 315-318.
* Vimala, T., & Nugroho, S. (2022). FORECASTING PENJUALAN OBAT MENGGUNAKAN METODE SINGLE, DOUBLE, DAN TRIPLE EXPONENTIAL SMOOTHING (STUDI KASUS: APOTEK MANDIRI MEDIKA). *Jurnal Penerapan Teknologi Informasi dan Komunikasi*, 90-99.

Berikut adalah penulisan ulang lengkap dari dokumen *Jupyter Notebook* "Forecasting_Tugas" tanpa diringkas. Kesalahan ketik akibat ekstraksi PDF (seperti spasi yang terputus pada kata *import*, *scipy*, *prediction*, *output*, dll.) telah diperbaiki agar kode Python valid dan teks mudah dibaca. Sesuai permintaan, setiap bagian yang merujuk pada output atau gambar dalam notebook telah diganti dengan **deskripsi detail** mengenai apa yang ditampilkan berdasarkan konteks materi.

***

# Forecasting Model

Pendekatan yang digunakan untuk memprediksi nilai-nilai masa depan berdasarkan data historis atau tren yang ada. Tujuan utama dari model peramalan adalah untuk memberikan perkiraan yang akurat tentang apa yang mungkin terjadi di masa mendatang.

## Import Libraries yang Diperlukan

```python
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import math
from sklearn.metrics import mean_squared_error
from numpy import array
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_log_error
from scipy.stats import kurtosis, skew
```

## Fungsi Sliding Window

Buat fungsi untuk merubah *time series* data menjadi input `X` dan output `y`, dengan teknik *sliding window*.

```python
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
            
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[out_end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
        
    return array(X), array(y)
```

## Fungsi Fitur Statistik

Fungsi tambahan untuk membuat *statistical features*, harapannya bisa meningkatkan akurasi.

```python
def stats_features(input_data):
    inp = list()
    for i in range(len(input_data)):
        inp2 = list()
        inp2 = input_data[i]
        
        min_val = float(np.min(inp2))
        max_val = float(np.max(inp2))
        diff = (max_val - min_val)
        std_val = float(np.std(inp2))
        mean_val = float(np.mean(inp2))
        median_val = float(np.median(inp2))
        kurt = float(kurtosis(inp2))
        sk = float(skew(inp2))
        
        inp2 = np.append(inp2, min_val)
        inp2 = np.append(inp2, max_val)
        inp2 = np.append(inp2, diff)
        inp2 = np.append(inp2, std_val)
        inp2 = np.append(inp2, mean_val)
        inp2 = np.append(inp2, median_val)
        inp2 = np.append(inp2, kurt)
        inp2 = np.append(inp2, sk)
        
        inp = np.append(inp, inp2)
        
    inp = inp.reshape(len(input_data), -1)
    return inp
```

## Fungsi Model Machine Learning

### 1. Model untuk Neural Network (MLP)

```python
def mlp(X_train, X_test, y_train, y_test):
    # mlp = multilayer perceptron / neural network for regression.
    # to setup parameter, please refer to: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    
    mlp_model = MLPRegressor(random_state=42)
    # mlp_model = MLPRegressor(hidden_layer_sizes=(100,100,), max_iter=1000, random_state=42)
    
    # the model learning from training data
    mlp_model.fit(X_train, y_train)
    
    # get the prediction output
    y_pred = mlp_model.predict(X_test)
    y_pred = np.round(y_pred, 0)
    
    # get the root mean square error between prediction and real test data
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    
    # get the pearson correlation
    corr, p_value = pearsonr(y_test, y_pred)
    
    # returning the output of RMSE, corr, and prediction result
    return rmse, corr, y_pred
```

### 2. Model untuk KNN

```python
def knn(X_train, X_test, y_train, y_test):
    # k nearest neighbor for regression.
    # to setup the parameter please refer to: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    
    knn_model = KNeighborsRegressor()
    
    # the model is learning from training data
    knn_model.fit(X_train, y_train)
    
    # get the prediction output
    y_pred = knn_model.predict(X_test)
    y_pred = np.round(y_pred, 0)
    
    # get the root mean square error between prediction and real test data
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    
    # get the pearson correlation
    corr, p_value = pearsonr(y_test, y_pred)
    
    # returning the output of RMSE, corr, and prediction result
    return rmse, corr, y_pred
```

### 3. Model untuk Decision Tree

```python
def dt(X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred, 0)
    
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    
    # get the pearson correlation
    corr, p_value = pearsonr(y_test, y_pred)
    
    # returning the output of RMSE, corr, and prediction result
    return rmse, corr, y_pred
```

### 4. Model untuk SVR

```python
def svm(X_train, X_test, y_train, y_test):
    model = SVR()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred, 0)
    
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    
    # get the pearson correlation
    corr, p_value = pearsonr(y_test, y_pred)
    
    # returning the output of RMSE, corr, and prediction result
    return rmse, corr, y_pred
```

### 5. Model Random Forest

```python
def rf(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred, 0)
    
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    
    # get the pearson correlation
    corr, p_value = pearsonr(y_test, y_pred)
    
    # returning the output of RMSE, corr, and prediction result
    return rmse, corr, y_pred
```

---

## Read Dataset

```python
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/ganjar87/data_science_practice/main/bikesharing_day.csv')
df.head()
```

> **[Deskripsi Output 1: Lima Baris Pertama Dataset (`df.head()`)]**
> *Tabel ini menampilkan 5 baris teratas dari dataset `bikesharing_day.csv`. Kolom yang ditampilkan meliputi: `instant`, `dteday`, `season`, `yr`, `mnth`, `holiday`, `weekday`, `workingday`, `weathersit`, `temp`, `atemp`, `hum`, `windspeed`, `casual`, `registered`, dan `cnt`. Data menunjukkan catatan harian peminjaman sepeda.*

```python
df_ori = df
df_ori['date'] = pd.to_datetime(df_ori['dteday'])
df_ori['cnt'].iloc[:10]
```

> **[Deskripsi Output 2: 10 Data Pertama Kolom Target (`cnt`)]**
> *Output ini menampilkan deret data (Series) dari 10 baris pertama kolom `cnt` (jumlah total peminjaman sepeda). Nilainya berturut-turut adalah: 985, 801, 1349, 1562, 1600, 1606, 1510, 959, 822, dan 1321.*

---

## Data Preprocessing & Sliding Window

```python
df_X = df_ori[['cnt', 'cnt']]
in_seq = df_X.astype(float).values

n_steps_in, n_steps_out = 7, 1
X, y = split_sequences(in_seq, n_steps_in, n_steps_out)

n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

X_train = stats_features(X_train)
X_test = stats_features(X_test)
```

> **[Deskripsi Output 3: Variabel `in_seq`]**
> *Output ini menampilkan array NumPy 2 dimensi yang berisi duplikasi kolom `cnt` (karena `df_X = df_ori[['cnt', 'cnt']]`). Setiap baris berisi dua nilai yang identik, merepresentasikan data input mentah sebelum di-reshape.*

> **[Deskripsi Output 4: Variabel `X` (Fitur)]**
> *Output ini menampilkan array NumPy 2 dimensi hasil dari fungsi `split_sequences` dan `reshape`. Setiap baris mewakili satu sampel data yang terdiri dari 7 nilai historis berturut-turut (window size = 7) yang akan digunakan untuk memprediksi nilai berikutnya.*

> **[Deskripsi Output 5: Bentuk Data `X.shape`]**
> *Output menampilkan tuple `(724, 7)`, yang berarti terdapat 724 sampel data (baris) dan 7 fitur (kolom) dalam array `X` sebelum penambahan fitur statistik.*

> **[Deskripsi Output 6: Sampel Pertama `X[0]`]**
> *Output menampilkan array 1 dimensi yang berisi 7 nilai numerik: `[985., 801., 1349., 1562., 1600., 1606., 1510.]`. Ini adalah 7 hari pertama dari data historis `cnt` yang menjadi input untuk memprediksi hari ke-8.*

> **[Deskripsi Output 7: Target Pertama `y[0]`]**
> *Output menampilkan nilai float `959.0`. Ini adalah nilai target (jumlah peminjaman sepeda) pada hari ke-8 yang sesuai dengan input `X[0]`.*

```python
df_new = df_ori[['date', 'cnt']]
df_new.set_index('date')
```

> **[Deskripsi Output 8: DataFrame `df_new`]**
> *Tabel ini menampilkan dua kolom: `date` (yang telah dikonversi menjadi format datetime dan dijadikan index) dan `cnt` (jumlah peminjaman). Data mencakup rentang waktu dari 2011-01-01 hingga 2012-12-31, total 731 baris.*

---

## Exploratory Data Analysis (EDA)

```python
fig1 = plt.figure(figsize=(20,6))
plt.plot(df_new['cnt'])
plt.ylabel('# total rented bikes', fontsize=20)
plt.legend(loc='upper left', fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
```
*(Catatan: Muncul peringatan `WARNING:matplotlib.legend:No artists with labels found...` karena argumen `label` tidak didefinisikan di dalam `plt.plot()`, namun grafik tetap ter-render).*

> **[Deskripsi Gambar 1: Visualisasi Deret Waktu (Time Series) Total Peminjaman Sepeda]**
> *Gambar ini menampilkan grafik garis (line chart) dengan ukuran figur 20x6. Sumbu X merepresentasikan waktu (tanggal) dari awal 2011 hingga akhir 2012. Sumbu Y merepresentasikan jumlah total sepeda yang disewa (`# total rented bikes`). Garis plot menunjukkan fluktuasi musiman yang jelas, dengan pola naik-turun yang berulang, mengindikasikan adanya tren mingguan atau musiman dalam data peminjaman sepeda.*

---

## Cek Input X dan Output Y (Setelah Preprocessing)

```python
X_test
```
> **[Deskripsi Output 9: Data `X_test`]**
> *Output ini menampilkan array NumPy 2 dimensi yang berisi data fitur untuk pengujian. Setiap baris kini memiliki lebih dari 7 kolom karena fungsi `stats_features` telah menambahkan 8 fitur statistik baru (min, max, diff, std, mean, median, kurtosis, skewness) ke dalam 7 data historis asli, sehingga total kolom menjadi 15.*

```python
y_test
```
> **[Deskripsi Output 10: Data `y_test`]**
> *Output ini menampilkan array 1 dimensi yang berisi nilai target aktual (jumlah peminjaman sepeda) untuk data testing. Array ini berisi ratusan nilai numerik (misalnya: 7286., 5786., 6299., ..., 1796., 2729.) yang akan digunakan sebagai pembanding untuk menilai akurasi prediksi model.*

---

## Panggil Fungsi Machine Learning Model

```python
# calling the function mlp.
# returning rmse, pearson correlation, and prediction output
rmse_mlp, corr_mlp, y_pred_mlp = mlp(X_train, X_test, y_train, y_test)

# calling the function knn
# returning rmse, pearson correlation, and prediction output
rmse_knn, corr_knn, y_pred_knn = knn(X_train, X_test, y_train, y_test)

rmse_dt, corr_dt, y_pred_dt = dt(X_train, X_test, y_train, y_test)

# rmse_svm, corr_svm, y_pred_svm = svm(X_train, X_test, y_train, y_test) # (Dikommentari dalam kode asli)

rmse_rf, corr_rf, y_pred_rf = rf(X_train, X_test, y_train, y_test)
```

---

## Visualisasi

```python
fig1 = plt.figure(figsize=(20,6))

# plotting the result
# because the total number of test data is high, so we just print the first 100 data
# y_test[0:100] get the first 100 data from y_test or real test data.
# y_pred_mlp is the prediction result from MLP

plt.plot(y_test, label='Real data')
plt.plot(y_pred_mlp, label='MLP')

plt.xlabel('Time t', fontsize=20)
plt.ylabel('rented bikes', fontsize=20)
plt.legend(loc='upper left', fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
```

> **[Deskripsi Gambar 2: Visualisasi Perbandingan Nilai Aktual vs Prediksi MLP]**
> *Gambar ini menampilkan grafik garis yang membandingkan dua deret data untuk 100 sampel pertama dari data testing. Garis pertama (biasanya biru, berlabel 'Real data') menunjukkan nilai peminjaman sepeda yang sebenarnya. Garis kedua (biasanya oranye, berlabel 'MLP') menunjukkan nilai yang diprediksi oleh model Neural Network (MLP). Kedua garis tersebut saling mengikuti dengan cukup erat, menunjukkan bahwa model MLP mampu menangkap pola fluktuasi data, meskipun terdapat beberapa deviasi atau selisih pada titik-titik puncak (peak) atau lembah (trough) yang ekstrem.*

---

## Evaluasi Model

```python
# print out the RMSE, pearson correlation coefficient for MLP and KNN
# low RMSE is better
# high pearson correlation coefficient is better.

print('________________________')
print('MLP')
print('RMSE: %.3f' % rmse_mlp)
print('Pearson correlation coefficient: %.3f' % corr_mlp)

print('________________________')
print('KNN')
print('RMSE: %.3f' % rmse_knn)
print('Pearson correlation coefficient: %.3f' % corr_knn)

print('________________________')
print('DT')
print('RMSE: %.3f' % rmse_dt)
print('Pearson correlation coefficient: %.3f' % corr_dt)

print('________________________')
print('RF')
print('RMSE: %.3f' % rmse_rf)
print('Pearson correlation coefficient: %.3f' % corr_rf)
```

> **[Deskripsi Output 11: Hasil Evaluasi Performa Model]**
> *Output teks ini menampilkan metrik evaluasi untuk empat model yang dijalankan:*
> - ***MLP:*** RMSE: 1190.549, Pearson correlation coefficient: 0.778
> - ***KNN:*** RMSE: 1357.710, Pearson correlation coefficient: 0.696
> - ***DT (Decision Tree):*** RMSE: 1854.465, Pearson correlation coefficient: 0.484
> - ***RF (Random Forest):*** RMSE: 1324.452, Pearson correlation coefficient: 0.716
> 
> *Berdasarkan output ini, model MLP memiliki RMSE terendah (kesalahan terkecil) dan koefisien korelasi Pearson tertinggi (hubungan linear terkuat dengan data asli), menjadikannya model dengan performa terbaik di antara keempat model yang diuji pada konfigurasi ini.*

---

## Tugas

1. Dengan menggunakan dataset ini: https://www.kaggle.com/datasets/timoboz/tesla-stock-data-from-2010-to-2020/data
2. Buatlah model *forecasting* untuk memprediksi target (`close`) 2 hari ke depan, dengan menggunakan data `close` 7 hari sebelumnya!
3. Buatlah model *forecasting* untuk memprediksi target (`close`) 2 hari ke depan, dengan menggunakan data `close` dan `open` 7 hari sebelumnya

Berikut adalah penulisan ulang lengkap dari materi presentasi "Session 5 - Regression" tanpa diringkas. Kesalahan ketik akibat ekstraksi PDF (seperti spasi yang terputus pada kata *supervised*, *output*, *overfitting*, *fitting*, dll.) telah diperbaiki agar mudah dibaca. Sesuai permintaan, setiap bagian yang merujuk pada gambar, diagram, atau tabel telah diganti dengan **deskripsi detail** mengenai apa yang ditampilkan berdasarkan konteks materi.

***

# What will We Learn Today?

## Table of Content
1. Regression
2. Linear Regression
3. Decision Tree and Random Forest Regression
4. Evaluation metrics for regression
5. Forecasting

---

## 1. Regression

**Regression**
* **Regression:** metode yang mencoba untuk menentukan kekuatan dan karakter hubungan antara satu variabel dependen dan serangkaian variabel lainnya (dikenal sebagai *independent variables*).
* **Regression algorithms** = *continuous values* (such as price, salary, age, etc).
* **Classification algorithms** = *discrete values* (such as stroke or normal, spam or not spam, etc).
* Keduanya masuk dalam kategori *supervised learning*.

> **[Deskripsi Gambar: Classification, Regression, Clustering]**
> *Diagram ini menampilkan tiga kategori utama dalam machine learning. Dua kotak pertama, "Classification" dan "Regression", dikelompokkan bersama di bawah payung "Supervised Learning" (karena menggunakan data berlabel). Kotak ketiga, "Clustering", berdiri sendiri di bawah kategori "Unsupervised Learning" (karena bekerja dengan data tanpa label untuk menemukan pola atau kelompok).*

---

## 2. Linear Regression

**Linear Regression**
* **Simple Linear Reg** vs **Multiple Linear Reg**

> **[Deskripsi Gambar: Simple vs Multiple Linear Regression]**
> *Diagram ini membandingkan dua jenis regresi linier. Sisi kiri (Simple Linear Regression) menampilkan grafik 2D dengan satu sumbu X (variabel independen) dan satu sumbu Y (variabel dependen), dengan sebuah garis lurus yang memotong titik-titik data. Sisi kanan (Multiple Linear Regression) menampilkan grafik 3D dengan dua sumbu X (variabel independen) dan satu sumbu Y, di mana model membentuk sebuah bidang datar (hyperplane) yang melewati titik-titik data.*

**Linear Regression**
* Membangun hubungan diantara dua variabel dengan garis lurus.
* Variabel independen merupakan variabel yang memengaruhi atau menyebabkan perubahan.
* Variabel dependen adalah variabel yang dipengaruhi atau yang menjadi akibat karena adanya variabel independen.

**Linear Regression**
* Linear regression (regresi linier) mencoba menggambar garis yang paling dekat dengan data dengan menemukan *slope* (kemiringan) dan *intercept*, serta meminimalkan *regression errors*.
* **Ordinary Least Squares (OLS)** adalah metode estimasi yang paling umum untuk model linier.
* Garis optimal yang memberikan nilai *sum of squared errors* (SSE) terendah.

**Example**
* $y$ (dependent variable) = *price* (harga rumah)
* $x$ (independent variable) = *sqft_living* (luas rumah)

> **[Deskripsi Gambar: Contoh Regresi Linier Harga Rumah]**
> *Grafik scatter plot ini menampilkan sumbu X sebagai `sqft_living` (luas rumah dalam kaki persegi) dan sumbu Y sebagai `price` (harga rumah). Titik-titik data tersebar membentuk tren naik. Sebuah garis regresi linier ditarik melalui tengah sebaran data tersebut. Terdapat anotasi khusus yang menunjuk ke titik pada garis regresi di mana $x = 1000$ sqft, dengan label: "Q: Rumah dengan luas 1000 square feet, berapa harganya kira-kira? A: USD 237,562.663".*

**Example**
* **House Sales in King County, USA.**
* Dataset ini berhubungan dengan harga rumah di King County, yang termasuk juga Seattle. Berhubungan dengan rumah yang dijual dari Mei 2014 sampai Mei 2015.
* **Source:** https://www.kaggle.com/harlfoxem/housesalesprediction

**Bias and Variance**
* Linear regression mencari nilai *coefficient* yang meminimalkan nilai *sum of squared errors* (SSE).
* Tetapi mungkin ini bukan model terbaik, karena akan memberikan *coefficient* untuk semua features.
* Termasuk feature yang mempunyai "kemampuan prediksi yang rendah".
* Ini akan menghasilkan model yang "high-variance, low bias".
* **Solusi = regularization**
  * Kita bisa memodifikasi *cost function* untuk memberi batasan nilai *coefficients*.
  * *Reference:* https://towardsdatascience.com/bias-variance-and-regularization-in-linear-regression-lasso-ridge-and-elastic-net-8bf81991d0c5

> **[Deskripsi Gambar: Bias and Variance Trade-off]**
> *Diagram ini mengilustrasikan konsep bias dan varians. Terdapat tiga grafik yang menunjukkan tingkat kompleksitas model: "Low Bias, High Variance" (garis yang sangat berliku mengikuti setiap titik data secara ketat, mengindikasikan overfitting), "High Bias, Low Variance" (garis lurus sederhana yang gagal menangkap pola data, mengindikasikan underfitting), dan "Optimal Balance" (garis lengkung halus yang menangkap tren umum tanpa mengikuti noise). Teks di sampingnya menjelaskan bahwa regularisasi (seperti Lasso, Ridge, atau Elastic Net) menambahkan penalti pada fungsi biaya untuk membatasi besarnya koefisien, sehingga mencegah model menjadi terlalu kompleks (high variance).*

---

## 3. DT and RF Regression

**Decision Tree Regression**
* Decision trees bisa diaplikasikan pada kasus *classification* dan *regression*.
* **Keuntungan:**
  * Mudah dipahami dan di-interpretasikan.
* **Kerugian:**
  * Bisa membuat "over-complex trees" yang tidak bisa *generalise* terhadap data baru. Atau disebut dengan *overfitting*.
  * **Solusi:** *pruning* (pemangkasan pohon).

**Random Forest Regression**
* Random forest adalah algoritma dalam *Supervised Learning* yang menggunakan *ensemble learning method* untuk kasus *classification* dan *regression*.
* Hasil prediksi adalah label terbanyak (untuk kasus *classification*) atau rata-rata hasil prediksi (untuk kasus *regression*) dari model tree yang banyak.

---

## 4. Evaluation metrics for Regression

**Evaluation metrics**
* **Pearson correlation coefficient ($r$):** mengukur kekuatan dan arah hubungan linier antara dua variabel (-1 to 1).
* **Coefficient of determination ($r^2$ or $r$ square):** memberikan proporsi varians (fluktuasi) dari satu variabel yang diprediksi dari variabel lainnya (0 to 1).
* **Root mean square error (RMSE):** merupakan besarnya tingkat kesalahan hasil prediksi. Semakin kecil (mendekati 0) semakin baik (*prediction errors*).

> **[Deskripsi Gambar: Coefficient of Determination ($R^2$)]**
> *Diagram ini menampilkan visualisasi konsep $R^2$. Terdapat sebuah grafik scatter plot dengan garis regresi. Area di sekitar garis regresi diarsir untuk menunjukkan "Explained Variance" (variansi yang dijelaskan oleh model), sedangkan area di luar garis (residuals) diarsir dengan pola berbeda untuk menunjukkan "Unexplained Variance" (variansi yang tidak dijelaskan). Rumus $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$ ditampilkan di sampingnya, menjelaskan bahwa $R^2$ yang mendekati 1 berarti model menjelaskan sebagian besar varians data.*

---

## 5. Forecasting

**Forecasting**
* **Forecasting model:** to predict future trends and outcomes based on historical data.
* *Reference:* https://scied.ucar.edu/learning-zone/climate-change-impacts/predictions-future-global-climate

**One step vs multi step Forecasting**
* **One-step forecast:** model predicts a single value for next time-step.
* **Multi-step forecast:** predicting few times-steps ahead.

> **[Deskripsi Gambar: One step vs Multi step Forecasting]**
> *Diagram ini menampilkan dua skenario peramalan deret waktu (time-series):*
> * *Bagian atas (One step ahead): Menampilkan urutan data historis [24, 25, 27, 28, 30] diikuti oleh satu kotak bertanda tanya [?], yang menunjukkan bahwa model hanya memprediksi satu langkah ke depan.*
> * *Bagian bawah (Multi step ahead): Menampilkan urutan data historis yang sama [24, 25, 27, 28, 30], tetapi diikuti oleh tiga kotak bertanda tanya [?, ?, ?], yang menunjukkan bahwa model memprediksi beberapa langkah ke depan secara berurutan.*

**Contoh Dataset (from CGM Sensor)**

> **[Deskripsi Tabel: Dataset Deret Waktu Glukosa Darah]**
> *Tabel ini menampilkan data deret waktu sederhana dari sensor Continuous Glucose Monitor (CGM). Terdapat dua kolom: "Time" (1 hingga 7) dan "Blood Glucose (mg/dL)" dengan nilai berturut-turut: 100, 125, 140, 120, 130, 160, 150. Data ini digunakan sebagai contoh untuk transformasi menjadi format supervised learning.*

**Sliding Window + Direct Method**
* **Window size** = 5
* **Prediction horizon** = 1 or 3 or 6

> **[Deskripsi Gambar: Sliding Window Direct Method]**
> *Diagram ini mengilustrasikan metode "Direct" untuk peramalan multi-langkah. Sebuah jendela geser (sliding window) berukuran 5 mengambil data historis untuk memprediksi langsung nilai pada horizon waktu tertentu (misalnya, langkah ke-1, ke-3, atau ke-6) menggunakan model yang terpisah atau output langsung, tanpa menggunakan prediksi sebelumnya sebagai input untuk langkah berikutnya.*

**Sliding Window + Iterative Method**
* **Window size** = 5
* **Prediction horizon** = 1

> **[Deskripsi Gambar: Sliding Window Iterative Method]**
> *Diagram ini mengilustrasikan metode "Iterative" (atau Recursive). Model dilatih untuk memprediksi satu langkah ke depan (horizon = 1). Untuk memprediksi langkah ke-2, prediksi dari langkah ke-1 dimasukkan kembali ke dalam jendela geser sebagai data input baru, dan proses ini diulang secara berantai untuk mencapai horizon yang diinginkan.*

**Feature Extraction**

> **[Deskripsi Tabel: Feature Extraction dengan Sliding Window]**
> *Tabel ini menunjukkan transformasi data deret waktu menjadi format supervised learning menggunakan sliding window. Dengan "Window size = 3" dan "Prediction horizon = 1" (atau 2), data historis diubah menjadi fitur (X) dan target (y).*
> * *Baris 1: X = [100, 125, 140], y = 120 (atau 130 jika horizon=2)*
> * *Baris 2: X = [125, 140, 120], y = 130 (atau 160 jika horizon=2)*
> * *Baris 3: X = [140, 120, 130], y = 160 (atau 150 jika horizon=2)*
> * *Proses ini memungkinkan algoritma Machine Learning berbasis regresi (Linear Reg, DT Reg, RF Reg) untuk mempelajari pola dari urutan waktu.*

---

## Let's Practice!

## Thank You