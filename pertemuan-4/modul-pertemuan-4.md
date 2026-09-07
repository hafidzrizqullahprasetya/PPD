# Modul Praktikum Penambangan Data – Teknologi Rekayasa Perangkat Lunak – 2025
## PERTEMUAN IV: Regression

### 4.1. TUJUAN PEMBELAJARAN
A. Mahasiswa mampu memahami konsep dasar regresi.
B. Mahasiswa mengetahui prinsip kerja berbagai algoritma regresi, khususnya Linear Regression, Decision Tree Regression, dan Random Forest Regression.
C. Mahasiswa dapat menerapkan berbagai algoritma regresi untuk membangun model.
D. Mahasiswa mampu mengevaluasi performa model regresi menggunakan metrik RMSE, R², dan koefisien korelasi (R).

---

### 4.2. DASAR TEORI

**• Regresi**
Regresi adalah metode statistik dalam *machine learning* yang digunakan untuk memodelkan hubungan antara satu atau lebih variabel independen (fitur) dengan variabel dependen (target) yang bersifat numerik atau kontinu. Tujuan utama regresi adalah untuk memahami pola hubungan antara fitur dan target, memperkirakan nilai masa depan berdasarkan data sebelumnya, dan mengidentifikasi seberapa besar pengaruh masing-masing fitur terhadap *output*.

Dalam *machine learning*, regresi termasuk dalam *supervised learning*, karena model dilatih menggunakan pasangan *input–output* sehingga bisa belajar memprediksi nilai target yang belum diketahui. Contoh penggunaan regresi antara lain yaitu prediksi harga rumah, estimasi permintaan barang, prediksi curah hujan, dan evaluasi risiko. Regresi bekerja dengan cara menemukan fungsi atau model matematis yang paling sesuai dengan pola data. Model tersebut kemudian digunakan untuk memprediksi nilai baru berdasarkan *input* baru.

**• Algoritma Regresi**
Algoritma regresi yang umum digunakan adalah Linear Regression, Decision Tree Regression, dan Random Forest Regression. Masing-masing algoritma memiliki karakteristik dan cara kerja yang berbeda sesuai jenis data dan pola hubungan antar variabel.

1. **Linear Regression**
   Linear Regression adalah algoritma regresi paling dasar yang mengasumsikan bahwa hubungan antara fitur dan target bersifat linier. Model ini berusaha membangun sebuah garis atau *hyperplane* terbaik yang meminimalkan *error* antara prediksi dan nilai sebenarnya. Berikut merupakan persamaan linear regression:
   $$y = b_0 + b_1x_1 + b_2x_2 + \dots + b_nx_n$$
   Dimana:
   * $b_0$ = *intercept* (nilai dasar)
   * $b_n$ = koefisien regresi
   * $x_n$ = nilai fitur
   * $y$ = target/prediksi

   Model linear regression mencari koefisien terbaik menggunakan teknik *Least Squares*, yaitu metode yang meminimalkan jumlah kuadrat *error*. Kelebihan linear regression yaitu sangat cepat dan efisien untuk dataset besar, mudah dipahami dan dijelaskan (*interpretable*), cocok untuk hubungan yang mendekati linear. Sedangkan kekurangannya adalah kurang cocok untuk pola non-linear dan sensitif terhadap *outlier*.

2. **Decision Tree Regression**
   Decision Tree Regression merupakan salah satu algoritma regresi yang bekerja dengan membangun struktur pohon keputusan sebagai representasi model. Algoritma ini membagi data menjadi beberapa subset berdasarkan titik *split* yang dianggap paling optimal dalam mengurangi *error*. Setiap pembagian *node* dilakukan secara rekursif hingga mencapai *leaf node* yang berisi nilai prediksi. Pemilihan *split* biasanya dilakukan berdasarkan pengukuran *error* seperti *Mean Squared Error* (MSE) sehingga pemisahan data menghasilkan kelompok dengan variasi nilai target yang lebih homogen.

   Decision Tree Regression sangat efektif untuk memodelkan hubungan non-linear karena struktur pohon memungkinkan model membuat batas keputusan yang kompleks. Selain itu, decision tree tidak memerlukan normalisasi atau *scaling* data dan dapat menangani fitur numerik maupun kategorikal. Meskipun demikian, algoritma ini memiliki kelemahan utama berupa kecenderungan *overfitting*, yaitu ketika model terlalu menyesuaikan diri dengan data *training* sehingga performanya menurun saat diuji pada data baru.

3. **Random Forest Regression**
   Random Forest Regression merupakan pengembangan dari Decision Tree yang menggunakan pendekatan *ensemble learning*, yaitu membangun banyak pohon keputusan dan menggabungkan prediksinya. Setiap pohon dibangun menggunakan sampel data yang dipilih secara acak melalui *bootstrap sampling*, serta subset fitur yang juga dipilih secara acak pada setiap pemisahan *node*. Proses ini menciptakan berbagai variasi pohon yang kemudian memberikan hasil prediksi lebih stabil dan tidak mudah *overfitting* dibanding *tree* tunggal.

   Prediksi akhir pada Random Forest biasanya diperoleh dari rata-rata hasil prediksi semua pohon, sehingga model mampu menangkap pola kompleks tanpa mengorbankan akurasi. Selain itu, Random Forest menyediakan informasi mengenai *feature importance*, yang menunjukkan fitur mana yang paling berpengaruh dalam proses prediksi. Hal ini membuat Random Forest menjadi salah satu algoritma regresi yang umum digunakan, terutama pada dataset yang besar dan memiliki banyak fitur. Meski demikian, model ini memerlukan sumber daya komputasi yang lebih besar.

**• Metrik Evaluasi Regresi**
Evaluasi model regresi dilakukan menggunakan metrik khusus yang mampu mengukur kesalahan prediksi nilai numerik. Untuk menilai performa model regresi, digunakan dua metrik utama.

1. **RMSE (Root Mean Squared Error)**
   RMSE mengukur rata-rata jarak antara nilai prediksi dan nilai sebenarnya dengan memberikan penalti lebih besar terhadap kesalahan yang besar. RMSE merupakan akar dari *Mean Squared Error*, sehingga unitnya sama dengan unit target dan lebih mudah diinterpretasikan dalam masalah nyata. Nilai RMSE yang lebih kecil menandakan bahwa model menghasilkan prediksi yang lebih akurat.
   $$RMSE = \sqrt{\frac{1}{n}\sum(y_{pred} - y_{true})^2}$$
   RMSE yang memiliki nilai rendah menunjukkan bahwa model mampu menghasilkan prediksi yang lebih mendekati nilai sebenarnya, sehingga tingkat akurasinya semakin baik. Karena RMSE memberikan penalti yang lebih besar pada kesalahan yang ekstrem, metrik ini menjadi sensitif terhadap *outlier*. Sensitivitas tersebut justru menjadikan RMSE cocok digunakan pada data kontinu seperti harga rumah, di mana perbedaan nilai yang besar dapat memberikan dampak signifikan terhadap evaluasi performa model.

2. **Coefficient of Determination (R²)**
   *Coefficient of Determination* atau R² menunjukkan seberapa besar proporsi variansi target yang dapat dijelaskan oleh model. Nilai R² berada pada rentang 0 hingga 1, di mana nilai mendekati 1 menandakan bahwa model memiliki kemampuan prediksi yang baik dan dapat menangkap pola hubungan dengan baik. Berbeda dengan metrik klasifikasi seperti akurasi atau *precision*, RMSE dan R² merupakan metrik yang dirancang khusus untuk kasus regresi, karena mampu menilai kualitas prediksi numerik secara tepat.
   $$0 \le R^2 \le 1$$
   Nilai R² yang semakin mendekati 1 menunjukkan bahwa model mampu menjelaskan proporsi variansi target dengan lebih baik, sehingga performanya dapat dikatakan semakin sesuai dengan pola data sebenarnya. Sebaliknya, jika nilai R² rendah, berarti model kurang mampu menangkap hubungan antara fitur dan target. Karena sifatnya yang mengukur seberapa besar variasi data yang dapat dijelaskan oleh model, R² sangat berguna dalam menilai tingkat kecocokan model regresi, terutama ketika ingin memahami seberapa representatif model tersebut terhadap pola keseluruhan dalam dataset.

---

### 4.3. ALAT DAN BAHAN
**• Perangkat Keras**
* Komputer/Laptop

**• Perangkat Lunak**
* Python 3
* Google Colaboratory

---

### 4.4. LANGKAH PERCOBAAN

#### 1. Impor Pustaka
> **[Deskripsi Gambar 4.4.1: Impor Library]**
> *Gambar ini menampilkan potongan kode Python (code snippet) yang berisi serangkaian pernyataan `import`. Kode tersebut memuat impor pustaka `pandas` untuk manipulasi dataframe, `numpy` untuk operasi array dan matematika, `matplotlib` dan `seaborn` untuk visualisasi data, serta berbagai modul dari `sklearn` seperti `train_test_split`, `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, `mean_squared_error`, dan `r2_score`.*

#### 2. Load Dataset
Dataset dapat diunduh pada link berikut: https://www.kaggle.com/datasets/shivachandel/kc-house-data

```python
df = pd.read_csv('https://raw.githubusercontent.com/ganjar87/data_science_practice/main/kc_house_data.csv')
```

#### 3. Exploratory Data Analysis (EDA)

**a. Melihat 5 baris pertama dan terakhir pada dataset dengan `df.head()` dan `df.tail()`.**

> **[Deskripsi Gambar 4.4.2: Lima Baris Pertama Dataset]**
> *Gambar ini menampilkan output tabel dari perintah `df.head()`. Tabel menunjukkan 5 baris teratas dari dataset `kc_house_data` dengan kolom-kolom: `id`, `date`, `price`, `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `zipcode`, `lat`, `long`, `sqft_living15`, dan `sqft_lot15`.*

> **[Deskripsi Gambar 4.4.3: Lima Baris Terakhir Dataset]**
> *Gambar ini menampilkan output tabel dari perintah `df.tail()`. Tabel menunjukkan 5 baris paling bawah dari dataset yang memiliki struktur kolom yang sama dengan gambar sebelumnya, memberikan gambaran tentang variasi data di akhir kumpulan dataset.*

**b. Melihat deskripsi statistik dengan `df.describe()`.**

> **[Deskripsi Gambar 4.4.4: Hasil Deskripsi Statistik Dataset]**
> *Gambar ini menampilkan output tabel statistik deskriptif. Tabel berisi baris metrik (count, mean, std, min, 25%, 50%, 75%, max) dan kolom berupa variabel numerik seperti `price`, `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, dll. Nilai mean dan standar deviasi memberikan gambaran sebaran data numerik, misalnya rata-rata harga rumah (`price`) dan luas bangunan (`sqft_living`).*

**c. Melihat histogram**

> **[Deskripsi Gambar 4.4.5: Kode untuk Menampilkan Histogram]**
> *Gambar ini menampilkan blok kode Python yang menggunakan fungsi bawaan pandas/matplotlib (`df.hist(figsize=(10,10))`) untuk memplot histogram bagi setiap kolom bertipe numerik dalam dataset.*

> **[Deskripsi Gambar 4.4.6: Histogram Seluruh Variabel Numerik]**
> *Gambar ini menampilkan grid yang berisi beberapa subplot histogram. Setiap subplot merepresentasikan distribusi frekuensi dari variabel numerik. Histogram `price` menunjukkan distribusi yang sangat condong ke kanan (*right-skewed*) dengan banyak outlier di nilai tinggi, sementara variabel seperti `bedrooms` atau `bathrooms` menunjukkan distribusi diskrit yang terkonsentrasi pada nilai-nilai kecil.*

**d. Mengecek korelasi atribut numerik dengan `df.corr(numeric_only=True)`**

> **[Deskripsi Gambar 4.4.7: Korelasi Atribut Numerik]**
> *Gambar ini menampilkan output tabel matriks korelasi (Correlation Matrix). Tabel menampilkan nilai korelasi Pearson antar seluruh variabel numerik. Misalnya, terlihat korelasi yang cukup kuat dan positif antara `price` dan `sqft_living` (sekitar 0.70), atau `price` dan `sqft_above`, yang mengindikasikan bahwa semakin luas rumah, semakin tinggi harganya.*

**e. Mengecek missing value dengan `df.isnull().sum()`**

> **[Deskripsi Gambar 4.4.8: Pengecekan Missing Value]**
> *Gambar ini menampilkan output teks dari perintah `df.isnull().sum()`. Output tersebut mendaftar seluruh kolom beserta jumlah nilai kosongnya. Hasil menunjukkan bahwa seluruh kolom pada dataset ini memiliki 0 nilai kosong (tidak ada missing value), sehingga data sudah bersih dari segi kekosongan nilai.*

**f. Mengecek atribut kategorikal**

> **[Deskripsi Gambar 4.4.9: Pengecekan Atribut Kategorikal]**
> *Gambar ini menampilkan output dari perintah pengecekan tipe data (seperti `df.select_dtypes(include=['object']).columns`). Output menyoroti kolom `date` sebagai satu-satunya kolom yang bertipe data objek/kategorikal, yang mengindikasikan bahwa kolom ini perlu ditangani lebih lanjut (misalnya dikonversi menjadi format datetime atau dihapus jika tidak informatif).*

---

#### 4. Modelling

**a. Linear Regression**

> **[Deskripsi Gambar 4.4.10: Kode untuk Membuat Model Linear Regression]**
> *Gambar ini menampilkan blok kode Python yang panjang dan terstruktur untuk tahap pemodelan Linear Regression. Kode tersebut mencakup:*
> 1. *Pemisahan fitur (`X`) dengan menghapus kolom `id`, `date`, dan `price`, serta target (`y`) yang berisi kolom `price`.*
> 2. *Konversi `X` dan `y` menjadi numpy array bertipe float.*
> 3. *Pembagian data training dan testing (70:30) menggunakan `train_test_split` dengan `random_state=42`.*
> 4. *Inisialisasi model `LinearRegression()` dan proses `.fit()` untuk melatih model.*
> 5. *Perhitungan nilai R² untuk data training dan testing.*
> 6. *Pengecekan parameter model menggunakan `reg.coef_` dan `reg.intercept_`.*

* **Menentukan fitur (x) dan target (y)**
  Proses pemodelan dimulai dengan menentukan fitur dan target yang akan digunakan oleh model. Pada kode ini, variabel `df_X` dibuat dengan menghapus kolom `id`, `date`, dan `price` dari dataset, karena kolom `price` akan menjadi target prediksi. Sementara kolom `id` dan `date` tidak bersifat informatif. Variabel `df_y` dibuat dengan kolom `price` sebagai label yang ingin diprediksi oleh model.
* **Mengonversi data ke dalam bentuk numerik**
  Setelah fitur dan target dipisahkan, kedua variabel tersebut diubah menjadi array numerik menggunakan `.astype(float).values`.
* **Membagi data menjadi data training dan testing**
  Tahap berikutnya adalah membagi data menjadi dua bagian, yaitu data training dan data testing menggunakan fungsi `train_test_split`. Pada kode ini, 70% data digunakan untuk melatih model dan 30% sisanya untuk menguji performa model. Parameter `random_state=42` digunakan agar pembagian data tetap konsisten setiap kali kode dijalankan.
* **Melatih model**
  Selanjutnya, objek model Linear Regression dibuat dengan `reg = LinearRegression()`. Model kemudian dilatih menggunakan data training melalui perintah `reg.fit(X_train, y_train)`, di mana model mempelajari hubungan linier antara fitur dan target.
* **Mengukur kemampuan model**
  Setelah tahap training selesai, nilai *coefficient of determination* atau R² dihitung baik untuk data training maupun testing.
* **Melihat parameter model**
  Model Linear Regression menghasilkan dua parameter, yaitu koefisien dan *intercept*. Koefisien menunjukkan pengaruh masing-masing fitur terhadap nilai prediksi, sedangkan *intercept* merupakan nilai dasar ketika semua fitur bernilai nol. Kedua parameter ini ditampilkan dalam kode melalui `reg.coef_` dan `reg.intercept_`.

* **Membuat prediksi model**
  Tahap selanjutnya adalah membuat prediksi pada data testing menggunakan `reg.predict(X_test)`. Hasil prediksi kemudian disimpan dalam variabel `y_pred`, kemudian ditampilkan untuk melihat 10 prediksi pertama. Untuk memastikan hasilnya masuk akal, nilai prediksi tersebut dibandingkan dengan 10 nilai sebenarnya dari `y_test`. Langkah ini memberikan gambaran awal mengenai akurasi model sebelum dilakukan evaluasi lebih lanjut menggunakan metrik seperti RMSE atau R².

> **[Deskripsi Gambar 4.4.11: Hasil Prediksi Model Linear Regression]**
> *Gambar ini menampilkan output array dari `y_pred` yang berisi 10 nilai prediksi harga rumah pertama, serta output array dari `y_test` yang berisi 10 nilai harga rumah sebenarnya. Perbandingan sekilas menunjukkan bahwa nilai prediksi memiliki kedekatan dengan nilai aktual, meskipun terdapat selisih (error) pada beberapa sampel.*

Tahap selanjutnya adalah evaluasi model dengan metrik RMSE dan R² untuk mengukur kualitas prediksi dari model regresi.

> **[Deskripsi Gambar 4.4.12: Kode Evaluasi Model Menggunakan RMSE dan R²]**
> *Gambar ini menampilkan blok kode Python untuk menghitung metrik evaluasi. Kode menggunakan fungsi `mean_squared_error(y_test, y_pred)` dari sklearn untuk mendapatkan nilai MSE, yang kemudian diakarkan menggunakan `np.sqrt(mse)` untuk mendapatkan RMSE. Selain itu, kode juga menggunakan fungsi `r2_score(y_test, y_pred)` untuk menghitung nilai R².*

Pertama, nilai *mean squared error* (MSE) dihitung menggunakan fungsi `mean_squared_error(y_test, y_pred)`. Nilai MSE menggambarkan rata-rata kuadrat selisih antara nilai prediksi dengan nilai sebenarnya. Semakin kecil nilai MSE, semakin baik performa model. Selanjutnya, nilai *Root Mean Squared Error* (RMSE) diperoleh dengan mengambil akar kuadrat dari MSE menggunakan `np.sqrt(mse)`. RMSE memberikan interpretasi kesalahan dalam satuan yang sama dengan target, sehingga lebih mudah untuk dibandingkan.

Kemudian, performa model juga dievaluasi menggunakan R² (*coefficient of determination*) melalui fungsi `r2_score(y_test, y_pred)`. Nilai R² menunjukkan seberapa besar variansi data yang dapat dijelaskan oleh model. Nilai yang mendekati 1 berarti model memiliki kemampuan prediksi yang sangat baik, sedangkan nilai mendekati 0 menandakan model kurang mampu menjelaskan pola hubungan dalam data.

> **[Deskripsi Gambar 4.4.13: Hasil Evaluasi Menggunakan Metrik RMSE dan R²]**
> *Gambar ini menampilkan output teks yang berisi nilai numerik dari metrik evaluasi. Nilai RMSE yang dihasilkan adalah sebesar 208296.72, dan nilai R² adalah sebesar 0.699.*

Berdasarkan hasil evaluasi, model menghasilkan nilai RMSE sebesar 208296.72, yang berarti rata-rata kesalahan prediksi model berada di kisaran angka tersebut terhadap nilai harga rumah sebenarnya. Sementara itu, nilai R² sebesar 0.699 menunjukkan bahwa model mampu menjelaskan sekitar 69.9% variansi dalam data harga rumah.

Nilai ini mengindikasikan bahwa model Linear Regression memiliki kemampuan yang cukup baik dalam memodelkan hubungan antara fitur dan harga rumah, meskipun masih terdapat sekitar 30% variansi yang belum dapat dijelaskan oleh model.

Tahap selanjutnya, dilakukan visualisasi untuk membandingkan antara nilai prediksi model dan nilai sebenarnya dari dataset testing.

> **[Deskripsi Gambar 4.4.14: Kode Visualisasi Perbandingan Nilai Prediksi dan Nilai Sebenarnya]**
> *Gambar ini menampilkan blok kode Python untuk membuat visualisasi. Kode mengambil 50 sampel awal dari `y_test` dan `y_pred`, mengubahnya menjadi objek Series, dan menggabungkannya ke dalam sebuah DataFrame bernama `df_new` dengan kolom `real values` dan `predicted values`. Kemudian, grafik batang dibuat menggunakan `df_new.plot(kind='bar', figsize=(15,3))`.*

> **[Deskripsi Gambar 4.4.15: Grafik Perbandingan Nilai Prediksi dan Nilai Sebenarnya]**
> *Gambar ini menampilkan grafik batang (bar chart) yang membandingkan 50 nilai `real values` (biru) dan `predicted values` (oranye). Terlihat bahwa pola batang prediksi secara umum mengikuti fluktuasi batang nilai sebenarnya, namun terdapat deviasi atau selisih ketinggian batang pada beberapa titik yang merepresentasikan error prediksi model.*

---

**b. Decision Tree Regression**

> **[Deskripsi Gambar 4.4.16: Kode untuk Membuat Model Decision Tree Regression]**
> *Gambar ini menampilkan blok kode untuk pemodelan Decision Tree. Kode mencakup pemisahan fitur dan target, konversi ke numpy array, pembagian data (70:30), inisialisasi model `DecisionTreeRegressor(max_depth=10)`, proses `.fit()`, perhitungan R² untuk train dan test, serta prediksi menggunakan `.predict()`.*

* **Menentukan fitur (x) dan target (y)**
  Proses pemodelan dimulai dengan menentukan fitur dan target yang akan digunakan oleh model. Pada kode ini, variabel `df_X` dibuat dengan menghapus kolom `id`, `date`, dan `price` dari dataset, karena kolom `price` akan menjadi target prediksi. Sementara kolom `id` dan `date` tidak bersifat informatif. Variabel `df_y` dibuat dengan kolom `price` sebagai label yang ingin diprediksi oleh model.
* **Mengonversi data ke dalam bentuk numerik**
  Setelah fitur dan target dipisahkan, kedua variabel tersebut diubah menjadi array numerik menggunakan `.astype(float).values`.
* **Membagi data menjadi data training dan testing**
  Tahap berikutnya adalah membagi data menjadi dua bagian, yaitu data training dan data testing menggunakan fungsi `train_test_split`. Pada kode ini, 70% data digunakan untuk melatih model dan 30% sisanya untuk menguji performa model. Parameter `random_state=42` digunakan agar pembagian data tetap konsisten setiap kali kode dijalankan.
* **Melatih model**
  Model Decision Tree Regression dibuat dengan objek `DecisionTreeRegressor(max_depth=10)` yang berfungsi sebagai representasi pohon keputusan untuk tugas regresi. Parameter `max_depth=10` digunakan untuk membatasi kedalaman maksimum pohon sehingga model tidak terlalu kompleks dan dapat mengurangi risiko *overfitting*. Setelah objek model dibuat, tahap training dilakukan dengan memanggil `dt.fit(X_train, y_train)` sehingga model mempelajari pola hubungan antara fitur pada data latih dan nilai targetnya.
* **Mengukur kemampuan model**
  Setelah tahap training selesai, nilai *coefficient of determination* atau R² dihitung baik untuk data training maupun testing.
* **Membuat prediksi model**
  Setelah model selesai dilatih menggunakan data training, langkah berikutnya adalah membuat prediksi terhadap data yang belum pernah dilihat model, yaitu data testing. Proses ini dilakukan menggunakan perintah `dt.predict(X_test)`, yang menghasilkan serangkaian nilai hasil prediksi dan disimpan dalam variabel `y_pred`.

> **[Deskripsi Gambar 4.4.17: Hasil Prediksi Model Decision Tree Regression]**
> *Gambar ini menampilkan output array dari `y_pred` untuk Decision Tree, yang berisi 10 nilai prediksi harga rumah pertama. Nilai-nilai ini menunjukkan estimasi yang dihasilkan oleh pohon keputusan.*

Selanjutnya dilakukan evaluasi model menggunakan metrik seperti RMSE dan R².

> **[Deskripsi Gambar 4.4.18: Kode Evaluasi Model Menggunakan RMSE dan R²]**
> *Gambar ini menampilkan blok kode Python yang identik dengan evaluasi sebelumnya, menggunakan `mean_squared_error`, `np.sqrt()`, dan `r2_score` untuk menghitung metrik evaluasi model Decision Tree.*

> **[Deskripsi Gambar 4.4.19: Hasil Evaluasi Model]**
> *Gambar ini menampilkan output teks yang berisi nilai numerik RMSE dan R² untuk model Decision Tree Regression. Nilai R² untuk Decision Tree umumnya lebih tinggi dibandingkan Linear Regression, menunjukkan kemampuan model yang lebih baik dalam menangkap pola non-linear pada data.*

Tahap selanjutnya, dilakukan visualisasi untuk membandingkan antara nilai prediksi model dan nilai sebenarnya dari dataset testing.

> **[Deskripsi Gambar 4.4.20: Kode Visualisasi Perbandingan Nilai Prediksi dan Nilai Sebenarnya]**
> *Gambar ini menampilkan blok kode Python yang mengambil 50 sampel awal dari data testing dan prediksi, lalu memplotnya dalam bentuk grafik batang menggunakan `df_new.plot(kind='bar')`.*

> **[Deskripsi Gambar 4.4.21: Grafik Perbandingan Nilai Prediksi dan Nilai Sebenarnya]**
> *Gambar ini menampilkan grafik batang perbandingan 50 nilai `real values` dan `predicted values` untuk model Decision Tree. Terlihat bahwa pola prediksi lebih rapat mengikuti nilai sebenarnya dibandingkan model Linear Regression, menandakan error yang lebih kecil pada beberapa titik.*

---

**c. Random Forest Regression**

> **[Deskripsi Gambar 4.4.22: Kode untuk Membuat Model Random Forest Regression]**
> *Gambar ini menampilkan blok kode untuk pemodelan Random Forest. Kode mencakup pemisahan fitur dan target, konversi ke numpy array, pembagian data (70:30), inisialisasi model `RandomForestRegressor()` dengan parameter default, proses `.fit()`, perhitungan R² untuk train dan test, serta prediksi menggunakan `.predict()`.*

* **Menentukan fitur (x) dan target (y)**
  Proses pemodelan dimulai dengan menentukan fitur dan target yang akan digunakan oleh model. Pada kode ini, variabel `df_X` dibuat dengan menghapus kolom `id`, `date`, dan `price` dari dataset, karena kolom `price` akan menjadi target prediksi. Sementara kolom `id` dan `date` tidak bersifat informatif. Variabel `df_y` dibuat dengan kolom `price` sebagai label yang ingin diprediksi oleh model.
* **Mengonversi data ke dalam bentuk numerik**
  Setelah fitur dan target dipisahkan, kedua variabel tersebut diubah menjadi array numerik menggunakan `.astype(float).values`.
* **Membagi data menjadi data training dan testing**
  Tahap berikutnya adalah membagi data menjadi dua bagian, yaitu data training dan data testing menggunakan fungsi `train_test_split`. Pada kode ini, 70% data digunakan untuk melatih model dan 30% sisanya untuk menguji performa model. Parameter `random_state=42` digunakan agar pembagian data tetap konsisten setiap kali kode dijalankan.
* **Melatih model**
  Tahap pemodelan Random Forest Regression dimulai dengan membentuk objek model melalui `RandomForestRegressor()`, yang secara *default* akan membuat sekumpulan pohon keputusan sebagai dasar prediksi. Model ini kemudian dilatih menggunakan data training dengan memanggil `rf.fit(X_train, y_train)`, sehingga setiap pohon dalam *ensemble* mempelajari pola hubungan antara fitur dan harga rumah dari subset data yang berbeda.
* **Mengukur kemampuan model**
  Setelah tahap training selesai, nilai *coefficient of determination* atau R² dihitung baik untuk data training maupun testing.
* **Membuat prediksi model**
  Tahap berikutnya adalah membuat prediksi menggunakan `rf.predict(X_test)`, yang menghasilkan nilai estimasi harga rumah dengan data testing.

> **[Deskripsi Gambar 4.4.23: Hasil Prediksi Model Random Forest Regression]**
> *Gambar ini menampilkan output array dari `y_pred` untuk Random Forest, yang berisi 10 nilai prediksi harga rumah pertama. Nilai-nilai ini merupakan hasil rata-rata dari seluruh pohon keputusan di dalam ensemble.*

Selanjutnya dilakukan evaluasi model menggunakan metrik seperti RMSE dan R².

> **[Deskripsi Gambar 4.4.24: Kode Evaluasi Model Menggunakan RMSE dan R²]**
> *Gambar ini menampilkan blok kode Python untuk menghitung metrik evaluasi RMSE dan R² khusus untuk model Random Forest Regression.*

> **[Deskripsi Gambar 4.4.25: Hasil Evaluasi Model]**
> *Gambar ini menampilkan output teks yang berisi nilai numerik RMSE dan R² untuk model Random Forest Regression. Model ini umumnya menghasilkan nilai RMSE terendah dan R² tertinggi di antara ketiga model, membuktikan keunggulan metode ensemble dalam menangkap pola data yang kompleks.*

Tahap selanjutnya, dilakukan visualisasi untuk membandingkan antara nilai prediksi model dan nilai sebenarnya dari dataset testing.

> **[Deskripsi Gambar 4.4.26: Kode Visualisasi Perbandingan Nilai Prediksi dan Nilai Sebenarnya]**
> *Gambar ini menampilkan blok kode Python yang mengambil 50 sampel awal dari data testing dan prediksi Random Forest, lalu memplotnya dalam bentuk grafik batang.*

> **[Deskripsi Gambar 4.4.27: Grafik Perbandingan Nilai Prediksi dan Nilai Sebenarnya]**
> *Gambar ini menampilkan grafik batang perbandingan 50 nilai `real values` dan `predicted values` untuk model Random Forest. Terlihat bahwa batang `predicted values` hampir berimpit atau sangat mendekati batang `real values`, menunjukkan bahwa model Random Forest memiliki tingkat error yang paling rendah dan akurasi prediksi yang paling tinggi dibandingkan model lainnya.*

---

### 4.5. TUGAS & ANALISIS
**Dataset:** https://www.kaggle.com/datasets/hellbuoy/car-price-prediction/data

1. Jelaskan apa tujuan penggunaan dataset ini dan definisikan atribut yang menjadi input dan *output*-nya!
2. Sebutkan variabel yang paling mempengaruhi *output* atau mempunyai nilai korelasi tinggi!
3. Silahkan membuat model regresi dengan Linear Regression, Decision Tree Regression, Random Forest Regression dari dataset diatas!
4. Buatlah tabel yang menjelaskan performa dari model *machine learning* untuk kasus dataset diatas! Kolom pertama "model", kolom selanjutnya "RMSE, R2, R".
5. Jelaskan dari hasil eksperimen di atas, model mana yang paling baik? Jelaskan alasan anda!

---

### 4.6. REFERENSI
* Han, J., Pei, J., & Tong, H. (2022). *Data Mining: Concepts and Techniques* (4th ed.). Morgan Kaufmann/Elsevier.

# Regression

Model regresi adalah alat statistik yang digunakan untuk memahami dan memodelkan hubungan antara satu atau lebih variabel independen (prediktor) dan satu variabel dependen (variabel respons) dengan cara menemukan persamaan matematis yang menggambarkan pola hubungan tersebut. Tujuan utama dari model regresi adalah untuk menggambarkan dan memprediksi nilai variabel dependen berdasarkan nilai variabel independen yang diberikan, dengan asumsi bahwa hubungan tersebut linier. Model regresi dapat digunakan untuk menganalisis dan memahami tren, mengidentifikasi faktor yang mempengaruhi variabel dependen, serta memperkirakan atau memprediksi nilai-nilai di masa depan.

---

## Tugas 1
1. Download dataset di https://www.kaggle.com/datasets/shivachandel/kc-house-data dan simpan di gdrive anda atau repository lain misal github anda!
2. Jelaskan apa tujuan dari penggunaan dataset ini? Jelaskan atribut nya!
3. Definisikan mana input dan output nya!

---

## Read the Dataset

```python
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/ganjar87/data_science_practice/main/kc_house_data.csv')
df.head()
```

> **[Deskripsi Gambar 1: Lima Baris Pertama Dataset (`df.head()`)]**
> *Gambar ini menampilkan output tabel yang berisi 5 baris teratas dari dataset `kc_house_data`. Kolom yang ditampilkan meliputi: `id`, `date`, `price`, `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, hingga `grade` dan `sqft_above`. Data menunjukkan variasi nilai numerik untuk fitur-fitur properti.*

```python
df.tail()
```

> **[Deskripsi Gambar 2: Lima Baris Terakhir Dataset (`df.tail()`)]**
> *Gambar ini menampilkan output tabel yang berisi 5 baris paling bawah dari dataset (indeks 21608 hingga 21612). Struktur kolomnya sama dengan gambar sebelumnya, memberikan gambaran tentang variasi data di akhir kumpulan dataset.*

```python
# untuk melihat statistical details
df.describe()
```

> **[Deskripsi Gambar 3: Hasil Deskripsi Statistik Dataset (`df.describe()`)]**
> *Gambar ini menampilkan tabel statistik deskriptif untuk variabel numerik. Baris berisi metrik: `count`, `mean`, `std`, `min`, `25%`, `50%`, `75%`, dan `max`. Kolom mencakup `id`, `price`, `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, dan `view`. Terlihat bahwa rata-rata harga (`price`) adalah sekitar 540.088, dan rata-rata `sqft_living` adalah sekitar 2079.*

---

## Check Histogram

```python
import matplotlib.pyplot as plt
# check histogram for continuous columns
df.hist(figsize=(10,10))
plt.show()
```

> **[Deskripsi Gambar 4: Kode untuk Menampilkan Histogram]**
> *Gambar ini menampilkan blok kode Python yang menggunakan fungsi bawaan pandas/matplotlib untuk memplot histogram bagi setiap kolom bertipe numerik dalam dataset.*

> **[Deskripsi Gambar 5: Histogram Seluruh Variabel Numerik]**
> *Gambar ini menampilkan grid yang berisi beberapa subplot histogram. Setiap subplot merepresentasikan distribusi frekuensi dari variabel numerik. Histogram `price` menunjukkan distribusi yang sangat condong ke kanan (*right-skewed*) dengan banyak outlier di nilai tinggi, sementara variabel seperti `bedrooms` atau `bathrooms` menunjukkan distribusi diskrit yang terkonsentrasi pada nilai-nilai kecil.*

---

## Check Correlation & Missing Values

```python
# check correlation coef
df.corr(numeric_only=True)
```

> **[Deskripsi Gambar 6: Matriks Korelasi Atribut Numerik]**
> *Gambar ini menampilkan output tabel matriks korelasi (Correlation Matrix). Tabel menampilkan nilai korelasi Pearson antar seluruh variabel numerik. Misalnya, terlihat korelasi yang cukup kuat dan positif antara `price` dan `sqft_living` (0.702035), `grade` (0.667434), dan `sqft_above` (0.605567), yang mengindikasikan bahwa semakin luas dan tinggi grade rumah, semakin tinggi harganya.*

```python
df.isnull().sum()
```

> **[Deskripsi Gambar 7: Pengecekan Missing Value]**
> *Gambar ini menampilkan output teks dari perintah `df.isnull().sum()`. Output tersebut mendaftar seluruh kolom (dari `id` hingga `sqft_lot15`) beserta jumlah nilai kosongnya. Hasil menunjukkan bahwa seluruh kolom pada dataset ini memiliki 0 nilai kosong (tidak ada missing value).*

---

## Check Categorical Attributes

```python
df_X = df.drop(['id', 'date', 'price'], axis=1)
df_y = df['price']
cats = df_X.select_dtypes(include=['object', 'bool']).columns
print(cats)
```

> **[Deskripsi Gambar 8: Pengecekan Atribut Kategorikal]**
> *Gambar ini menampilkan output `Index([], dtype='object')`. Hal ini menunjukkan bahwa setelah kolom `id`, `date`, dan `price` di-drop, tidak ada lagi kolom yang bertipe data objek atau boolean di dalam `df_X`. Semua fitur yang tersisa sudah berupa numerik.*

---

## Linear Regression

Model linear regresi adalah pendekatan statistik yang digunakan untuk memodelkan hubungan linier antara satu atau lebih variabel independen (prediktor) dan satu variabel dependen (variabel respons). Model ini didasarkan pada asumsi bahwa hubungan antara variabel-variabel tersebut dapat dijelaskan dengan menggunakan persamaan garis lurus, di mana perubahan dalam variabel independen secara proporsional terkait dengan perubahan dalam variabel dependen. Dalam model linear regresi, tujuan utama adalah untuk menemukan koefisien regresi yang terbaik yang meminimalkan selisih antara nilai-nilai yang diamati dari variabel dependen dan nilai-nilai yang diprediksi oleh model. Dengan menggunakan teknik *least squares* atau metode lainnya, model linear regresi dapat digunakan untuk melakukan prediksi, penjelajahan pola, dan pemahaman hubungan antarvariabel dalam berbagai bidang.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df_X = df.drop(['id','date','price'], axis=1)
df_y = df['price']

X = df_X.astype(float).values
y = df_y.astype(float).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

print('coef of determination training', reg.score(X_train, y_train))
print('coef of determination testing', reg.score(X_test, y_test))

print('coefficient')
print(reg.coef_)
print('intercept')
print(reg.intercept_)

print('prediction')
y_pred = reg.predict(X_test)
print(y_pred[:10])
print('real value')
print(y_test[0:10])
```

> **[Deskripsi Gambar 9: Kode untuk Membuat Model Linear Regression]**
> *Gambar ini menampilkan blok kode Python untuk pemodelan Linear Regression. Kode mencakup pemisahan fitur dan target, konversi ke numpy array, pembagian data (70:30), inisialisasi model `LinearRegression()`, proses `.fit()`, perhitungan R² (score) untuk train dan test, pengecekan koefisien dan intercept, serta prediksi 10 data pertama.*

> **[Deskripsi Gambar 10: Output Pelatihan dan Parameter Model Linear Regression]**
> *Gambar ini menampilkan output teks dari kode di atas:*
> * *coef of determination training: 0.6995155846436756*
> * *coef of determination testing: 0.6994627057969875*
> * *coefficient: Array berisi bobot untuk setiap fitur (misal: -3.43e+04, 4.03e+04, dst).*
> * *intercept: 6641646.708118638*
> * *prediction: Array berisi 10 nilai prediksi harga rumah pertama.*
> * *real value: Array berisi 10 nilai harga rumah sebenarnya untuk perbandingan.*

### Tugas 1 (lanjutan)
4. Jelaskan apa itu coefficient determination untuk training dan testing!
5. Jelaskan apa itu coefficient dan intercept!
6. Jelaskan apa itu prediction value dan real value!
7. Untuk menghitung performa model regresi, bisa menggunakan RMSE dan R2. Jelaskan alasannya, dan kenapa tidak menggunakan accuracy precision recall?
8. Dari tiga model regresi, mana yang performanya paling bagus? Jelaskan alasannya!

```python
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('rmse:', rmse)
print('r2:', r2)
```

> **[Deskripsi Gambar 11: Kode Evaluasi Model Linear Regression]**
> *Gambar ini menampilkan blok kode Python untuk menghitung metrik evaluasi. Kode menggunakan `mean_squared_error` untuk mendapatkan MSE, mengakarkannya untuk RMSE, dan menggunakan `r2_score` untuk R².*

> **[Deskripsi Gambar 12: Hasil Evaluasi Model Linear Regression]**
> *Gambar ini menampilkan output teks:*
> * *rmse: 208296.72772118967*
> * *r2: 0.6994627057969875*

```python
import seaborn as sns
import matplotlib.pyplot as plt

data1 = pd.Series(y_test[:50].ravel())
data2 = pd.Series(y_pred[:50].ravel())

df_new = pd.concat([data1, data2], keys=['real values', 'predicted values'], axis=1)
# df_new.plot.bar() # bisa pakai cara 1
df_new.plot(kind='bar', figsize=(15,3)) # bisa pakai cara 2

plt.title("Linear regression")
plt.xlabel('Sample i')
plt.ylabel('House Price')
plt.show()
```

> **[Deskripsi Gambar 13: Kode Visualisasi Perbandingan Linear Regression]**
> *Gambar ini menampilkan blok kode Python yang mengambil 50 sampel awal dari data testing dan prediksi, lalu memplotnya dalam bentuk grafik batang.*

> **[Deskripsi Gambar 14: Grafik Batang Perbandingan Nilai Prediksi dan Sebenarnya (Linear Regression)]**
> *Gambar ini menampilkan grafik batang (bar chart) yang membandingkan 50 nilai `real values` dan `predicted values`. Terlihat bahwa pola batang prediksi secara umum mengikuti fluktuasi batang nilai sebenarnya, namun terdapat deviasi atau selisih ketinggian batang pada beberapa titik yang merepresentasikan error prediksi model.*

---

## Decision Tree Regression

Model decision tree regression adalah algoritma pembelajaran mesin yang digunakan untuk memprediksi nilai variabel target berkelanjutan dengan membagi ruang fitur menjadi segmen-segmen yang lebih kecil dan lebih sederhana. Dalam model ini, pohon keputusan dibangun dengan membagi data berdasarkan aturan keputusan yang berurutan hingga mencapai titik di mana segmentasi tidak dapat ditingkatkan lebih lanjut. Setiap node dalam pohon mewakili aturan keputusan untuk membagi data, dan setiap daun mewakili nilai prediksi untuk segmen data yang sesuai. Model ini cocok untuk dataset dengan hubungan non-linear dan interaksi antara fitur, memungkinkan untuk memahami struktur data yang rumit dan menghasilkan prediksi yang akurat.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

df_X = df.drop(['id','date','price'], axis=1)
df_y = df['price']

X = df_X.astype(float).values
y = df_y.astype(float).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt = DecisionTreeRegressor(max_depth=10)
dt.fit(X_train, y_train)

print('coef of determination training', dt.score(X_train, y_train))
print('coef of determination testing', dt.score(X_test, y_test))

print('prediction')
y_pred = dt.predict(X_test)
print(y_pred[:10])
print('real value')
print(y_test[0:10])
```

> **[Deskripsi Gambar 15: Kode untuk Membuat Model Decision Tree Regression]**
> *Gambar ini menampilkan blok kode untuk pemodelan Decision Tree. Kode mencakup pemisahan fitur dan target, pembagian data, inisialisasi model `DecisionTreeRegressor(max_depth=10)`, proses `.fit()`, perhitungan R², serta prediksi.*

> **[Deskripsi Gambar 16: Output Pelatihan dan Prediksi Model Decision Tree]**
> *Gambar ini menampilkan output teks:*
> * *coef of determination training: 0.9185274272472612*
> * *coef of determination testing: 0.7635675475462345*
> * *prediction: Array berisi 10 nilai prediksi harga rumah.*
> * *real value: Array berisi 10 nilai harga rumah sebenarnya.*

```python
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('rmse:', rmse)
print('r2:', r2)
```

> **[Deskripsi Gambar 17: Kode Evaluasi Model Decision Tree]**
> *Gambar ini menampilkan blok kode Python yang identik dengan evaluasi sebelumnya untuk menghitung RMSE dan R² model Decision Tree.*

> **[Deskripsi Gambar 18: Hasil Evaluasi Model Decision Tree]**
> *Gambar ini menampilkan output teks:*
> * *rmse: 184751.00302163974*
> * *r2: 0.7635675475462345*

```python
import seaborn as sns
import matplotlib.pyplot as plt

data1 = pd.Series(y_test[:50].ravel())
data2 = pd.Series(y_pred[:50].ravel())

df_new = pd.concat([data1, data2], keys=['real values', 'predicted values'], axis=1)
df_new.plot(kind='bar', figsize=(15,3))

plt.title("DT regression")
plt.xlabel('Sample i')
plt.ylabel('House Price')
plt.show()
```

> **[Deskripsi Gambar 19: Kode Visualisasi Decision Tree]**
> *Gambar ini menampilkan blok kode Python untuk memplot 50 sampel pertama perbandingan nilai aktual dan prediksi Decision Tree.*

> **[Deskripsi Gambar 20: Grafik Batang Perbandingan (Decision Tree)]**
> *Gambar ini menampilkan grafik batang perbandingan 50 nilai `real values` dan `predicted values` untuk model Decision Tree. Terlihat bahwa pola prediksi lebih rapat mengikuti nilai sebenarnya dibandingkan model Linear Regression, menandakan error yang lebih kecil.*

---

## Random Forest Regression

Model random forest regression adalah algoritma *ensemble learning* yang menggunakan sejumlah besar pohon keputusan yang dibangun secara acak untuk memprediksi nilai variabel target berkelanjutan. Setiap pohon dalam hutan tersebut dibangun secara independen dengan sampel acak dari data pelatihan dan fitur, dan kemudian nilai prediksi dari setiap pohon diambil rata-rata atau digabungkan untuk menghasilkan prediksi akhir. Dengan memanfaatkan keragaman antara pohon-pohon tersebut, model ini mampu mengatasi masalah *overfitting* dan meningkatkan kinerja prediksi, serta mampu menangani hubungan non-linear dan interaksi antara fitur secara efektif dalam dataset yang kompleks. Model random forest regression juga memiliki kemampuan untuk mengevaluasi pentingnya fitur (*feature importance*) dalam membuat prediksi, membuatnya sangat berguna untuk pemodelan dan analisis data yang luas.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df_X = df.drop(['id','date','price'], axis=1)
df_y = df['price']

X = df_X.astype(float).values
y = df_y.astype(float).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

print('coef of determination training', rf.score(X_train, y_train))
print('coef of determination testing', rf.score(X_test, y_test))

print('prediction')
y_pred = rf.predict(X_test)
print(y_pred[:10])
print('real value')
print(y_test[0:10])
```

> **[Deskripsi Gambar 21: Kode untuk Membuat Model Random Forest Regression]**
> *Gambar ini menampilkan blok kode untuk pemodelan Random Forest. Kode mencakup inisialisasi model `RandomForestRegressor()` dengan parameter default, proses `.fit()`, perhitungan R², serta prediksi.*

> **[Deskripsi Gambar 22: Output Pelatihan dan Prediksi Model Random Forest]**
> *Gambar ini menampilkan output teks:*
> * *coef of determination training: 0.9820816679041049*
> * *coef of determination testing: 0.8572069468830394*
> * *prediction: Array berisi 10 nilai prediksi harga rumah.*
> * *real value: Array berisi 10 nilai harga rumah sebenarnya.*

```python
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('rmse:', rmse)
print('r2:', r2)
```

> **[Deskripsi Gambar 23: Kode Evaluasi Model Random Forest]**
> *Gambar ini menampilkan blok kode Python untuk menghitung metrik evaluasi RMSE dan R² khusus untuk model Random Forest Regression.*

> **[Deskripsi Gambar 24: Hasil Evaluasi Model Random Forest]**
> *Gambar ini menampilkan output teks:*
> * *rmse: 143577.6368885783*
> * *r2: 0.8572069468830394*

```python
import seaborn as sns
import matplotlib.pyplot as plt

data1 = pd.Series(y_test[:50].ravel())
data2 = pd.Series(y_pred[:50].ravel())

df_new = pd.concat([data1, data2], keys=['real values', 'predicted values'], axis=1)
df_new.plot(kind='bar', figsize=(15,3))

plt.title("RF regression")
plt.xlabel('Sample i')
plt.ylabel('House Price')
plt.show()
```

> **[Deskripsi Gambar 25: Kode Visualisasi Random Forest]**
> *Gambar ini menampilkan blok kode Python yang mengambil 50 sampel awal dari data testing dan prediksi Random Forest, lalu memplotnya dalam bentuk grafik batang.*

> **[Deskripsi Gambar 26: Grafik Batang Perbandingan (Random Forest)]**
> *Gambar ini menampilkan grafik batang perbandingan 50 nilai `real values` dan `predicted values` untuk model Random Forest. Terlihat bahwa batang `predicted values` hampir berimpit atau sangat mendekati batang `real values`, menunjukkan bahwa model Random Forest memiliki tingkat error yang paling rendah dan akurasi prediksi yang paling tinggi dibandingkan model lainnya.*

---

## Tugas 2

**Dataset:** https://www.kaggle.com/datasets/hellbuoy/car-price-prediction/data

1. Dengan menggunakan dataset diatas, simpan ke dalam google drive atau repository lain, kemudian tampilkan dalam bentuk dataframe!
2. Jelaskan apa tujuan penggunaan dataset ini?
3. Definisikan atribut yang menjadi input dan output nya?
4. Sebutkan 3 variable yang paling mempengaruhi output / mempunyai nilai korelasi tinggi!
5. Silahkan membuat model regresi dengan Lin regression, DT reg, RF reg dari dataset diatas!
6. Buatlah tabel yang menjelaskan performa dari model machine learning untuk kasus dataset diatas! Kolom pertama "model", kolom selanjutnya "RMSE, R2, R".
7. Jelaskan dari hasil eksperimen diatas, model mana yang paling baik? Jelaskan alasan anda!

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