# Submission 1: French Twitter Sentiment Analysis
Nama: Niko Febrianur

Username dicoding: nikofebrianur

![twiter sentiment](https://github.com/nikofebrianur/French-Twitter-Sentiment-Analysis/assets/42314371/13d5370b-7a19-4473-97c8-aab7710560d0)

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [French Twitter Sentiment Analysis](https://www.kaggle.com/datasets/hbaflast/french-twitter-sentiment-analysis) |
| Masalah | Salah satu tantangan utama dalam bisnis modern adalah memahami dan merespons perasaan dan pendapat pelanggan secara real-time. Dalam era media sosial, pelanggan secara aktif berbagi pengalaman mereka tentang produk atau layanan di platform seperti Twitter. Masalah yang dihadapi perusahaan adalah bagaimana mengukur dan menganalisis sentimen pelanggan terhadap produk atau layanan mereka dalam skala besar. Sentimen dapat berkisar dari positif (puas) hingga negatif (tidak puas), dan ini dapat memengaruhi citra merek, retensi pelanggan, dan keputusan bisnis. <br><br> Oleh karena itu, masalah yang ingin diangkat adalah bagaimana memanfaatkan data dari media sosial, khususnya tweet-tweet dalam bahasa Prancis, untuk mengidentifikasi sentimen pelanggan terhadap produk atau layanan. Dalam konteks ini, kita ingin mengembangkan model atau algoritma yang dapat secara otomatis menganalisis tweet-tweet ini dan mengkategorikannya sebagai sentimen positif atau negatif. Hal ini akan memungkinkan perusahaan untuk secara proaktif merespons umpan balik pelanggan, mengukur keberhasilan kampanye pemasaran, dan meningkatkan kualitas produk atau layanan mereka berdasarkan data pelanggan. |
| Solusi machine learning | Untuk mengatasi masalah analisis sentimen terhadap produk atau layanan di media sosial, kita dapat mengembangkan model machine learning klasifikasi. Solusi ini akan memungkinkan kita untuk secara otomatis mengkategorikan tweet-tweet dalam bahasa Prancis sebagai sentimen positif atau negatif terhadap produk atau layanan yang disebutkan. Prosesnya dapat dibagi menjadi beberapa tahap: <br><br> 1. **Pra-pemrosesan Teks**: Tahap awal melibatkan pra-pemrosesan teks, di mana kita membersihkan, tokenisasi, dan mengubah teks tweet ke dalam representasi numerik yang dapat dipahami oleh model. Ini juga melibatkan penghapusan kata-kata berhenti (stop words) dan mungkin penerapan teknik seperti stemming atau lemmatization. <br> 2. **Ekstraksi Fitur**: Kami dapat menggunakan metode seperti TF-IDF (Term Frequency-Inverse Document Frequency) atau Word Embeddings (misalnya, Word2Vec) untuk mengubah teks menjadi vektor numerik yang merepresentasikan setiap tweet. <br> 3. **Pemilihan Model**: Pilih model machine learning yang sesuai untuk tugas klasifikasi ini. Beberapa opsi termasuk Naive Bayes, Support Vector Machine (SVM), atau model Deep Learning seperti LSTM. <br> 4. **Pelatihan Model**: Gunakan dataset yang telah dipisahkan menjadi set pelatihan dan set pengujian untuk melatih model. Ini melibatkan penyesuaian parameter dan pemantauan kinerja model. <br> 5. **Evaluasi Model**: Model harus dievaluasi dengan menggunakan metrik seperti akurasi, presisi, recall, F1-score, dan mungkin matriks kebingungan (confusion matrix) untuk memahami kinerjanya. <br> 6. **Penyimpanan Model**: Simpan model terlatih untuk digunakan pada tweet-tweet baru di masa depan. Model ini akan dapat secara otomatis menganalisis sentimen terhadap produk atau layanan berdasarkan tweet-tweet yang masuk. <br><br> Dengan solusi ini, perusahaan atau organisasi dapat mengambil keputusan berdasarkan data, merespons umpan balik pelanggan dengan cepat, dan memahami secara lebih baik bagaimana produk |
| Metode pengolahan | Dalam pengolahan data untuk analisis sentimen dengan dataset tersebut, kita perlu melakukan beberapa tahap pra-pemrosesan data dan pembagian data. Berikut adalah langkah-langkah yang dapat diambil: <br><br> 1. **Pra-pemrosesan Teks**: <br> - Mengubah teks dalam kolom "text" menjadi huruf kecil (lowercase) agar tidak ada perbedaan antara huruf besar dan kecil dalam analisis. <br> - Melakukan tokenisasi untuk memecah teks menjadi kata-kata atau token individual. <br><br> 2. **Pengubahan Label**: <br> - Mengubah label dalam kolom "label" menjadi format integer. Label 0 dapat dianggap sebagai sentimen negatif, sedangkan label 1 dapat dianggap sebagai sentimen positif. <br><br> 3. **Pembagian Data**: <br> - Memisahkan data menjadi dua bagian: data pelatihan (training data) dan data evaluasi (evaluation data) dalam rasio 80:20 atau sesuai dengan kebutuhan. Data pelatihan digunakan untuk melatih model machine learning, sementara data evaluasi digunakan untuk menguji kinerja model. Ini dapat dilakukan dengan menggunakan teknik pembagian data acak. <br><br> Dengan melakukan pra-pemrosesan ini dan pembagian data, kita mempersiapkan dataset agar siap digunakan dalam pelatihan dan pengujian model machine learning untuk analisis sentimen. Data telah dikonversi ke dalam format yang sesuai dan siap untuk dieksploitasi oleh algoritma pembelajaran mesin. |
| Arsitektur model | Dalam proyek ini, kita menggunakan arsitektur model yang sesuai untuk analisis sentimen berdasarkan teks. Berikut adalah komponen-komponen utama dari arsitektur model ini: <br><br> 1. **Layer Vectorize**: <br> - Layer pertama adalah `vectorize_layer`, yang bertanggung jawab untuk mengubah teks menjadi representasi numerik. <br><br> 2. **Layer Embedding**: <br> - Dilanjutkan dengan layer `Embedding` dengan dimensi embedding sebesar 16. Layer ini bertugas untuk memetakan kata-kata dalam teks ke dalam ruang vektor berdimensi 16. <br><br> 3. **Layer AveragePooling1D**: <br> - Layer `AveragePooling1D` digunakan karena data berbentuk teks. Ini membantu dalam mengurangi dimensi dan mengekstraksi fitur penting dari teks. <br><br> 4. **Layer Dense**: <br> - Terdapat dua layer `Dense` berturut-turut dengan unit 64 dan 32, yang menggunakan aktivasi ReLU (Rectified Linear Unit) untuk memproses informasi. <br><br> 5. **Layer Output**: <br> - Layer output terakhir menggunakan aktivasi sigmoid karena kita melakukan klasifikasi biner antara dua label sentimen (0 dan 1). <br><br> - Loss function yang digunakan adalah `binary_crossentropy` karena kita melakukan klasifikasi biner. <br> - Optimizer yang digunakan adalah `Adam`, yang merupakan optimizer yang umum digunakan dalam pelatihan model. <br> - Metrik yang digunakan adalah `BinaryAccuracy` untuk mengukur akurasi prediksi model dalam konteks klasifikasi biner. <br><br> Dengan arsitektur ini, kita dapat mengembangkan model yang dapat memahami dan mengklasifikasikan sentimen dari teks tweet dengan baik. |
| Metrik evaluasi | Dalam mengevaluasi kinerja model analisis sentimen, kita menggunakan beberapa metrik evaluasi berikut: <br><br> 1. **ExampleCount**: Metrik ini memberikan jumlah total contoh yang digunakan dalam evaluasi, yang mencakup data yang diklasifikasikan dengan benar dan yang salah. <br><br> 2. **AUC (Area Under the Curve)**: AUC adalah metrik yang digunakan untuk mengukur area di bawah kurva Receiver Operating Characteristic (ROC). Ini memberikan gambaran tentang sejauh mana model dapat membedakan antara kelas positif dan negatif. <br><br> 3. **FalsePositives**: Ini adalah jumlah kasus di mana model secara salah mengklasifikasikan contoh yang sebenarnya negatif sebagai positif. <br><br> 4. **TruePositives**: Ini adalah jumlah kasus di mana model dengan benar mengklasifikasikan contoh yang sebenarnya positif sebagai positif. <br><br> 5. **FalseNegatives**: Ini adalah jumlah kasus di mana model secara salah mengklasifikasikan contoh yang sebenarnya positif sebagai negatif. <br><br> 6. **TrueNegatives**: Ini adalah jumlah kasus di mana model dengan benar mengklasifikasikan contoh yang sebenarnya negatif sebagai negatif. <br><br> 7. **BinaryAccuracy**: Metrik ini mengukur akurasi prediksi model dalam konteks klasifikasi biner, yaitu sejauh mana model benar-benar memprediksi kelas dengan benar. <br><br> Metrik-metrik ini digunakan untuk memberikan pemahaman yang komprehensif tentang seberapa baik model dapat melakukan analisis sentimen terhadap teks tweet dalam dataset. Dengan melihat metrik-metrik ini, kita dapat mengevaluasi dan mengoptimalkan kinerja model untuk tujuan analisis sentimen. |
| Performa model | (sample) Evaluasi model diperoleh yaitu AUC sebesar 82%, kemudian example_count 575, dengan BinaryAccuracy 75%, dan loss sebesar 1.364. Untuk False Negatives 68, False Positive 75, True Negative 201 dan True Positive 231. Model yang telah dibuat dapat dilakukan peningkatan performa, karena model belum cukup baik karena BinaryAccuracy masih dibawah 80% |

## Untuk menjalankan proyek ML pipeline, lakukan perintah berikut ini:
-  Membuat environment dan aktifkan
``sh
conda create --name example-env python=3.9.15
conda activate example-env
``

- Install library dan menjalankan Jupyter notebook di localhost
``sh
pip install jupyter scikit-learn tensorflow tfx==1.11.0 flask joblib
jupyter-notebook notebook.ipynb
``
