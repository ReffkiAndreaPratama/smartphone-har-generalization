# Analisis Generalisasi Model Pengenalan Aktivitas Manusia Berbasis Sensor Smartphone pada Perbedaan Pengguna dan Perangkat

Repository riset yang dapat direproduksi untuk skripsi **"Analisis Generalisasi Model
Pengenalan Aktivitas Manusia Berbasis Sensor Smartphone pada Perbedaan Pengguna dan
Perangkat"**.

| | |
| --- | --- |
| **Dataset** | UCI HAR (data nyata, 10.299 window, 561 fitur, 30 subjek, 6 aktivitas) |
| **Model** | Random Forest, SVM (RBF), 1D-CNN |
| **Eksperimen** | A (baseline) · B (lintas pengguna) · C (lintas perangkat) · D (strategi peningkatan) |
| **Lingkungan** | Python 3.11 · CPU-only · scikit-learn + PyTorch |
| **Pengujian** | 77 tes otomatis (`pytest -q`) |
| **Lisensi data** | UCI HAR — AS-IS, penggunaan komersial dilarang, sitasi wajib (lihat [Lisensi & Etika](#lisensi--etika)) |

> **Tidak ada angka yang dikarang di repository ini.** Semua metrik pada README,
> dashboard, dan laporan berasal dari berkas JSON/CSV yang benar-benar dihasilkan
> oleh `scripts/run_experiments.py`. Mode demo sintetis selalu diberi label
> **DATA DEMO SINTETIS** dan tidak boleh dicampur dengan hasil penelitian.

---

## Daftar Isi

1. [Ringkasan Eksekutif](#ringkasan-eksekutif)
2. [Tujuan Penelitian dan Ruang Lingkup](#tujuan-penelitian-dan-ruang-lingkup)
3. [Dataset UCI HAR](#dataset-uci-har)
4. [Struktur Repositori](#struktur-repositori)
5. [Instalasi](#instalasi)
6. [Menjalankan Pipeline](#menjalankan-pipeline)
7. [Metodologi](#metodologi)
8. [Hasil Eksperimen](#hasil-eksperimen)
9. [Dashboard Streamlit](#dashboard-streamlit)
10. [Artefak Keluaran](#artefak-keluaran)
11. [Determinisme dan Reproduksi](#determinisme-dan-reproduksi)
12. [Pengujian](#pengujian)
13. [Batasan Penelitian](#batasan-penelitian)
14. [Lisensi & Etika](#lisensi--etika)
15. [Menyimpan ke GitHub](#menyimpan-ke-github)
16. [Referensi](#referensi)
17. [Daftar File Penting](#daftar-file-penting)

---

## Ringkasan Eksekutif

Repository ini menjawab satu pertanyaan riset secara terukur: **seberapa besar
performansi model pengenalan aktivitas menurun ketika model digunakan pada pengguna
(atau perangkat) yang berbeda dari data latihnya?**

Ringkasan alur kerja:

1. Dataset UCI HAR diunduh, diverifikasi terhadap struktur yang diharapkan
   (561 fitur, 30 subjek, 7.352/2.947 sampel, tanpa tumpang tindih subjek), lalu
   dicache menjadi array NumPy.
2. **Eksperimen A — Baseline.** Tiga model dilatih pada split resmi UCI
   (21 subjek latih / 9 subjek uji) dengan penjaga anti-leakage.
3. **Eksperimen B — Generalisasi lintas pengguna.** Random Forest dan SVM diuji
   dengan *Leave-One-Subject-Out* (30 fold, setiap subjek diuji tepat sekali).
   1D-CNN diuji dengan *GroupKFold(6)* karena biaya pelatihan CPU, dengan setiap
   subjek tetap diuji tepat sekali.
4. **Eksperimen C — Generalisasi lintas perangkat.** **Tidak dijalankan**, karena
   UCI HAR hanya dikumpulkan dengan satu perangkat (Samsung Galaxy S II). Status
   `not_performed` beserta alasannya ditulis eksplisit di
   `reports/results/cross_device.json`; protyek lengkap ada di
   [`docs/cross_device_protocol.md`](docs/cross_device_protocol.md).
5. **Eksperimen D — Strategi peningkatan.** Normalisasi per-subjek, kalibrasi
   label target, dan augmentasi dibandingkan pada subset evaluasi yang identik.

Temuan utama, lengkap dengan angkanya, ada di bagian
[Hasil Eksperimen](#hasil-eksperimen).

---

## Tujuan Penelitian dan Ruang Lingkup

**Tujuan.** Mengkuantifikasi generalisasi model HAR pada perbedaan pengguna dan
menganji beberapa strategi mitigasi (normalisasi, kalibrasi, augmentasi).

**Yang dikerjakan.**

* Empat eksperimen (A, B, C, D) dengan protokol dan guardrail yang eksplisit.
* Tiga arsitektur model yang berbeda (ensemble pohon, kernel, convolutional).
* Pelaporan metrik lengkap: accuracy, balanced accuracy, macro-F1, weighted-F1,
  Cohen's kappa, precision/recall/F1 per kelas, confusion matrix, variasi per subjek,
  dan waktu komputasi.
* Dashboard web yang membaca artefak hasil (tidak menghitung ulang apa pun).

**Yang tidak dikerjakan, dan alasannya.**

* **Eksperimen lintas perangkat tidak dapat dihitung** dari UCI HAR karena hanya ada
  satu perangkat. repository ini menyediakan modul eksekusi penuh
  (*leave-one-device-out*) plus prosedur pengumpulan data mandiri, tetapi **tidak
  memalsukan angka lintas perangkat**.
* Tidak ada klaim deployment di luar domain (mis. perangkat berbeda, orientasi
  domisili, population berbeda) karena dataset tidak mendukungnya.

---

## Dataset UCI HAR

[Human Activity Recognition Using Smartphones](https://doi.org/10.24432/C54S4K)
oleh Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, dan Jorge L. Reyes-Ortiz
(ESANN 2013).

| Atribut | Nilai |
| --- | --- |
| Sampel | 10.299 window (7.352 latih + 2.947 uji) |
| Fitur | 561 fitur (prakomputasi oleh pembuat dataset) |
| Subjek | 30 orang (21 latih, 9 uji pada split resmi) |
| Aktivitas | WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING |
| Perangkat | Samsung Galaxy S II (satu perangkat, dipasang di pinggang) |
| Sinyal | 50 Hz, window 128 sampel (2,56 s), overlap 50%, 9 kanal inersial |
| Kanal | `body_acc_{x,y,z}`, `body_gyro_{x,y,z}`, `total_acc_{x,y,z}` |
| Lisensi | AS-IS; sitasi wajib; penggunaan komersial dilarang |

Ringkasan kelas untuk seluruh 10.299 window: WALKING 1.722 (16,72 %),
WALKING_UPSTAIRS 1.544 (14,99 %), WALKING_DOWNSTAIRS 1.406 (13,65 %), SITTING 1.777
(17,25 %), STANDING 1.906 (18,51 %), LAYING 1.944 (18,88 %). Rincian per partisi
termuat di `reports/tables/dataset_class_distribution.csv`.

### Unduhan otomatis

```bash
python scripts/prepare_data.py --download
```

Perintah tersebut mengunduh arsip, memverifikasi isi ZIP, mengekstrak folder
`UCI HAR Dataset`, menjalankan verifikasi struktur, membuat cache NumPy, dan menulis
`reports/tables/dataset_summary.json` + `reports/tables/dataset_verification.json`.

Mirror yang dipakai saat pengembangan (SHA-256):

```text
50dabbc800629611831a85b8b71c87040525ca4af6a152c6ba9360ecee6b92dc
```

Hash ini bersifat informatif untuk mirror CloudFront; berkas ZIP dari URL utama UCI
dapat berbeda byte-wise meski isinya identik. Verifikasi yang dipakai pipeline adalah
`zipfile.testzip()` (CRC per entri) plus pemeriksaan jumlah fitur/subjek/sampel dan
ketiadaan tumpang tindih subjek.

### Catatan penting tentang nama fitur

`features.txt` berisi **561 baris nama fitur dengan hanya 477 nama unik** — 84 baris
menduplikasi nama fitur `bandsEnergy`. Repository ini menamai ulang duplikat dengan
sufiks `__dupN` (mis. `fBodyAcc-bandsEnergy()-mean()__dup1`) **tanpa mengubah urutan
kolom**, sehingga aman dipakai pada DataFrame/CSV. Jumlah duplikat tercatat pada
`reports/tables/dataset_verification.json`. Nilai fitur tidak pernah diubah.

### Sinyal mentah

Repository menyimpan dan memuat 9 kanal mentah (128 × 9 per window) dari folder
`Inertial Signals`. Sinyal ini dipakai oleh model 1D-CNN dan oleh modul
eksperimen lintas perangkat. Fitur 561 kolom yang dipakai RF/SVM adalah fitur
prakomputasi UCI, **bukan** hasil ekstraksi ulang oleh pipeline ini.

---

## Struktur Repositori

```text
smartphone-har-generalization/
├── app.py                     # Dashboard Streamlit (5 halaman)
├── config/
│   └── config.yaml            # Seluruh pengaturan: seed, model, protokol, path
├── src/
│   ├── utils.py               # Path, seed, logging, JSON/IO, error domain
│   ├── data_loader.py         # Unduh, verifikasi, muat, cache UCI HAR
│   ├── demo_data.py           # Generator data demo sintetis deterministik
│   ├── preprocessing.py       # Standardisasi (hanya data latih), statistik subjek
│   ├── split_strategy.py      # Split resmi, LOSO, GroupKFold, split kalibrasi
│   ├── feature_extraction.py  # Ekstraksi fitur window + parsing grup fitur UCI
│   ├── augmentation.py        # Jitter/scaling/time-shift untuk sinyal CNN
│   ├── models.py              # Pabrik model sklearn + arsitektur 1D-CNN
│   ├── train.py               # Loop pelatihan, seleksi hiperparameter, save/load
│   ├── evaluate.py            # Metrik, per-kelas, per-subjek, agregasi
│   ├── generalization.py      # Orkestrasi Eksperimen A, B, D
│   ├── cross_device.py        # Eksperimen C (leave-one-device-out) + status eksplisit
│   ├── visualization.py       # Seluruh figur laporan
│   └── dashboard_data.py      # Pembaca artefak untuk dashboard (read-only)
├── scripts/
│   ├── prepare_data.py        # Unduh/verifikasi/cache + figur eksplorasi
│   ├── run_experiments.py     # CLI pipeline seluruh tahap
│   ├── generate_report.py     # Tabel, figur, experiment_summary.md
│   └── update_readme_results.py # Isi blok hasil README dari artefak JSON
├── tests/                     # 77 tes otomatis (pytest)
├── notebooks/                 # 2 notebook analisis (hanya membaca artefak)
├── docs/
│   └── cross_device_protocol.md   # Protokol eksperimen C + format manifest
├── data/                      # Dataset & cache (tidak di-commit; lihat data/README.md)
├── models/                    # Model terlatih (tidak di-commit; dapat direproduksi)
└── reports/                   # Hasil eksperimen, tabel, figur, laporan
```

Prinsip struktur: **`src/` berisi logika, `scripts/` berisi CLI, `reports/` berisi
keluaran, `tests/` memverifikasi semuanya.** Tidak ada angka yang ditulis manual di
kode.

---

## Instalasi

**Kebutuhan sistem**

* Python 3.11 (diuji pada 3.11)
* ±2 GB RAM bebas — arsip dataset 60 MB (ekstraksi 270 MB), cache NumPy 41 MB,
  model RF 14 MB, SVM 9 MB, CNN 0,2 MB
* Koneksi internet hanya untuk tahap unduh dataset
* GPU **tidak** diperlukan; seluruh eksperimen berjalan di CPU

**Langkah**

```bash
git clone <url-repository> smartphone-har-generalization
cd smartphone-har-generalization

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Untuk PyTorch khusus CPU (disarankan agar unduhan lebih kecil):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Verifikasi instalasi:

```bash
python -c "import sklearn, torch, streamlit, yaml; print(sklearn.__version__, torch.__version__)"
pytest -q
```

---

## Menjalankan Pipeline

### 1. Siapkan dan verifikasi dataset

```bash
python scripts/prepare_data.py --download
```

### 2. Jalankan seluruh eksperimen

```bash
python scripts/run_experiments.py
```

Tahap yang dijalankan: `baseline` → `loso` → `group-cv` → `improvement` → `cross-device`
→ pembuatan laporan dan figur. Log lengkap tersimpan di konsol; manifest run ditulis
ke `reports/results/run_manifest.json`.

Setelah run selesai, perbarui otomatis bagian **Hasil Eksperimen** pada README:

```bash
python scripts/update_readme_results.py
```

Skrip ini membaca `reports/results/*.json` lalu menulis ulang blok
"Hasil Eksperimen" di `README.md`. Karena itu **angka pada README selalu
identik dengan artefak hasil** — tidak ada angka yang diketik manual.

Perintah yang sering dipakai:

```bash
# hanya satu tahap
python scripts/run_experiments.py --stages baseline
python scripts/run_experiments.py --stages loso,group-cv
python scripts/run_experiments.py --stages improvement
python scripts/run_experiments.py --stages cross-device

# uji cepat (data demo, model kecil) - BUKAN hasil penelitian
python scripts/run_experiments.py --demo --quick --fresh

# ulangi hanya tahap yang gagal / belum selesai (resume per fold)
python scripts/run_experiments.py

# paksa ulang semuanya
python scripts/run_experiments.py --fresh

# ganti ukuran protokol
python scripts/run_experiments.py --max-loso-folds 10 --n-splits 3 --cnn-epochs 20
```

| Flag | Fungsi |
| --- | --- |
| `--stages` | Pilih tahap: `baseline`, `loso`, `group-cv`, `improvement`, `cross-device`, `all` |
| `--models` | Batasi model, mis. `--models random_forest,svm` |
| `--demo` | Pakai data demo sintetis (wajib diberi label, bukan hasil penelitian) |
| `--quick` | Konfigurasi cepat untuk smoke test |
| `--download` | Unduh dataset bila belum ada |
| `--fresh` | Hapus artefak hasil mode ini sebelum menjalankan |
| `--no-resume` | Latih ulang semua fold, abaikan hasil parsial |
| `--no-report` | Lewati pembuatan laporan/figur |
| `--seed` | Ganti seed acak (default 42 dari `config/config.yaml`) |
| `--max-loso-folds`, `--n-splits`, `--cnn-epochs`, `--calibration-fraction` | Ubah ukuran protokol |

### 3. Buka dashboard

```bash
streamlit run app.py
```

Buka `http://localhost:8501` (sesuaikan dengan port yang ditampilkan).

### 4. Jalankan notebook (opsional)

```bash
cd notebooks
jupyter notebook
```

---

## Metodologi

### Pipeline umum

```text
unduh → verifikasi → cache → split (anti-leakage) → fit preprocessing pada data latih
      → latih model → evaluasi (per kelas, per subjek, confusion matrix, timing)
      → simpan JSON/CSV/NPZ → figur + tabel + laporan → dashboard
```

**Penjaga anti-leakage** (`src/split_strategy.py`):

* `assert_no_subject_overlap` — nama subjek pada partisi latih dan uji harus benar-benar
  berbeda; pelanggaran memicu `SubjectLeakageError`.
* `assert_indices_disjoint` — indeks latih dan uji tidak boleh tumpang tindih.
* Pemeriksaan ini dipanggil pada **setiap fold** di LOSO, GroupKFold, dan Eksperimen D,
  bukan hanya sekali di awal.
* Standardisasi fitur di dalam pipeline scikit-learn (`StandardScaler`) di-**fit hanya
  pada data latih** fold tersebut. Random Forest dilatih pada fitur asli tanpa scaling.
* Normalisasi per-subjek pada Eksperimen D memakai statistik target **tanpa label**
  (transductive) dan ditandai eksplisit di laporan.

### Model

| Model | Representasi input | Catatan |
| --- | --- | --- |
| Random Forest | 561 fitur UCI | 300 pohon, `max_features=sqrt`; hiperparameter dipilih sekali via GroupKFold pada subjek latih |
| SVM (RBF) | 561 fitur UCI | `StandardScaler` di dalam pipeline, `probability=False` |
| 1D-CNN | Sinyal mentah 128 × 9 | 3 blok Conv1D (32/64/128) + pooling + dropout 0,3; early stopping pada subjek validasi |

Seleksi hiperparameter RF/SVM memakai `GroupKFold(3)` **hanya pada subjek data latih
split resmi**, lalu konfigurasi hasil seleksi **dibekukan** untuk seluruh fold LOSO
dan Eksperimen D. Ini adalah trade-off biaya komputasi yang dinyatakan terbuka —
pemilihan ulang hiperparameter per fold tidak dilakukan.

### Eksperimen A — Baseline (split resmi)

* Protokol: `official_subject_disjoint`.
* Latih: 21 subjek / 7.352 sampel. Uji: 9 subjek / 2.947 sampel.
* Guardrail: `assert_no_subject_overlap` dijalankan sebelum pelatihan.

### Eksperimen B — Generalisasi lintas pengguna

| Model | Protokol | Jumlah fold | Sifat |
| --- | --- | --- | --- |
| Random Forest | Leave-One-Subject-Out | 30 | Setiap subjek diuji tepat sekali |
| SVM (RBF) | Leave-One-Subject-Out | 30 | Setiap subjek diuji tepat sekali |
| 1D-CNN | GroupKFold(6) | 6 | Setiap subjek diuji tepat sekali |

Perbedaan jumlah fold antar model **disengaja** (biaya pelatihan CPU untuk CNN) dan
dinyatakan terbuka. Yang dijaga adalah invarian yang penting: setiap subjek selalu
berada di sisi uji tepat satu kali, dan tidak pernah di sisi latih fold-nya.

### Eksperimen C — Generalisasi lintas perangkat

Status: **tidak dijalankan** (`not_performed`).

Alasannya faktual, bukan pilihan metodologis: UCI HAR dikumpulkan dengan satu perangkat
(Samsung Galaxy S II). Menjalankan "leave-one-device-out" pada satu perangkat akan
menjadi Training-On-Test, sehingga tidak bermakna. `reports/results/cross_device.json`
menuliskan status, alasan, daftar kebutuhan data, dan tautan prosedur. Modul
`src/cross_device.py` sudah menyediakan jalur eksekusi penuh untuk data mandiri
multi-perangkat (lihat [`docs/cross_device_protocol.md`](docs/cross_device_protocol.md)).

### Eksperimen D — Strategi peningkatan generalisasi

Protokol: `GroupKFold(6)` — **fold yang sama** dipakai oleh seluruh model dan seluruh
strategi, sehingga perbandingan unfair tidak mungkin terjadi.

| Strategi | Cara kerja | Sifat |
| --- | --- | --- |
| `base` | Model dilatih pada subjek sumber; evaluasi pada subset target | Pembanding |
| `subject_norm` | Standardisasi per-subjek memakai statistik subjek target **tanpa label** | Transductive |
| `calibration` | 10 % baris target **berlabel** ditambahkan ke data latih; evaluasi pada sisa target | Butuh label target |
| `augmentation` | Jitter/scaling/time-shift pada data latih CNN saja | Hanya CNN |

Pada strategi `calibration`, baris kalibrasi dikeluarkan dari subset evaluasi
(diperiksa dengan `assert_indices_disjoint`), sehingga tidak ada baris yang sekaligus
melatih dan dinilai.

### Metrik

* **Accuracy** — proporsi prediksi benar.
* **Balanced accuracy** — rata-rata recall per kelas; lebih informatif untuk data
  yang kelasnya hampir seimbang namun tetap dihitung sebagai ukuran tambahan.
* **Macro-F1** — rata-rata F1 semua kelas; metrik utama dalam laporan ini.
* **Weighted-F1** — F1 berbobot support.
* **Cohen's kappa** — koreksi terhadap tebakan acak.
* **Precision/recall/F1 per kelas** dengan support.
* **Confusion matrix** 6 × 6 plus daftar pasangan confusion teratas.
* **Variasi per subjek** — metrik dihitung ulang per subjek uji pada LOSO, lalu
  diringkas (min/maks/mean/std).
* **Waktu komputasi** — `fit_time_s`, `predict_time_s`, baik per fold maupun total.

> **Catatan jujur tentang skor SVM.** SVM dilatih dengan `probability=False`
> (karena biaya komputasi dan kalibrasi Platt yang mahal). Skor yang ditampilkan adalah
> *decision function* yang dinormalisasi menjadi jumlah 1, **bukan probabilitas
> terkalibrasi**. Dashboard dan laporan menyatakan hal ini eksplisit pada setiap
> penyajian skor SVM.

---

## Hasil Eksperimen

<!-- BEGIN-HASIL-EKSPERIMEN -->

Angka di bawah ini dibaca langsung dari berkas JSON pada `reports/results/` yang dihasilkan `python scripts/run_experiments.py`. Laporan lengkap dengan tautan figur tersedia di [`reports/experiment_summary.md`](reports/experiment_summary.md).

### Provenance eksekusi

| Properti | Nilai |
| --- | --- |
| Run id | 2026-10-22T13:19:37+07:00 |
| Status run | complete |
| Mode | research |
| Seed | 42 |
| Mulai | 2026-10-22T13:19:37+07:00 |
| Selesai | 2026-10-22T13:46:37+07:00 |
| Durasi total | 27.0 menit |
| Git commit | null (run dieksekusi sebelum repository diinisialisasi) |
| Python | 3.11.9 |
| Platform | Windows-10-10.0.19045-SP0 / Intel64 Family 6 Model 79 Stepping 1, GenuineIntel |
| Versi pustaka | numpy 2.4.6, scikit-learn 1.5.2, torch 2.5.1+cpu, streamlit 1.39.0 |

| Tahap | Status | Durasi |
| --- | --- | --- |
| baseline | complete | 1.5 menit |
| loso | complete | 7.5 menit |
| group-cv | complete | 2.7 menit |
| improvement | complete | 14.4 menit |
| cross-device | complete | 0.0 menit |

### Eksperimen A — Baseline pada split resmi

Protokol `official_subject_disjoint` — latih 7352 sampel dari 21 subjek, uji 2947 sampel dari 9 subjek. Guardrail: Tidak ada subjek/indeks yang tumpang tindih.

| Model | Accuracy | Balanced acc. | Macro-F1 | Weighted F1 | Kappa | Latih (s) | Inferensi (s) | Jenis skor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.9250 | 0.9217 | 0.9229 | 0.9248 | 0.9098 | 6.42 | 0.100 | probabilitas |
| SVM (RBF) | 0.9549 | 0.9531 | 0.9540 | 0.9547 | 0.9458 | 2.58 | 3.647 | decision score ternormalisasi (bukan probabilitas) |
| 1D-CNN | 0.9104 | 0.9117 | 0.9121 | 0.9107 | 0.8923 | 18.84 | 0.261 | probabilitas |

**F1 per kelas pada partisi uji resmi** (precision / recall / F1)

| Aktivitas | Random Forest (P / R / F1) | SVM (RBF) (P / R / F1) | 1D-CNN (P / R / F1) |
| --- | --- | --- | --- |
| Walking | 0.892 / 0.964 / 0.926 | 0.955 / 0.974 / 0.964 | 0.961 / 0.994 / 0.977 |
| Walking Upstairs | 0.892 / 0.911 / 0.901 | 0.932 / 0.964 / 0.948 | 0.984 / 0.900 / 0.940 |
| Walking Downstairs | 0.957 / 0.843 / 0.896 | 0.975 / 0.917 / 0.945 | 0.926 / 0.983 / 0.954 |
| Sitting | 0.916 / 0.888 / 0.902 | 0.963 / 0.896 / 0.928 | 0.861 / 0.756 / 0.805 |
| Standing | 0.899 / 0.925 / 0.912 | 0.915 / 0.968 / 0.941 | 0.766 / 0.887 / 0.822 |
| Laying | 1.000 / 1.000 / 1.000 | 0.996 / 1.000 / 0.998 | 1.000 / 0.950 / 0.974 |

### Eksperimen B — Generalisasi lintas pengguna

| Model | Protokol | Fold | Macro-F1 (mean ± std) | Accuracy | Balanced acc. | Macro-F1 subjek min | Macro-F1 subjek max | Latih/fold (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | leave_one_subject_out | 30 | 0.9310 ± 0.0667 | 0.9358 | 0.9357 | 0.7384 | 0.9972 | 8.55 |
| SVM (RBF) | leave_one_subject_out | 30 | 0.9447 ± 0.0576 | 0.9484 | 0.9481 | 0.7297 | 0.9972 | 4.82 |
| 1D-CNN | group_kfold_6 | 6 | 0.9317 ± 0.0414 | 0.9283 | 0.9326 | 0.6894 | 1.0000 | 26.05 |

**Selisih terhadap baseline**

| Model | Baseline macro-F1 | Lintas pengguna macro-F1 | Selisih absolut | Selisih relatif |
| --- | --- | --- | --- | --- |
| Random Forest | 0.9229 | 0.9310 | -0.0080 | -0.9 % |
| SVM (RBF) | 0.9540 | 0.9447 | 0.0092 | 1.0 % |
| 1D-CNN | 0.9121 | 0.9317 | -0.0197 | -2.2 % |

> Baseline memakai split resmi (9 subjek uji), sedangkan lintas pengguna merata-ratakan fold dengan subjek yang berbeda. Perbedaan protokol harus dipertimbangkan saat menafsirkan selisih.
> Tanda positif berarti performa lintas pengguna lebih rendah daripada baseline. Perhatikan bahwa LOSO melatih model pada 29 subjek (bukan 21 subjek pada split resmi), sehingga selisih ini **bukan** ukuran degradasi murni.

**Variasi per subjek (fold lintas pengguna)**

- **Random Forest** — subjek terlemah: S14 (macro-F1 0.7384); terkuat: S24 (macro-F1 0.9972).
- **SVM (RBF)** — subjek terlemah: S14 (macro-F1 0.7297); terkuat: S30 (macro-F1 0.9972).
- **1D-CNN** — subjek terlemah: S14 (macro-F1 0.6894); terkuat: S27 (macro-F1 1.0000).

### Eksperimen D — Strategi peningkatan generalisasi

| Model | Strategi | Macro-F1 (mean ± std) | Δ Macro-F1 vs base | Balanced acc. | Accuracy | Fold | Latih/fold (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | base | 0.9326 ± 0.0284 | 0.0000 | 0.9332 | 0.9325 | 6 | 6.96 |
| Random Forest | normalisasi per-subjek | 0.9529 ± 0.0181 | 0.0203 | 0.9533 | 0.9518 | 6 | 6.68 |
| Random Forest | kalibrasi 10 % label target | 0.9448 ± 0.0216 | 0.0122 | 0.9452 | 0.9444 | 6 | 7.12 |
| SVM (RBF) | base | 0.9428 ± 0.0306 | 0.0000 | 0.9435 | 0.9439 | 6 | 3.32 |
| SVM (RBF) | normalisasi per-subjek | 0.9711 ± 0.0121 | 0.0283 | 0.9713 | 0.9702 | 6 | 2.91 |
| SVM (RBF) | kalibrasi 10 % label target | 0.9676 ± 0.0139 | 0.0248 | 0.9681 | 0.9676 | 6 | 3.51 |
| 1D-CNN | base | 0.9308 ± 0.0429 | 0.0000 | 0.9316 | 0.9274 | 6 | 25.55 |
| 1D-CNN | normalisasi per-subjek | 0.9435 ± 0.0403 | 0.0127 | 0.9443 | 0.9414 | 6 | 29.39 |
| 1D-CNN | kalibrasi 10 % label target | 0.9516 ± 0.0254 | 0.0208 | 0.9517 | 0.9483 | 6 | 43.87 |

**Catatan protokol**

- **equal evaluation** — Semua strategi dievaluasi pada subset evaluasi target yang identik; baris kalibrasi (10%) dikeluarkan dari evaluasi.
- **transductive normalisation** — Standardisasi per-subjek memakai statistik subjek itu sendiri tanpa label (transductive; data target tanpa label dipakai).
- **augmentation** — Hanya tersedia untuk CNN karena membutuhkan sinyal mentah.

### Eksperimen C — Generalisasi lintas perangkat

Status: **not_performed** (tidak dihitung, bukan angka yang dipalsukan).

**Alasan:** UCI HAR hanya menyediakan satu perangkat (Samsung Galaxy S II, waist-mounted). Tidak ada variasi perangkat yang dapat dipakai untuk menguji generalisasi lintas perangkat.

**Kebutuhan data untuk menjalankannya:**
- Minimal dua perangkat smartphone berbeda dengan sensor yang sama jenisnya.
- Protokol aktivitas identik (6 aktivitas, urutan dan durasi sama).
- Posisi pemasangan dan orientasi dicatat per sesi.
- Setiap perangkat memiliki data dari beberapa subjek (idealnya >= 5).
- Metadata perangkat: model, OS, vendor sensor, sampling rate, resolusi.
- Sinyal mentah tersedia dalam window 128 sampel x 9 kanal (50 Hz, overlap 50%).

**Perangkat yang tersedia pada dataset:** Samsung Galaxy S II (satu perangkat)

Prosedur lengkap: [`docs/cross_device_protocol.md`](docs/cross_device_protocol.md)

### Interpretasi singkat

- **Eksperimen A.** Macro-F1 tertinggi pada split resmi: SVM (RBF) (0.9540, accuracy 0.9549); terendah: 1D-CNN (0.9121).
- **Eksperimen B (Random Forest).** Rata-rata macro-F1 lintas pengguna 0.9310 ± 0.0667 dengan rentang antar-subjek 0.7384–0.9972 pada 30 fold (leave_one_subject_out).
- **Eksperimen B (SVM (RBF)).** Rata-rata macro-F1 lintas pengguna 0.9447 ± 0.0576 dengan rentang antar-subjek 0.7297–0.9972 pada 30 fold (leave_one_subject_out).
- **Eksperimen B (1D-CNN).** Rata-rata macro-F1 lintas pengguna 0.9317 ± 0.0414 dengan rentang antar-subjek 0.6894–1.0000 pada 6 fold (group_kfold_6).
- Selisih baseline → lintas pengguna untuk Random Forest: -0.0080 absolut (-0.9 % relatif). Karena LOSO melatih model pada 29 subjek sedangkan split resmi hanya 21 subjek, selisih ini perbandingan antar protokol, bukan bukti degradasi generalisasi.
- Selisih baseline → lintas pengguna untuk SVM (RBF): 0.0092 absolut (1.0 % relatif). Karena LOSO melatih model pada 29 subjek sedangkan split resmi hanya 21 subjek, selisih ini perbandingan antar protokol, bukan bukti degradasi generalisasi.
- Selisih baseline → lintas pengguna untuk 1D-CNN: -0.0197 absolut (-2.2 % relatif). Karena LOSO melatih model pada 29 subjek sedangkan split resmi hanya 21 subjek, selisih ini perbandingan antar protokol, bukan bukti degradasi generalisasi.
- **Eksperimen D (Random Forest).** Δ macro-F1 terhadap `base`: normalisasi per-subjek +0.0203, kalibrasi 10 % label target +0.0122.
- **Eksperimen D (SVM (RBF)).** Δ macro-F1 terhadap `base`: normalisasi per-subjek +0.0283, kalibrasi 10 % label target +0.0248.
- **Eksperimen D (1D-CNN).** Δ macro-F1 terhadap `base`: normalisasi per-subjek +0.0127, kalibrasi 10 % label target +0.0208.
- **Eksperimen C.** Tidak dapat dihitung dari UCI HAR karena hanya ada satu perangkat; status `not_performed` beserta alasannya tercatat di `reports/results/cross_device.json`.

### Figur pendukung

| Figur | Isi |
| --- | --- |
| `reports/figures/01_class_distribution.png` | Distribusi kelas |
| `reports/figures/02_samples_per_subject.png` | Jumlah sampel per subjek |
| `reports/figures/04_model_comparison.png` | Perbandingan model pada baseline |
| `reports/figures/05_per_class_f1.png` | F1 per kelas |
| `reports/figures/07_baseline_vs_cross_user.png` | Baseline vs lintas pengguna |
| `reports/figures/08_cross_user_per_subject.png` | Performa per subjek |
| `reports/figures/10_improvement_comparison.png` | Perbandingan strategi |
| `reports/figures/11_timing.png` | Waktu komputasi |

<!-- END-HASIL-EKSPERIMEN -->

---

## Dashboard Streamlit

```bash
streamlit run app.py
```

| Halaman | Isi |
| --- | --- |
| **Ringkasan** | Fakta dataset, status tiap eksperimen, ringkasan performa, provenance eksekusi (seed, commit, fingerprint, waktu run) |
| **Dataset** | Split resmi,integritas data, distribusi kelas, figur eksplorasi, hasil verifikasi struktur |
| **Baseline** | Metrik A per model, F1 per kelas, confusion matrix, pratinjau prediksi pada sampel uji dengan model tersimpan |
| **Lintas Pengguna** | Tabel B (mean ± std, min/maks per subjek), selisih terhadap baseline, rincian per fold, pratinjau prediksi out-of-fold |
| **Strategi** | Tabel D (Δ macro-F1 tiap strategi terhadap `base`), status Eksperimen C beserta alasannya, inventaris seluruh artefak |

Aturan dashboard:

* **Hanya menampilkan artefak yang benar-benar ada.** Bagian yang belum dijalankan
  menampilkan perintah yang harus dijalankan, bukan angka placeholder.
* **Mode demo terisolasi.** Mode demo membaca folder `reports/demo/` dan
  `models/demo/`, tidak pernah menyentuh artefak penelitian; mode penelitian
  memuat dataset nyata. Mode demo menampilkan banner **DATA DEMO SINTETIS**.
* **Tidak ada training di dashboard.** Model dimuat dari `models/`; metrik dibaca
  dari JSON.

---

## Artefak Keluaran

```text
reports/
├── experiment_summary.md      # Laporan otomatis: ringkasan, tabel, tautan figur
├── report_manifest.json       # Daftar file keluaran + jumlah baris
├── results/
│   ├── baseline.json          # Eksperimen A
│   ├── cross_user.json        # Eksperimen B (LOSO + GroupKFold CNN)
│   ├── improvement.json       # Eksperimen D
│   ├── cross_device.json      # Eksperimen C (status not_performed + alasan)
│   ├── model_config.json      # Hiperparameter yang dibekukan
│   └── run_manifest.json      # Provenance run: seed, env, durasi tiap tahap
├── tables/                    # CSV + JSON ringkasan
├── figures/                   # PNG seluruh grafik laporan
└── predictions/               # NPZ y_true/y_pred/subject per fold (tidak di-commit)
```

Tabel CSV yang dihasilkan: `model_comparison.csv`, `per_class_f1.csv`,
`cross_user_per_fold.csv`, `cross_user_per_subject.csv`, `improvement_comparison.csv`,
`dataset_class_distribution.csv`, `dataset_samples_per_subject.csv`.

---

## Determinisme dan Reproduksi

* Seed global tunggal: **42** (`config/config.yaml` → `project.seed`), diterapkan ke
  Python `random`, NumPy, dan PyTorch.
* `set_seed()` dipanggil pada awal setiap skrip CLI.
* Split LOSO/GroupKFold dan split kalibrasi bersifat deterministik terhadap seed.
* Cache dataset tervalidasi terhadap **fingerprint berkas mentah**; bila berkas
  sumber berubah, cache otomatis dibangun ulang, bukan dipakai diam-diam.
* Manifest run menyimpan versi pustaka, platform, seed, dan durasi tiap tahap
  (`reports/results/run_manifest.json`).
* `run_manifest.json` juga menyimpan commit Git bila repository sudah di-commit.

> Waktu komputasi bergantung perangkat, jumlah thread (`n_jobs`), dan versi
> pustaka. Angka waktu pada laporan adalah hasil pengukuran pada mesin
> pengembangan dan akan berbeda pada mesin lain.

---

## Pengujian

```bash
pytest -q
```

77 tes otomatis, seluruhnya berjalan tanpa GPU dan tanpa akses internet (kecuali
tes yang secara eksplisit *skip* bila dataset belum diunduh).

| Berkas tes | Cakupan |
| --- | --- |
| `tests/test_data_loader.py` | Validasi dataset (NaN, label di luar rentang, panjang nama fitur), cache dan invalidasinya, ringkasan dataset, penolakan ringkasan sintetis pada mode penelitian |
| `tests/test_preprocessing.py` | Standardisasi hanya pada data latih, statistik per subjek, standardisasi sinyal mentah, roundtrip save/load |
| `tests/test_split_strategy.py` | Split resmi tanpa tumpang tindih subjek, LOSO menguji setiap subjek tepat sekali, GroupKFold deterministik & lengkap, split kalibrasi, assertion leakage |
| `tests/test_evaluate.py` | Metrik, confusion matrix, metrik per kelas dan per subjek, agregasi mean/std/CI95, validasi input |
| `tests/test_models.py` | Pabrik model sklearn, jenis skor (probabilitas vs decision score), arsitektur CNN, simpan/muat, seleksi hiperparameter, loop pelatihan |
| `tests/test_generalization.py` | Integrasi Eksperimen A/B/D pada data demo: artefak terbentuk, split bebas subjek, fold identik antar strategi, delta terhadap `base` |
| `tests/test_feature_extraction.py` | Windowing, ekstraksi fitur deterministik, parsing grup fitur UCI, augmentasi reproducible |
| `tests/test_cross_device.py` | Manifest multi-perangkat (termasuk penolakan satu perangkat), status `not_performed`, proteksi terhadap penimpaan hasil yang sudah lengkap |
| `tests/test_dashboard_and_report.py` | Pemuatan bundel, kondisi artefak belum ada, isolasi mode demo/penelitian, tabel dashboard, pembangun laporan |

---

## Batasan Penelitian

Batasan berikut **wajib** dibaca bersama angka hasil:

1. **Satu perangkat saja.** UCI HAR memakai satu Samsung Galaxy S II. Tidak ada
   variasi perangkat, versi OS, posisi pemasangan, atau orientasi. Karena itu
   generalisasi lintas perangkat **tidak dapat diukur** dari dataset ini dan
   Eksperimen C tidak dijalankan.
2. **Populasi terbatas.** 30 subjek sehat dewasa; hasil tidak otomatis berlaku untuk
   anak, lanjut usia, penyandang disabilitas, atau populasi klinis.
3. **Tugas terkontrol.** Aktivitas dilakukan tanpa gangguan; performa di lapangan
   (berisik, berjalan sambil memegang benda, aktivitas sehari-hari yang tidak
   terklasifikasi) pasti lebih rendah.
4. **Fitur prakomputasi.** 561 fitur dihitung oleh pembuat dataset, bukan oleh
   pipeline ini. Pipeline tidak mengulang ekstraksi itu, sehingga perbandingan
   langsung dengan literatur yang mengekstraksi fitur sendiri perlu hati-hati.
5. **Hiperparameter dibekukan untuk LOSO.** Seleksi dilakukan sekali pada subjek
   latih split resmi lalu dibekukan. Seleksi ulang per fold tidak dilakukan karena
   biaya komputasi; ini metode yang sedikit bias-ifavorit ke data validasi.
6. **Jumlah fold tidak sama antar model.** CNN memakai GroupKFold(6) sementara
   RF/SVM memakai LOSO(30). Invariant yang dijaga (setiap subjek diuji tepat sekali)
   tetap berlaku, tetapi rata-rata antar model tidak bandingkan apples-to-apples.
7. **Transduktif pada `subject_norm`.** Normalisasi per-subjek memakai data target
   tanpa label;.ini sah secara metodologis untuk skenario adaptasi, tetapi berarti
   model tidak sepenuhnya "cold start".
8. **`calibration` membutuhkan label target.** Strategi ini realistis pada skenario
   perangkat yang sudah dipakai sebentar, bukan pada pengguna baru.
9. **Tanpa pengulangan statistik.** `random_seed_repeats = 1`; belum ada analisis
   signifikansi/bootstrap antar fold. Perbedaan kecil antar model sebaiknya tidak
   ditafsirkan berlebihan.
10. **Skor SVM bukan probabilitas.** Sudah dijelaskan pada bagian Metrik.

---

## Lisensi & Etika

* **Kode repository ini** dibuat untuk keperluan penelitian akademik.
* **Data UCI HAR** berada di bawah lisensi *AS-IS* dari UCI: penggunaan dalam
  publikasi **wajib** menyertakan sitasi berikut, penggunaan komersial **dilarang**,
  dan pengguna bertanggung jawab atas penggunaan data sesuai ketentuan UCI.

  > Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L.
  > Reyes-Ortiz. *A Public Domain Dataset for Human Activity Recognition Using
  > Smartphones.* 21st European Symposium on Artificial Neural Networks,
  > Computational Intelligence and Machine Learning (ESANN), 2013.

* Dataset berisi data sensor dalam aktivitasmotorik, **bukan data pribadi** dalam
  arti nama, wajah, atau rekaman; yang direkam adalah sinyal inersial.
* Pengumpulan data mandiri untuk Eksperimen C memerlukan persetujuan etis/wawancara
  sesuai ketentuan universitas masing-masing, serta persetujuan peserta.
* Mode demo memakai **data sintetis deterministik** yang dibuat oleh generator pada
  `src/demo_data.py`; tidak berisi data nyata dan tidak boleh dilaporkan sebagai
  hasil penelitian.

---

## Menyimpan ke GitHub

Repository ini sengaja **belum** di-*push* ke GitHub: pada mesin pengembangan tidak
terpasang kredensial GitHub (`gh` tidak terautentikasi). Untuk mempublikasikannya:

```bash
# 1. Inisialisasi (jika belum)
cd smartphone-har-generalization
git init
git branch -M main

# 2. Commit logis
git add config src scripts tests app.py notebooks docs data/README.md README.md requirements.txt pyproject.toml .gitignore
git commit -m "feat: pipeline eksperimen A-D, dashboard Streamlit, dan 77 tes otomatis"

# 3. Hubungkan ke repositori GitHub (butuh login/token)
git remote add origin https://github.com/<user>/smartphone-har-generalization.git
git push -u origin main
```

Atau lewat GitHub CLI:

```bash
gh auth login
gh repo create smartphone-har-generalization --public --source=. --push
```

Berkas besar (dataset, model biner, prediksi per fold) **sengaja tidak** di-commit;
lihat `.gitignore`. Setelah repository di-push, jalankan sekali
`python scripts/run_experiments.py` agar `reports/` terisi hasil yang dapat
diperiksa pembaca.

---

## Referensi

1. Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013).
   *A Public Domain Dataset for Human Activity Recognition Using Smartphones.*
   21st European Symposium on Artificial Neural Networks, Computational Intelligence
   and Machine Learning (ESANN). DOI: <https://doi.org/10.24432/C54S4K>
2. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
3. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
4. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444.
5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.*
   MIT Press. — rujukan umum perancangan 1D-CNN yang diadaptasi pada
   `src/models.py` (blok Conv1D + BatchNorm + MaxPool + Dropout).
6. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.
   *Journal of Machine Learning Research*, 12, 2825–2830.
7. Paszke, D., et al. (2019). PyTorch: An imperative style, high-performance deep
   learning library. *Advances in Neural Information Processing Systems*, 32.
8. Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362.
9. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment.
   *Computing in Science & Engineering*, 9(3), 90–95.
10. Streamlit documentation — <https://docs.streamlit.io/>

---

## Daftar File Penting

| Berkas | Peran |
| --- | --- |
| `config/config.yaml` | Satu-satunya sumber pengaturan: seed, model, protokol, path |
| `src/data_loader.py` | Unduh, verifikasi, muat, cache, ringkasan dataset |
| `src/split_strategy.py` | Semua strategi split + guardrail anti-leakage |
| `src/preprocessing.py` | Standardisasi (train-only) dan statistik per subjek |
| `src/models.py` | Pabrik model sklearn, arsitektur 1D-CNN, jenis skor |
| `src/train.py` | Loop pelatihan, seleksi hiperparameter, simpan/muat model |
| `src/evaluate.py` | Metrik, per kelas, per subjek, agregasi |
| `src/generalization.py` | Orkestrasi Eksperimen A, B, D |
| `src/cross_device.py` | Eksperimen C + status `not_performed` yang eksplisit |
| `src/demo_data.py` | Generator data demo sintetis deterministik |
| `src/dashboard_data.py` | Pembaca artefak read-only untuk dashboard |
| `src/visualization.py` | Seluruh figur laporan |
| `scripts/run_experiments.py` | CLI pipeline utama |
| `scripts/prepare_data.py` | CLI penyiapan dataset |
| `scripts/generate_report.py` | CLI laporan, tabel, figur |
| `scripts/update_readme_results.py` | Mengisi blok "Hasil Eksperimen" di README dari artefak JSON |
| `app.py` | Dashboard Streamlit |
| `docs/cross_device_protocol.md` | Prosedur Eksperimen C + format manifest |
| `data/README.md` | Fakta dataset, checksum, catatan integritas, sitasi |
| `notebooks/01_eksplorasi_dataset.ipynb` | Eksplorasi dataset (hanya membaca artefak) |
| `notebooks/02_hasil_eksperimen.ipynb` | Analisis hasil eksperimen (hanya membaca artefak) |

---

## Pertanyaan umum

**Q: Apakah angka di README bisa dipakai langsung untuk laporan?**
Ya, dengan catatan ikut menyertakan berkas sumbernya di `reports/results/` dan
menjelaskan protokolesnya. Jangan pernah memindahkan angka dari mode demo.

**Q: Mengapa Eksperimen C kosong?**
Karena UCI HAR hanya memiliki satu perangkat. Menjalankan leave-one-device-out pada
satu perangkat tidak bermakna. Repository ini menyiapkan jalur eksekusi dan prosedur
pengumpulan data mandiri sebagai gantinya.

**Q: Berapa lama seluruh pipeline berjalan?**
Bergantung perangkat. Pada mesin pengembangan CPU biasa, tahap baseline + lintas
pengguna + CNN GroupKFold + peningkatan memerlukan waktu `27 menit`. Rincian durasi
per tahap ada di `reports/results/run_manifest.json` (`stage_results.*.duration_s`).

**Q: Apakah bisa memakai GPU?**
Bisa secara teknis, tetapi pipeline ini sengaja berjalan entirely di CPU: tensor
tidak pernah dipindahkan ke GPU (`src/utils.py` bahkan mencatat
`cuda_available` di environment info). Untuk memakai GPU perlu menambahkan
pemindahan tensor ke `cuda` pada `src/train.py` dan `src/models.py`. Perlu diingat
bahwa bottleneck pipeline berada pada Random Forest/SVM (scikit-learn), bukan pada
pelatihan CNN, sehingga waktu total tidak akan berubah drastis.

**Q: Mengapa `features.txt` berubah?**
Tidak berubah. Hanya nama fitur duplikat yang diberi sufiks `__dupN` agar unik
saat disimpan sebagai tabel, dengan urutan kolom tetap sama persis.
