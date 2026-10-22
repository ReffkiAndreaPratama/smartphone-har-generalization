# Folder `data/`

Folder ini menyimpan dataset dan hasil preprocessing. Isi folder yang besar
**tidak** di-commit ke Git (lihat `.gitignore`); semua dapat diunduh ulang atau
dihasilkan ulang dengan perintah di bawah.

```text
data/
├── raw/                 # dataset UCI HAR hasil unduhan (tidak di-commit)
│   ├── uci_har.zip      # arsip ZIP asli
│   └── UCI HAR Dataset/ # folder dataset hasil ekstraksi
├── processed/           # cache numpy dari dataset (dapat dibuat ulang)
├── demo/                # placeholder untuk data demo/manual
└── cross_device/        # (opsional) manifest data multi-perangkat mandiri
```

## Mengunduh dataset UCI HAR

```bash
python scripts/prepare_data.py --download
```

Perintah ini:

1. mengunduh arsip dari UCI (dengan mirror CloudFront sebagai cadangan),
2. memverifikasi CRC arsip ZIP,
3. mengekstrak folder `UCI HAR Dataset`,
4. memverifikasi struktur: 561 fitur, 30 subjek, 7.352 sampel train,
   2.947 sampel test, subjek train dan test tidak beririsan,
5. membuat cache numpy di `data/processed/`,
6. menulis `reports/tables/dataset_summary.json` dan
   `reports/tables/dataset_verification.json`.

## Dataset UCI HAR yang dipakai

| Atribut | Nilai |
| --- | --- |
| Nama | Human Activity Recognition Using Smartphones |
| DOI | <https://doi.org/10.24432/C54S4K> |
| Jumlah sampel | 10.299 window (7.352 train + 2.947 test) |
| Jumlah fitur | 561 (fitur prakomputasi oleh pembuat dataset) |
| Jumlah subjek | 30 (21 latih, 9 uji pada split resmi) |
| Aktivitas | WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING |
| Perangkat | Samsung Galaxy S II (satu perangkat, pinggang) |
| Sinyal | 50 Hz, window 128 sampel (2,56 s), overlap 50%, 9 kanal inersial |
| Lisensi | AS-IS; penggunaan dalam publikasi wajib mensitasi Anguita et al. (2013); penggunaan komersial dilarang |

Sitasi yang wajib disertakan:

> Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L.
> Reyes-Ortiz. *A Public Domain Dataset for Human Activity Recognition Using
> Smartphones.* 21st European Symposium on Artificial Neural Networks,
> Computational Intelligence and Machine Learning (ESANN), 2013.

## Catatan integritas

* `features.txt` berisi 477 nama unik dari 561 baris; 84 baris memiliki nama
  duplikat (fitur `bandsEnergy`). Repository ini memberi sufiks `__dupN` pada
  nama duplikat **tanpa mengubah urutan kolom**, agar aman dipakai pada
  DataFrame/CSV. Jumlahnya tercatat pada `dataset_verification.json`.
* Dataset tidak memiliki nilai kosong dan tidak memiliki baris duplikat pada
  partisi train maupun test (diverifikasi otomatis oleh `verify_uci_structure`).
* Data multi-perangkat **tidak** ada pada UCI HAR. Jika Anda ingin menjalankan
  Eksperimen C, lihat `docs/cross_device_protocol.md` dan isi
  `data/cross_device/manifest.csv` beserta file sinyal Anda.
