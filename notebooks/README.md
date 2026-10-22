# Notebooks

Dua notebook pendukung untuk laporan. Keduanya hanya **membaca** artefak yang sudah
dihasilkan pipeline; tidak ada model yang dilatih di dalam notebook.

| Notebook | Isi | Prasyarat |
| --- | --- | --- |
| `01_eksplorasi_dataset.ipynb` | Distribusi kelas, sampel per subjek, komposisi fitur, hasil verifikasi integritas | `python scripts/prepare_data.py --download` |
| `02_hasil_eksperimen.ipynb` | Tabel dan grafik hasil Eksperimen A, B, D, dan status Eksperimen C | `python scripts/run_experiments.py` |

Menjalankan notebook dari dalam folder `notebooks/`:

```bash
cd notebooks
jupyter notebook
```

Seluruh angka yang ditampilkan berasal dari berkas JSON/CSV pada `reports/`.
Bila sebuah eksperimen belum dijalankan, notebook menandakan kondisi tersebut
alih-alih menampilkan angka.
