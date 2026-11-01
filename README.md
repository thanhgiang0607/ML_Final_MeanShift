# Mean-Shift: cài đặt từ đầu và so sánh

Tự cài thuật toán Mean-Shift theo hướng đối tượng (chỉ dùng NumPy/SciPy), đối chiếu với `sklearn.cluster.MeanShift`, so sánh với K-Means, DBSCAN, GMM và đo thời gian chạy trên 3 bộ dữ liệu thật có sẵn trong scikit-learn: Iris, Wine, Breast Cancer.

## Cấu trúc thư mục

```
├── meanshift/                 # Package Mean-Shift tự cài
│   ├── core.py                #   MeanShiftCustom: fit / predict / fit_predict
│   ├── base.py                #   Lớp cơ sở: get_params, set_params
│   ├── kernels.py             #   FlatKernel, GaussianKernel
│   ├── neighbors.py           #   BruteForceSearch, KDTreeSearch
│   ├── bandwidth.py           #   estimate_bandwidth
│   └── seeding.py             #   get_bin_seeds (bin seeding)
├── tests/                     # Unit test (pytest)
├── experiments/
│   ├── common.py              #   Đọc dữ liệu thật (Iris, Wine, Breast Cancer), chuẩn hoá, mẫu con
│   ├── e1_compare_sklearn.py  #   E1 – đối chiếu với sklearn
│   ├── e2_compare_algorithms.py  # E2 – so với K-Means, DBSCAN, GMM
│   ├── e3_runtime.py          #   E3 – thời gian chạy và bộ nhớ
│   └── fig_*.py               #   Vẽ hình minh hoạ
├── results/                   # Kết quả E1–E3 (hình, bảng tổng hợp, metadata)
└── report/                    # Báo cáo E1–E3, sơ đồ luồng fit() và sơ đồ lớp UML
```

## Cài đặt và chạy

```bash
pip install -r requirements.txt
python3 -m pytest                             # unit test

python3 experiments/e1_compare_sklearn.py     # E1 – đối chiếu sklearn (--seeds 1 để chạy thử)
python3 experiments/e2_compare_algorithms.py  # E2 – so với K-Means, DBSCAN, GMM
python3 experiments/e3_runtime.py             # E3 – thời gian chạy và bộ nhớ
for f in experiments/fig_*.py; do python3 "$f"; done   # hình trong report/figures
```

Kết quả lưu ở `results/eX/`. Thiết kế và kết quả: [`report/THUC_NGHIEM_E1_E2.md`](report/THUC_NGHIEM_E1_E2.md), [`report/THUC_NGHIEM_E3.md`](report/THUC_NGHIEM_E3.md).

## Dùng thử

```python
from meanshift import MeanShiftCustom, estimate_bandwidth

h = estimate_bandwidth(X)
ms = MeanShiftCustom(bandwidth=h, kernel="flat", neighbor_search="kdtree").fit(X)
ms.cluster_centers_, ms.labels_
```
