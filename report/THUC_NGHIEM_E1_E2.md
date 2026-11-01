# Thực nghiệm E1, E2 — Đối chiếu sklearn và so sánh thuật toán trên dữ liệu thật

## 1. Dữ liệu

Ba bộ dữ liệu thật có nhãn, lấy trực tiếp từ `sklearn.datasets` (không cần tải):

| Bộ | n | d | Số lớp | Ghi chú |
| --- | --- | --- | --- | --- |
| Iris | 150 | 4 | 3 | *versicolor* và *virginica* chồng lấn |
| Wine | 178 | 13 | 3 | các đặc trưng có thang đo rất khác nhau |
| Breast Cancer Wisconsin | 569 | 30 | 2 | nhiều chiều nhất |

- Mọi bộ được chuẩn hoá bằng `StandardScaler` và chạy trên **đủ số chiều**.
- **Seed 0** là toàn bộ dữ liệu. **Seed 1–9** là mẫu con phân tầng 80% (giữ tỉ lệ lớp), chuẩn hoá lại trên từng mẫu con. Cách này cho 10 lần chạy trên mỗi bộ, tương tự 10 seed của dữ liệu tổng hợp trước đây.
- Các hình chiếu dữ liệu xuống PCA 2D chỉ để trực quan hoá. Hình `report/figures/datasets.png` cho thấy PCA 2D giữ 96% phương sai của Iris, nhưng chỉ 55% của Wine và 63% của Breast Cancer.

Chạy lại:

```bash
python3 experiments/e1_compare_sklearn.py
python3 experiments/e2_compare_algorithms.py
```

## 2. E1 — Đối chiếu `MeanShiftCustom` với `sklearn.cluster.MeanShift`

### 2.1. Thiết lập

- **Lưới chạy:** 3 bộ × 10 seed × α ∈ {0.25, 0.5, 1} × bin seeding {tắt, bật} × tìm láng giềng {brute, kdtree} = **360 lượt**.
- **Tham số:** h = α · `estimate_bandwidth(X, quantile=0.3)`, nhân phẳng, `tol = 1e-3`, `max_iter = 300`, `cluster_all=True`, 1 luồng.
- **So sánh:** số cụm, nhãn (ARI và giống hệt từng điểm), tâm cụm (ghép Hungarian, sai lệch chia cho h), số vòng lặp `n_iter_`.

### 2.2. Kết quả

| Tiêu chí | Kết quả |
| --- | --- |
| Lượt chạy được | 340/360 |
| Cùng số cụm | 340/340 (100%) |
| Nhãn giống hệt từng điểm | 340/340 (100%), ARI nhỏ nhất = 1,0 |
| Sai lệch tâm cụm lớn nhất / h | 0 |
| Cùng `n_iter_` | 340/340 |
| Cả hai cùng báo lỗi | 20/360 |
| Chỉ một bên báo lỗi | 0 |

**20 lượt lỗi** đều thuộc đúng một cấu hình: Breast Cancer, α = 0,25, bật bin seeding. Ở 30 chiều với h nhỏ, mỗi điểm rơi vào một ô lưới riêng, và tâm các ô nằm xa hơn h so với mọi điểm dữ liệu. Vì vậy không seed nào có láng giềng, và **cả hai cài đặt cùng báo `ValueError`**. Lúc này sklearn báo "No point was within bandwidth", còn bản tự cài báo "Không có điểm nào trong bán kính h". Đây là hành vi giống nhau, không phải sai khác.

Ngoài ra còn kiểm tra KDE Gauss tính trực tiếp so với `sklearn.neighbors.KernelDensity` tại các điểm trên quỹ đạo: sai số tuyệt đối lớn nhất là 2,6·10⁻¹⁶ (`results/e1/e1_kde_check.json`).

**Kết luận RQ1:** trên dữ liệu thật đến 30 chiều, `MeanShiftCustom` cho kết quả **trùng khớp hoàn toàn** với sklearn, kể cả khi bật bin seeding và với cả hai cách tìm láng giềng.

### 2.3. Số cụm theo h (trung vị trên 10 seed, bin seeding tắt)

| Bộ | α = 0,25 | α = 0,5 | α = 1 |
| --- | --- | --- | --- |
| Iris | 50,5 | 10 | 2 |
| Wine | 142 | 55,5 | 2 |
| Breast Cancer | 437 | 129,5 | 13,5 |

Ở số chiều cao, h nhỏ khiến gần như mỗi điểm thành một cụm riêng. Ví dụ Wine với α = 0,25 có trung vị 0 vòng lặp: không seed nào có láng giềng ngoài chính nó.

### 2.4. Minh hoạ quá trình dịch chuyển

Hai hình dưới đây dùng **Iris chiếu PCA 2D**, vì quỹ đạo chỉ vẽ được trong 2 chiều:

- `report/figures/trajectory_iris.png`: 40 quỹ đạo ngẫu nhiên và quỹ đạo dài nhất (từ X[59], 15 bước) trên nền mật độ KDE Epanechnikov, với nhân phẳng và h = 0,5·h₀ = 0,74. Kết quả là 6 mode, trong đó 2 mode là điểm lẻ nằm xa.
- `report/figures/density_steps.png`: mật độ tại điểm đang dịch chuyển từ X[59].
  - Nhân phẳng tăng mật độ KDE Epanechnikov từ 0,081 lên 0,147 (×1,83) sau 15 bước.
  - Nhân Gauss tăng mật độ KDE Gauss từ 0,052 lên 0,087 (×1,68) sau 21 bước.
  - Cả hai đều tăng đơn điệu, đúng với lý thuyết: Mean-Shift là phương pháp leo dốc trên KDE.

## 3. E2 — So sánh với K-Means, DBSCAN, GMM

### 3.1. Thiết lập

- **Mean-Shift (h mặc định):** h₀ = `estimate_bandwidth`, không cần biết trước số cụm.
- **Mean-Shift (h chọn):** h = α·h₀, α ∈ {0.25, 0.5, 0.75, 1, 1.5, 2}.
- **DBSCAN:** `min_samples = 5`, eps ∈ {0.1, …, 3.0}.
- **K-Means và GMM:** được cho **biết trước số lớp thật**. K-Means dùng `n_init = 10`, GMM dùng `full`.
- **Chọn tham số không dùng nhãn:**
  - α và eps được chọn theo **silhouette score** cao nhất trên seed 0 (toàn bộ dữ liệu). Silhouette chỉ dựa vào khoảng cách giữa các điểm, giống cách phân cụm được dùng ngoài thực tế khi không có nhãn.
  - Chỉ xét cấu hình có ≥ 2 cụm và ít nhất 50% số điểm không bị DBSCAN coi là nhiễu. Silhouette tính trên các điểm không phải nhiễu.
  - Nếu không cấu hình nào thoả điều kiện thì bỏ điều kiện về tỉ lệ nhiễu. Trường hợp này chỉ xảy ra với DBSCAN trên Breast Cancer.
- **Báo cáo:** tham số giữ nguyên cho seed 1–9, và bảng dưới là trung bình trên **seed 1–9**.

Nhãn **không được dùng** ở bất kỳ bước chọn tham số nào. ARI của từng ứng viên vẫn được ghi trong `results/e2/e2_selection_params.csv`, nhưng chỉ để đối chiếu.

| Bộ | α chọn | silhouette | eps chọn | silhouette |
| --- | --- | --- | --- | --- |
| Iris | 1,0 | 0,582 | 0,5 | 0,656 |
| Wine | 0,75 | 0,251 | 2,0 | 0,240 |
| Breast Cancer | 1,5 | 0,591 | 2,0 (điều kiện nới) | 0,098 |

### 3.2. Kết quả (ARI trung bình ± độ lệch chuẩn, seed 1–9)

| Bộ | MS h mặc định | MS h chọn | K-Means | DBSCAN | GMM |
| --- | --- | --- | --- | --- | --- |
| Iris | 0,567 ± 0,000 | 0,567 ± 0,000 | 0,617 ± 0,031 | 0,428 ± 0,072 | **0,847 ± 0,106** |
| Wine | −0,007 ± 0,002 | 0,535 ± 0,187 | **0,886 ± 0,019** | 0,251 ± 0,053 | 0,813 ± 0,161 |
| Breast Cancer | 0,009 ± 0,027 | 0,004 ± 0,003 | 0,644 ± 0,019 | 0,065 ± 0,020 | **0,740 ± 0,037** |

Hình: `results/e2/e2_ari_bar.png` (biểu đồ cột) và `results/e2/e2_grid.png` (nhãn thật và kết quả của 5 phương pháp, chiếu PCA 2D).

### 3.3. Nhận xét

- **Iris:** Mean-Shift ổn định tuyệt đối (độ lệch chuẩn 0) nhưng chỉ ra 2 cụm, tách *setosa* khỏi phần còn lại. *Versicolor* và *virginica* chồng lấn nên tạo thành một đỉnh mật độ chung, do đó không có mode riêng cho từng loài. Silhouette cũng chọn h cho 2 cụm, vì 2 cụm này tách nhau rõ nhất về hình học. GMM thắng nhờ được biết k = 3.
- **Wine:**
  - Với h mặc định, Mean-Shift thất bại (ARI ≈ 0).
  - Chọn α = 0,75 theo silhouette cho 3 cụm và ARI 0,535. Kết quả tốt hơn nhưng dao động lớn giữa các mẫu con (độ lệch chuẩn 0,19).
- **Breast Cancer, chọn h không dùng nhãn thất bại:**
  - Silhouette chọn α = 1,5. Cấu hình này cho 2 cụm, nhưng một cụm chỉ gồm vài điểm lẻ nằm xa, nên ARI chỉ 0,004.
  - Cấu hình α = 0,5 cho ARI 0,749 nhưng có silhouette thấp hơn (0,316 so với 0,591), nên không được chọn.
  - Đây là điểm yếu đã biết của silhouette: tách riêng vài điểm ngoại lai thì silhouette cao, dù không phản ánh cấu trúc lớp.
- **DBSCAN:** gặp cùng vấn đề mật độ ở nhiều chiều. Trên Wine và Breast Cancer, DBSCAN đánh dấu 59–70% số điểm là nhiễu.
- **K-Means và GMM:** ổn định nhất, nhưng được lợi vì biết trước số cụm.

**Kết luận RQ2:**
- Khi không dùng nhãn (đúng như ngoài thực tế), Mean-Shift chỉ cạnh tranh được ở dữ liệu ít chiều (Iris).
- Ở 13–30 chiều, Mean-Shift kém K-Means và GMM rõ rệt.
- Nguyên nhân chính là **chọn h**: h mặc định thì gộp tất cả, còn tiêu chí không giám sát như silhouette có thể chọn sai, như ở Breast Cancer.

## 4. Hạn chế

- Các mẫu con 80% chồng lấn nhau nhiều, nên độ lệch chuẩn phản ánh độ ổn định chứ không phải khả năng tổng quát hoá sang dữ liệu mới.
- Mean-Shift và DBSCAN chọn tham số không dùng nhãn, còn K-Means và GMM được biết trước k. Vì vậy so sánh vẫn nghiêng về K-Means và GMM.
- Silhouette chỉ là một trong nhiều tiêu chí không giám sát. Các tiêu chí khác, như Davies–Bouldin, có thể chọn khác.
- Chỉ có 3 bộ dữ liệu, kích thước nhỏ (≤ 569 điểm).
