# Thực nghiệm E1, E2 — Đối chiếu sklearn và so sánh thuật toán trên dữ liệu thật (2 chiều)

## 1. Dữ liệu

Ba bộ dữ liệu thật có nhãn, lấy từ `sklearn.datasets` (bản CSV ở `data/`):

| Bộ | n | Số chiều gốc | Số lớp | PCA 2D giữ lại |
| --- | --- | --- | --- | --- |
| Iris | 150 | 4 | 3 | 95,8% phương sai |
| Wine | 178 | 13 | 3 | 55,4% phương sai |
| Breast Cancer Wisconsin | 569 | 30 | 2 | 63,2% phương sai |

**Tiền xử lý:** mỗi bộ được chuẩn hoá (`StandardScaler`) rồi giảm về **2 chiều** bằng PCA. Mọi thuật toán chạy trên dữ liệu 2D này. Các trục PC1 và PC2 giữ nguyên thang đo sau chuẩn hoá (không làm trắng), nên PC1 có độ phân tán lớn hơn PC2.

**Các lần chạy:**
- **Seed 0** là toàn bộ dữ liệu.
- **Seed 1–9** là mẫu con phân tầng 80%. Mỗi mẫu con được chuẩn hoá và PCA lại riêng.
- Bản 2D của toàn bộ dữ liệu được lưu ở `data/<bộ>_pca2.csv`.

Hình `report/figures/datasets.png` cho thấy ba bộ dữ liệu sau khi giảm chiều:
- **Iris:** *setosa* tách hẳn; *versicolor* và *virginica* chồng lấn một phần.
- **Wine:** ba lớp xếp thành hình chữ V, tách khá rõ.
- **Breast Cancer:** lớp lành tính tạo thành một khối dày đặc; lớp ác tính trải rộng thành một đuôi dài.

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
| Lượt chạy được | 360/360 (không lượt nào báo lỗi) |
| Cùng số cụm | 360/360 (100%) |
| Nhãn giống hệt từng điểm | 360/360 (100%), ARI nhỏ nhất = 1,0 |
| Sai lệch tâm cụm lớn nhất / h | 0 |
| Cùng `n_iter_` | 360/360 |

KDE Gauss tính trực tiếp khớp với `sklearn.neighbors.KernelDensity` tại các điểm trên quỹ đạo, với sai số tuyệt đối lớn nhất 2,6·10⁻¹⁶ (`results/e1/e1_kde_check.json`).

**Kết luận RQ1:** trên dữ liệu thật 2 chiều, `MeanShiftCustom` cho kết quả **trùng khớp hoàn toàn** với sklearn ở mọi cấu hình: mọi mức h, có hoặc không bin seeding, brute hay KD-Tree. Hình `results/e1/e1_examples.png` đặt kết quả của hai cài đặt cạnh nhau, tại α = 1 trên toàn bộ dữ liệu.

### 2.3. Số cụm theo h (trung vị trên 10 seed, bin seeding tắt)

| Bộ | α = 0,25 | α = 0,5 | α = 1 |
| --- | --- | --- | --- |
| Iris | 28,5 | 7 | 2 |
| Wine | 33 | 8 | 3 |
| Breast Cancer | 65 | 18 | 3 |

Ở 2D, số cụm giảm đều và có thể đoán trước khi h tăng. Với α = 1 (h₀ mặc định), số cụm đã gần số lớp thật: Wine ra 3, Iris ra 2 vì *versicolor* và *virginica* chồng lấn.

### 2.4. Minh hoạ quá trình dịch chuyển (Iris 2D)

- **`report/figures/trajectory_iris.png`:**
  - Vẽ 40 quỹ đạo ngẫu nhiên và quỹ đạo dài nhất (từ X[59], 15 bước) trên nền mật độ KDE Epanechnikov.
  - Thiết lập: nhân phẳng, h = 0,5·h₀ = 0,74.
  - Kết quả: 6 mode, trong đó 2 mode là điểm lẻ nằm xa.
  - Vì dữ liệu đã là 2D, hình vẽ đúng không gian mà thuật toán chạy, không phải hình chiếu.
- **`report/figures/density_steps.png`:** mật độ tại điểm đang dịch chuyển từ X[59].
  - Nhân phẳng tăng mật độ Epanechnikov từ 0,081 lên 0,147 (×1,83) sau 15 bước.
  - Nhân Gauss tăng mật độ Gauss từ 0,052 lên 0,087 (×1,68) sau 21 bước.
  - Cả hai tăng đơn điệu, đúng với lý thuyết leo dốc trên KDE.

## 3. E2 — So sánh với K-Means, DBSCAN, GMM

### 3.1. Thiết lập

- **Mean-Shift (h mặc định):** h₀ = `estimate_bandwidth`, không cần biết trước số cụm.
- **Mean-Shift (h chọn):** h = α·h₀, α ∈ {0.25, 0.5, 0.75, 1, 1.5, 2}.
- **DBSCAN:** `min_samples = 5`, eps ∈ {0.1, …, 3.0}.
- **K-Means, GMM:** được cho **biết trước số lớp thật**. K-Means dùng `n_init = 10`, GMM dùng `full`.
- **Chọn tham số không dùng nhãn:**
  - α và eps được chọn theo **silhouette** cao nhất trên seed 0.
  - Chỉ xét cấu hình có ≥ 2 cụm và ≥ 50% số điểm không bị DBSCAN coi là nhiễu. Silhouette tính trên các điểm không phải nhiễu.
  - Ở 2D, cả ba bộ đều có cấu hình thoả điều kiện, nên không cần nới.
- **Báo cáo:** trung bình trên **seed 1–9**.

ARI của từng ứng viên được ghi trong `results/e2/e2_selection_params.csv` chỉ để đối chiếu, không dùng khi chọn.

| Bộ | α chọn | silhouette | eps chọn | silhouette |
| --- | --- | --- | --- | --- |
| Iris | 1,0 | 0,615 | 0,4 | 0,670 |
| Wine | 0,75 | 0,561 | 0,5 | 0,523 |
| Breast Cancer | 1,0 | 0,457 | 1,5 | 0,408 |

### 3.2. Kết quả (ARI trung bình ± độ lệch chuẩn, seed 1–9)

| Bộ | MS h mặc định | MS h chọn | K-Means | DBSCAN | GMM |
| --- | --- | --- | --- | --- | --- |
| Iris | 0,567 ± 0,000 | 0,567 ± 0,000 | 0,619 ± 0,027 | 0,462 ± 0,012 | **0,739 ± 0,098** |
| Wine | 0,866 ± 0,025 | 0,781 ± 0,122 | **0,879 ± 0,028** | 0,399 ± 0,061 | 0,830 ± 0,089 |
| Breast Cancer | 0,163 ± 0,041 | 0,163 ± 0,041 | **0,639 ± 0,021** | 0,042 ± 0,007 | 0,616 ± 0,025 |

NMI cho thứ tự xếp hạng giống ARI trên Wine và Breast Cancer. Riêng Iris, Mean-Shift có NMI 0,734, cao hơn K-Means (0,663): NMI không phạt việc gộp *versicolor* và *virginica* nặng bằng ARI (`results/e2/e2_summary.csv`). Hình: `results/e2/e2_ari_bar.png` và `results/e2/e2_grid.png` (nhãn thật và kết quả của 5 phương pháp, vẽ trực tiếp trên mặt phẳng 2D).

### 3.3. Nhận xét

- **Wine, nơi Mean-Shift mạnh nhất:**
  - Với h mặc định, Mean-Shift ra đúng 3 cụm ở cả 9 mẫu con. ARI đạt 0,866, ngang K-Means (0,879) và cao hơn GMM (0,830), dù **không được biết trước số cụm**.
  - Ba lớp của Wine là ba vùng mật độ tách nhau trong mặt phẳng PCA, nên mode mật độ trùng với lớp thật.
- **Iris:**
  - Mean-Shift ổn định tuyệt đối (độ lệch chuẩn 0) nhưng chỉ ra 2 cụm, vì *versicolor* và *virginica* chồng lấn tạo thành một đỉnh mật độ chung.
  - Silhouette cũng ưu tiên 2 cụm này, vì chúng tách nhau rõ nhất về hình học.
  - GMM thắng nhờ biết k = 3 và mô hình hoá được cụm hình ellipse.
- **Breast Cancer, nơi Mean-Shift yếu nhất:**
  - Lớp lành tính là một khối dày đặc, còn lớp ác tính là một đuôi thưa trải dài. Ranh giới giữa hai lớp không phải là vùng mật độ thấp, nên không có hai mode tương ứng hai lớp.
  - Mean-Shift ra 3 cụm: một cụm lớn gồm khối dày đặc và phần lớn đuôi, cùng các nhóm nhỏ ở rìa. ARI chỉ 0,163.
  - Silhouette chọn α = 1 (silhouette 0,457, ARI 0,096 trên seed 0). Cấu hình α = 0,75 có ARI 0,626 nhưng silhouette thấp hơn (0,430), nên không được chọn.
- **"MS h chọn" không tốt hơn h mặc định:** chỉ trên Wine là có khác biệt, và còn thấp hơn (0,781 so với 0,866). Lý do: α = 0,75 đôi khi tách thành 4 cụm trên mẫu con. Silhouette chọn α theo hình học trên seed 0, nhưng không bảo đảm α đó tốt cho các mẫu khác.
- **DBSCAN** kém nhất trên cả ba bộ. Một eps chung không phù hợp khi mật độ thay đổi mạnh, đặc biệt ở Breast Cancer với lõi dày và đuôi thưa.
- **K-Means và GMM** ổn định nhất, nhưng được lợi vì biết trước số cụm.

**Kết luận RQ2:**
- Ở 2 chiều, Mean-Shift **cạnh tranh được với K-Means và GMM khi các lớp tương ứng với các vùng mật độ tách biệt** (Wine), mà không cần biết số cụm.
- Mean-Shift kém khi các lớp chồng lấn (Iris) hoặc khi lớp không phải một vùng mật độ riêng (Breast Cancer).
- Chọn h bằng silhouette không cải thiện so với h₀ mặc định.

## 4. Hạn chế

- PCA 2D chỉ giữ 55% phương sai của Wine và 63% của Breast Cancer. Kết quả phản ánh cấu trúc trong mặt phẳng PCA, không phải toàn bộ dữ liệu gốc.
- Các mẫu con 80% chồng lấn nhau nhiều, nên độ lệch chuẩn phản ánh độ ổn định chứ không phải khả năng tổng quát hoá.
- K-Means và GMM được biết trước k, còn Mean-Shift và DBSCAN thì không, nên so sánh vẫn nghiêng về K-Means và GMM.
- Silhouette chỉ là một trong nhiều tiêu chí không giám sát. Chỉ có 3 bộ dữ liệu, kích thước nhỏ (≤ 569 điểm).
