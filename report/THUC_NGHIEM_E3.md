# Thực nghiệm E3 — Thời gian chạy và bộ nhớ trên dữ liệu thật (2 chiều)

## 1. Mục tiêu

Trả lời RQ3: KD-Tree và bin seeding giảm chi phí tính toán bao nhiêu, trong điều kiện nào, và có làm đổi kết quả không? So sánh với `sklearn.cluster.MeanShift`.

## 2. Thiết lập

- **Dữ liệu:** Iris, Wine, Breast Cancer, chuẩn hoá rồi giảm về **2 chiều** bằng PCA (giống E1, E2).
- **Năm bản cài đặt:** `custom brute`, `custom kdtree`, `custom kdtree + bin` (bin seeding), `sklearn`, `sklearn + bin`. Tất cả dùng nhân phẳng; sklearn chạy với `n_jobs=1`, mọi thư viện giới hạn 1 luồng.
- **Theo n:** dùng Breast Cancer 2D, lấy mẫu con phân tầng n ∈ {100, 200, 300, 400, 500, 569}. h giữ cố định bằng `estimate_bandwidth` của toàn bộ dữ liệu.
- **Theo h:** chạy cả 3 bộ với đủ dữ liệu, h = α·h₀, α ∈ {0.25, 0.5, 1, 2}.
- **Đo:**
  - Thời gian lấy trung vị 3 lần chạy.
  - Bộ nhớ đỉnh đo bằng `tracemalloc` ở 1 lần chạy riêng.
  - Ghi thêm số seed, số cụm và số vòng lặp.
- **Máy:** Apple M1, Python 3.14.5, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.8.0 (`results/e3/env.txt`).

Chạy lại: `python3 experiments/e3_runtime.py` (khoảng 1 phút).

## 3. Kết quả theo n (Breast Cancer 2D, h cố định)

| n | custom brute | custom kdtree | custom kdtree + bin | sklearn | sklearn + bin |
| --- | --- | --- | --- | --- | --- |
| 100 | 0,0067 s | 0,0083 s | 0,0021 s | 0,0611 s | 0,0168 s |
| 300 | 0,0386 s | 0,0581 s | 0,0060 s | 0,2838 s | 0,0319 s |
| 569 | 0,1060 s | 0,2221 s | 0,0120 s | 0,5972 s | 0,0432 s |

Hình: `results/e3/e3_time_vs_n.png`, `results/e3/e3_memory_vs_n.png`.

- **KD-Tree chậm hơn brute** ở mọi n, chỉ đạt 0,48–0,80 lần tốc độ của brute. Ở 2D, nguyên nhân không phải số chiều mà là **n quá nhỏ**:
  - Bản brute tính khoảng cách cho cả lô seed bằng một phép nhân ma trận (BLAS). Với n ≤ 569, ma trận khoảng cách rất nhỏ nên phép tính này rất nhanh.
  - Bản KD-Tree gọi `cKDTree.query_ball_point`, trả về một danh sách Python cho từng seed, rồi phải chuyển đổi và sắp xếp từng danh sách. Chi phí này lớn hơn phần tính toán mà cây tiết kiệm được.
  - Lợi thế của KD-Tree chỉ xuất hiện khi n đủ lớn để chi phí O(n) của mỗi truy vấn brute trở nên đáng kể, vùng này nằm ngoài phạm vi 3 bộ dữ liệu.
- **Bin seeding** giảm số seed từ 569 xuống 27 (−95%) và nhanh hơn brute **8,8 lần** ở n = 569. **Kết quả không đổi:** vẫn 3 cụm, số vòng lặp gần như nhau (24 so với 23 khi không dùng bin).
- **So với sklearn:** bản tự cài nhanh hơn rõ.
  - Ở n = 569: `custom brute` mất 0,106 s so với 0,597 s của sklearn (nhanh hơn 5,6 lần); `custom kdtree + bin` mất 0,012 s so với 0,043 s của `sklearn + bin`.
  - Hai cài đặt cho cùng nhãn (E1). Khác biệt đến từ việc bản tự cài dịch chuyển mọi seed theo lô bằng NumPy, còn sklearn xử lý tuần tự từng seed.
- **Bộ nhớ:**
  - `custom brute` và `custom kdtree` tăng nhanh theo n (7,7 MB và 9,4 MB ở n = 569), vì giữ khoảng cách và chỉ số láng giềng của cả lô seed.
  - sklearn dùng dưới 0,8 MB.
  - Bin seeding giảm bộ nhớ của bản tự cài xuống 0,28 MB.

## 4. Kết quả theo h

| Bộ | α | custom brute | custom kdtree | kdtree + bin | sklearn | số seed (bin) | số cụm (không bin / bin) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Iris | 0,5 | 0,0117 s | 0,0138 s | 0,0037 s | 0,1082 s | 35 | 6 / 6 |
| Iris | 1 | 0,0141 s | 0,0161 s | 0,0024 s | 0,1191 s | 17 | 2 / 2 |
| Wine | 1 | 0,0335 s | 0,0193 s | 0,0017 s | 0,1238 s | 13 | 3 / 3 |
| Breast Cancer | 0,25 | 0,0688 s | 0,0790 s | 0,0166 s | 0,5752 s | 160 | 64 / 68 |
| Breast Cancer | 0,5 | 0,1458 s | 0,2093 s | 0,0177 s | 0,9094 s | 68 | 19 / 17 |
| Breast Cancer | 2 | 0,0834 s | 0,1665 s | 0,0046 s | 0,3704 s | 12 | 1 / 1 |

Hình: `results/e3/e3_time_vs_h.png`. Bảng đầy đủ: `results/e3/e3_summary.csv`.

- **KD-Tree so với brute:**
  - Trong 12 cấu hình theo h, KD-Tree chỉ nhanh hơn brute 2 lần: Iris α = 0,25 (×1,09) và Wine α = 1 (×1,73). Các trường hợp còn lại đạt 0,50–0,91 lần tốc độ brute.
  - Riêng Wine α = 1, thời gian của brute cao bất thường so với các α lân cận, nên trường hợp này có thể do nhiễu đo.
  - Khi h lớn, mỗi truy vấn trả về nhiều điểm nên KD-Tree càng mất lợi thế (Breast Cancer α = 2: ×0,50).
- **Thời gian không đơn điệu theo h:** h trung bình cần nhiều vòng lặp nhất (Breast Cancer α = 0,5: 36 vòng), còn h lớn thì hội tụ nhanh (α = 2: 9 vòng).
- **Bin seeding càng lợi khi h lớn:** lưới thô hơn nên ít seed hơn. Bin seeding nhanh hơn brute 2–20 lần (lớn nhất ở Wine α = 1: ×19,7; Breast Cancer α = 2: ×18,2).
- **Kết quả với bin seeding ở 2D:** số cụm **giữ nguyên** ở mọi cấu hình có α ≥ 0,5, trừ Breast Cancer α = 0,5 (19 → 17 cụm). Ở α = 0,25, số cụm lệch nhẹ (Iris 29 → 27, Breast Cancer 64 → 68), vì khi h nhỏ thì nhiều mode nhỏ dễ bị bỏ sót hoặc tách thêm.
- **sklearn** chậm nhất ở mọi cấu hình: chậm hơn `custom brute` 4–12 lần.

## 5. Kết luận RQ3

- Trên dữ liệu thật 2D với n ≤ 569, **KD-Tree không giúp tăng tốc** so với brute đã vector hoá: chậm hơn tới khoảng 50%. Lý do là n nhỏ và chi phí xử lý kết quả truy vấn theo từng seed, chứ không phải số chiều.
- **Bin seeding** là cách tăng tốc hiệu quả nhất (nhanh hơn brute tới gần 20 lần, bộ nhớ giảm khoảng 25 lần). Ở 2D, nó **hầu như không đổi kết quả** khi h ≥ 0,5·h₀; sai khác chỉ đáng kể khi h nhỏ.
- Bản tự cài nhanh hơn sklearn 4–12 lần nhờ xử lý theo lô, đổi lại tốn bộ nhớ nhiều hơn.

## 6. Hạn chế

- n tối đa chỉ 569, nên chưa quan sát được vùng n lớn nơi KD-Tree có thể vượt brute. Mọi thời gian đều dưới 1 s nên dễ bị ảnh hưởng bởi tải của máy.
- Phép đo theo n chỉ dùng một bộ (Breast Cancer 2D).
- `tracemalloc` chỉ đo bộ nhớ do Python/NumPy cấp phát, không đo phần bộ nhớ riêng bên trong thư viện C của `cKDTree`.
