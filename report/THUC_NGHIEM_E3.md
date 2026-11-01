# Thực nghiệm E3 — Thời gian thực thi trên dữ liệu thật (2 chiều)

## 1. Mục tiêu

Trả lời RQ3: KD-Tree và bin seeding giảm thời gian thực thi bao nhiêu, trong điều kiện nào, và có làm đổi kết quả không? So sánh với `sklearn.cluster.MeanShift`.

## 2. Thiết lập

- **Dữ liệu:** Iris, Wine, Breast Cancer, chuẩn hoá rồi giảm về **2 chiều** bằng PCA (giống E1, E2).
- **Năm bản cài đặt:** `custom brute`, `custom kdtree`, `custom kdtree + bin` (bin seeding), `sklearn`, `sklearn + bin`. Tất cả dùng nhân phẳng; sklearn chạy với `n_jobs=1`, mọi thư viện giới hạn 1 luồng.
- **Theo n:** dùng Breast Cancer 2D, lấy mẫu con phân tầng n ∈ {100, 200, 300, 400, 500, 569}. h giữ cố định bằng `estimate_bandwidth` của toàn bộ dữ liệu.
- **Theo h:** chạy cả 3 bộ với đủ dữ liệu, h = α·h₀, α ∈ {0.25, 0.5, 1, 2}.
- **Đo:** thời gian thực thi của `fit`, lấy trung vị 3 lần chạy (`time.perf_counter`). Ghi thêm số seed, số cụm và số vòng lặp để biết việc tăng tốc có làm đổi kết quả không.
- **Máy:** Apple M1, Python 3.14.5, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.8.0 (`results/e3/env.txt`).

Chạy lại: `python3 experiments/e3_runtime.py` (khoảng 1 phút).

## 3. Kết quả theo n (Breast Cancer 2D, h cố định)

| n | custom brute | custom kdtree | custom kdtree + bin | sklearn | sklearn + bin |
| --- | --- | --- | --- | --- | --- |
| 100 | 0,0065 s | 0,0074 s | 0,0024 s | 0,0574 s | 0,0161 s |
| 300 | 0,0360 s | 0,0560 s | 0,0067 s | 0,2855 s | 0,0309 s |
| 569 | 0,1535 s | 0,1943 s | 0,0118 s | 0,6151 s | 0,0446 s |

Hình: `results/e3/e3_time_vs_n.png`.

- **KD-Tree chậm hơn brute** ở mọi n, chỉ đạt 0,54–0,88 lần tốc độ của brute. Ở 2D, nguyên nhân không phải số chiều mà là **n quá nhỏ**:
  - Bản brute tính khoảng cách cho cả lô seed bằng một phép nhân ma trận (BLAS). Với n ≤ 569, ma trận khoảng cách rất nhỏ nên phép tính này rất nhanh.
  - Bản KD-Tree gọi `cKDTree.query_ball_point`, trả về một danh sách Python cho từng seed, rồi phải chuyển đổi và sắp xếp từng danh sách. Chi phí này lớn hơn phần tính toán mà cây tiết kiệm được.
  - Lợi thế của KD-Tree chỉ xuất hiện khi n đủ lớn để chi phí O(n) của mỗi truy vấn brute trở nên đáng kể, vùng này nằm ngoài phạm vi 3 bộ dữ liệu.
- **Bin seeding** giảm số seed từ 569 xuống 27 (−95%) và nhanh hơn brute **13 lần** ở n = 569. **Kết quả không đổi:** vẫn 3 cụm, số vòng lặp gần như nhau (24 so với 23 khi không dùng bin).
- **So với sklearn:** bản tự cài nhanh hơn rõ.
  - Ở n = 569: `custom brute` mất 0,154 s so với 0,615 s của sklearn (nhanh hơn 4 lần); `custom kdtree + bin` mất 0,012 s so với 0,045 s của `sklearn + bin`.
  - Hai cài đặt cho cùng nhãn (E1). Khác biệt đến từ việc bản tự cài dịch chuyển mọi seed theo lô bằng NumPy, còn sklearn xử lý tuần tự từng seed.

## 4. Kết quả theo h

| Bộ | α | custom brute | custom kdtree | kdtree + bin | sklearn | số seed (bin) | số cụm (không bin / bin) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Iris | 0,5 | 0,0138 s | 0,0130 s | 0,0037 s | 0,1166 s | 35 | 6 / 6 |
| Iris | 1 | 0,0119 s | 0,0155 s | 0,0025 s | 0,1091 s | 17 | 2 / 2 |
| Wine | 1 | 0,0152 s | 0,0194 s | 0,0017 s | 0,1261 s | 13 | 3 / 3 |
| Breast Cancer | 0,25 | 0,0744 s | 0,0743 s | 0,0156 s | 0,5603 s | 160 | 64 / 68 |
| Breast Cancer | 0,5 | 0,1505 s | 0,2106 s | 0,0190 s | 0,9587 s | 68 | 19 / 17 |
| Breast Cancer | 2 | 0,0865 s | 0,1721 s | 0,0046 s | 0,3845 s | 12 | 1 / 1 |

Hình: `results/e3/e3_time_vs_h.png`. Bảng đầy đủ: `results/e3/e3_summary.csv`.

- **KD-Tree so với brute:**
  - Trong 12 cấu hình, KD-Tree chỉ nhanh hơn brute một chút ở Iris α = 0,5 (×1,06) và ngang brute ở Breast Cancer α = 0,25 (×1,00). Các trường hợp còn lại đạt 0,50–0,94 lần tốc độ brute.
  - Khi h lớn, mỗi truy vấn trả về nhiều điểm nên KD-Tree càng mất lợi thế (Breast Cancer α = 2: ×0,50).
- **Thời gian không đơn điệu theo h:** h trung bình cần nhiều vòng lặp nhất (Breast Cancer α = 0,5: 36 vòng), còn h lớn thì hội tụ nhanh (α = 2: 9 vòng).
- **Bin seeding càng lợi khi h lớn:**
  - Lưới thô hơn nên ít seed hơn. Khi h nhỏ, bin seeding gần như không có lợi (Wine α = 0,25: 83/178 seed, ×0,99). Khi h lớn, nó nhanh hơn brute tới **18,9 lần** (Breast Cancer α = 2).
  - **Kết quả ở 2D:** số cụm **giữ nguyên** ở mọi cấu hình có α ≥ 0,5, trừ Breast Cancer α = 0,5 (19 → 17 cụm). Ở α = 0,25, số cụm lệch nhẹ (Iris 29 → 27, Breast Cancer 64 → 68), vì khi h nhỏ thì nhiều mode nhỏ dễ bị bỏ sót hoặc tách thêm.
- **sklearn** chậm nhất ở mọi cấu hình: chậm hơn `custom brute` 4–10 lần.

## 5. Kết luận RQ3

- Trên dữ liệu thật 2D với n ≤ 569, **KD-Tree không giúp tăng tốc** so với brute đã vector hoá: chậm hơn tới khoảng 50%. Lý do là n nhỏ và chi phí xử lý kết quả truy vấn theo từng seed, chứ không phải số chiều.
- **Bin seeding** là cách tăng tốc hiệu quả nhất (nhanh hơn brute tới khoảng 19 lần khi h lớn). Ở 2D, nó **hầu như không đổi kết quả** khi h ≥ 0,5·h₀; sai khác chỉ đáng kể khi h nhỏ, và lúc đó cũng gần như không tăng tốc.
- Bản tự cài nhanh hơn sklearn 4–10 lần nhờ dịch chuyển các seed theo lô.

## 6. Hạn chế

- n tối đa chỉ 569, nên chưa quan sát được vùng n lớn nơi KD-Tree có thể vượt brute. Mọi thời gian đều dưới 1 s nên dễ bị ảnh hưởng bởi tải của máy.
- Phép đo theo n chỉ dùng một bộ (Breast Cancer 2D).
- Chỉ đo thời gian thực thi, không đo bộ nhớ.
