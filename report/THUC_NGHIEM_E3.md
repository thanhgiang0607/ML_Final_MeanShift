# Thực nghiệm E3 — Thời gian chạy và bộ nhớ trên dữ liệu thật

## 1. Mục tiêu

Trả lời RQ3: KD-Tree và bin seeding giảm chi phí tính toán bao nhiêu, và trong điều kiện nào? So sánh với `sklearn.cluster.MeanShift`.

## 2. Thiết lập

- **Năm bản cài đặt:** `custom brute`, `custom kdtree`, `custom kdtree + bin` (bin seeding), `sklearn`, `sklearn + bin`. Tất cả dùng nhân phẳng; sklearn chạy với `n_jobs=1`, mọi thư viện giới hạn 1 luồng.
- **Theo n:** dùng Breast Cancer (d = 30), chuẩn hoá toàn bộ rồi lấy mẫu con phân tầng n ∈ {100, 200, 300, 400, 500, 569}. h giữ cố định bằng `estimate_bandwidth` của toàn bộ dữ liệu, để chỉ n thay đổi.
- **Theo h:** chạy cả 3 bộ (Iris, Wine, Breast Cancer) với đủ dữ liệu, h = α·h₀, α ∈ {0.25, 0.5, 1, 2}.
- **Đo:**
  - Thời gian lấy trung vị 3 lần chạy (`time.perf_counter`).
  - Bộ nhớ đỉnh đo bằng `tracemalloc` ở 1 lần chạy riêng.
  - Ghi thêm số seed, số cụm và số vòng lặp.
- **Máy:** Apple M1, Python 3.14.5, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.8.0 (`results/e3/env.txt`).

Chạy lại: `python3 experiments/e3_runtime.py` (khoảng 40 giây).

## 3. Kết quả theo n (Breast Cancer, h cố định)

| n | custom brute | custom kdtree | custom kdtree + bin | sklearn | sklearn + bin |
| --- | --- | --- | --- | --- | --- |
| 100 | 0,0082 s | 0,0113 s | 0,0011 s | 0,0372 s | 0,0048 s |
| 300 | 0,0410 s | 0,0726 s | 0,0041 s | 0,1235 s | 0,0124 s |
| 569 | 0,1469 s | 0,2691 s | 0,0111 s | 0,2952 s | 0,0216 s |

Hình: `results/e3/e3_time_vs_n.png` và `results/e3/e3_memory_vs_n.png`.

- **KD-Tree chậm hơn brute** ở mọi n, chỉ đạt 0,55–0,72 lần tốc độ của brute. Ở 30 chiều, gần như mọi nút của cây đều giao với hình cầu bán kính h, nên cây không loại bớt được điểm nào mà còn tốn thêm chi phí duyệt cây. Kết quả này khớp với lý thuyết: KD-Tree chỉ hiệu quả khi số chiều thấp.
- **Bin seeding** giảm số seed từ 569 xuống 71 (−88%) và nhanh hơn brute khoảng **13 lần** ở n = 569. Tuy nhiên, **kết quả đổi**: số cụm từ 14 còn 1. Với h này ở 30 chiều, lưới bin quá thô so với cấu trúc dữ liệu.
- **So với sklearn:** bản tự cài nhanh hơn ở mọi cấu hình. Ở n = 569, `custom brute` mất 0,147 s so với 0,295 s của sklearn; `custom kdtree + bin` mất 0,011 s so với 0,022 s của `sklearn + bin`. Cả hai cho cùng nhãn (đã kiểm chứng ở E1), nên khác biệt chỉ nằm ở cách cài đặt: bản tự cài dịch chuyển mọi seed theo lô bằng NumPy.
- **Bộ nhớ:** `custom brute` và `custom kdtree` tăng gần theo n² (9–12 MB ở n = 569) vì lưu khoảng cách của cả lô seed. sklearn dùng dưới 1 MB vì xử lý từng seed một. Bin seeding giảm bộ nhớ của bản tự cài xuống còn 0,8 MB.

## 4. Kết quả theo h

| Bộ (d) | α | custom brute | custom kdtree | kdtree + bin | sklearn | số cụm (không bin / bin) |
| --- | --- | --- | --- | --- | --- | --- |
| Iris (4) | 0,5 | 0,0118 s | 0,0138 s | 0,0060 s | 0,1034 s | 8 / 7 |
| Iris (4) | 1 | 0,0122 s | 0,0148 s | 0,0031 s | 0,0980 s | 2 / 2 |
| Wine (13) | 1 | 0,0337 s | 0,0456 s | 0,0073 s | 0,2356 s | 2 / 2 |
| Breast Cancer (30) | 0,5 | 0,1064 s | 0,1819 s | 0,0133 s | 0,3635 s | 150 / 2 |
| Breast Cancer (30) | 2 | 0,1079 s | 0,1892 s | 0,0052 s | 0,1986 s | 2 / 1 |

Hình: `results/e3/e3_time_vs_h.png`. Bảng đầy đủ: `results/e3/e3_summary.csv`.

- **KD-Tree so với brute:** ở Iris (4 chiều), KD-Tree gần bằng brute (0,75–0,92 lần); càng nhiều chiều thì KD-Tree càng chậm tương đối (Breast Cancer: 0,57–0,85 lần). Với dữ liệu nhỏ (≤ 569 điểm), chưa có cấu hình nào mà KD-Tree vượt brute.
- **Thời gian không đơn điệu theo h:** h nhỏ thì mỗi seed ít láng giềng và hội tụ ngay (Wine α = 0,25: 0 vòng lặp). h trung bình cần nhiều vòng lặp nhất (Breast Cancer α = 0,5: 32 vòng). h lớn thì mỗi lần tìm trả về nhiều điểm hơn nhưng số vòng lặp giảm.
- **Bin seeding càng lợi khi h lớn:** lưới thô hơn nên ít seed hơn, ví dụ Wine α = 2 chỉ còn 2 seed và nhanh hơn brute 17 lần. Ngược lại, khi h nhỏ ở 30 chiều (Breast Cancer α = 0,25), bin seeding làm **cả hai cài đặt báo lỗi**: tâm các ô lưới nằm xa mọi điểm hơn h.
- **Bin seeding có thể làm đổi kết quả:** Breast Cancer α = 0,5 ra 150 cụm khi không dùng bin, nhưng chỉ 2 cụm khi dùng bin. Vì vậy cần đọc số cụm cùng với thời gian.

## 5. Kết luận RQ3

- Trên dữ liệu thật nhỏ và nhiều chiều, **KD-Tree không giúp tăng tốc**: chậm hơn brute tới 45% (tốc độ chỉ bằng 0,55–0,97 lần brute), vì ở 13–30 chiều cây không loại bớt được điểm khi tìm láng giềng.
- **Bin seeding** là cách tăng tốc hiệu quả nhất (nhanh hơn brute tới 21 lần), nhưng **đánh đổi kết quả**: số cụm có thể thay đổi mạnh, và với h nhỏ ở nhiều chiều thì thuật toán không chạy được.
- Bản tự cài nhanh hơn sklearn 2–10 lần nhờ xử lý theo lô, đổi lại tốn bộ nhớ nhiều hơn (tăng theo n²).

## 6. Hạn chế

- n tối đa chỉ 569 nên chưa thấy được vùng n lớn, nơi KD-Tree có thể có lợi hơn nếu số chiều thấp. Mọi thời gian đều dưới 0,4 s nên dễ bị ảnh hưởng bởi tải của máy.
- Phép đo theo n chỉ dùng một bộ (Breast Cancer, d = 30), nên không tách được ảnh hưởng của n và ảnh hưởng của d.
- `tracemalloc` chỉ đo bộ nhớ do Python/NumPy cấp phát, không đo phần bộ nhớ riêng bên trong thư viện C của `cKDTree`.
