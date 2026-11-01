# Thực nghiệm E3 — Tốc độ và bộ nhớ

## 1. Mục tiêu
Trả lời RQ3: KD-Tree và bin seeding giảm chi phí tính toán bao nhiêu, và trong điều kiện nào?

## 2. Cấu hình thực nghiệm
- Dữ liệu: `make_blobs`, centers=5, cluster_std=1.0, d=2, random_state=0.
- Bandwidth tham chiếu: h0=1.5541824, ước lượng một lần trên mẫu n=2000 điểm với quantile=0.1.
- Quét n: 500, 1000, 2000, 5000, 10000, 20000, 50000; giữ h=h0.
- Quét h: n=5000; alpha=0.25, 0.5, 1.0, 2.0; h=alpha*h0.
- Số lượt đo thời gian mỗi cấu hình: 3; timeout: 600 giây cho từng lượt fit.
- Các biến thể: Custom brute; Custom KD-Tree; Custom KD-Tree + bin seeding; sklearn MeanShift (bin seeding off/on, n_jobs=1).
- **Thời gian:** `time.perf_counter()` quanh `fit(X)`, lặp N lần, không bật `tracemalloc` và không gắn wrapper thống kê; báo cáo median và IQR.
- **Bộ nhớ:** một lượt `fit(X)` riêng cho mỗi cấu hình, bật `tracemalloc` ngay trước fit và không gắn wrapper thống kê. IQR bộ nhớ không báo cáo khi chỉ có một quan sát.
- **Chẩn đoán:** một lượt fit riêng không bật `tracemalloc`; chỉ lượt này gắn wrapper để đếm truy vấn/láng giềng và số vòng lặp. Thời gian của lượt chẩn đoán không dùng trong benchmark.
- Môi trường: CPU Apple M1; RAM tổng (GB) 16.0; Python 3.14.5 (v3.14.5:5607950ef23, May 10 2026, 07:38:09) [Clang 21.0.0 (clang-2100.0.123.102)]; NumPy 2.4.6; SciPy 1.17.1; scikit-learn 1.8.0.

## 3. Kết quả tóm tắt
Bảng chi tiết nằm ở `../results/e3/e3_summary.csv`; từng phép đo nằm ở `../results/e3/e3_runs.csv` (không commit theo quy ước nhóm).

### 3.1. Thời gian, bộ nhớ và số cụm theo n
Bảng có cột **Số cụm trung vị** để kiểm tra bandwidth có đang gộp quá nhiều cụm hay không.

| n | Phương pháp | Median thời gian (s) | IQR thời gian (s) | Bộ nhớ đỉnh (MB) | Số cụm trung vị | Số seed trung vị | Láng giềng TB/truy vấn | Vòng lặp TB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 500 | Custom brute | 0.06936 | 0.001399 | 4.555 | 6 | 500 | 70.57 | 7.794 |
| 500 | Custom KD-Tree | 0.07967 | 0.00145 | 2.21 | 6 | 500 | 70.57 | 7.794 |
| 500 | Custom KD-Tree + bin seeding | 0.01067 | 0.00036 | 0.2256 | 5 | 48 | 65.33 | 8.771 |
| 500 | sklearn (bin seeding off) | 0.5219 | 0.0103 | 0.8455 | 6 | 500 | 70.49 | 7.794 |
| 500 | sklearn (bin seeding on) | 0.08485 | 0.00824 | 0.691 | 5 | 48 | 64.67 | 8.771 |
| 1000 | Custom brute | 0.2429 | 0.01012 | 15.22 | 5 | 1000 | 152 | 9.584 |
| 1000 | Custom KD-Tree | 0.2958 | 0.06676 | 9.601 | 5 | 1000 | 152 | 9.584 |
| 1000 | Custom KD-Tree + bin seeding | 0.01673 | 0.0003682 | 0.5386 | 4 | 56 | 133.2 | 9.75 |
| 1000 | sklearn (bin seeding off) | 1.232 | 0.0306 | 1.012 | 5 | 1000 | 151.9 | 9.584 |
| 1000 | sklearn (bin seeding on) | 0.1049 | 0.036 | 0.7309 | 4 | 56 | 132.4 | 9.75 |
| 2000 | Custom brute | 0.7373 | 0.05555 | 39.02 | 6 | 2000 | 293.8 | 10.85 |
| 2000 | Custom KD-Tree | 0.999 | 0.003867 | 39.86 | 6 | 2000 | 293.8 | 10.85 |
| 2000 | Custom KD-Tree + bin seeding | 0.03476 | 0.0007007 | 1.186 | 6 | 60 | 265.9 | 11.92 |
| 2000 | sklearn (bin seeding off) | 2.749 | 0.0257 | 1.388 | 6 | 2000 | 293.7 | 10.85 |
| 2000 | sklearn (bin seeding on) | 0.1538 | 0.02307 | 0.7728 | 6 | 60 | 263.9 | 11.92 |
| 5000 | Custom brute | 4.962 | 0.05851 | 165.5 | 6 | 5000 | 756.1 | 16.64 |
| 5000 | Custom KD-Tree | 8.596 | 0.1122 | 253.1 | 6 | 5000 | 756.1 | 16.64 |
| 5000 | Custom KD-Tree + bin seeding | 0.1043 | 0.02248 | 3.495 | 5 | 68 | 690.7 | 16.24 |
| 5000 | sklearn (bin seeding off) | 11.09 | 0.1458 | 2.652 | 6 | 5000 | 756 | 16.64 |
| 5000 | sklearn (bin seeding on) | 0.1922 | 0.005257 | 0.8853 | 5 | 68 | 687.8 | 16.24 |
| 10000 | Custom brute | 15.87 | 0.1792 | 562.4 | 4 | 10000 | 1484 | 14.92 |
| 10000 | Custom KD-Tree | 29.9 | 0.2972 | 1019 | 4 | 10000 | 1484 | 14.92 |
| 10000 | Custom KD-Tree + bin seeding | 0.2374 | 0.01601 | 7.498 | 5 | 73 | 1360 | 16.05 |
| 10000 | sklearn (bin seeding off) | 22.21 | 0.1524 | 4.457 | 4 | 10000 | 1484 | 14.92 |
| 10000 | sklearn (bin seeding on) | 0.2332 | 0.009931 | 1.03 | 5 | 73 | 1354 | 16.05 |
| 20000 | Custom brute | 59.52 | 3.075 | 2034 | 4 | 20000 | 2969 | 13.74 |
| 20000 | Custom KD-Tree | 112.6 | 2.916 | 4087 | 4 | 20000 | 2969 | 13.74 |
| 20000 | Custom KD-Tree + bin seeding | 0.4206 | 0.008113 | 16.64 | 5 | 81 | 2654 | 14.51 |
| 20000 | sklearn (bin seeding off) | 50.17 | 0.4618 | 8.13 | 4 | 20000 | 2969 | 13.74 |
| 20000 | sklearn (bin seeding on) | 0.2892 | 0.0123 | 1.306 | 5 | 81 | 2644 | 14.51 |
| 50000 | Custom brute | 433.9 | 5.906 | — | 5 | 50000 | — | — |
| 50000 | Custom KD-Tree | — | — | — | — | — | — | — |
| 50000 | Custom KD-Tree + bin seeding | 1.077 | 0.001581 | 43.79 | 6 | 87 | 6496 | 13.89 |
| 50000 | sklearn (bin seeding off) | 192 | 18.35 | 15.76 | 5 | 50000 | 7366 | 13.49 |
| 50000 | sklearn (bin seeding on) | 0.4863 | 0.04403 | 2.142 | 6 | 87 | 6466 | 13.89 |

### 3.2. Thời gian, bộ nhớ và số cụm theo h
| alpha | h | Phương pháp | Median thời gian (s) | IQR thời gian (s) | Bộ nhớ đỉnh (MB) | Số cụm trung vị | Số seed trung vị | Láng giềng TB/truy vấn | Vòng lặp TB |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.38855 | Custom brute | 2.063 | 0.08047 | 67.86 | 210 | 5000 | 55.14 | 10.75 |
| 0.25 | 0.38855 | Custom KD-Tree | 1.125 | 0.01241 | 18.18 | 210 | 5000 | 55.14 | 10.75 |
| 0.25 | 0.38855 | Custom KD-Tree + bin seeding | 0.1147 | 0.0006014 | 1.618 | 188 | 651 | 41.73 | 8.972 |
| 0.25 | 0.38855 | sklearn (bin seeding off) | 6.985 | 0.8186 | 2.305 | 210 | 5000 | 54.97 | 10.75 |
| 0.25 | 0.38855 | sklearn (bin seeding on) | 0.8186 | 0.01548 | 1.011 | 188 | 651 | 40.63 | 8.972 |
| 0.5 | 0.77709 | Custom brute | 4.859 | 0.1341 | 95.16 | 12 | 5000 | 236.1 | 21.57 |
| 0.5 | 0.77709 | Custom KD-Tree | 4.608 | 0.03458 | 82.57 | 12 | 5000 | 236.1 | 21.57 |
| 0.5 | 0.77709 | Custom KD-Tree + bin seeding | 0.1876 | 0.02319 | 3.36 | 11 | 204 | 215.3 | 24.8 |
| 0.5 | 0.77709 | sklearn (bin seeding off) | 13.72 | 0.3254 | 2.812 | 12 | 5000 | 236.1 | 21.57 |
| 0.5 | 0.77709 | sklearn (bin seeding on) | 0.6782 | 0.01582 | 0.9332 | 11 | 204 | 214.9 | 24.8 |
| 1 | 1.5542 | Custom brute | 5.019 | 0.2362 | 165.5 | 6 | 5000 | 756.1 | 16.64 |
| 1 | 1.5542 | Custom KD-Tree | 9.175 | 0.07121 | 253.1 | 6 | 5000 | 756.1 | 16.64 |
| 1 | 1.5542 | Custom KD-Tree + bin seeding | 0.1045 | 0.001874 | 3.496 | 5 | 68 | 690.7 | 16.24 |
| 1 | 1.5542 | sklearn (bin seeding off) | 11.79 | 0.6048 | 2.654 | 6 | 5000 | 756 | 16.64 |
| 1 | 1.5542 | sklearn (bin seeding on) | 0.2047 | 0.02354 | 0.8833 | 5 | 68 | 687.8 | 16.24 |
| 2 | 3.1084 | Custom brute | 7.743 | 0.3355 | 319.7 | 2 | 5000 | 2132 | 14.85 |
| 2 | 3.1084 | Custom KD-Tree | 22.32 | 0.3669 | 605.5 | 2 | 5000 | 2132 | 14.85 |
| 2 | 3.1084 | Custom KD-Tree + bin seeding | 0.0844 | 0.005278 | 2.555 | 2 | 23 | 1989 | 14.3 |
| 2 | 3.1084 | sklearn (bin seeding off) | 13.36 | 0.8006 | 2.945 | 2 | 5000 | 2132 | 14.85 |
| 2 | 3.1084 | sklearn (bin seeding on) | 0.1259 | 0.01001 | 0.8144 | 2 | 23 | 1978 | 14.3 |

## 4. Trả lời định lượng các câu hỏi RQ3

### 4.1. KD-Tree bắt đầu nhanh hơn brute ở n nào?
Tốc độ tương đối được tính bằng `median_time(brute) / median_time(KD-Tree)`. Giá trị > 1 nghĩa là KD-Tree nhanh hơn; giá trị < 1 nghĩa là KD-Tree chậm hơn trong cấu hình và môi trường đã đo.

| n | Brute median (s) | KD-Tree median (s) | Brute/KD-Tree (×) | Láng giềng TB/truy vấn của KD-Tree |
|---:|---:|---:|---:|---:|
| 500 | 0.069357 | 0.07967 | 0.871 | 70.57 |
| 1000 | 0.24285 | 0.2958 | 0.821 | 152 |
| 2000 | 0.73728 | 0.99902 | 0.738 | 293.8 |
| 5000 | 4.9625 | 8.5961 | 0.577 | 756.1 |
| 10000 | 15.872 | 29.899 | 0.531 | 1484 |
| 20000 | 59.519 | 112.59 | 0.529 | 2969 |
| 50000 | 433.88 | timeout (> 600) | < 0.723 | — |

**Kết quả:** chưa ghi nhận KD-Tree nhanh hơn brute ở bất kỳ mức n nào có đủ số liệu so sánh (6 mức hợp lệ). Tại n lớn nhất có thể so sánh (n=20,000), brute mất 59.519 s, KD-Tree mất 112.59 s; tỷ số brute/KD-Tree = 0.529×.
Ở n=50000, brute vẫn chạy xong (433.88 s) còn KD-Tree vượt timeout 600 s, nên tỷ số chắc chắn nhỏ hơn 0.723. Phân tích nguyên nhân ở mục 5.1.

**Độ dốc log–log thực nghiệm** (hệ số góc của log10(thời gian median) theo log10(n), chỉ tính tại các điểm đo hợp lệ):
- Custom brute: độ dốc = **1.884**, dựa trên 7 mức n.
- Custom KD-Tree: độ dốc = **1.993**, dựa trên 6 mức n.
- Custom KD-Tree + bin seeding: độ dốc = **1.038**, dựa trên 7 mức n.
- sklearn (bin seeding off): độ dốc = **1.273**, dựa trên 7 mức n.
- sklearn (bin seeding on): độ dốc = **0.359**, dựa trên 7 mức n.
Độ dốc là mô tả thực nghiệm trong khoảng n đã đo; không nên đồng nhất trực tiếp với độ phức tạp tiệm cận lý thuyết.

### 4.2. Khi h tăng, số láng giềng và thời gian thay đổi thế nào?
Bảng sau so sánh mức alpha nhỏ nhất và lớn nhất có số liệu hợp lệ của từng phương pháp. Phần trăm thay đổi được tính từ điểm đầu đến điểm cuối; phần nhận xét về xu hướng dùng tất cả điểm alpha có dữ liệu.

| Phương pháp | alpha đầu → cuối | Láng giềng TB/truy vấn đầu → cuối | Thay đổi láng giềng | Thời gian median đầu → cuối (s) | Thay đổi thời gian | Xu hướng thời gian qua các alpha |
|---|---:|---:|---:|---:|---:|---|
| Custom brute | 0.25 → 2 | 55.14 → 2132 | +3766.4% | 2.0625 → 7.7427 | +275.4% | tăng hoặc giữ nguyên qua các mức đã đo |
| Custom KD-Tree | 0.25 → 2 | 55.14 → 2132 | +3766.4% | 1.1252 → 22.323 | +1883.9% | tăng hoặc giữ nguyên qua các mức đã đo |
| Custom KD-Tree + bin seeding | 0.25 → 2 | 41.73 → 1989 | +4667.1% | 0.1147 → 0.084398 | -26.4% | biến động không đơn điệu qua các mức đã đo |
| sklearn (bin seeding off) | 0.25 → 2 | 54.97 → 2132 | +3778.3% | 6.9851 → 13.364 | +91.3% | biến động không đơn điệu qua các mức đã đo |
| sklearn (bin seeding on) | 0.25 → 2 | 40.63 → 1978 | +4768.8% | 0.81863 → 0.12593 | -84.6% | giảm hoặc giữ nguyên qua các mức đã đo |

**Diễn giải KD-Tree:** số láng giềng trung bình tăng hoặc giữ nguyên qua các mức đã đo theo các alpha đã đo; từ alpha=0.25 đến alpha=2, giá trị đổi từ 55.14 sang 2132 (+3766.4%).
Thời gian median của KD-Tree tăng hoặc giữ nguyên qua các mức đã đo theo alpha; từ alpha=0.25 đến alpha=2, thời gian đổi từ 1.1252 s sang 22.323 s (+1883.9%).
Phân tích chi tiết ở mục 5.2.

### 4.3. Bin seeding giảm bao nhiêu seed và thời gian?
So sánh `custom_kdtree_bin` với `custom_kdtree` tại cùng n và h. Phần trăm tiết kiệm seed = (seed không bin − seed có bin) / seed không bin; phần trăm tiết kiệm thời gian được tính tương tự. Số âm nghĩa là cấu hình có bin seeding tốn nhiều tài nguyên/thời gian hơn ở phép đo đó.

| n | Seed không bin | Seed có bin | Seed giảm | Seed giảm (%) | Số cụm không bin | Số cụm có bin | Thời gian không bin (s) | Thời gian có bin (s) | Thời gian tiết kiệm (s) | Thời gian tiết kiệm (%) | Speedup có bin (×) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 500 | 48 | 452 | +90.4% | 6 | 5 | 0.07967 | 0.010672 | +0.068998 | +86.6% | 7.465 |
| 1000 | 1000 | 56 | 944 | +94.4% | 5 | 4 | 0.2958 | 0.016725 | +0.27908 | +94.3% | 17.686 |
| 2000 | 2000 | 60 | 1940 | +97.0% | 6 | 6 | 0.99902 | 0.034764 | +0.96426 | +96.5% | 28.737 |
| 5000 | 5000 | 68 | 4932 | +98.6% | 6 | 5 | 8.5961 | 0.1043 | +8.4918 | +98.8% | 82.414 |
| 10000 | 10000 | 73 | 9927 | +99.3% | 4 | 5 | 29.899 | 0.23744 | +29.661 | +99.2% | 125.920 |
| 20000 | 20000 | 81 | 19919 | +99.6% | 4 | 5 | 112.59 | 0.42064 | +112.17 | +99.6% | 267.659 |
| 50000 | — | 87 | — | — | — | 6 | — | 1.077 | — | — | — |

Mức giảm seed lớn nhất theo tỷ lệ là +99.6% tại n=20,000 (từ 20000 xuống 81 seed).
Mức tiết kiệm thời gian lớn nhất theo tỷ lệ là +99.6% tại n=20,000 (+112.17 s; speedup=267.659×). Số âm nghĩa là bin seeding chậm hơn KD-Tree không bin ở cấu hình đó.
Cần đọc đồng thời cột số cụm: giảm seed/tăng tốc không tự động có nghĩa là kết quả phân cụm giữ nguyên. Bảng E3 ghi số cụm để đối chiếu; đánh giá sai khác chất lượng phân cụm chi tiết thuộc các thực nghiệm phù hợp như E1/E6.

## 5. Nhận xét

### 5.1. KD-Tree không nhanh hơn brute khi h cố định
- Với h = h0, KD-Tree chậm hơn brute ở **mọi** mức n đã đo, và khoảng cách nới rộng dần khi n tăng: tỷ số brute/KD-Tree giảm từ 0.87 (n=500) xuống 0.53 (n=10000 và n=20000). Ở n=50000, brute chạy xong trong 434 s, còn KD-Tree vượt timeout. Như vậy trong thiết lập này **không có ngưỡng n** nào mà từ đó KD-Tree bắt đầu nhanh hơn.
- Nguyên nhân là số láng giềng. Khi giữ h cố định và tăng n, dữ liệu dày lên nên mỗi truy vấn luôn trả về khoảng 14–15% số điểm (70.6 láng giềng ở n=500, 2969 ở n=20000). Một tỷ lệ cố định của dữ liệu nằm trong bán kính nên cây không loại bỏ được phần lớn không gian tìm kiếm. Chi phí khi đó chủ yếu nằm ở việc trả về và xử lý láng giềng, phần này tăng theo n ở cả hai cách. Brute tính khoảng cách theo khối bằng NumPy (vector hoá). KD-Tree nhận kết quả của `cKDTree.query_ball_point` dưới dạng list Python cho từng seed rồi mới chuyển thành mảng, tức tốn thêm một bước chậm.
- Số đo bộ nhớ cũng ủng hộ giải thích này: ở n lớn, KD-Tree tốn khoảng 1.8–2.0 lần bộ nhớ của brute (1019 MB so với 562 MB ở n=10000; 4087 MB so với 2034 MB ở n=20000), do phải giữ các list láng giềng trung gian.
- KD-Tree chỉ có lợi khi lân cận nhỏ. Kết quả quét h tại n=5000:

| alpha | Láng giềng TB/truy vấn | Brute (s) | KD-Tree (s) | Brute/KD-Tree (×) |
|---:|---:|---:|---:|---:|
| 0.25 | 55 | 2.06 | 1.13 | 1.83 |
| 0.5 | 236 | 4.86 | 4.61 | 1.05 |
| 1 | 756 | 5.02 | 9.17 | 0.55 |
| 2 | 2132 | 7.74 | 22.32 | 0.35 |

  Điều kiện để KD-Tree có lợi là **số láng giềng mỗi truy vấn nhỏ so với n** (khoảng 1% dữ liệu ở α=0.25), chứ không phải n lớn. Tuy vậy, ở α=0.25 kết quả bị vỡ thành 210 cụm, tức vùng h mà KD-Tree có lợi lại là vùng h quá nhỏ đối với bộ dữ liệu này.

### 5.2. Ảnh hưởng của h
- Từ α=0.25 đến α=2, số láng giềng tăng khoảng 39 lần (55 → 2132). Thời gian của brute chỉ tăng 3.75 lần (2.06 → 7.74 s), vì phần tính khoảng cách tới mọi điểm không phụ thuộc h; chỉ phần xử lý láng giềng tăng. Thời gian của KD-Tree tăng 19.8 lần (1.13 → 22.3 s), gần với mức tăng láng giềng hơn, vì chi phí của nó gắn trực tiếp với số điểm trả về.
- Số vòng lặp trung bình không đơn điệu theo h (10.7; 21.6; 16.6; 14.9), nên thời gian không chỉ phụ thuộc số láng giềng mà còn phụ thuộc số bước cần để hội tụ.
- Với bin seeding, h lớn lại chạy nhanh hơn: lưới có cạnh h thô hơn nên số seed giảm từ 651 xuống 23, bù cho số láng giềng tăng. Thời gian giảm từ 0.115 s xuống 0.084 s, trừ α=0.5 (0.188 s) là mức có số vòng lặp trung bình cao nhất (24.8).

### 5.3. Bin seeding là cải tiến hiệu quả nhất
- Số seed gần như không đổi theo n (48 seed ở n=500, 87 seed ở n=50000), vì nó phụ thuộc số ô lưới có dữ liệu, tức phụ thuộc vùng dữ liệu chiếm và h, không phụ thuộc n. Chi phí khi đó chỉ còn tỷ lệ với số láng giềng, tức tỷ lệ với n: độ dốc log–log của KD-Tree + bin seeding là 1.04, so với 1.88 của brute và 1.99 của KD-Tree.
- Mức tăng tốc so với KD-Tree không bin tăng dần theo n: 7.5× ở n=500, 82× ở n=5000, 268× ở n=20000. Ở n=50000, KD-Tree không bin vượt timeout, còn có bin seeding chỉ mất 1.08 s. Bộ nhớ cũng giảm theo cùng tỷ lệ (16.6 MB so với 4087 MB ở n=20000).
- Đánh đổi là số cụm có thể khác: số cụm có bin và không bin khác nhau ở 5/6 mức n so sánh được (ví dụ n=10000: 4 cụm không bin, 5 cụm có bin). Dữ liệu được sinh với 5 tâm và cách có bin ra 5 cụm ở phần lớn các mức n, nên ở đây bin seeding không làm kết quả xấu đi. Dù vậy, kết quả có phụ thuộc cách chọn seed; đánh giá chất lượng phân cụm chi tiết thuộc E1/E6.

### 5.4. So sánh với sklearn
- **Không bin seeding:** custom brute nhanh hơn sklearn ở n ≤ 10000 (7.5× ở n=500, 2.2× ở n=5000, 1.4× ở n=10000), nhưng chậm hơn từ n=20000 (59.5 s so với 50.2 s) và chậm hơn rõ ở n=50000 (434 s so với 192 s). Độ dốc của sklearn thấp hơn (1.27 so với 1.88). sklearn dịch chuyển từng seed một, mỗi vòng lặp gọi một truy vấn `radius_neighbors`, nên tốn chi phí cố định cho mỗi seed ở n nhỏ. Custom dịch chuyển mọi seed còn active trong cùng một lượt nên nhanh ở n vừa phải, nhưng phải giữ danh sách láng giềng của tất cả seed cùng lúc, nên chi phí tăng nhanh hơn khi n lớn.
- **Bộ nhớ:** cũng vì cách xử lý theo lô ở trên mà custom tốn bộ nhớ nhiều hơn sklearn hai bậc độ lớn: 2034 MB (custom brute, n=20000) so với 8.1 MB (sklearn, n=20000). Đây là đánh đổi bộ nhớ lấy tốc độ.
- **Có bin seeding:** custom nhanh hơn ở n nhỏ (8× ở n=500, 1.8× ở n=5000), ngang nhau ở n=10000 (0.237 s so với 0.233 s), và sklearn nhanh hơn từ n=20000 (2.2× ở n=50000).
- Số cụm và số vòng lặp trung bình của custom khớp sklearn ở mọi cấu hình, xác nhận hai bản cài đặt cho cùng kết quả; khác biệt chỉ nằm ở chi phí.

### 5.5. Trả lời RQ3
- **KD-Tree:** không giảm chi phí khi h cố định và n tăng, mà còn chậm hơn brute 1.1–1.9 lần và tốn bộ nhớ gấp khoảng 2 lần. KD-Tree chỉ có lợi khi h nhỏ, số láng giềng mỗi truy vấn chỉ khoảng 1% dữ liệu (nhanh hơn 1.83× ở α=0.25).
- **Bin seeding:** giảm 90–99.6% số seed và 87–99.6% thời gian, tăng tốc 7.5–268× tuỳ n, đưa độ dốc log–log về khoảng 1. Đây là yếu tố quyết định để chạy được n lớn.
- Các số liệu trên đo trên Apple M1 và chỉ có giá trị so sánh tương đối giữa các phương pháp trên cùng máy.

## 6. Biểu đồ
- `../results/e3/e3_time_vs_n.png` — thời gian theo n, trục log–log.
- `../results/e3/e3_memory_vs_n.png` — bộ nhớ đỉnh theo n.
- `../results/e3/e3_time_vs_h.png` — thời gian theo h.

## 7. Giới hạn và lưu ý diễn giải
- Lần chạy đầu trên máy Windows (Intel thế hệ 13) cho thời gian dao động tới 4.5 lần giữa các lần đo cùng một cấu hình, nên toàn bộ số liệu trong file này lấy từ lần chạy lại trên Apple M1. Tại đây, cùng cấu hình n=5000, h=h0 xuất hiện ở cả hai lượt quét và lệch nhau không quá 7%.
- Ở n=50000, lượt đo bộ nhớ của brute vượt timeout (tracemalloc làm chậm fit), nên ô bộ nhớ, láng giềng và vòng lặp của dòng này để trống; thời gian vẫn hợp lệ.
- `tracemalloc` ghi nhận bộ nhớ Python-traced; có thể không bao quát đầy đủ bộ nhớ native do NumPy, SciPy, KD-Tree hoặc sklearn cấp phát. Không diễn giải số đo này như RSS tổng của tiến trình.
- Thời gian báo cáo chỉ lấy từ các lượt fit không bật tracemalloc và không gắn wrapper. Lượt đo bộ nhớ và lượt chẩn đoán là các lần fit riêng; thời gian của chúng không tham gia tính median thời gian.
- Bộ nhớ được đo một lần riêng cho mỗi cấu hình. Do chỉ có một quan sát, median được báo cáo nhưng Q1/Q3/IQR bộ nhớ để trống.
- Các lượt timeout/skip được ghi trong CSV; chúng không được dùng để tính median thời gian thành công.
- Số láng giềng và vòng lặp lấy từ lượt chẩn đoán. Nếu API thư viện thay đổi khiến không thu được số liệu, để trống/NA thay vì tự suy ra.
- Số cụm được lấy từ `len(cluster_centers_)` sau mỗi lượt fit; cần đọc cột này cùng bandwidth h để xác định liệu các cụm có bị gộp quá mức hay không.

## 8. File đầu ra
- `../results/e3/e3_runs.csv` — dữ liệu từng phép đo, có cột `measurement` (`time`, `memory`, `diagnostics`).
- `../results/e3/e3_summary.csv` — median/Q1/Q3/IQR cho thời gian, peak memory khi có đủ quan sát, speedup so với custom brute và số cụm trung vị.
- `../results/e3/env.txt` và `../results/e3/e3_metadata.json` — môi trường và cấu hình chạy.
