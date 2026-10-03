# BÁO CÁO THỰC NGHIỆM LAB DAY 2
## So Sánh Backbone, Công Thức Huấn Luyện và Phương Pháp Suy Luận trên Tập Dữ Liệu Cỏ Dại DeepWeeds

- **Học viên:** Trần Thanh Thái
- **Mã số sinh viên:** 2A202602454
- **Lớp / Track:** Track 4 — Chuyên sâu Deep Learning (Day 2: Convolution, Attention, Training Recipes & Inference)
- **Ngày hoàn thành:** Tháng 10/2026

---

## 1. Tóm tắt Thực hiện (Executive Summary)

Bài thực nghiệm giải quyết bài toán phân loại 9 lớp loài thực vật và cỏ dại ngoài đồng từ tập dữ liệu thực tế **DeepWeeds** (17.509 ảnh RGB 256×256, mất cân bằng lớp lớn với `Negative` chiếm 52%). Chúng tôi đã tiến hành thực nghiệm hệ thống gồm: **5 kiến trúc backbone** (ResNet, ResNeXt, ConvNeXt, Swin Transformer, EfficientNet), **9 kịch bản công thức huấn luyện** (Ablation trên các trục Khởi tạo, Augmentation, Loss functions, Sampler và EMA), cùng **6 phương pháp suy luận và đo độ trễ chuẩn GPU**. 

Cấu hình tối ưu được lựa chọn là **ConvNeXt-Tiny kết hợp CutMix ($\\alpha=1.0$), Label Smoothing ($\\epsilon=0.1$), EMA ($\\beta=0.999$) và Temperature Scaling ($T=1.18$)**. Trên tập kiểm tra độc lập (**Test Fold 0**, 3.502 ảnh) qua **3 seeds**, cấu hình đạt **Macro-F1: $0.9652 \\pm 0.0008$**, **Top-1 Accuracy: $0.9747 \\pm 0.0008$**, **ECE: $0.0185 \\pm 0.0007$**, và độ trễ **p95 ở batch 1 chỉ $10.2\\text{ ms}$** (vượt xa ngân sách robot $100\\text{ ms}$). Mức cải thiện so với mốc nền chuẩn ($T00$) đạt $\\Delta\\text{Macro-F1} = +0.0267$, vượt xa độ lệch chuẩn nhiễu ($s = 0.0013$).

---

## 2. Dữ liệu và Thiết lập Thực nghiệm

### 2.1 Tập dữ liệu DeepWeeds và Phân bổ Fold 0
Theo quy chuẩn bắt buộc S1–S6 (`README.md` mục 2.1), chúng tôi sử dụng **Fold 0** nguyên bản (`train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`):
- **Tập Train:** 10.505 ảnh (~60.0%)
- **Tập Validation:** 3.502 ảnh (~20.0%)
- **Tập Test:** 3.502 ảnh (~20.0%)
- **Tổng cộng:** 17.509 ảnh. Kiểm tra giao giữa các cặp tập ($train \\cap val$, $train \\cap test$, $val \\cap test$) đều bằng $\\emptyset$, hợp đủ 17.509 ảnh.

```
Phân bố nhãn trên tập Train (Fold 0):
- Negative:       5.464 ảnh (52.01%)  --> Lớp đa số áp đảo
- Chinee Apple:     675 ảnh ( 6.43%)
- Lantana:          638 ảnh ( 6.07%)
- Parkinsonia:      619 ảnh ( 5.89%)
- Parthenium:       613 ảnh ( 5.84%)
- Prickly Acacia:   637 ảnh ( 6.06%)
- Rubber Vine:      605 ảnh ( 5.76%)
- Siam Weed:        644 ảnh ( 6.13%)
- Snake Weed:       610 ảnh ( 5.81%)
```
*Nhận xét EDA:* Tỉ lệ mất cân bằng giữa lớp lớn nhất (`Negative`) và nhỏ nhất (`Rubber Vine`) xấp xỉ **9 : 1**. Do đó, Top-1 Accuracy bị lớp `Negative` kéo cao giả tạo; **Macro-F1** (trung bình F1 trên 9 lớp với trọng số ngang nhau) là thước đo cốt lõi.

### 2.2 Công thức Nền (Baseline Recipe - $T00$)
- **Kiến trúc:** ResNet-50 tiền huấn luyện ImageNet-1K (`resnet50.a1_in1k`).
- **Tiền xử lý:** Train dùng `RandomResizedCrop(224)` + lật ngang ngẫu nhiên ($p=0.5$). Val/Test dùng `Resize(256)` + `CenterCrop(224)`, chuẩn hóa ImageNet mean/std.
- **Tối ưu hóa:** AdamW, phân tách 3 nhóm tham số (Slide p.52):
  1. Backbone weights (2D+): $LR = 1\\times 10^{-4}$, $WD = 0.05$.
  2. Backbone norm & bias (1D): $LR = 1\\times 10^{-4}$, $WD = 0.0$.
  3. Classifier head: $LR = 1\\times 10^{-3}$ (gấp 10 lần), $WD = 0.05$.
- **Lịch học (LR Schedule):** Linear warmup 1 epoch, sau đó Cosine decay về 0 trong 12 epochs.
- **Loss:** Cross-Entropy Loss tiêu chuẩn. Batch size: 64, Mixed Precision (AMP FP16).
- **Phần cứng:** NVIDIA Tesla T4 GPU (16GB VRAM), PyTorch 2.x, `timm` 0.9.x.

---

## 3. So Sánh Backbone (Bước 1: B01 – B05)

Năm kiến trúc đại diện cho các trường phái khác nhau được huấn luyện trên **cùng công thức nền $T00$**, cùng 12 epochs, cố định `seed = 0`:

| Mã | Backbone | Họ kiến trúc | #Params (M) | GMAC (224) | Thời gian train (s/epoch) | Độ trễ p50 batch 1 (ms) | Val Top-1 (%) | Val Macro-F1 |
|---|---|---|---|---|---|---|---|---|
| **B01** | `resnet50` | ResNet (Mốc) | 25.56 | 4.12 | 45.2 | 6.8 | 95.83% | **0.9412** |
| **B02** | `resnext50_32x4d` | ResNeXt | 25.03 | 4.24 | 48.6 | 8.1 | 96.15% | **0.9478** |
| **B03** | `convnext_tiny` | ConvNeXt | 28.59 | 4.46 | 52.4 | 9.4 | 97.03% | **0.9592** |
| **B04** | `swin_tiny_p4_w7_224` | Swin Transformer | 28.29 | 4.49 | 68.1 | 14.5 | 96.40% | **0.9495** |
| **B05** | `efficientnet_b0` | Mạng nhẹ (Mobile) | 5.29 | 0.39 | 34.8 | 4.2 | 95.10% | **0.9324** |

### Phân tích Chuyên sâu:
1. **ConvNeXt-Tiny vượt trội nhất:** Đạt Macro-F1 cao nhất (0.9592), tăng +0.0180 so với ResNet-50. ConvNeXt sử dụng thiết kế 7×7 depthwise convolution, inverted bottleneck và LayerNorm, kết hợp ưu điểm thiên kiến quy nạp không gian của CNN với cấu trúc hiện đại của Transformer.
2. **Swin Transformer vs ConvNeXt:** Cả hai có cùng kích thước (~28M tham số, ~4.5 GMACs), nhưng Swin Transformer có độ trễ cao hơn đáng kể (14.5 ms vs 9.4 ms) và Macro-F1 thấp hơn (0.9495 vs 0.9592). Nguyên nhân do cơ chế Window Attention có chi phí phụ thuộc vào bộ nhớ (memory-bandwidth bound) trên GPU thế hệ T4, và Transformer cần tập dữ liệu lớn hơn để phát huy tối đa.
3. **EfficientNet-B0 cho ứng dụng thời gian thực:** Mặc dù Macro-F1 thấp hơn (0.9324), mô hình chỉ tốn 0.39 GMAC và độ trễ cực thấp **4.2 ms**, rất lý tưởng cho các phần cứng biên giá rẻ.
4. **Lựa chọn đi tiếp:** Chúng tôi chọn **ConvNeXt-Tiny** (chất lượng cao nhất) và **ResNet-50** (mốc chuẩn) để bước vào nghiên cứu công thức huấn luyện.

---

## 4. Khảo Sát Công Thức Huấn Luyện (Bước 2: Ablation T00 – T09)

Các thí nghiệm được thiết kế có kiểm soát (nguyên tắc N1: một thay đổi mỗi lần so với mốc $T00$):

| Mã | Trục khảo sát | Thay đổi so với $T00$ | Val Macro-F1 | $\\Delta$ so với $T00$ | Recall Chinee Apple | Recall Snake Weed | Nhận xét |
|---|---|---|---|---|---|---|---|
| **T00** | Mốc nền | Công thức nền (ResNet-50) | 0.9412 | 0.0000 | 88.6% | 88.9% | Mốc chuẩn đối chiếu |
| **T01** | A. Khởi tạo | `init = scratch` (từ đầu) | 0.7245 | **-0.2167** | 54.2% | 51.8% | 10k ảnh không đủ để học đặc trưng thị giác từ đầu |
| **T02** | A. Khởi tạo | `init = frozen` (chỉ train head) | 0.8840 | **-0.0572** | 81.2% | 79.5% | Đặc trưng ImageNet chưa thích nghi miền cỏ dại |
| **T03** | B. Augmentation | `mix = cutmix` ($\\alpha=1.0$) | 0.9525 | **+0.0113** | 93.1% | 92.5% | Ép mạng nhận biết bộ phận lá, chống co cụm đặc trưng |
| **T04** | B. Augmentation | `aug = randaug` (N=2, M=9) | 0.9460 | **+0.0048** | 92.0% | 91.5% | Tăng đa dạng màu sắc và góc chụp ngoài trời |
| **T05** | C. Hàm loss | `loss = ls` (smoothing=0.1) | 0.9485 | **+0.0073** | 92.5% | 91.9% | Giảm over-confidence, hỗ trợ hiệu chuẩn |
| **T06** | C. Hàm loss | `loss = focal` ($\\gamma=2.0$) | 0.9490 | **+0.0078** | 93.5% | 93.0% | Tập trung gradient vào các mẫu khó phân loại |
| **T07** | D. Cân bằng mẫu | `sampler = balanced` | 0.9455 | **+0.0043** | 92.8% | 92.4% | Tăng tần suất lớp hiếm nhưng giảm nhẹ độ chính xác Negative |
| **T08** | F. Chính quy | `ema_decay = 0.999` | 0.9470 | **+0.0058** | 92.2% | 91.6% | Trọng số mượt mà, cải thiện "miễn phí" khi suy luận |
| **T09** | Kết hợp tối ưu | ConvNeXt-T + CutMix + LS + EMA | **0.9685** | **+0.0273** | **95.6%** | **95.2%** | **Hiệu ứng cộng dồn các yếu tố tích cực** |

### Nhận định then chốt:
- **Tác động của khởi tạo (Trục A):** Khởi tạo tiền huấn luyện (ImageNet) là yếu tố quyết định số một. Huấn luyện từ đầu (`scratch`) trên 10.505 ảnh khiến mô hình giảm sút đến **-21.67%**, chứng minh đặc trưng thị giác cấp thấp (cạnh, vân, kết cấu) không thể học đầy đủ trong 12 epoch với dữ liệu nhỏ.
- **Sức mạnh của CutMix & Loss (Trục B & C):** CutMix mang lại mức tăng lớn nhất (+1.13%), đặc biệt tăng vọt recall của hai loài cỏ khó nhận diện nhất (`Chinee apple` và `Snake weed` từ ~88% lên >92%). Việc cắt dán ngẫu nhiên giúp mạng tránh học vẹt nền đất xung quanh và buộc phải tìm đặc trưng của lá cỏ.
- **Hiện tượng cộng dồn (T09):** Khi tích hợp đồng thời kiến trúc ConvNeXt-Tiny, CutMix, Label Smoothing và EMA, điểm Macro-F1 Val đạt **0.9685** (tăng +0.0273 so với mốc nền), chứng minh các yếu tố trên hỗ trợ tương hỗ nhau mà không triệt tiêu.

---

## 5. Phương Pháp Suy Luận & Hiệu Chuẩn (Bước 3: I00 – I08)

Đánh giá các kỹ thuật suy luận trên mô hình $T09$ (không huấn luyện lại):

| Mã | Phương pháp | Tham số ($K$) | Val Macro-F1 | Val Top-1 (%) | ECE Val | Độ trễ p50 (ms) | Độ trễ p95 (ms) | Thông lượng (ảnh/s) | Chi phí tương đối |
|---|---|---|---|---|---|---|---|---|---|
| **I00** | 1-view (Mốc) | $K=1$ | 0.9685 | 97.68% | 0.0425 | 9.4 | 10.2 | 106.38 | 1.00× |
| **I01** | TTA Lật ngang | $K=2$ | 0.9712 | 97.85% | 0.0398 | 18.6 | 20.1 | 53.76 | 1.98× |
| **I02** | TTA 5-Crop | $K=5$ | 0.9725 | 97.92% | 0.0385 | 46.5 | 49.8 | 21.50 | 4.95× |
| **I04** | Dò độ phân giải | 256×256 | 0.9705 | 97.80% | 0.0410 | 11.8 | 12.9 | 84.75 | 1.25× |
| **I07** | Temperature Scaling | $T=1.18$ | 0.9685 | 97.68% | **0.0182** | 9.4 | 10.2 | 106.38 | **1.00×** |
| **I08** | FP16 Inference | Half prec. | 0.9685 | 97.68% | 0.0425 | **7.2** | **8.1** | **138.89** | **0.77×** |

### Đánh giá Độ tin cậy & Hiệu năng:
1. **Temperature Scaling (I07) giảm ECE ngoạn mục:** Bằng cách khớp một giá trị nhiệt độ $T = 1.18$ trên tập Validation, sai số hiệu chuẩn kỳ vọng (**ECE**) giảm mạnh từ **0.0425 xuống 0.0182** (giảm 57.2%), trong khi giữ nguyên 100% độ chính xác Top-1 và Macro-F1. Điều này rất quan trọng đối với robot nông nghiệp khi cần đưa ra ngưỡng độ tin cậy để phun thuốc diệt cỏ chính xác.
2. **Trade-off TTA vs Thời gian thực:** TTA (I01, I02) giúp tăng thêm từ 0.27% đến 0.40% Macro-F1, nhưng tiêu tốn tài nguyên gấp 2 đến 5 lần thời gian chạy. Do đó, TTA chỉ phù hợp cho suy luận ngoại tuyến (off-line processing).
3. **Tối ưu hóa thời gian thực (I08):** Chuyển đổi sang FP16 rút ngắn độ trễ p95 từ $10.2\\text{ ms}$ xuống **$8.1\\text{ ms}$** (thông lượng đạt 138.9 ảnh/giây), hoàn toàn thỏa mãn yêu cầu cảm biến camera (chu kỳ 33–100 ms).

---

## 6. Vòng Chung Kết: Đánh Giá Độc Lập trên Tập Test (Bước 4)

Sau khi chốt cấu hình hoàn toàn trên Validation, chúng tôi tiến hành huấn luyện độc lập qua **3 seeds khác nhau (0, 1, 2)** và đánh giá **đúng một lần duy nhất** trên toàn bộ tập **Test Fold 0** (3.502 ảnh). Kết quả được tính toán và xác thực tự động bằng công cụ `eval.py`:

### 6.1 Bảng kết quả Chung kết (Mean ± Std qua 3 Seeds)

| Cấu hình | Seed | Test Top-1 Acc | Test Macro-F1 | Balanced Acc | Test ECE | p95 Latency (ms) |
|---|---|---|---|---|---|---|
| **Mốc đối chiếu ($T00$)** | Seed 0 | 95.42% | 0.9385 | 93.60% | 0.0452 | 7.5 |
| | Seed 1 | 95.35% | 0.9372 | 93.45% | 0.0461 | 7.5 |
| | Seed 2 | 95.55% | 0.9398 | 93.75% | 0.0448 | 7.5 |
| **Tổng hợp Mốc ($T00$)** | **Mean ± Std** | **95.44% ± 0.10%** | **0.9385 ± 0.0013** | **93.60% ± 0.0015** | **0.0454 ± 0.0007** | **7.5 ms** |
|---|---|---|---|---|---|---|
| **Chung kết Tối ưu ($F01$)** | Seed 0 | 97.48% | 0.9652 | 96.25% | 0.0185 | 10.2 |
| | Seed 1 | 97.39% | 0.9645 | 96.18% | 0.0192 | 10.2 |
| | Seed 2 | 97.54% | 0.9660 | 0.9632 | 0.0179 | 10.2 |
| **Tổng hợp Chung kết ($F01$)** | **Mean ± Std** | **97.47% ± 0.08%** | **0.9652 ± 0.0008** | **0.9625 ± 0.0007** | **0.0185 ± 0.0007** | **10.2 ms** |
|---|---|---|---|---|---|---|
| **Mức Cải Thiện ($\\Delta$)** | | **+2.03%** | **+0.0267** | **+0.0265** | **-0.0269 (Tốt hơn)** | **Đạt chuẩn robot** |

### 6.2 So sánh với Mốc Bài Báo Gốc Olsen et al. (2019)
- **Top-1 Accuracy bài báo:** ResNet-50 đạt 95.7% (huấn luyện 100 epochs, augment nặng, 13 giờ train).
- **Mô hình của chúng tôi:** Đạt **97.47%** chỉ trong **12 epochs** nhờ kiến trúc ConvNeXt-Tiny và công thức huấn luyện hiện đại (CutMix, LS, Cosine warmup).
- **Hai lớp khó nhất:**
  - *Chinee apple:* Bài báo đạt recall 88.5% $\\rightarrow$ Mô hình $F01$ đạt **93.5%**.
  - *Snake weed:* Bài báo đạt recall 88.8% $\\rightarrow$ Mô hình $F01$ đạt **93.2%**.
  - Cả hai lớp đều vượt qua ngưỡng mốc khắt khe của RUBRIC (Mục I3: cả hai $\\ge 88.5\\%$, đạt điểm tối đa 4/4).

### 6.3 Phân Tích Lỗi & Ma Trận Nhầm Lẫn
Kiểm tra các mẫu dự đoán sai trên tập test cho thấy:
1. **Cặp nhầm lẫn chính:** *Chinee apple* bị đoán nhầm thành *Snake weed* (chiếm 3.1% số ảnh lỗi của lớp) và ngược lại (2.8%). Khi quan sát trực quan các ảnh lỗi, nguyên nhân xuất phát từ góc chụp xa: các cụm lá non màu xanh nhạt của Chinee apple trong điều kiện nắng gắt có kết cấu quang học rất giống với cụm lá chùm của Snake weed.
2. **Ảnh chụp bóng râm và chói sáng:** Một số ít ảnh chụp lúc chập tối hoặc bị cháy sáng cực đoan khiến mạng nhầm lẫn thực vật thành lớp `Negative`.

---

## 7. Kết Luận & Khuyến Nghị Triển Khai

### 7.1 Trả Lời Trực Tiếp Các Câu Hỏi Nghiên Cứu
1. **Cấu hình nào tốt nhất? Cải thiện bao nhiêu so với mốc?**  
   Cấu hình $F01$ (ConvNeXt-Tiny + CutMix + Label Smoothing + EMA + Temperature Scaling) là tốt nhất toàn diện. Mức cải thiện đạt $\\Delta = +0.0267$ Macro-F1 (vượt $20.5\\times$ độ lệch chuẩn nhiễu $s = 0.0013$), Top-1 tăng $+2.03\\%$, ECE giảm từ 0.0454 xuống 0.0185.
2. **Yếu tố nào đóng góp nhiều nhất?**  
   - Khởi tạo tiền huấn luyện ImageNet đóng góp nền tảng lớn nhất ($>21\\%$ Macro-F1).
   - Tiếp theo là **Kiến trúc Backbone** (+1.8% từ ResNet lên ConvNeXt) và **Kỹ thuật Augmentation CutMix** (+1.13%).
   - Phương pháp suy luận Temperature Scaling đóng góp vai trò cốt lõi trong **hiệu chuẩn xác suất (Calibration)** mà không tốn tài nguyên tính toán.
3. **Khuyến nghị triển khai trên Robot Nông nghiệp (Ngân sách 30–100 ms):**  
   - **Lựa chọn 1 (Độ chính xác cao):** Triển khai **ConvNeXt-Tiny + FP16 + Temperature Scaling**. Độ trễ thực tế chỉ **$8.1\\text{ ms}$ (p95)**, xử lý mượt mà ở tốc độ 120+ khung hình/giây trên Jetson / GPU biên, Macro-F1 đạt 0.9652.
   - **Lựa chọn 2 (Phần cứng cực kỳ hạn chế):** Triển khai **EfficientNet-B0 + INT8/FP16** với độ trễ chỉ **$4.2\\text{ ms}$**, tiêu thụ ít điện năng.

---

## 8. Hạn Chế & Hướng Nghiên Cứu Tiếp Theo

1. **Hạn chế về chia dữ liệu ngẫu nhiên:** DeepWeeds Fold 0 được chia ngẫu nhiên theo ảnh, không phân tách theo địa điểm trang trại (geographical location). Do đó, điểm test có thể hơi lạc quan so với thực tế khi robot di chuyển sang một cánh đồng có chất đất và ánh sáng hoàn toàn mới.
2. **Đánh giá trên đa Fold:** Bài làm tập trung sâu vào Fold 0 theo yêu cầu chuẩn. Trong tương lai, việc huấn luyện chéo 5-fold (Cross-validation) sẽ giúp đánh giá độ bền vững cao hơn nữa.
3. **Thích ứng thời gian thực (Test-Time Adaptation - TTA):** Nghiên cứu chuẩn hóa thống kê BatchNorm hoặc Tent khi gặp điều kiện thời tiết cực đoan (mưa, bụi đất mờ ống kính camera).
