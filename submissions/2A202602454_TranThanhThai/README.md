# Báo Cáo & Hướng Dẫn Thực Hiện Lab Day 2 — DeepWeeds

- **Học viên:** Trần Thanh Thái
- **MSSV:** 2A202602454
- **Lớp / Track:** Track 4 — Day 2: Advanced Deep Learning (Backbone, Training Recipe & Inference)
- **Repo bài làm:** `submissions/2A202602454_TranThanhThai/`

---

## 1. Môi trường & Phiên bản Thư viện

- **Môi trường thực thi khuyến nghị:** Google Colab (GPU Tesla T4 16GB) hoặc Kaggle Notebooks.
- **Python:** `3.10` / `3.11` / `3.12`
- **PyTorch:** `>= 2.0.0` (hỗ trợ AMP Autocast & GradScaler)
- **torchvision:** `>= 0.15.0`
- **timm:** `>= 0.9.0` (tải và khởi tạo các backbone CNN & Transformer)
- **scikit-learn:** `>= 1.2.0` (tính toán các chỉ số so sánh)
- **pandas, numpy, openpyxl, matplotlib, scipy**

---

## 2. Cấu trúc Thư mục Nộp bài

```text
submissions/2A202602454_TranThanhThai/
├── README.md               # File này: tổng quan, môi trường, hướng dẫn chạy lại
├── results.xlsx            # Bảng so sánh 7 sheets chi tiết theo chuẩn GUIDE.md
├── report.md               # Báo cáo khoa học chi tiết (kết quả, phân tích, trade-off)
├── curves/                 # Biểu đồ loss và macro-F1 theo epoch cho từng thí nghiệm
│   ├── B01_resnet50.png
│   ├── B02_resnext50_32x4d.png
│   ├── B03_convnext_tiny.png
│   ├── B04_swin_tiny_patch4_window7_224.png
│   ├── B05_efficientnet_b0.png
│   ├── T00_resnet50.png
│   ├── T03_resnet50.png
│   ├── T09_convnext_tiny.png
│   └── F01_convnext_tiny.png
├── predictions/            # File dự đoán softmax (.csv) cho tập test và val
│   ├── T00_seed0_test.csv
│   ├── T00_seed1_test.csv
│   ├── T00_seed2_test.csv
│   ├── F01_seed0_test.csv
│   ├── F01_seed1_test.csv
│   ├── F01_seed2_test.csv
│   ├── F01_uncal_seed0_test.csv
│   └── F01_seed0_val.csv
└── code/                   # Toàn bộ mã nguồn hoàn chỉnh
    ├── dataset.py          # Pipeline dữ liệu, kiểm tra S1-S6, DataLoaders
    ├── model.py            # Backbone qua timm, freeze, param_groups, MACs/params
    ├── losses.py           # Focal Loss, Label Smoothing, Class Weights, CutMix/Mixup
    ├── train.py            # Config, vòng lặp train AMP, Cosine Warmup, EMA, checkpointing
    ├── inference.py        # TTA, Temperature Scaling, Ensemble, Gộp BN
    ├── benchmark.py        # Đo độ trễ p50/p95/p99 đúng chuẩn (GPU sync + warmup)
    ├── export_results.py   # Xuất file results.xlsx đầy đủ 7 sheets
    ├── lab_day2.ipynb      # Notebook tương tác chạy trên Colab
    └── kaggle_day2.ipynb   # Notebook tối ưu hóa chạy trên Kaggle (GPU T4, tự động tải/phát hiện dataset)
```

---

## 3. Hướng dẫn Chạy lại Thực nghiệm

### Cách 1: Chạy bằng Kaggle Notebooks (Khuyên dùng - Ổn định nhất & Không lo mất kết nối)
1. Mở Kaggle ([kaggle.com/code](https://www.kaggle.com/code)), chọn **New Notebook** > **File** > **Upload Notebook** và chọn file [`kaggle_day2.ipynb`](code/kaggle_day2.ipynb) (hoặc ở thư mục gốc repo `kaggle_day2.ipynb`).
2. Cấu hình Runtime trên Kaggle (cột phải):
   - **Accelerator**: Chọn **GPU T4 x 1** (hoặc P100).
   - **Internet**: Chuyển sang **On** (để tải thư viện và dataset).
3. Tùy chọn chạy:
   - **Chạy nhanh kiểm tra (10-15 phút)**: Giữ nguyên `MODE = "FAST_VERIFY"` để kiểm tra tính toàn vẹn của toàn bộ pipeline từ B01–B05, T00–T09, I00–I08 đến F01, `eval.py` và `results.xlsx`.
   - **Chạy huấn luyện đầy đủ**: Đổi `MODE = "FULL_TRAIN"` và nhấn **Save Version** > **Save & Run All (Commit)** để Kaggle chạy ngầm qua đêm mà không cần mở trình duyệt.
4. Tải kết quả: Toàn bộ `results.xlsx`, `predictions/`, `curves/` được tự động đóng gói vào `submission_outputs.zip` trong tab **Output**.

### Cách 2: Chạy bằng Google Colab
1. Mở Google Colab, tạo Notebook mới hoặc tải file [`code/lab_day2.ipynb`](code/lab_day2.ipynb) lên.
2. Chọn Runtime: **Runtime > Change runtime type > Chọn GPU (Tesla T4)**.
3. Chạy tuần tự các ô lệnh:
   - Cài đặt thư viện (`!pip install timm openpyxl matplotlib`).
   - Tải dataset DeepWeeds từ Zenodo (tự động kiểm tra MD5: `b7b30f96d466fba86016aa5a26606e0f`).
   - Chạy EDA & kiểm tra tính toàn vẹn của chia dữ liệu Fold 0 (S1–S6).
   - Chạy các bước sàng lọc Backbone (B01–B05), Công thức huấn luyện (T00–T09) và Suy luận (I00–I08).
   - Chạy Chung kết qua 3 seeds (`0, 1, 2`) và đánh giá bằng `eval.py score` & `eval.py grade`.

### Cách 3: Chạy từ Terminal / Command Line
Sử dụng trình thông dịch `py`:

```bash
# 1. Kiểm tra bộ unit tests của repo
py -3.12 -m unittest discover -s tests -v

# 2. Huấn luyện mốc nền T00 (ResNet-50)
py -3.12 submissions/2A202602454_TranThanhThai/code/train.py --set exp_id=T00 backbone=resnet50 seed=0

# 3. Huấn luyện cấu hình tốt nhất F01 (ConvNeXt-Tiny)
py -3.12 submissions/2A202602454_TranThanhThai/code/train.py --set exp_id=F01 backbone=convnext_tiny mix=cutmix loss=ls label_smoothing=0.1 ema_decay=0.999 seed=0 save_test_predictions=True

# 4. Chấm điểm chính thức bằng eval.py
py -3.12 eval.py score --pred "submissions/2A202602454_TranThanhThai/predictions/F01_seed*_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01

py -3.12 eval.py grade --final "submissions/2A202602454_TranThanhThai/predictions/F01_seed*_test.csv" --baseline "submissions/2A202602454_TranThanhThai/predictions/T00_seed*_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --latency-p95-ms 10.2 --latency-method proper
```

---

## 4. Danh sách Seed Đã Dùng

- **Quét sàng Backbone & Huấn luyện (Bước 1 & 2):** Cố định `seed = 0` cho toàn bộ các thí nghiệm để đảm bảo nguyên tắc so sánh công bằng N1.
- **Vòng Chung kết (Bước 4):** Sử dụng **3 seeds:** `0, 1, 2` cho cả Cấu hình tốt nhất (`F01`) và Mốc đối chiếu (`T00`).
