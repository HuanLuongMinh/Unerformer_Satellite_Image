# UNet CNN — Satellite Image Semantic Segmentation

An ablation study comparing CNN/Transformer encoders on the [OpenEarthMap](https://open-earth-map.org/) dataset using a CNN U-Net decoder (Channel Projection Block + DecoderBlock), kept identical across encoder choices for fair comparison.

Two independent experiment tracks exist side-by-side in this repo:

| Track | Encoder | Loss | Base LR | Scripts/configs |
|---|---|---|---|---|
| **A — baseline (gốc)** | ResNet-101 (ImageNet) | CE + Dice | 6e-4 | `src/train_cnn.py`, `configs/cnn/`, `run_cnn_experiment.sh` |
| **B — resnext101_32x16d (mới)** | ResNeXt-101_32x16d (SWSL, ~1B ảnh pretrain) | CE thuần | 1e-5 | `src/train_cnn_resnext101_32.py`, `configs/cnn_reshnet101_32/`, `run_cnn_reshnet101_32_experiment.sh` |

Track B chỉ thay đổi 3 biến (encoder/loss/lr) so với Track A, **giữ nguyên decoder, dữ liệu, batch size, số iteration, seed** — để so sánh công bằng. Hai track hoàn toàn độc lập về code (không file nào bị chia sẻ/sửa đổi chéo).

---

## Architecture

```
Input (1024×1024, train crop 512×512)
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│  Encoder (theo track)                                          │
│  Track A — ResNet-101        → channels [256, 512, 1024, 2048] │
│  Track B — ResNeXt-101_32x16d → channels [256, 512, 1024, 2048]│
│  (ResNet-18 / MiT-B0 cũng được hỗ trợ trong unetcnn.py gốc)    │
└────────────────────────────────────────────────────────────────┘
     │  4 multi-scale feature maps (stride 4→32)
     ▼
┌─────────────────────────────────────┐
│  Channel Projection Block           │
│  1×1 Conv → unified [64,128,256,512]│
└─────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  CNN Decoder (fixed)         │  3× DecoderBlock (upsample + skip + ConvBNReLU) │
└──────────────────────────────┘
     │
     ▼
Segmentation Map (9 classes)
```

**Key design**: Channel Projection Block chuẩn hóa output encoder về cùng kích thước trước khi decode, nên CNN decoder giống nhau tuyệt đối giữa Track A và Track B — đảm bảo so sánh công bằng khi đổi encoder.

---

## Dataset — OpenEarthMap

| Property | Value |
|---|---|
| Task | Multi-class semantic segmentation |
| Classes | 9 |
| Image format | GeoTIFF (`.tif`) |
| Patch size | 1024 × 1024 px |
| Training crop | 512 × 512 px (random crop) |
| Val patches | 2,000 (fixed split) |

**Classes:**

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | Background | 5 | Tree |
| 1 | Bareland | 6 | Water |
| 2 | Rangeland | 7 | Agriculture |
| 3 | Developed | 8 | Building |
| 4 | Road | | |

---

## Ablation Experiments

### Track A — ResNet-101 baseline (gốc)

| Experiment | Train images | Config |
|---|---|---|
| Luot 1 | 500 | `configs/cnn/luot1_500.yaml` |
| Luot 2 | 1,000 | `configs/cnn/luot2_1000.yaml` |
| Luot 3 | 1,500 | `configs/cnn/luot3_1500.yaml` |

Cả 3 lượt dùng chung 1 seed (42) (model init + sampler shuffle), giống Track B.

### Track B — ResNeXt-101_32x16d, CE thuần, lr=1e-5 (mới)

| Experiment | Train images | Config |
|---|---|---|
| Luot 1 | 500 | `configs/cnn_reshnet101_32/luot1_500.yaml` |
| Luot 2 | 1,000 | `configs/cnn_reshnet101_32/luot2_1000.yaml` |
| Luot 3 | 1,500 | `configs/cnn_reshnet101_32/luot3_1500.yaml` |

Cả 3 lượt trong mỗi track dùng **chung 1 seed (42)** (model init + sampler shuffle), chỉ khác lượng dữ liệu train, để cô lập đúng biến đang khảo sát.

---

## Project Structure

```
UnetFormer Satellite Image/
├── configs/
│   ├── cnn/                              # Track A configs
│   │   ├── luot1_500.yaml
│   │   ├── luot2_1000.yaml
│   │   └── luot3_1500.yaml
│   └── cnn_reshnet101_32/                # Track B configs
│       ├── luot1_500.yaml
│       ├── luot2_1000.yaml
│       └── luot3_1500.yaml
├── src/
│   ├── train_cnn.py                      # Track A training script (DDP + AMP)
│   ├── train_cnn_resnext101_32.py        # Track B training script (DDP + AMP + checkpoint/resume)
│   ├── data/
│   │   ├── dataset.py                    # OpenEarthMap PyTorch Dataset
│   │   └── transforms.py                 # Albumentations pipelines
│   ├── models/
│   │   ├── unetcnn.py                    # Track A: multi-encoder CNN U-Net (resnet18/resnet101/mit_b0)
│   │   └── unetcnn_resnext101_32.py      # Track B: CNN U-Net với encoder resnext101_32x16d
│   └── utils/
│       ├── losses.py                     # Cross-Entropy + Dice loss (dùng cho Track A)
│       ├── metrics.py                    # Confusion-matrix mIoU
│       ├── callbacks.py                  # Early stopping
│       └── visualizer.py                 # RGB prediction visualiser
├── Tools/
│   ├── create_splits.py                  # Generate train split .txt files
│   ├── prepare_splits.py                 # Validate & prepare val split
│   └── get_resume_checkpoint.py          # Xác định/upload checkpoint để resume (Track B)
├── dataset/
│   └── val_2000_fixed.txt                # Pre-generated validation split
├── requirements.txt
├── run_cnn_experiment.sh                 # Launcher Track A
├── run_cnn_reshnet101_32_experiment.sh   # Launcher Track B (chạy mới từ đầu)
├── resume_checkpoint.sh                  # Resume Track B sau khi session bị ngắt
└── read_resume.md                        # Hướng dẫn resume Track B chi tiết
```

---

## Requirements

- Python ≥ 3.9
- CUDA ≥ 12.8
- 2× NVIDIA GPU (tested on 2× T4 16 GB — Kaggle)

Install dependencies:

```bash
pip install -r requirements.txt
```

**Key packages:**

```
torch==2.10.0+cu128   torchvision==0.25.0+cu128
timm==1.0.26          albumentations>=1.3.1
rasterio>=1.4         pyyaml>=6.0.3
matplotlib>=3.9       einops>=0.7.0
```

---

## Usage

### Track A — ResNet-101 baseline

```bash
# Train with 500 / 1000 / 1500 images
bash run_cnn_experiment.sh 1
bash run_cnn_experiment.sh 2
bash run_cnn_experiment.sh 3

# Quick smoke-test (5 iterations only)
bash run_cnn_experiment.sh 1 --dry-run
```

Manual launch:

```bash
python Tools/create_splits.py \
    --data-root /kaggle/input/datasets/dyiyacao/openearthmap \
    --output-dir /kaggle/working/unetcnn-openearthmap

torchrun --nproc_per_node=2 src/train_cnn.py --config configs/cnn/luot1_500.yaml
```

### Track B — ResNeXt-101_32x16d (mới)

```bash
# Train with 500 / 1000 / 1500 images (chạy mới từ đầu)
bash run_cnn_reshnet101_32_experiment.sh 1
bash run_cnn_reshnet101_32_experiment.sh 2
bash run_cnn_reshnet101_32_experiment.sh 3

# Quick smoke-test (5 iterations only)
bash run_cnn_reshnet101_32_experiment.sh 1 --dry-run
```

Manual launch:

```bash
python Tools/create_splits.py \
    --data-root /kaggle/input/datasets/dyiyacao/openearthmap \
    --output-dir /kaggle/working/unetcnn-resnext101-openearthmap

torchrun --nproc_per_node=2 src/train_cnn_resnext101_32.py --config configs/cnn_reshnet101_32/luot1_500.yaml
```

**Nếu Kaggle session bị ngắt giữa lúc train Track B**, dùng `resume_checkpoint.sh` để tiếp tục từ `latest_checkpoint.pth` (lưu sau mỗi lần validation) — xem chi tiết tại [read_resume.md](read_resume.md):

```bash
bash resume_checkpoint.sh 1
# hoặc bỏ qua câu hỏi nếu đã biết đường dẫn checkpoint:
bash resume_checkpoint.sh 1 --path /kaggle/working/.../latest_checkpoint.pth
```

---

## Training Configuration

| Parameter | Track A (gốc) | Track B (mới) |
|---|---|---|
| Encoder | ResNet-101 (ImageNet pretrained) | ResNeXt-101_32x16d (SWSL, ~1B ảnh pretrain) |
| Loss | CrossEntropy + Dice | CrossEntropy thuần |
| Optimizer | AdamW | AdamW |
| Base LR | 6e-4 | 1e-5 |
| LR schedule | Polynomial decay (power=0.9) | Polynomial decay (power=0.9) |
| Warmup | 500 iterations | 500 iterations |
| Seed | 42 (chung cho cả 3 lượt) | 42 (chung cho cả 3 lượt) |
| Batch size | 2/GPU × 2 GPU = 4 total | 2/GPU × 2 GPU = 4 total |
| Max iterations | 40,000 | 40,000 |
| Validation interval | every 4,000 iterations | every 4,000 iterations |
| Early stopping patience | 3 | 3 |
| Gradient clipping | 5.0 | 5.0 |
| Mixed precision | AMP (FP16) | AMP (FP16) |
| Distributed training | DDP via `torchrun` | DDP via `torchrun` |
| Checkpoint/resume | mỗi 12,000 iter (chỉ trọng số) | **mỗi lần validation** (đầy đủ model+optimizer+scaler+early-stopping, hỗ trợ `--resume` thật) |

---

## Output Artifacts

### Track A — `work_dirs/<experiment>/`

```
work_dirs/luot1_500/
├── best_model.pth              # Best checkpoint (highest val mIoU)
├── checkpoint_iter012000.pth   # Periodic checkpoint (every 12k iters, weights only)
├── benchmark_results.csv       # Validation metrics per checkpoint
├── learning_curves.png         # mIoU + val loss over iterations
├── per_class_iou_best.png      # Per-class IoU bar chart at best epoch
└── vis/
    └── iter004000_s0.png       # Prediction visualisations (RGB | GT | Pred)
```

### Track B — `work_dirs/<experiment>/`

```
work_dirs/luot1_500/
├── best_model.pth              # Best checkpoint (ghi đè ngay khi có mIoU mới tốt nhất)
├── latest_checkpoint.pth       # Checkpoint đầy đủ (model+optimizer+scaler+early-stopping),
│                                 ghi đè sau MỖI lần validation — dùng để --resume
├── benchmark_results.csv       # Ghi tăng dần sau mỗi validation, có cột is_best (True/False)
├── learning_curves.png         # mIoU + val loss over iterations
├── per_class_iou_best.png      # Per-class IoU bar chart at best epoch
└── vis/
    └── iter004000_s0.png       # Prediction visualisations (RGB | GT | Pred)
```

Track B không còn file `checkpoint_iter*.pth` định kỳ — đã thay bằng `latest_checkpoint.pth` ghi đè mỗi validation (đỡ tốn dung lượng, đủ để resume).

---

## Estimated Training Time (2× T4 on Kaggle)

### Track A — ResNet-101

| Experiment | ~Iterations | ~Duration |
|---|---|---|
| Luot 1 (500 imgs) | 20,000–24,000 | ~1.8 h |
| Luot 2 (1,000 imgs) | 28,000–36,000 | ~2.5 h |
| Luot 3 (1,500 imgs) | 36,000–40,000 | ~3.2 h |
| **Total (sequential)** | | **~7.5–8 h** |

Early stopping (patience=3) có thể giảm đáng kể thời gian với dataset nhỏ.

### Track B — ResNeXt-101_32x16d (ước tính, chưa đo thực tế)

Encoder nặng hơn ~3.5–4.5× (FLOPs/throughput) so với ResNet-101, và lr=1e-5 thấp hơn nhiều nên khó early-stop sớm như Track A — nhiều khả năng cả 3 lượt chạy gần hết 40,000 iter.

| Experiment | ~Duration |
|---|---|
| Mỗi lượt (500/1000/1500) | ~11–15 h |
| **Total (3 lượt, tuần tự)** | **~34–44 h** |

⚠️ Mỗi lượt có thể **vượt giới hạn session GPU ~9h của Kaggle** — xem [read_resume.md](read_resume.md) để resume sau khi session bị ngắt. Khuyến nghị chạy `--dry-run` trước để đo tốc độ thực tế và hiệu chỉnh lại ước tính này.

---

## License

This project is for academic and research purposes.
