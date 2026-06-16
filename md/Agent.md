# ROLE & CONTEXT
You are an elite Deep Learning Engineer specializing in Remote Sensing and Semantic Segmentation. Your task is to generate a complete, production-grade, modular PyTorch codebase for a satellite imagery segmentation research project on the OpenEarthMap dataset.

We are implementing an **Ablation Study** to compare pure Convolutional Encoders (ResNet-18, ResNet-101) versus a Vision Transformer Encoder (MiT). The project utilizes a dynamic "Multi-Encoder" UNetFormer architecture.

# ENVIRONMENT & HARDWARE CONFIGURATION
- **Target Platform:** Kaggle Notebooks executed via background commits (`Save & Run All`).
- **Hardware:** 2x NVIDIA T4 GPUs.
- **Distributed Training:** Must strictly use PyTorch `DistributedDataParallel` (DDP) initialized via `torchrun`.
- **Memory Optimization:** Must strictly incorporate Automatic Mixed Precision (AMP) via `torch.cuda.amp`.

# PROFESSIONAL REPOSITORY ARCHITECTURE
Generate the full production-grade code adhering strictly to the following modular directory layout. Avoid placeholders like `# Implement here`. Provide complete, runnable file implementations.

```text
unetformer-openearthmap/
├── requirements.txt
├── val_2000_fixed.txt         # Pre-generated validation split file
├── configs/
│   ├── luot1_500.yaml
│   ├── luot2_1000.yaml
│   └── luot3_1500.yaml
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── unetformer.py
│   └── utils/
│       ├── __init__.py
│       ├── losses.py
│       ├── metrics.py
│       ├── callbacks.py
│       └── visualizer.py
│── md/
│── prompt/
DETAILED SPECIFICATIONS PER MODULE
1. Requirements (requirements.txt)
List all essential libraries pinning production-safe versions: torch, torchvision, timm, segmentation_models_pytorch, albumentations, matplotlib, pyyaml, pandas.

2. Configuration System (configs/*.yaml)
Create the YAML structure supporting dynamic overriding. Provide templates for three experimental tiers.

Example structure for configs/luot1_500.yaml:
DATASET:
  ROOT_DIR: "/kaggle/input/open-earth-map" 
  TRAIN_IMG_DIR: "images/train"
  TRAIN_MASK_DIR: "labels/train"
  VAL_IMG_DIR: "images/val"
  VAL_MASK_DIR: "labels/val"
  TRAIN_SPLIT_FILE: "train_500_fixed.txt"
  VAL_SPLIT_FILE: "/kaggle/working/unetformer-openearthmap/val_2000_fixed.txt"

MODEL:
  ARCH: "UnetFormer"
  ENCODER: "resnet101"  # Options: "resnet18", "resnet101", "mit_b0"
  PRETRAINED: True

TRAIN:
  NUM_CLASSES: 9
  MAX_ITERS: 40000
  VAL_INTERVAL: 4000
  EARLY_STOPPING_PATIENCE: 2
  BATCH_SIZE_PER_GPU: 2

OPTIMIZER:
  BASE_LR: 0.0006

3. Data Infrastructure (src/data/)
    dataset.py: A robust PyTorch Dataset for OpenEarthMap. Input patches are 1024x1024.

        Parsing Pre-defined Splits (CRUCIAL): The provided split text files contain lines with nested paths (e.g., images/val/svaneti_44_aug0). In __init__, you MUST read the file, strip the line, and use os.path.basename(line) to extract just the filename prefix (e.g., svaneti_44_aug0). Append .tif (or appropriate suffix) and join with ROOT_DIR and IMG_DIR/MASK_DIR to construct absolute paths.

        Include explicit path validity checks (assert os.path.exists).

    transforms.py: Utilize albumentations.

        Train Augmentations: Resize to 1024x1024, Random Crop to 512x512. Apply Linear/Geometric mutations to BOTH image and mask. Apply Color mutations ONLY to the source image.

        Validation Pipeline: Strictly deterministic. Resize to 1024x1024, Normalize. NO data augmentation.
4. Dynamic Architecture (src/models/unetformer.py)
Implement a dynamic UnetFormer class supporting Ablation Studies:
    Encoder Factory Logic: Inside __init__, read the encoder_name.
        If "resnet18": Load pre-trained ResNet-18 (Channels: [64, 128, 256, 512]).
        If "resnet101": Load pre-trained ResNet-101 (Channels: [256, 512, 1024, 2048]).
        If "mit_b0": Load pre-trained MiT (Channels: [32, 64, 160, 256]).
    Channel Projection Block (CRUCIAL): Because ResNet-101 outputs massively different dimensions compared to ResNet-18 or MiT, you MUST implement a dynamic $1 \times 1$ Convolution block immediately after the encoder. This block checks the encoder_name and maps its specific output channels into a unified, standard set of dimensions (e.g., [64, 128, 256, 512]) BEFORE passing them to the Decoder.
    Decoder: A custom Convolutional Decoder expecting the unified channel dimensions from the projection block. DO NOT insert attention mechanisms here.
5. Custom Utilities (src/utils/)
    losses.py: Multi-class Loss (Cross-Entropy + Dice/IoU Loss). Guard against NaN errors under float16 AMP.

    metrics.py: Confusion-matrix based calculation for mIoU and per-class IoU for 9 classes.

    callbacks.py: State-tracking Early Stopping. Break loop if validation mIoU fails to improve after EARLY_STOPPING_PATIENCE checks.

    visualizer.py: Color-code the 9 classes based on the OpenEarthMap RGB palette. Generate a 3-column visualization subplot: [Original | Ground Truth | Prediction]. Save as PNGs.
6. Execution Core (src/train.py)
    Parse arguments (--config and --dry-run).

    Dry-run Mechanism: If active, set MAX_ITERS=5, VAL_INTERVAL=2, and stream only 4 samples.

    Initialize init_process_group, wrap model in DistributedDataParallel, set up DistributedSampler.

    Loop training steps using AdamW and PolyLR decay policy.

    Every VAL_INTERVAL iterations, evaluate on validation set. On early stop or completion, export best_model.pth, benchmark_results.csv, and learning_curves.png.
OUTPUT DELIVERY EXPECTATION
Generate all modular files completely with professional code practices. Do not truncate code. Ensure the Channel Projection Block logic handling the massive [256, 512, 1024, 2048] channels of ResNet-101 is accurately implemented.