#!/usr/bin/env bash
# resume_checkpoint.sh — cài requirements, tạo lại split files, hỏi người dùng
# upload checkpoint (latest_checkpoint.pth) lên /kaggle/working, rồi resume
# training thí nghiệm resnext101_32x16d (configs/cnn_reshnet101_32/) từ đó.
#
# Dùng khi Kaggle session bị ngắt giữa lúc train thí nghiệm mới. File này
# HOÀN TOÀN ĐỘC LẬP — không động tới run_cnn_experiment.sh /
# run_cnn_reshnet101_32_experiment.sh hay bất kỳ script cũ nào khác.
#
# Cách dùng:
#   bash resume_checkpoint.sh <1|2|3>
#     1 → resume luot1 (500 ảnh)
#     2 → resume luot2 (1000 ảnh)
#     3 → resume luot3 (1500 ảnh)
#
#   Bỏ qua hỏi nếu đã biết sẵn đường dẫn checkpoint:
#   bash resume_checkpoint.sh <1|2|3> --path /kaggle/working/.../latest_checkpoint.pth

set -e

DATA_ROOT="/kaggle/input/datasets/dyiyacao/openearthmap"
WORK_BASE="/kaggle/working/unetcnn-resnext101-openearthmap"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPERIMENT="${1:-}"
shift || true
EXTRA_PATH_VALUE=""
if [[ "${1:-}" == "--path" ]]; then
    EXTRA_PATH_VALUE="${2:-}"
fi

if [[ -z "$EXPERIMENT" ]]; then
    echo "Cách dùng: bash resume_checkpoint.sh <1|2|3> [--path <checkpoint.pth>]"
    echo "  1 → resume luot1 (500 ảnh)"
    echo "  2 → resume luot2 (1000 ảnh)"
    echo "  3 → resume luot3 (1500 ảnh)"
    exit 1
fi

case "$EXPERIMENT" in
    1) CONFIG="configs/cnn_reshnet101_32/luot1_500.yaml";  WORK_DIR="$WORK_BASE/work_dirs/luot1_500";  LABEL="luot1 — 500 ảnh"  ;;
    2) CONFIG="configs/cnn_reshnet101_32/luot2_1000.yaml"; WORK_DIR="$WORK_BASE/work_dirs/luot2_1000"; LABEL="luot2 — 1000 ảnh" ;;
    3) CONFIG="configs/cnn_reshnet101_32/luot3_1500.yaml"; WORK_DIR="$WORK_BASE/work_dirs/luot3_1500"; LABEL="luot3 — 1500 ảnh" ;;
    *)
        echo "Lỗi: experiment phải là 1, 2 hoặc 3 (nhận được: '$EXPERIMENT')"
        exit 1
        ;;
esac

echo "========================================================"
echo " Resume training (resnext101_32x16d) — $LABEL"
echo "========================================================"

# ── Bước 1: Cài requirements ─────────────────────────────────────────────────
echo ""
echo "[1/4] Cài đặt requirements ..."
pip install -q -r "$SCRIPT_DIR/requirements.txt" 2>&1 \
    | grep -v -E "pip's dependency resolver|requires .*(incompatible|which is not installed)" \
    || true
echo "      Done."

# ── Bước 2: Tạo lại split files ───────────────────────────────────────────────
echo ""
echo "[2/4] Tạo split files ..."
python "$SCRIPT_DIR/Tools/create_splits.py" \
    --data-root "$DATA_ROOT" \
    --output-dir "$WORK_BASE"
echo "      Done."

# ── Bước 3: Hỏi người dùng upload + xác định checkpoint ─────────────────────
echo ""
echo "[3/4] Xác định checkpoint để resume ..."
mkdir -p "$WORK_DIR"
DEFAULT_CKPT="$WORK_DIR/latest_checkpoint.pth"

set +e
if [[ -n "$EXTRA_PATH_VALUE" ]]; then
    CKPT_PATH="$(python "$SCRIPT_DIR/Tools/get_resume_checkpoint.py" \
        --default-path "$DEFAULT_CKPT" --path "$EXTRA_PATH_VALUE")"
else
    CKPT_PATH="$(python "$SCRIPT_DIR/Tools/get_resume_checkpoint.py" \
        --default-path "$DEFAULT_CKPT")"
fi
GET_CKPT_STATUS=$?
set -e

if [[ $GET_CKPT_STATUS -ne 0 || -z "$CKPT_PATH" ]]; then
    echo "Lỗi: không xác định được checkpoint hợp lệ để resume."
    exit 1
fi
echo "      Checkpoint: $CKPT_PATH"

# ── Bước 4: Resume training ───────────────────────────────────────────────────
echo ""
echo "[4/4] Resume training: $CONFIG"
echo "      Kết quả lưu tại: $WORK_DIR"
echo ""

cd "$SCRIPT_DIR"
torchrun --nproc_per_node=2 src/train_cnn_resnext101_32.py \
    --config "$CONFIG" --resume "$CKPT_PATH"

echo ""
echo "========================================================"
echo " Hoàn thành (resume): $LABEL"
echo " Xem kết quả tại: $WORK_DIR"
echo "========================================================"
