# Resume training khi Kaggle bị ngắt session (thí nghiệm resnext101_32x16d)

Áp dụng cho thí nghiệm mới `configs/cnn_reshnet101_32/` (encoder resnext101_32x16d,
CE thuần, lr=1e-5) — chạy qua `src/train_cnn_resnext101_32.py`. Script này lưu
checkpoint đầy đủ (model + optimizer + scaler AMP + iteration + early-stopping
state) sau **mỗi lần validation**, ghi đè vào 1 file duy nhất `latest_checkpoint.pth`
trong `WORK_DIR` của từng lượt (luot1/2/3), nên có thể tiếp tục train đúng vị trí
nếu Kaggle session bị ngắt giữa lúc chạy.

## Cách 1 — Upload checkpoint qua browse trên trình duyệt (khuyên dùng)

Chạy trong **1 cell của Kaggle/Jupyter notebook** (không chạy qua bash/terminal):

```python
%run "Tools/get_resume_checkpoint.py" --browser --default-path /kaggle/working/unetcnn-resnext101-openearthmap/work_dirs/luot1_500/latest_checkpoint.pth
```

- Lệnh này hiện ra nút **Upload** — bấm vào sẽ mở hộp thoại browse file của trình
  duyệt, chọn file `latest_checkpoint.pth` đã lưu từ session trước trên máy bạn.
- Sau khi thấy dòng `✔ Đã lưu checkpoint vào: ...`, file đã nằm đúng vị trí
  `WORK_DIR` của lượt tương ứng.
- Đổi `luot1_500` thành `luot2_1000` hoặc `luot3_1500` tùy lượt đang resume.

Sau đó chạy resume bình thường ở terminal/cell khác:

```bash
bash resume_checkpoint.sh 1
```

Vì checkpoint đã có sẵn ở đường dẫn mặc định, chỉ cần nhấn Enter khi được hỏi
(hoặc dùng `--path` để bỏ qua hẳn câu hỏi, xem Cách 2).

## Cách 2 — Upload thủ công qua panel Kaggle + nhập đường dẫn ở terminal

1. Upload file `latest_checkpoint.pth` lên `/kaggle/working/...` qua panel
   "Data" bên phải hoặc File Browser của notebook (kéo-thả file).
2. Chạy:
   ```bash
   bash resume_checkpoint.sh <1|2|3>
   ```
   Script sẽ hỏi đường dẫn checkpoint — nhấn Enter để dùng mặc định (nếu đã
   upload đúng vị trí gợi ý) hoặc nhập đường dẫn khác.
3. Hoặc bỏ qua hẳn câu hỏi bằng cách truyền sẵn đường dẫn:
   ```bash
   bash resume_checkpoint.sh <1|2|3> --path /kaggle/working/.../latest_checkpoint.pth
   ```

## Lưu ý quan trọng

- Chế độ `--browser` **chỉ hoạt động khi chạy trực tiếp trong notebook cell**
  (qua `%run`), **không hoạt động** qua bash/subprocess (vì widget upload cần
  kết nối trực tiếp tới kernel hiển thị trên trình duyệt). Đây là lý do
  `resume_checkpoint.sh` (chạy qua bash) chỉ hỗ trợ Cách 2 (hỏi đường dẫn ở
  terminal), còn Cách 1 (browse) phải chạy riêng trong 1 cell trước.
- `resume_checkpoint.sh` tự cài lại `requirements.txt` và tạo lại split files
  trước khi resume — không cần làm thủ công các bước này.
- 1/2/3 tương ứng với `luot1_500` / `luot2_1000` / `luot3_1500`, đúng như
  `run_cnn_reshnet101_32_experiment.sh`.
