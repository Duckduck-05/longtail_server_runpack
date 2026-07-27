# Runbook server — CM + CORAL long-tail baseline

## Bàn giao

Runpack là độc lập. `.env.local` nằm cạnh README đã có W&B credentials cho
private hand-off; không in nội dung file đó vào terminal hay W&B config.

Trên server CUDA nhận runpack, chạy đúng một lệnh:

```bash
bash scripts/run_server.sh
```

Lệnh này tự dựng environment đã pin, bootstrap third-party ports, tải
CIFAR-10/100, tải archive ImageNet ILSVRC2012 + public ImageNet-LT manifests,
kiểm tra checksum/data contract, rồi chạy CM và CORAL từ scratch.

## Data contract

- CIFAR-10-LT và CIFAR-100-LT: `torchvision` tự tải.
- ImageNet-LT: một ImageNet source duy nhất; ảnh được resize/crop lúc train
  thành hai benchmark 32×32 và 64×64. Preflight bắt buộc 115,846 train rows,
  1,000 classes, và reference split 20,000 ảnh (20/lớp).
- Nếu endpoint ImageNet không truy cập được, dùng private mirror qua
  `LTX_IMAGENET_SOURCE=custom_archive` cùng URL/SHA256 trong `.env.local`.

## Hai protocol tái lập tách biệt

- CM: DDPM/CBDM/OC/CM × seeds 0,1,2 trên CIFAR-10-LT IR100,
  CIFAR-100-LT IR100, ImageNet-LT 32 và ImageNet-LT 64: 48 tasks; source CM,
  200k/300k steps, metric FID/KID.
- CORAL: DDPM/CBDM/T2H/CORAL × seeds 0,1,2 trên ba CIFAR paper cells: 36 tasks;
  source CORAL/OC, 150k/200k steps, metric FID/IS/F-score/Recall.
- Đây **không phải một bảng chung**. DDPM/CBDM ở CIFAR IF100 xuất hiện hai lần
  vì được tái lập theo hai paper protocol khác nhau; không so sánh/chung bình
  metric hay reuse một run giữa hai suite.
- Báo cáo fail-closed: thiếu seed hoặc metric thì không tạo bảng so sánh giả.
- W&B và local reports: `runs/cm_baselines_v1/report/` và
  `runs/coral2025_cifar_v1/report/`.

Đây là **baseline-only**. Chỉ sau khi task của method mới được port thật và
`LTX_CANDIDATE_METHOD` được đặt, report mới tính paired bootstrap CI và nhãn
`WIN`.

## Giới hạn được ghi rõ

CM ImageNet-LT là source port có kiểm soát: public CM release không có YAML
ImageNet-LT của tác giả. Vì vậy report dùng Table-5 làm reference, không tuyên
bố tái lập bit-for-bit kết quả paper.
