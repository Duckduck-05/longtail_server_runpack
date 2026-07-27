# Runbook — Unified CIFAR-LT table

Trên server CUDA, ở root của runpack độc lập này chạy:

```bash
bash scripts/run_server.sh
```

Không cần clone `Longtail` hay thư mục nào khác. Lệnh tự tạo/cập nhật conda
environment từ `environment.yml`, bootstrap các third-party đã vendored, đọc
`.env.local`, tự tải CIFAR-10/CIFAR-100 bằng `torchvision`, tạo metric assets,
rồi chạy/resume campaign 45 task.

## Campaign được khóa

- Cells: CIFAR-10-LT IF100, CIFAR-10-LT IF1000, CIFAR-100-LT IF100.
- Methods: DDPM, CBDM, T2H, CM, CORAL.
- Seeds: 0, 1, 2 cho mọi data × method.
- Train: 200,000 updates; batch 64; LR 2e-4; T=1000; conditional CFG;
  exponential LT split với `split_seed=0`.
- Eval: 50,000 ảnh 32×32 với nhãn điều kiện đúng class-uniform, ancestral
  DDPM 1,000 reverse steps, một shared evaluator cho FID/IS/F₈/F₁⁄₈/IPR.

`OC` là tên repository của paper T2H; không được thêm `oc` thành một method
thứ sáu. Preflight sẽ fail nếu matrix, seed, budget, sampler family, label
schedule hoặc metric contract bị lệch.

## Theo dõi và kết quả

```bash
source .venv/bin/activate
python -m ltx.cli status --config configs/unified_cifar.yaml --watch 30
```

Kết quả paper-facing chỉ đọc từ:

```text
runs/unified_cifar_v1/report/table.md
runs/unified_cifar_v1/report/per_seed.csv
runs/unified_cifar_v1/report/summary.json
```

W&B có năm run groups theo cell/method, loss và system telemetry theo task,
sample grids, `comparison/per_seed`, `comparison/unified_main_table`, và
report artifact. Report fail-closed nếu một trong ba seed hoặc metric bị thiếu.

Không gọi kết quả này là “paper reproduction”: đây là protocol chung mới với
source-native implementations. Chi tiết khoa học ở
[UNIFIED_CIFAR_PROTOCOL.md](UNIFIED_CIFAR_PROTOCOL.md).
