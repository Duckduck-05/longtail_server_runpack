# Runbook — Unified CIFAR-LT table

**Scope hiện tại: chỉ CIFAR-100-LT** (18 task — CIFAR-10-LT sẽ chạy sau).
Trên server CUDA, ở root của runpack độc lập này chạy:

```bash
bash scripts/run_server_c100.sh
```

Không cần clone `Longtail` hay thư mục nào khác. Lệnh tự tạo/cập nhật conda
environment từ `environment.yml`, bootstrap các third-party đã vendored, đọc
`.env.local`, tự tải CIFAR-100 bằng `torchvision`, tạo metric assets, rồi
chạy/resume campaign 18 task (`configs/unified_cifar_c100.yaml`).

Khi nào cần chạy đủ cả CIFAR-10-LT + CIFAR-100-LT (54 task, protocol khoá
cứng gốc), dùng `bash scripts/run_server.sh` thay thế — cùng máy, cùng
environment, không cần setup gì thêm.

## Chọn mức song song (GPU packing)

Mặc định lệnh trên tự dò VRAM trống của từng GPU và tự quyết định số task
chạy chung một GPU (`machine.tasks_per_gpu: auto` trong `configs/server.yaml`),
bắt đầu từ ước lượng 12 GB/task rồi tự hiệu chỉnh theo footprint thật của
task đầu tiên hoàn thành. Không cần biết trước cấu hình máy thuê.

Ép tay khi cần — ví dụ máy đang cotenant hoặc muốn giới hạn GPU cụ thể:

```bash
bash scripts/run_server_c100.sh --per-gpu 3 --gpus 0,1,2,3
bash scripts/run_server_c100.sh --jobs 8        # trần tổng số task chạy đồng thời
```

Hoặc đặt trong `.env.local`: `LTX_TASKS_PER_GPU`, `LTX_GPU_IDS`,
`LTX_MAX_CONCURRENT`. Hai task chung một GPU sẽ tự chia nhỏ `num_workers` của
dataloader để không quá tải CPU của host.

## Campaign được khóa

Bản đầy đủ (`configs/unified_cifar.yaml`, `scripts/run_server.sh`) khoá cứng
matrix dưới đây; bản đang chạy (`configs/unified_cifar_c100.yaml`,
`scripts/run_server_c100.sh`) chỉ dùng đúng dòng CIFAR-100-LT IF100 — cùng
methods/seeds/budget/contract, chỉ khác `fairness_contract.cells` còn một cell.

- Cells: CIFAR-10-LT IF100, CIFAR-10-LT IF1000, CIFAR-100-LT IF100.
- Methods: DDPM, CBDM, T2H, CM, CORAL, CCUA.
- Seeds: 0, 1, 2 cho mọi data × method.
- Train: 300,000 updates; batch 64; LR 2e-4; T=1000; conditional CFG;
  exponential LT split với `split_seed=0`.
- U-Net ch=128 [1,2,2,2] attn[1] 2 blocks, EMA 0.9999 — pin trong contract,
  preflight fail nếu lệch. Checkpoint ghi mỗi 50k bước chỉ để resume khi crash;
  như upstream, chỉ giữ cái mới nhất.
- 300k×64 = 19.2M ảnh, đúng bằng ngân sách của CBDM (300k×64), CM (300k×64)
  và CORAL (150k×128), nên không baseline nào bị train thiếu so với paper gốc.
- Eval: 50,000 ảnh 32×32 với nhãn điều kiện đúng class-uniform, ancestral
  DDPM 1,000 reverse steps, một shared evaluator cho FID/IS/F₈/F₁⁄₈/IPR.

`OC` là tên repository của paper T2H; không được thêm `oc` thành một method
riêng — nó sẽ nhân đôi đúng một method. Tương tự, `CCUA` chỉ lấy nhánh U-Net
(`CCUA-DDPM`); nhánh `CCUA-SiT` dùng backbone Diffusion Transformer nên không
thuộc bảng này. Preflight sẽ fail nếu matrix, seed, budget, sampler family,
label schedule hoặc metric contract bị lệch.

## Theo dõi và kết quả

```bash
source .venv/bin/activate
python -m ltx.cli status --config configs/unified_cifar_c100.yaml --watch 30
```

Kết quả paper-facing chỉ đọc từ (campaign name `unified_cifar_c100_v1` cho
bản CIFAR-100-LT, `unified_cifar_v1` cho bản đầy đủ):

```text
runs/unified_cifar_c100_v1/report/table.md
runs/unified_cifar_c100_v1/report/per_seed.csv
runs/unified_cifar_c100_v1/report/summary.json
runs/unified_cifar_c100_v1/report/results.log       # bảng + fingerprint + trạng thái từng task + link W&B, gộp một file
runs/unified_cifar_c100_v1/report/campaign_run.log  # snapshot stdout toàn campaign (bootstrap, GPU packing, launch, lỗi)
runs/unified_cifar_c100_v1/latest.log               # stdout live của lần chạy gần nhất (symlink)
```

W&B có sáu run groups theo cell/method, loss và system telemetry theo task,
sample grids, `comparison/per_seed`, `comparison/unified_main_table`, và
report artifact. Report fail-closed nếu một trong ba seed hoặc metric bị thiếu.
Với `WANDB_API_KEY` đã có trong `.env.local`, `--wandb` (script chính luôn
bật cờ này) sẽ tự đặt project thành public-read và tạo một W&B Report tổng
hợp bảng + biểu đồ; link report được in ra cuối lệnh và ghi vào `results.log`.

Toàn bộ file trong `report/` (kể cả `results.log` và `campaign_run.log`) được
upload thành artifact `evaluation-report` của run report. Nghĩa là người chạy
hộ chỉ cần đưa lại **một link W&B** — bảng, trạng thái từng task, và toàn bộ
stdout của campaign đều đọc được trên đó, không cần quyền vào máy.

Không gọi kết quả này là “paper reproduction”: đây là protocol chung mới với
source-native implementations. Chi tiết khoa học ở
[UNIFIED_CIFAR_PROTOCOL.md](UNIFIED_CIFAR_PROTOCOL.md).
