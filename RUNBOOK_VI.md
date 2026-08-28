# Runbook — Unified CIFAR-LT table

**Scope hiện tại: chỉ CIFAR-100-LT** (18 task — CIFAR-10-LT sẽ chạy sau).
Trên server CUDA, ở root của runpack độc lập này chạy:

```bash
bash scripts/run_server_c100.sh
```

Không cần clone `Longtail` hay thư mục nào khác. Lệnh tự tạo/cập nhật conda
environment từ `environment.yml`, bootstrap các third-party đã vendored, đọc
`.env.local`, tự tải CIFAR-100 bằng `torchvision`, tạo metric assets, rồi
chạy/resume campaign 27 task (`configs/unified_cifar_c100.yaml`).

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
- Eval: 50,000 ảnh 32×32 với nhãn điều kiện đúng class-uniform, DDIM-100 ở
  omega=1.5, một shared evaluator cho FID/IS/F₈/F₁⁄₈/IPR/KID và FID theo
  class/Many-Medium-Few. Initial Inception/FID dùng micro-batch 16 để không
  OOM khi nhiều task dùng chung GPU; đổi bằng `eval.inception_batch_size` nếu
  cần.

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

Không chạy các file `/tmp/*.sh` hoặc gọi trực tiếp `third_party/*/main.py`:
đường đó bypass state database/scheduler và không có parent W&B run. Hãy pull
đúng repository này trên server rồi chạy `bash scripts/run_server_c100.sh`.
Nếu `WANDB_MODE=online` mà thiếu key, runner sẽ dừng trước khi chạy thay vì
hoàn tất mà không có kết quả online.

## Dừng, bật lại, và task đã fail

Chạy lại `bash scripts/run_server_c100.sh` luôn là lệnh đúng: nó bootstrap lại
(idempotent), nhận lại các task có worker đã chết, và resume từ checkpoint mới
nhất của từng task.

**Dừng.** `python -m ltx.cli stop --config configs/unified_cifar_c100.yaml` gửi
SIGTERM cho process group của mọi worker đang chạy. Lần chạy sau tự nhặt lại
(trạng thái `retry`, không phải `failed`). Checkpoint ghi mỗi 50k bước và chỉ
giữ cái mới nhất, nên mỗi lần dừng mất tối đa 50k bước (~5 giờ ở 2.6 it/s) của
task đang train. Nếu định dừng/bật nhiều lần, hạ `save_step` trong
`configs/unified_cifar_c100.yaml` trước — chỉ giữ một checkpoint nên không tốn
thêm đĩa.

**Task `failed` không tự resume.** `run` chỉ nhặt `pending`/`retry`, phải
requeue tay:

```bash
python -m ltx.cli retry-failed --config configs/unified_cifar_c100.yaml
bash scripts/run_server_c100.sh
```

Task được requeue sẽ skip mọi phase đã có output, nên task T2H đã train xong
`ckpt_300000.pt` mà chết ở eval chỉ chạy lại eval + metrics, **không train lại**.

Để dùng checkpoint ngoài run directory, phải chỉ rõ method, seed và mode. Full
checkpoint khôi phục model/EMA/optimizer/scheduler; checkpoint cũ chỉ có
`ema_model` là **warm start không exact**, cần opt-in:

```bash
python -m ltx.cli run --config configs/unified_cifar_c100.yaml \
  --resume-method ddpm --resume-seed 0 \
  --resume-checkpoint /path/to/ckpt_200000.pt \
  --resume-step 200000 --resume-mode ema_only
```

Không có hành vi tự động lấy checkpoint từ campaign khác. Nếu có một file cho
mỗi seed, dùng `{seed}` trong đường dẫn và bỏ `--resume-seed`. Resume explicit
được ghi vào task provenance/W&B; state database cũ có fingerprint khác thì
phải dùng campaign name hoặc `LTX_RUNS_ROOT` mới, tránh trộn hai protocol.

**Lỗi eval của T2H/OC với `torch.compile`.** `OC_LT/main.py` bọc U-Net bằng
`torch.compile` nên mọi key trong `net_model` có tiền tố `_orig_mod.`, trong khi
`ddpm_gen.py` dựng `UNet` thuần → `RuntimeError: Error(s) in loading state_dict
for UNet`, chết ở phase eval sau khi đã train xong. `patches/apply_oc_compiled_ckpt.py`
strip tiền tố đó ở đường eval (giống cách upstream đã làm trong `ema()`);
`scripts/bootstrap.sh` tự apply, nên chỉ cần chạy lại script launch là có fix.
Preflight fail-closed nếu thiếu marker
`third_party/OC_LT/.ltx_oc_compiled_ckpt_patch_v1`.

Muốn kiểm chứng trên GPU trong ~2 phút (train 20 bước rồi chạy đúng nhánh eval
ancestral-DDPM với stride thô):

```bash
source .venv/bin/activate
python -m ltx.cli run --config configs/smoke_t2h.yaml --skip-preflight
```

Nó ghi `t2h_samples.npy`, file nhãn class-uniform và marker `SUCCESS` trong
`runs/smoke_t2h_v1/`, và lên W&B với tag `smoke`. FID/IS ở đó là `nan` theo
thiết kế: 64 ảnh chỉ để kiểm tra đường ống, không phải để đo.

**Log tiến trình.** tqdm vẽ lại bằng `\r`, khi output bị pipe thì mỗi lần vẽ
lại thành một record — phase train 31 giờ từng ghi ~300k record vào
`stdout.log` (phase eval còn nhiều hơn, mỗi batch một thanh 1.000 bước) và tất
cả đều được upload làm W&B artifact. Worker giờ chỉ giữ một redraw mỗi
`progress_log_every_seconds` (mặc định 30) và set `TQDM_MININTERVAL` (mặc định
10) cho tiến trình con; hai knob nằm trong `runtime:` của `configs/server.yaml`,
đặt `progress_log_every_seconds: 0` để quay lại hành vi cũ. Dòng kết thúc bằng
newline thật (log thường, traceback, dòng in metric, trạng thái cuối của mỗi
thanh) không bao giờ bị bỏ; cuối mỗi phase in ra số redraw đã lược.

Không gọi kết quả này là “paper reproduction”: đây là protocol chung mới với
source-native implementations. Chi tiết khoa học ở
[UNIFIED_CIFAR_PROTOCOL.md](UNIFIED_CIFAR_PROTOCOL.md).
