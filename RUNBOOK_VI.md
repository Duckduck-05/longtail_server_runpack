# Runbook — Native CIFAR-LT baseline table

**Scope hiện tại: chỉ CIFAR-100-LT IF100** (9 task native — CIFAR-10-LT sẽ chạy sau).
Trên server CUDA, ở root của runpack độc lập này chạy:

```bash
bash scripts/run_server_c100.sh
```

Không cần clone `Longtail` hay thư mục nào khác. Lệnh tự tạo/cập nhật conda
environment từ `environment.yml`, bootstrap các third-party đã vendored, đọc
`.env.local`, tự tải CIFAR-100 bằng `torchvision`, tạo metric assets, rồi
chạy/resume `configs/native_cifar100_if100.yaml`: DDPM/CBDM/CCUA đều qua
`CCUA-DDPM`, nhưng objective được bật/tắt tường minh theo từng row. CM,
CORAL, T2H/unified và IP-SVT tạm dừng cho đến khi đủ chín baseline rows.

Trên Viettel, nên đặt run mới ở volume còn dung lượng:

```bash
export LTX_RUNS_ROOT=/home/nvidia-lab/data_mount/longtail_server_runpack/runs
```

Checkpoint DDPM cũ dưới `runs/` của project là Coral schema có projection head
riêng, nên không được link vào CCUA-DDPM. Launcher giữ run cũ để audit và chỉ
resume checkpoint được tạo trong đúng CCUA-DDPM method/seed directory.

Các launcher unified cũ vẫn giữ lại để truy hồi job đang chạy, nhưng không
được dùng cho bảng chính.

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

Campaign chính (`configs/native_cifar100_if100.yaml`,
`scripts/run_server_c100.sh`) khoá đúng ma trận native dưới đây.

- Cell hiện tại: CIFAR-100-LT IF100 (CIFAR-10-LT sẽ chạy sau).
- Methods: DDPM, CBDM, CCUA.
- Seeds: 0, 1, 2 cho mọi data × method.
- Train: 300,000 updates; batch 64; LR 2e-4; T=1000; conditional CFG;
  exponential LT split với `split_seed=0`.
- U-Net ch=128 [1,2,2,2] attn[1] 2 blocks, EMA 0.9999 — pin trong contract,
  preflight fail nếu lệch. Checkpoint v2 ghi mỗi 50k bước và giữ lại theo
  namespace để kiểm tra lineage/resume gần điểm dừng; nếu thiếu disk thì giảm
  `save_step` trước khi chạy.
- 300k×64 = 19.2M ảnh, giữ nguyên outer budget đã khóa cho bảng chính.
- Eval: 50,000 ảnh 32×32 với nhãn điều kiện đúng class-uniform, official
  DDIM-100 ở omega=1.5, một shared evaluator cho FID/IS/F₈/F₁⁄₈/IPR/KID và FID theo
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
python -m ltx.cli status --config configs/native_cifar100_if100.yaml --watch 30
```

Kết quả paper-facing chỉ đọc từ campaign `native_cifar100_if100_v1`:

```text
runs/native_cifar100_if100_v1/report/per_seed.csv
runs/native_cifar100_if100_v1/report/summary.json
runs/native_cifar100_if100_v1/report/results.log       # bảng + trạng thái từng task + link W&B
runs/native_cifar100_if100_v1/latest.log               # stdout live của lần chạy gần nhất (symlink)
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

**Dừng.** `python -m ltx.cli stop --config configs/native_cifar100_if100.yaml` gửi
SIGTERM cho process group của mọi worker đang chạy. Lần chạy sau tự nhặt lại
(trạng thái `retry`, không phải `failed`). Checkpoint ghi mỗi 50k bước và chỉ
giữ trong đúng native run directory để tiếp tục từ checkpoint gần nhất, nên
mỗi lần dừng mất tối đa 50k bước của task đang train. Nếu disk pressure tăng,
hạ `save_step` trong `configs/native_cifar100_if100.yaml` trước.

**Task `failed` không tự resume.** `run` chỉ nhặt `pending`/`retry`, phải
requeue tay:

```bash
python -m ltx.cli retry-failed --config configs/native_cifar100_if100.yaml
bash scripts/run_server_c100.sh
```

Task được requeue sẽ skip phase train khi `ckpt_300000.pt` đã tồn tại trong
đúng native run directory; nếu chết ở eval thì chỉ chạy lại eval + metrics,
**không train lại**.

Để dùng checkpoint Coral ngoài run directory, phải chỉ rõ method, seed và
mode. Không dùng checkpoint unified, checkpoint của objective khác, hoặc
checkpoint khác seed:

```bash
python -m ltx.cli run --config configs/native_cifar100_if100.yaml \
  --resume-method ddpm --resume-seed 0 \
  --resume-checkpoint /path/to/ckpt_200000.pt \
  --resume-step 200000 --resume-mode full
```

Không có hành vi tự động lấy checkpoint từ campaign khác. Nếu có một file cho
mỗi seed, dùng `{seed}` trong đường dẫn và bỏ `--resume-seed`. Resume explicit
được ghi vào task provenance/W&B; state database cũ có fingerprint khác thì
phải dùng campaign name hoặc `LTX_RUNS_ROOT` mới, tránh trộn hai protocol.

**Native checkpoint/eval.** Train, sample và metric đi qua CCUA-DDPM adapter
với objective đúng của từng row. `ckpt_300000.pt` chỉ được tiếp tục trong
đúng objective/seed; Coral hoặc unified checkpoint không được nạp vào bảng
native. Sampler chính thức vẫn khóa DDIM-100 và evaluator dùng chung.

Muốn kiểm chứng trên GPU trong ~2 phút (train 20 bước rồi chạy đúng nhánh eval
DDIM của common host với stride thô):

```bash
source .venv/bin/activate
python -m ltx.cli run --config configs/smoke_t2h.yaml --skip-preflight
```

Nó ghi sample array/nhãn class-uniform theo namespace v2 và marker `SUCCESS`
trong `runs/smoke_t2h_v1/`, và lên W&B với tag `smoke`. FID/IS ở đó là `nan` theo
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

Không gọi kết quả này là “paper reproduction” bit-for-bit: đây là protocol
outer-control native mới. Chi tiết khoa học ở
[UNIFIED_CIFAR_PROTOCOL.md](UNIFIED_CIFAR_PROTOCOL.md).
