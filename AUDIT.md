# Legacy audit — semantic deadline campaign

> This audit applies to `configs/deadline_full.yaml`, not the default standalone
> CM + CORAL baseline runpack. See `README.md` and `RUNBOOK_VI.md` for the
> current launch/report contract.

## Scientific audit

### Câu hỏi bắt buộc

**Nó giải quyết được bài toán long-tail diffusion chưa? — Chưa.**

Run này kiểm tra gate quyết định hẹp nhưng quan trọng: class-local inferred reweighting có sửa terminal within-class semantic-mass allocation hay không. Một PASS chứng minh method xử lý failure axis đã khóa trong controlled benchmark; nó chưa tự động chứng minh giải quyết toàn bộ long-tail diffusion trên các benchmark chuẩn.

Để được claim rộng hơn, sau PASS vẫn cần:

1. tail/few FID và Recall cải thiện trên CIFAR10/100-LT;
2. head non-inferiority;
3. thắng class-balancing/frequency controls;
4. cạnh tranh với CBDM, OC/T2H, CORAL và CM dưới reporting công bằng;
5. robustness với K/representation ngoài frozen controlled setup.

### Điều run này bảo vệ

- Same architecture/optimizer/steps/sampler/evaluator cho decisive arms.
- LT và weighted arms đều dùng replacement sampling.
- Fine labels bị khóa khỏi practical training arms.
- Oracle được đánh dấu riêng.
- Matched permutation kiểm tra gain có đến từ semantic correspondence hay chỉ weight spectrum/ESS.
- Point-fit là đối chứng bắt buộc.
- Bootstrap CI và paired seeds được dùng cho verdict.
- Built-in CIFAR FID của CORAL bị tắt ở custom semantic dataset; evaluator frozen phải cung cấp metrics đúng reference.

### Điều tuyệt đối không được claim từ kết quả này

- inferred components là ground-truth modes;
- fixed K=5 là unknown-K discovery;
- weighted empirical DSM đúng bằng continuous predictive target;
- average DSM improvement bảo đảm terminal recovery;
- head/population sharing đã được chứng minh;
- timestep gate, router, donor selection hoặc OT geometry là mechanism.

## Engineering audit

### Đã kiểm tra trong package

- `python -m compileall` pass.
- 9 unit/integration tests pass:
  - campaign expansion/task matrix;
  - CORAL patch idempotency/fail-closed anchors;
  - frozen manifest/weight/firewall/matched permutation preflight;
  - synthetic PASS/KILL aggregation;
  - CM/OC command generation.
- SQLite state, stale-worker recovery, retry delay, deterministic W&B IDs.
- Child process nằm cùng process group để stop không bỏ orphan GPU process.
- Provenance ghi task, commands, git commits/status, Python, Torch, CUDA/cuDNN.
- CM/OC eval dùng success marker để không skip partial output.

### Chưa thể kiểm tra trong môi trường đóng gói

- CUDA runtime thực trên server của bạn;
- compatibility với exact package versions đang cài trên server;
- private frozen manifest/weights/evaluator vì chúng chưa được cung cấp trong runpack;
- end-to-end 50k sampling của từng official repository.

Vì vậy smoke test trên server là bắt buộc. Full preflight fail-closed thay vì đoán hoặc tự reconstruct locked experiment.

## Operational audit

- W&B secret nằm trong `.env`, không commit/hardcode.
- Project mặc định `longtail`.
- Decisive tasks priority 100; official baseline priority thấp hơn.
- Disk guard dừng launch mới khi free space dưới threshold.
- OOM retry giảm batch và log rõ confound operational.
- Không tự tune hyperparameter từ kết quả đang chạy.
