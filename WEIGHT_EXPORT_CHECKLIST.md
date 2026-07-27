# Checklist export locked weights

Mỗi `.npy` phải aligned tuyệt đối với `manifest.sample_ids`.

Sidecar tối thiểu:

```json
{
  "dataset_name": "c100_semantic_locked",
  "dataset_fingerprint": "...",
  "sample_ids_sha256": "64-hex",
  "num_samples": 2000,
  "weights_file": "/abs/predictive.npy",
  "weights_sha256": "64-hex",
  "method": "predictive",
  "normalization": "sum_one_per_class",
  "effective_sample_size": 1234.5,
  "estimator_commit": "git-sha",
  "representation": "DINOv2-vitb14-frozen",
  "K": 5,
  "alpha": 1.0,
  "fine_labels_used_for_training": false
}
```

Rules:

- `oracle`: `fine_labels_used_for_training=true`.
- `predictive`, `pointfit`, `permutation`: false.
- Normalize so every coarse class has identical total weight.
- Permutation is within each coarse class and preserves predictive weight multiset exactly.
- Do not regenerate weights per model seed. The frozen estimator is shared; only diffusion initialization changes.
- Do not update weights after seeing terminal generation metrics.
