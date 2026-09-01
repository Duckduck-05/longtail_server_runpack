# Frozen contracts

## Semantic evaluator output

The command in `LTX_SEMANTIC_EVAL_COMMAND` is formatted with
`{samples}`, `{labels}`, `{manifest}`, `{method}`, `{seed}`, `{run_dir}`, and `{output}`.
It must write `semantic_metrics.json` matching `semantic_metrics.schema.json`.

`bootstrap_draws_file` points to an NPZ (absolute or relative to the run directory) containing aligned arrays:

```text
js                 [B]
rare_mode_mass     [B]
coarse_consistency [B]
memorization       [B]
FID                [B]  optional but recommended
KID                [B]  optional but recommended
Recall             [B]  optional but recommended
```

For a fixed model seed, draw `b` must use the same frozen clean-image/class resample for every arm. The aggregator preserves this pairing and then performs a hierarchical bootstrap over model seeds. Bootstrap draws quantify uncertainty; they are not counted as independent model replications.

## Weight sidecars

Every frozen weight array must have a same-stem JSON sidecar matching `weight_manifest.schema.json`. The decisive preflight checks exact sample-order hash, array checksum, class-total mass, the fine-label firewall, estimator descriptors, and per-class spectrum matching between predictive and permutation arms.
