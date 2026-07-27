# Vendored third-party sources

This directory is part of the delivered runpack and is deliberately independent
of `../Longtail`. It contains working-tree copies of CBDM-pytorch, IGD-ML,
ImbDiff-CM, OC_LT, and CORAL, without their nested git metadata.

Excluded from the delivery are only regenerable/downloadable material: dataset
payloads, cached metric features/statistics, Python bytecode, and the literal
download-cache directories named `...`. The runner downloads CIFAR and builds
the balanced metric assets on the target server. Exact source commits, local
worktree state, and Python-file hashes are recorded in
`THIRD_PARTY_MANIFEST.json`.
