# Dataset scope

| Dataset | Paper reports | This independent runpack | Reason |
| --- | --- | --- | --- |
| CIFAR10-LT IF100 / IF1000 | Yes | Runnable, auto-download | Public CORAL/CBDM loader supports it. |
| CIFAR100-LT IF100 | Yes | Runnable, auto-download | Public CORAL/CBDM loader supports it. |
| CelebA-5 | Yes | Intentionally blocked | Public third-party code omits the paper's five-class split/labels. |
| ImageNet-LT | Yes | Runnable, auto-download | `scripts/run_server.sh` obtains official ILSVRC2012 train images plus checksum-pinned public ImageNet-LT manifests, reconstructs `train/<synset>`, then validates 115,846 train images and the exact 1,000-class × 20-image reference split. |
| Places-LT | No | Not added | It would be a new benchmark, not a reproduction. |

The CM campaign is 4 dataset/resolution cells × 4 methods × 3 seeds = 48
tasks: CIFAR-10-LT IR100, CIFAR-100-LT IR100, ImageNet-LT 32 and ImageNet-LT
64. The separate CORAL campaign is 3 CIFAR cells × 4 methods × 3 seeds = 36
tasks. The ImageNet source must be the exact authorized payload referenced by
the supplied manifests; the runpack never substitutes a generic ImageNet or
Places split.
