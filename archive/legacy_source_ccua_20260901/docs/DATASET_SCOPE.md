# Dataset scope — Unified CIFAR Benchmark v1

| Dataset | Included in the one-table campaign | Acquisition | Why |
|---|---:|---|---|
| CIFAR-10-LT IF100 | Yes | `torchvision` automatic download | All five methods support this public 32×32 cell. |
| CIFAR-10-LT IF1000 | Yes | `torchvision` automatic download | Stress-test cell shared by all five methods. |
| CIFAR-100-LT IF100 | Yes | `torchvision` automatic download | Public 100-class cell shared by all five methods. |
| CelebA-5 | No | Not used | Exact five-class split is not present in the vendored public sources. |
| ImageNet-LT 32/64 | No | Retained only as legacy tooling | Forcing CM-only support onto every method would not be a fair common table. |
| Places-LT | No | Not used | Not a published common cell for all five vendored methods. |

Each included long-tail split uses the same exponential CIFAR construction and
`split_seed=0`. The launcher makes no ImageNet download for the unified
campaign; this is intentional, not an omitted dependency.
