#!/usr/bin/env bash
# This compatibility bridge is intentionally disabled for the current native
# campaign. The old DDPM checkpoint was produced by Coral and contains Coral's
# extra mean_proj/logvar_proj parameters; the current baseline runner uses the
# CCUA-DDPM U-Net, whose checkpoint schema is different. Linking it would make
# resume/eval fail (or, worse, turn a non-exact weight import into a claimed
# continuation). Keep the old run for audit and start CCUA-DDPM baselines in
# their own directories.
set -euo pipefail

echo "[reconcile] skipped: Coral checkpoint is not schema-compatible with CCUA-DDPM"
exit 0
