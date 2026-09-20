#!/usr/bin/env bash
# Full pipeline on one dataset: index -> main methods -> ablations -> sweeps -> generic cache -> score.
set -euo pipefail
DS=${1:-fincontab}
python -m experiments.build_index --dataset "$DS"
for M in qwen pcrc pcrc_nocache; do python -m experiments.run_main --dataset "$DS" --method "$M"; done
for V in random all semantic_only spatial_only shared_only exclusive_only no_gating; do
  python -m experiments.run_ablations --dataset "$DS" --variant "$V"
done
for P in k_shared k_excl tau_shared tau_excl lam; do python -m experiments.run_sweeps --dataset "$DS" --param "$P"; done
# requires a running vLLM server (scripts/serve_vllm.sh)
for P in pcrc_schedule single_judgment; do python -m experiments.run_generic_cache --dataset "$DS" --protocol "$P"; done
python -m experiments.score --datasets "$DS"
