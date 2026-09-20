# When Tables Need Context: Progressive Context-Reusable Cache (PCRC)

Code and demo data for the anonymous submission *When Tables Need Context:
Query-Conditioned Table Extraction with a Progressive Context-Reusable Cache*.

Query-conditioned table extraction identifies the tables of a document that
satisfy a natural-language condition and reconstructs only those tables as
HTML. PCRC ranks table-relevant document evidence (Section 3.1), judges
candidate tables progressively with a reliability gate, and reuses KV states
across tables, judgment rounds, and the extraction turn through a fixed-head /
replaceable-tail cache frame (Section 3.2, Algorithm 1).

## Repository layout

```
pcrc/
  config.py                    paths, dataset aliases, defaults (K_shared=4, tau_shared=0.12, K_excl=3, tau_excl=0.3, lambda=1)
  data/                        example schema (A.1.1), corpus / condition readers
  preprocessing/
    parsing.py                 front-end parsing (PP-Structure / PyMuPDF), DegConTab text-layer masking
    embedding.py               NV-Embed encoder, s_sem = max(0, cos)
    relevance.py               s_spa = exp(-lambda * d_hat), r = s_sem * s_spa (Eq. 1), factor quantiles (Table 11/12)
    partition.py               g(P_j) mean, S(D) and E(T_i) (Eq. 2-3), random / all / shared-only / exclusive-only variants
  inference/
    prompts.py                 the single-shot and PCRC prompt templates of Appendix A.2, block for block
    engine.py                  multimodal cache frame: head prefill, tail append, progressive append, judge, extract, discard
    pcrc.py                    Algorithm 1 (with the no-gating, single-judgment, and no-cache variants)
    single_shot.py             the Qwen baseline (A.2.1)
    generic_cache.py           PCRC schedule / single judgment through a generic prefix cache (vLLM), A.3.5
    api_backbones.py           frontier VLMs as single-shot baselines and the extraction-only context ablation
    accounting.py              prefill (actual / no-cache) and decode token accounting, R.R.
  baselines/                   TATR / MinerU extractors, TAPAS / Qwen judges, the six modular pipelines (5.1)
  evaluation/                  micro P/R/F1, cell-count weighted TEDS on TP, composite P_c/R_c/F1, cost, per-category
  datasets/                    condition generation (A.1.5-A.1.7) and DegConTab synthesis (A.1.3-A.1.4)
experiments/                   one entry point per table / figure (see below)
deps/teds/                     official PubTabNet TEDS implementation
data/                          demo condition files (see Data)
corpus/                        demo corpora (see Data)
```

## Data

The bundled corpora under `corpus/` are a **demo subset** of the three
datasets (FinConTab, PubConTab, DegConTab; directory names `fintabnet`,
`pubtables`, `wild`). Each dataset directory holds

```
docs/<doc_id>/doc_meta.json, tables/*.png, table_html/*.html, table_ocr/*.txt
segment_json/<doc_id>_p<P>.json     per-page blocks {key: {type, bbox, ocr}}
partition/<doc_id>.json             the relevance index I(D), written by experiments.build_index (not bundled)
```

`data/demo_per_doc_<dataset>.json` maps documents to their conditions
(`condition_id`, `condition`, `gold_yes_tables`; a `category` field, when
present, drives the per-category scores of Table 8);
`data/demo_conditions_<dataset>.json` lists the demo subset of condition ids.
`experiments.build_index` writes the relevance index under `partition/`.

## Setup

```bash
pip install -r requirements.txt
export QWEN_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct      # or a local path
export NVEMBED_MODEL_PATH=nvidia/NV-Embed-v2
export PCRC_CORPUS_ROOT=$(pwd)/corpus                   # default
export PCRC_OUTPUT_ROOT=$(pwd)/outputs                  # default
```

Optional components: `paddleocr` (PP-Structure parsing of PubConTab pages;
word boxes for the TATR extractor), `pymupdf` (text layer of FinConTab /
DegConTab), `mineru` and the table-transformer weights for the modular
baselines, `vllm` for the generic prefix-cache comparison, `anthropic` /
`openai` clients for the API backbones.

## Reproducing the paper

| Paper | Command |
|---|---|
| Section 3.1 index | `python -m experiments.build_index --dataset fincontab` |
| Tables 2, 3, 7 | `python -m experiments.run_main --dataset fincontab --method {qwen,pcrc,pcrc_nocache,qwen+tatr,...}` |
| Table 8 (per category) | `python -m experiments.score --datasets fincontab pubcontab degcontab` |
| Table 9 / Figure 3 | `python -m experiments.run_ablations --dataset fincontab --variant {random,all,semantic_only,spatial_only,shared_only,exclusive_only,no_gating}` |
| Table 10 | `python -m experiments.run_sweeps --dataset fincontab --param {k_shared,k_excl,tau_shared,tau_excl}` |
| Tables 11, 12 | `python -m experiments.score_distribution` and `python -m experiments.run_sweeps --dataset fincontab --param lam` |
| Table 13 | `python -m experiments.run_backbones --dataset fincontab --backbone {claude-opus-4.5,gpt-5.2,qwen3-vl-plus,glm-4.5v}` |
| Table 14 | `python -m experiments.run_context_ablation --dataset degcontab --backbone gpt-5.2` |
| Table 9 (cache management) / A.3.5 | `python -m experiments.run_generic_cache --dataset fincontab --protocol {pcrc_schedule,single_judgment}` |
| Appendix A.1 | `python -m experiments.build_conditions`, `python -m experiments.synthesize_degcontab` |

Every method writes one JSONL record per (document, condition, table) under
`outputs/records/<dataset>/<method>.jsonl` with the predicted label, the
extracted HTML, the judgment rounds, and the token counts; `experiments.score`
aggregates them into judgment P/R/F1, composite P_c/R_c/F1, per-condition
prefill / decode / total tokens, and the recomputation ratio
R.R. = sum(prefill actual) / sum(prefill without cache reuse).
Add `--subset` to restrict any run to the demo condition subset.

## Inference contract

```python
from pcrc.inference.engine import VLMEngine
from pcrc.inference.pcrc import PCRCOptions, PCRCRunner

engine = VLMEngine()                                  # Qwen2.5-VL-7B-Instruct
runner = PCRCRunner(engine, sample_image, sample_html, PCRCOptions(reuse=True))
for outcome in runner.run_document(doc, index):       # doc / index from pcrc.data.corpus
    print(outcome.table_id, outcome.predicted, outcome.rounds, outcome.prefill_actual)
```

`PCRCOptions(reuse=False)` gives PCRC w/o cache, `gating=False` removes the
reliability gate, `single_judgment=True` supplies all selected evidence in one
call, and `evidence="shared_only" | "exclusive_only"` restricts the evidence
types.

## License

Code is released under the MIT License (`LICENSE`). The bundled TEDS source
retains its original Apache 2.0 license.
