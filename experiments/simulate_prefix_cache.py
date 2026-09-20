"""Offline simulation of a generic block-level prefix cache (vLLM-style) over a
logged sequence of VLM calls (Appendix A.3.5), giving the prefill actually
computed and the recomputation ratio R.R. = computed / no-cache prefill.

Input: a JSONL file, one request per line, in the order the calls were issued:

    {"doc": "d17", "cond": "c2", "table": "t3", "round": 1, "call": "judge",
     "segments": [["head:d17:c2", 812], ["img:t3", 1024], ["ocr:t3", 190],
                  ["para:p41", 143]]}

`segments` lists the prompt as (segment_id, token_count) pairs in prompt order.
A segment id names a piece of content that is byte-identical wherever it recurs
(the fixed head of a document-condition pair, a table image, a table's OCR, a
paragraph, the sample image/HTML, an instruction block).  Judgment answers are
NOT part of the next prompt (Algorithm 1 drops them), so they are not listed.
If you have real token ids, pass them instead with "tokens": [...]; segment
ids are then ignored.

The cache follows vLLM's automatic prefix caching: the prompt is cut into
blocks of --block-size tokens, each block's hash chains on the previous
block's hash (so a block is reusable only when the whole prefix before it is
identical), only complete blocks are reusable, and eviction is LRU over blocks
with --capacity-blocks (0 = unbounded).  --interleave N serves N documents
round-robin to emulate cache pressure.

Usage:
    python -m experiments.simulate_prefix_cache calls.jsonl
    python -m experiments.simulate_prefix_cache calls.jsonl --capacity-blocks 4000 --interleave 8
    python -m experiments.simulate_prefix_cache --selftest
"""
import argparse
import collections
import hashlib
import json
import sys


def tokens_of(req):
    if "tokens" in req:
        return [str(t) for t in req["tokens"]]
    toks = []
    for seg_id, n in req["segments"]:
        toks.extend(f"{seg_id}#{i}" for i in range(int(n)))
    return toks


def block_hashes(toks, block):
    """Chained hashes of the complete blocks of a token list."""
    out, prev = [], ""
    for b in range(len(toks) // block):
        chunk = "|".join(toks[b * block:(b + 1) * block])
        prev = hashlib.sha1((prev + "||" + chunk).encode()).hexdigest()
        out.append(prev)
    return out


def interleave(reqs, n):
    """Round-robin the requests of n documents at a time (cache-pressure setting)."""
    by_doc = collections.OrderedDict()
    for r in reqs:
        by_doc.setdefault(r["doc"], []).append(r)
    docs = list(by_doc.values())
    out = []
    for g in range(0, len(docs), n):
        group = [collections.deque(d) for d in docs[g:g + n]]
        while any(group):
            for q in group:
                if q:
                    out.append(q.popleft())
    return out


def simulate(reqs, block=16, capacity=0):
    cache = collections.OrderedDict()   # block hash -> None, LRU order
    total = computed = 0
    per_doc = collections.defaultdict(lambda: [0, 0])
    for r in reqs:
        toks = tokens_of(r)
        hashes = block_hashes(toks, block)
        hit = 0
        for h in hashes:                 # longest cached prefix of complete blocks
            if h in cache:
                hit += 1
                cache.move_to_end(h)
            else:
                break
        n_tot = len(toks)
        n_comp = n_tot - hit * block
        for h in hashes[hit:]:           # insert newly computed blocks
            cache[h] = None
            cache.move_to_end(h)
            if capacity and len(cache) > capacity:
                cache.popitem(last=False)
        total += n_tot
        computed += n_comp
        per_doc[r["doc"]][0] += n_tot
        per_doc[r["doc"]][1] += n_comp
    return total, computed, per_doc


def selftest():
    """Two documents x two tables x (2 judge rounds + 1 extraction)."""
    reqs = []
    for d in ("d1", "d2"):
        head = [(f"head:{d}", 800)]
        for t in ("t1", "t2"):
            tail = [(f"img:{d}:{t}", 1024), (f"ocr:{d}:{t}", 200)]
            reqs.append({"doc": d, "cond": "c", "table": t, "round": 1, "call": "judge",
                         "segments": head + tail})
            reqs.append({"doc": d, "cond": "c", "table": t, "round": 2, "call": "judge",
                         "segments": head + tail + [(f"para:{d}:{t}:1", 150)]})
            reqs.append({"doc": d, "cond": "c", "table": t, "round": 2, "call": "extract",
                         "segments": head + tail + [(f"para:{d}:{t}:1", 150),
                                                    ("sample", 900)]})
    tot, comp, _ = simulate(reqs)
    print(f"selftest: no-cache prefill {tot} tokens, computed {comp}, R.R. {100*comp/tot:.1f}%")
    # exact expectation: head once per doc (800), tail once per table (1224),
    # paragraph once (150), sample once per extraction (900) => 2*800 + 4*1224 + 4*150 + 4*900
    exact = 2 * 800 + 4 * 1224 + 4 * 150 + 4 * 900
    print(f"          ideal explicit-cache computed {exact} ({100*exact/tot:.1f}%); "
          f"block-granularity overhead {comp-exact} tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="?")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--capacity-blocks", type=int, default=0, help="0 = unbounded")
    ap.add_argument("--interleave", type=int, default=1, help="documents served round-robin")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.jsonl:
        selftest()
        return
    reqs = [json.loads(l) for l in open(a.jsonl, encoding="utf-8") if l.strip()]
    if a.interleave > 1:
        reqs = interleave(reqs, a.interleave)
    tot, comp, per_doc = simulate(reqs, a.block_size, a.capacity_blocks)
    n_cond = len({(r["doc"], r["cond"]) for r in reqs})
    print(f"requests {len(reqs)}  conditions {n_cond}  block {a.block_size}  "
          f"capacity {a.capacity_blocks or 'inf'} blocks  interleave {a.interleave}")
    print(f"no-cache prefill per condition {tot/n_cond/1000:.2f}k   "
          f"computed per condition {comp/n_cond/1000:.2f}k   R.R. {100*comp/tot:.2f}%")


if __name__ == "__main__":
    sys.exit(main())
