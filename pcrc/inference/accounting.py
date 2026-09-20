"""Token accounting (Section 5.3).

For every forward call we record

* ``prefill_actual``   tokens whose key/value states were actually computed
                       (cache misses: the head once, each tail once, each
                       appended paragraph once, the extraction turn once);
* ``prefill_nocache``  tokens the same call would encode if every call
                       re-encoded its full prefix (the no-cache reference);
* ``decode``           generated tokens.

Per condition the three quantities are summed over all calls (judgment
rounds and extraction included) and averaged over conditions in scoring.
The recomputation ratio is

    R.R. = sum(prefill_actual) / sum(prefill_nocache),

whose denominator is the prefill of the same call sequence, with the same
inputs and outputs, when every call re-encodes its full prefix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class CallRecord:
    label: str            # judge_1 | judge_2 | forced | extract | single_shot
    prefill_actual: int
    prefill_nocache: int
    decode: int


@dataclass
class TokenAccountant:
    calls: List[CallRecord] = field(default_factory=list)
    _pending_actual: int = 0        # tokens encoded since the last completed call

    def encoded(self, n_tokens: int) -> None:
        """Register newly computed prefix tokens (head / tail / paragraph / suffix)."""
        self._pending_actual += int(n_tokens)

    def complete_call(self, label: str, sequence_length: int, decode_tokens: int) -> CallRecord:
        """Close one generation call.

        ``sequence_length`` is the full prompt length at generation time; it is
        exactly what a no-cache run would prefill for this call.
        """
        rec = CallRecord(label, self._pending_actual, int(sequence_length), int(decode_tokens))
        self.calls.append(rec)
        self._pending_actual = 0
        return rec

    def totals(self) -> dict:
        return {
            "prefill_actual": sum(c.prefill_actual for c in self.calls),
            "prefill_nocache": sum(c.prefill_nocache for c in self.calls),
            "decode": sum(c.decode for c in self.calls),
            "n_calls": len(self.calls),
        }

    def reset(self) -> None:
        self.calls.clear()
        self._pending_actual = 0


def recomputation_ratio(prefill_actual: float, prefill_nocache: float) -> float:
    return 100.0 * prefill_actual / prefill_nocache if prefill_nocache else float("nan")
