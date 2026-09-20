"""Cache-frame engine for open-weight VLM backbones (Section 3.2).

The engine exposes the contract of Algorithm 1 on top of a HuggingFace
``DynamicCache``:

    frame = engine.new_frame(system)
    frame.prefill_head(segments)           # Pi_head = [sys; judge policy; q; S(D)], encoded once
    frame.append(tail_segment)             # C <- C_head (+) [image(T_i); ocr(T_i)]
    a = frame.generate("judge_1", ...)     # Judge(C); the request + answer kv states are dropped
    frame.append(exclusive_segment)        # C <- C (+) e_k
    ...
    frame.append(extraction_turn); html = frame.generate("extract", ...)
    frame.discard_tail()                   # back to C_head for the next table

Images live in the tail: a segment may carry images, and the engine encodes
the text and vision tokens of a segment in one incremental forward with the
backbone's own multimodal rotary positions (``get_rope_index`` of Qwen2.5-VL
and GLM-4.5V), so the table-image tokens are computed once per table and
reused across rounds and by the extraction turn. Chat markup and image
placeholders are taken from each backbone's chat template (``ChatGlue``).

With ``reuse=False`` the same API replays every call from scratch (PCRC w/o
cache): every generation re-encodes its full prefix, and the accountant
records exactly that.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from .. import config
from .accounting import TokenAccountant
from .prompts import Segment

_SENTINEL = "\x00PCRC_USER_CONTENT\x00"


def _load_backbone(model_path: str, device: str, dtype_name: str):
    import torch
    from transformers import AutoProcessor

    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    try:
        from transformers import AutoModelForImageTextToText as Cls
    except ImportError:  # older transformers: pick the architecture by name
        if "glm" in model_path.lower():
            from transformers import Glm4vMoeForConditionalGeneration as Cls
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    model = Cls.from_pretrained(model_path, torch_dtype=dtype,
                                attn_implementation=config.ATTN_IMPLEMENTATION, device_map=device).eval()
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=config.MIN_PIXELS, max_pixels=config.MAX_PIXELS)
    return model, processor


class ChatGlue:
    """Backbone-specific chat markup, derived from the processor's own chat template.

    Rendering a system + user turn around a sentinel yields the text that
    precedes the user content (``prefix``), the text that closes the turn and
    opens the assistant turn (``suffix``), and the placeholder the processor
    expands into vision tokens (``image``). For Qwen2.5-VL these are the ChatML
    markers and ``<|vision_start|><|image_pad|><|vision_end|>``; for GLM-4.5V
    the ``[gMASK]<sop><|system|>...<|user|>...<|assistant|>`` markers and
    ``<|begin_of_image|><|image|><|end_of_image|>``.
    """

    def __init__(self, processor, system: str = "SYSTEM"):
        self._processor = processor
        rendered = self._render(system, [{"type": "text", "text": _SENTINEL}])
        prefix, suffix = rendered.split(_SENTINEL)
        self._before_system, self._after_system = prefix.split(system, 1)
        self.suffix = suffix
        with_image = self._render(system, [{"type": "image"}, {"type": "text", "text": _SENTINEL}])
        self.image = with_image[len(prefix): with_image.index(_SENTINEL)]

    def _render(self, system: str, content: List[dict]) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        return self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def prefix(self, system: str) -> str:
        return self._before_system + system + self._after_system


class VLMEngine:
    """Model + processor + chat-template glue shared by all inference modes."""

    def __init__(self, model_path: str = config.QWEN_MODEL_PATH, device: str = "cuda:0"):
        self.model_path = model_path
        self.device = device
        self.model, self.processor = _load_backbone(model_path, device, config.DTYPE)
        self.tokenizer = self.processor.tokenizer
        self.glue = ChatGlue(self.processor)
        self.accountant = TokenAccountant()
        eos = self.model.generation_config.eos_token_id
        self.stop_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {self.tokenizer.eos_token_id}
        self.stop_ids.discard(None)

    def chat_prefix(self, system: str) -> str:
        """System turn plus the opening of the user turn, in the backbone's markup."""
        return self.glue.prefix(system)

    def generation_suffix(self) -> str:
        """Close the user turn and open the assistant turn."""
        return self.glue.suffix

    def encode_segment(self, segment: Segment) -> dict:
        """Tokenise one segment; images become vision tokens through the processor."""
        text = "".join(self.glue.image if it["type"] == "image" else it["text"] for it in segment.content())
        images = [it["image"] for it in segment.content() if it["type"] == "image"] or None
        inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=False)
        return {k: v.to(self.device) for k, v in inputs.items()}

    # ---------------------------------------------------- single-shot path
    def generate_once(self, messages: List[dict], max_new_tokens: int, label: str = "single_shot") -> str:
        """One fresh forward + greedy decode (Appendix A.2.1 baseline; no cache to reuse)."""
        import torch

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = [it["image"] for m in messages if isinstance(m["content"], list)
                  for it in m["content"] if it["type"] == "image"] or None
        inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=True).to(self.device)
        n_in = int(inputs.input_ids.shape[1])
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        gen = out[:, n_in:]
        self.accountant.encoded(n_in)
        self.accountant.complete_call(label, n_in, int(gen.shape[1]))
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0]

    def new_frame(self, system: str, reuse: bool = True) -> "CacheFrame":
        return CacheFrame(self, system, reuse=reuse)


class CacheFrame:
    """Fixed head + replaceable tail over one DynamicCache (Section 3.2)."""

    def __init__(self, engine: VLMEngine, system: str, reuse: bool = True):
        self.engine = engine
        self.reuse = reuse
        self.system = system
        self.segments: List[Segment] = []     # prefix segments after the chat prefix, in order
        self.head_segments = 0
        self._reset_cache()

    # ---------------------------------------------------------- low level
    def _reset_cache(self) -> None:
        import torch
        from transformers.cache_utils import DynamicCache

        self.cache = DynamicCache()
        self.length = 0
        self.head_length = 0
        self._ids = torch.zeros((1, 0), dtype=torch.long, device=self.engine.device)  # full sequence so far
        self._grids: List = []                                                        # image_grid_thw rows so far
        for obj in (self.engine.model, getattr(self.engine.model, "model", None)):
            if obj is not None and hasattr(obj, "rope_deltas"):
                obj.rope_deltas = None

    def _positions(self, new_ids, new_grids):
        """Multimodal rotary positions of ``new_ids`` given everything encoded before them."""
        import torch

        seq = torch.cat([self._ids, new_ids], dim=1)
        grids = self._grids + list(new_grids)
        core = getattr(self.engine.model, "model", self.engine.model)
        if hasattr(core, "get_rope_index"):
            pos, _ = core.get_rope_index(seq, torch.stack(grids) if grids else None, attention_mask=torch.ones_like(seq))
        else:
            pos = torch.arange(seq.shape[1], device=seq.device).view(1, 1, -1).expand(3, 1, -1)
        return pos[..., self.length:].contiguous(), seq

    def _forward(self, inputs: dict, count: bool = True):
        """Extend the cache with ``inputs`` (ids + optional pixels); return last-position logits."""
        import torch

        ids = inputs["input_ids"]
        n = int(ids.shape[1])
        grids = list(inputs["image_grid_thw"]) if "image_grid_thw" in inputs else []
        position_ids, seq = self._positions(ids, grids)
        kwargs = dict(input_ids=ids, position_ids=position_ids, past_key_values=self.cache, use_cache=True,
                      cache_position=torch.arange(self.length, self.length + n, device=ids.device),
                      attention_mask=torch.ones((1, self.length + n), device=ids.device, dtype=torch.long))
        if "pixel_values" in inputs:
            kwargs["pixel_values"], kwargs["image_grid_thw"] = inputs["pixel_values"], inputs["image_grid_thw"]
        with torch.no_grad():
            out = self.engine.model(**kwargs)
        self._ids, self._grids, self.length = seq, self._grids + grids, self.length + n
        if count:
            self.engine.accountant.encoded(n)
        return out.logits[:, -1, :]

    def _forward_segment(self, segment: Segment):
        return self._forward(self.engine.encode_segment(segment))

    def _forward_token(self, token_id: int):
        import torch
        ids = torch.tensor([[token_id]], dtype=torch.long, device=self.engine.device)
        return self._forward({"input_ids": ids}, count=False)   # decode steps are counted as decode tokens

    def _rebuild(self) -> None:
        """No-cache mode: re-encode the whole prefix from scratch before a call."""
        segs = list(self.segments)
        self._reset_cache()
        self._forward_segment(Segment.text("sys", self.engine.chat_prefix(self.system)))
        for s in segs:
            self._forward_segment(s)
        self.segments = segs

    # ------------------------------------------------------------ contract
    def prefill_head(self, head_segments: Sequence[Segment]) -> None:
        """Encode Pi_head once and remember its boundary (C_head).

        In no-cache mode nothing is encoded here: every call rebuilds its full
        prefix in ``generate``, which is exactly what the accountant must see.
        """
        self._reset_cache()
        self.segments = list(head_segments)
        self.head_segments = len(self.segments)
        if self.reuse:
            self._forward_segment(Segment.text("sys", self.engine.chat_prefix(self.system)))
            for s in head_segments:
                self._forward_segment(s)
            self.head_length = self.length

    def append(self, segment: Segment) -> None:
        """C <- C (+) segment (tail, exclusive paragraph, or extraction turn)."""
        self.segments.append(segment)
        if self.reuse:
            self._forward_segment(segment)

    def generate(self, label: str, max_new_tokens: int, request: Optional[Segment] = None) -> str:
        """Close the user turn, decode greedily, then drop the request and answer states.

        The request segment (judgment / final judgment / extraction request) is
        part of this call only; it is cropped together with the generated tokens
        so that the next round extends the same prefix (Algorithm 1, line 4).
        """
        import torch

        if not self.reuse:
            self._rebuild()
        restore_to = self.length
        if request is not None:
            self._forward_segment(request)
        logits = self._forward_segment(Segment.text("suffix", self.engine.generation_suffix()))
        prompt_len = self.length

        generated: List[int] = []
        for _ in range(max_new_tokens):
            nxt = int(torch.argmax(logits, dim=-1).item())
            if nxt in self.engine.stop_ids:
                break
            generated.append(nxt)
            logits = self._forward_token(nxt)
        text = self.engine.tokenizer.decode(generated, skip_special_tokens=True)
        self.engine.accountant.complete_call(label, prompt_len, len(generated))
        self.crop_to(restore_to)
        return text

    def crop_to(self, target_length: int) -> None:
        """Truncate the cache and its bookkeeping to ``target_length`` tokens."""
        if target_length >= self.length:
            return
        if hasattr(self.cache, "crop"):
            self.cache.crop(target_length)
        else:  # older cache layouts
            for layer in getattr(self.cache, "layers", []):
                layer.keys, layer.values = layer.keys[..., :target_length, :], layer.values[..., :target_length, :]
        self._ids = self._ids[:, :target_length]
        self.length = target_length
        # keep only the image grids whose tokens survive the crop
        image_token_id = getattr(self.engine.model.config, "image_token_id", None)
        if image_token_id is not None:
            remaining = int((self._ids == image_token_id).sum().item())
            merge = getattr(self.engine.processor.image_processor, "merge_size", 2) ** 2
            kept, acc = [], 0
            for g in self._grids:
                n_tok = int(g.prod().item()) // merge
                if acc + n_tok <= remaining:
                    kept.append(g)
                    acc += n_tok
            self._grids = kept

    def discard_tail(self) -> None:
        """Remove the table-specific tail; keep C_head (Algorithm 1, line 8)."""
        self.segments = self.segments[: self.head_segments]
        if self.reuse:
            self.crop_to(self.head_length)

    def close(self) -> None:
        self.cache = None
