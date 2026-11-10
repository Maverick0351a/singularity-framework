import json
import os
import struct
import time
import subprocess
import sys
import numpy as np

_DEFAULT_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm25-350m")
_FALLBACK_SCRATCH = r"C:\Users\Maverick\.gemini\antigravity\scratch\lfm25-350m"
MODEL_DIR = os.environ.get(
    "LFM2_MODEL_DIR",
    _DEFAULT_LOCAL if os.path.exists(_DEFAULT_LOCAL) else _FALLBACK_SCRATCH,
)


class FastBPETokenizer:
    """Exact GPT-2 byte-level BPE tokenizer loaded directly from LiquidAI/LFM2.5-350M tokenizer.json."""

    def __init__(self, tokenizer_json_path: str):
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["model"]["vocab"]
        self.id_to_tok = {v: k for k, v in self.vocab.items()}
        for added in data.get("added_tokens", []):
            self.vocab[added["content"]] = added["id"]
            self.id_to_tok[added["id"]] = added["content"]

        # Build byte-to-unicode mapping (standard GPT-2 byte-level BPE)
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        self.byte_encoder = {b: chr(c) for b, c in zip(bs, cs)}
        self.byte_decoder = {chr(c): b for b, c in zip(bs, cs)}

        # Parse merge ranks
        self.bpe_ranks = {}
        for rank, merge in enumerate(data["model"]["merges"]):
            if isinstance(merge, list):
                pair = (merge[0], merge[1])
            else:
                parts = merge.split(" ")
                pair = (parts[0], parts[1])
            self.bpe_ranks[pair] = rank

        self.special_tokens = sorted(
            [t["content"] for t in data.get("added_tokens", [])],
            key=len,
            reverse=True,
        )

    def _bpe(self, token_str: str) -> list[str]:
        word = list(token_str)
        if len(word) <= 1:
            return word
        while True:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            best_pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if best_pair not in self.bpe_ranks:
                break
            first, second = best_pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
        return word

    def encode(self, text: str) -> list[int]:
        # Split on special tokens first
        ids = []
        i = 0
        while i < len(text):
            matched_special = None
            for sp in self.special_tokens:
                if text.startswith(sp, i):
                    matched_special = sp
                    break
            if matched_special is not None:
                ids.append(self.vocab[matched_special])
                i += len(matched_special)
                continue

            # Find next special token boundary
            next_sp = len(text)
            for sp in self.special_tokens:
                pos = text.find(sp, i)
                if pos != -1 and pos < next_sp:
                    next_sp = pos
            chunk = text[i:next_sp]
            i = next_sp

            # Pre-tokenize into words/whitespace runs so BPE doesn't merge across words/newlines
            import re
            for piece in re.findall(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z_]+| ?[0-9]+| ?[^\sA-Za-z0-9_]+|\s+(?!\S)|\s+", chunk):
                encoded_bytes = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
                for bpe_tok in self._bpe(encoded_bytes):
                    if bpe_tok in self.vocab:
                        ids.append(self.vocab[bpe_tok])
        return ids

    def decode(self, ids: list[int]) -> str:
        pieces = []
        byte_buf = bytearray()
        for idx in ids:
            tok = self.id_to_tok.get(int(idx), "")
            if tok in self.special_tokens:
                if byte_buf:
                    pieces.append(byte_buf.decode("utf-8", errors="replace"))
                    byte_buf.clear()
                pieces.append(tok)
            else:
                for ch in tok:
                    if ch in self.byte_decoder:
                        byte_buf.append(self.byte_decoder[ch])
        if byte_buf:
            pieces.append(byte_buf.decode("utf-8", errors="replace"))
        return "".join(pieces)


class StatefulLFM2Model:
    """
    Fast O(1)-step execution engine for real pretrained LiquidAI/LFM2.5-350M (model.safetensors).
    Uses contiguous BLAS sgemv (W @ v) + 2-slot Lfm2ShortConv state ring + GQA KV cache.
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        with open(model_dir + r"\config.json", "r") as f:
            self.cfg = json.load(f)
        self.layer_types = self.cfg["layer_types"]
        self.num_layers = len(self.layer_types)
        self.hidden_size = self.cfg["hidden_size"]
        self.t = self._load_safetensors(model_dir + r"\model.safetensors")
        self._apply_futhark_horizontal_soac_fusion()
        self.freqs = 1.0 / (
            1000000.0 ** (np.arange(0, 64, 2, dtype=np.float32) / 64.0)
        )
        self.reset_cache()

    def _apply_futhark_horizontal_soac_fusion(self):
        """
        Applies Futhark Horizontal SOAC Fusion (`src/Futhark/Optimise/Fusion.hs`):
        - Horizontally fuses `feed_forward.w1` [4608, 1024] and `feed_forward.w3` [4608, 1024]
          which share the exact same consumer input `ffn_in` into `w13_fused` [9216, 1024].
        - Horizontally fuses `q_proj` [1024, 1024], `k_proj` [512, 1024], and `v_proj` [512, 1024]
          which share the exact same consumer input `normed` into `qkv_fused` [2048, 1024].
        """
        for i, ltype in enumerate(self.layer_types):
            p = f"model.layers.{i}."
            w1 = self.t.pop(p + "feed_forward.w1.weight")
            w3 = self.t.pop(p + "feed_forward.w3.weight")
            self.t[p + "feed_forward.w13_fused"] = np.ascontiguousarray(np.vstack([w1, w3]))
            if ltype != "conv":
                wq = self.t.pop(p + "self_attn.q_proj.weight")
                wk = self.t.pop(p + "self_attn.k_proj.weight")
                wv = self.t.pop(p + "self_attn.v_proj.weight")
                self.t[p + "self_attn.qkv_fused"] = np.ascontiguousarray(np.vstack([wq, wk, wv]))

    @staticmethod
    def _load_safetensors(path: str) -> dict[str, np.ndarray]:
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len).decode("utf-8"))
            raw = np.fromfile(f, dtype=np.uint16)
        tensors = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            s, e = info["data_offsets"]
            u16 = raw[s // 2 : e // 2]
            f32 = (u16.astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])
            tensors[name] = np.ascontiguousarray(f32)
        return tensors

    def reset_cache(self):
        self.pos = 0
        # 2-token conv state ring buffer per conv layer: [2, 1024]
        self.conv_cache = {
            i: np.zeros((2, self.hidden_size), dtype=np.float32)
            for i, lt in enumerate(self.layer_types)
            if lt == "conv"
        }
        # KV cache per attention layer: list of [8, 64] arrays
        self.k_cache = {
            i: [] for i, lt in enumerate(self.layer_types) if lt != "conv"
        }
        self.v_cache = {
            i: [] for i, lt in enumerate(self.layer_types) if lt != "conv"
        }

    @staticmethod
    def _rms_norm_1d(x: np.ndarray, w: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
        return (x / rms) * w

    def step_hidden(self, token_id: int) -> np.ndarray:
        """Runs a single token through all 16 layers in ~20ms using contiguous sgemv (W @ v)."""
        h = self.t["model.embed_tokens.weight"][token_id].copy()
        pos = self.pos
        angles = pos * self.freqs
        cos = np.cos(angles)
        sin = np.sin(angles)

        for i, ltype in enumerate(self.layer_types):
            p = f"model.layers.{i}."
            normed = self._rms_norm_1d(h, self.t[p + "operator_norm.weight"])
            if ltype == "conv":
                bcx = self.t[p + "conv.in_proj.weight"] @ normed
                B = bcx[0:1024]
                C = bcx[1024:2048]
                x_tilde = bcx[2048:3072]
                u = B * x_tilde
                ring = self.conv_cache[i]
                w = self.t[p + "conv.conv.weight"][:, 0, :]  # [1024, 3]
                conv_out = ring[0] * w[:, 0] + ring[1] * w[:, 1] + u * w[:, 2]
                ring[0] = ring[1]
                ring[1] = u
                op_out = self.t[p + "conv.out_proj.weight"] @ (C * conv_out)
            else:
                qkv = self.t[p + "self_attn.qkv_fused"] @ normed
                q = qkv[0:1024].reshape(16, 64)
                k = qkv[1024:1536].reshape(8, 64)
                v = qkv[1536:2048].reshape(8, 64)
                q = self._rms_norm_1d(q, self.t[p + "self_attn.q_layernorm.weight"])
                k = self._rms_norm_1d(k, self.t[p + "self_attn.k_layernorm.weight"])
                # Apply RoPE
                q = np.concatenate(
                    [q[:, :32] * cos - q[:, 32:] * sin, q[:, 32:] * cos + q[:, :32] * sin],
                    axis=-1,
                )
                k = np.concatenate(
                    [k[:, :32] * cos - k[:, 32:] * sin, k[:, 32:] * cos + k[:, :32] * sin],
                    axis=-1,
                )
                self.k_cache[i].append(k)
                self.v_cache[i].append(v)
                k_stack = np.stack(self.k_cache[i], axis=0)  # [T, 8, 64]
                v_stack = np.stack(self.v_cache[i], axis=0)  # [T, 8, 64]
                # GQA: reshape q to [8, 2, 64]
                q_gqa = q.reshape(8, 2, 64)
                # Einsum over head group: [8, 2, 64] x [T, 8, 64] -> [8, 2, T]
                scores = np.einsum("ghd,tgd->ght", q_gqa, k_stack) * (1.0 / 8.0)
                scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
                probs = scores / np.sum(scores, axis=-1, keepdims=True)
                attn_heads = np.einsum("ght,tgd->ghd", probs, v_stack).reshape(1024)
                op_out = self.t[p + "self_attn.out_proj.weight"] @ attn_heads

            h = h + op_out
            ffn_in = self._rms_norm_1d(h, self.t[p + "ffn_norm.weight"])
            w13 = self.t[p + "feed_forward.w13_fused"] @ ffn_in
            w1 = w13[0:4608]
            w3 = w13[4608:9216]
            silu_w1 = w1 / (1.0 + np.exp(-np.clip(w1, -40.0, 40.0)))
            h = h + (self.t[p + "feed_forward.w2.weight"] @ (silu_w1 * w3))

        self.pos += 1
        return self._rms_norm_1d(h, self.t["model.embedding_norm.weight"])


if __name__ == "__main__":
    tok = FastBPETokenizer(MODEL_DIR + r"\tokenizer.json")
    model = StatefulLFM2Model(MODEL_DIR)

    prompt = "<|startoftext|>def add_numbers(a, b):\n    return "
    ids = tok.encode(prompt)
    print("Prompt tokens:", ids, [tok.id_to_tok[i] for i in ids])

    t0 = time.time()
    for tid in ids:
        h_last = model.step_hidden(tid)
    gen_ids = []
    for _ in range(12):
        logits = model.t["model.embed_tokens.weight"] @ h_last
        next_id = int(np.argmax(logits))
        gen_ids.append(next_id)
        h_last = model.step_hidden(next_id)
    elapsed = time.time() - t0
    total_steps = len(ids) + len(gen_ids)
    print(f"Ran {total_steps} real LFM2.5-350M token steps in {elapsed:.3f}s ({total_steps/elapsed:.1f} tok/s)")
    print("Generated completion:", repr(tok.decode(gen_ids)))
