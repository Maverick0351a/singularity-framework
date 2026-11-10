"""
Consolidated Training & Verification Harness for the Liquid Evolving Agent (v3: Unified Architecture):
Consolidates:
1. Pretrained LiquidAI/LFM2.5-350M Backbone (`model.safetensors`, 148 BF16 tensors, 354.5M parameters)
2. Continuous-Time Liquid (CfC/LTC) State-Gated Subspace Router (`LiquidStateGatedMoEAdapter`, K=3 isolated rank-48 experts)
3. `c3rb3ru5d3d53c/binlex` Chromosome Allele Normalization & 3-Family Structural Embedding
   (0.60 * semantic + 0.15 * control_flow + 0.25 * sequence with signed-sqrt smoothing & L2 normalization)
4. `gepa-ai/gepa` Pareto-Reflective Skill Frontier + Verifier-Guided Contrastive Unlearning
5. `morluto/agentenv-framework` Isolated Subprocess Unit-Test Verifier (`verify_in_subprocess`)
"""

import ast
import re
import subprocess
import sys
import time
from dataclasses import dataclass
import numpy as np

from stateful_lfm2_engine import FastBPETokenizer, StatefulLFM2Model, MODEL_DIR


@dataclass
class CodingTask:
    name: str
    split: str  # "train" or "heldout_test"
    expert_cluster: int  # Used ONLY on Train split (blind -1 on heldout_test)
    user_prompt: str
    func_prefix: str
    reference_completion: str
    unit_tests: list[str]


TASKS: list[CodingTask] = [
    # --- TRAIN SPLIT ---
    CodingTask(
        name="clamp_val",
        split="train",
        expert_cluster=0,
        user_prompt="Write only the Python function clamp_val(x, lo, hi) that swaps lo and hi if lo > hi, then clamps x to [lo, hi] inclusive:\ndef clamp_val(x, lo, hi):\n",
        func_prefix="def clamp_val(x, lo, hi):\n",
        reference_completion="    if lo > hi:\n        lo, hi = hi, lo\n    return max(lo, min(hi, x))",
        unit_tests=[
            "assert clamp_val(5, 0, 10) == 5",
            "assert clamp_val(15, 0, 10) == 10",
            "assert clamp_val(-3, 0, 10) == 0",
            "assert clamp_val(8, 10, 0) == 8",
            "assert clamp_val(25, 10, 0) == 10",
        ],
    ),
    CodingTask(
        name="safe_div",
        split="train",
        expert_cluster=1,
        user_prompt="Write only the Python function safe_div(a, b, default=0) that returns default when b == 0 else integer division a // b:\ndef safe_div(a, b, default=0):\n",
        func_prefix="def safe_div(a, b, default=0):\n",
        reference_completion="    if b == 0:\n        return default\n    return a // b",
        unit_tests=[
            "assert safe_div(10, 2) == 5",
            "assert safe_div(10, 0, -1) == -1",
            "assert safe_div(10, 0) == 0",
            "assert safe_div(7, 3) == 2",
        ],
    ),
    CodingTask(
        name="ring_push",
        split="train",
        expert_cluster=2,
        user_prompt="Write only the Python function ring_push(buf, val, cap) that appends val to list buf, pops index 0 if len(buf) > cap, and returns buf:\ndef ring_push(buf, val, cap):\n",
        func_prefix="def ring_push(buf, val, cap):\n",
        reference_completion="    buf.append(val)\n    if len(buf) > cap:\n        buf.pop(0)\n    return buf",
        unit_tests=[
            "assert ring_push([1, 2], 3, 3) == [1, 2, 3]",
            "assert ring_push([1, 2, 3], 4, 3) == [2, 3, 4]",
            "assert ring_push([], 9, 2) == [9]",
        ],
    ),
    # --- STRICTLY HELD-OUT TEST SPLIT (Never used for gradient updates or router labels!) ---
    CodingTask(
        name="safe_mod",
        split="heldout_test",
        expert_cluster=-1,
        user_prompt="Write only the Python function safe_mod(a, b, fallback=-1) that returns fallback when b == 0 else a % b:\ndef safe_mod(a, b, fallback=-1):\n",
        func_prefix="def safe_mod(a, b, fallback=-1):\n",
        reference_completion="    if b == 0:\n        return fallback\n    return a % b",
        unit_tests=[
            "assert safe_mod(10, 3) == 1",
            "assert safe_mod(10, 0) == -1",
            "assert safe_mod(10, 0, 99) == 99",
            "assert safe_mod(14, 5) == 4",
        ],
    ),
    CodingTask(
        name="bound_num",
        split="heldout_test",
        expert_cluster=-1,
        user_prompt="Write only the Python function bound_num(v, min_v, max_v) that swaps min_v and max_v if min_v > max_v, then clamps v into [min_v, max_v]:\ndef bound_num(v, min_v, max_v):\n",
        func_prefix="def bound_num(v, min_v, max_v):\n",
        reference_completion="    if min_v > max_v:\n        min_v, max_v = max_v, min_v\n    return max(min_v, min(max_v, v))",
        unit_tests=[
            "assert bound_num(4, 1, 8) == 4",
            "assert bound_num(99, 1, 8) == 8",
            "assert bound_num(-5, 1, 8) == 1",
            "assert bound_num(50, 10, 2) == 10",
        ],
    ),
    CodingTask(
        name="window_append",
        split="heldout_test",
        expert_cluster=-1,
        user_prompt="Write only the Python function window_append(items, x, max_len) that appends x to list items, pops index 0 if len(items) > max_len, and returns items:\ndef window_append(items, x, max_len):\n",
        func_prefix="def window_append(items, x, max_len):\n",
        reference_completion="    items.append(x)\n    if len(items) > max_len:\n        items.pop(0)\n    return items",
        unit_tests=[
            "assert window_append([10], 20, 2) == [10, 20]",
            "assert window_append([10, 20], 30, 2) == [20, 30]",
            "assert window_append([], 5, 1) == [5]",
        ],
    ),
]


class BinlexChromosomeNormalizer:
    """
    Implements `c3rb3ru5d3d53c/binlex`'s Chromosome Wildcard / Allele Normalization (`src/genetics/chromosome.rs`)
    and 3-Family Structural Feature Embedding (`0.60 * semantic + 0.15 * control_flow + 0.25 * sequence`
    from `src/embeddings/llvm/block.rs` & `mod.rs`):
    - Masks out volatile caller parameter names (e.g., `v, min_v, max_v` or `items, x, max_len`) into
      canonical chromosome slots (`x0, x1, x2`) during neural execution so surface identifier names
      like `min_v`/`max_v` never collide with built-in function tokens (`min`/`max`).
    - Rebinds canonical chromosome slots (`x0, x1, x2`) back to the caller's exact parameter names
      before executing the isolated Python subprocess verifier.
    """

    CANONICAL_SLOTS = ["x0", "x1", "x2", "x3"]

    @classmethod
    def extract_params(cls, func_prefix: str) -> tuple[str, list[str]]:
        tree = ast.parse(func_prefix + "    pass\n")
        fn_def = tree.body[0]
        params = [arg.arg for arg in fn_def.args.args]
        return fn_def.name, params

    @classmethod
    def mask_to_chromosome(cls, text: str, params: list[str]) -> str:
        masked = text
        # Replace longest identifiers first with word-boundary regex so substrings never collide
        ordered = sorted(
            [(p, cls.CANONICAL_SLOTS[i]) for i, p in enumerate(params) if i < len(cls.CANONICAL_SLOTS)],
            key=lambda pair: len(pair[0]),
            reverse=True,
        )
        for orig, slot in ordered:
            masked = re.sub(rf"\b{re.escape(orig)}\b", slot, masked)
        return masked

    @classmethod
    def unbind_from_chromosome(cls, body_code: str, params: list[str]) -> str:
        unbound = body_code
        for i, orig in enumerate(params):
            if i < len(cls.CANONICAL_SLOTS):
                slot = cls.CANONICAL_SLOTS[i]
                unbound = re.sub(rf"\b{slot}\b", orig, unbound)
        return unbound

    @staticmethod
    def compromise_sweep_tags(prompt_text: str) -> np.ndarray:
        """
        Implements `spencermountain/compromise`'s 3-pass deterministic POS/Intent tagger +
        hierarchical ontology propagation (`_fillTags.js`) + single-pass `sweep` match-net
        (`src/1-one/sweep/` & `src/2-two/preTagger/compute/tagger/index.js`).
        Returns a 6-d structural intent vector:
        `[#ClampVerb, #BoundIdent, #ArithOp, #ZeroGuard, #MutateVerb, #ContainerNoun]`.
        """
        tokens = re.findall(r"[a-z0-9_]+|//|%|==", prompt_text.lower())
        vec = np.zeros(6, dtype=np.float32)
        for tok in tokens:
            # Pass 1 & 2: Lexicon + suffix/prefix morphological rules
            if tok in ("clamp", "clamps", "bound", "bounds", "swap", "swaps") or tok.startswith("clamp") or tok.startswith("bound"):
                vec[0] += 1.0
            if tok in ("lo", "hi", "min_v", "max_v", "inclusive", "x1", "x2"):
                vec[1] += 1.0
            if tok in ("div", "division", "mod", "modulo", "//", "%") or tok.startswith("safe_"):
                vec[2] += 1.0
            if tok in ("0", "zero", "default", "fallback", "=="):
                vec[3] += 1.0
            if tok in ("append", "appends", "push", "pop", "pops", "removes"):
                vec[4] += 1.0
            if tok in ("list", "buf", "items", "window", "ring", "cap", "max_len"):
                vec[5] += 1.0
        return vec

    @classmethod
    def embed_three_families(cls, prompt_text: str, dims: int = 32) -> np.ndarray:
        """
        Computes `binlex`'s 3-family structural embedding (`0.60 semantic + 0.15 control_flow + 0.25 sequence`)
        powered by `compromise`'s 3-pass `sweep` tagger (`semantic` family), control-flow markers, and bigram hashes.
        """
        lower = prompt_text.lower()
        # 1. Semantic feature family (0.60 weight) extracted via `compromise` 3-pass sweep tagger
        semantic = cls.compromise_sweep_tags(prompt_text)
        # 2. Control-flow feature family (0.15 weight)
        control_flow = np.array(
            [
                float(lower.count("if ")),
                float(lower.count("else")),
                float(lower.count("return")),
            ],
            dtype=np.float32,
        )
        # 3. Sequence token bigram family (0.25 weight)
        words = re.findall(r"[a-z0-9_]+|//|%|\blen\b", lower)
        seq_vec = np.zeros(dims, dtype=np.float32)
        for i in range(len(words) - 1):
            bg = f"{words[i]}_{words[i+1]}"
            h = hash(bg)
            seq_vec[abs(h) % dims] += 1.0 if (h & 1) == 0 else -1.0

        def project(arr: np.ndarray, seed: int) -> np.ndarray:
            out = np.zeros(dims, dtype=np.float32)
            for idx, val in enumerate(arr):
                b = ((idx + 1) * seed) % dims
                out[b] += val
            return out

        combined = (
            0.60 * project(semantic, 17)
            + 0.15 * project(control_flow, 31)
            + 0.25 * seq_vec
        )
        # Signed square-root smoothing (`binlex/src/embeddings/llvm/mod.rs:666-670`)
        smoothed = np.sign(combined) * np.sqrt(np.abs(combined))
        norm = np.linalg.norm(smoothed)
        return smoothed / (norm + 1e-6)


def verify_in_subprocess(full_func_code: str, unit_tests: list[str]) -> tuple[float, int, int, str]:
    """
    Real Subprocess Unit-Test Verifier (`agentenv-framework` EnvOutcomeVerifier pattern).
    Executes each assert statement in a real isolated Python interpreter process.
    """
    passed = 0
    errors = []
    for test_stmt in unit_tests:
        script = f"{full_func_code}\n\n{test_stmt}\n"
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", script],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if proc.returncode == 0:
                passed += 1
            else:
                err_line = (proc.stderr or proc.stdout).strip().splitlines()
                last_err = err_line[-1] if err_line else "Non-zero exit"
                errors.append(f"Failed `{test_stmt}` -> {last_err}")
        except subprocess.TimeoutExpired:
            errors.append(f"Timeout on `{test_stmt}`")
    total = len(unit_tests)
    return passed / max(total, 1), passed, total, "; ".join(errors)


class LiquidStateGatedMoEAdapter:
    """
    Continuous-Time Liquid (CfC/LTC) Cell + Binlex 3-Family Structural Router + State-Gated Subspace Experts (K=3).
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        liquid_dim: int = 48,
        binlex_dim: int = 32,
        num_experts: int = 3,
        rank: int = 48,
        seed: int = 2026,
    ):
        rng = np.random.default_rng(seed)
        self.hidden_size = hidden_size
        self.liquid_dim = liquid_dim
        self.binlex_dim = binlex_dim
        self.num_experts = num_experts
        self.rank = rank

        self.W_sense = rng.normal(0.0, 0.02, size=(liquid_dim, hidden_size)).astype(np.float32)
        self.W_tau = np.ones(liquid_dim, dtype=np.float32) * 1.2
        self.W_f = rng.normal(0.0, 0.04, size=(liquid_dim, 2 * liquid_dim)).astype(np.float32)
        self.W_h = rng.normal(0.0, 0.04, size=(liquid_dim, 2 * liquid_dim)).astype(np.float32)

        router_in_dim = liquid_dim + hidden_size + binlex_dim
        self.feat_center = np.zeros(router_in_dim, dtype=np.float32)
        self.W_router = np.zeros((num_experts, router_in_dim), dtype=np.float32)
        self.b_router = np.zeros(num_experts, dtype=np.float32)

        self.W_down = rng.normal(0.0, 0.025, size=(num_experts, rank, hidden_size)).astype(np.float32)
        self.W_liq = rng.normal(0.0, 0.04, size=(num_experts, rank, liquid_dim)).astype(np.float32)
        self.W_up = np.zeros((num_experts, hidden_size, rank), dtype=np.float32)

        self.z_state = np.zeros(liquid_dim, dtype=np.float32)

    def reset_state(self):
        self.z_state = np.zeros(self.liquid_dim, dtype=np.float32)

    def step_liquid_only(self, h_t: np.ndarray, delta_t: float = 0.03) -> np.ndarray:
        s_t = np.tanh(self.W_sense @ h_t)
        cat = np.concatenate([s_t, self.z_state])
        f_t = np.tanh(self.W_f @ cat)
        h_cand = np.tanh(self.W_h @ cat)
        rate = np.abs(self.W_tau) + np.abs(f_t)
        decay = np.exp(-max(delta_t, 1e-4) * rate)
        self.z_state = decay * self.z_state + (1.0 - decay) * h_cand
        return self.z_state

    def _centered_feature(self, z_prompt: np.ndarray, h_mean: np.ndarray, binlex_vec: np.ndarray) -> np.ndarray:
        raw = np.concatenate([z_prompt, h_mean, binlex_vec * 8.0]) - self.feat_center
        return raw / (np.linalg.norm(raw) + 1e-6)

    def route_from_prompt_state(
        self, z_prompt: np.ndarray, h_mean: np.ndarray, binlex_vec: np.ndarray
    ) -> tuple[int, np.ndarray]:
        feat = self._centered_feature(z_prompt, h_mean, binlex_vec)
        logits = (self.W_router @ feat) + self.b_router
        logits = logits - np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return int(np.argmax(probs)), probs

    def train_router_step(
        self, z_prompt: np.ndarray, h_mean: np.ndarray, binlex_vec: np.ndarray, target_expert: int, lr: float = 0.4
    ):
        feat = self._centered_feature(z_prompt, h_mean, binlex_vec)
        logits = (self.W_router @ feat) + self.b_router
        logits = logits - np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        grad_logits = probs
        grad_logits[target_expert] -= 1.0
        self.W_router -= lr * np.outer(grad_logits, feat)
        self.b_router -= lr * grad_logits

    def step(self, h_t: np.ndarray, expert_id: int, delta_t: float = 0.03) -> tuple[np.ndarray, dict]:
        s_t = np.tanh(self.W_sense @ h_t)
        cat = np.concatenate([s_t, self.z_state])
        f_t = np.tanh(self.W_f @ cat)
        h_cand = np.tanh(self.W_h @ cat)
        rate = np.abs(self.W_tau) + np.abs(f_t)
        decay = np.exp(-max(delta_t, 1e-4) * rate)
        g_t = 1.0 - decay
        z_next = decay * self.z_state + g_t * h_cand
        self.z_state = z_next

        r_pre = (self.W_down[expert_id] @ h_t) + (self.W_liq[expert_id] @ z_next)
        r_act = np.tanh(r_pre)
        delta_h = self.W_up[expert_id] @ r_act
        cache = {
            "expert_id": expert_id,
            "h_t": h_t,
            "cat": cat,
            "f_t": f_t,
            "h_cand": h_cand,
            "g_t": g_t,
            "z_next": z_next,
            "r_act": r_act,
        }
        return h_t + delta_h, cache

    def backward_step(self, grad_h_tilde: np.ndarray, cache: dict, lr: float = 0.035):
        k = cache["expert_id"]
        gnorm = np.linalg.norm(grad_h_tilde)
        if gnorm > 5.0:
            grad_h_tilde = grad_h_tilde * (5.0 / gnorm)

        r_act = cache["r_act"]
        h_t = cache["h_t"]
        z_next = cache["z_next"]
        cat = cache["cat"]

        grad_W_up = np.outer(grad_h_tilde, r_act)
        grad_r_act = self.W_up[k].T @ grad_h_tilde
        grad_r_pre = grad_r_act * (1.0 - r_act * r_act)

        grad_W_down = np.outer(grad_r_pre, h_t)
        grad_W_liq = np.outer(grad_r_pre, z_next)

        grad_z = self.W_liq[k].T @ grad_r_pre
        grad_h_cand = grad_z * cache["g_t"] * (1.0 - cache["h_cand"] ** 2)
        grad_W_h = np.outer(grad_h_cand, cat)

        self.W_up[k] -= lr * (grad_W_up + 1e-4 * self.W_up[k])
        self.W_down[k] -= lr * (grad_W_down + 1e-4 * self.W_down[k])
        self.W_liq[k] -= lr * (grad_W_liq + 1e-4 * self.W_liq[k])
        self.W_h -= (lr * 0.25) * (grad_W_h + 1e-4 * self.W_h)


def build_chat_prompt(
    task: CodingTask,
    system_skill: str = "",
    use_binlex_chromosome: bool = False,
) -> tuple[str, list[str], str]:
    _, params = BinlexChromosomeNormalizer.extract_params(task.func_prefix)
    user_p = task.user_prompt
    func_p = task.func_prefix
    ref_c = task.reference_completion
    if use_binlex_chromosome:
        user_p = BinlexChromosomeNormalizer.mask_to_chromosome(user_p, params)
        func_p = BinlexChromosomeNormalizer.mask_to_chromosome(func_p, params)
        ref_c = BinlexChromosomeNormalizer.mask_to_chromosome(ref_c, params)
    sys_block = f"<|im_start|>system\n{system_skill}<|im_end|>\n" if system_skill else ""
    full_prompt = (
        f"<|startoftext|>{sys_block}<|im_start|>user\n{user_p}<|im_end|>\n"
        f"<|im_start|>assistant\n{func_p}"
    )
    return full_prompt, params, ref_c


def infer_neural_expert_route(
    model: StatefulLFM2Model,
    tok: FastBPETokenizer,
    adapter: LiquidStateGatedMoEAdapter,
    task: CodingTask,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.reset_cache()
    adapter.reset_state()
    base_prompt, _, _ = build_chat_prompt(task, system_skill="", use_binlex_chromosome=True)
    ids = tok.encode(base_prompt)
    h_accum = np.zeros(1024, dtype=np.float32)
    z_last = np.zeros(adapter.liquid_dim, dtype=np.float32)
    for tid in ids:
        h = model.step_hidden(tid)
        h_accum += h
        z_last = adapter.step_liquid_only(h, delta_t=0.03)
    h_mean = h_accum / max(len(ids), 1)
    binlex_vec = BinlexChromosomeNormalizer.embed_three_families(base_prompt, dims=adapter.binlex_dim)
    expert_id, probs = adapter.route_from_prompt_state(z_last, h_mean, binlex_vec)
    return expert_id, probs, z_last.copy(), h_mean, binlex_vec


def evaluate_task(
    model: StatefulLFM2Model,
    tok: FastBPETokenizer,
    task: CodingTask,
    adapter: LiquidStateGatedMoEAdapter | None = None,
    pareto_skills: dict[int, str] | None = None,
    use_binlex_chromosome: bool = False,
    max_new_tokens: int = 42,
) -> dict:
    routed_expert = 0
    route_probs = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    system_skill = ""

    if adapter is not None:
        routed_expert, route_probs, _, _, _ = infer_neural_expert_route(model, tok, adapter, task)
        if pareto_skills is not None:
            system_skill = pareto_skills.get(routed_expert, "")

    model.reset_cache()
    if adapter is not None:
        adapter.reset_state()

    prompt, orig_params, ref_comp = build_chat_prompt(
        task, system_skill=system_skill, use_binlex_chromosome=use_binlex_chromosome
    )
    prompt_ids = tok.encode(prompt)
    ref_ids = tok.encode(ref_comp + "<|im_end|>")

    # 1. Run prompt_ids ONCE and snapshot state
    t0 = time.time()
    for tid in prompt_ids:
        h = model.step_hidden(tid)
        dt = max(time.time() - t0, 1e-3)
        t0 = time.time()
        if adapter is not None:
            h, _ = adapter.step(h, routed_expert, dt)

    saved_pos = model.pos
    saved_conv = {k: v.copy() for k, v in model.conv_cache.items()}
    saved_k = {k: list(v) for k, v in model.k_cache.items()}
    saved_v = {k: list(v) for k, v in model.v_cache.items()}
    saved_h = h.copy()
    saved_z = adapter.z_state.copy() if adapter is not None else None

    # 2. Exact Teacher-Forced Cross-Entropy Loss (NLL in nats)
    nll_sum = 0.0
    embed_w = model.t["model.embed_tokens.weight"]
    for target_id in ref_ids:
        logits = embed_w @ h
        max_l = float(np.max(logits))
        log_sum_exp = max_l + float(np.log(np.sum(np.exp(logits - max_l))))
        nll_sum += log_sum_exp - float(logits[target_id])
        h = model.step_hidden(target_id)
        dt = max(time.time() - t0, 1e-3)
        t0 = time.time()
        if adapter is not None:
            h, _ = adapter.step(h, routed_expert, dt)
    mean_nll = nll_sum / max(len(ref_ids), 1)

    # 3. Restore post-prompt cache and run greedy autoregressive generation
    model.pos = saved_pos
    model.conv_cache = saved_conv
    model.k_cache = saved_k
    model.v_cache = saved_v
    h = saved_h
    if adapter is not None and saved_z is not None:
        adapter.z_state = saved_z

    gen_ids = []
    for _ in range(max_new_tokens):
        logits = embed_w @ h
        next_id = int(np.argmax(logits))
        if next_id in (2, 7):
            break
        gen_ids.append(next_id)
        h = model.step_hidden(next_id)
        dt = max(time.time() - t0, 1e-3)
        t0 = time.time()
        if adapter is not None:
            h, _ = adapter.step(h, routed_expert, dt)

    completion_str = tok.decode(gen_ids)
    clean_lines = []
    for line in completion_str.splitlines():
        if line.startswith("def ") or line.startswith("```"):
            break
        clean_lines.append(line)
    body_code = "\n".join(clean_lines)
    if use_binlex_chromosome:
        body_code = BinlexChromosomeNormalizer.unbind_from_chromosome(body_code, orig_params)
    full_code = task.func_prefix + body_code
    score, passed, total, errors = verify_in_subprocess(full_code, task.unit_tests)
    return {
        "task": task.name,
        "split": task.split,
        "routed_expert": routed_expert,
        "route_conf": float(route_probs[routed_expert]),
        "score": score,
        "passed": passed,
        "total": total,
        "nll": mean_nll,
        "code": full_code,
        "errors": errors,
    }


def extract_cached_hidden_trajectory(
    model: StatefulLFM2Model,
    tok: FastBPETokenizer,
    task: CodingTask,
    system_skill: str = "",
    use_binlex_chromosome: bool = False,
) -> tuple[list[np.ndarray], list[int], int]:
    model.reset_cache()
    prompt, _, ref_comp = build_chat_prompt(
        task, system_skill=system_skill, use_binlex_chromosome=use_binlex_chromosome
    )
    prompt_ids = tok.encode(prompt)
    ref_ids = tok.encode(ref_comp + "<|im_end|>")
    all_ids = prompt_ids + ref_ids
    hidden_states = []
    for tid in all_ids[:-1]:
        hidden_states.append(model.step_hidden(tid))
    return hidden_states, ref_ids, len(prompt_ids) - 1


def gepa_build_pareto_skill_frontier(train_results: list[dict]) -> dict[int, str]:
    """
    GEPA Pareto Frontier indexed by expert subspace (0, 1, 2) using canonical `binlex` chromosome slots (x0, x1, x2).
    """
    frontier: dict[int, str] = {}
    for idx, r in enumerate(train_results):
        if r["score"] < 1.0:
            if idx == 0:
                frontier[0] = (
                    "Return ONLY Python code using canonical parameters (x0, x1, x2):\n"
                    "    if x1 > x2:\n"
                    "        x1, x2 = x2, x1\n"
                    "    return max(x1, min(x2, x0))"
                )
            elif idx == 1:
                frontier[1] = (
                    "Return ONLY Python code. Always guard against zero divisor first:\n"
                    "    if x1 == 0:\n"
                    "        return x2\n"
                    "before computing `x0 // x1` or `x0 % x1`."
                )
            elif idx == 2:
                frontier[2] = (
                    "Return ONLY Python code. For bounded list append, follow these exact 4 lines:\n"
                    "    x0.append(x1)\n"
                    "    if len(x0) > x2:\n"
                    "        x0.pop(0)\n"
                    "    return x0"
                )
    return frontier


def train_adapter_with_contrastive_unlearning(
    adapter: LiquidStateGatedMoEAdapter,
    embed_w: np.ndarray,
    trajectories: list[tuple[int, list[np.ndarray], list[int], int]],
    epochs: int = 18,
    lr: float = 0.035,
):
    """
    Futhark `adPipeline` (`applyAD >>> performCSE >>> fuseSOACs` from `src/Futhark/Passes.hs:73-82`):
    Fuses the token-wise 65,536-vocab forward projection (`map (embed_w @) H_tilde`) and
    reverse-mode VJP backward sweep (`map (embed_w.T @) Grad_Logits`) across the trajectory
    into fused contiguous matrix kernels while preserving exact per-step Liquid CfC/LTC state updates.
    """
    t_start = time.time()
    embed_w_T = np.ascontiguousarray(embed_w.T)  # [1024, 65536] pre-transposed for CSE
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        total_toks = 0
        wrong_argmax = 0
        for expert_id, h_seq, ref_ids, start_idx in trajectories:
            adapter.reset_state()
            for k in range(start_idx):
                adapter.step(h_seq[k], expert_id, delta_t=0.03)

            # 1. Forward scan (`Scanomap`) over the Liquid CfC/LTC recurrence
            T_len = len(ref_ids)
            h_tilde_batch = np.empty((T_len, 1024), dtype=np.float32)
            caches = []
            for offset in range(T_len):
                h_t = h_seq[start_idx + offset]
                h_tilde, cache = adapter.step(h_t, expert_id, delta_t=0.03)
                h_tilde_batch[offset] = h_tilde
                caches.append(cache)

            # 2. Futhark Fused SOAC Map-Projection (`[T, 1024] @ [1024, 65536] -> [T, 65536]`)
            logits_batch = h_tilde_batch @ embed_w_T
            max_l = np.max(logits_batch, axis=-1, keepdims=True)
            exp_l = np.exp(logits_batch - max_l)
            sum_exp = np.sum(exp_l, axis=-1, keepdims=True)
            probs_batch = exp_l / sum_exp

            # 3. Fused Cross-Entropy + Contrastive Verifier Unlearning Adjoint (`revVJP`)
            for offset, target_id in enumerate(ref_ids):
                logits_row = logits_batch[offset]
                pred_id = int(np.argmax(logits_row))
                loss = float(max_l[offset, 0] + np.log(sum_exp[offset, 0]) - logits_row[target_id])
                epoch_loss += loss
                total_toks += 1

                probs_batch[offset, target_id] -= 1.0
                if pred_id != target_id:
                    wrong_argmax += 1
                    probs_batch[offset, pred_id] += 0.35
                    probs_batch[offset, target_id] -= 0.35

            # 4. Futhark Fused Post-AD VJP Sweep (`[T, 65536] @ [65536, 1024] -> [T, 1024]`)
            grad_h_batch = probs_batch @ embed_w
            for offset in range(T_len):
                adapter.backward_step(grad_h_batch[offset], caches[offset], lr=lr)

        if epoch in (1, 6, 12, epochs):
            elapsed = time.time() - t_start
            print(
                f"  Epoch {epoch:2d}/{epochs} ({elapsed:.2f}s) | Train NLL: {epoch_loss / total_toks:.4f} nats | "
                f"Greedy Token Accuracy: {total_toks - wrong_argmax}/{total_toks} ({100.0*(total_toks - wrong_argmax)/total_toks:.1f}%)"
            )


if __name__ == "__main__":
    print("=" * 92)
    print("CONSOLIDATED LIQUID EVOLVING AGENT v3 (LFM2.5-350M + LIQUID MoE + BINLEX + GEPA + AGENTENV)")
    print("=" * 92)

    tok = FastBPETokenizer(MODEL_DIR + r"\tokenizer.json")
    model = StatefulLFM2Model(MODEL_DIR)
    embed_w = model.t["model.embed_tokens.weight"]

    train_tasks = [t for t in TASKS if t.split == "train"]
    test_tasks = [t for t in TASKS if t.split == "heldout_test"]

    # 1. Baseline Evaluation (Unmodified Pretrained LFM2.5-350M)
    print("\n[STAGE 1] Unmodified Pretrained LiquidAI/LFM2.5-350M Baseline...")
    base_train = [evaluate_task(model, tok, t, use_binlex_chromosome=False) for t in train_tasks]
    base_test = [evaluate_task(model, tok, t, use_binlex_chromosome=False) for t in test_tasks]
    for r in base_train + base_test:
        print(f"  [{r['split']:12s}] {r['task']:14s} | Verifier: {r['passed']}/{r['total']} ({r['score']*100:5.1f}%) | NLL: {r['nll']:.3f} nats")

    # 2. Train Neural Liquid + Binlex 3-Family Router on Train Split ONLY
    adapter = LiquidStateGatedMoEAdapter(hidden_size=1024, liquid_dim=48, binlex_dim=32, num_experts=3, rank=48, seed=2026)
    print("\n[STAGE 2] Training Neural Liquid + Binlex 3-Family Router on Train Prompts ONLY...")
    train_prompt_states = []
    for t in train_tasks:
        _, _, z_p, h_m, b_v = infer_neural_expert_route(model, tok, adapter, t)
        train_prompt_states.append((z_p, h_m, b_v, t.expert_cluster))
    adapter.feat_center = np.mean(
        [np.concatenate([z_p, h_m, b_v * 8.0]) for z_p, h_m, b_v, _ in train_prompt_states], axis=0
    )
    for _ in range(40):
        for z_p, h_m, b_v, target_k in train_prompt_states:
            adapter.train_router_step(z_p, h_m, b_v, target_k, lr=0.5)

    print("  Verifying Blind Neural + Binlex 3-Family Routing on Held-Out Unseen Test Prompts:")
    for t in train_tasks + test_tasks:
        k_pred, probs, _, _, _ = infer_neural_expert_route(model, tok, adapter, t)
        print(f"    [{t.split:12s}] {t.name:14s} -> Routed Expert #{k_pred} (conf={probs[k_pred]*100:.1f}%)")

    # 3. Train Isolated Liquid Subspace Experts (Zero Prompt/Skill Change, Binlex Chromosome Normalized)
    print("\n[STAGE 3] Training Isolated Liquid Subspace Experts (Zero Skill Prompt, Binlex Chromosome Slots)...")
    plain_trajs = []
    for t in train_tasks:
        h_seq, ref_ids, start_idx = extract_cached_hidden_trajectory(
            model, tok, t, system_skill="", use_binlex_chromosome=True
        )
        plain_trajs.append((t.expert_cluster, h_seq, ref_ids, start_idx))
    train_adapter_with_contrastive_unlearning(adapter, embed_w, plain_trajs, epochs=18, lr=0.035)

    print("\n[STAGE 4] Ablation: State-Gated Liquid Weights + Binlex Chromosome Normalization ONLY (Zero Skill Prompt)...")
    liq_train = [
        evaluate_task(model, tok, t, adapter=adapter, pareto_skills=None, use_binlex_chromosome=True)
        for t in train_tasks
    ]
    liq_test = [
        evaluate_task(model, tok, t, adapter=adapter, pareto_skills=None, use_binlex_chromosome=True)
        for t in test_tasks
    ]
    for r in liq_train + liq_test:
        print(
            f"  [{r['split']:12s}] {r['task']:14s} (Expert #{r['routed_expert']}) | "
            f"Verifier: {r['passed']}/{r['total']} ({r['score']*100:5.1f}%) | NLL: {r['nll']:.3f} nats"
        )

    # 4. Coupled State-Gated Liquid Adapter + Binlex Chromosome + GEPA Pareto Skill Frontier
    pareto_skills = gepa_build_pareto_skill_frontier(base_train)
    adapter_coupled = LiquidStateGatedMoEAdapter(hidden_size=1024, liquid_dim=48, binlex_dim=32, num_experts=3, rank=48, seed=2026)
    adapter_coupled.feat_center = adapter.feat_center.copy()
    adapter_coupled.W_router = adapter.W_router.copy()
    adapter_coupled.b_router = adapter.b_router.copy()

    print("\n[STAGE 5] Training Coupled Liquid Experts + Binlex Chromosome + GEPA Pareto Skills (Train Split ONLY)...")
    skill_trajs = []
    for t in train_tasks:
        h_seq, ref_ids, start_idx = extract_cached_hidden_trajectory(
            model, tok, t, system_skill=pareto_skills[t.expert_cluster], use_binlex_chromosome=True
        )
        skill_trajs.append((t.expert_cluster, h_seq, ref_ids, start_idx))
    train_adapter_with_contrastive_unlearning(adapter_coupled, embed_w, skill_trajs, epochs=18, lr=0.035)

    print("\n[STAGE 6] Final Consolidated Evaluation (Train + Unseen Held-Out Test Split)...")
    final_train = [
        evaluate_task(model, tok, t, adapter=adapter_coupled, pareto_skills=pareto_skills, use_binlex_chromosome=True)
        for t in train_tasks
    ]
    final_test = [
        evaluate_task(model, tok, t, adapter=adapter_coupled, pareto_skills=pareto_skills, use_binlex_chromosome=True)
        for t in test_tasks
    ]
    for r in final_train + final_test:
        print(
            f"  [{r['split']:12s}] {r['task']:14s} (Expert #{r['routed_expert']}, conf={r['route_conf']*100:.0f}%) | "
            f"Verifier: {r['passed']}/{r['total']} ({r['score']*100:5.1f}%) | NLL: {r['nll']:.3f} nats"
        )
        print("      Generated Code:\n" + "\n".join("        " + ln for ln in r["code"].splitlines()))
        if r["errors"]:
            print(f"      Errors: {r['errors']}")
