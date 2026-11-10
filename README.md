# Singularity Framework: Liquid Evolving Agent & Hardware-Timed Execution Substrate

An open research and engineering repository uniting **Liquid Foundation Models (`LFM2.5` / `LFM3`)**, **Continuous-Time Liquid Neural Networks (`CfC` / `LTC`)**, **Pareto-Reflective Skill Evolution (`GEPA`)**, and **Real Subprocess / Container Task Verification (`agentenv-framework` pattern)**.

---

## 1. Real Pretrained Liquid Evolving Agent (`python_liquid_agent/`)

Located in [`python_liquid_agent/`](./python_liquid_agent), this pipeline runs and trains a **Continuous-Time Liquid State-Gated Subspace Adapter (`LiquidStateGatedMoEAdapter`)** directly on top of Liquid AI's official **`LiquidAI/LFM2.5-350M`** (`354,483,200` `BF16` parameters across 16 hybrid layers: 10 `Lfm2ShortConv` width-3 gated convolution layers + 6 Grouped-Query Attention RoPE layers + SwiGLU MLPs).

### Key Architectural Findings
1. **Stateful $O(1)$ `Lfm2ShortConv` + GQA Execution (`stateful_lfm2_engine.py`):**
   - Loads all 148 `BF16` tensors directly from `model.safetensors` and executes token steps at **~29.4 tokens/sec on CPU** via contiguous BLAS `sgemv` (`W @ v`) and a 2-token `Lfm2ShortConv` ring buffer.
2. **Eliminating Cross-Task Interference in Compact (`350M`) Models via Liquid State Routing:**
   - Training a single monolithic low-rank adapter or monolithic prompt across distinct coding tasks causes **cross-task interference** (for example, `if b == 0: return default` from safe division bleeds into bounded list operations as `if val == 0: return`).
   - Our **`LiquidStateGatedMoEAdapter`** feeds `LFM2.5-350M`'s 1024-d hidden states through a continuous-time `CfC`/`LTC` cell to form a prompt state vector $[z_{\text{prompt}}, \bar{h}_{\text{prompt}}]$, which neutrally routes each task to isolated low-rank Liquid subspace experts ($K=3$) and task-specialized GEPA Pareto skill cards.
3. **Ground-Truth Subprocess Verification (Zero Synthetic Proxy Metrics):**
   - Every generated Python function is graded strictly by executing deterministic `assert` unit tests inside an isolated `python -I -c` subprocess (`verify_in_subprocess`), alongside exact teacher-forced cross-entropy loss (`NLL` in nats) over the 65,536-token vocabulary.

### Verified Experimental Results (`python_liquid_agent/train_liquid_evolving_agent.py`)

| Task | Split | Baseline `LFM2.5-350M` (Unmodified) | v1 Monolithic Adapter | **v2 State-Gated Liquid Weights ONLY** (Zero Prompt Change) | **v2 Full Coupled** (State-Gated Liquid + GEPA Pareto Frontier) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`clamp_val`** | Train | `2/5` (`40%`) \| NLL `0.971` | `2/5` (`40%`) \| NLL `0.193` | **`5/5` (`100%`)** \| NLL **`0.035`** | **`5/5` (`100%`)** \| NLL **`0.024`** |
| **`safe_div`** | Train | `2/4` (`50%`) \| NLL `1.112` | `4/4` (`100%`) \| NLL `0.146` | **`4/4` (`100%`)** \| NLL **`0.035`** | **`4/4` (`100%`)** \| NLL **`0.035`** |
| **`ring_push`** | Train | `0/3` (`0.0%`) \| NLL `2.582` | `0/3` (`0.0%`) \| NLL `0.606` | **`3/3` (`100%`)** \| NLL **`0.037`** | **`3/3` (`100%`)** \| NLL **`0.025`** |
| **`safe_mod`** | **Held-Out Test** | `4/4` (`100%`) \| NLL `0.989` | `4/4` (`100%`) \| NLL `0.118` | **`4/4` (`100%`)** \| NLL **`0.024`** | **`4/4` (`100%`)** \| NLL **`0.023`** |
| **`window_append`** | **Held-Out Test** | `0/3` (`0.0%`) \| NLL `2.836` | `0/3` (`0.0%`) \| NLL `1.127` | `0/3` (`0.0%`) \| NLL **`0.529`** | **`3/3` (`100%`)** \| NLL **`0.485`** |
| **`bound_num`** | **Held-Out Test** | `0/4` (`0.0%`) \| NLL `1.516` | `1/4` (`25%`) \| NLL `0.825` | `0/4` (`0.0%`) \| NLL **`0.801`** | **`1/4` (`25%`)** \| NLL **`0.203`** |

### Quickstart (Python Liquid Evolving Agent)
```bash
# 1. Download the official LiquidAI/LFM2.5-350M checkpoint (708 MB) from HuggingFace
python python_liquid_agent/download_lfm2_weights.py

# 2. Run the 16-layer stateful LFM2.5-350M engine benchmark
python python_liquid_agent/stateful_lfm2_engine.py

# 3. Run the full 6-stage Liquid State-Gated Subspace Router + Subprocess Verifier training loop
python python_liquid_agent/train_liquid_evolving_agent.py
```

---

## 2. Rust Workspace (`crates/`)

The Rust workspace implements clean-room systems and neural building blocks across 5 crates:

- **`crates/liquid-core`:** Clean-room Rust implementation of `CfCCell` (Closed-form Continuous-time cell), `LTCCell` (semi-implicit ODE solver with online Hebbian synapse plasticity), `AutoNcpWiring` (4-layer sparse C. elegans wiring), `Lfm2ShortConv`, and `Lfm3TriHybridBlock` (width-7 segmented `Lfm3ShortConv` + `Lfm3ZeroTailRoPE` + gated GQA + 4-way Sandwich RMSNorm).
- **`crates/substrate-override`:** In-RAM 3-layer virtual image substrate (`ImgChunk` + `OverrideChunk` + `VirtChunk`) inspired by `ventoy/Ventoy`, enabling zero-disk copy-on-write byte patching for evolving programs.
- **`crates/silicon-trace`:** Hardware cycle-accurate (`LFENCE + RDTSC`) ring-buffer flight recorder and callstack compressor inspired by `janestreet/magic-trace`.
- **`crates/darwin-evolver`:** Evolutionary population engine implementing `p75` sigmoid-scaled parent selection, Neighborhood-2 lineage learning logs, `GEPA` Pareto frontier selection, Differential Evolution (`DE/best/1/bin`), and PSO velocity updates.
- **`crates/singularity-engine`:** Integrates the sandboxed bytecode VM, hardware `RDTSC` profiler, `Radiate` graph evolution, and MIT CSAIL weight-space analysis (`real_pretrained_weights/`).

### Quickstart (Rust Workspace)
```bash
cargo test --release
cargo run --release -p singularity-engine
```
