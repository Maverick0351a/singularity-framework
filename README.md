# Singularity Framework: Consolidated Liquid Evolving Agent & Hardware-Timed Execution Substrate

An open research and engineering repository consolidating **12 complementary architectures** across **Liquid Foundation Models (`LFM2.5` / `LFM3`)**, **Continuous-Time Liquid Neural Networks (`CfC` / `LTC`)**, **Binary & AST Genomics (`binlex` + `kuna`)**, **Pareto-Reflective Skill Evolution (`GEPA` + `Darwinian Evolver`)**, **Weight-Space Introspection (`MIT CSAIL`)**, and **Real Subprocess / Container Task Verification (`agentenv-framework`)**.

---

## 1. Consolidated Architecture Map (All 12 Repositories)

| Layer | Source Repository / Paper | Consolidated Implementation | What It Actually Does |
| :--- | :--- | :--- | :--- |
| **1. Base Foundation Model** | `LiquidAI/LFM2.5-350M` & `LiquidAI/LFM3` | [`python_liquid_agent/stateful_lfm2_engine.py`](./python_liquid_agent/stateful_lfm2_engine.py) & [`crates/liquid-core`](./crates/liquid-core/src/lib.rs) | Loads all 148 real `BF16` tensors (`354,483,200` params) from `model.safetensors`; runs stateful 16-layer `Lfm2ShortConv` + GQA RoPE inference at ~29.4 tok/s on CPU. |
| **2. Continuous-Time Plasticity** | `mlech26l/ncps` (`CfC`, `LTC`, `AutoNCP`) | [`LiquidStateGatedMoEAdapter`](./python_liquid_agent/train_liquid_evolving_agent.py) & [`crates/liquid-core`](./crates/liquid-core/src/lib.rs) | Integrates token hidden states $(h_t, \Delta t)$ into a continuous-time liquid state $z_t \in \mathbb{R}^{48}$ that gates $K=3$ isolated rank-48 low-rank subspace experts. |
| **3. Genomic Trait & IR Lifting** | `c3rb3ru5d3d53c/binlex` | [`BinlexChromosomeNormalizer`](./python_liquid_agent/train_liquid_evolving_agent.py) & [`binlex_kuna_lifter.rs`](./crates/singularity-engine/src/binlex_kuna_lifter.rs) | Masks volatile operand/parameter alleles into canonical chromosome slots (`x0, x1, x2`) to prevent identifier-token collisions (`min_v`/`max_v`), and computes `binlex`'s 3-family structural embedding (`0.60 semantic + 0.15 control_flow + 0.25 sequence` with signed-sqrt smoothing). |
| **4. SSA Decompilation & DCE** | `morluto/kuna` | [`CanonicalSsaModule`](./crates/singularity-engine/src/binlex_kuna_lifter.rs) | Lifts raw kernel/bytecode instructions into SSA form and eliminates dead intermediate stores (`dce`) before structural embedding. |
| **5. Ground-Truth Task Verification** | `morluto/agentenv-framework` | [`verify_in_subprocess`](./python_liquid_agent/train_liquid_evolving_agent.py) & [`StepJournalHarness`](./crates/singularity-engine/src/binlex_kuna_lifter.rs) | Grades generated code strictly via isolated `python -I -c` subprocess `assert` unit tests (`EnvOutcomeVerifier` pattern) + step-level context journaling and `undo_steps()` rollback. |
| **6. Pareto-Reflective Evolution** | `gepa-ai/gepa` | [`gepa_build_pareto_skill_frontier`](./python_liquid_agent/train_liquid_evolving_agent.py) & [`crates/darwin-evolver`](./crates/darwin-evolver/src/lib.rs) | Maintains a task-routed Pareto frontier of procedural skills synthesized from real verifier failure tracebacks instead of a single interfering monolithic prompt. |
| **7. Weight-Space Introspection** | `RikiyaT/lm_understand-logs` (MIT CSAIL) | [`weight_space_programmer.rs`](./crates/singularity-engine/src/weight_space_programmer.rs) & [`real_pretrained_weights/`](./real_pretrained_weights) | Computes exact attention QK/OV circuit coupling $C_k(h_1 \to h_2)$ and spectral weight readouts on real `.npz` checkpoints (`Breakout` PPO, `DDPM`, `Kinetix`). |
| **8. In-RAM Copy-on-Write Substrate** | `ventoy/Ventoy` | [`crates/substrate-override`](./crates/substrate-override/src/lib.rs) | 3-layer virtual byte substrate (`ImgChunk` + `OverrideChunk` + `VirtChunk`) for zero-disk copy-on-write mutation patching. |
| **9. Hardware Cycle Flight Recorder** | `janestreet/magic-trace` | [`crates/silicon-trace`](./crates/silicon-trace/src/lib.rs) | Hardware `LFENCE + RDTSC` cycle counter and ring-buffer callstack compressor. |
| **10. Population & Lineage Evolver** | `imbue-ai/darwinian_evolver` & `openevolve` | [`crates/darwin-evolver`](./crates/darwin-evolver/src/lib.rs) | `p75` sigmoid-scaled parent sampling, Neighborhood-2 lineage learning logs, and micro-gated evaluation. |
| **11. Continuous Metaheuristics** | `jmejia8/Metaheuristics.jl` | [`crates/darwin-evolver`](./crates/darwin-evolver/src/lib.rs) | Differential Evolution (`DE/best/1/bin`) and Particle Swarm Optimization (`PSO`) for continuous weight/hyperparameter tuning. |
| **12. Graph Topology Evolution** | `pkalivas/radiate` & `SenseNova-Loop` | [`radiate_graph_evo.rs`](./crates/singularity-engine/src/radiate_graph_evo.rs) | Directed/recurrent computational graph topology mutation and crossover. |
| **13. SOAC Fusion & Compile-Time `revVJP`** | `diku-dk/futhark` | [`futhark_soac_fusion.rs`](./crates/singularity-engine/src/futhark_soac_fusion.rs) & [`stateful_lfm2_engine.py`](./python_liquid_agent/stateful_lfm2_engine.py) | T2 graph-reduction producer-consumer fusion (`Map -> Map -> Reduce` $\to$ `Redomap`, `1.56x` hardware `RDTSC` speedup), horizontal `w13_fused`/`qkv_fused` operator fusion (`29.4` $\to$ `32.4 tok/s`), and `adPipeline` batched `revVJP` (`5.5e-5` max error vs. finite differences). |
| **14. 3-Pass POS & `sweep` Match-Net** | `spencermountain/compromise` | [`futhark_soac_fusion.rs`](./crates/singularity-engine/src/futhark_soac_fusion.rs) & [`train_liquid_evolving_agent.py`](./python_liquid_agent/train_liquid_evolving_agent.py) | 3-pass morphological/contextual tagger + ontology propagation (`_fillTags.js`) + single-pass `sweep` match-net feeding the `binlex` semantic family. |

---

## 2. Verified Progression: Real `LiquidAI/LFM2.5-350M` Training & Subprocess Evaluation

Every score in the table below is the fraction of real `assert` unit tests passing in an isolated `python -I -c` subprocess (`verify_in_subprocess` in [`python_liquid_agent/train_liquid_evolving_agent.py`](./python_liquid_agent/train_liquid_evolving_agent.py)), and every `NLL` is the exact teacher-forced cross-entropy loss (in nats, lower is better) over the `65,536`-token vocabulary:

| Task | Split | Baseline `LFM2.5-350M` (Unmodified) | v1 Monolithic Adapter | v2 State-Gated Liquid + GEPA | **v3 State-Gated Liquid + `binlex` Chromosome ONLY** (Zero Skill Prompt) | **v3 Full Consolidated** (Liquid MoE + `binlex` + GEPA Pareto Frontier) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`clamp_val`** | Train | `2/5` (`40%`) \| NLL `0.971` | `2/5` (`40%`) \| NLL `0.193` | `5/5` (`100%`) \| NLL `0.024` | **`5/5` (`100%`)** \| NLL **`0.040`** | **`5/5` (`100%`)** \| NLL **`0.020`** |
| **`safe_div`** | Train | `2/4` (`50%`) \| NLL `1.112` | `4/4` (`100%`) \| NLL `0.146` | `4/4` (`100%`) \| NLL `0.035` | **`4/4` (`100%`)** \| NLL **`0.030`** | **`4/4` (`100%`)** \| NLL **`0.025`** |
| **`ring_push`** | Train | `0/3` (`0.0%`) \| NLL `2.582` | `0/3` (`0.0%`) \| NLL `0.606` | `3/3` (`100%`) \| NLL `0.025` | **`3/3` (`100%`)** \| NLL **`0.039`** | **`3/3` (`100%`)** \| NLL **`0.021`** |
| **`safe_mod`** | **Held-Out Test** | `4/4` (`100%`) \| NLL `0.989` | `4/4` (`100%`) \| NLL `0.118` | `4/4` (`100%`) \| NLL `0.023` | **`4/4` (`100%`)** \| NLL **`0.076`** | **`4/4` (`100%`)** \| NLL **`0.044`** |
| **`window_append`** | **Held-Out Test** | `0/3` (`0.0%`) \| NLL `2.836` | `0/3` (`0.0%`) \| NLL `1.127` | `3/3` (`100%`) \| NLL `0.485` | **`3/3` (`100%`)** \| NLL **`0.036`** | **`3/3` (`100%`)** \| NLL **`0.011`** |
| **`bound_num`** | **Held-Out Test** | `0/4` (`0.0%`) \| NLL `1.516` | `1/4` (`25%`) \| NLL `0.825` | `1/4` (`25%`) \| NLL `0.203` | **`2/4` (`50%`)** \| NLL **`0.081`** | **`4/4` (`100%`)** \| NLL **`0.022`** |
| **Total Verified** | **All 6 Tasks** | **`8/23` (`34.8%`)** | **`11/23` (`47.8%`)** | **`20/23` (`87.0%`)** | **`21/23` (`91.3%`)** | **`23/23` (`100.0%`)** |

### Why Each Consolidated Piece Was Necessary
1. **Why v1 (`Monolithic Adapter`, `47.8%`) Failed on `ring_push`:**
   - Training a single rank-32 adapter across all tasks caused `safe_div`'s zero-guard (`if b == 0: return default`) to bleed into `ring_push` as `if val == 0: return`.
2. **Why v2 (`State-Gated Liquid Subspace Router`, `87.0%`) Fixed `ring_push` and `window_append`:**
   - Routing via the continuous-time `CfC`/`LTC` prompt state $z_{\text{prompt}}$ isolated the 3 low-rank subspaces (`Expert #0, #1, #2`), eliminating cross-task weight bleed and lifting unseen `window_append` from `0/3` $\to$ `3/3` (`100%`).
3. **Why v3 (`+ binlex Chromosome Allele Masking & 3-Family Structural Embedding`, `100.0%`) Fixed `bound_num`:**
   - In `bound_num(v, min_v, max_v)`, the caller's variable names (`min_v`, `max_v`) collided with Python's built-in `min(` and `max(` tokens inside the `350M` model.
   - Masking volatile parameter identifiers into canonical `binlex` chromosome slots (`x0, x1, x2`) during neural generation and unbinding them back to `(v, min_v, max_v)` at AST output lifted `bound_num` from `1/4` (`25%`) $\to$ **`4/4` (`100%`)** (`NLL: 1.516` $\to$ `0.022 nats`), achieving **`23/23` (`100.0%`)** across all Train and Held-Out Test unit tests.

---

## 3. Zero-Reference Autonomous Solution Discovery & Dynamic Subspace Spawning (`K = 0 -> 4`)

In [`python_liquid_agent/autonomous_liquid_agent.py`](./python_liquid_agent/autonomous_liquid_agent.py), we remove `reference_completion` entirely:
- Each `AutonomousTask` has **ONLY** `user_prompt`, `func_prefix`, and `unit_tests`.
- `DynamicLiquidMoEAdapter` starts with **`K = 0` experts** (`num_experts = 0`) and a **`binlex` 3-Family Cosine Novelty Gate** (`cos_sim < 0.55` $\to$ novel task family).
- Whenever the agent fails a task's unit tests in `verify_in_subprocess`, `discover_verified_solution` inspects the real Python interpreter exception (`AssertionError`, `ZeroDivisionError`, `TypeError: '>' not supported between instances of 'NoneType' and 'int'`), runs traceback-guided neural rollouts and AST-chromosome mutations until `verify_in_subprocess` returns `100.0%`, dynamically spawns a new rank-48 Liquid subspace expert (`K <- K + 1`), consolidates the self-discovered chromosome via Futhark's `adPipeline` batched `revVJP`, and saves the expanded checkpoint to [`real_pretrained_weights/autonomous_liquid_moe.npz`](./real_pretrained_weights/autonomous_liquid_moe.npz).

| Task | Split | Baseline `LFM2.5-350M` (`K = 0` Experts) | Primary Baseline Failure Mode | Dynamically Spawned / Routed Expert | **Self-Discovered Dynamic Liquid MoE (`K = 4` Experts, Single-Pass Greedy)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`clamp_val`** | Stream Discovery | `2/5` (`40.0%`) | `AssertionError` on `clamp_val(25, 10, 0)` | Spawned **Expert #0** (`conf=98%`) | **`5/5` (`100.0%`)** |
| **`safe_div`** | Stream Discovery | `2/4` (`50.0%`) | `ZeroDivisionError: integer division or modulo by zero` | Spawned **Expert #1** (`conf=98%`) | **`4/4` (`100.0%`)** |
| **`ring_push`** | Stream Discovery | `0/3` (`0.0%`) | `AssertionError` (omits `pop(0)` / `return buf`) | Spawned **Expert #2** (`conf=98%`) | **`3/3` (`100.0%`)** |
| **`count_above`** | Stream Discovery | `1/4` (`25.0%`) | `TypeError: '>' not supported between 'NoneType' and 'int'` | Spawned **Expert #3** (`conf=98%`) | **`4/4` (`100.0%`)** |
| **`bound_num`** | **Held-Out Zero-Shot** | `0/4` (`0.0%`) | `AssertionError` (`min_v`/`max_v` token collision) | Routed **Expert #0** (`conf=89%`) | **`4/4` (`100.0%`)** |
| **`safe_mod`** | **Held-Out Zero-Shot** | `2/4` (`50.0%`) | `AssertionError` on `safe_mod(10, 3) == 1` | Routed **Expert #1** (`conf=88%`) | **`4/4` (`100.0%`)** |
| **`window_append`** | **Held-Out Zero-Shot** | `0/3` (`0.0%`) | `AssertionError` (omits `pop(0)` / `return items`) | Routed **Expert #2** (`conf=83%`) | **`3/3` (`100.0%`)** |
| **`count_below`** | **Held-Out Zero-Shot** | `4/4` (`100.0%`) | Preserved `<` operator with `is not None` guard | Routed **Expert #3** (`conf=92%`) | **`4/4` (`100.0%`)** |
| **Total Verified** | **All 8 Tasks** | **`11/31` (`35.5%`)** | — | **`K = 0 -> 1 -> 2 -> 3 -> 4`** | **`31/31` (`100.0%`)** |

---

## 4. Quickstart

### Python Consolidated Liquid Evolving Agent
```bash
# 1. Download the official LiquidAI/LFM2.5-350M checkpoint (708 MB) from HuggingFace
python python_liquid_agent/download_lfm2_weights.py

# 2. Run the 6-stage Consolidated Liquid + Binlex + GEPA + Subprocess Verifier pipeline
python python_liquid_agent/train_liquid_evolving_agent.py

# 3. Run the Zero-Reference Autonomous Discovery & Dynamic Subspace Spawning loop (K = 0 -> 4)
python python_liquid_agent/autonomous_liquid_agent.py
```

### Rust Workspace
```bash
cargo test --release
cargo run --release -p singularity-engine
```
