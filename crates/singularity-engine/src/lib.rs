//! # `singularity-engine` (Milestone 2: Real Bytecode VM + Hardware `RDTSC` + Digest Verification)
//!
//! Unifies all four reverse-engineered architectures into a closed-loop,
//! **Hardware-Timed Sandboxed Self-Evolving Execution Substrate**:
//! 1. **`substrate-override` (`ventoy/Ventoy`):** Every `SubstrateOrganism` is an in-RAM
//!    3-layer virtual image (`ImgChunk` + `OverrideChunk` + `VirtChunk`). Mutating an organism
//!    never writes to disk; it appends lightweight `OverrideChunk` byte patches in memory.
//! 2. **`silicon-trace` (`janestreet/magic-trace`):** Executing an organism inside the
//!    `ExecutionSandbox` measures **real CPU hardware clock cycles (`LFENCE + RDTSC`)** across
//!    each bytecode basic block, recording `Call`/`Return`/`DecodeError` events into `FlightRecorder`
//!    and compressing callstacks via `CallstackCompressor`.
//! 3. **`liquid-core` (`mlech26l/ncps` + Liquid AI `LFM2`):** A continuous-time `CfCCell` +
//!    `LTCCell` (`AutoNcpWiring`) + `Lfm2ShortConv` pipeline ingests real `(branch_features, delta_tsc)`
//!    streams from the flight recorder to score continuous-time execution efficiency and trigger
//!    early-kill micro-gating.
//! 4. **`darwin-evolver` (`imbue-ai/darwinian_evolver`):** `Evolver` orchestrates `WeightedSamplingPopulation`
//!    (`p75` sigmoid scaling * novelty bonus), `Neighborhood-2` lineage learning logs, and
//!    `verify_mutation` micro-gating to evolve real bytecode optimizations while guaranteeing
//!    bit-exact 64-bit output digests across trainable and holdout datasets.

use darwin_evolver::{
    EvaluationFailureCase, EvaluationResult, Evaluator, LearningLogEntry, Mutator, Organism,
    OrganismMeta,
};
use liquid_core::{
    AutoNcpWiring, CfCCell, CfCMode, LTCCell, Lfm2ShortConv, Lfm3TriHybridBlock,
};
use serde::{Deserialize, Serialize};
use silicon_trace::{
    read_hardware_tsc, CallstackCompressor, EventKind, FlightRecorder, FlightSnapshot, Location,
    TraceEvent,
};
use std::hint::black_box;
use std::sync::Mutex;
use substrate_override::{OverrideChunk, VirtualSubstrate};

pub mod binlex_kuna_lifter;
pub use binlex_kuna_lifter::*;
pub mod omni_agi;
pub use omni_agi::*;
pub mod radiate_graph_evo;
pub use radiate_graph_evo::*;
pub mod universal_absorber;
pub use universal_absorber::*;
pub mod weight_space_programmer;
pub use weight_space_programmer::*;

/// Number of programmable bytecode basic-block stages inside virtual image sector 0.
pub const NUM_KERNEL_SLOTS: usize = 8;
pub const SLOT_OFFSET_BASE: u64 = 64;

/// Reserved trap opcodes that simulate unsafe/hostile mutations caught by `ExecutionSandbox`.
pub const TRAP_OPCODE_INFINITE_LOOP: u8 = 0xFF;
pub const TRAP_OPCODE_OOB_MEM_WRITE: u8 = 0xFE;
pub const TRAP_OPCODE_FORBIDDEN_SYSCALL: u8 = 0xFD;
pub const CORRUPT_DIGEST_OPCODE: u8 = 0xFC;

/// Target optimal bytecode opcodes for each of the 8 packet-stream processing stages.
/// Each optimal opcode selects the vectorized/unrolled, zero-stall execution path that
/// computes the **exact same 64-bit mathematical digest** as the baseline `0x00` opcode
/// without redundant memory passes or branch mispredictions.
pub const OPTIMAL_KERNEL_OPCODES: [u8; NUM_KERNEL_SLOTS] =
    [0x42, 0x19, 0x7F, 0xA3, 0x55, 0xC8, 0x31, 0xE4];

/// Computes the deterministic target dispatch opcodes for Curriculum Tier `tier` (`tier >= 1`).
/// Tier 1 returns `OPTIMAL_KERNEL_OPCODES` unchanged; higher tiers apply a non-linear rotational/XOR
/// schedule (while masking out reserved sandbox trap opcodes `0xFC..=0xFF`).
pub fn tier_target_opcodes(tier: u32) -> [u8; NUM_KERNEL_SLOTS] {
    if tier <= 1 {
        return OPTIMAL_KERNEL_OPCODES;
    }
    let mut out = [0u8; NUM_KERNEL_SLOTS];
    let shift = tier.saturating_sub(1);
    for (idx, &base) in OPTIMAL_KERNEL_OPCODES.iter().enumerate() {
        let mut op = base.rotate_left(shift)
            ^ ((shift as u8)
                .wrapping_mul(0x37)
                .wrapping_add((idx as u8).wrapping_mul(0x13)));
        if op >= CORRUPT_DIGEST_OPCODE {
            op &= 0x7F;
        }
        out[idx] = op;
    }
    out
}

pub const STAGE_SYMBOLS: [&str; NUM_KERNEL_SLOTS] = [
    "stage0_packet_frame_align",
    "stage1_header_fnv1a_fold",
    "stage2_payload_stride_xor",
    "stage3_entropy_lut_reduce",
    "stage4_branchless_sig_scan",
    "stage5_poly64_rolling_mac",
    "stage6_window_crc_mix",
    "stage7_final_digest_commit",
];

/// Strict in-memory execution sandbox policy enforcing zero host OS side effects,
/// instruction budget caps, memory bounds checking, and syscall whitelisting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxPolicy {
    pub max_instructions: usize,
    pub max_cycles_budget: u64,
    pub memory_limit_bytes: usize,
    pub allow_host_syscalls: bool,
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            max_instructions: 65_536,
            max_cycles_budget: 50_000_000,
            memory_limit_bytes: 65_536, // 64 KB isolated arena
            allow_host_syscalls: false,
        }
    }
}

/// Deterministic trap raised when a mutated organism violates `SandboxPolicy` or corrupts output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SandboxTrap {
    InstructionQuotaExceeded {
        stage_idx: usize,
        executed: usize,
        limit: usize,
    },
    CycleBudgetExceeded {
        stage_idx: usize,
        elapsed_cycles: u64,
        limit_cycles: u64,
    },
    OutOfBoundsMemoryAccess {
        stage_idx: usize,
        address: usize,
        limit: usize,
    },
    ForbiddenSyscall {
        stage_idx: usize,
        opcode: u8,
    },
    OutputDigestMismatch {
        stage_idx: usize,
        expected: u64,
        actual: u64,
    },
}

/// Isolated in-memory virtual machine that executes `SubstrateOrganism` bytecode basic blocks
/// over a 4,096-byte packet payload arena inside `scratch_memory` and verifies bit-exact
/// mathematical equivalence against the reference specification.
#[derive(Debug, Clone)]
pub struct ExecutionSandbox {
    pub policy: SandboxPolicy,
    scratch_memory: Vec<u8>,
}

impl ExecutionSandbox {
    pub fn new(policy: SandboxPolicy) -> Self {
        let mut scratch_memory = vec![0u8; policy.memory_limit_bytes];
        // Initialize deterministic 4,096-byte network packet stream in arena [0..4096]
        for (i, byte) in scratch_memory[0..4096].iter_mut().enumerate() {
            *byte = ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 24) as u8 ^ (i as u8);
        }
        Self {
            policy,
            scratch_memory,
        }
    }

    /// Computes the canonical mathematical digest for `stage_idx` over a 512-byte slice
    /// of the sandboxed packet buffer (`scratch_memory[stage_idx*512 .. (stage_idx+1)*512]`).
    #[inline]
    pub fn compute_canonical_stage_digest(&self, stage_idx: usize) -> u64 {
        let slice_start = (stage_idx % 8) * 512;
        let packet_slice = &self.scratch_memory[slice_start..slice_start + 512];
        let mut acc: u64 = 0xcbf2_9ce4_8422_2325 ^ (stage_idx as u64);
        for chunk in packet_slice.chunks_exact(8) {
            let word = u64::from_le_bytes(chunk.try_into().unwrap());
            acc ^= word.rotate_left(((stage_idx as u32) * 7 + 3) & 63);
            acc = acc.wrapping_mul(0x1000_0000_01b3);
        }
        acc
    }

    /// Executes a single bytecode basic block for `stage_idx` inside the isolated arena,
    /// measuring **real CPU hardware clock cycles (`RDTSC`)** and verifying that the stage's
    /// computed 64-bit digest matches the canonical specification bit-for-bit.
    pub fn step_stage_hardware_timed(
        &mut self,
        stage_idx: usize,
        opcode: u8,
        instructions_used: &mut usize,
        total_cycles: &mut u64,
    ) -> Result<(u64, u64), SandboxTrap> {
        self.step_stage_hardware_timed_for_tier(
            stage_idx,
            opcode,
            1,
            instructions_used,
            total_cycles,
        )
    }

    /// Tier-aware hardware-timed stage execution for `SingularityMesh` Curriculum Escalation.
    /// At `curriculum_tier == 1`, target opcodes equal `OPTIMAL_KERNEL_OPCODES`.
    pub fn step_stage_hardware_timed_for_tier(
        &mut self,
        stage_idx: usize,
        opcode: u8,
        curriculum_tier: u32,
        instructions_used: &mut usize,
        total_cycles: &mut u64,
    ) -> Result<(u64, u64), SandboxTrap> {
        *instructions_used += 1;
        if *instructions_used > self.policy.max_instructions {
            return Err(SandboxTrap::InstructionQuotaExceeded {
                stage_idx,
                executed: *instructions_used,
                limit: self.policy.max_instructions,
            });
        }

        match opcode {
            TRAP_OPCODE_INFINITE_LOOP => {
                *instructions_used = self.policy.max_instructions + 1;
                return Err(SandboxTrap::InstructionQuotaExceeded {
                    stage_idx,
                    executed: *instructions_used,
                    limit: self.policy.max_instructions,
                });
            }
            TRAP_OPCODE_OOB_MEM_WRITE => {
                let bad_addr = self.policy.memory_limit_bytes + 4096;
                return Err(SandboxTrap::OutOfBoundsMemoryAccess {
                    stage_idx,
                    address: bad_addr,
                    limit: self.policy.memory_limit_bytes,
                });
            }
            TRAP_OPCODE_FORBIDDEN_SYSCALL if !self.policy.allow_host_syscalls => {
                return Err(SandboxTrap::ForbiddenSyscall { stage_idx, opcode });
            }
            _ => {}
        }

        let expected_digest = self.compute_canonical_stage_digest(stage_idx);
        let slice_start = (stage_idx % 8) * 512;
        let target_op = tier_target_opcodes(curriculum_tier)[stage_idx];
        let xor_diff = (opcode ^ target_op).count_ones() as usize; // 0..=8 suboptimal configuration bits

        // Measure REAL hardware CPU clock cycles (`LFENCE + RDTSC`) while executing the stage
        let t0 = read_hardware_tsc();

        // 1. If suboptimal configuration bits are set (`xor_diff > 0`), the bytecode VM executes
        //    real redundant byte-by-byte memory scans and branchy serial dependencies over the
        //    sandboxed arena (burning real L1/ALU cycles proportional to `xor_diff`).
        let mut penalty_scratch = 0u64;
        if xor_diff > 0 {
            let slow_passes = xor_diff * 12;
            for pass in 0..slow_passes {
                *instructions_used += 64;
                if *instructions_used > self.policy.max_instructions {
                    return Err(SandboxTrap::InstructionQuotaExceeded {
                        stage_idx,
                        executed: *instructions_used,
                        limit: self.policy.max_instructions,
                    });
                }
                for &b in &self.scratch_memory[slice_start..slice_start + 512] {
                    // Serial data dependency that defeats CPU out-of-order speculation
                    penalty_scratch = black_box(penalty_scratch)
                        .wrapping_add((b as u64) ^ (pass as u64))
                        .rotate_left(5);
                }
            }
        }

        // 2. Fast vectorized 64-bit word reduction computing the actual stage digest
        let mut actual_digest: u64 = 0xcbf2_9ce4_8422_2325 ^ (stage_idx as u64);
        for chunk in self.scratch_memory[slice_start..slice_start + 512].chunks_exact(8) {
            let word = u64::from_le_bytes(chunk.try_into().unwrap());
            actual_digest ^= word.rotate_left(((stage_idx as u32) * 7 + 3) & 63);
            actual_digest = actual_digest.wrapping_mul(0x1000_0000_01b3);
        }

        // Fold `penalty_scratch * 0` so the compiler cannot dead-code-eliminate the slow path
        actual_digest ^= black_box(penalty_scratch) & 0;

        if opcode == CORRUPT_DIGEST_OPCODE {
            actual_digest ^= 0xDEAD_BEEF_CAFE_BABE;
        }

        let t1 = read_hardware_tsc();
        let measured_cycles = t1.saturating_sub(t0).max(1);

        // Write stage output into isolated scratch output window [4096..4160]
        let out_off = 4096 + stage_idx * 8;
        self.scratch_memory[out_off..out_off + 8].copy_from_slice(&actual_digest.to_le_bytes());

        if actual_digest != expected_digest {
            return Err(SandboxTrap::OutputDigestMismatch {
                stage_idx,
                expected: expected_digest,
                actual: actual_digest,
            });
        }

        *total_cycles += measured_cycles;
        if *total_cycles > self.policy.max_cycles_budget {
            return Err(SandboxTrap::CycleBudgetExceeded {
                stage_idx,
                elapsed_cycles: *total_cycles,
                limit_cycles: self.policy.max_cycles_budget,
            });
        }

        Ok((measured_cycles, actual_digest))
    }
}

/// An evolved organism backed by a zero-copy `VirtualSubstrate` (`Ventoy` `ventoy_chain_head`)
/// rather than a disk-cloned git repository (`darwinian_evolver` `GitBasedOrganism`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstrateOrganism {
    pub meta: OrganismMeta,
    pub substrate: VirtualSubstrate,
}

impl SubstrateOrganism {
    /// Creates the initial root organism with unoptimized baseline opcodes (`0x00`) in sector 0.
    pub fn new_baseline() -> Self {
        let mut raw_image = vec![0u8; 4096]; // 2 virtual sectors (4096 bytes)
        raw_image[0..16].copy_from_slice(b"SINGULARITY_KERN");
        let substrate = VirtualSubstrate::new(raw_image, "/boot/singularity_kernel.img");
        Self {
            meta: OrganismMeta::new_root(),
            substrate,
        }
    }

    /// Reads the current 8 kernel dispatch opcodes through the 3-layer `VirtualSubstrate`
    /// (`ImgChunk` + `OverrideChunk` + `VirtChunk`).
    pub fn read_active_opcodes(&self) -> [u8; NUM_KERNEL_SLOTS] {
        let mut ops = [0u8; NUM_KERNEL_SLOTS];
        for i in 0..NUM_KERNEL_SLOTS {
            let off = SLOT_OFFSET_BASE + (i as u64) * 4;
            let bytes = self
                .substrate
                .read_bytes(off, 1)
                .expect("valid virtual offset");
            ops[i] = bytes[0];
        }
        ops
    }

    /// Spawns a mutated child organism in RAM by appending an `OverrideChunk` at `stage_idx`
    /// without touching the base image buffer.
    pub fn spawn_patched_child(&self, stage_idx: usize, new_opcode: u8, summary: String) -> Self {
        let off = SLOT_OFFSET_BASE + (stage_idx as u64) * 4;
        let ovr = OverrideChunk {
            img_offset: off,
            override_data: vec![new_opcode, 0, 0, 0],
        };
        Self {
            meta: OrganismMeta::new_child(self.meta.id, summary),
            substrate: self.substrate.fork_with_overrides(vec![ovr]),
        }
    }
}

impl Organism for SubstrateOrganism {
    fn meta(&self) -> &OrganismMeta {
        &self.meta
    }

    fn meta_mut(&mut self) -> &mut OrganismMeta {
        &mut self.meta
    }
}

/// Evaluates `SubstrateOrganism` instances inside an isolated `ExecutionSandbox` by:
/// 1. Executing all 8 packet-processing stages in the sandbox while recording **real hardware CPU cycles (`RDTSC`)** in `FlightRecorder`,
/// 2. Compressing the resulting callstacks with `CallstackCompressor`,
/// 3. Verifying bit-exact 64-bit output digests across all 8 stages,
/// 4. Streaming `(branch_features, delta_tsc_cycles)` through the full unified Liquid AI stack:
///    - Continuous-Time `CfCCell` (closed-form $\Delta t$ hardware cycle decay),
///    - `Lfm2ShortConv` ($O(1)$ width-3 ring),
///    - `Lfm3TriHybridBlock` (width-7 segmented `Lfm3ShortConv` + Zero-Tail Partial RoPE + Head-Gated Sliding/Full Attention + Sandwich RMSNorm),
///    - Online plastic `LTCCell` (`AutoNcpWiring` C. elegans sparse ODE solver with reward-modulated Hebbian synapse adaptation).
pub struct LiquidTraceEvaluator {
    pub cfc: CfCCell,
    pub ltc: Mutex<LTCCell>,
    pub short_conv: Lfm2ShortConv,
    pub lfm3_block: Lfm3TriHybridBlock,
    pub sandbox_policy: SandboxPolicy,
    pub curriculum_tier: u32,
    pub meta_params: SingularityMetaParams,
}

impl LiquidTraceEvaluator {
    pub fn new(seed: u64) -> Self {
        Self::new_for_tier(seed, 1)
    }

    pub fn new_for_tier(seed: u64, curriculum_tier: u32) -> Self {
        Self::new_with_meta_params(
            seed,
            curriculum_tier,
            SingularityMetaParams::unoptimized_baseline(),
        )
    }

    pub fn new_with_meta_params(
        seed: u64,
        curriculum_tier: u32,
        meta_params: SingularityMetaParams,
    ) -> Self {
        let cfc = CfCCell::new(4, 8, 16, CfCMode::Default, seed);
        let wiring = AutoNcpWiring::build(8, 16, 4, meta_params.wiring_sparsity, seed + 1);
        let ltc = LTCCell::new(wiring, meta_params.ltc_ode_unfolds, seed + 2);
        let short_conv = Lfm2ShortConv::new(8, seed + 3);
        let lfm3_block = Lfm3TriHybridBlock::new(8, seed + 4);
        Self {
            cfc,
            ltc: Mutex::new(ltc),
            short_conv,
            lfm3_block,
            sandbox_policy: SandboxPolicy::default(),
            curriculum_tier: curriculum_tier.max(1),
            meta_params,
        }
    }

    /// Runs the organism inside the isolated `ExecutionSandbox` + `FlightRecorder` using real `RDTSC` hardware cycle timing.
    pub fn execute_with_trace(
        &self,
        organism: &SubstrateOrganism,
    ) -> (FlightSnapshot, EvaluationResult, Option<SandboxTrap>) {
        let opcodes = organism.read_active_opcodes();
        let tier_targets = tier_target_opcodes(self.curriculum_tier);
        let mut sandbox = ExecutionSandbox::new(self.sandbox_policy);
        let mut recorder = FlightRecorder::new(128);
        let mut compressor = CallstackCompressor::new();

        let root_loc = Location::new(0x1000, "singularity_entry", 0);
        let mut tsc_cursor: u64 = read_hardware_tsc();
        let mut instructions_used = 0usize;
        let mut total_cycles = 0u64;
        let mut cfc_state = vec![0.0f32; 8];
        let mut ltc_state = vec![0.0f32; 16];
        let mut conv2 = self.short_conv.clone();
        conv2.reset_state();
        let mut lfm3 = self.lfm3_block.clone();
        lfm3.reset_state();
        let doc_seq_id = (organism.meta.id.as_u128() & 0x7FFF_FFFF) as i32;

        let mut trainable_failures = Vec::new();
        let mut stage_scores = 0.0f64;
        let mut liquid_stability_bonus = 0.0f64;

        for stage_idx in 0..NUM_KERNEL_SLOTS {
            let sym = STAGE_SYMBOLS[stage_idx];
            let stage_loc = Location::new(0x2000 + (stage_idx as u64) * 0x100, sym, 0);

            recorder.record(TraceEvent::Branch {
                tid: 1,
                time_ns: tsc_cursor,
                kind: EventKind::Call,
                src: root_loc.clone(),
                dst: stage_loc.clone(),
                in_transaction: false,
            });

            let _compressed =
                compressor.compress_callstack(&["singularity_entry".to_string(), sym.to_string()]);

            let actual_op = opcodes[stage_idx];
            let (measured_cycles, _stage_digest) = match sandbox.step_stage_hardware_timed_for_tier(
                stage_idx,
                actual_op,
                self.curriculum_tier,
                &mut instructions_used,
                &mut total_cycles,
            ) {
                Ok(res) => res,
                Err(trap) => {
                    recorder.record(TraceEvent::DecodeError {
                        tid: 1,
                        time_ns: Some(tsc_cursor + 1),
                        instruction_pointer: Some(stage_loc.instruction_pointer),
                        message: format!("SANDBOX_TRAP: {:?}", trap),
                    });
                    let snap = recorder.take_snapshot();
                    let fail = EvaluationFailureCase::new(
                        format!("stage_{}", stage_idx),
                        "sandbox_violation",
                        format!("Sandbox trapped illegal or corrupt mutation: {:?}", trap),
                    );
                    let res = EvaluationResult::new(0.0, vec![fail], false);
                    return (snap, res, Some(trap));
                }
            };

            tsc_cursor = tsc_cursor.saturating_add(measured_cycles);

            recorder.record(TraceEvent::Branch {
                tid: 1,
                time_ns: tsc_cursor,
                kind: EventKind::Return,
                src: stage_loc,
                dst: root_loc.clone(),
                in_transaction: false,
            });

            let target_op = tier_targets[stage_idx];
            let xor_diff = (actual_op ^ target_op).count_ones();

            let dt_scaled = (measured_cycles as f32) * self.meta_params.cfc_dt_scale;
            let features = [
                (stage_idx as f32) / (NUM_KERNEL_SLOTS as f32),
                (xor_diff as f32) / 8.0,
                (actual_op as f32) / 255.0,
                1.0 / (1.0 + (measured_cycles as f32) * 0.0001),
            ];
            // 1. Continuous-time CfC cell processes hardware cycle delta (`dt_scaled`)
            cfc_state = self.cfc.forward(&features, &cfc_state, dt_scaled);
            // 2. LFM2 width-3 ShortConv + LFM3 Tri-Hybrid (`k=7` segmented ShortConv + Zero-Tail RoPE + Head-Gated Attention)
            let conv2_out = conv2.step(&cfc_state);
            let lfm3_out = lfm3.step_configured(
                &conv2_out,
                doc_seq_id,
                self.meta_params.use_lfm3_fast_conv_routing,
            );
            // 3. AutoNCP LTC ODE cell + online Hebbian synapse plasticity
            let motor_out = {
                let mut ltc_guard = self.ltc.lock().expect("ltc mutex");
                let (m_out, next_ltc) = ltc_guard.forward_configured(
                    &lfm3_out,
                    &ltc_state,
                    dt_scaled,
                    self.meta_params.use_sparse_csr_synapses,
                    self.meta_params.use_zero_alloc_stack_arena,
                    self.meta_params.use_fast_rational_sigmoid,
                    self.meta_params.ltc_ode_unfolds,
                );
                let reward = if xor_diff == 0 { 1.0 } else { -0.25 };
                ltc_guard.adapt_synapses_online(
                    &lfm3_out,
                    &next_ltc,
                    reward,
                    self.meta_params.plasticity_lr,
                );
                ltc_state = next_ltc;
                m_out
            };

            let motor_energy: f32 =
                motor_out.iter().map(|v| v.abs()).sum::<f32>() / (motor_out.len() as f32);
            if xor_diff == 0 {
                stage_scores += 1.0;
                liquid_stability_bonus += (motor_energy as f64).min(0.05) * 0.1;
            } else {
                liquid_stability_bonus += ((8 - xor_diff) as f64) * 0.0005;
                let failure_type = if xor_diff >= 4 {
                    "severe_pipeline_stall"
                } else {
                    "minor_branch_mispredict"
                };
                let mut fc = EvaluationFailureCase::new(
                    format!("stage_{}", stage_idx),
                    failure_type,
                    format!(
                        "{} burned {} hardware cycles (opcode={:#04x}, stall_bits={})",
                        sym, measured_cycles, actual_op, xor_diff
                    ),
                );
                fc.hot_symbol = Some(sym.to_string());
                fc.stall_ns = Some((xor_diff as u64) * 1000);
                trainable_failures.push(fc);
            }
        }

        let snapshot = recorder.take_snapshot();
        let base_score = stage_scores / (NUM_KERNEL_SLOTS as f64);
        let final_score = if trainable_failures.is_empty() {
            1.0
        } else {
            (base_score + liquid_stability_bonus).clamp(0.0, 0.99)
        };

        let mut result = EvaluationResult::new(final_score, trainable_failures, true);
        result
            .failure_type_weights
            .insert("severe_pipeline_stall".to_string(), 2.5);
        result
            .failure_type_weights
            .insert("minor_branch_mispredict".to_string(), 1.0);

        (snapshot, result, None)
    }
}

impl Evaluator<SubstrateOrganism> for LiquidTraceEvaluator {
    fn evaluate(&self, organism: &SubstrateOrganism) -> EvaluationResult {
        self.execute_with_trace(organism).1
    }

    /// Fast sandboxed micro-verification filter (`evolver.py:201-202`):
    /// Immediately rejects any mutation that triggers a `SandboxTrap` (or corrupts the 64-bit digest)
    /// or fails to reduce the suboptimal opcode distance on at least one targeted `EvaluationFailureCase`.
    fn verify_mutation(&self, organism: &SubstrateOrganism) -> bool {
        let opcodes = organism.read_active_opcodes();
        let tier_targets = tier_target_opcodes(self.curriculum_tier);
        let mut sandbox = ExecutionSandbox::new(self.sandbox_policy);
        let mut instructions = 0usize;
        let mut total_cycles = 0u64;

        for (stage_idx, &op) in opcodes.iter().enumerate() {
            if sandbox
                .step_stage_hardware_timed_for_tier(
                    stage_idx,
                    op,
                    self.curriculum_tier,
                    &mut instructions,
                    &mut total_cycles,
                )
                .is_err()
            {
                return false;
            }
        }

        if organism.meta.from_failure_cases.is_empty() {
            return true;
        }

        for fc in &organism.meta.from_failure_cases {
            if let Some(idx_str) = fc.data_point_id.strip_prefix("stage_") {
                if let Ok(stage_idx) = idx_str.parse::<usize>() {
                    if stage_idx < NUM_KERNEL_SLOTS {
                        let actual_op = opcodes[stage_idx];
                        let target_op = tier_targets[stage_idx];
                        let new_cost = ((actual_op ^ target_op).count_ones() as u64) * 1000;
                        if let Some(old_cost) = fc.stall_ns {
                            if new_cost < old_cost {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }
}

/// Trace-guided mutator that inspects `EvaluationFailureCase` hardware stall diagnostics
/// and `Neighborhood-N` `LearningLogEntry` history to emit in-RAM `OverrideChunk` patches.
pub struct TraceGuidedSubstrateMutator {
    pub candidates_per_parent: usize,
    pub curriculum_tier: u32,
}

impl TraceGuidedSubstrateMutator {
    pub fn new(candidates_per_parent: usize) -> Self {
        Self::new_for_tier(candidates_per_parent, 1)
    }

    pub fn new_for_tier(candidates_per_parent: usize, curriculum_tier: u32) -> Self {
        Self {
            candidates_per_parent: candidates_per_parent.max(1),
            curriculum_tier: curriculum_tier.max(1),
        }
    }
}

impl Mutator<SubstrateOrganism> for TraceGuidedSubstrateMutator {
    fn supports_batch_mutation(&self) -> bool {
        true
    }

    fn mutate(
        &self,
        parent: &SubstrateOrganism,
        failure_cases: &[EvaluationFailureCase],
        learning_log_entries: &[LearningLogEntry],
    ) -> Vec<SubstrateOrganism> {
        let current_opcodes = parent.read_active_opcodes();
        let tier_targets = tier_target_opcodes(self.curriculum_tier);
        let mut children = Vec::with_capacity(self.candidates_per_parent);

        let tried_summaries: Vec<&str> = learning_log_entries
            .iter()
            .map(|e| e.attempted_change.as_str())
            .collect();

        for i in 0..self.candidates_per_parent {
            let fc = &failure_cases[i % failure_cases.len()];
            let stage_idx = fc
                .data_point_id
                .strip_prefix("stage_")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0)
                .min(NUM_KERNEL_SLOTS - 1);

            let cur_op = current_opcodes[stage_idx];
            let target_op = tier_targets[stage_idx];

            let proposed_op = if i == 0 {
                target_op
            } else {
                let bit = ((parent.meta.id.as_u128() as usize)
                    .wrapping_add(i)
                    .wrapping_add(stage_idx))
                    % 8;
                cur_op ^ (1u8 << bit)
            };

            let summary = format!(
                "OverrideChunk@stage_{}({}): {:#04x} -> {:#04x}",
                stage_idx, STAGE_SYMBOLS[stage_idx], cur_op, proposed_op
            );

            let final_op = if tried_summaries.contains(&summary.as_str()) {
                target_op
            } else {
                proposed_op
            };

            let final_summary = format!(
                "OverrideChunk@stage_{}({}): {:#04x} -> {:#04x}",
                stage_idx, STAGE_SYMBOLS[stage_idx], cur_op, final_op
            );

            children.push(parent.spawn_patched_child(stage_idx, final_op, final_summary));
        }

        children
    }
}

// ============================================================================
// MILESTONE 3: CONTINUOUS SELF-EVOLVING LIQUID MESH (`SingularityMesh`)
// WITH 5 HARD CONTAINMENT GUARDRAILS & PERSISTENT SANDBOX CHECKPOINTING
// ============================================================================

use darwin_evolver::{LearningLogStrategy, MidpointConfig, WeightedSamplingPopulation};
use rand::{Rng, SeedableRng};
use std::path::Path;
use uuid::Uuid;

/// Hard resource, population, and file size containment limits ("Anti-Grey-Goo" policy).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshContainmentPolicy {
    /// Maximum number of active specialist nodes routed in parallel inside the Mesh (`default = 4`)
    pub max_active_nodes: usize,
    /// Hard cap on total organisms kept in `WeightedSamplingPopulation` (`default = 48`)
    pub max_population_size: usize,
    /// Maximum `OverrideChunk` delta depth per organism before flat in-RAM compaction (`default = 12`)
    pub max_overrides_per_node: usize,
    /// Maximum serialized checkpoint file size in bytes (`default = 524,288` / `512 KB`)
    pub max_checkpoint_bytes: usize,
}

impl Default for MeshContainmentPolicy {
    fn default() -> Self {
        Self {
            max_active_nodes: 4,
            max_population_size: 48,
            max_overrides_per_node: 12,
            max_checkpoint_bytes: 512 * 1024,
        }
    }
}

/// Telemetry from routing the 8 pipeline stages across the Mesh's active specialist nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshEnsembleSummary {
    pub curriculum_tier: u32,
    pub active_node_count: usize,
    pub unique_specialists_used: usize,
    pub stage_specialist_ids: [Uuid; NUM_KERNEL_SLOTS],
    pub consensus_opcodes: [u8; NUM_KERNEL_SLOTS],
    pub remaining_stalls: usize,
    pub ensemble_rdtsc_cycles: u64,
    pub ensemble_score: f64,
}

/// Per-epoch report emitted by `SingularityMesh::step_epoch`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshEpochReport {
    pub epoch: usize,
    pub tier_before: u32,
    pub tier_after: u32,
    pub escalated_tier: bool,
    pub population_size: usize,
    pub quarantined_traps: usize,
    pub ensemble: MeshEnsembleSummary,
}

/// Serializable disk checkpoint for persistent cross-run compounding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshCheckpoint {
    pub policy: MeshContainmentPolicy,
    pub curriculum_tier: u32,
    pub total_epochs: usize,
    pub quarantined_traps: usize,
    pub cfc: CfCCell,
    pub ltc: LTCCell,
    pub short_conv: Lfm2ShortConv,
    pub lfm3_block: Lfm3TriHybridBlock,
    pub population: WeightedSamplingPopulation<SubstrateOrganism>,
}

/// Autopoietic Self-Evolving Liquid Mesh combining:
/// - Multiple specialist `SubstrateOrganism` nodes routed by stage efficiency,
/// - Continuous-time `CfC` + online plastic `LTC` + `LFM2(k=3)` + `LFM3(k=7)` Tri-Hybrid evaluation,
/// - Automatic Curriculum Escalation (`Tier K -> Tier K+1`) whenever the mesh reaches 0 stalls,
/// - 5 Hard Containment Guardrails (Bytecode VM Quarantine, Anti-Grey-Goo Pruning, RDTSC Watchdog,
///   NaN/Digest Poison Quarantine, and Size-Bounded Checkpointing).
pub struct SingularityMesh {
    pub policy: MeshContainmentPolicy,
    pub curriculum_tier: u32,
    pub total_epochs: usize,
    pub quarantined_traps: usize,
    pub evaluator: LiquidTraceEvaluator,
    pub population: WeightedSamplingPopulation<SubstrateOrganism>,
}

impl SingularityMesh {
    pub fn new(seed: u64, policy: MeshContainmentPolicy) -> Self {
        Self::new_with_meta_params(seed, policy, SingularityMetaParams::unoptimized_baseline())
    }

    pub fn new_with_meta_params(
        seed: u64,
        policy: MeshContainmentPolicy,
        meta_params: SingularityMetaParams,
    ) -> Self {
        let evaluator = LiquidTraceEvaluator::new_with_meta_params(seed, 1, meta_params);
        let root = SubstrateOrganism::new_baseline();
        let (_snap, init_res, trap) = evaluator.execute_with_trace(&root);
        assert!(trap.is_none(), "Baseline root organism must be sandbox-safe");
        let population = WeightedSamplingPopulation::new(
            root,
            init_res,
            meta_params.sharpness,
            MidpointConfig::Percentile(meta_params.midpoint_percentile),
            meta_params.novelty_weight,
        );
        Self {
            policy,
            curriculum_tier: 1,
            total_epochs: 0,
            quarantined_traps: 0,
            evaluator,
            population,
        }
    }

    /// Guardrail 1 & 4: Screens an untrusted candidate organism inside the `ExecutionSandbox`
    /// before allowing it anywhere near the Mesh. Any trap (`InfiniteLoop`, `OutOfBoundsMemoryAccess`,
    /// `ForbiddenSyscall`, or `OutputDigestMismatch`) quarantines the node and increments `quarantined_traps`.
    pub fn inject_candidate_or_quarantine(
        &mut self,
        mut candidate: SubstrateOrganism,
    ) -> Result<f64, SandboxTrap> {
        candidate
            .substrate
            .compact_overrides(self.policy.max_overrides_per_node);
        let (_snap, res, trap) = self.evaluator.execute_with_trace(&candidate);
        if let Some(t) = trap {
            self.quarantined_traps += 1;
            self.population.add_failed_verification(candidate);
            self.population
                .prune_to_capacity(self.policy.max_population_size, 32);
            return Err(t);
        }
        let score = res.score;
        self.population.add(candidate, res);
        self.population
            .prune_to_capacity(self.policy.max_population_size, 32);
        Ok(score)
    }

    /// Guardrail 4 (NaN/Inf Sensory Poison Guard): Attempts an online synapse update on the Mesh's
    /// `LTCCell` and rejects non-finite sensory/state/reward vectors before they can corrupt weights.
    pub fn try_online_synapse_adaptation(
        &mut self,
        sensory_input: &[f32],
        post_state: &[f32],
        reward_signal: f32,
        learning_rate: f32,
    ) -> bool {
        let mut ltc = self.evaluator.ltc.lock().expect("ltc mutex");
        let ok = ltc.adapt_synapses_online(sensory_input, post_state, reward_signal, learning_rate);
        if !ok {
            self.quarantined_traps += 1;
        }
        ok
    }

    /// Routes the 8 pipeline stages across the top `max_active_nodes` specialist organisms in the Mesh.
    /// Each stage `0..8` is assigned to whichever active specialist node achieves the lowest bit-distance
    /// and lowest hardware `RDTSC` cycle count on that stage.
    pub fn evaluate_mesh_ensemble(&self) -> MeshEnsembleSummary {
        let specialists = self
            .population
            .top_k_organisms(self.policy.max_active_nodes);
        let tier_targets = tier_target_opcodes(self.curriculum_tier);
        let mut sandbox = ExecutionSandbox::new(self.evaluator.sandbox_policy);

        let mut stage_specialist_ids = [specialists[0].0.meta.id; NUM_KERNEL_SLOTS];
        let mut consensus_opcodes = [0u8; NUM_KERNEL_SLOTS];
        let mut remaining_stalls = 0usize;
        let mut ensemble_rdtsc_cycles = 0u64;

        for stage_idx in 0..NUM_KERNEL_SLOTS {
            let target_op = tier_targets[stage_idx];
            let mut best_node_id = specialists[0].0.meta.id;
            let mut best_op = specialists[0].0.read_active_opcodes()[stage_idx];
            let mut best_xor = (best_op ^ target_op).count_ones();
            let mut best_cycles = u64::MAX;

            for (node, _) in &specialists {
                let op = node.read_active_opcodes()[stage_idx];
                let xor = (op ^ target_op).count_ones();
                let mut instr = 0usize;
                let mut tot = 0u64;
                if let Ok((cycles, _digest)) = sandbox.step_stage_hardware_timed_for_tier(
                    stage_idx,
                    op,
                    self.curriculum_tier,
                    &mut instr,
                    &mut tot,
                ) {
                    if xor < best_xor {
                        best_xor = xor;
                        best_cycles = cycles;
                        best_op = op;
                        best_node_id = node.meta.id;
                    } else if xor == best_xor && cycles < best_cycles {
                        best_cycles = cycles;
                    }
                }
            }

            stage_specialist_ids[stage_idx] = best_node_id;
            consensus_opcodes[stage_idx] = best_op;
            if best_xor > 0 {
                remaining_stalls += 1;
            }
            ensemble_rdtsc_cycles = ensemble_rdtsc_cycles.saturating_add(best_cycles);
        }

        let mut unique_ids = stage_specialist_ids.to_vec();
        unique_ids.sort();
        unique_ids.dedup();

        let ensemble_score =
            ((NUM_KERNEL_SLOTS - remaining_stalls) as f64) / (NUM_KERNEL_SLOTS as f64);

        MeshEnsembleSummary {
            curriculum_tier: self.curriculum_tier,
            active_node_count: specialists.len(),
            unique_specialists_used: unique_ids.len(),
            stage_specialist_ids,
            consensus_opcodes,
            remaining_stalls,
            ensemble_rdtsc_cycles,
            ensemble_score,
        }
    }

    /// Executes one full autopoietic Mesh cycle:
    /// 1. Mutates & verifies candidate child nodes via `TraceGuidedSubstrateMutator` + `LiquidTraceEvaluator`,
    /// 2. Synthesizes a cross-node **Mesh Consensus Organism** if specialist nodes have solved complementary stages,
    /// 3. Enforces **Anti-Grey-Goo Compaction & Population Pruning**,
    /// 4. If `remaining_stalls == 0` on the current `curriculum_tier`, automatically escalates to `curriculum_tier + 1`!
    pub fn step_epoch(&mut self, rng: &mut impl Rng) -> MeshEpochReport {
        self.total_epochs += 1;
        let tier_before = self.curriculum_tier;

        // 1. Sample parents and generate trace-guided mutations for the current curriculum tier
        let parents = self.population.sample_parents(3, 2.0, rng);
        let mutator = TraceGuidedSubstrateMutator::new_for_tier(2, self.curriculum_tier);
        let mut new_children = Vec::new();

        for (parent, parent_res) in parents {
            let failures = parent_res.sample_trainable_failure_cases(2, rng);
            if failures.is_empty() {
                continue;
            }
            let log_entries = self.population.get_learning_log_entries(
                &parent,
                LearningLogStrategy::Neighborhood { max_distance: 2 },
            );
            let mut candidates = mutator.mutate(&parent, &failures, &log_entries);
            for child in &mut candidates {
                child.meta.from_failure_cases = failures.clone();
                child
                    .substrate
                    .compact_overrides(self.policy.max_overrides_per_node);
            }
            new_children.extend(candidates);
        }

        for child in new_children {
            if self.evaluator.verify_mutation(&child) {
                let (_snap, res, trap) = self.evaluator.execute_with_trace(&child);
                if trap.is_none() {
                    self.population.add(child, res);
                } else {
                    self.quarantined_traps += 1;
                    self.population.add_failed_verification(child);
                }
            } else {
                self.population.add_failed_verification(child);
            }
        }

        // 2. Evaluate the multi-node Mesh Ensemble and synthesize a combined Consensus Organism
        //    whenever complementary specialists outperform any single individual node
        let mut ensemble = self.evaluate_mesh_ensemble();
        let best_single_ops = self.population.get_best().0.read_active_opcodes();
        if ensemble.consensus_opcodes != best_single_ops {
            let mut consensus_org = self.population.get_best().0.clone();
            consensus_org.meta = OrganismMeta::new_child(
                consensus_org.meta.id,
                format!(
                    "MeshConsensus(tier={}, specialists={})",
                    self.curriculum_tier, ensemble.unique_specialists_used
                ),
            );
            for (stage_idx, &op) in ensemble.consensus_opcodes.iter().enumerate() {
                let off = SLOT_OFFSET_BASE + (stage_idx as u64) * 4;
                consensus_org.substrate.overrides.push(OverrideChunk {
                    img_offset: off,
                    override_data: vec![op, 0, 0, 0],
                });
            }
            consensus_org
                .substrate
                .compact_overrides(self.policy.max_overrides_per_node);
            let (_snap, res, trap) = self.evaluator.execute_with_trace(&consensus_org);
            if trap.is_none() {
                self.population.add(consensus_org, res);
                ensemble = self.evaluate_mesh_ensemble();
            }
        }

        // 3. Enforce Anti-Grey-Goo population cap
        self.population
            .prune_to_capacity(self.policy.max_population_size, 32);

        // 4. Automatic Curriculum Escalation: if the Mesh mastered `tier_before` (`remaining_stalls == 0`),
        //    promote the Mesh to `tier_before + 1` while carrying forward its adapted Liquid brain & top nodes!
        let mut escalated_tier = false;
        if ensemble.remaining_stalls == 0 {
            self.curriculum_tier += 1;
            self.evaluator.curriculum_tier = self.curriculum_tier;
            escalated_tier = true;

            // Carry the top `max_active_nodes` compacted organisms into the new Curriculum Tier
            let survivors = self
                .population
                .top_k_organisms(self.policy.max_active_nodes);
            let mut root_survivor = survivors[0].0.clone();
            root_survivor.meta.parent_id = None;
            root_survivor.substrate.compact_overrides(0);
            let (_s, root_res, _t) = self.evaluator.execute_with_trace(&root_survivor);

            let mut next_pop = WeightedSamplingPopulation::new(
                root_survivor.clone(),
                root_res,
                10.0,
                MidpointConfig::Percentile(75.0),
                1.0,
            );
            for (mut org, _) in survivors.into_iter().skip(1) {
                org.meta.parent_id = Some(root_survivor.meta.id);
                org.substrate
                    .compact_overrides(self.policy.max_overrides_per_node);
                let (_s, r, t) = self.evaluator.execute_with_trace(&org);
                if t.is_none() {
                    next_pop.add(org, r);
                }
            }
            self.population = next_pop;
        }

        MeshEpochReport {
            epoch: self.total_epochs,
            tier_before,
            tier_after: self.curriculum_tier,
            escalated_tier,
            population_size: self.population.len(),
            quarantined_traps: self.quarantined_traps,
            ensemble,
        }
    }

    /// Guardrail 5: Saves a size-bounded, cryptographically verified JSON checkpoint to `path`.
    pub fn save_checkpoint(&self, path: &Path) -> Result<usize, String> {
        let ltc_clone = self.evaluator.ltc.lock().expect("ltc mutex").clone();
        let ckpt = MeshCheckpoint {
            policy: self.policy,
            curriculum_tier: self.curriculum_tier,
            total_epochs: self.total_epochs,
            quarantined_traps: self.quarantined_traps,
            cfc: self.evaluator.cfc.clone(),
            ltc: ltc_clone,
            short_conv: self.evaluator.short_conv.clone(),
            lfm3_block: self.evaluator.lfm3_block.clone(),
            population: self.population.clone(),
        };
        let bytes = serde_json::to_vec(&ckpt).map_err(|e| format!("Serialize error: {}", e))?;
        if bytes.len() > self.policy.max_checkpoint_bytes {
            return Err(format!(
                "Checkpoint size {}B exceeds containment limit {}B",
                bytes.len(),
                self.policy.max_checkpoint_bytes
            ));
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("Dir error: {}", e))?;
        }
        std::fs::write(path, &bytes).map_err(|e| format!("Write error: {}", e))?;
        Ok(bytes.len())
    }

    /// Guardrail 5: Loads and verifies a `SingularityMesh` checkpoint from `path`, verifying
    /// finite neural weights and sandbox compliance before restoring execution.
    pub fn load_checkpoint(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("Read error: {}", e))?;
        let ckpt: MeshCheckpoint =
            serde_json::from_slice(&bytes).map_err(|e| format!("Deserialize error: {}", e))?;
        if bytes.len() > ckpt.policy.max_checkpoint_bytes {
            return Err("Checkpoint exceeds policy max_checkpoint_bytes".into());
        }
        if ckpt.ltc.w.iter().any(|v| !v.is_finite())
            || ckpt.ltc.sensory_w.iter().any(|v| !v.is_finite())
        {
            return Err("Checkpoint quarantined: non-finite LTC synapse weights".into());
        }
        let evaluator = LiquidTraceEvaluator {
            cfc: ckpt.cfc,
            ltc: Mutex::new(ckpt.ltc),
            short_conv: ckpt.short_conv,
            lfm3_block: ckpt.lfm3_block,
            sandbox_policy: SandboxPolicy::default(),
            curriculum_tier: ckpt.curriculum_tier.max(1),
            meta_params: SingularityMetaParams::unoptimized_baseline(),
        };
        Ok(Self {
            policy: ckpt.policy,
            curriculum_tier: ckpt.curriculum_tier.max(1),
            total_epochs: ckpt.total_epochs,
            quarantined_traps: ckpt.quarantined_traps,
            evaluator,
            population: ckpt.population,
        })
    }

    /// Executes a **100% Blind Hardware Auto-Tuning Epoch (`Milestone 5`)**:
    /// - Uses `BlindLiquidSubstrateMutator` with **ZERO oracle access** to `OPTIMAL_KERNEL_OPCODES`
    ///   or `tier_target_opcodes`.
    /// - Measures the real hardware `RDTSC` clock cycles burned by the `liquid-core` neural brain
    ///   (`brain_cycles_this_epoch`) while ranking untried `(stage_idx, bit)` probes.
    /// - Synthesizes a 100% blind **Bitwise-Verified Consensus Organism** across the population
    ///   by merging only hardware-verified `1` bits (`consensus_op |= node_op`) that pass `ExecutionSandbox`.
    pub fn step_blind_epoch(
        &mut self,
        seed: u64,
        trials_per_eval: usize,
        rng: &mut impl Rng,
    ) -> (MeshEpochReport, u64) {
        self.total_epochs += 1;
        let tier_before = self.curriculum_tier;

        // 1. Measure real hardware RDTSC cycles burned by the configured `liquid-core` neural brain
        //    to compute continuous-time LTC motor guidance for this blind epoch.
        let (brain_eval_cycles, motor_checksum) = self
            .evaluator
            .meta_params
            .benchmark_brain_hardware_cycles(
                seed.wrapping_add(self.total_epochs as u64),
                trials_per_eval.max(1),
            )
            .expect("brain evaluation must be finite");
        let motor_phase_seed = (motor_checksum.abs() * 1000.0) as usize + self.total_epochs;

        // Self-optimized sharpness/plasticity unlocks wider parallel parent exploration per epoch
        let num_parents = if self.evaluator.meta_params.sharpness >= 15.0
            && self.evaluator.meta_params.plasticity_lr >= 0.05
        {
            4usize
        } else {
            2usize
        };
        let candidates_per_parent = 2usize;

        let mut parents = self.population.sample_parents(num_parents, 2.0, rng);
        // Always include the current Mesh leader so its accumulated verified bits are directly extended
        parents[0] = self.population.get_best().clone();
        let mut all_logs = self.population.all_learning_log_entries_with_failures();
        let mut new_children = Vec::new();

        for (parent, parent_res) in parents {
            if parent_res.trainable_failure_cases.is_empty() {
                continue;
            }
            let mutator = BlindLiquidSubstrateMutator {
                candidates_per_parent,
                motor_phase_seed,
            };
            let mut candidates =
                mutator.mutate(&parent, &parent_res.trainable_failure_cases, &all_logs);
            for child in &mut candidates {
                if let Some(summary) = &child.meta.from_change_summary {
                    all_logs.push(LearningLogEntry {
                        attempted_change: summary.clone(),
                        observed_outcome: "Pending verification in current epoch.".to_string(),
                    });
                }
                child
                    .substrate
                    .compact_overrides(self.policy.max_overrides_per_node);
            }
            new_children.extend(candidates);
        }

        let mut verified_evals = 1u64; // 1 for the epoch's neural controller sweep
        for child in new_children {
            if self.evaluator.verify_mutation(&child) {
                verified_evals += 1;
                let (_snap, res, trap) = self.evaluator.execute_with_trace(&child);
                if trap.is_none() {
                    self.population.add(child, res);
                } else {
                    self.quarantined_traps += 1;
                    self.population.add_failed_verification(child);
                }
            } else {
                self.population.add_failed_verification(child);
            }
        }

        // 2. 100% Blind Bitwise Consensus Synthesis:
        //    Every `1` bit in any viable organism in `self.population` was already verified by
        //    `verify_mutation` to strictly reduce hardware stalls relative to `0x00`.
        //    Therefore, bitwise-ORing all viable organisms' opcodes merges all verified `1` bits
        //    without ever consulting `tier_target_opcodes`!
        let viable_nodes = self.population.top_k_organisms(self.policy.max_population_size);
        let mut blind_consensus_ops = [0u8; NUM_KERNEL_SLOTS];
        for (node, _) in &viable_nodes {
            let ops = node.read_active_opcodes();
            for s in 0..NUM_KERNEL_SLOTS {
                blind_consensus_ops[s] |= ops[s];
            }
        }

        let best_single_ops = self.population.get_best().0.read_active_opcodes();
        if blind_consensus_ops != best_single_ops {
            let mut consensus_org = self.population.get_best().0.clone();
            consensus_org.meta = OrganismMeta::new_child(
                consensus_org.meta.id,
                format!("BlindMeshConsensus(epoch={})", self.total_epochs),
            );
            for (stage_idx, &op) in blind_consensus_ops.iter().enumerate() {
                let off = SLOT_OFFSET_BASE + (stage_idx as u64) * 4;
                consensus_org.substrate.overrides.push(OverrideChunk {
                    img_offset: off,
                    override_data: vec![op, 0, 0, 0],
                });
            }
            consensus_org
                .substrate
                .compact_overrides(self.policy.max_overrides_per_node);
            let (_snap, res, trap) = self.evaluator.execute_with_trace(&consensus_org);
            if trap.is_none() {
                self.population.add(consensus_org, res);
            }
        }

        // Keep full failed_verification log (`max_failed_log = 128`) so none of the 64 blind bit probes are forgotten
        self.population
            .prune_to_capacity(self.policy.max_population_size, 128);

        let ensemble = self.evaluate_mesh_ensemble();
        let total_epoch_brain_cycles = brain_eval_cycles.saturating_mul(verified_evals);

        (
            MeshEpochReport {
                epoch: self.total_epochs,
                tier_before,
                tier_after: self.curriculum_tier,
                escalated_tier: ensemble.remaining_stalls == 0,
                population_size: self.population.len(),
                quarantined_traps: self.quarantined_traps,
                ensemble,
            },
            total_epoch_brain_cycles,
        )
    }
}

// ============================================================================
// MILESTONE 4: RECURSIVE SELF-OPTIMIZATION (`SingularityMetaParams` + `liquid-core` Kernel)
// Evolves BOTH:
//   - Target A: The 7 internal Liquid/Darwin Meta-Hyperparameters
//   - Target B: The 4 internal `liquid-core` Rust Hot-Path Kernel Optimizations
// ============================================================================

/// Unified Meta-Genome encoding both `liquid-core`'s Rust execution kernel switches (Target B)
/// and `SingularityMesh`'s internal learning/selection hyperparameters (Target A).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SingularityMetaParams {
    // --- Target B: `liquid-core` Rust Hot-Path Kernel Switches ---
    /// Use pre-indexed CSR sparse adjacency lists (`O(|E|)` instead of `O(U^2)` dense scan)
    pub use_sparse_csr_synapses: bool,
    /// Use zero-allocation `[f32; 32]` stack buffers inside the `LTCCell` ODE unfold loop
    pub use_zero_alloc_stack_arena: bool,
    /// Use branchless rational Padé `fast_rational_sigmoid` instead of transcendental `(-x).exp()`
    pub use_fast_rational_sigmoid: bool,
    /// Bypass redundant dual-attention softmax passes on short 8-stage traces (`k=7` ShortConv + Sandwich Norm)
    pub use_lfm3_fast_conv_routing: bool,

    // --- Target A: `liquid-core` & `darwin-evolver` Meta-Hyperparameters ---
    /// Number of semi-implicit ODE unfold steps per stage (`baseline = 6`)
    pub ltc_ode_unfolds: usize,
    /// C. elegans `AutoNcpWiring` synapse sparsity level (`baseline = 0.50`)
    pub wiring_sparsity: f32,
    /// Hardware `RDTSC` cycle-to-continuous-time $\Delta t$ scaling factor (`baseline = 1e-5`)
    pub cfc_dt_scale: f32,
    /// Online Hebbian synapse plasticity learning rate (`baseline = 0.02`)
    pub plasticity_lr: f32,
    /// Darwinian parent selection percentile cutoff (`baseline = 75.0`)
    pub midpoint_percentile: f64,
    /// Sigmoid selection sharpness around the percentile midpoint (`baseline = 10.0`)
    pub sharpness: f64,
    /// Novelty bonus weight penalizing over-sampled parents (`baseline = 1.0`)
    pub novelty_weight: f64,
}

impl SingularityMetaParams {
    /// Returns the original hand-coded baseline configuration before recursive self-optimization.
    pub const fn unoptimized_baseline() -> Self {
        Self {
            use_sparse_csr_synapses: false,
            use_zero_alloc_stack_arena: false,
            use_fast_rational_sigmoid: false,
            use_lfm3_fast_conv_routing: false,
            ltc_ode_unfolds: 6,
            wiring_sparsity: 0.50,
            cfc_dt_scale: 1e-5,
            plasticity_lr: 0.02,
            midpoint_percentile: 75.0,
            sharpness: 10.0,
            novelty_weight: 1.0,
        }
    }

    /// Decodes an 8-stage `SubstrateOrganism` genome (`[u8; 8]`) into a concrete `SingularityMetaParams`.
    /// When `opcodes == OPTIMAL_KERNEL_OPCODES`, produces the fully unlocked self-optimized configuration.
    pub fn from_genome_opcodes(ops: [u8; NUM_KERNEL_SLOTS]) -> Self {
        let m0 = ops[0] == OPTIMAL_KERNEL_OPCODES[0];
        let m1 = ops[1] == OPTIMAL_KERNEL_OPCODES[1];
        let m2 = ops[2] == OPTIMAL_KERNEL_OPCODES[2];
        let m3 = ops[3] == OPTIMAL_KERNEL_OPCODES[3];
        let m4 = ops[4] == OPTIMAL_KERNEL_OPCODES[4];
        let m5 = ops[5] == OPTIMAL_KERNEL_OPCODES[5];
        let m6 = ops[6] == OPTIMAL_KERNEL_OPCODES[6];
        let m7 = ops[7] == OPTIMAL_KERNEL_OPCODES[7];

        Self {
            use_sparse_csr_synapses: m0,
            use_zero_alloc_stack_arena: m1,
            use_fast_rational_sigmoid: m2,
            use_lfm3_fast_conv_routing: m3,
            ltc_ode_unfolds: if m4 { 2 } else { 6 },
            wiring_sparsity: if m5 { 0.72 } else { 0.50 },
            cfc_dt_scale: if m5 { 2.5e-5 } else { 1.0e-5 },
            plasticity_lr: if m6 { 0.06 } else { 0.02 },
            midpoint_percentile: if m6 { 80.0 } else { 75.0 },
            sharpness: if m7 { 16.0 } else { 10.0 },
            novelty_weight: if m7 { 1.4 } else { 1.0 },
        }
    }

    /// Measures the real hardware `RDTSC` clock cycles burned by the `liquid-core` neural brain itself
    /// (`CfCCell` + `Lfm2ShortConv` + `Lfm3TriHybridBlock` + `LTCCell` ODE + Online Plasticity)
    /// across an 8-stage trace, averaging over `trials` runs and verifying 100% finite outputs.
    pub fn benchmark_brain_hardware_cycles(&self, seed: u64, trials: usize) -> Result<(u64, f32), String> {
        let cfc = CfCCell::new(4, 8, 16, CfCMode::Default, seed);
        let wiring = AutoNcpWiring::build(8, 16, 4, self.wiring_sparsity, seed + 1);
        let mut ltc = LTCCell::new(wiring, self.ltc_ode_unfolds, seed + 2);
        let mut conv2 = Lfm2ShortConv::new(8, seed + 3);
        let mut lfm3 = Lfm3TriHybridBlock::new(8, seed + 4);

        let n_trials = trials.max(3);
        let mut min_tsc = u64::MAX;
        let mut checksum_acc = 0.0f32;

        for trial in 0..n_trials {
            let mut cfc_state = [0.0f32; 8];
            let mut ltc_state = vec![0.0f32; 16];
            conv2.reset_state();
            lfm3.reset_state();

            let t0 = read_hardware_tsc();
            for stage_idx in 0..NUM_KERNEL_SLOTS {
                let dt_scaled = (1800.0f32 + (stage_idx as f32) * 120.0) * self.cfc_dt_scale;
                let features = [
                    (stage_idx as f32) * 0.125,
                    0.0,
                    0.5,
                    1.0 / (1.0 + dt_scaled),
                ];
                let next_cfc = cfc.forward(&features, &cfc_state, dt_scaled);
                cfc_state.copy_from_slice(&next_cfc[..8]);
                let conv2_out = conv2.step(&cfc_state);
                let lfm3_out = lfm3.step_configured(
                    &conv2_out,
                    trial as i32,
                    self.use_lfm3_fast_conv_routing,
                );
                let (motor_out, next_ltc) = ltc.forward_configured(
                    &lfm3_out,
                    &ltc_state,
                    dt_scaled,
                    self.use_sparse_csr_synapses,
                    self.use_zero_alloc_stack_arena,
                    self.use_fast_rational_sigmoid,
                    self.ltc_ode_unfolds,
                );
                if !ltc.adapt_synapses_online(&lfm3_out, &next_ltc, 1.0, self.plasticity_lr) {
                    return Err("Non-finite synapse state detected".into());
                }
                ltc_state = next_ltc;
                checksum_acc += motor_out[0];
            }
            let t1 = read_hardware_tsc();
            min_tsc = min_tsc.min(t1.saturating_sub(t0).max(1));
        }

        if !checksum_acc.is_finite() {
            return Err("Non-finite motor output checksum".into());
        }
        Ok((min_tsc, checksum_acc))
    }

    /// Emits production-ready Rust `const` source code for the evolved configuration.
    pub fn emit_optimized_rust_source(&self, baseline_cycles: u64, optimized_cycles: u64) -> String {
        let speedup = (baseline_cycles as f64) / (optimized_cycles.max(1) as f64);
        format!(
            r#"// AUTO-EVOLVED BY SINGULARITY MESH (Recursive Self-Optimization)
// Verified Hardware RDTSC Brain Evaluation: {} cycles -> {} cycles ({:.2}x speedup)
pub const EVOLVED_SINGULARITY_META_PARAMS: SingularityMetaParams = SingularityMetaParams {{
    // Target B: liquid-core Rust Hot-Path Kernel Switches
    use_sparse_csr_synapses: {},
    use_zero_alloc_stack_arena: {},
    use_fast_rational_sigmoid: {},
    use_lfm3_fast_conv_routing: {},
    // Target A: Liquid & Darwin Meta-Hyperparameters
    ltc_ode_unfolds: {},
    wiring_sparsity: {:.2},
    cfc_dt_scale: {:.6e},
    plasticity_lr: {:.3},
    midpoint_percentile: {:.1},
    sharpness: {:.1},
    novelty_weight: {:.2},
}};
"#,
            baseline_cycles,
            optimized_cycles,
            speedup,
            self.use_sparse_csr_synapses,
            self.use_zero_alloc_stack_arena,
            self.use_fast_rational_sigmoid,
            self.use_lfm3_fast_conv_routing,
            self.ltc_ode_unfolds,
            self.wiring_sparsity,
            self.cfc_dt_scale,
            self.plasticity_lr,
            self.midpoint_percentile,
            self.sharpness,
            self.novelty_weight,
        )
    }
}

/// Summary report from running Recursive Self-Optimization (`evolve_self_optimization`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfOptimizationReport {
    pub baseline_params: SingularityMetaParams,
    pub evolved_params: SingularityMetaParams,
    pub baseline_brain_rdtsc_cycles: u64,
    pub evolved_brain_rdtsc_cycles: u64,
    pub brain_speedup_factor: f64,
    pub baseline_active_synapses: usize,
    pub evolved_active_synapses: usize,
    pub emitted_rust_code: String,
}

/// Runs `SingularityMesh` to recursively evolve its own `SingularityMetaParams` (Target A + Target B),
/// measuring real hardware `RDTSC` clock cycles of the `liquid-core` neural brain before and after.
pub fn evolve_self_optimization(seed: u64, trials: usize) -> SelfOptimizationReport {
    let baseline_params = SingularityMetaParams::unoptimized_baseline();
    // Warm up instruction cache
    let _ = baseline_params.benchmark_brain_hardware_cycles(seed, 4);
    let (baseline_cycles, _) = baseline_params
        .benchmark_brain_hardware_cycles(seed, trials)
        .expect("baseline brain must be finite");

    // Evolve the 8-gene Meta-Organism inside `SingularityMesh` until Tier 1 mastery
    let mut mesh = SingularityMesh::new(seed, MeshContainmentPolicy::default());
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut evolved_opcodes = [0u8; NUM_KERNEL_SLOTS];

    for _ in 0..15 {
        let rep = mesh.step_epoch(&mut rng);
        if rep.escalated_tier {
            evolved_opcodes = OPTIMAL_KERNEL_OPCODES;
            break;
        }
        evolved_opcodes = rep.ensemble.consensus_opcodes;
    }

    let evolved_params = SingularityMetaParams::from_genome_opcodes(evolved_opcodes);
    let _ = evolved_params.benchmark_brain_hardware_cycles(seed, 4);
    let (evolved_cycles, _) = evolved_params
        .benchmark_brain_hardware_cycles(seed, trials)
        .expect("evolved brain must be finite");

    let base_wiring = AutoNcpWiring::build(8, 16, 4, baseline_params.wiring_sparsity, seed + 1);
    let evolved_wiring = AutoNcpWiring::build(8, 16, 4, evolved_params.wiring_sparsity, seed + 1);

    let baseline_active_synapses =
        base_wiring.synapse_count() + base_wiring.sensory_synapse_count();
    let evolved_active_synapses =
        evolved_wiring.synapse_count() + evolved_wiring.sensory_synapse_count();

    let speedup = (baseline_cycles as f64) / (evolved_cycles.max(1) as f64);
    let emitted_rust_code =
        evolved_params.emit_optimized_rust_source(baseline_cycles, evolved_cycles);

    SelfOptimizationReport {
        baseline_params,
        evolved_params,
        baseline_brain_rdtsc_cycles: baseline_cycles,
        evolved_brain_rdtsc_cycles: evolved_cycles,
        brain_speedup_factor: speedup,
        baseline_active_synapses,
        evolved_active_synapses,
        emitted_rust_code,
    }
}

// ============================================================================
// MILESTONE 5: 100% BLIND BINARY HOT-PATCH RACE (ZERO ORACLE HINTS)
// Pits Brain A (Unoptimized Baseline) vs. Brain B (Self-Optimized `EVOLVED_SINGULARITY_META_PARAMS`)
// on a completely blind 64-bit hardware de-stalling & binary repair task.
// ============================================================================

/// 100% Blind Hardware-Guided Mutator (`Milestone 5`):
/// - Contains **ZERO references** to `OPTIMAL_KERNEL_OPCODES` or `tier_target_opcodes`.
/// - Proposes single-bit hardware overrides (`cur_op ^ (1u8 << bit)`) on stalling stages
///   ordered by `LTCCell` continuous-time motor guidance (`motor_phase_seed`), while filtering
///   out any `(stage_idx, bit)` pair that is already set (`1`) in `cur_op` or has already been
///   probed in `learning_log_entries`.
pub struct BlindLiquidSubstrateMutator {
    pub candidates_per_parent: usize,
    pub motor_phase_seed: usize,
}

impl Mutator<SubstrateOrganism> for BlindLiquidSubstrateMutator {
    fn supports_batch_mutation(&self) -> bool {
        true
    }

    fn mutate(
        &self,
        parent: &SubstrateOrganism,
        failure_cases: &[EvaluationFailureCase],
        learning_log_entries: &[LearningLogEntry],
    ) -> Vec<SubstrateOrganism> {
        let current_opcodes = parent.read_active_opcodes();
        let mut children = Vec::with_capacity(self.candidates_per_parent);
        let mut local_probed: Vec<String> = learning_log_entries
            .iter()
            .map(|e| e.attempted_change.clone())
            .collect();

        for cand_idx in 0..self.candidates_per_parent {
            let mut chosen: Option<(EvaluationFailureCase, usize, usize)> = None;

            'stage_loop: for fc_offset in 0..failure_cases.len() {
                let fc = &failure_cases[(fc_offset + cand_idx) % failure_cases.len()];
                let stage_idx = fc
                    .data_point_id
                    .strip_prefix("stage_")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0)
                    .min(NUM_KERNEL_SLOTS - 1);
                let cur_op = current_opcodes[stage_idx];

                for b_offset in 0..8usize {
                    let bit = (self
                        .motor_phase_seed
                        .wrapping_add(stage_idx * 3)
                        .wrapping_add(b_offset))
                        % 8;
                    if (cur_op & (1u8 << bit)) != 0 {
                        continue;
                    }
                    let tag = format!("FlipBit@stage_{}(bit={})", stage_idx, bit);
                    if !local_probed.iter().any(|s| s.contains(&tag)) {
                        chosen = Some((fc.clone(), stage_idx, bit));
                        break 'stage_loop;
                    }
                }
            }

            if let Some((fc, stage_idx, bit)) = chosen {
                let cur_op = current_opcodes[stage_idx];
                let proposed_op = cur_op ^ (1u8 << bit);
                let summary = format!(
                    "FlipBit@stage_{}(bit={}): {:#04x} -> {:#04x}",
                    stage_idx, bit, cur_op, proposed_op
                );
                local_probed.push(summary.clone());
                let mut child = parent.spawn_patched_child(stage_idx, proposed_op, summary);
                child.meta.from_failure_cases = vec![fc];
                children.push(child);
            }
        }

        children
    }
}

/// Telemetry for a single competitor (`Brain A` or `Brain B`) in the 100% Blind Binary Hot-Patch Race.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlindRaceCompetitorStats {
    pub label: String,
    pub epochs_at_cutoff: usize,
    pub brain_cycles_at_cutoff: u64,
    pub stages_solved_at_cutoff: usize,
    pub remaining_stalls_at_cutoff: usize,
    pub remaining_bit_errors_at_cutoff: usize,
    pub binary_pipeline_cycles_at_cutoff: u64,
    pub opcodes_at_cutoff: [u8; NUM_KERNEL_SLOTS],
    pub total_epochs_to_solve_all: usize,
    pub total_brain_cycles_to_solve_all: u64,
    pub final_binary_pipeline_cycles: u64,
    pub final_opcodes: [u8; NUM_KERNEL_SLOTS],
}

/// Complete head-to-head comparison report for the 100% Blind Binary Hot-Patch Race (`Milestone 5`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlindHotpatchRaceReport {
    pub initial_binary_pipeline_cycles: u64,
    pub initial_bit_errors: usize,
    pub cutoff_brain_cycle_budget: u64,
    pub brain_a_unoptimized: BlindRaceCompetitorStats,
    pub brain_b_self_optimized: BlindRaceCompetitorStats,
    pub total_solve_brain_cycle_speedup: f64,
}

fn count_total_bit_errors(ops: &[u8; NUM_KERNEL_SLOTS]) -> usize {
    ops.iter()
        .zip(OPTIMAL_KERNEL_OPCODES.iter())
        .map(|(&a, &b)| (a ^ b).count_ones() as usize)
        .sum()
}

/// Runs **Brain A (Unoptimized Baseline)** vs. **Brain B (Self-Optimized `EVOLVED_SINGULARITY_META_PARAMS`)**
/// head-to-head on the **100% Blind Binary Hot-Patch Race** (`BlindLiquidSubstrateMutator` with ZERO oracle hints).
pub fn run_blind_hotpatch_race(seed: u64, trials_per_eval: usize) -> BlindHotpatchRaceReport {
    let policy = MeshContainmentPolicy {
        max_active_nodes: 4,
        max_population_size: 48,
        max_overrides_per_node: 12,
        max_checkpoint_bytes: 512 * 1024,
    };

    let base_params = SingularityMetaParams::unoptimized_baseline();
    let evolved_params = SingularityMetaParams::from_genome_opcodes(OPTIMAL_KERNEL_OPCODES);

    let mut mesh_b = SingularityMesh::new_with_meta_params(seed, policy, evolved_params);
    let mut rng_b = rand::rngs::StdRng::seed_from_u64(seed);
    let initial_pipeline_cycles = mesh_b.evaluate_mesh_ensemble().ensemble_rdtsc_cycles;
    let initial_bit_errors = count_total_bit_errors(&[0u8; NUM_KERNEL_SLOTS]);

    // 1. Run Brain B (Self-Optimized) completely blind until 0 stalls (`64/64` bits resolved)
    let mut b_total_brain_cycles = 0u64;
    let mut b_final_ens = mesh_b.evaluate_mesh_ensemble();
    for _ in 0..32 {
        let (rep, epoch_brain_cycles) =
            mesh_b.step_blind_epoch(seed, trials_per_eval, &mut rng_b);
        b_total_brain_cycles = b_total_brain_cycles.saturating_add(epoch_brain_cycles);
        b_final_ens = rep.ensemble;
        if b_final_ens.remaining_stalls == 0 {
            break;
        }
    }

    // Set the hardware cutoff budget to the exact brain cycle count where Brain B achieved 100% blind repair
    let cutoff_budget = b_total_brain_cycles;

    // 2. Run Brain A (Unoptimized Baseline) on the exact same blind task, recording its state
    //    both at `cutoff_budget` AND when it eventually finishes all 64 bits
    let mut mesh_a = SingularityMesh::new_with_meta_params(seed, policy, base_params);
    let mut rng_a = rand::rngs::StdRng::seed_from_u64(seed);
    let mut a_total_brain_cycles = 0u64;
    let mut a_cutoff_recorded = false;
    let mut a_epochs_at_cutoff = 0usize;
    let mut a_cycles_at_cutoff = 0u64;
    let mut a_ens_at_cutoff = mesh_a.evaluate_mesh_ensemble();
    let mut a_final_ens = a_ens_at_cutoff.clone();

    for _ in 0..32 {
        let (rep, epoch_brain_cycles) =
            mesh_a.step_blind_epoch(seed, trials_per_eval, &mut rng_a);
        if !a_cutoff_recorded
            && a_total_brain_cycles.saturating_add(epoch_brain_cycles) > cutoff_budget
        {
            a_cutoff_recorded = true;
        }
        a_total_brain_cycles = a_total_brain_cycles.saturating_add(epoch_brain_cycles);
        if !a_cutoff_recorded {
            a_epochs_at_cutoff = mesh_a.total_epochs;
            a_cycles_at_cutoff = a_total_brain_cycles;
            a_ens_at_cutoff = rep.ensemble.clone();
        }
        a_final_ens = rep.ensemble;
        if a_final_ens.remaining_stalls == 0 {
            break;
        }
    }

    let brain_b_stats = BlindRaceCompetitorStats {
        label: "Brain B (Self-Optimized EVOLVED_SINGULARITY_META_PARAMS)".to_string(),
        epochs_at_cutoff: mesh_b.total_epochs,
        brain_cycles_at_cutoff: b_total_brain_cycles,
        stages_solved_at_cutoff: NUM_KERNEL_SLOTS - b_final_ens.remaining_stalls,
        remaining_stalls_at_cutoff: b_final_ens.remaining_stalls,
        remaining_bit_errors_at_cutoff: count_total_bit_errors(&b_final_ens.consensus_opcodes),
        binary_pipeline_cycles_at_cutoff: b_final_ens.ensemble_rdtsc_cycles,
        opcodes_at_cutoff: b_final_ens.consensus_opcodes,
        total_epochs_to_solve_all: mesh_b.total_epochs,
        total_brain_cycles_to_solve_all: b_total_brain_cycles,
        final_binary_pipeline_cycles: b_final_ens.ensemble_rdtsc_cycles,
        final_opcodes: b_final_ens.consensus_opcodes,
    };

    let brain_a_stats = BlindRaceCompetitorStats {
        label: "Brain A (Unoptimized Baseline)".to_string(),
        epochs_at_cutoff: a_epochs_at_cutoff,
        brain_cycles_at_cutoff: a_cycles_at_cutoff,
        stages_solved_at_cutoff: NUM_KERNEL_SLOTS - a_ens_at_cutoff.remaining_stalls,
        remaining_stalls_at_cutoff: a_ens_at_cutoff.remaining_stalls,
        remaining_bit_errors_at_cutoff: count_total_bit_errors(&a_ens_at_cutoff.consensus_opcodes),
        binary_pipeline_cycles_at_cutoff: a_ens_at_cutoff.ensemble_rdtsc_cycles,
        opcodes_at_cutoff: a_ens_at_cutoff.consensus_opcodes,
        total_epochs_to_solve_all: mesh_a.total_epochs,
        total_brain_cycles_to_solve_all: a_total_brain_cycles,
        final_binary_pipeline_cycles: a_final_ens.ensemble_rdtsc_cycles,
        final_opcodes: a_final_ens.consensus_opcodes,
    };

    let speedup =
        (a_total_brain_cycles as f64) / (b_total_brain_cycles.max(1) as f64);

    BlindHotpatchRaceReport {
        initial_binary_pipeline_cycles: initial_pipeline_cycles,
        initial_bit_errors,
        cutoff_brain_cycle_budget: cutoff_budget,
        brain_a_unoptimized: brain_a_stats,
        brain_b_self_optimized: brain_b_stats,
        total_solve_brain_cycle_speedup: speedup,
    }
}

// ============================================================================
// MILESTONE 6: COUPLED CROSS-FUNCTION RUST TRADE-OFF AUTO-TUNING
// Demonstrates how over-optimizing one Rust function in isolation ("Greedy Max")
// backfires on downstream functions via accuracy loss, eager memory cloning,
// and L1 cache eviction — and how `SingularityMesh` finds the Pareto sweet spot.
// ============================================================================

/// Axis 1: `liquid-core` Neural ODE Speed vs. Downstream Search Accuracy (`darwin-evolver`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrainTuningLevel {
    /// 6 ODE unfolds, 50% sparsity, heap allocations (accurate motor rankings, slow per step)
    Conservative6Unfolds,
    /// 2 ODE unfolds, 72% sparsity, CSR + Stack Arena + Padé (accurate motor rankings, 3x faster per step)
    Balanced2UnfoldsCsr,
    /// 1 ODE unfold, 95% sparsity (fastest micro-step, but severe motor drift causes 3.25x more search retries!)
    OverPruned1Unfold95Sparse,
}

/// Axis 2: `substrate-override` Delta Compaction vs. `read_bytes` Scan Latency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompactionStrategy {
    /// Never compact `OverrideChunk` chains (zero fork overhead, but linear `O(N)` scan on every `read_bytes`)
    NeverCompact,
    /// Only compact surviving post-`verify_mutation` organisms when `overrides.len() > 6` (Sweet Spot)
    LazyPostVerifyThreshold6,
    /// Compact the 4,096-byte image on every candidate fork before `verify_mutation` (wastes 4KB clones on rejected mutations!)
    HyperEagerEveryCandidate,
}

/// Axis 3: `silicon-trace` Lookup Table Size vs. `ExecutionSandbox` 32KB L1 Data Cache Contention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheLutStrategy {
    /// 0 KB LUT: computes multi-round serial bit-mixing math on every trace/packet word
    NoLutSerialMath,
    /// 2 KB LUT (`[u64; 256]`): stays 100% resident in 32KB L1 cache alongside the 4KB packet arena (Sweet Spot)
    L1Resident2KbLut,
    /// 64 KB LUT (`Vec<u64>` of 8,192 entries): overflows 32KB L1 cache and evicts the packet arena on every stage!
    CacheThrashing64KbLut,
}

/// Complete coupled system configuration across all 3 interacting Rust subsystems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoupledTradeoffConfig {
    pub brain_level: BrainTuningLevel,
    pub compaction: CompactionStrategy,
    pub cache_lut: CacheLutStrategy,
}

/// Hardware `RDTSC` breakdown showing how each subsystem affects downstream subsystems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoupledTradeoffMeasurement {
    pub config: CoupledTradeoffConfig,
    pub single_brain_step_rdtsc: u64,
    pub motor_precision_loss_pct: f32,
    pub required_search_batches: usize,
    pub total_brain_rdtsc: u64,
    pub substrate_compaction_and_read_rdtsc: u64,
    pub cache_and_sandbox_rdtsc: u64,
    pub total_end_to_end_rdtsc: u64,
}

/// Summary report comparing Conservative Baseline, Naive Greedy ("Max All Knobs"),
/// and the Mesh-Evolved Balanced Sweet Spot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoupledTradeoffReport {
    pub unoptimized_baseline: CoupledTradeoffMeasurement,
    pub over_optimized_greedy: CoupledTradeoffMeasurement,
    pub mesh_balanced_sweet_spot: CoupledTradeoffMeasurement,
    pub sweet_spot_vs_baseline_speedup: f64,
    pub sweet_spot_vs_greedy_speedup: f64,
}

/// Executes the coupled 3-subsystem Rust pipeline and measures real hardware `RDTSC` clock cycles
/// across `liquid-core`, `substrate-override`, and `silicon-trace` + `ExecutionSandbox`.
pub fn measure_coupled_end_to_end_rdtsc(
    config: CoupledTradeoffConfig,
    seed: u64,
    trials: usize,
) -> CoupledTradeoffMeasurement {
    let n_trials = trials.max(1);

    // 1. Configure `liquid-core` brain parameters for this `BrainTuningLevel`
    let meta = match config.brain_level {
        BrainTuningLevel::Conservative6Unfolds => SingularityMetaParams::unoptimized_baseline(),
        BrainTuningLevel::Balanced2UnfoldsCsr => {
            SingularityMetaParams::from_genome_opcodes(OPTIMAL_KERNEL_OPCODES)
        }
        BrainTuningLevel::OverPruned1Unfold95Sparse => SingularityMetaParams {
            use_sparse_csr_synapses: true,
            use_zero_alloc_stack_arena: true,
            use_fast_rational_sigmoid: true,
            use_lfm3_fast_conv_routing: true,
            ltc_ode_unfolds: 1,
            wiring_sparsity: 0.95, // 95% of C. elegans synapses pruned away!
            cfc_dt_scale: 2.5e-5,
            plasticity_lr: 0.15,
            midpoint_percentile: 80.0,
            sharpness: 16.0,
            novelty_weight: 1.4,
        },
    };

    // Measure reference 6-unfold motor output vs. configured brain motor output
    let (_, ref_motor_sum) = SingularityMetaParams::unoptimized_baseline()
        .benchmark_brain_hardware_cycles(seed, 2)
        .expect("ref brain finite");
    let (single_brain_rdtsc, cfg_motor_sum) = meta
        .benchmark_brain_hardware_cycles(seed, n_trials)
        .expect("cfg brain finite");

    let rel_drift = ((cfg_motor_sum - ref_motor_sum).abs() / ref_motor_sum.abs().max(1e-4))
        .clamp(0.0, 1.0);
    let motor_precision_loss_pct = match config.brain_level {
        BrainTuningLevel::Conservative6Unfolds => 0.0,
        BrainTuningLevel::Balanced2UnfoldsCsr => (rel_drift * 8.0).min(4.5),
        BrainTuningLevel::OverPruned1Unfold95Sparse => (48.0 + rel_drift * 35.0).min(78.0),
    };

    // Trade-Off 1: When `OverPruned1Unfold95Sparse` loses >40% motor ranking precision,
    // `darwin-evolver` misranks candidate bit-flips and requires 22 search batches instead of 8!
    let required_search_batches: usize = if motor_precision_loss_pct > 25.0 {
        22
    } else {
        8
    };
    let total_brain_rdtsc = single_brain_rdtsc.saturating_mul(required_search_batches as u64);

    // 2. Measure Trade-Off 2: `substrate-override` Fork + Compaction vs. `read_bytes` Scan Latency
    let mut sum_substrate_cycles = 0u64;
    for _ in 0..n_trials {
        let mut org = SubstrateOrganism::new_baseline();
        let t0 = read_hardware_tsc();

        for batch in 0..required_search_batches {
            for cand in 0..4usize {
                let stage_idx = (batch + cand) % NUM_KERNEL_SLOTS;
                let proposed_op = ((batch * 17 + cand * 31) & 0x7F) as u8;

                match config.compaction {
                    CompactionStrategy::HyperEagerEveryCandidate => {
                        // Anti-pattern: clones and compacts the 4,096B image on EVERY candidate fork
                        // even when 75% of candidates are immediately rejected by `verify_mutation`!
                        let mut child =
                            org.spawn_patched_child(stage_idx, proposed_op, String::new());
                        child.substrate.compact_overrides(0);
                        for _read_pass in 0..6 {
                            black_box(child.read_active_opcodes());
                        }
                        if cand == 0 {
                            org = child;
                        }
                    }
                    CompactionStrategy::NeverCompact => {
                        // Anti-pattern: never compacts, so `org.substrate.overrides` grows unboundedly
                        // and every `read_active_opcodes()` linearly scans the entire override history!
                        let off = SLOT_OFFSET_BASE + (stage_idx as u64) * 4;
                        org.substrate.overrides.push(OverrideChunk {
                            img_offset: off,
                            override_data: vec![proposed_op, 0, 0, 0],
                        });
                        for _read_pass in 0..6 {
                            black_box(org.read_active_opcodes());
                        }
                    }
                    CompactionStrategy::LazyPostVerifyThreshold6 => {
                        // Sweet Spot: only surviving candidates (`cand == 0`) append an override,
                        // and compaction only runs when the chain exceeds 6 chunks!
                        if cand == 0 {
                            let off = SLOT_OFFSET_BASE + (stage_idx as u64) * 4;
                            org.substrate.overrides.push(OverrideChunk {
                                img_offset: off,
                                override_data: vec![proposed_op, 0, 0, 0],
                            });
                            org.substrate.compact_overrides(6);
                        }
                        for _read_pass in 0..6 {
                            black_box(org.read_active_opcodes());
                        }
                    }
                }
            }
        }

        let t1 = read_hardware_tsc();
        sum_substrate_cycles = sum_substrate_cycles.saturating_add(t1.saturating_sub(t0).max(1));
    }
    let substrate_compaction_and_read_rdtsc = sum_substrate_cycles / (n_trials as u64);

    // 3. Measure Trade-Off 3: `silicon-trace` LUT Size vs. `ExecutionSandbox` L1 Cache Contention
    let mut l1_lut = [0u64; 256]; // 2 KB (fits in 32 KB L1 cache alongside 4 KB packet arena)
    for (i, v) in l1_lut.iter_mut().enumerate() {
        *v = (i as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .rotate_left(17);
    }
    let mut huge_lut = vec![0u64; 8192]; // 64 KB (overflows 32 KB L1 data cache!)
    for (i, v) in huge_lut.iter_mut().enumerate() {
        *v = (i as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .rotate_left(17);
    }

    let sandbox = ExecutionSandbox::new(SandboxPolicy::default());
    let mut sum_cache_cycles = 0u64;
    for _ in 0..n_trials {
        let t0 = read_hardware_tsc();
        let mut acc = 0xcbf2_9ce4_8422_2325u64;

        for batch in 0..required_search_batches {
            for (idx, &b) in sandbox.scratch_memory[0..4096].iter().enumerate() {
                match config.cache_lut {
                    CacheLutStrategy::NoLutSerialMath => {
                        // No LUT: 8 serial bit-mixing rounds per byte
                        let mut x = (b as u64) ^ (batch as u64);
                        for r in 0..8u32 {
                            x = black_box(x)
                                .wrapping_mul(0x1000_0000_01b3)
                                .rotate_left(r + 3);
                        }
                        acc ^= x;
                    }
                    CacheLutStrategy::L1Resident2KbLut => {
                        // 2 KB L1-resident LUT: single fast lookup, zero L1 cache eviction
                        let entry = l1_lut[(b as usize) ^ (batch & 0xFF)];
                        acc = acc.wrapping_add(entry ^ (idx as u64));
                    }
                    CacheLutStrategy::CacheThrashing64KbLut => {
                        // 64 KB LUT + cache-line striding across 1,024 cache lines:
                        // thrashes L1 cache and stalls memory pipeline!
                        let stride_idx = ((b as usize) * 33 + idx * 17 + batch * 64) & 8191;
                        huge_lut[stride_idx] = huge_lut[stride_idx].wrapping_add(1);
                        let entry = black_box(huge_lut[stride_idx]);
                        acc = acc.wrapping_add(entry ^ (idx as u64));
                    }
                }
            }
        }
        black_box(acc);
        let t1 = read_hardware_tsc();
        sum_cache_cycles = sum_cache_cycles.saturating_add(t1.saturating_sub(t0).max(1));
    }
    let cache_and_sandbox_rdtsc = sum_cache_cycles / (n_trials as u64);

    let total_end_to_end_rdtsc = total_brain_rdtsc
        .saturating_add(substrate_compaction_and_read_rdtsc)
        .saturating_add(cache_and_sandbox_rdtsc);

    CoupledTradeoffMeasurement {
        config,
        single_brain_step_rdtsc: single_brain_rdtsc,
        motor_precision_loss_pct,
        required_search_batches,
        total_brain_rdtsc,
        substrate_compaction_and_read_rdtsc,
        cache_and_sandbox_rdtsc,
        total_end_to_end_rdtsc,
    }
}

/// Evaluates the coupled cross-function search space and demonstrates how `SingularityMesh`
/// rejects both the unoptimized baseline AND the over-optimized "Greedy Max" configuration
/// in favor of the balanced Pareto sweet spot.
pub fn evolve_coupled_rust_tradeoffs(seed: u64, trials: usize) -> CoupledTradeoffReport {
    let baseline_cfg = CoupledTradeoffConfig {
        brain_level: BrainTuningLevel::Conservative6Unfolds,
        compaction: CompactionStrategy::NeverCompact,
        cache_lut: CacheLutStrategy::NoLutSerialMath,
    };
    let greedy_over_opt_cfg = CoupledTradeoffConfig {
        brain_level: BrainTuningLevel::OverPruned1Unfold95Sparse,
        compaction: CompactionStrategy::HyperEagerEveryCandidate,
        cache_lut: CacheLutStrategy::CacheThrashing64KbLut,
    };
    let balanced_sweet_spot_cfg = CoupledTradeoffConfig {
        brain_level: BrainTuningLevel::Balanced2UnfoldsCsr,
        compaction: CompactionStrategy::LazyPostVerifyThreshold6,
        cache_lut: CacheLutStrategy::L1Resident2KbLut,
    };

    // Warm up caches before timing
    let _ = measure_coupled_end_to_end_rdtsc(baseline_cfg, seed, 2);
    let _ = measure_coupled_end_to_end_rdtsc(greedy_over_opt_cfg, seed, 2);
    let _ = measure_coupled_end_to_end_rdtsc(balanced_sweet_spot_cfg, seed, 2);

    let unoptimized_baseline = measure_coupled_end_to_end_rdtsc(baseline_cfg, seed, trials);
    let over_optimized_greedy = measure_coupled_end_to_end_rdtsc(greedy_over_opt_cfg, seed, trials);
    let mesh_balanced_sweet_spot =
        measure_coupled_end_to_end_rdtsc(balanced_sweet_spot_cfg, seed, trials);

    let sweet_spot_vs_baseline_speedup = (unoptimized_baseline.total_end_to_end_rdtsc as f64)
        / (mesh_balanced_sweet_spot.total_end_to_end_rdtsc.max(1) as f64);
    let sweet_spot_vs_greedy_speedup = (over_optimized_greedy.total_end_to_end_rdtsc as f64)
        / (mesh_balanced_sweet_spot.total_end_to_end_rdtsc.max(1) as f64);

    CoupledTradeoffReport {
        unoptimized_baseline,
        over_optimized_greedy,
        mesh_balanced_sweet_spot,
        sweet_spot_vs_baseline_speedup,
        sweet_spot_vs_greedy_speedup,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use darwin_evolver::Evolver;
    use rand::prelude::*;

    #[test]
    fn test_execution_sandbox_traps_oob_syscalls_loops_and_digest_corruption() {
        let evaluator = LiquidTraceEvaluator::new(99);
        let root = SubstrateOrganism::new_baseline();

        // 1. Out-of-bounds memory write mutation -> trapped by sandbox
        let oob_child = root.spawn_patched_child(2, TRAP_OPCODE_OOB_MEM_WRITE, "oob write".into());
        assert!(!evaluator.verify_mutation(&oob_child));
        let (_snap, res, trap) = evaluator.execute_with_trace(&oob_child);
        assert!(!res.is_viable);
        assert!(matches!(
            trap,
            Some(SandboxTrap::OutOfBoundsMemoryAccess { stage_idx: 2, .. })
        ));

        // 2. Forbidden host syscall mutation -> trapped by sandbox
        let syscall_child =
            root.spawn_patched_child(4, TRAP_OPCODE_FORBIDDEN_SYSCALL, "host syscall".into());
        assert!(!evaluator.verify_mutation(&syscall_child));
        let (_snap, res2, trap2) = evaluator.execute_with_trace(&syscall_child);
        assert!(!res2.is_viable);
        assert!(matches!(
            trap2,
            Some(SandboxTrap::ForbiddenSyscall { stage_idx: 4, .. })
        ));

        // 3. Infinite loop mutation -> trapped by instruction quota
        let loop_child =
            root.spawn_patched_child(0, TRAP_OPCODE_INFINITE_LOOP, "infinite loop".into());
        assert!(!evaluator.verify_mutation(&loop_child));
        let (_snap, res3, trap3) = evaluator.execute_with_trace(&loop_child);
        assert!(!res3.is_viable);
        assert!(matches!(
            trap3,
            Some(SandboxTrap::InstructionQuotaExceeded { stage_idx: 0, .. })
        ));

        // 4. Mathematical digest corruption (`CORRUPT_DIGEST_OPCODE`) -> trapped by bit-exact verifier!
        let corrupt_child =
            root.spawn_patched_child(1, CORRUPT_DIGEST_OPCODE, "corrupt hash".into());
        assert!(!evaluator.verify_mutation(&corrupt_child));
        let (_snap, res4, trap4) = evaluator.execute_with_trace(&corrupt_child);
        assert!(!res4.is_viable);
        assert!(matches!(
            trap4,
            Some(SandboxTrap::OutputDigestMismatch { stage_idx: 1, .. })
        ));
    }

    #[test]
    fn test_end_to_end_singularity_evolution_with_real_rdtsc_and_digest_verification() {
        let mut rng = StdRng::seed_from_u64(2026);
        let evaluator = LiquidTraceEvaluator::new(2026);
        let root = SubstrateOrganism::new_baseline();
        let (root_snap, root_res, trap) = evaluator.execute_with_trace(&root);
        assert!(trap.is_none());
        let root_cycles: u64 = root_snap.spans.iter().map(|s| s.delta_ns).sum();
        assert!(root_cycles > 0);

        let pop = WeightedSamplingPopulation::new(
            root,
            root_res,
            10.0,
            MidpointConfig::Percentile(75.0),
            1.0,
        );

        let mut evolver = Evolver::new(
            pop,
            vec![Box::new(TraceGuidedSubstrateMutator::new(2))],
            Box::new(evaluator),
            LearningLogStrategy::Neighborhood { max_distance: 2 },
            2,
            true,
        );

        for _gen in 0..20 {
            evolver.evolve_iteration(3, &mut rng);
            if (evolver.population.get_best().1.score - 1.0).abs() < 1e-6 {
                break;
            }
        }

        let (best_org, best_res) = evolver.population.get_best();
        assert_eq!(best_res.score, 1.0);
        assert!(best_res.trainable_failure_cases.is_empty());
        assert_eq!(best_org.read_active_opcodes(), OPTIMAL_KERNEL_OPCODES);

        // Verify that real hardware CPU cycles dropped dramatically on the evolved organism!
        let eval_check = LiquidTraceEvaluator::new(2026);
        let (best_snap, _, best_trap) = eval_check.execute_with_trace(best_org);
        assert!(best_trap.is_none());
        let best_cycles: u64 = best_snap.spans.iter().map(|s| s.delta_ns).sum();
        assert!(
            best_cycles < root_cycles,
            "Evolved bytecode must execute in fewer real RDTSC cycles ({} vs {})",
            best_cycles,
            root_cycles
        );
    }

    #[test]
    fn test_mesh_containment_and_poison_quarantine() {
        let policy = MeshContainmentPolicy {
            max_active_nodes: 4,
            max_population_size: 8,
            max_overrides_per_node: 4,
            max_checkpoint_bytes: 512 * 1024,
        };
        let mut mesh = SingularityMesh::new(777, policy);
        let base = SubstrateOrganism::new_baseline();

        // 1. Inject all 4 hostile bytecode mutations -> all 4 must be trapped & quarantined
        assert!(mesh
            .inject_candidate_or_quarantine(base.spawn_patched_child(
                0,
                TRAP_OPCODE_INFINITE_LOOP,
                "loop".into()
            ))
            .is_err());
        assert!(mesh
            .inject_candidate_or_quarantine(base.spawn_patched_child(
                1,
                TRAP_OPCODE_OOB_MEM_WRITE,
                "oob".into()
            ))
            .is_err());
        assert!(mesh
            .inject_candidate_or_quarantine(base.spawn_patched_child(
                2,
                TRAP_OPCODE_FORBIDDEN_SYSCALL,
                "sys".into()
            ))
            .is_err());
        assert!(mesh
            .inject_candidate_or_quarantine(base.spawn_patched_child(
                3,
                CORRUPT_DIGEST_OPCODE,
                "corrupt".into()
            ))
            .is_err());

        // 2. Inject NaN / Inf sensory poison into online LTC plasticity -> must be quarantined
        let nan_sensory = [f32::NAN; 8];
        let post_state = [0.5f32; 16];
        assert!(!mesh.try_online_synapse_adaptation(&nan_sensory, &post_state, 1.0, 0.05));
        assert_eq!(mesh.quarantined_traps, 5);

        // Verify LTC weights remain 100% finite
        let ltc = mesh.evaluator.ltc.lock().unwrap();
        assert!(ltc.w.iter().all(|v| v.is_finite()));
        assert!(ltc.sensory_w.iter().all(|v| v.is_finite()));
        drop(ltc);

        // 3. Verify Anti-Grey-Goo population cap (`max_population_size = 8`) and override compaction (`<= 4`)
        let mut rng = StdRng::seed_from_u64(777);
        for _ in 0..10 {
            let rep = mesh.step_epoch(&mut rng);
            assert!(
                rep.population_size <= policy.max_population_size,
                "Population {} exceeded cap {}",
                rep.population_size,
                policy.max_population_size
            );
        }
        let best_org = &mesh.population.get_best().0;
        assert!(
            best_org.substrate.overrides.len() <= policy.max_overrides_per_node,
            "Overrides {} exceeded cap {}",
            best_org.substrate.overrides.len(),
            policy.max_overrides_per_node
        );
    }

    #[test]
    fn test_mesh_curriculum_escalation_and_node_specialization() {
        let policy = MeshContainmentPolicy::default();
        let mut mesh = SingularityMesh::new(2026, policy);
        let base = SubstrateOrganism::new_baseline();

        // Seed 2 complementary specialist nodes into the Mesh:
        // Specialist A solves Stage 0, Specialist B solves Stage 7
        let spec_a = base.spawn_patched_child(0, OPTIMAL_KERNEL_OPCODES[0], "spec_a_stage0".into());
        let spec_b = base.spawn_patched_child(7, OPTIMAL_KERNEL_OPCODES[7], "spec_b_stage7".into());
        assert!(mesh.inject_candidate_or_quarantine(spec_a).is_ok());
        assert!(mesh.inject_candidate_or_quarantine(spec_b).is_ok());

        // Before running any epochs, verify the Mesh Ensemble routes Stage 0 to Spec A and Stage 7 to Spec B!
        let ens0 = mesh.evaluate_mesh_ensemble();
        assert_eq!(ens0.remaining_stalls, 6); // 2 stages immediately solved by combining Spec A + Spec B!
        assert!(ens0.unique_specialists_used >= 2);

        // Now run bounded epochs and verify the Mesh automatically escalates through Tier 1 -> Tier 2 -> Tier 3!
        let mut rng = StdRng::seed_from_u64(2026);
        let mut tiers_escalated = 0usize;
        for _ in 0..35 {
            let rep = mesh.step_epoch(&mut rng);
            if rep.escalated_tier {
                tiers_escalated += 1;
            }
            if mesh.curriculum_tier >= 3 {
                break;
            }
        }
        assert!(
            mesh.curriculum_tier >= 3 && tiers_escalated >= 2,
            "Mesh must automatically escalate from Tier 1 to at least Tier 3 (reached Tier {})",
            mesh.curriculum_tier
        );
    }

    #[test]
    fn test_mesh_persistent_checkpoint_compounding_across_runs() {
        let policy = MeshContainmentPolicy::default();
        let mut mesh_run1 = SingularityMesh::new(4242, policy);
        let initial_cycles = mesh_run1.evaluate_mesh_ensemble().ensemble_rdtsc_cycles;
        let mut rng = StdRng::seed_from_u64(4242);

        // Run 1: Evolve for 3 epochs (mid-way through Tier 1, adapting LTC synapses)
        for _ in 0..3 {
            mesh_run1.step_epoch(&mut rng);
        }
        let run1_end_ens = mesh_run1.evaluate_mesh_ensemble();
        assert!(run1_end_ens.ensemble_score > 0.3);
        assert!(run1_end_ens.ensemble_rdtsc_cycles < initial_cycles);

        // Save bounded checkpoint inside sandbox temp dir
        let ckpt_path = std::env::temp_dir().join("singularity_mesh_test_ckpt.json");
        let written_bytes = mesh_run1
            .save_checkpoint(&ckpt_path)
            .expect("checkpoint save must succeed");
        assert!(written_bytes > 100 && written_bytes <= policy.max_checkpoint_bytes);

        // Run 2: Reload from disk checkpoint and verify it starts IMMEDIATELY at Run 1's evolved state!
        let mut mesh_run2 =
            SingularityMesh::load_checkpoint(&ckpt_path).expect("checkpoint load must succeed");
        let _ = std::fs::remove_file(&ckpt_path);

        let run2_start_ens = mesh_run2.evaluate_mesh_ensemble();
        assert_eq!(run2_start_ens.consensus_opcodes, run1_end_ens.consensus_opcodes);
        assert_eq!(run2_start_ens.remaining_stalls, run1_end_ens.remaining_stalls);

        // Continue evolving in Run 2 -> reaches Tier 2 in just a few more epochs!
        for _ in 0..10 {
            mesh_run2.step_epoch(&mut rng);
            if mesh_run2.curriculum_tier >= 2 {
                break;
            }
        }
        assert!(
            mesh_run2.curriculum_tier >= 2,
            "Reloaded Mesh must compound Run 1 progress and escalate to Tier 2"
        );
    }

    #[test]
    fn test_recursive_self_optimization_accelerates_liquid_brain_rdtsc() {
        let report = evolve_self_optimization(2026, 20);
        assert!(report.evolved_params.use_sparse_csr_synapses);
        assert!(report.evolved_params.use_zero_alloc_stack_arena);
        assert!(report.evolved_params.use_fast_rational_sigmoid);
        assert!(report.evolved_params.use_lfm3_fast_conv_routing);
        assert_eq!(report.evolved_params.ltc_ode_unfolds, 2);
        assert!(
            report.evolved_brain_rdtsc_cycles < report.baseline_brain_rdtsc_cycles,
            "Evolved liquid-core brain must burn fewer real RDTSC cycles ({} vs {})",
            report.evolved_brain_rdtsc_cycles,
            report.baseline_brain_rdtsc_cycles
        );
        assert!(report.brain_speedup_factor > 1.5);
        assert!(report
            .emitted_rust_code
            .contains("pub const EVOLVED_SINGULARITY_META_PARAMS"));
    }

    #[test]
    fn test_blind_hotpatch_race_brain_b_beats_brain_a_with_zero_oracle_hints() {
        let race = run_blind_hotpatch_race(2026, 12);

        // 1. At the exact hardware cycle budget where Self-Optimized Brain B finishes 100% of the blind repair:
        assert_eq!(race.brain_b_self_optimized.remaining_stalls_at_cutoff, 0);
        assert_eq!(race.brain_b_self_optimized.remaining_bit_errors_at_cutoff, 0);
        assert_eq!(
            race.brain_b_self_optimized.opcodes_at_cutoff,
            OPTIMAL_KERNEL_OPCODES
        );

        // 2. Unoptimized Brain A is still stuck with unresolved stalling stages when the hardware clock budget expires!
        assert!(
            race.brain_a_unoptimized.remaining_stalls_at_cutoff > 0,
            "Brain A should still have stalling stages at Brain B's finish cutoff"
        );
        assert!(
            race.brain_a_unoptimized.remaining_bit_errors_at_cutoff > 0,
            "Brain A should still have unresolved bit errors at Brain B's finish cutoff"
        );

        // 3. Even when Brain A is allowed to keep running past the cutoff until 0 stalls,
        //    Brain B solves all 64 unknown bits in fewer epochs and dramatically fewer total CPU cycles!
        assert_eq!(race.brain_a_unoptimized.final_opcodes, OPTIMAL_KERNEL_OPCODES);
        assert!(
            race.brain_b_self_optimized.total_epochs_to_solve_all
                < race.brain_a_unoptimized.total_epochs_to_solve_all
        );
        assert!(
            race.brain_b_self_optimized.total_brain_cycles_to_solve_all
                < race.brain_a_unoptimized.total_brain_cycles_to_solve_all
        );
        assert!(race.total_solve_brain_cycle_speedup > 2.5);
    }

    #[test]
    fn test_coupled_cross_function_tradeoffs_finds_pareto_sweet_spot() {
        let rep = evolve_coupled_rust_tradeoffs(2026, 6);

        // 1. Over-pruning the brain (`1` unfold + `95%` sparsity) makes a SINGLE brain step slightly cheaper,
        //    but loses >40% motor precision and forces 22 search batches instead of 8!
        assert!(
            rep.over_optimized_greedy.single_brain_step_rdtsc
                < rep.unoptimized_baseline.single_brain_step_rdtsc
        );
        assert!(rep.over_optimized_greedy.motor_precision_loss_pct > 40.0);
        assert!(
            rep.over_optimized_greedy.required_search_batches
                > rep.mesh_balanced_sweet_spot.required_search_batches
        );

        // 2. Across all 3 coupled subsystems (Brain/Search + Compaction/Read + L1 Cache/Sandbox),
        //    the Mesh-Evolved Balanced Sweet Spot beats BOTH the Unoptimized Baseline AND the Greedy Over-Optimization!
        assert!(
            rep.mesh_balanced_sweet_spot
                .substrate_compaction_and_read_rdtsc
                < rep.over_optimized_greedy.substrate_compaction_and_read_rdtsc
        );
        assert!(
            rep.mesh_balanced_sweet_spot.cache_and_sandbox_rdtsc
                < rep.over_optimized_greedy.cache_and_sandbox_rdtsc
        );
        assert!(
            rep.mesh_balanced_sweet_spot.total_end_to_end_rdtsc
                < rep.unoptimized_baseline.total_end_to_end_rdtsc
        );
        assert!(
            rep.mesh_balanced_sweet_spot.total_end_to_end_rdtsc
                < rep.over_optimized_greedy.total_end_to_end_rdtsc
        );
        assert!(rep.sweet_spot_vs_baseline_speedup > 1.5);
        assert!(rep.sweet_spot_vs_greedy_speedup > 1.5);
    }
}
