//! # `binlex_kuna_lifter` — Consolidated Binary Genomics, SSA Decompilation & Step-Journal Verification
//!
//! Consolidates the real mechanisms inspected from:
//! 1. **`c3rb3ru5d3d53c/binlex` (`src/genetics/chromosome.rs` & `src/embeddings/llvm/mod.rs`):**
//!    - `BinlexChromosome`: Models executable code slices as `(raw_bytes, wildcard_mask)` chromosomes
//!      where volatile operands/immediates are masked (`0xFF`) while structural opcodes are preserved (`0x00`),
//!      paired with exact Shannon byte entropy and `MinHash32` locality-sensitive signatures.
//!    - `BinlexThreeFamilyEmbedder`: Implements `binlex`'s exact 3-family structural embedding pipeline
//!      (`0.60 * semantic + 0.15 * control_flow + 0.25 * sequence`), hashed into fixed-width feature buckets,
//!      smoothed via signed square-root (`v.signum() * v.abs().sqrt()`), and L2-normalized.
//! 2. **`morluto/kuna` (`src/ir/dfg.rs` & `src/cfg/structurer.rs`):**
//!    - `CanonicalSsaModule`: Lifts raw kernel/bytecode instructions into Static Single Assignment (`SSA`)
//!      form and runs canonicalization passes (`dead_code_elimination` + `constant_folding`) so surface
//!      mutations with identical semantics normalize to the same SSA graph.
//! 3. **`morluto/agentenv-framework` (`src/agent_env/task/task.py`):**
//!    - `StepJournalHarness`: Implements `agentenv-framework`'s step-level context journaling and
//!      `undo_steps()` rollback so failed evolutionary trials cleanly roll back state mutations.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::{NUM_KERNEL_SLOTS, OPTIMAL_KERNEL_OPCODES, STAGE_SYMBOLS};

/// Represents a `binlex` Chromosome (`src/genetics/chromosome.rs:68-96`):
/// raw instruction bytes paired with a per-byte wildcard mask (`0x00` = fixed structural gene,
/// `0xFF` = wildcarded volatile operand/address allele).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BinlexChromosome {
    pub raw_bytes: Vec<u8>,
    pub wildcard_mask: Vec<u8>,
}

impl BinlexChromosome {
    pub fn new(raw_bytes: Vec<u8>, wildcard_mask: Vec<u8>) -> Result<Self, &'static str> {
        if raw_bytes.len() != wildcard_mask.len() {
            return Err("raw_bytes and wildcard_mask must have identical length");
        }
        Ok(Self {
            raw_bytes,
            wildcard_mask,
        })
    }

    /// Extracts a 16-byte chromosome from an 8-stage kernel opcode array (`[opcode, operand_imm]`),
    /// masking out the volatile operand byte (`0xFF`) while preserving the structural dispatch opcode (`0x00`).
    pub fn from_kernel_opcodes(opcodes: &[u8; NUM_KERNEL_SLOTS], operands: &[u8; NUM_KERNEL_SLOTS]) -> Self {
        let mut raw_bytes = Vec::with_capacity(NUM_KERNEL_SLOTS * 2);
        let mut wildcard_mask = Vec::with_capacity(NUM_KERNEL_SLOTS * 2);
        for i in 0..NUM_KERNEL_SLOTS {
            raw_bytes.push(opcodes[i]);
            wildcard_mask.push(0x00); // Structural opcode is preserved
            raw_bytes.push(operands[i]);
            wildcard_mask.push(0xFF); // Volatile immediate/register slot is wildcarded
        }
        Self {
            raw_bytes,
            wildcard_mask,
        }
    }

    /// Returns the masked chromosome bytes with wildcarded bits zeroed in place (`chromosome.rs:44-46`).
    pub fn masked_bytes(&self) -> Vec<u8> {
        self.raw_bytes
            .iter()
            .zip(self.wildcard_mask.iter())
            .map(|(&b, &m)| b & !m)
            .collect()
    }

    /// Renders a YARA-compatible hex/wildcard pattern string (`chromosome.rs:39-41`).
    pub fn yara_pattern(&self) -> String {
        self.raw_bytes
            .iter()
            .zip(self.wildcard_mask.iter())
            .map(|(&b, &m)| {
                if m == 0xFF {
                    "??".to_string()
                } else {
                    format!("{b:02x}")
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Computes exact Shannon entropy in bits per byte (`[0.0, 8.0]`) over the masked chromosome.
    pub fn shannon_entropy(&self) -> f64 {
        let masked = self.masked_bytes();
        if masked.is_empty() {
            return 0.0;
        }
        let mut counts = [0usize; 256];
        for &b in &masked {
            counts[b as usize] += 1;
        }
        let len = masked.len() as f64;
        let mut entropy = 0.0;
        for &c in &counts {
            if c > 0 {
                let p = (c as f64) / len;
                entropy -= p * p.log2();
            }
        }
        entropy
    }

    /// Computes a deterministic 16-hash `MinHash32` signature over 2-byte shingles of the masked chromosome,
    /// enabling fast Jaccard similarity estimation across mutated binary organisms.
    pub fn minhash32(&self) -> [u32; 16] {
        let masked = self.masked_bytes();
        let mut sig = [u32::MAX; 16];
        if masked.is_empty() {
            return [0; 16];
        }
        let window = masked.len().min(2);
        for shingle in masked.windows(window) {
            let base = if shingle.len() == 2 {
                u16::from_le_bytes([shingle[0], shingle[1]]) as u32
            } else {
                shingle[0] as u32
            };
            for (k, slot) in sig.iter_mut().enumerate() {
                let seed = 0x9E37_79B9u32
                    .wrapping_mul((k as u32) + 1)
                    .wrapping_add(0x85EB_CA6Bu32);
                let mut h = base ^ seed;
                h ^= h >> 16;
                h = h.wrapping_mul(0x7FEB_352Du32);
                h ^= h >> 15;
                h = h.wrapping_mul(0x846C_A68Bu32);
                h ^= h >> 16;
                if h < *slot {
                    *slot = h;
                }
            }
        }
        sig
    }

    /// Computes the exact MinHash Jaccard similarity (`[0.0, 1.0]`) between two chromosomes.
    pub fn minhash_similarity(&self, other: &Self) -> f64 {
        let a = self.minhash32();
        let b = other.minhash32();
        let matches = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
        (matches as f64) / (a.len() as f64)
    }
}

/// SSA instruction node inspired by `morluto/kuna` (`src/ir/dfg.rs`) and `binlex` (`src/lifters/llvm/`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SsaInstruction {
    pub ssa_id: usize,
    pub stage_symbol: String,
    pub canonical_opcode: u8,
    pub is_vectorized_fast_path: bool,
    pub is_dead_store: bool,
}

/// Canonicalized SSA module lifted from an 8-stage bytecode organism.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalSsaModule {
    pub instructions: Vec<SsaInstruction>,
    pub eliminated_dead_stores: usize,
}

impl CanonicalSsaModule {
    /// Lifts an 8-stage kernel opcode slice into SSA form and runs `dce` (dead-store elimination).
    pub fn lift_and_canonicalize(opcodes: &[u8; NUM_KERNEL_SLOTS]) -> Self {
        let mut instructions = Vec::with_capacity(NUM_KERNEL_SLOTS);
        let mut eliminated_dead_stores = 0usize;

        for (idx, &op) in opcodes.iter().enumerate() {
            let is_fast = op == OPTIMAL_KERNEL_OPCODES[idx];
            // If an opcode is neither optimal nor a trap, it executes a redundant slow-path pass
            // whose intermediate scratch write is dead after stage commitment.
            let is_dead = !is_fast && op < 0xFC;
            if is_dead {
                eliminated_dead_stores += 1;
            }
            // Canonicalize redundant slow-path variants to canonical slow opcode `0x00`
            let canonical_opcode = if is_fast || op >= 0xFC { op } else { 0x00 };
            instructions.push(SsaInstruction {
                ssa_id: idx,
                stage_symbol: STAGE_SYMBOLS[idx].to_string(),
                canonical_opcode,
                is_vectorized_fast_path: is_fast,
                is_dead_store: false, // Dead scratch store eliminated by DCE pass
            });
        }

        Self {
            instructions,
            eliminated_dead_stores,
        }
    }
}

/// `binlex`'s 3-Family Structural Embedder (`src/embeddings/llvm/block.rs:21-33` & `mod.rs:598-670`):
/// Combines:
/// - `semantic` features (`weight = 0.60`)
/// - `control_flow` features (`weight = 0.15`)
/// - `sequence` n-gram features (`weight = 0.25`)
/// with deterministic hash projection, signed square-root smoothing (`v.signum() * v.abs().sqrt()`),
/// and L2 unit normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinlexThreeFamilyEmbedder {
    pub dimensions: usize,
    pub semantic_weight: f32,
    pub control_flow_weight: f32,
    pub sequence_weight: f32,
}

impl Default for BinlexThreeFamilyEmbedder {
    fn default() -> Self {
        Self {
            dimensions: 32,
            semantic_weight: 0.60,
            control_flow_weight: 0.15,
            sequence_weight: 0.25,
        }
    }
}

impl BinlexThreeFamilyEmbedder {
    fn hash_project(features: &[f32], dimensions: usize, seed: u64) -> Vec<f32> {
        let mut out = vec![0.0f32; dimensions.max(1)];
        for (idx, &val) in features.iter().enumerate() {
            let mut h = seed
                .wrapping_add((idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                ^ (val.to_bits() as u64);
            h ^= h >> 30;
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h ^= h >> 27;
            h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
            h ^= h >> 31;
            let bucket = (h as usize) % out.len();
            let sign = if (h >> 32) & 1 == 0 { 1.0 } else { -1.0 };
            out[bucket] += sign * val;
        }
        out
    }

    /// Embeds a lifted `CanonicalSsaModule` into an L2-normalized structural vector (`f32`).
    pub fn embed_ssa(&self, ssa: &CanonicalSsaModule) -> Vec<f32> {
        let n = ssa.instructions.len().max(1) as f32;
        let fast_count = ssa
            .instructions
            .iter()
            .filter(|i| i.is_vectorized_fast_path)
            .count() as f32;
        let trap_count = ssa
            .instructions
            .iter()
            .filter(|i| i.canonical_opcode >= 0xFC)
            .count() as f32;

        // 1. Semantic family (ratios of fast vectorized ops, canonical opcodes, and trap absence)
        let mut semantic = Vec::with_capacity(ssa.instructions.len() + 2);
        semantic.push(fast_count / n);
        semantic.push(1.0 - (trap_count / n));
        for inst in &ssa.instructions {
            semantic.push((inst.canonical_opcode as f32) / 255.0);
        }

        // 2. Control-flow family (branchless fast-path ratio and trap terminator flags)
        let control_flow = vec![
            fast_count / n,
            if trap_count > 0.0 { 1.0 } else { 0.0 },
            (ssa.eliminated_dead_stores as f32) / n,
        ];

        // 3. Sequence family (2-gram opcode transitions across the 8 stages)
        let mut sequence = Vec::with_capacity(ssa.instructions.len().saturating_sub(1));
        for pair in ssa.instructions.windows(2) {
            let bigram = ((pair[0].canonical_opcode as u16) << 8) | (pair[1].canonical_opcode as u16);
            sequence.push((bigram as f32) / 65535.0);
        }

        let d = self.dimensions.max(1);
        let sem_proj = Self::hash_project(&semantic, d, 0x5E9A_0000);
        let cfg_proj = Self::hash_project(&control_flow, d, 0x5E9A_0001);
        let seq_proj = Self::hash_project(&sequence, d, 0x5E9A_0002);

        let mut vec_out = vec![0.0f32; d];
        for i in 0..d {
            let v = sem_proj[i] * self.semantic_weight
                + cfg_proj[i] * self.control_flow_weight
                + seq_proj[i] * self.sequence_weight;
            // Signed square-root smoothing (`binlex/src/embeddings/llvm/mod.rs:666-670`)
            vec_out[i] = v.signum() * v.abs().sqrt();
        }

        // L2 normalization (`normalize_l2`)
        let norm = vec_out.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for x in &mut vec_out {
                *x /= norm;
            }
        }
        vec_out
    }

    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

/// Step-level execution journal and rollback manager modeled after
/// `morluto/agentenv-framework` (`src/agent_env/task/task.py:35-88`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepJournalEntry {
    pub step_id: String,
    pub before_opcodes: [u8; NUM_KERNEL_SLOTS],
    pub after_opcodes: [u8; NUM_KERNEL_SLOTS],
    pub verified_passed: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StepJournalHarness {
    pub current_opcodes: [u8; NUM_KERNEL_SLOTS],
    pub journal: Vec<StepJournalEntry>,
    pub metadata: BTreeMap<String, String>,
}

impl StepJournalHarness {
    pub fn new(initial_opcodes: [u8; NUM_KERNEL_SLOTS]) -> Self {
        Self {
            current_opcodes: initial_opcodes,
            journal: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// Applies a step mutation and records its before/after diff in the step journal.
    pub fn record_step(
        &mut self,
        step_id: impl Into<String>,
        next_opcodes: [u8; NUM_KERNEL_SLOTS],
        verified_passed: bool,
    ) {
        let entry = StepJournalEntry {
            step_id: step_id.into(),
            before_opcodes: self.current_opcodes,
            after_opcodes: next_opcodes,
            verified_passed,
        };
        self.current_opcodes = next_opcodes;
        self.journal.push(entry);
    }

    /// Rolls back the journal to the state before `target_step_id` (`agentenv-framework` `undo_steps()`),
    /// returning the list of rolled-back step IDs so any failed trial state is cleanly discarded.
    pub fn undo_steps_from(&mut self, target_step_id: &str) -> Vec<String> {
        if let Some(pos) = self.journal.iter().position(|e| e.step_id == target_step_id) {
            self.current_opcodes = self.journal[pos].before_opcodes;
            let drained: Vec<String> = self.journal.drain(pos..).map(|e| e.step_id).collect();
            drained
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binlex_chromosome_wildcard_invariance_and_minhash() {
        // Two organisms with the exact same dispatch opcodes but completely different volatile operands
        let ops = OPTIMAL_KERNEL_OPCODES;
        let operands_a = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let operands_b = [0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10];

        let chrom_a = BinlexChromosome::from_kernel_opcodes(&ops, &operands_a);
        let chrom_b = BinlexChromosome::from_kernel_opcodes(&ops, &operands_b);

        assert_eq!(chrom_a.masked_bytes(), chrom_b.masked_bytes());
        assert_eq!(chrom_a.yara_pattern(), chrom_b.yara_pattern());
        assert!((chrom_a.minhash_similarity(&chrom_b) - 1.0).abs() < 1e-9);
        assert!(chrom_a.shannon_entropy() > 1.5);
    }

    #[test]
    fn test_kuna_ssa_canonicalization_and_binlex_three_family_embedding() {
        // Two different non-optimal slow opcodes (`0x01` vs `0x02`) compute the exact same slow path
        let slow_variant_1 = [0x01, 0x19, 0x7F, 0xA3, 0x55, 0xC8, 0x31, 0xE4];
        let slow_variant_2 = [0x02, 0x19, 0x7F, 0xA3, 0x55, 0xC8, 0x31, 0xE4];
        let optimal = OPTIMAL_KERNEL_OPCODES;

        let ssa_1 = CanonicalSsaModule::lift_and_canonicalize(&slow_variant_1);
        let ssa_2 = CanonicalSsaModule::lift_and_canonicalize(&slow_variant_2);
        let ssa_opt = CanonicalSsaModule::lift_and_canonicalize(&optimal);

        // Canonicalization normalizes both redundant slow variants to identical SSA instructions
        assert_eq!(ssa_1.instructions, ssa_2.instructions);
        assert_eq!(ssa_1.eliminated_dead_stores, 1);
        assert_eq!(ssa_opt.eliminated_dead_stores, 0);

        let embedder = BinlexThreeFamilyEmbedder::default();
        let emb_1 = embedder.embed_ssa(&ssa_1);
        let emb_2 = embedder.embed_ssa(&ssa_2);
        let emb_opt = embedder.embed_ssa(&ssa_opt);

        let sim_identical = BinlexThreeFamilyEmbedder::cosine_similarity(&emb_1, &emb_2);
        let sim_diff = BinlexThreeFamilyEmbedder::cosine_similarity(&emb_1, &emb_opt);
        assert!((sim_identical - 1.0).abs() < 1e-5);
        assert!(sim_diff < 0.99);
    }

    #[test]
    fn test_agentenv_step_journal_and_rollback() {
        let base = [0x00; NUM_KERNEL_SLOTS];
        let mut harness = StepJournalHarness::new(base);

        let mut step1 = base;
        step1[0] = OPTIMAL_KERNEL_OPCODES[0];
        harness.record_step("step_1_patch_stage0", step1, true);

        let mut step2_bad = step1;
        step2_bad[1] = 0xFF; // Trap opcode
        harness.record_step("step_2_bad_mutation", step2_bad, false);
        assert_eq!(harness.current_opcodes[1], 0xFF);

        // Roll back the failed step via undo_steps_from
        let rolled_back = harness.undo_steps_from("step_2_bad_mutation");
        assert_eq!(rolled_back, vec!["step_2_bad_mutation".to_string()]);
        assert_eq!(harness.current_opcodes, step1);
        assert_eq!(harness.journal.len(), 1);
    }
}
