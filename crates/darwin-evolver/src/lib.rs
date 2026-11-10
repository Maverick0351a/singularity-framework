//! # `darwin-evolver`
//!
//! Clean-room Rust implementation of the **Darwin Gödel Machine (DGM) Evolutionary Framework**
//! reverse-engineered from `imbue-ai/darwinian_evolver`:
//! - `darwinian_evolver/problem.py` (`Organism`, `EvaluationFailureCase`, `EvaluationResult` with
//!   two-stage weighted failure-type batch sampling `sample_trainable_failure_cases`)
//! - `darwinian_evolver/population.py` (`WeightedSamplingPopulation` with dynamic `p75` linear-interpolated
//!   percentile midpoint, sigmoid performance scaling, and `1 / (1 + novelty_weight * num_children)`
//!   novelty bonus with `expected_children_per_parent` replacement updates)
//! - `darwinian_evolver/learning_log.py` & `learning_log_view.py` (`Empty`, `Ancestors(max_depth)`,
//!   and `Neighborhood(max_distance)` bidirectional lineage graph traversal)
//! - `darwinian_evolver/evolver.py` (`Evolver::evolve_iteration` with `verify_mutation` micro-gate
//!   and atomic generation-end population commit)

use rand::distributions::WeightedIndex;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Lineage metadata carried by every evolved organism (`problem.py:31-59`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismMeta {
    pub id: Uuid,
    pub parent_id: Option<Uuid>,
    pub additional_parent_ids: Vec<Uuid>,
    pub from_failure_cases: Vec<EvaluationFailureCase>,
    pub from_learning_log_entries: Vec<LearningLogEntry>,
    pub from_change_summary: Option<String>,
}

impl OrganismMeta {
    pub fn new_root() -> Self {
        Self {
            id: Uuid::new_v4(),
            parent_id: None,
            additional_parent_ids: Vec::new(),
            from_failure_cases: Vec::new(),
            from_learning_log_entries: Vec::new(),
            from_change_summary: None,
        }
    }

    pub fn new_child(parent_id: Uuid, change_summary: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            parent_id: Some(parent_id),
            additional_parent_ids: Vec::new(),
            from_failure_cases: Vec::new(),
            from_learning_log_entries: Vec::new(),
            from_change_summary: Some(change_summary.into()),
        }
    }
}

pub trait Organism: Clone + Send + Sync {
    fn meta(&self) -> &OrganismMeta;
    fn meta_mut(&mut self) -> &mut OrganismMeta;
}

/// Specific failure case identified during evaluation (`problem.py:61-71`),
/// augmented with optional hardware trace diagnostics from `silicon-trace`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationFailureCase {
    pub data_point_id: String,
    pub failure_type: String,
    pub diagnostic_detail: String,
    pub hot_symbol: Option<String>,
    pub stall_ns: Option<u64>,
}

impl EvaluationFailureCase {
    pub fn new(
        data_point_id: impl Into<String>,
        failure_type: impl Into<String>,
        diagnostic_detail: impl Into<String>,
    ) -> Self {
        Self {
            data_point_id: data_point_id.into(),
            failure_type: failure_type.into(),
            diagnostic_detail: diagnostic_detail.into(),
            hot_symbol: None,
            stall_ns: None,
        }
    }
}

/// Quantitative score and categorized failure cases (`problem.py:73-166`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub score: f64,
    pub trainable_failure_cases: Vec<EvaluationFailureCase>,
    pub holdout_failure_cases: Vec<EvaluationFailureCase>,
    pub is_viable: bool,
    pub failure_type_weights: HashMap<String, f64>,
}

impl EvaluationResult {
    pub fn new(score: f64, trainable_failure_cases: Vec<EvaluationFailureCase>, is_viable: bool) -> Self {
        Self {
            score,
            trainable_failure_cases,
            holdout_failure_cases: Vec::new(),
            is_viable,
            failure_type_weights: HashMap::new(),
        }
    }

    /// Formats the observed outcome relative to the parent's evaluation (`problem.py:88-110`).
    pub fn format_observed_outcome(&self, parent_result: Option<&EvaluationResult>) -> String {
        if !self.is_viable {
            return "Inconclusive - the resulting organism was not viable.".to_string();
        }
        let rounded = (self.score * 100.0).round() / 100.0;
        let mut outcome = format!("The organism achieved an overall fitness score of {:.2}.", rounded);
        if let Some(parent) = parent_result {
            let parent_rounded = (parent.score * 100.0).round() / 100.0;
            if rounded > parent_rounded {
                outcome.push_str(&format!(
                    " This was an improvement over the parent's score of {:.2}.",
                    parent_rounded
                ));
            } else if rounded < parent_rounded {
                outcome.push_str(&format!(
                    " This was worse than the parent's score of {:.2}.",
                    parent_rounded
                ));
            } else {
                outcome.push_str(&format!(
                    " This was the same as the parent's score of {:.2}.",
                    parent_rounded
                ));
            }
        }
        outcome
    }

    /// Two-stage weighted failure case batch sampler (`problem.py:118-152`):
    /// 1. Tallies frequency of each `failure_type` and multiplies by `failure_type_weights`.
    /// 2. Samples a single `failure_type` proportional to weighted frequency.
    /// 3. Uniformly samples up to `batch_size` failure cases of that single type without replacement.
    pub fn sample_trainable_failure_cases<R: Rng>(
        &self,
        batch_size: usize,
        rng: &mut R,
    ) -> Vec<EvaluationFailureCase> {
        if self.trainable_failure_cases.is_empty() || batch_size == 0 {
            return Vec::new();
        }

        let mut type_order: Vec<String> = Vec::new();
        let mut type_freq: HashMap<String, f64> = HashMap::new();
        for fc in &self.trainable_failure_cases {
            if !type_freq.contains_key(&fc.failure_type) {
                type_order.push(fc.failure_type.clone());
            }
            *type_freq.entry(fc.failure_type.clone()).or_insert(0.0) += 1.0;
        }

        for (ft, &w) in &self.failure_type_weights {
            if let Some(freq) = type_freq.get_mut(ft) {
                *freq *= w.max(1e-9);
            }
        }

        let weights: Vec<f64> = type_order.iter().map(|t| type_freq[t]).collect();
        let dist = WeightedIndex::new(&weights).unwrap();
        let chosen_type = &type_order[dist.sample(rng)];

        let matching: Vec<EvaluationFailureCase> = self
            .trainable_failure_cases
            .iter()
            .filter(|fc| &fc.failure_type == chosen_type)
            .cloned()
            .collect();

        let take_n = batch_size.min(matching.len());
        matching.choose_multiple(rng, take_n).cloned().collect()
    }
}

/// Entry in the evolutionary learning log (`learning_log.py:7-12`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LearningLogEntry {
    pub attempted_change: String,
    pub observed_outcome: String,
}

/// Strategy for exposing historical mutation learnings to mutators (`learning_log_view.py:10-92`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LearningLogStrategy {
    None,
    Ancestors { max_depth: Option<usize> },
    Neighborhood { max_distance: usize },
}

/// Midpoint configuration for `WeightedSamplingPopulation` (`population.py:240-265`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MidpointConfig {
    Fixed(f64),
    Percentile(f64), // Default in darwinian_evolver: 75.0 (p75)
}

/// Population archive with Darwin Gödel Machine parent selection (`population.py:19-417`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedSamplingPopulation<O: Organism> {
    organisms: Vec<(O, EvaluationResult)>,
    index_by_id: HashMap<Uuid, usize>,
    children_by_id: HashMap<Uuid, Vec<Uuid>>,
    learning_log: HashMap<Uuid, LearningLogEntry>,
    failed_verification: Vec<O>,
    pub sharpness: f64,
    pub midpoint: MidpointConfig,
    pub novelty_weight: f64,
}

impl<O: Organism> WeightedSamplingPopulation<O> {
    pub fn new(
        initial_organism: O,
        initial_result: EvaluationResult,
        sharpness: f64,
        midpoint: MidpointConfig,
        novelty_weight: f64,
    ) -> Self {
        assert!(
            initial_organism.meta().parent_id.is_none(),
            "Initial organism must not have a parent"
        );
        assert!(initial_result.is_viable, "Initial organism must be viable");

        let mut pop = Self {
            organisms: Vec::new(),
            index_by_id: HashMap::new(),
            children_by_id: HashMap::new(),
            learning_log: HashMap::new(),
            failed_verification: Vec::new(),
            sharpness,
            midpoint,
            novelty_weight,
        };
        pop.add(initial_organism, initial_result);
        pop
    }

    pub fn add(&mut self, organism: O, result: EvaluationResult) {
        let id = organism.meta().id;
        assert!(
            !self.index_by_id.contains_key(&id),
            "Organism already in population"
        );

        if let Some(change) = &organism.meta().from_change_summary {
            let parent_res = organism
                .meta()
                .parent_id
                .and_then(|pid| self.get_by_id(pid))
                .map(|(_, r)| r);
            let outcome = result.format_observed_outcome(parent_res);
            self.learning_log.insert(
                id,
                LearningLogEntry {
                    attempted_change: change.clone(),
                    observed_outcome: outcome,
                },
            );
        }

        if let Some(pid) = organism.meta().parent_id {
            self.children_by_id.entry(pid).or_default().push(id);
        }

        let idx = self.organisms.len();
        self.organisms.push((organism, result));
        self.index_by_id.insert(id, idx);
    }

    pub fn add_failed_verification(&mut self, organism: O) {
        self.failed_verification.push(organism);
    }

    pub fn get_by_id(&self, id: Uuid) -> Option<&(O, EvaluationResult)> {
        self.index_by_id.get(&id).map(|&idx| &self.organisms[idx])
    }

    pub fn get_best(&self) -> &(O, EvaluationResult) {
        self.organisms
            .iter()
            .max_by(|a, b| a.1.score.partial_cmp(&b.1.score).unwrap())
            .expect("Population cannot be empty")
    }

    /// Returns up to `k` highest-scoring viable organisms to serve as active specialist nodes in the Mesh.
    pub fn top_k_organisms(&self, k: usize) -> Vec<(O, EvaluationResult)> {
        let mut sorted: Vec<(O, EvaluationResult)> = self
            .organisms
            .iter()
            .filter(|(_, r)| r.is_viable)
            .cloned()
            .collect();
        sorted.sort_by(|a, b| {
            b.1.score
                .partial_cmp(&a.1.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(k.max(1));
        sorted
    }

    /// Anti-Grey-Goo Memory Cap: Evicts the lowest-scoring organisms when population exceeds `max_organisms`
    /// and bounds `failed_verification` history to `max_failed_log`.
    pub fn prune_to_capacity(&mut self, max_organisms: usize, max_failed_log: usize) {
        let cap = max_organisms.max(2);
        if self.organisms.len() > cap {
            self.organisms.sort_by(|a, b| {
                b.1.score
                    .partial_cmp(&a.1.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.organisms.truncate(cap);
            self.index_by_id.clear();
            for (idx, (org, _)) in self.organisms.iter().enumerate() {
                self.index_by_id.insert(org.meta().id, idx);
            }
        }
        if self.failed_verification.len() > max_failed_log {
            let drain_count = self.failed_verification.len() - max_failed_log;
            self.failed_verification.drain(0..drain_count);
        }
    }

    pub fn len(&self) -> usize {
        self.organisms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.organisms.is_empty()
    }

    pub fn failed_verification_count(&self) -> usize {
        self.failed_verification.len()
    }

    /// Returns all `LearningLogEntry` records across both accepted organisms (`learning_log`)
    /// and rejected mutations (`failed_verification`), enabling 100% blind hardware mutators
    /// to avoid re-probing single-bit overrides that have already been tested.
    pub fn all_learning_log_entries_with_failures(&self) -> Vec<LearningLogEntry> {
        let mut entries: Vec<LearningLogEntry> = self.learning_log.values().cloned().collect();
        for failed_org in &self.failed_verification {
            if let Some(change) = &failed_org.meta().from_change_summary {
                entries.push(LearningLogEntry {
                    attempted_change: change.clone(),
                    observed_outcome:
                        "Rejected by verify_mutation hardware micro-gate (stalls increased)."
                            .to_string(),
                });
            }
        }
        entries
    }

    /// Computes linear-interpolated score percentile (`population.py:181-208`).
    pub fn compute_percentile(&self, percentile: f64) -> f64 {
        let mut scores: Vec<f64> = self.organisms.iter().map(|(_, r)| r.score).collect();
        if scores.is_empty() {
            return 0.0;
        }
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = scores.len();
        if n == 1 {
            return scores[0];
        }
        let k = ((n - 1) as f64) * (percentile.clamp(0.0, 100.0) / 100.0);
        let f = k.floor() as usize;
        let c = k.ceil() as usize;
        if f == c {
            scores[f]
        } else {
            scores[f] * ((c as f64) - k) + scores[c] * (k - (f as f64))
        }
    }

    pub fn current_midpoint_score(&self) -> f64 {
        match self.midpoint {
            MidpointConfig::Fixed(val) => val,
            MidpointConfig::Percentile(p) => self.compute_percentile(p),
        }
    }

    /// Samples `k` parents using Darwin Gödel Machine weighted sampling (`population.py:317-416`):
    /// `weight = sigmoid(sharpness * (score - midpoint)) * (1 / (1 + novelty_weight * num_children))`
    /// while incrementing `temp_children_counts` by `expected_children_per_parent` on each draw.
    pub fn sample_parents<R: Rng>(
        &self,
        k: usize,
        expected_children_per_parent: f64,
        rng: &mut R,
    ) -> Vec<(O, EvaluationResult)> {
        let eligible: Vec<&(O, EvaluationResult)> = self
            .organisms
            .iter()
            .filter(|(_, res)| res.is_viable && !res.trainable_failure_cases.is_empty())
            .collect();

        if eligible.is_empty() {
            // If an organism solved all trainable failure cases (perfect score), return the best
            return vec![self.get_best().clone()];
        }

        let midpoint = self.current_midpoint_score();
        let mut temp_children: HashMap<Uuid, f64> = eligible
            .iter()
            .map(|(org, _)| {
                let count = self
                    .children_by_id
                    .get(&org.meta().id)
                    .map(|v| v.len() as f64)
                    .unwrap_or(0.0);
                (org.meta().id, count)
            })
            .collect();

        let mut sampled = Vec::with_capacity(k);
        for _ in 0..k {
            let weights: Vec<f64> = eligible
                .iter()
                .map(|(org, res)| {
                    let sig = 1.0 / (1.0 + (-self.sharpness * (res.score - midpoint)).exp());
                    let n_child = temp_children[&org.meta().id];
                    let novelty = 1.0 / (1.0 + self.novelty_weight * n_child);
                    (sig * novelty).max(1e-12)
                })
                .collect();

            let dist = WeightedIndex::new(&weights).unwrap();
            let chosen_idx = dist.sample(rng);
            let chosen = eligible[chosen_idx];
            *temp_children.get_mut(&chosen.0.meta().id).unwrap() += expected_children_per_parent;
            sampled.push((*chosen).clone());
        }
        sampled
    }

    /// Retrieves learning log entries visible to `organism` (`learning_log_view.py:30-92`).
    pub fn get_learning_log_entries(
        &self,
        organism: &O,
        strategy: LearningLogStrategy,
    ) -> Vec<LearningLogEntry> {
        match strategy {
            LearningLogStrategy::None => Vec::new(),
            LearningLogStrategy::Ancestors { max_depth } => {
                let mut entries = Vec::new();
                let mut cur_id = Some(organism.meta().id);
                let mut depth = 0usize;
                while let Some(id) = cur_id {
                    if max_depth.map_or(false, |max| depth > max) {
                        break;
                    }
                    if let Some(entry) = self.learning_log.get(&id) {
                        entries.push(entry.clone());
                    }
                    cur_id = self.get_by_id(id).and_then(|(o, _)| o.meta().parent_id);
                    depth += 1;
                }
                entries
            }
            LearningLogStrategy::Neighborhood { max_distance } => {
                let mut visited = HashSet::new();
                let mut in_range_ids = Vec::new();
                self.traverse_neighborhood(organism.meta().id, 0, max_distance, &mut visited, &mut in_range_ids);
                in_range_ids
                    .into_iter()
                    .filter_map(|id| self.learning_log.get(&id).cloned())
                    .collect()
            }
        }
    }

    fn traverse_neighborhood(
        &self,
        current_id: Uuid,
        dist: usize,
        max_dist: usize,
        visited: &mut HashSet<Uuid>,
        out: &mut Vec<Uuid>,
    ) {
        if dist > max_dist || !visited.insert(current_id) {
            return;
        }
        out.push(current_id);
        if let Some((org, _)) = self.get_by_id(current_id) {
            if let Some(pid) = org.meta().parent_id {
                self.traverse_neighborhood(pid, dist + 1, max_dist, visited, out);
            }
        }
        if let Some(children) = self.children_by_id.get(&current_id) {
            for &cid in children {
                self.traverse_neighborhood(cid, dist + 1, max_dist, visited, out);
            }
        }
    }
}

pub trait Mutator<O: Organism>: Send + Sync {
    fn supports_batch_mutation(&self) -> bool {
        false
    }

    fn mutate(
        &self,
        parent: &O,
        failure_cases: &[EvaluationFailureCase],
        learning_log_entries: &[LearningLogEntry],
    ) -> Vec<O>;
}

pub trait Evaluator<O: Organism>: Send + Sync {
    fn evaluate(&self, organism: &O) -> EvaluationResult;

    /// Fast post-mutation verification filter (`evolver.py:201-202`).
    fn verify_mutation(&self, organism: &O) -> bool {
        let _ = organism;
        true
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvolverStats {
    pub num_mutate_calls: usize,
    pub num_failure_cases_supplied: usize,
    pub num_generated_mutations: usize,
    pub num_mutations_after_verification: usize,
    pub num_evaluate_calls: usize,
    pub num_verify_mutation_calls: usize,
    pub num_learning_log_entries_supplied: usize,
}

/// Orchestrates the Darwinian evolutionary loop (`evolver.py:73-225`).
pub struct Evolver<O: Organism> {
    pub population: WeightedSamplingPopulation<O>,
    pub mutators: Vec<Box<dyn Mutator<O>>>,
    pub evaluator: Box<dyn Evaluator<O>>,
    pub learning_log_strategy: LearningLogStrategy,
    pub batch_size: usize,
    pub should_verify_mutations: bool,
}

impl<O: Organism> Evolver<O> {
    pub fn new(
        population: WeightedSamplingPopulation<O>,
        mutators: Vec<Box<dyn Mutator<O>>>,
        evaluator: Box<dyn Evaluator<O>>,
        learning_log_strategy: LearningLogStrategy,
        batch_size: usize,
        should_verify_mutations: bool,
    ) -> Self {
        Self {
            population,
            mutators,
            evaluator,
            learning_log_strategy,
            batch_size: batch_size.max(1),
            should_verify_mutations,
        }
    }

    /// Runs a single evolutionary generation (`evolver.py:113-199`):
    /// 1. Samples `num_parents` using sigmoid(`p75`) * novelty bonus.
    /// 2. Samples failure cases and lineage `learning_log_entries` for each parent.
    /// 3. Invokes mutators to produce candidate offspring.
    /// 4. Runs `verify_mutation` micro-gate before full `evaluate`.
    /// 5. Atomically commits all evaluated offspring to the population at the end of the generation.
    pub fn evolve_iteration<R: Rng>(&mut self, num_parents: usize, rng: &mut R) -> EvolverStats {
        let mut stats = EvolverStats::default();
        let parents = self.population.sample_parents(num_parents, 1.0, rng);

        let mut candidate_children: Vec<O> = Vec::new();

        for (parent_org, parent_eval) in &parents {
            if parent_eval.trainable_failure_cases.is_empty() {
                continue;
            }
            let failure_cases = parent_eval.sample_trainable_failure_cases(self.batch_size, rng);
            let learning_entries = self
                .population
                .get_learning_log_entries(parent_org, self.learning_log_strategy);

            for mutator in &self.mutators {
                let cases_for_mutator = if mutator.supports_batch_mutation() {
                    failure_cases.clone()
                } else {
                    vec![failure_cases[0].clone()]
                };

                let mut children = mutator.mutate(parent_org, &cases_for_mutator, &learning_entries);
                stats.num_mutate_calls += 1;
                stats.num_failure_cases_supplied += cases_for_mutator.len();
                stats.num_learning_log_entries_supplied += learning_entries.len();
                stats.num_generated_mutations += children.len();

                for child in &mut children {
                    let meta = child.meta_mut();
                    if meta.parent_id.is_none() {
                        meta.parent_id = Some(parent_org.meta().id);
                    }
                    if meta.from_failure_cases.is_empty() {
                        meta.from_failure_cases = cases_for_mutator.clone();
                    }
                    if meta.from_learning_log_entries.is_empty() {
                        meta.from_learning_log_entries = learning_entries.clone();
                    }
                }
                candidate_children.extend(children);
            }
        }

        // Phase 2: Post-mutation verification filter + Evaluation
        let mut evaluated_batch: Vec<(O, EvaluationResult)> = Vec::new();
        for child in candidate_children {
            let passed = if self.should_verify_mutations {
                stats.num_verify_mutation_calls += 1;
                self.evaluator.verify_mutation(&child)
            } else {
                true
            };

            if passed {
                stats.num_mutations_after_verification += 1;
                stats.num_evaluate_calls += 1;
                let res = self.evaluator.evaluate(&child);
                evaluated_batch.push((child, res));
            } else {
                self.population.add_failed_verification(child);
            }
        }

        // Phase 3: Atomic commit to population so same-iteration children don't leak into learning logs early
        for (child, res) in evaluated_batch {
            self.population.add(child, res);
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
    struct DummyOrganism {
        meta: OrganismMeta,
        val: i32,
    }

    impl Organism for DummyOrganism {
        fn meta(&self) -> &OrganismMeta {
            &self.meta
        }
        fn meta_mut(&mut self) -> &mut OrganismMeta {
            &mut self.meta
        }
    }

    #[test]
    fn test_population_percentile_and_neighborhood_learning_log() {
        let root = DummyOrganism {
            meta: OrganismMeta::new_root(),
            val: 10,
        };
        let root_id = root.meta.id;
        let res0 = EvaluationResult::new(
            0.25,
            vec![EvaluationFailureCase::new("case1", "math", "off by 1")],
            true,
        );

        let mut pop = WeightedSamplingPopulation::new(
            root,
            res0,
            10.0,
            MidpointConfig::Percentile(75.0),
            1.0,
        );

        let child1 = DummyOrganism {
            meta: OrganismMeta::new_child(root_id, "incremented val to 20"),
            val: 20,
        };
        let child1_id = child1.meta.id;
        pop.add(
            child1.clone(),
            EvaluationResult::new(
                0.75,
                vec![EvaluationFailureCase::new("case2", "math", "still off")],
                true,
            ),
        );

        let child2 = DummyOrganism {
            meta: OrganismMeta::new_child(root_id, "decremented val to 5"),
            val: 5,
        };
        pop.add(
            child2,
            EvaluationResult::new(
                0.10,
                vec![EvaluationFailureCase::new("case1", "math", "worse")],
                true,
            ),
        );

        // Sibling (`child2`) is distance 2 from `child1` (child1 -> root -> child2)
        let entries_d1 =
            pop.get_learning_log_entries(&child1, LearningLogStrategy::Neighborhood { max_distance: 1 });
        assert_eq!(entries_d1.len(), 1); // Only child1 has a change summary in distance 1 (root has none)

        let entries_d2 =
            pop.get_learning_log_entries(&child1, LearningLogStrategy::Neighborhood { max_distance: 2 });
        assert_eq!(entries_d2.len(), 2); // Both child1 and sibling child2!
        assert_eq!(pop.get_best().0.meta.id, child1_id);
        assert_eq!(pop.get_best().0.val, 20);
    }
}
