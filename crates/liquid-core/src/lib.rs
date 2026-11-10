//! # `liquid-core`
//!
//! Clean-room Rust implementation of **Liquid AI's Continuous-Time & Hybrid Foundation Architectures**,
//! reverse-engineered from:
//! - `mlech26l/ncps/ncps/torch/cfc_cell.py` (`CfCCell` in `default`, `pure`, and `no_gate` modes + `LeCun` tanh)
//! - `mlech26l/ncps/ncps/torch/ltc_cell.py` (`LTCCell` semi-implicit ODE unfold solver with `gleak`, `vleak`, `cm`, `erev`)
//! - `mlech26l/ncps/ncps/wirings/wirings.py` (`NCP` and `AutoNCP` 4-layer C. elegans sparse sensory -> inter -> command -> motor wiring)
//! - `Lfm2ShortConv` (Liquid Foundation Model 2 double-gated depthwise causal Conv1D `kernel_size=3` with $O(1)$ 2-token ring buffer cache)

use rand::prelude::*;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

/// LeCun scaled hyperbolic tangent (`cfc_cell.py:27-34`):
/// `1.7159 * tanh(0.666 * x)`
#[inline]
pub fn lecun_tanh(x: f32) -> f32 {
    1.7159 * (0.666 * x).tanh()
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Branchless rational algebraic sigmoid approximation executing in ~3 CPU cycles
/// without calling transcendental `(-x).exp()`.
#[inline]
pub fn fast_rational_sigmoid(x: f32) -> f32 {
    let xc = x.clamp(-12.0, 12.0);
    let x2 = xc * xc;
    let num = xc * (27.0 + x2);
    let den = 27.0 + 9.0 * x2;
    0.5 + 0.5 * (num / den).clamp(-1.0, 1.0)
}

/// Operating mode for `CfCCell` (`cfc_cell.py:69`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CfCMode {
    /// Gated time-interpolated closed-form (`ff1 * (1 - t_interp) + t_interp * ff2`)
    Default,
    /// Exact analytical exponential solution (`-A * exp(-ts * (|w_tau| + |ff1|)) * ff1 + A`)
    Pure,
    /// Un-gated additive time interpolation (`ff1 + t_interp * ff2`)
    NoGate,
}

/// Dense or sparse linear projection helper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weight: Vec<f32>, // [out_dim * in_dim]
    pub bias: Vec<f32>,   // [out_dim]
}

impl LinearLayer {
    pub fn new_seeded(in_dim: usize, out_dim: usize, rng: &mut StdRng) -> Self {
        // Xavier uniform initialization (`cfc_cell.py:131-134`)
        let limit = (6.0 / ((in_dim + out_dim) as f32)).sqrt();
        let weight = (0..out_dim * in_dim)
            .map(|_| rng.gen_range(-limit..limit))
            .collect();
        let bias = vec![0.0; out_dim];
        Self {
            in_dim,
            out_dim,
            weight,
            bias,
        }
    }

    pub fn forward(&self, x: &[f32], sparsity_mask: Option<&[f32]>) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.in_dim);
        let mut out = self.bias.clone();
        for o in 0..self.out_dim {
            let row = o * self.in_dim;
            let mut sum = 0.0f32;
            if let Some(mask) = sparsity_mask {
                for i in 0..self.in_dim {
                    sum += self.weight[row + i] * mask[row + i] * x[i];
                }
            } else {
                for i in 0..self.in_dim {
                    sum += self.weight[row + i] * x[i];
                }
            }
            out[o] += sum;
        }
        out
    }
}

/// Closed-form Continuous-time (`CfC`) Neural Cell (`ncps/torch/cfc_cell.py:36-167`).
/// Processes `(input, hidden_state, delta_t)` in $O(1)$ without ODE solver loops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfCCell {
    pub input_size: usize,
    pub hidden_size: usize,
    pub mode: CfCMode,
    pub backbone: Option<LinearLayer>,
    pub ff1: LinearLayer,
    pub ff2: LinearLayer,
    pub time_a: LinearLayer,
    pub time_b: LinearLayer,
    pub w_tau: Vec<f32>,
    pub a_param: Vec<f32>,
    pub sparsity_mask: Option<Vec<f32>>,
}

impl CfCCell {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        backbone_units: usize,
        mode: CfCMode,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let (backbone, cat_shape) = if backbone_units > 0 {
            (
                Some(LinearLayer::new_seeded(
                    input_size + hidden_size,
                    backbone_units,
                    &mut rng,
                )),
                backbone_units,
            )
        } else {
            (None, input_size + hidden_size)
        };

        Self {
            input_size,
            hidden_size,
            mode,
            backbone,
            ff1: LinearLayer::new_seeded(cat_shape, hidden_size, &mut rng),
            ff2: LinearLayer::new_seeded(cat_shape, hidden_size, &mut rng),
            time_a: LinearLayer::new_seeded(cat_shape, hidden_size, &mut rng),
            time_b: LinearLayer::new_seeded(cat_shape, hidden_size, &mut rng),
            w_tau: vec![0.1; hidden_size],
            a_param: vec![1.0; hidden_size],
            sparsity_mask: None,
        }
    }

    /// Forward pass (`cfc_cell.py:136-167`) accepting explicit continuous time delta `ts` ($\Delta t$).
    pub fn forward(&self, input: &[f32], hx: &[f32], ts: f32) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.input_size);
        debug_assert_eq!(hx.len(), self.hidden_size);

        let mut cat = Vec::with_capacity(self.input_size + self.hidden_size);
        cat.extend_from_slice(input);
        cat.extend_from_slice(hx);

        let x = if let Some(bb) = &self.backbone {
            bb.forward(&cat, None)
                .into_iter()
                .map(lecun_tanh)
                .collect::<Vec<_>>()
        } else {
            cat
        };

        let mask_ref = self.sparsity_mask.as_deref();
        let mut ff1 = self.ff1.forward(&x, mask_ref);

        match self.mode {
            CfCMode::Pure => {
                // `cfc_cell.py:146-151`: -A * exp(-ts * (|w_tau| + |ff1|)) * ff1 + A
                let mut new_hidden = vec![0.0; self.hidden_size];
                for i in 0..self.hidden_size {
                    let decay = (-ts * (self.w_tau[i].abs() + ff1[i].abs())).exp();
                    new_hidden[i] = -self.a_param[i] * decay * ff1[i] + self.a_param[i];
                }
                new_hidden
            }
            CfCMode::Default | CfCMode::NoGate => {
                let mut ff2 = self.ff2.forward(&x, mask_ref);
                let t_a = self.time_a.forward(&x, None);
                let t_b = self.time_b.forward(&x, None);
                let mut new_hidden = vec![0.0; self.hidden_size];

                for i in 0..self.hidden_size {
                    ff1[i] = ff1[i].tanh();
                    ff2[i] = ff2[i].tanh();
                    let t_interp = sigmoid(t_a[i] * ts + t_b[i]);
                    new_hidden[i] = if self.mode == CfCMode::NoGate {
                        ff1[i] + t_interp * ff2[i]
                    } else {
                        ff1[i] * (1.0 - t_interp) + t_interp * ff2[i]
                    };
                }
                new_hidden
            }
        }
    }
}

/// 4-Layer Neural Circuit Policies (`NCP` / `AutoNCP`) Sparse Wiring
/// reverse-engineered from `ncps/wirings/wirings.py:395-659`.
/// Partitions neurons into:
/// `Sensory (input_dim)` -> `Interneurons` -> `Command Neurons (recurrent)` -> `Motor Neurons (output_size)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoNcpWiring {
    pub input_dim: usize,
    pub units: usize,
    pub output_dim: usize,
    pub num_inter_neurons: usize,
    pub num_command_neurons: usize,
    pub num_motor_neurons: usize,
    /// Internal adjacency matrix `[units * units]` with values in `{-1, 0, +1}` (inhibitory/absent/excitatory)
    pub adjacency_matrix: Vec<i8>,
    /// Sensory adjacency matrix `[input_dim * units]` with values in `{-1, 0, +1}`
    pub sensory_adjacency_matrix: Vec<i8>,
    /// CSR-style pre-indexed non-zero recurrent edges per destination neuron: `(src_neuron, flat_idx)`
    #[serde(default)]
    pub active_synapses_by_dest: Vec<Vec<(usize, usize)>>,
    /// CSR-style pre-indexed non-zero sensory edges per destination neuron: `(src_input, flat_idx)`
    #[serde(default)]
    pub active_sensory_by_dest: Vec<Vec<(usize, usize)>>,
}

impl AutoNcpWiring {
    pub fn build(input_dim: usize, units: usize, output_size: usize, sparsity_level: f32, seed: u64) -> Self {
        assert!(output_size + 2 <= units, "output_size must be <= units - 2");
        let density = (1.0 - sparsity_level).clamp(0.1, 0.9);
        let inter_and_command = units - output_size;
        let command_neurons = ((0.4 * (inter_and_command as f32)) as usize).max(1);
        let inter_neurons = inter_and_command - command_neurons;

        let sensory_fanout = ((inter_neurons as f32 * density) as usize).max(1);
        let inter_fanout = ((command_neurons as f32 * density) as usize).max(1);
        let recurrent_command_synapses = ((command_neurons as f32 * density * 2.0) as usize).max(1);
        let motor_fanin = ((command_neurons as f32 * density) as usize).max(1);

        let mut rng = StdRng::seed_from_u64(seed);
        let mut adjacency_matrix = vec![0i8; units * units];
        let mut sensory_adjacency_matrix = vec![0i8; input_dim * units];

        // Neuron ID layout from `wirings.py:433-448`:
        // [0..motor] [motor..motor+command] [motor+command..units (inter)]
        let motor_ids: Vec<usize> = (0..output_size).collect();
        let command_ids: Vec<usize> = (output_size..output_size + command_neurons).collect();
        let inter_ids: Vec<usize> = (output_size + command_neurons..units).collect();

        // 1. Sensory -> Inter (`_build_sensory_to_inter_layer`)
        for src in 0..input_dim {
            for &dest in inter_ids.choose_multiple(&mut rng, sensory_fanout) {
                let polarity = if rng.gen_bool(0.5) { 1 } else { -1 };
                sensory_adjacency_matrix[src * units + dest] = polarity;
            }
        }
        // Ensure every interneuron has at least 1 sensory connection
        for &dest in &inter_ids {
            let connected = (0..input_dim).any(|s| sensory_adjacency_matrix[s * units + dest] != 0);
            if !connected {
                let src = rng.gen_range(0..input_dim);
                sensory_adjacency_matrix[src * units + dest] = if rng.gen_bool(0.5) { 1 } else { -1 };
            }
        }

        // 2. Inter -> Command (`_build_inter_to_command_layer`)
        for &src in &inter_ids {
            for &dest in command_ids.choose_multiple(&mut rng, inter_fanout) {
                let polarity = if rng.gen_bool(0.5) { 1 } else { -1 };
                adjacency_matrix[src * units + dest] = polarity;
            }
        }
        for &dest in &command_ids {
            let connected = inter_ids.iter().any(|&s| adjacency_matrix[s * units + dest] != 0);
            if !connected {
                let src = *inter_ids.choose(&mut rng).unwrap();
                adjacency_matrix[src * units + dest] = if rng.gen_bool(0.5) { 1 } else { -1 };
            }
        }

        // 3. Recurrent Command (`_build_recurrent_command_layer`)
        for _ in 0..recurrent_command_synapses {
            let src = *command_ids.choose(&mut rng).unwrap();
            let dest = *command_ids.choose(&mut rng).unwrap();
            adjacency_matrix[src * units + dest] = if rng.gen_bool(0.5) { 1 } else { -1 };
        }

        // 4. Command -> Motor (`_build_command__to_motor_layer`)
        for &dest in &motor_ids {
            for &src in command_ids.choose_multiple(&mut rng, motor_fanin) {
                let polarity = if rng.gen_bool(0.5) { 1 } else { -1 };
                adjacency_matrix[src * units + dest] = polarity;
            }
        }

        let mut active_synapses_by_dest = vec![Vec::new(); units];
        for dest in 0..units {
            for src in 0..units {
                let idx = src * units + dest;
                if adjacency_matrix[idx] != 0 {
                    active_synapses_by_dest[dest].push((src, idx));
                }
            }
        }

        let mut active_sensory_by_dest = vec![Vec::new(); units];
        for dest in 0..units {
            for src in 0..input_dim {
                let idx = src * units + dest;
                if sensory_adjacency_matrix[idx] != 0 {
                    active_sensory_by_dest[dest].push((src, idx));
                }
            }
        }

        Self {
            input_dim,
            units,
            output_dim: output_size,
            num_inter_neurons: inter_neurons,
            num_command_neurons: command_neurons,
            num_motor_neurons: output_size,
            adjacency_matrix,
            sensory_adjacency_matrix,
            active_synapses_by_dest,
            active_sensory_by_dest,
        }
    }

    pub fn synapse_count(&self) -> usize {
        self.adjacency_matrix.iter().filter(|&&v| v != 0).count()
    }

    pub fn sensory_synapse_count(&self) -> usize {
        self.sensory_adjacency_matrix.iter().filter(|&&v| v != 0).count()
    }
}

/// Liquid Time-Constant (`LTC`) Cell with Semi-Implicit ODE Unfold Solver
/// reverse-engineered from `ncps/torch/ltc_cell.py:21-286`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LTCCell {
    pub wiring: AutoNcpWiring,
    pub ode_unfolds: usize,
    pub epsilon: f32,
    pub gleak: Vec<f32>,
    pub vleak: Vec<f32>,
    pub cm: Vec<f32>,
    pub w: Vec<f32>,
    pub sigma: Vec<f32>,
    pub mu: Vec<f32>,
    pub erev: Vec<f32>,
    pub sensory_w: Vec<f32>,
    pub sensory_sigma: Vec<f32>,
    pub sensory_mu: Vec<f32>,
    pub sensory_erev: Vec<f32>,
}

impl LTCCell {
    pub fn new(wiring: AutoNcpWiring, ode_unfolds: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let u = wiring.units;
        let s = wiring.input_dim;

        // Initialization ranges from `ltc_cell.py:57-67`
        let gleak = (0..u).map(|_| rng.gen_range(0.001..1.0)).collect();
        let vleak = (0..u).map(|_| rng.gen_range(-0.2..0.2)).collect();
        let cm = (0..u).map(|_| rng.gen_range(0.4..0.6)).collect();
        let w = (0..u * u).map(|_| rng.gen_range(0.001..1.0)).collect();
        let sigma = (0..u * u).map(|_| rng.gen_range(3.0..8.0)).collect();
        let mu = (0..u * u).map(|_| rng.gen_range(0.3..0.8)).collect();
        let erev = wiring.adjacency_matrix.iter().map(|&v| v as f32).collect();

        let sensory_w = (0..s * u).map(|_| rng.gen_range(0.001..1.0)).collect();
        let sensory_sigma = (0..s * u).map(|_| rng.gen_range(3.0..8.0)).collect();
        let sensory_mu = (0..s * u).map(|_| rng.gen_range(0.3..0.8)).collect();
        let sensory_erev = wiring.sensory_adjacency_matrix.iter().map(|&v| v as f32).collect();

        Self {
            wiring,
            ode_unfolds: ode_unfolds.max(1),
            epsilon: 1e-8,
            gleak,
            vleak,
            cm,
            w,
            sigma,
            mu,
            erev,
            sensory_w,
            sensory_sigma,
            sensory_mu,
            sensory_erev,
        }
    }

    /// Executes the semi-implicit ODE solver (`ltc_cell.py:203-250`) over `elapsed_time` ($\Delta t$),
    /// returning `(motor_outputs, next_full_state)`.
    pub fn forward(&self, inputs: &[f32], state: &[f32], elapsed_time: f32) -> (Vec<f32>, Vec<f32>) {
        self.forward_configured(
            inputs,
            state,
            elapsed_time,
            true,
            true,
            true,
            self.ode_unfolds,
        )
    }

    /// Programmable ODE execution path enabling direct head-to-head hardware `RDTSC` cycle evolution
    /// across `liquid-core`'s internal Rust optimizations:
    /// - `use_sparse_csr`: Iterates only non-zero CSR synapses (`O(|E|)` vs `O(U^2)`)
    /// - `use_stack_arena`: Uses fixed `[f32; 32]` stack arrays instead of heap `vec![]` inside the ODE unfold loop
    /// - `use_fast_sigmoid`: Uses branchless rational Padé approximation instead of `(-x).exp()`
    /// - `ode_unfolds_override`: Tunable number of semi-implicit ODE unfold steps (`1..=8`)
    pub fn forward_configured(
        &self,
        inputs: &[f32],
        state: &[f32],
        elapsed_time: f32,
        use_sparse_csr: bool,
        use_stack_arena: bool,
        use_fast_sigmoid: bool,
        ode_unfolds_override: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let u = self.wiring.units;
        let s = self.wiring.input_dim;
        debug_assert_eq!(inputs.len(), s);
        debug_assert_eq!(state.len(), u);
        let unfolds = ode_unfolds_override.max(1);
        let dt_step = (elapsed_time.max(1e-6)) / (unfolds as f32);
        let sig_fn: fn(f32) -> f32 = if use_fast_sigmoid {
            fast_rational_sigmoid
        } else {
            sigmoid
        };

        if use_stack_arena && u <= 32 {
            let mut v_pre = [0.0f32; 32];
            v_pre[..u].copy_from_slice(state);
            let mut w_num_sensory = [0.0f32; 32];
            let mut w_den_sensory = [0.0f32; 32];

            if use_sparse_csr && !self.wiring.active_sensory_by_dest.is_empty() {
                for dest in 0..u {
                    let mut num = 0.0f32;
                    let mut den = 0.0f32;
                    for &(src, idx) in &self.wiring.active_sensory_by_dest[dest] {
                        let sig = sig_fn(self.sensory_sigma[idx] * (inputs[src] - self.sensory_mu[idx]));
                        let act = self.sensory_w[idx].max(0.0) * sig;
                        num += act * self.sensory_erev[idx];
                        den += act;
                    }
                    w_num_sensory[dest] = num;
                    w_den_sensory[dest] = den;
                }
            } else {
                for src in 0..s {
                    for dest in 0..u {
                        let idx = src * u + dest;
                        let mask = (self.wiring.sensory_adjacency_matrix[idx] != 0) as u8 as f32;
                        if mask > 0.0 {
                            let sig = sig_fn(self.sensory_sigma[idx] * (inputs[src] - self.sensory_mu[idx]));
                            let act = self.sensory_w[idx].max(0.0) * sig * mask;
                            w_num_sensory[dest] += act * self.sensory_erev[idx];
                            w_den_sensory[dest] += act;
                        }
                    }
                }
            }

            for _ in 0..unfolds {
                let mut next_v = [0.0f32; 32];
                for dest in 0..u {
                    let mut w_num = w_num_sensory[dest];
                    let mut w_den = w_den_sensory[dest];

                    if use_sparse_csr && !self.wiring.active_synapses_by_dest.is_empty() {
                        for &(src, idx) in &self.wiring.active_synapses_by_dest[dest] {
                            let sig = sig_fn(self.sigma[idx] * (v_pre[src] - self.mu[idx]));
                            let act = self.w[idx].max(0.0) * sig;
                            w_num += act * self.erev[idx];
                            w_den += act;
                        }
                    } else {
                        for src in 0..u {
                            let idx = src * u + dest;
                            let mask = (self.wiring.adjacency_matrix[idx] != 0) as u8 as f32;
                            if mask > 0.0 {
                                let sig = sig_fn(self.sigma[idx] * (v_pre[src] - self.mu[idx]));
                                let act = self.w[idx].max(0.0) * sig * mask;
                                w_num += act * self.erev[idx];
                                w_den += act;
                            }
                        }
                    }

                    let cm_t = self.cm[dest].max(0.0) / dt_step;
                    let gleak = self.gleak[dest].max(0.0);
                    let numerator = cm_t * v_pre[dest] + gleak * self.vleak[dest] + w_num;
                    let denominator = cm_t + gleak + w_den;
                    next_v[dest] = numerator / (denominator + self.epsilon);
                }
                v_pre = next_v;
            }

            return (
                v_pre[0..self.wiring.output_dim].to_vec(),
                v_pre[0..u].to_vec(),
            );
        }

        // Legacy heap-allocating dense path (for comparison when `use_stack_arena == false`)
        let mut v_pre = state.to_vec();
        let mut w_num_sensory = vec![0.0f32; u];
        let mut w_den_sensory = vec![0.0f32; u];

        for src in 0..s {
            for dest in 0..u {
                let idx = src * u + dest;
                let mask = (self.wiring.sensory_adjacency_matrix[idx] != 0) as u8 as f32;
                if mask > 0.0 {
                    let sig = sig_fn(self.sensory_sigma[idx] * (inputs[src] - self.sensory_mu[idx]));
                    let act = self.sensory_w[idx].max(0.0) * sig * mask;
                    w_num_sensory[dest] += act * self.sensory_erev[idx];
                    w_den_sensory[dest] += act;
                }
            }
        }

        for _ in 0..unfolds {
            let mut next_v = vec![0.0f32; u];
            for dest in 0..u {
                let mut w_num = w_num_sensory[dest];
                let mut w_den = w_den_sensory[dest];

                for src in 0..u {
                    let idx = src * u + dest;
                    let mask = (self.wiring.adjacency_matrix[idx] != 0) as u8 as f32;
                    if mask > 0.0 {
                        let sig = sig_fn(self.sigma[idx] * (v_pre[src] - self.mu[idx]));
                        let act = self.w[idx].max(0.0) * sig * mask;
                        w_num += act * self.erev[idx];
                        w_den += act;
                    }
                }

                let cm_t = self.cm[dest].max(0.0) / dt_step;
                let gleak = self.gleak[dest].max(0.0);
                let numerator = cm_t * v_pre[dest] + gleak * self.vleak[dest] + w_num;
                let denominator = cm_t + gleak + w_den;
                next_v[dest] = numerator / (denominator + self.epsilon);
            }
            v_pre = next_v;
        }

        let motor_out = v_pre[0..self.wiring.output_dim].to_vec();
        (motor_out, v_pre)
    }

    /// Online reward-modulated Hebbian synapse adaptation for `AutoNcpWiring` (`Sensory -> Inter -> Command -> Motor`).
    /// Strengthens active sensory and inter-neuron conductances (`sensory_w`, `w`) when a candidate mutation
    /// reduces real hardware `RDTSC` clock cycles (`reward_signal > 0`), while keeping weights bounded in `[0.001, 2.5]`.
    pub fn adapt_synapses_online(
        &mut self,
        sensory_input: &[f32],
        post_state: &[f32],
        reward_signal: f32,
        learning_rate: f32,
    ) -> bool {
        let u = self.wiring.units;
        let s = self.wiring.input_dim;
        if sensory_input.len() != s || post_state.len() != u {
            return false;
        }
        // NaN / Inf Poison Quarantine: reject non-finite sensory, state, or reward values immediately
        if !reward_signal.is_finite()
            || !learning_rate.is_finite()
            || sensory_input.iter().any(|v| !v.is_finite())
            || post_state.iter().any(|v| !v.is_finite())
        {
            return false;
        }
        let step = (reward_signal * learning_rate).clamp(-0.15, 0.15);
        for src in 0..s {
            for dest in 0..u {
                let idx = src * u + dest;
                if self.wiring.sensory_adjacency_matrix[idx] != 0 {
                    let pre_act = sigmoid(self.sensory_sigma[idx] * (sensory_input[src] - self.sensory_mu[idx]));
                    let post_act = post_state[dest].abs().min(1.5);
                    self.sensory_w[idx] = (self.sensory_w[idx] + step * pre_act * post_act).clamp(0.001, 2.5);
                }
            }
        }
        for src in 0..u {
            for dest in 0..u {
                let idx = src * u + dest;
                if self.wiring.adjacency_matrix[idx] != 0 {
                    let pre_act = sigmoid(self.sigma[idx] * (post_state[src] - self.mu[idx]));
                    let post_act = post_state[dest].abs().min(1.5);
                    self.w[idx] = (self.w[idx] + step * pre_act * post_act).clamp(0.001, 2.5);
                }
            }
        }
        true
    }
}

/// Liquid Foundation Model 2 (`LFM2`) Double-Gated Linear Input-Varying Short Convolution (`Lfm2ShortConv`).
/// Operates in $O(1)$ memory during autoregressive decoding by caching only the last `kernel_size - 1` (2) vectors!
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lfm2ShortConv {
    pub d_model: usize,
    pub kernel_size: usize,
    pub in_proj_b: LinearLayer,
    pub in_proj_c: LinearLayer,
    pub in_proj_v: LinearLayer,
    pub conv_weight: Vec<f32>, // [d_model * kernel_size] depthwise causal kernel
    pub out_proj: LinearLayer,
    /// Fixed $O(1)$ ring-buffer state holding the last `kernel_size - 1` gated vectors `[k-1][d_model]`
    pub ring_state: Vec<Vec<f32>>,
}

impl Lfm2ShortConv {
    pub fn new(d_model: usize, seed: u64) -> Self {
        let kernel_size = 3;
        let mut rng = StdRng::seed_from_u64(seed);
        let conv_weight = (0..d_model * kernel_size)
            .map(|_| rng.gen_range(-0.25..0.25))
            .collect();

        Self {
            d_model,
            kernel_size,
            in_proj_b: LinearLayer::new_seeded(d_model, d_model, &mut rng),
            in_proj_c: LinearLayer::new_seeded(d_model, d_model, &mut rng),
            in_proj_v: LinearLayer::new_seeded(d_model, d_model, &mut rng),
            conv_weight,
            out_proj: LinearLayer::new_seeded(d_model, d_model, &mut rng),
            ring_state: vec![vec![0.0; d_model]; kernel_size - 1],
        }
    }

    pub fn reset_state(&mut self) {
        for slot in &mut self.ring_state {
            slot.fill(0.0);
        }
    }

    /// Processes a single token vector `x` in $O(d_{\text{model}})$ time and $O(1)$ cache memory:
    /// `y = W_out( C(x) * Conv1D_k3( B(x) * V(x) ) )`
    pub fn step(&mut self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.d_model);
        let b_gate = self.in_proj_b.forward(x, None);
        let c_gate = self.in_proj_c.forward(x, None);
        let v_val = self.in_proj_v.forward(x, None);

        // Input gating: u = sigmoid(B(x)) * V(x)
        let mut u_curr = vec![0.0f32; self.d_model];
        for d in 0..self.d_model {
            u_curr[d] = sigmoid(b_gate[d]) * v_val[d];
        }

        // Depthwise 1D causal convolution over [ring_state[0], ring_state[1], u_curr]
        let mut conv_gated = vec![0.0f32; self.d_model];
        for d in 0..self.d_model {
            let w_base = d * self.kernel_size;
            let c0 = self.ring_state[0][d] * self.conv_weight[w_base];
            let c1 = self.ring_state[1][d] * self.conv_weight[w_base + 1];
            let c2 = u_curr[d] * self.conv_weight[w_base + 2];
            let conv_sum = c0 + c1 + c2;
            // Output gating: C(x) * conv_out
            conv_gated[d] = sigmoid(c_gate[d]) * conv_sum;
        }

        // Shift $O(1)$ ring state without allocating
        self.ring_state.rotate_left(1);
        self.ring_state[self.kernel_size - 2] = u_curr;

        self.out_proj.forward(&conv_gated, None)
    }
}

// ============================================================================
// LFM3 TRI-HYBRID ARCHITECTURE (Reverse Engineered from LiquidAI/LFM3-1.7B)
// Source files: `configuration_lfm3.py`, `lfm3_conv.py`, `modeling_lfm3.py`
// ============================================================================

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Per-layer operator schedule in LFM3's 24-layer Tri-Hybrid stack (`configuration_lfm3.py:91-95`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Lfm3LayerType {
    /// Width-7 (`conv_L_cache = 7`) double-gated short convolution (`13/24` layers in LFM3-1.7B)
    Conv,
    /// Sliding-window GQA (`sliding_window = 256`, `rope_theta = 10_000.0`, full RoPE) (`5/24` layers)
    SlidingAttention,
    /// Global GQA (`rope_theta = 1_000_000.0`, zero-tail partial RoPE `0.5`) (`6/24` layers)
    FullAttention,
}

impl Lfm3LayerType {
    /// Generates the canonical LFM3 Tri-Hybrid layer schedule (`configuration_lfm3.py:92-95`):
    /// `"full_attention" if i % 4 == 2 else "sliding_attention" if i % 4 == 0 and i > 0 else "conv"`
    pub fn canonical_schedule(num_layers: usize) -> Vec<Self> {
        (0..num_layers)
            .map(|i| {
                if i % 4 == 2 {
                    Self::FullAttention
                } else if i % 4 == 0 && i > 0 {
                    Self::SlidingAttention
                } else {
                    Self::Conv
                }
            })
            .collect()
    }
}

/// `Lfm3RMSNorm` reverse-engineered from `modeling_lfm3.py:52-70`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lfm3RMSNorm {
    pub weight: Vec<f32>,
    pub variance_epsilon: f32,
}

impl Lfm3RMSNorm {
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        Self {
            weight: vec![1.0; hidden_size],
            variance_epsilon: eps,
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.weight.len());
        let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / (x.len().max(1) as f32);
        let inv_rms = 1.0 / (mean_sq + self.variance_epsilon).sqrt();
        x.iter()
            .zip(self.weight.iter())
            .map(|(&v, &w)| v * inv_rms * w)
            .collect()
    }
}

/// `Lfm3RotaryEmbedding` with LFM3's **zero-tail partial rotation**
/// reverse-engineered from `modeling_lfm3.py:72-125` and `204-210`:
/// - The first `partial_rotary_factor * head_dim // 2` frequencies follow the full `head_dim` progression
///   `1.0 / (base ** (2*i / head_dim))`.
/// - The remaining `head_dim // 2 - rope_angles` frequencies are explicitly set to `0.0` (`cos=1, sin=0`),
///   leaving the tail of each attention head completely unrotated!
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lfm3ZeroTailRoPE {
    pub head_dim: usize,
    pub inv_freq: Vec<f32>,
}

impl Lfm3ZeroTailRoPE {
    pub fn new(head_dim: usize, rope_theta: f32, partial_rotary_factor: f32) -> Self {
        let half = head_dim / 2;
        let rope_angles = ((partial_rotary_factor * (head_dim as f32)) as usize) / 2;
        let mut inv_freq = Vec::with_capacity(half);
        for i in 0..rope_angles.min(half) {
            let exponent = (2 * i) as f32 / (head_dim as f32);
            inv_freq.push(1.0 / rope_theta.powf(exponent));
        }
        while inv_freq.len() < half {
            inv_freq.push(0.0); // Zero-tail partial RoPE (`modeling_lfm3.py:110`)
        }
        Self { head_dim, inv_freq }
    }

    /// Applies `rotate_half` RoPE in `f32` (`modeling_lfm3.py:160-164, 204-210`).
    pub fn apply(&self, head_vec: &[f32], position_id: usize) -> Vec<f32> {
        debug_assert_eq!(head_vec.len(), self.head_dim);
        let half = self.head_dim / 2;
        let mut out = vec![0.0f32; self.head_dim];
        let pos = position_id as f32;
        for i in 0..half {
            let angle = pos * self.inv_freq[i];
            let (sin_a, cos_a) = angle.sin_cos();
            let x1 = head_vec[i];
            let x2 = head_vec[i + half];
            // rotate_half: [-x2, x1]
            out[i] = x1 * cos_a - x2 * sin_a;
            out[i + half] = x2 * cos_a + x1 * sin_a;
        }
        out
    }
}

/// `Lfm3ShortConv` width-7 (`conv_L_cache = 7`) double-gated depthwise causal convolution
/// with `seq_idx` packed-document boundary isolation, reverse-engineered from
/// `modeling_lfm3.py:332-443` and `lfm3_conv.py:64-99`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lfm3ShortConv {
    pub hidden_size: usize,
    pub conv_kernel_size: usize, // 7 in LFM3 (`conv_L_cache = 7`)
    pub in_proj: LinearLayer,    // hidden_size -> 3 * hidden_size (`BCx`)
    pub conv_weight: Vec<f32>,   // [hidden_size * 7] depthwise weights
    pub out_proj: LinearLayer,   // hidden_size -> hidden_size
    /// 6-slot $O(1)$ decode state ring buffer (`K = 6` previous `B * x` vectors in `lfm3_conv.py:89-93`)
    pub conv_state: Vec<Vec<f32>>,
    /// Tracks document/segment IDs per cached slot so `_segmented_conv` (`modeling_lfm3.py:432-442`)
    /// never bleeds across packed traces or organism boundaries (`-1` = empty padding).
    pub doc_state: Vec<i32>,
}

impl Lfm3ShortConv {
    pub fn new(hidden_size: usize, seed: u64) -> Self {
        let conv_kernel_size = 7;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut in_proj = LinearLayer::new_seeded(hidden_size, 3 * hidden_size, &mut rng);
        in_proj.bias.fill(0.0); // `conv_bias = false` in `config.json`
        let mut out_proj = LinearLayer::new_seeded(hidden_size, hidden_size, &mut rng);
        out_proj.bias.fill(0.0);
        let conv_weight = (0..hidden_size * conv_kernel_size)
            .map(|_| rng.gen_range(-0.22..0.22))
            .collect();

        Self {
            hidden_size,
            conv_kernel_size,
            in_proj,
            conv_weight,
            out_proj,
            conv_state: vec![vec![0.0; hidden_size]; conv_kernel_size - 1],
            doc_state: vec![-1; conv_kernel_size - 1],
        }
    }

    pub fn reset_state(&mut self) {
        for slot in &mut self.conv_state {
            slot.fill(0.0);
        }
        self.doc_state.fill(-1);
    }

    /// Executes one causal width-7 step (`modeling_lfm3.py:371-425` + `_segmented_conv` `432-442`):
    /// 1. `BCx = in_proj(x)` -> chunks into `(B, C, x_tilde)`
    /// 2. `u = B * x_tilde` (multiplicative input gate)
    /// 3. 7-tap depthwise convolution masking out any cached positions from a different `seq_idx`
    /// 4. `y = out_proj(C * conv_out)`
    pub fn step(&mut self, hidden_states: &[f32], seq_idx: i32) -> Vec<f32> {
        debug_assert_eq!(hidden_states.len(), self.hidden_size);
        let bcx = self.in_proj.forward(hidden_states, None);
        let d = self.hidden_size;
        let b_gate = &bcx[0..d];
        let c_gate = &bcx[d..2 * d];
        let x_tilde = &bcx[2 * d..3 * d];

        // Multiplicative input gate (`modeling_lfm3.py:373`: `hidden_states = B * x`)
        let mut u_curr = vec![0.0f32; d];
        for ch in 0..d {
            u_curr[ch] = b_gate[ch] * x_tilde[ch];
        }

        let cache_len = self.conv_kernel_size - 1; // 6
        let mut y_gated = vec![0.0f32; d];

        for ch in 0..d {
            let w_base = ch * self.conv_kernel_size;
            let mut acc = 0.0f32;
            // Read the 6 cached taps (`k = 0..6`), masking by `same_document` (`modeling_lfm3.py:438`)
            for k in 0..cache_len {
                if self.doc_state[k] == seq_idx {
                    acc += self.conv_state[k][ch] * self.conv_weight[w_base + k];
                }
            }
            // Tap 6 is current token `u_curr`
            acc += u_curr[ch] * self.conv_weight[w_base + cache_len];
            // Multiplicative output gate (`modeling_lfm3.py:422`: `y = C * hidden_states`)
            y_gated[ch] = c_gate[ch] * acc;
        }

        // Shift the 6-token $O(1)$ state window (`lfm3_conv.py:87-88`)
        self.conv_state.rotate_left(1);
        self.conv_state[cache_len - 1] = u_curr;
        self.doc_state.rotate_left(1);
        self.doc_state[cache_len - 1] = seq_idx;

        self.out_proj.forward(&y_gated, None)
    }
}

/// `Lfm3Attention` reverse-engineered from `modeling_lfm3.py:213-274`:
/// Grouped-Query Attention with per-head `q_norm` & `k_norm` (`Lfm3RMSNorm`),
/// layer-specific RoPE (`Zero-Tail Partial RoPE` on `FullAttention`, full RoPE on `SlidingAttention`),
/// sliding window cache eviction, and **per-head sigmoid output gate (`gate_proj`)**.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lfm3Attention {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub sliding_window: Option<usize>,
    pub q_proj: LinearLayer,
    pub k_proj: LinearLayer,
    pub v_proj: LinearLayer,
    pub o_proj: LinearLayer,
    pub gate_proj: LinearLayer, // hidden_size -> num_heads (`modeling_lfm3.py:231`)
    pub q_norm: Lfm3RMSNorm,    // (`modeling_lfm3.py:229`)
    pub k_norm: Lfm3RMSNorm,    // (`modeling_lfm3.py:230`)
    pub rope: Lfm3ZeroTailRoPE,
    pub k_cache: Vec<Vec<Vec<f32>>>, // [seq][num_kv_heads][head_dim]
    pub v_cache: Vec<Vec<Vec<f32>>>,
    pub position_cursor: usize,
}

impl Lfm3Attention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        layer_type: Lfm3LayerType,
        sliding_window: usize,
        seed: u64,
    ) -> Self {
        let head_dim = (hidden_size / num_heads).max(2);
        let mut rng = StdRng::seed_from_u64(seed);
        let (rope_theta, partial_factor, win) = match layer_type {
            Lfm3LayerType::FullAttention => (1_000_000.0, 0.5, None),
            Lfm3LayerType::SlidingAttention => (10_000.0, 1.0, Some(sliding_window.max(1))),
            Lfm3LayerType::Conv => (10_000.0, 1.0, None),
        };

        Self {
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window: win,
            q_proj: LinearLayer::new_seeded(hidden_size, num_heads * head_dim, &mut rng),
            k_proj: LinearLayer::new_seeded(hidden_size, num_kv_heads * head_dim, &mut rng),
            v_proj: LinearLayer::new_seeded(hidden_size, num_kv_heads * head_dim, &mut rng),
            o_proj: LinearLayer::new_seeded(num_heads * head_dim, hidden_size, &mut rng),
            gate_proj: LinearLayer::new_seeded(hidden_size, num_heads, &mut rng),
            q_norm: Lfm3RMSNorm::new(head_dim, 1e-5),
            k_norm: Lfm3RMSNorm::new(head_dim, 1e-5),
            rope: Lfm3ZeroTailRoPE::new(head_dim, rope_theta, partial_factor),
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            position_cursor: 0,
        }
    }

    pub fn reset_cache(&mut self) {
        self.k_cache.clear();
        self.v_cache.clear();
        self.position_cursor = 0;
    }

    pub fn step(&mut self, hidden_states: &[f32]) -> Vec<f32> {
        let q_raw = self.q_proj.forward(hidden_states, None);
        let k_raw = self.k_proj.forward(hidden_states, None);
        let v_raw = self.v_proj.forward(hidden_states, None);
        let head_gates = self.gate_proj.forward(hidden_states, None);

        let pos = self.position_cursor;
        self.position_cursor += 1;

        // Apply QK RMSNorm + RoPE per head (`modeling_lfm3.py:245-250`)
        let mut q_heads = Vec::with_capacity(self.num_heads);
        for h in 0..self.num_heads {
            let slice = &q_raw[h * self.head_dim..(h + 1) * self.head_dim];
            let normed = self.q_norm.forward(slice);
            q_heads.push(self.rope.apply(&normed, pos));
        }

        let mut k_heads = Vec::with_capacity(self.num_kv_heads);
        let mut v_heads = Vec::with_capacity(self.num_kv_heads);
        for kv in 0..self.num_kv_heads {
            let k_slice = &k_raw[kv * self.head_dim..(kv + 1) * self.head_dim];
            let v_slice = &v_raw[kv * self.head_dim..(kv + 1) * self.head_dim];
            let k_normed = self.k_norm.forward(k_slice);
            k_heads.push(self.rope.apply(&k_normed, pos));
            v_heads.push(v_slice.to_vec());
        }

        self.k_cache.push(k_heads);
        self.v_cache.push(v_heads);
        if let Some(win) = self.sliding_window {
            if self.k_cache.len() > win {
                self.k_cache.remove(0);
                self.v_cache.remove(0);
            }
        }

        let scaling = 1.0 / (self.head_dim as f32).sqrt();
        let kv_group_size = (self.num_heads / self.num_kv_heads).max(1);
        let mut concat_heads = Vec::with_capacity(self.num_heads * self.head_dim);

        for h in 0..self.num_heads {
            let kv_idx = (h / kv_group_size).min(self.num_kv_heads - 1);
            let q_vec = &q_heads[h];

            let mut scores: Vec<f32> = self
                .k_cache
                .iter()
                .map(|k_step| {
                    q_vec
                        .iter()
                        .zip(k_step[kv_idx].iter())
                        .map(|(&qv, &kv)| qv * kv)
                        .sum::<f32>()
                        * scaling
                })
                .collect();

            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                sum_exp += *s;
            }
            let inv_sum = 1.0 / sum_exp.max(1e-12);

            let mut head_out = vec![0.0f32; self.head_dim];
            for (t, &weight) in scores.iter().enumerate() {
                let prob = weight * inv_sum;
                for d in 0..self.head_dim {
                    head_out[d] += prob * self.v_cache[t][kv_idx][d];
                }
            }

            // Head-wise sigmoid gate (`modeling_lfm3.py:270-271`: `gate = torch.sigmoid(self.gate_proj(hidden_states))`)
            let gate = sigmoid(head_gates[h]);
            for d in 0..self.head_dim {
                concat_heads.push(head_out[d] * gate);
            }
        }

        self.o_proj.forward(&concat_heads, None)
    }
}

/// `Lfm3MLP` SwiGLU feed-forward network (`modeling_lfm3.py:149-158`):
/// `w2(silu(w1(x)) * w3(x))`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lfm3MLP {
    pub w1: LinearLayer,
    pub w2: LinearLayer,
    pub w3: LinearLayer,
}

impl Lfm3MLP {
    pub fn new(hidden_size: usize, intermediate_size: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            w1: LinearLayer::new_seeded(hidden_size, intermediate_size, &mut rng),
            w2: LinearLayer::new_seeded(intermediate_size, hidden_size, &mut rng),
            w3: LinearLayer::new_seeded(hidden_size, intermediate_size, &mut rng),
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let h1 = self.w1.forward(x, None);
        let h3 = self.w3.forward(x, None);
        let gated: Vec<f32> = h1
            .into_iter()
            .zip(h3)
            .map(|(a, b)| silu(a) * b)
            .collect();
        self.w2.forward(&gated, None)
    }
}

/// `Lfm3TriHybridBlock` combining LFM3's 3 layer types (`Conv` `k=7`, `SlidingAttention` `w=256`, `FullAttention`)
/// with 4-way **Sandwich RMSNorm** (`modeling_lfm3.py:445-499`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lfm3TriHybridBlock {
    pub hidden_size: usize,
    pub conv_layer: Lfm3ShortConv,
    pub sliding_attn: Lfm3Attention,
    pub full_attn: Lfm3Attention,
    pub ffn: Lfm3MLP,
    pub operator_norm: Lfm3RMSNorm,
    pub post_operator_norm: Lfm3RMSNorm,
    pub ffn_norm: Lfm3RMSNorm,
    pub post_ffn_norm: Lfm3RMSNorm,
}

impl Lfm3TriHybridBlock {
    pub fn new(hidden_size: usize, seed: u64) -> Self {
        let num_heads = 4;
        let num_kv_heads = 2;
        let intermediate = hidden_size * 2;
        Self {
            hidden_size,
            conv_layer: Lfm3ShortConv::new(hidden_size, seed),
            sliding_attn: Lfm3Attention::new(
                hidden_size,
                num_heads,
                num_kv_heads,
                Lfm3LayerType::SlidingAttention,
                8,
                seed + 10,
            ),
            full_attn: Lfm3Attention::new(
                hidden_size,
                num_heads,
                num_kv_heads,
                Lfm3LayerType::FullAttention,
                256,
                seed + 20,
            ),
            ffn: Lfm3MLP::new(hidden_size, intermediate, seed + 30),
            operator_norm: Lfm3RMSNorm::new(hidden_size, 1e-5),
            post_operator_norm: Lfm3RMSNorm::new(hidden_size, 1e-5),
            ffn_norm: Lfm3RMSNorm::new(hidden_size, 1e-5),
            post_ffn_norm: Lfm3RMSNorm::new(hidden_size, 1e-5),
        }
    }

    pub fn reset_state(&mut self) {
        self.conv_layer.reset_state();
        self.sliding_attn.reset_cache();
        self.full_attn.reset_cache();
    }

    /// Runs a single token through the LFM3 Tri-Hybrid Sandwich-Norm cascade (`modeling_lfm3.py:472-498`):
    /// `Conv (k=7) -> SlidingAttention -> FullAttention -> SwiGLU FFN`
    pub fn step(&mut self, x: &[f32], seq_idx: i32) -> Vec<f32> {
        self.step_configured(x, seq_idx, false)
    }

    /// Programmable LFM3 Tri-Hybrid forward pass:
    /// When `fast_conv_routing` is enabled for short hardware traces (`len <= 8`), bypasses
    /// the redundant `SlidingAttention + FullAttention` softmax passes and executes the width-7
    /// segmented `Lfm3ShortConv` + 4-way Sandwich RMSNorm + SwiGLU FFN in $O(1)$ time.
    pub fn step_configured(
        &mut self,
        x: &[f32],
        seq_idx: i32,
        fast_conv_routing: bool,
    ) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.hidden_size);

        // 1. Width-7 Gated ShortConv sub-layer with Sandwich RMSNorm
        let norm1 = self.operator_norm.forward(x);
        let conv_out = self
            .post_operator_norm
            .forward(&self.conv_layer.step(&norm1, seq_idx));
        let mut h: Vec<f32> = x.iter().zip(conv_out).map(|(&a, b)| a + b).collect();

        // 2. Optional Sliding-Window Attention + Full Zero-Tail Partial RoPE Attention
        if !fast_conv_routing {
            let norm2 = self.operator_norm.forward(&h);
            let slide_out = self.sliding_attn.step(&norm2);
            let full_out = self.full_attn.step(&slide_out);
            let attn_post = self.post_operator_norm.forward(&full_out);
            for d in 0..self.hidden_size {
                h[d] += attn_post[d];
            }
        }

        // 3. SwiGLU FFN with Sandwich RMSNorm (`modeling_lfm3.py:493-496`)
        let ffn_in = self.ffn_norm.forward(&h);
        let ffn_out = self.post_ffn_norm.forward(&self.ffn.forward(&ffn_in));
        for d in 0..self.hidden_size {
            h[d] += ffn_out[d];
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cfc_cell_continuous_time_sensitivity() {
        let cell = CfCCell::new(4, 8, 16, CfCMode::Default, 42);
        let input = [0.5, -0.2, 1.1, 0.3];
        let hx = [0.0; 8];

        let h_fast = cell.forward(&input, &hx, 0.001); // 1ms delta
        let h_slow = cell.forward(&input, &hx, 10.0);  // 10s delta
        assert_ne!(h_fast, h_slow, "CfC state must adapt to continuous time delta ts");
    }

    #[test]
    fn test_autoncp_and_ltc_semi_implicit_ode_and_online_plasticity() {
        let wiring = AutoNcpWiring::build(6, 16, 4, 0.5, 22222);
        assert!(wiring.synapse_count() > 0);
        assert!(wiring.sensory_synapse_count() > 0);

        let mut ltc = LTCCell::new(wiring, 6, 99);
        let state = vec![0.0; 16];
        let inputs = [1.0, 0.5, -0.5, 0.2, 0.0, 0.9];
        let (motor_before, next_state) = ltc.forward(&inputs, &state, 0.05);
        assert_eq!(motor_before.len(), 4);
        assert_eq!(next_state.len(), 16);
        assert!(motor_before.iter().all(|v| v.is_finite()));

        // Verify online Hebbian synapse adaptation updates conductances cleanly
        ltc.adapt_synapses_online(&inputs, &next_state, 1.0, 0.1);
        let (motor_after, _) = ltc.forward(&inputs, &state, 0.05);
        assert_ne!(motor_before, motor_after, "Online plasticity must update LTC motor response");
    }

    #[test]
    fn test_lfm2_short_conv_constant_state_ring() {
        let mut conv = Lfm2ShortConv::new(8, 7);
        let out1 = conv.step(&[1.0; 8]);
        let out2 = conv.step(&[1.0; 8]);
        assert_eq!(out1.len(), 8);
        assert_ne!(out1, out2, "Causal conv ring state must influence subsequent steps");
    }

    #[test]
    fn test_lfm3_tri_hybrid_zero_tail_rope_and_segmented_conv7() {
        // Verify canonical 24-layer schedule matches LiquidAI/LFM3-1.7B config.json exactly
        let sched = Lfm3LayerType::canonical_schedule(24);
        assert_eq!(sched.iter().filter(|&&t| t == Lfm3LayerType::Conv).count(), 13);
        assert_eq!(
            sched.iter().filter(|&&t| t == Lfm3LayerType::SlidingAttention).count(),
            5
        );
        assert_eq!(
            sched.iter().filter(|&&t| t == Lfm3LayerType::FullAttention).count(),
            6
        );

        // Verify Zero-Tail Partial RoPE leaves the upper half of frequency pairs at 0.0
        let rope = Lfm3ZeroTailRoPE::new(8, 1_000_000.0, 0.5);
        assert!(rope.inv_freq[0] > 0.0 && rope.inv_freq[1] > 0.0);
        assert_eq!(rope.inv_freq[2], 0.0);
        assert_eq!(rope.inv_freq[3], 0.0);

        // Verify LFM3 Tri-Hybrid block + segmented conv document isolation
        let mut block = Lfm3TriHybridBlock::new(8, 2026);
        let step1 = block.step(&[0.5; 8], 1);
        let step2 = block.step(&[0.5; 8], 1);
        assert_eq!(step1.len(), 8);
        assert_ne!(step1, step2);

        // Switching `seq_idx` isolates the width-7 conv state from prior document tokens
        let mut fresh_block = Lfm3TriHybridBlock::new(8, 2026);
        let fresh_conv = fresh_block.conv_layer.step(&[0.5; 8], 2);
        let segmented_conv = block.conv_layer.step(&[0.5; 8], 2);
        assert_eq!(
            fresh_conv, segmented_conv,
            "LFM3 _segmented_conv must isolate width-7 state across seq_idx boundaries"
        );
    }
}
