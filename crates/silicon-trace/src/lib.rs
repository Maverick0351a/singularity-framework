//! # `silicon-trace`
//!
//! Clean-room Rust implementation of the **Hardware Branch Flight Recorder & Callstack Compressor**
//! reverse-engineered from `janestreet/magic-trace`:
//! - `src/event.ml` (`Event.Kind`, `Location`, `Trace`, `Decode_error`)
//! - `src/new_trace_writer.ml` (`Mapped_time`, `#155 hack` 1ns monotonic decode-error slide,
//!   hardware TSX `in_transaction` / `Tx_abort` deque buffering, duration span reconstruction)
//! - `src/callstack_compression.ml` (`follow_table` predictive next-symbol callstack compression)
//! - `src/breakpoint_stubs.c` (`magic_breakpoint_next_stub` ring-buffer `data_head`/`data_tail`
//!   snapshot trigger capturing `(timestamp, passed_timestamp, passed_val, tid, ip)`)

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Reads the CPU hardware Time Stamp Counter (`RDTSC` with `LFENCE` serialization on `x86_64`)
/// for cycle-accurate profiling inside `FlightRecorder`.
#[inline]
pub fn read_hardware_tsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_lfence();
        let tsc = core::arch::x86_64::_rdtsc();
        core::arch::x86_64::_mm_lfence();
        tsc
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    }
}

/// Branch/control-flow event classification (`src/event.ml:3-16`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventKind {
    Async,
    Call,
    Return,
    Syscall,
    Sysret,
    HardwareInterrupt,
    Interrupt,
    Iret,
    Jump,
    TxAbort,
}

/// Instruction location descriptor (`src/event.ml:26-34`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Location {
    pub instruction_pointer: u64,
    pub symbol: String,
    pub symbol_offset: u64,
}

impl Location {
    pub fn new(instruction_pointer: u64, symbol: impl Into<String>, symbol_offset: u64) -> Self {
        Self {
            instruction_pointer,
            symbol: symbol.into(),
            symbol_offset,
        }
    }
}

/// Hardware breakpoint hit metadata (`src/breakpoint_stubs.c:139-185` & `src/new_trace_writer.ml:168-206`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BreakpointHit {
    pub symbol: String,
    pub timestamp_ns: u64,
    pub passed_timestamp_ns: u64,
    pub passed_val: u64,
    pub tid: u32,
    pub ip: u64,
}

/// Trace event payload (`src/event.ml:75-116`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceEvent {
    Branch {
        tid: u32,
        time_ns: u64,
        kind: EventKind,
        src: Location,
        dst: Location,
        in_transaction: bool,
    },
    DecodeError {
        tid: u32,
        time_ns: Option<u64>,
        instruction_pointer: Option<u64>,
        message: String,
    },
}

impl TraceEvent {
    pub fn time_ns(&self) -> Option<u64> {
        match self {
            TraceEvent::Branch { time_ns, .. } => Some(*time_ns),
            TraceEvent::DecodeError { time_ns, .. } => *time_ns,
        }
    }

    pub fn tid(&self) -> u32 {
        match self {
            TraceEvent::Branch { tid, .. } | TraceEvent::DecodeError { tid, .. } => *tid,
        }
    }
}

/// Relative timestamp mapped against `base_time` (`src/new_trace_writer.ml:10-33`).
/// Prevents float precision loss in Perfetto trace viewers when boot-relative timestamps are large.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MappedTime(pub u64);

impl MappedTime {
    pub const START_OF_TRACE: Self = Self(0);

    pub fn from_raw(raw_ns: u64, base_time_ns: u64) -> Self {
        Self(raw_ns.saturating_sub(base_time_ns))
    }

    pub fn add_ns(self, delta_ns: u64) -> Self {
        Self(self.0.saturating_add(delta_ns))
    }
}

/// Reconstructed function execution span emitted by the trace writer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurationSpan {
    pub tid: u32,
    pub symbol: String,
    pub start_ns: u64,
    pub end_ns: u64,
    pub delta_ns: u64,
    pub instruction_pointer: u64,
}

/// Lock-free / bounded circular flight-recorder buffer (`perf_event_mmap_page` model in `breakpoint_stubs.c`
/// combined with the TSX transaction buffer and `#155` monotonic timestamp fix in `new_trace_writer.ml`).
#[derive(Debug, Clone)]
pub struct FlightRecorder {
    capacity: usize,
    ring: VecDeque<TraceEvent>,
    transaction_queue: VecDeque<TraceEvent>,
    hits: Vec<BreakpointHit>,
}

impl FlightRecorder {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(16),
            ring: VecDeque::with_capacity(capacity.min(4096)),
            transaction_queue: VecDeque::new(),
            hits: Vec::new(),
        }
    }

    /// Records a hardware branch or decode error into the circular ring buffer,
    /// implementing exact TSX transaction buffering (`src/new_trace_writer.ml:448-481`):
    /// - Events with `in_transaction = true` are queued in `transaction_queue`.
    /// - `EventKind::TxAbort` drops all queued transaction events.
    /// - Normal event flushes queued transaction events into the ring buffer.
    pub fn record(&mut self, event: TraceEvent) {
        match &event {
            TraceEvent::Branch {
                kind: EventKind::TxAbort,
                ..
            } => {
                self.transaction_queue.clear();
                self.push_to_ring(event);
            }
            TraceEvent::Branch {
                in_transaction: true,
                ..
            } => {
                self.transaction_queue.push_back(event);
            }
            TraceEvent::Branch {
                in_transaction: false,
                ..
            } => {
                while let Some(queued) = self.transaction_queue.pop_front() {
                    self.push_to_ring(queued);
                }
                self.push_to_ring(event);
            }
            TraceEvent::DecodeError { .. } => {
                self.transaction_queue.clear();
                self.push_to_ring(event);
            }
        }
    }

    pub fn record_breakpoint_hit(&mut self, hit: BreakpointHit) {
        self.hits.push(hit);
    }

    fn push_to_ring(&mut self, ev: TraceEvent) {
        if self.ring.len() >= self.capacity {
            self.ring.pop_front();
        }
        self.ring.push_back(ev);
    }

    /// Takes an atomic snapshot of the current ring buffer and reconstructs `DurationSpan`s
    /// using `MappedTime` and the `#155 hack` (`src/new_trace_writer.ml:264-300`).
    pub fn take_snapshot(&self) -> FlightSnapshot {
        let earliest_event_ns = self
            .ring
            .iter()
            .filter_map(|e| e.time_ns())
            .min()
            .unwrap_or(0);
        let base_time_ns = self
            .hits
            .iter()
            .map(|h| h.timestamp_ns)
            .fold(earliest_event_ns, u64::min);

        let mut last_event_time: HashMap<u32, MappedTime> = HashMap::new();
        let mut last_decode_err_time: HashMap<u32, MappedTime> = HashMap::new();
        let mut call_stacks: HashMap<u32, Vec<(String, u64, MappedTime)>> = HashMap::new();
        let mut spans = Vec::new();
        let mut branch_deltas_ns = Vec::new();
        let mut prev_mapped_ns: Option<u64> = None;

        for ev in &self.ring {
            let tid = ev.tid();
            let unadjusted = match ev.time_ns() {
                Some(raw_t) => {
                    let mt = MappedTime::from_raw(raw_t, base_time_ns);
                    last_event_time.insert(tid, mt);
                    mt
                }
                None => *last_event_time.get(&tid).unwrap_or(&MappedTime::START_OF_TRACE),
            };

            // `#155 hack` (`new_trace_writer.ml:278-284`):
            // If an event occurs at the exact same timestamp as the previous decode error,
            // slide it forward by 1ns so callstacks never invert.
            let last_err = *last_decode_err_time
                .get(&tid)
                .unwrap_or(&MappedTime::START_OF_TRACE);
            let mapped = if unadjusted == last_err && last_err != MappedTime::START_OF_TRACE {
                unadjusted.add_ns(1)
            } else {
                unadjusted
            };

            if let Some(prev) = prev_mapped_ns {
                branch_deltas_ns.push((ev.clone(), mapped.0.saturating_sub(prev)));
            } else {
                branch_deltas_ns.push((ev.clone(), 0));
            }
            prev_mapped_ns = Some(mapped.0);

            match ev {
                TraceEvent::DecodeError { .. } => {
                    last_decode_err_time.insert(tid, mapped);
                    if let Some(stack) = call_stacks.get_mut(&tid) {
                        while let Some((sym, ip, start)) = stack.pop() {
                            spans.push(DurationSpan {
                                tid,
                                symbol: sym,
                                start_ns: start.0,
                                end_ns: mapped.0,
                                delta_ns: mapped.0.saturating_sub(start.0),
                                instruction_pointer: ip,
                            });
                        }
                    }
                }
                TraceEvent::Branch {
                    kind, src, dst, ..
                } => match kind {
                    EventKind::Call | EventKind::Syscall | EventKind::Interrupt => {
                        call_stacks.entry(tid).or_default().push((
                            dst.symbol.clone(),
                            dst.instruction_pointer,
                            mapped,
                        ));
                    }
                    EventKind::Return | EventKind::Sysret | EventKind::Iret => {
                        if let Some(stack) = call_stacks.get_mut(&tid) {
                            if let Some((sym, ip, start)) = stack.pop() {
                                spans.push(DurationSpan {
                                    tid,
                                    symbol: sym,
                                    start_ns: start.0,
                                    end_ns: mapped.0,
                                    delta_ns: mapped.0.saturating_sub(start.0),
                                    instruction_pointer: ip,
                                });
                            } else {
                                spans.push(DurationSpan {
                                    tid,
                                    symbol: src.symbol.clone(),
                                    start_ns: mapped.0,
                                    end_ns: mapped.0,
                                    delta_ns: 0,
                                    instruction_pointer: src.instruction_pointer,
                                });
                            }
                        }
                    }
                    _ => {}
                },
            }
        }

        FlightSnapshot {
            base_time_ns,
            spans,
            branch_deltas_ns,
            hits: self.hits.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlightSnapshot {
    pub base_time_ns: u64,
    pub spans: Vec<DurationSpan>,
    pub branch_deltas_ns: Vec<(TraceEvent, u64)>,
    pub hits: Vec<BreakpointHit>,
}

/// Predictive Follow-Table Callstack Compressor (`src/callstack_compression.ml:1-86`).
/// Interns symbol strings into integer IDs and maintains a single-step `follow_table`
/// predicting `next_symbol` from `prev_symbol`, compressing arbitrary callstacks into
/// `(from_symbol_id, run_length_steps)` tuples.
#[derive(Debug, Default, Clone)]
pub struct CallstackCompressor {
    symbol_to_id: HashMap<String, usize>,
    id_to_symbol: HashMap<usize, String>,
    next_symbol_id: usize,
    follow_table: HashMap<usize, usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompressedCallstackEvent {
    pub new_symbols: Vec<(String, usize)>,
    pub callstack: Vec<(usize, usize)>,
}

impl CallstackCompressor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn compress_callstack(&mut self, callstack: &[String]) -> CompressedCallstackEvent {
        let mut new_symbols = Vec::new();
        let mut symbol_ids = Vec::with_capacity(callstack.len());

        for sym in callstack {
            let id = if let Some(&existing_id) = self.symbol_to_id.get(sym) {
                existing_id
            } else {
                let assigned = self.next_symbol_id;
                self.next_symbol_id += 1;
                self.symbol_to_id.insert(sym.clone(), assigned);
                self.id_to_symbol.insert(assigned, sym.clone());
                new_symbols.push((sym.clone(), assigned));
                assigned
            };
            symbol_ids.push(id);
        }

        let compressed = self.compress_with_follow_table(&symbol_ids);
        self.update_follow_table(&symbol_ids);

        CompressedCallstackEvent {
            new_symbols,
            callstack: compressed,
        }
    }

    pub fn decompress_callstack(&mut self, ev: &CompressedCallstackEvent) -> Vec<String> {
        for (sym, id) in &ev.new_symbols {
            self.id_to_symbol.insert(*id, sym.clone());
            self.symbol_to_id.insert(sym.clone(), *id);
        }
        let symbol_ids = self.decompress_with_follow_table(&ev.callstack);
        self.update_follow_table(&symbol_ids);
        symbol_ids
            .into_iter()
            .map(|id| self.id_to_symbol.get(&id).cloned().unwrap_or_default())
            .collect()
    }

    fn update_follow_table(&mut self, stack: &[usize]) {
        for pair in stack.windows(2) {
            self.follow_table.insert(pair[0], pair[1]);
        }
    }

    fn compress_with_follow_table(&self, stack: &[usize]) -> Vec<(usize, usize)> {
        let mut acc: Vec<(usize, usize, usize)> = Vec::new();
        for &x in stack {
            match acc.last_mut() {
                None => acc.push((x, x, 0)),
                Some((_from_s, to_s, steps)) => {
                    if self.follow_table.get(to_s) == Some(&x) {
                        *to_s = x;
                        *steps += 1;
                    } else {
                        acc.push((x, x, 0));
                    }
                }
            }
        }
        acc.into_iter().map(|(from_s, _, steps)| (from_s, steps)).collect()
    }

    fn decompress_with_follow_table(&self, comp_stack: &[(usize, usize)]) -> Vec<usize> {
        let mut out = Vec::new();
        for &(from_s, steps) in comp_stack {
            out.push(from_s);
            let mut cur = from_s;
            for _ in 0..steps {
                if let Some(&next) = self.follow_table.get(&cur) {
                    out.push(next);
                    cur = next;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_callstack_follow_table_compression_roundtrip() {
        let mut enc = CallstackCompressor::new();
        let mut dec = CallstackCompressor::new();

        let stack1 = vec![
            "main".to_string(),
            "parse_iso".to_string(),
            "read_sector".to_string(),
            "apply_override".to_string(),
        ];
        let ev1 = enc.compress_callstack(&stack1);
        assert_eq!(ev1.callstack.len(), 4); // First time: 4 independent entries
        assert_eq!(dec.decompress_callstack(&ev1), stack1);

        // Second identical stack compresses to a SINGLE `(from_id=0, steps=3)` tuple!
        let ev2 = enc.compress_callstack(&stack1);
        assert_eq!(ev2.new_symbols.len(), 0);
        assert_eq!(ev2.callstack, vec![(0, 3)]);
        assert_eq!(dec.decompress_callstack(&ev2), stack1);
    }

    #[test]
    fn test_flight_recorder_tsx_abort_and_155_monotonic_hack() {
        let mut rec = FlightRecorder::new(64);
        let main_loc = Location::new(0x1000, "main", 0);
        let work_loc = Location::new(0x2000, "hot_loop", 0);

        rec.record(TraceEvent::Branch {
            tid: 1,
            time_ns: 1_000,
            kind: EventKind::Call,
            src: main_loc.clone(),
            dst: work_loc.clone(),
            in_transaction: false,
        });

        // Queued inside TSX transaction, then aborted -> must be discarded!
        rec.record(TraceEvent::Branch {
            tid: 1,
            time_ns: 1_050,
            kind: EventKind::Call,
            src: work_loc.clone(),
            dst: Location::new(0x3000, "aborted_fn", 0),
            in_transaction: true,
        });
        rec.record(TraceEvent::Branch {
            tid: 1,
            time_ns: 1_060,
            kind: EventKind::TxAbort,
            src: work_loc.clone(),
            dst: work_loc.clone(),
            in_transaction: false,
        });

        // Decode error at t=1_200, followed by a Call at exact same t=1_200 (#155 hack slides it by +1ns)
        rec.record(TraceEvent::DecodeError {
            tid: 1,
            time_ns: Some(1_200),
            instruction_pointer: Some(0x2010),
            message: "pt_overflow".to_string(),
        });
        rec.record(TraceEvent::Branch {
            tid: 1,
            time_ns: 1_200,
            kind: EventKind::Call,
            src: main_loc.clone(),
            dst: work_loc.clone(),
            in_transaction: false,
        });
        rec.record(TraceEvent::Branch {
            tid: 1,
            time_ns: 1_500,
            kind: EventKind::Return,
            src: work_loc,
            dst: main_loc,
            in_transaction: false,
        });

        let snap = rec.take_snapshot();
        assert_eq!(snap.spans.len(), 2);
        // First span closed by DecodeError at mapped t=200ns
        assert_eq!(snap.spans[0].symbol, "hot_loop");
        assert_eq!(snap.spans[0].delta_ns, 200);
        // Second span started at t=201ns (+1ns #155 slide) and ended at t=500ns -> 299ns
        assert_eq!(snap.spans[1].delta_ns, 299);
    }
}
