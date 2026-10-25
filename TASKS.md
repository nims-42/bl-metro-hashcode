# TASKS: swim-lifeguard

To-do list for the build described in `CONTEXT.md`. Section numbers (§) refer to that file.
Tick a box when the task is done and committed.

## 1. Scaffolding
- [x] Repo layout from §10 (`src/`, `tests/`, `experiments/`, `results/`)
- [x] `.gitignore` for run output and Python caches
- [x] `TASKS.md` (this file)

## 2. Protocol messages: `src/messages.py` (§5)
- [x] Piggyback header with bounded telemetry (at most N entries) and gossip (at most G entries)
- [x] Gossip updates: `SUSPECT` / `ALIVE` / `CONFIRM` / `LEAVE`
- [x] Side-channel messages: `PING_REQ` / `ACK` / `NACK`
- [x] Length-prefixed JSON frames over TCP, with the NumPy chunk as raw bytes after the header

## 3. Membership: `src/membership.py` (§6.4)
- [x] Member table with `Alive` / `Suspect` / `Failed` / `Left`, incarnation and class
- [x] SWIM precedence rules (ALIVE i > j, SUSPECT i > j / i ≥ j, CONFIRM overrides both)
- [x] Only the node itself can raise its own incarnation (refutation)
- [x] `Failed` keeps its class and the node keeps training; `LEAVE` → `Left`
- [x] `tests/test_membership.py`

## 4. Lifeguard: `src/lifeguard.py` (§6.2, §6.3, §6.5)
- [x] LHM saturating counter 0..S and its five events
- [x] Probe timeout / probe interval / suspicion threshold scaled by (LHM + 1)
- [x] LHA-Suspicion timeout (Min, Max, C, K) with an injectable clock
- [x] Self-check ("am I the slow one?") that blocks accusations
- [x] Buddy System gossip ordering
- [x] Classification: compute vs network
- [x] `tests/test_lifeguard.py`

## 5. Metrics: `src/metrics.py` (§3.2 L7, §9)
- [x] Per-node, per-step JSON line to `/results/<run_id>/<node_id>.jsonl`, flushed every step
- [x] Message counters, events, member states, LHM

## 6. SWIM layer: `src/swim.py` (§3.3, §6.1, §6.4)
- [x] Ring view of everyone's `compute_t` / `wait_t` / `pred_lag` from relayed telemetry
- [x] Lateness rule: 1.1 × ring median, 10% jitter filter, N_consec late steps in a row
- [x] Indirect probe: `PING_REQ` to k distinct peers (never the accuser or the suspect)
- [x] Answering `PING_REQ` from this node's own view (agree / disagree / NACK)
- [x] Gossip buffer: λ·log(n) retransmits, least-sent first, at most G per message
- [x] Suspicion timers → `CONFIRM`; refutation with a higher incarnation
- [x] `tests/test_swim.py`

## 7. Node: `src/node.py` (§3.1, §4, §7, §8)
- [x] `Config` dataclass with env overrides and validation (N ≥ 3, k ≤ N − 2)
- [x] TCP ring: connect to successor and accept from predecessor, retrying with backoff
- [x] Ring all-reduce (reduce-scatter + all-gather), padding when D is not divisible by N
- [x] `wait_t` and `pred_lag` measured with `time.monotonic()`
- [x] NumPy training stand-in that times `compute_t`
- [x] Fault hooks: sleep on the compute path or on the send path
- [x] TCP ping-req listener on its own port
- [x] `LEAVE` on clean shutdown

## 8. Containers (§3.1, §10)
- [x] `Dockerfile` (python:3.12-slim, `python -m src.node`)
- [x] `compose.yaml`: gpu0…gpu4 on `ring-net`, shared `x-node` anchor, `./results` → `/results`

## 9. Experiments: `experiments/run_experiments.py` (§3.4, §8, §9)
- [x] Scenarios: no fault, compute fault, network fault, jitter only, slow accuser
- [x] Generated per-run compose file for any N; `docker update --cpus` fault (verified under Docker Compose)
- [x] Ground truth to `results/<run_id>/ground_truth.json`
- [x] Summary: detection latency (steps, seconds, full dissemination), FP / FP-, classification accuracy, extra messages
- [x] Overhead: step time with detection on vs off
- [x] `summary.json` / `summary.csv`

## 10. Docs and verification
- [x] `README.md`: build, run a scenario, read the results
- [x] All tests pass at several values of N
- [x] End-to-end run of every scenario at N = 5 (`--local` processes; N = 8 checked too)
- [x] End-to-end run under Docker Compose, including `cpu_limit` (0 false positives across all six scenarios)
- [x] Training stand-in does real CPU work (`COMPUTE_MODE=work`) so `docker update --cpus` actually slows a node; `--local` uses `sleep`
- [x] A suspected node re-checks itself every step and refutes once it is back on time (fixes false positives from transient host noise)

## 11. Demo support
- [x] Console lines prefixed with the node name: startup, status every `LOG_EVERY` steps, detection events
- [x] Console output is best-effort (a closed stdout never stops a node)
- [x] Per-node processing time: `per_node_compute` scenario, `--set gpu<i>.KEY=VALUE`, `GPU<i>_COMPUTE_MS` in `compose.yaml`

## 12. Cost of a straggler (after the run)
- [x] `slow_impact` = running sum of the successor's `pred_lag` since onset; GPU time = N × that
- [x] `--action-cost R`: first step where acting pays off (FALCON ski-rental rule)
- [x] `RING_UP` event (connect and calibration time, state and chunk bytes, warmup steps) and `steps` on every JSONL line
- [x] `results/history.csv` and `--action-cost auto` (R = past ring formation + state transfer + warmup)
- [ ] Live version: carry the running cost in `CONFIRM` events (needed before any automatic mitigation)
- [ ] Measure R for real once a drop/rebalance exists, and replace the estimate with observed costs

## 13. Fixes from the WSL demo run
- [x] Lag explained by compute difference no longer counts (`residual_lag`): a fast node no longer accuses its normal predecessor
- [x] Only the critical-path node is accused (`critical`, by delay score: compute excess or unexplained successor lag): a node hidden behind a bigger straggler costs nothing. A first version used "waits least", which wrongly cleared slow-link stragglers under Docker noise.
- [x] Cost = how far the straggler finished behind the slowest *other* node; `step_time` method by default, `pred_lag` as fallback, both reported
- [x] `uneven_compute_fault` scenario reproduces the demo; `slow_accuser` now expects gpu3 to stay hidden
- [ ] Once mitigation exists: drop a removed node from the critical-path check so the next straggler surfaces
- [ ] Re-run the Docker sweep with the delay-score rule: the Docker `network_fault` was missed by the wait-based first version, and the sweep with the fix was stopped before Docker ran (verified locally and by `test_slow_link_straggler_does_not_refute_while_its_link_is_slow`)

## Open findings from the first runs
- [x] `slow_accuser`: the slow successor used to hold back its accusation and get accused itself, so the real straggler was caught late or never. Fixed by the critical-path rule: the hidden successor accuses the real straggler and is not accused itself.
- [ ] Detection takes ~8 s at N = 5 with α = 5, β = 6 (suspicion timeout). Try α = β = 2 and re-measure false positives under jitter.
- [ ] Under Docker, host noise briefly suspects healthy nodes (1 per 300-step healthy run, 7 under 30 ms jitter); all were refuted before the timeout. Watch this when scaling N.
- [ ] Under Docker, measured compute is ~20–35% above `COMPUTE_MS` (calibrated on an idle core). Fine for relative demos; recalibrate if absolute numbers matter.
- [ ] With auto R ≈ 1.1–1.6 s (N = 5), acting would pay off ~10–15 steps after onset, before the ~27–37-step confirmation. Detection latency, not R, now dominates the cost of a straggler.
- [ ] A confirmed straggler stays `Failed` after its fault ends (CONFIRM is final in SWIM). Decide on recovery once mitigation is in scope.
- [ ] Reconcile with the `nims/skeleton` branch, which uses a different layout (`swim_straggler/`).

## Later (out of scope for now, §0 rule 6)
- [ ] Dashboard: `wait_t` per GPU per step, detected node and class, time to detect, FP vs ground truth
- [ ] FALCON S2 mitigation hooked on `Failed` + `class`
- [ ] Scale to N = 8–10 (the deck's minimum) and re-tune α, β, N_consec
