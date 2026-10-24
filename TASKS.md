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
- [ ] Ring view of everyone's `compute_t` / `wait_t` / `pred_lag` from relayed telemetry
- [ ] Lateness rule: 1.1 × ring median, 10% jitter filter, N_consec late steps in a row
- [ ] Indirect probe: `PING_REQ` to k distinct peers (never the accuser or the suspect)
- [ ] Answering `PING_REQ` from this node's own view (agree / disagree / NACK)
- [ ] Gossip buffer: λ·log(n) retransmits, least-sent first, at most G per message
- [ ] Suspicion timers → `CONFIRM`; refutation with a higher incarnation
- [ ] `tests/test_swim.py`

## 7. Node: `src/node.py` (§3.1, §4, §7, §8)
- [ ] `Config` dataclass with env overrides and validation (N ≥ 3, k ≤ N − 2)
- [ ] TCP ring: connect to successor and accept from predecessor, retrying with backoff
- [ ] Ring all-reduce (reduce-scatter + all-gather), padding when D is not divisible by N
- [ ] `wait_t` and `pred_lag` measured with `time.monotonic()`
- [ ] NumPy training stand-in that times `compute_t`
- [ ] Fault hooks: sleep on the compute path or on the send path
- [ ] TCP ping-req listener on its own port
- [ ] `LEAVE` on clean shutdown

## 8. Containers (§3.1, §10)
- [ ] `Dockerfile` (python:3.12-slim, `python -m src.node`)
- [ ] `compose.yaml`: gpu0…gpu4 on `ring-net`, shared `x-node` anchor, `./results` → `/results`

## 9. Experiments: `experiments/run_experiments.py` (§3.4, §8, §9)
- [ ] Scenarios: no fault, compute fault, network fault, jitter only, slow accuser
- [ ] Generated per-run compose file for any N; `docker update --cpus` fault
- [ ] Ground truth to `results/<run_id>/ground_truth.json`
- [ ] Summary: detection latency (steps, seconds, full dissemination), FP / FP-, classification accuracy, extra messages
- [ ] Overhead: step time with detection on vs off
- [ ] `summary.json` / `summary.csv`

## 10. Docs and verification
- [ ] `README.md`: build, run a scenario, read the results
- [ ] All tests pass at several values of N
- [ ] End-to-end run of every scenario at N = 5

## Later (out of scope for now, §0 rule 6)
- [ ] Dashboard: `wait_t` per GPU per step, detected node and class, time to detect, FP vs ground truth
- [ ] FALCON S2 mitigation hooked on `Failed` + `class`
- [ ] Scale to N = 8–10 (the deck's minimum) and re-tune α, β, N_consec
