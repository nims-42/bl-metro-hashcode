# swim-lifeguard: decentralised straggler detection

Blue Line Metro (Nimalan K E, Selvaganesh Arunmozhi, Shashidharan VS), PES University. Hashcode 14, Track 7: Networks.

N workers train in lock-step over a ring all-reduce. A slow-but-alive worker (throttled GPU or congested link) slows every step without raising an error. Each worker watches its predecessor through the chunks the all-reduce already carries. A worker that keeps seeing lag from its predecessor runs Lifeguard's "am I the slow one?" check, asks k random peers to confirm, then spreads the verdict by SWIM gossip on the ring traffic. There is no central monitor. The only extra messages are the ping-reqs sent while a node is under suspicion.

The full design is in `CONTEXT.md` and the build status is in `TASKS.md`.

## Layout

| Path | What it does |
|---|---|
| `src/node.py` | Config (env overrides), TCP ring all-reduce, ping-req server, NumPy training stand-in, fault hooks, event loop |
| `src/swim.py` | Lateness rule on ring arrivals, ping-req to k peers, gossip buffer, suspicion → confirm |
| `src/lifeguard.py` | Local Health Multiplier, scaled timeouts, LHA-Suspicion timeout, self-check, Buddy System, compute/network classification |
| `src/membership.py` | `Alive` / `Suspect` / `Failed` / `Left` with SWIM incarnation precedence |
| `src/messages.py` | Piggyback header, gossip updates, `PING_REQ` / `ACK` / `NACK`, length-prefixed framing |
| `src/metrics.py` | One JSON line per node per step in `results/<run_id>/<node>.jsonl` |
| `experiments/run_experiments.py` | Scenarios, fault injection, ground truth, `summary.json` / `summary.csv` |

## How a straggler is found

- **Signal.** For every received chunk, a node records how long it was blocked on it. The sender's header carries `fwd_wait`, how long the sender itself was blocked just before forwarding that chunk. `pred_lag = Σ max(0, my_wait − fwd_wait)` is the lag that **originates at the predecessor**. Delay inherited from further up the ring cancels out, so only the successor of a slow GPU or a slow link sees a large `pred_lag`. No clocks are compared across nodes.
- **Lateness (FALCON).** The predecessor is late in a step if its `pred_lag` or its reported `compute_t` is above 1.1 × the ring median and above 10% of the step time. A suspicion needs N_consec late steps in a row (scaled by LHM + 1). Step time must also have risen by at least 10%; otherwise the streak is jitter.
- **Self-check (Lifeguard).** A node that the ring view itself implicates does not accuse anyone. It is implicated if its compute is slow, it waits least, or its own successor reports lag from it.
- **Confirm (SWIM).** `PING_REQ` goes to k = 2 random peers. Each answers from its own copy of the ring telemetry: agree, disagree, or `NACK` if it has no data. If a majority agrees, `SUSPECT` is gossiped and the Lifeguard suspicion timer starts, already shortened by the agreeing peers. If the suspected node's own view clears it, it refutes with a higher incarnation. Otherwise the timer ends in `CONFIRM` with the class. If peers disagree, the suspicion is dropped and the accuser's LHM goes up.
- **Recovery.** While suspected, a node re-runs its self-check every step. Once N_consec steps in a row no longer implicate it, it refutes with a higher incarnation, so a brief slowdown is cleared before the suspicion times out.
- **Class (FALCON).** If the suspect's `compute_t` is slow it is a **compute** straggler; otherwise it is a **network** straggler.

## Run with Docker (5 nodes)

Prerequisites: Docker Desktop (WSL 2) or Docker Engine 24+ with Compose v2, and Python 3.12 with `pip install -r requirements.txt`. A per-user Docker Desktop install puts `docker` in `%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin`; open a new terminal after installing so it is on `PATH`.

```bash
# healthy ring, logs in results/manual/
docker compose up --build

# one slow GPU: gpu2 sleeps an extra 100 ms per step from step 60
FAULT_NODE=2 FAULT_TYPE=compute FAULT_DELAY_MS=100 FAULT_START_STEP=60 RUN_ID=demo docker compose up --build
```

Scenarios with ground truth and scoring:

```bash
python experiments/run_experiments.py list                       # scenario definitions
python experiments/run_experiments.py run compute_fault --n 5    # one scenario
python experiments/run_experiments.py all --n 5                  # every scenario
python experiments/run_experiments.py overhead --n 5             # step time, detection on vs off
python experiments/run_experiments.py summarize <run_id>         # re-score an existing run
python experiments/run_experiments.py run compute_fault --set ALPHA=2 --set BETA=2   # tune
python experiments/run_experiments.py run no_fault --set gpu2.COMPUTE_MS=250          # one node only
```

`run_experiments.py` writes a compose file for any `--n` to `results/<run_id>/compose.generated.yaml`, so scaling up needs no edits to `compose.yaml`.

Scenarios: `no_fault`, `compute_fault`, `network_fault` (sleep on the send path), `jitter_only` (0–30 ms random extra compute on every node), `slow_accuser` (the straggler's successor is slow as well), `cpu_limit` (`docker update --cpus`, Docker only), and `per_node_compute` (every node has its own processing time from the start: 100, 110, 200, 105 and 90 ms at N = 5; only the 200 ms node should be flagged).

`--set KEY=VALUE` applies to every node; `--set gpu2.KEY=VALUE` applies to gpu2 only and wins.

### Per-node processing time and console output

Each node prints lines prefixed with its name: a startup line, a status line every `LOG_EVERY` steps (compute, wait, pred_lag, LHM, and any member that isn't Alive), and one line per detection event. Under Compose, each line also carries the container prefix (`gpu3-1 |`):

```text
gpu2-1  | [gpu2] ring up: pred gpu1, succ gpu3, compute 200 ms (work)
gpu3-1  | [gpu3] step   12 | SUSPECT_RAISED gpu2 (compute), asking gpu0, gpu4
gpu3-1  | [gpu3] step   13 | PING_REQ_RESULT gpu2: agree 2, disagree 0, nack 0, missing 0
gpu2-1  | [gpu2] step   13 | SUSPECTED_SELF gpu2 (compute) from gpu3
gpu3-1  | [gpu3] step   25 | compute  126.7 ms  wait  129.6 ms  pred_lag  126.8 ms  LHM 0 | gpu2 Suspect (compute)
gpu0-1  | [gpu0] step   41 | CONFIRM gpu2 (compute) from gpu3
```

To give nodes different processing times by hand, set `COMPUTE_MS` for all nodes and `GPU<i>_COMPUTE_MS` for one node:

```bash
GPU2_COMPUTE_MS=200 GPU1_COMPUTE_MS=110 GPU4_COMPUTE_MS=90 RUN_ID=hetero docker compose up --build
docker compose logs gpu3        # one node's console output only
```

Under Docker the measured `compute_t` runs about 20–35% above `COMPUTE_MS`. The CPU work is calibrated on an idle core, and N busy containers then run slower. The ratios between nodes hold. The JSONL logs record the measured value.

## Run without Docker

Add `--local` to start the same N nodes as processes on 127.0.0.1, each with its own ports. Local nodes use `COMPUTE_MODE=sleep`: CPU limits only exist under Docker, and N busy processes on one desktop add enough noise to cause false suspicions. `cpu_limit` is skipped.

```bash
python experiments/run_experiments.py all --n 5 --local
```

## Results

Each run writes these files to `results/<run_id>/`:

- `gpu<i>.jsonl`: one line per step with `compute_t`, `wait_t`, `pred_lag`, `step_t`, `lhm`, message counters, `member_states`, `events` (`SUSPECT_RAISED`, `PING_REQ_RESULT`, `SUSPECT`, `REFUTE`, `CONFIRM`, `JITTER_IGNORED`, `FAULT_START`, …) and `chunk_lags`. This is the data the deferred dashboard will read.
- `ground_truth.json`: the scenario, which node, which fault, and when.
- `summary.json` / `summary.csv`: per fault, whether it was detected, by whom, the class and whether it is correct, latency to suspect and to confirm (steps and seconds), and time to full dissemination. Also false positives (FP, and FP- for those seen by healthy observers), jitter streaks ignored, ping-reqs against ring frames, median step time before and during the fault, and max LHM.
- `results/history.csv`: one row per scored run (ring formation time, state and chunk sizes, bandwidth, healthy step time), used to estimate R below.

## Cost of a straggler

Computed after the run from the logs. Nothing acts on it: dropping, rebalancing and restarting are out of scope (CONTEXT.md §6.6).

```
pred_lag_succ(s) = time the whole ring lost to the straggler on step s, measured by its successor
slow_impact(k)   = Σ pred_lag_succ(s) over the k steps since the slowdown began   (job time lost)
GPU time lost    = N · slow_impact
act once slow_impact ≥ R                                                           (R = one-off cost of acting)
```

The rule is FALCON's ski-rental heuristic: acting once the waiting cost has reached R never costs more than 2× the best choice made in hindsight, whatever the slowdown's duration. If R is off from the truth by a factor x, the bound becomes `1 + max(x, 1/x)`.

**Per detection** in `summary.json` / `summary.csv`:

| Field | Meaning |
|---|---|
| `onset_step` | When the slowdown began: the start of the accuser's late streak, or the injected start if it was never suspected |
| `cost_measured_by` | The straggler's successor, whose `pred_lag` is summed |
| `cost_per_step_s` | Median per-step loss |
| `slow_impact_at_confirm_s`, `gpu_s_at_confirm` | What detection itself cost |
| `slow_impact_total_s`, `gpu_s_total` | What the straggler cost over the whole run |
| `action_cost_s`, `action_cost_source` | R and where it came from |
| `act_at_step` | First step where acting would have paid off (`null`: waiting was cheaper) |

**Choosing R:** pass `--action-cost 5` for a fixed R, or `--action-cost auto` to estimate it for "drop the node and re-form the ring":

```
R = T_ring    past ring formation time at the same mode and N (median; --r-percentile 90 to be cautious)
  + T_state   state bytes / this run's effective all-reduce bandwidth
  + T_warmup  warmup steps x this run's healthy step time
```

- `T_ring` comes from the `RING_UP` event each node logs at startup. Per run it takes the **smallest** value across nodes: the last node to start waits least, which is closest to re-forming a ring whose members are all running.
- With no history at that N, the nearest N is used; with no history at all, the run's own figures.
- Runs from before `RING_UP` was added are left out of history.
- The summary prints the breakdown, e.g. `R = 1.562 s = ring 0.2944 + state 0.0046 + warmup 1.263 (ring time: p50 of 1 previous docker run(s) at N=5)`.

```bash
python experiments/run_experiments.py run compute_fault --action-cost auto
python experiments/run_experiments.py summarize <run_id> --action-cost 5
```

**Assumptions:**
1. Training is synchronous.
2. There is one straggler, and its successor isn't also slow (in `slow_accuser`, the cost is understated).
3. Acting costs R once and fully removes the slowdown.

**What R doesn't cover:** rebalancing (FALCON S2, R ≈ 0) and checkpoint/restart (needs checkpoint size and disk speed). The bandwidth is "effective", dominated by latency with our small chunks.

## Tests

```bash
python -m pytest -q
```

The tests cover SWIM precedence, LHM and the LHA-Suspicion formula with an injectable clock, the lateness rule and jitter filter, ping-req peer choice, gossip limits, and ring all-reduce correctness for N ∈ {3, 5, 8, 10} with D not divisible by N.

## Main parameters (env var = default)

`N=5`, `K=2` (≤ N − 2), `N_CONSEC=3`, `LATENESS_FACTOR=1.1`, `JITTER_THRESHOLD=0.10`, `LHM_MAX=8`, `BASE_PROBE_TIMEOUT=0.5`, `PROBE_INTERVAL=1.0`, `ALPHA=5`, `BETA=6` (use 2, 2 for faster detection), `K_CONFIRM=3`, `GOSSIP_LAMBDA=3`, `GOSSIP_MAX=6`, `COMPUTE_MS=100`, `COMPUTE_MODE=work` (calibrated CPU work; `sleep` for local runs), `STEPS=300`, `LOG_EVERY=25` (console status line interval; 0 = events only). The full list is the `Config` dataclass in `src/node.py`.
