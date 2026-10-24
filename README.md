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
```

`run_experiments.py` writes a compose file for any `--n` to `results/<run_id>/compose.generated.yaml`, so scaling up needs no edits to `compose.yaml`.

Scenarios: `no_fault`, `compute_fault`, `network_fault` (sleep on the send path), `jitter_only` (0–30 ms random extra compute on every node), `slow_accuser` (the straggler's successor is slow as well), and `cpu_limit` (`docker update --cpus`, Docker only).

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

## Tests

```bash
python -m pytest -q
```

The tests cover SWIM precedence, LHM and the LHA-Suspicion formula with an injectable clock, the lateness rule and jitter filter, ping-req peer choice, gossip limits, and ring all-reduce correctness for N ∈ {3, 5, 8, 10} with D not divisible by N.

## Main parameters (env var = default)

`N=5`, `K=2` (≤ N − 2), `N_CONSEC=3`, `LATENESS_FACTOR=1.1`, `JITTER_THRESHOLD=0.10`, `LHM_MAX=8`, `BASE_PROBE_TIMEOUT=0.5`, `PROBE_INTERVAL=1.0`, `ALPHA=5`, `BETA=6` (use 2, 2 for faster detection), `K_CONFIRM=3`, `GOSSIP_LAMBDA=3`, `GOSSIP_MAX=6`, `COMPUTE_MS=100`, `COMPUTE_MODE=work` (calibrated CPU work; `sleep` for local runs), `STEPS=300`. The full list is the `Config` dataclass in `src/node.py`.
