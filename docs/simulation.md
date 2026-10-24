# The simulator (optional extra)

**Everything the simulator produces is a simulation, not a measurement.** Nothing is trained and
nothing is timed. It exists to ask how the detectors behave with more workers than a laptop can run
(64, 256, 1024), under a stated model of the timings.

## What is real and what is made up
| real | made up |
|---|---|
| the detectors: the unchanged SWIM, central-monitor and step-time code, one per worker | compute times: `compute_ms` × a random factor (log-normal, spread `compute_noise`) |
| what each detector receives: its compute time, its wait in every ring round, the gossip of every ring message, in the same order as in the real ring | message times: `message_latency_ms` × a random factor (spread `latency_noise`) + `allreduce_data_ms` / N |
| the event log format: summaries, sweeps, the dashboard and trace.log all work on simulated runs | faults: compute slowdowns and link delays only (no container faults) |

One step in the model: each worker starts when its previous all-reduce ended, computes, then runs the
2(N-1) rounds of the ring. In every round it sends to its successor as soon as it is ready. The message
arrives after the latency plus its share of the work, and the receiver waits if the message is not there
yet. The gossip of every message is handed to the successor's detector, exactly as in the real ring.
The simulator skips turning gossip into numbers and back; a test checks that this changes no decision.

The default model was fitted to the three healthy "off" runs of the main sweep (4 gloo workers on a
2-processor machine): compute 11.7 ms, spread 0.33, latency 0.6 ms, all-reduce work 1.3 ms. To fit it
to your own machine:
```
swim-straggler simulate --workers 4 --fit-from runs/<a healthy measured run>
```
This prints the fitted values and uses them for this run; copy them into the `simulation` section of a config to keep them.

## Running it
```
swim-straggler simulate --workers 64                     # one run (experiments/configs/simulated.yaml)
swim-straggler simulate --workers 64 --trace             # with the reasoning behind every decision (trace.log)
swim-straggler sweep --spec experiments/sweeps/simulation.yaml          # 4 to 256 workers, every detector
swim-straggler sweep --spec experiments/sweeps/simulation_large.yaml    # 512 and 1024 workers (slow)
```
Any config with `cluster.backend: simulated` also runs with `swim-straggler train`.

The simulation is exactly reproducible: the same config and seed give the same events. That is why the
simulator sweeps delete each run's folder once its numbers are in `runs.jsonl` (`keep_run_folders: false`);
a 256-worker run logs about 50,000 step events.

Speed: the cost grows with N² (2(N-1) rounds per step, each passing N messages of gossip through Python).
A SWIM step takes about 1 ms to simulate at 4 workers, 35 ms at 64, 130 ms at 128, 0.6 s at 256 and
about 10 s at 1024. The baselines pass no gossip and are much faster.

## Limits: what could mislead
- **Independent noise.** Every worker's compute time is drawn independently every step. Real noise
  comes in bursts and is shared (another program on the machine slows several workers at once). The
  model can be too optimistic, or too pessimistic, about false alarms.
- **Model fitted to one small, noisy machine.** A spread of 0.33 is a lot: it comes from 4 workers
  sharing 2 processors. With more processors the real spread is likely smaller, and false alarms rarer.
  Fit the model to your own runs.
- **No real network.** No congestion, no bandwidth limit, no packet loss.
- **The check against measurement is at 4 workers only.** The simulator sweep runs 4 workers with the
  main sweep's settings and compares them with the measured main sweep (`compare_with`). Agreement there
  says the model is not far off at 4 workers. It says nothing direct about 1024 workers.
- **Detection load is computed, not measured.** The load table and plot count the numbers each worker
  sends for detection, from the design.
