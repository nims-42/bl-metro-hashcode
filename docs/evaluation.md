# Evaluating the detectors

All numbers come from these commands; none are typed by hand.

## Quick comparison (one config, every detector)
```
swim-straggler demo --compare --workers 4               # one run per detector
swim-straggler demo --compare --workers 4 --repeats 5   # five runs each, medians and ranges
swim-straggler demo --compare --config experiments/configs/docker_cpu.yaml --repeats 3
```
Every repeat uses a different seed; every detector gets the same seeds, so they are compared on the same data.
For container faults the table also shows the delay counted from when the fault command returned
(Docker commands can take a second, which is many steps).

## Sweeps
```
swim-straggler sweep --spec experiments/sweeps/main.yaml      # processes, about 10-15 minutes
swim-straggler sweep --spec experiments/sweeps/docker.yaml    # containers, needs Docker and the image
```
A sweep file lists faults, detectors and the number of repeats; every combination is run.
Results go to `results/<sweep name>/`:

| file | what it is |
|---|---|
| `runs.jsonl` | one line per finished run with all its numbers (the source of everything else) |
| `summary.json` | one entry per (detector, fault) pair |
| `results.md` | the tables and plots, ready to read |
| `*.png` | the plots |
| `machine.json` | where it was measured (processors, platform, versions, date) |
| `sweep.yaml` | a copy of the sweep definition |

A sweep that is stopped (Ctrl+C, a crash, a closed laptop) continues where it stopped when started again:
runs already in `runs.jsonl` are skipped. To start from scratch, delete the results folder.

## Rebuilding the tables and plots
```
swim-straggler report --results results/main
```
This rebuilds `results.md` and the plots from `runs.jsonl` (a sweep already does this when it finishes).

## What the numbers mean
- **culprit found**: every faulty worker was confirmed slow at or after the step its fault started.
- **delay**: steps from the fault's start to the first confirmation by any worker.
- **everyone knew after**: steps from the fault's start until every worker held the verdict (SWIM spreads it by gossip;
  the central monitor alone decides, so this is not applicable there).
- **false confirmation**: a healthy worker (or the later culprit before its fault started) declared slow.
- **measured slowdown**: how much longer the faulty worker really computed during the fault (checks the fault worked).
- **cost**: mean step time on healthy runs against the detector switched off.

Timing depends on the machine. Results measured on a machine with few processors and many workers are noisy;
compare detectors within one sweep, not across machines.
