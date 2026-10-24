# Testing with Docker containers

One container per worker; faults are injected from outside with real tools.

## One-time setup
1. Docker Desktop running (WSL2 backend on Windows). Check `docker version` works in your terminal.
2. Build the image from the project folder (a few minutes the first time, it downloads torch):
   `swim-straggler build-image`
   The containers run the code of your project folder (mounted read-only), so code changes need no rebuild.
   Rebuild only when the Dockerfile or the dependencies change, or set `cluster.mount_source_code: false`.

## Runs
| What | Command | Expected |
|---|---|---|
| Worker 2's container gets half a processor at step 50 (in 1 ms slices, so every burst of computing takes ~2x longer) | `swim-straggler demo --config experiments/configs/docker_cpu.yaml` | measured effect x1.5 to x2.5 on its compute time; worker 2 then confirmed a few steps later |
| Noisy neighbour (2 x stress-ng) next to worker 2 | `swim-straggler demo --config experiments/configs/docker_stress.yaml` | bursty slowdown; read the "measured effect" line before judging detection |
| Slow link (tc netem, 10 ms per packet) from worker 2 | `swim-straggler demo --config experiments/configs/docker_network.yaml` | NOT confirmed: compute time is normal, peers vote no (link detection is milestone 10) |
| Frozen container for 2 s | `swim-straggler demo --config experiments/configs/docker_pause.yaml` | a stall, not a slowdown; look at the wait times in `events.jsonl` |

Use `fault.fault_tool: pumba` in the config to run the same fault with Pumba instead of `docker update` / `docker exec`.
Everything a run produced is in `runs/<name>_<time>/`: `events.jsonl` (merged), `compose.yaml`,
`config.yaml` (what the containers read), `container_logs/`.

## Check the "measured effect" line first
The summary prints how much longer the faulty worker really computed (median, during vs before the fault).
If it says `x1.0` with a WARNING, the fault did not bite and the run says nothing about the detector.
(Your first runs showed exactly this: a plain `docker update --cpus 0.5` is a quota per 100 ms, which a worker
computing 12 ms per 30 ms step never uses up. That is why the cpu fault now throttles in 1 ms slices.)

## If something fails
- The error message prints the last lines of the failing container's log; the full logs are in `container_logs/`.
- `docker ps -a` shows leftover containers; `docker rm -f $(docker ps -aq --filter name=swim_)` removes them.
- The step in `fault_started` (from `events_controller.jsonl`) is read from worker 0's log and is accurate to about
  1-2 steps. `step_after_command` is the step seen after the fault command returned.
- `docker update` / `docker exec` take 0.4 to 1.3 s to return on some setups, which is 10 to 45 steps. The summary
  therefore gives the detection delay both from the step where the command was issued and from the step where it returned.
