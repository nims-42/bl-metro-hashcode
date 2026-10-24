# The step-by-step trace (trace.log)

Every run writes `trace.log` next to `events.jsonl`: what happened at every training step, in plain sentences.

```
swim-straggler train --config experiments/configs/fault_demo.yaml --trace
swim-straggler demo --workers 4 --trace
```

Without `--trace` the log holds only the decisions (suspected, confirmed, rejected, cleared, learned).
With `--trace` (or `detector.trace: true` in a config) every worker also logs the reasoning behind them.
Tracing does not change any decision; it only makes the event log longer, so it is meant for short runs.

## Reading it
```
step 57   step 51.4 ms   loss 0.085
  compute ms : w0   17.9  w1   19.0  w2   34.3  w3    7.1
  wait ms    : w0   27.9  w1   20.6  w2    8.1  w3   39.1
  w0: waits point to w2 (round 1 waited 20.96 ms, other rounds 1.86 ms): 5 of 5 steps
  w0: SUSPECTS w2, asks w3, w1 to confirm
  w0: answers about w2: SLOW (w2 computes 26.2 ms, typical 14.1 ms, limit x1.3)
  beliefs now: w0 thinks w2 suspect; w3 thinks w2 suspect
```
- The first lines of each step are the measurements: the step time, the loss, and each worker's compute and wait time.
- `FAULT` lines are the ground truth: when the injected fault became active. For Docker faults there are two lines:
  when the command was issued, and when it returned. Docker commands take up to about a second.
- `wN:` lines say what worker N did or concluded.
- `beliefs now` appears whenever someone's opinion changed: who thinks which worker is suspect or a straggler.
- The file ends with a summary: detection delay, false suspicions and confirmations, and what each worker believed at the end.

Lines are ordered by step, and within a step by worker. Worker clocks are never compared, so the order
between two workers inside one step says nothing about which came first.

## What each line means (SWIM detector, with --trace)
| line | meaning |
|---|---|
| waits point to wX (round r ...): n of k steps | this worker's longest wait was in round r, which points to wX; counted n times in a row, k needed to suspect |
| stops counting wX | the streak broke: the waits no longer point to wX, or wX is already suspect or confirmed |
| is slow itself ..., accuses nobody | the local health check: a slow worker does not trust its own waits |
| SUSPECTS wX, asks ... | the streak reached k: the suspicion starts and these peers are consulted |
| hears from wY that wX is suspected | a suspicion notice arrived by gossip |
| answers about wX: SLOW / not slow | this worker's opinion, from the compute times in its own table |
| hears wY's answer about wX | a consulted peer's answer arrived; the vote counts these |
| CONFIRMS / drops its suspicion | the majority of the consulted peers agreed / disagreed or did not answer in time |
| would suspect wX, but ... rejected | after a rejected suspicion a worker waits `recovery_steps` before suspecting again |
| learns wX is a straggler via gossip from wY | the verdict arrived from its ring predecessor wY |
| sees wX slow again ...; recovery count starts over | a confirmed straggler must look normal for `recovery_steps` steps in a row to be cleared |
| CLEARS wX | it did |

The central monitor traces its counting (`(central monitor) wX computes ... : n of k steps`), and the
step-time detector traces its own step time against the normal one it learned.

## Only a part
```
swim-straggler trace runs/<run folder>                    # the whole trace (also rebuilds it for older runs)
swim-straggler trace runs/<run folder> --worker 0         # only what worker 0 did
swim-straggler trace runs/<run folder> --about 2          # only what concerns worker 2
swim-straggler trace runs/<run folder> --steps 50-70      # only steps 50 to 70
swim-straggler trace runs/<run folder> --about 2 --output w2.log
```
The dashboard shows the same text, with the same filters, under "Trace" (below the detector events).
