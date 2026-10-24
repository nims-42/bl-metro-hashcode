# Experimental features (milestone 10)

Everything here is **experimental**: it works in the runs below, on a small and noisy machine, and has
not been tested widely. None of it changes the main detector unless switched on.

## 1. Slow links: "the delay is not compute" (`detector.diagnose_network: true`)
**What a slow link looks like.** A delayed message holds up the whole ring, one hop after another. So a
slow link makes workers wait in a pattern much like a slow worker's. The waits then point at some worker,
and not always the sender of the slow link. The simulator shows the same, without noise.

**What the detector does with it.** A worker's waits point at someone for a full streak, so it suspects
that worker. The consulted peers check its compute time, see it is normal, and vote no. So **a slow
link never gets a healthy worker confirmed**; the vote protects against it. With `diagnose_network`
on, a worker whose suspicions are rejected `diagnosis_rejections` times (default 3) within
`diagnosis_window_steps` (default 60), each time with a normally computing subject, logs
`delay_not_compute`: "the delay is probably not compute, possibly the network". It appears in the run
summary, in trace.log, and in `network_diagnoses` of the summary.

**What it cannot do.** It does not say *which* link is slow: the waits do not point at the sender
reliably. It is also slow to fire, and noise can trigger it.

Measured (gloo, 4 workers on 2 processors, 160 steps, a 4 ms link delay on worker 2 from step 50):

| run | confirmed anybody | diagnosis |
|---|---|---|
| link, seed 0 | no | worker 0, step 145 (95 steps after the fault started) |
| link, seed 1 | no | none |
| healthy, seed 0 | 2 false confirmations (noise, not the diagnosis) | none |
| healthy, seed 1 | no | none |

With threshold 2 instead of 3, a healthy run also produced a diagnosis, which is why the default is 3.
Two runs per case is far too few to state a rate. Treat this as a demonstration that the pattern exists,
not as a working link detector.

```
swim-straggler train --config experiments/configs/link_demo.yaml --trace
swim-straggler train --config experiments/configs/docker_network.yaml        # with Docker: tc netem
```

## 2. The diagnosis label on every verdict
Every `confirmed` event now carries `diagnosis: compute` and `compute_ratio`: the subject's compute time
against the typical one, from the confirming worker's table. Every `rejected` event carries `compute_ratio`
too. A rejection with a ratio near 1 means "my waits pointed here, but this worker computes normally".
trace.log shows the ratio, for example "CONFIRMS w2 is a straggler (2 of 2 consulted agree, computes x2.04
the typical)".

## 3. Does the signal exist in real DDP?
The detector reads waits from our own ring. `experiments/ddp_signal_check.py` checks the idea inside real
`DistributedDataParallel`. It alternates steps with and without DDP's all-reduce (`no_sync()`), and calls
the difference each worker's wait. Only each worker's own clock is used.

```
python experiments/ddp_signal_check.py --workers 4 --slow-worker 2 --multiplier 2.0
```

Measured (4 workers on 2 processors):

| slowed | waits of the others (ms) | wait of the slowed worker (ms) | result |
|---|---|---|---|
| w2, x2 | 15.2 to 17.6 | 5.4 | holds |
| w1, x3 | 25.7 to 27.3 | 5.3 | holds |

So in DDP too, the slow worker waits least and the others wait about as long as it was slowed. Two
limits:
- DDP hides its all-reduce inside backward(). This measures the total wait, not the wait per ring round,
  so it confirms the "who waits least" signal but not the round-by-round culprit estimate.
- The ring detector would need hooks into DDP (a communication hook) to run there; that is not built.
