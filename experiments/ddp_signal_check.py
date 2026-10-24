"""EXPERIMENTAL check: does the wait signal also exist in real PyTorch DistributedDataParallel (DDP)?

The detector reads stragglers from how long each worker waits in our hand-written ring. Real training
uses DDP, whose all-reduce is hidden inside backward(). This script checks the core idea there: with one
slow worker, every other worker waits for it, and the slow worker waits least.

How it measures, on each worker, using only that worker's own clock:
  * synced step   : forward + backward with DDP's all-reduce (backward returns after the gradients arrived)
  * unsynced step : forward + backward inside `no_sync()` (no all-reduce: pure compute)
  * wait          = median(synced) - median(unsynced)
Steps alternate between the two kinds. DDP is used only here, never in the detector's main path.

Run it:  python experiments/ddp_signal_check.py --workers 4 --slow-worker 2 --multiplier 2.0
"""

import argparse
import os
import socket
import statistics
import time

import torch
import torch.distributed as dist
from torch import multiprocessing
from torch.nn import functional
from torch.nn.parallel import DistributedDataParallel

from swim_straggler.train.model import SmallConvolutionalNetwork


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def worker(rank, world_size, port, slow_worker, multiplier, steps, results):
    torch.set_num_threads(1)
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(0)
    model = DistributedDataParallel(SmallConvolutionalNetwork())
    images, labels = torch.randn(64, 1, 28, 28), torch.randint(0, 10, (64,))

    def one_step():
        start = time.perf_counter()
        loss = functional.cross_entropy(model(images), labels)
        if rank == slow_worker:
            time.sleep((multiplier - 1.0) * last_compute[0])  # slow down like the injected compute faults
        loss.backward()
        return time.perf_counter() - start

    last_compute = [0.01]
    synced, unsynced = [], []
    for step in range(steps):
        if step % 2 == 0:
            with model.no_sync():
                seconds = one_step()
            last_compute[0] = seconds if rank != slow_worker else seconds / multiplier
            unsynced.append(seconds)
        else:
            synced.append(one_step())
        model.zero_grad()
        dist.barrier()  # every step starts together, so unsynced steps measure compute only
    warmup = 5
    results[rank] = (statistics.median(unsynced[warmup:]), statistics.median(synced[warmup:]))
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--slow-worker", type=int, default=2)
    parser.add_argument("--multiplier", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=80)
    arguments = parser.parse_args()
    if arguments.workers > (os.cpu_count() or 1):
        print(f"WARNING: {arguments.workers} workers on {os.cpu_count()} processors: timings will be noisy")

    results = multiprocessing.Manager().dict()
    multiprocessing.spawn(worker, args=(arguments.workers, free_port(), arguments.slow_worker, arguments.multiplier,
                                        arguments.steps, results), nprocs=arguments.workers, join=True)
    print(f"EXPERIMENTAL DDP check: {arguments.workers} workers, worker {arguments.slow_worker} "
          f"slowed x{arguments.multiplier} (medians over {arguments.steps // 2 - 5} steps of each kind)")
    print("worker  compute only (ms)  with DDP all-reduce (ms)  wait (ms)")
    waits = {}
    for rank in range(arguments.workers):
        compute, synced = results[rank]
        waits[rank] = synced - compute
        marker = "  <- slowed" if rank == arguments.slow_worker else ""
        print(f"w{rank:<6} {compute * 1000:17.1f}  {synced * 1000:24.1f}  {waits[rank] * 1000:9.1f}{marker}")
    least = min(waits, key=waits.get)
    holds = least == arguments.slow_worker
    print(f"the worker that waited least: w{least} -> the signal {'HOLDS' if holds else 'does NOT hold'} in DDP here")


if __name__ == "__main__":
    main()
