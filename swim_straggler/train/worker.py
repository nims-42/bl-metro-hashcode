"""The training loop that every worker runs.

One training step on one worker:
  1. compute  : forward pass, loss, backward pass (plus any injected slowdown)
  2. all-reduce: add the gradients of all workers with the hand-written ring
  3. update   : average the gradients and take an optimizer step

The loss value travels as one extra element at the end of the gradient vector, so after the
all-reduce every worker also knows the average loss over all workers, at no extra cost.
"""

import threading
import time

import torch
from torch.nn import functional

from swim_straggler.allreduce.header import Header, HeaderCodec
from swim_straggler.allreduce.ring import ring_allreduce
from swim_straggler.config import ConfigError
from swim_straggler.detector.swim import StragglerDetector
from swim_straggler.faults.inject import FaultInjector, SlowLinkTransport
from swim_straggler.telemetry.events import EventLogger
from swim_straggler.train.data import BatchProvider, load_rank_shard
from swim_straggler.train.model import SmallConvolutionalNetwork


MODEL_CREATION_LOCK = threading.Lock()


def create_model_with_seed(seed):
    """Build the model so that every worker gets exactly the same starting weights.

    PyTorch draws the starting weights from one global random generator. When workers run as
    threads of one process (the pipes backend) they share that generator, so without the lock
    two workers could interleave their draws and end up with different weights.
    """
    with MODEL_CREATION_LOCK:
        torch.manual_seed(seed)
        return SmallConvolutionalNetwork()


def create_detector(config, rank, world_size):
    """The straggler detector of this worker, or None when detection is switched off."""
    mode = config.detector.mode
    if mode == "off":
        return None
    if mode == "swim":
        return StragglerDetector(rank, world_size, config.detector, seed=config.train.seed)
    raise ConfigError(f"detector.mode '{mode}' is not implemented yet (the baselines arrive in milestone 6)")


def flatten_gradients_with_loss(model, loss):
    """One vector: every gradient value in parameter order, then the loss as the last element."""
    gradient_pieces = [parameter.grad.flatten() for parameter in model.parameters()]
    gradient_pieces.append(loss.detach().reshape(1))
    return torch.cat(gradient_pieces)


def write_averaged_gradients_back(model, summed_vector, world_size):
    """Copy the averaged gradients into the model and return the average loss."""
    averaged_vector = summed_vector / world_size
    position = 0
    for parameter in model.parameters():
        parameter_size = parameter.numel()
        parameter.grad.copy_(averaged_vector[position : position + parameter_size].view_as(parameter))
        position += parameter_size
    return averaged_vector[position].item()


def parameter_checksum(model):
    """A single number summarising all weights. Equal on all workers means the models are in sync."""
    return sum(parameter.sum().item() for parameter in model.parameters())


def train_one_rank(transport, config, dataset_source, run_directory):
    """Train on one worker until config.train.steps steps are done. Writes this worker's event file."""
    rank = transport.rank
    world_size = transport.world_size
    logger = EventLogger(run_directory / f"events_rank_{rank}.jsonl", rank)

    model = create_model_with_seed(config.train.seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.train.learning_rate)

    images, labels = load_rank_shard(dataset_source, config.train, rank, world_size)
    batches = BatchProvider(images, labels, config.train.batch_size, config.train.seed, rank)

    fault_injector = FaultInjector(config.fault, rank)
    transport = SlowLinkTransport(transport, fault_injector)
    detector = create_detector(config, rank, world_size)
    codec = HeaderCodec(gossip_length=detector.gossip_length if detector else 0)

    if rank == 0:
        logger.log("run_start", config=config.to_dict(), dataset_source=dataset_source, world_size=world_size)

    fault_start_logged = False
    previous_wait_seconds = 0.0
    run_start_time = time.perf_counter()

    for step in range(config.train.steps):
        step_start_time = time.perf_counter()
        batch_images, batch_labels = batches.batch_for_step(step)

        # 1. compute
        compute_start_time = time.perf_counter()
        optimizer.zero_grad()
        loss = functional.cross_entropy(model(batch_images), batch_labels)
        loss.backward()
        measured_compute_seconds = time.perf_counter() - compute_start_time
        injected_seconds = fault_injector.slow_down_compute(step, measured_compute_seconds)
        compute_seconds = time.perf_counter() - compute_start_time

        fault_is_active = fault_injector.is_fault_active(step)
        if fault_is_active and not fault_start_logged:
            logger.log("fault_started", step=step, fault_kind=config.fault.kind)
            fault_start_logged = True

        # 2. all-reduce (the detector rides along: its gossip travels in the header of every message)
        def build_header():
            gossip = tuple(detector.make_gossip()) if detector else ()
            return Header(sender=rank, step=step, compute_time=compute_seconds,
                          wait_time=previous_wait_seconds, gossip=gossip)

        def absorb_header(received_header, round_number):
            detector.receive_gossip(received_header.gossip)

        if detector:
            detector.start_step(step, compute_seconds)

        transport.current_step = step
        vector = flatten_gradients_with_loss(model, loss)
        allreduce_start_time = time.perf_counter()
        wait_seconds_per_round = ring_allreduce(
            transport, vector, codec, make_header=build_header,
            on_header_received=absorb_header if detector else None,
        )
        allreduce_seconds = time.perf_counter() - allreduce_start_time
        wait_seconds = sum(wait_seconds_per_round)
        previous_wait_seconds = wait_seconds

        if detector:
            for detector_event in detector.finish_step(step, wait_seconds_per_round):
                logger.log("detector", **detector_event)

        # 3. update
        average_loss = write_averaged_gradients_back(model, vector, world_size)
        optimizer.step()

        logger.log(
            "step",
            step=step,
            loss=average_loss,
            compute_seconds=compute_seconds,
            wait_seconds=wait_seconds,
            wait_seconds_per_round=wait_seconds_per_round,
            allreduce_seconds=allreduce_seconds,
            step_seconds=time.perf_counter() - step_start_time,
            fault_active=fault_is_active,
            injected_compute_seconds=injected_seconds,
        )

    logger.log(
        "run_end",
        parameter_checksum=parameter_checksum(model),
        believed_stragglers=detector.straggler_ranks() if detector else [],
        seconds_since_start=time.perf_counter() - run_start_time,
    )
    logger.close()
