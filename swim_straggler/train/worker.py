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
from swim_straggler.detector.factory import create_detector
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


def flatten_gradients_with_loss(model, loss, extra_values=None):
    """One vector: every gradient value in parameter order, then the loss, then any extra values."""
    gradient_pieces = [parameter.grad.flatten() for parameter in model.parameters()]
    gradient_pieces.append(loss.detach().reshape(1))
    if extra_values is not None:
        gradient_pieces.append(extra_values)
    return torch.cat(gradient_pieces)


def compute_report_slots(rank, world_size, compute_seconds):
    """N numbers: my compute time in my own slot, zeros elsewhere. Summed over all workers, they hold everyone's."""
    slots = torch.zeros(world_size)
    slots[rank] = compute_seconds
    return slots


def write_averaged_gradients_back(model, summed_vector, world_size):
    """Copy the averaged gradients into the model and return the average loss."""
    averaged_vector = summed_vector / world_size
    position = 0
    for parameter in model.parameters():
        parameter_size = parameter.numel()
        parameter.grad.copy_(averaged_vector[position : position + parameter_size].view_as(parameter))
        position += parameter_size
    return averaged_vector[position].item()


def log_detector_events(logger, detector_events):
    """Decisions are logged as "detector" events, the reasoning behind them (detector.trace) as "trace" events."""
    for detector_event in detector_events:
        event_name = "trace" if detector_event.pop("trace", False) else "detector"
        logger.log(event_name, **detector_event)


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
    codec = HeaderCodec(gossip_length=detector.gossip_length)

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

        # 2. all-reduce (the SWIM detector rides along: its gossip travels in the header of every message)
        def build_header():
            return Header(sender=rank, step=step, compute_time=compute_seconds,
                          wait_time=previous_wait_seconds, gossip=tuple(detector.make_gossip()))

        def absorb_header(received_header, round_number):
            detector.receive_gossip(received_header.gossip, sender=received_header.sender)

        detector.start_step(step, compute_seconds)

        transport.current_step = step
        report_slots = compute_report_slots(rank, world_size, compute_seconds) if detector.needs_compute_reports else None
        vector = flatten_gradients_with_loss(model, loss, report_slots)
        allreduce_start_time = time.perf_counter()
        wait_seconds_per_round = ring_allreduce(
            transport, vector, codec, make_header=build_header, on_header_received=absorb_header,
        )
        allreduce_seconds = time.perf_counter() - allreduce_start_time
        wait_seconds = sum(wait_seconds_per_round)
        previous_wait_seconds = wait_seconds

        compute_of_every_worker = vector[-world_size:].tolist() if detector.needs_compute_reports else None
        log_detector_events(logger, detector.finish_step(step, wait_seconds_per_round, compute_of_every_worker))

        # 3. update
        average_loss = write_averaged_gradients_back(model, vector, world_size)
        optimizer.step()

        step_seconds = time.perf_counter() - step_start_time
        logger.log(
            "step",
            step=step,
            loss=average_loss,
            compute_seconds=compute_seconds,
            wait_seconds=wait_seconds,
            wait_seconds_per_round=wait_seconds_per_round,
            allreduce_seconds=allreduce_seconds,
            step_seconds=step_seconds,
            fault_active=fault_is_active,
            injected_compute_seconds=injected_seconds,
        )
        log_detector_events(logger, detector.after_update(step, step_seconds))

    logger.log(
        "run_end",
        parameter_checksum=parameter_checksum(model),
        believed_stragglers=detector.straggler_ranks(),
        seconds_since_start=time.perf_counter() - run_start_time,
    )
    logger.close()
