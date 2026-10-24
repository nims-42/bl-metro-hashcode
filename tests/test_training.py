import pytest

from swim_straggler.config import ClusterConfig, Config, DetectorConfig, FaultConfig, TrainConfig
from swim_straggler.telemetry.events import read_events, summarize_training
from swim_straggler.train.launcher import run_training


def make_config(tmp_path, backend="pipes", world_size=4, steps=40, fault=None):
    return Config(
        name="test",
        run_directory=str(tmp_path / "runs"),
        cluster=ClusterConfig(world_size=world_size, backend=backend),
        train=TrainConfig(steps=steps, batch_size=32, dataset="synthetic", data_directory=str(tmp_path / "data")),
        fault=fault or FaultConfig(),
        detector=DetectorConfig(mode="off"),
    )


def check_training_log(events, world_size, steps):
    step_events = [event for event in events if event["event"] == "step"]
    assert len(step_events) == world_size * steps
    summary = summarize_training(events)
    assert summary["loss_at_end"] < 0.6 * summary["loss_at_start"]  # the network is learning
    checksums = [event["parameter_checksum"] for event in events if event["event"] == "run_end"]
    assert len(checksums) == world_size
    assert max(checksums) - min(checksums) < 1e-4  # all workers hold the same weights


def test_training_on_pipes_learns_and_stays_in_sync(tmp_path):
    result = run_training(make_config(tmp_path))
    assert result.dataset_source == "synthetic"
    events = read_events(result.events_path)
    assert events[0]["event"] == "run_start" and events[0]["dataset_source"] == "synthetic"
    check_training_log(events, world_size=4, steps=40)


@pytest.mark.slow
def test_training_on_gloo_learns_and_stays_in_sync(tmp_path):
    result = run_training(make_config(tmp_path, backend="gloo", world_size=3, steps=40))
    check_training_log(read_events(result.events_path), world_size=3, steps=40)
