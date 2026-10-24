from pathlib import Path

import pytest

from swim_straggler.cli import main
from swim_straggler.config import Config, ConfigError, load_config

CONFIG_DIRECTORY = Path(__file__).resolve().parents[1] / "experiments" / "configs"


def test_demo_config_loads():
    config = load_config(CONFIG_DIRECTORY / "demo.yaml")
    assert config.cluster.world_size == 8
    assert config.fault.workers == [4]
    assert config.fault.multiplier == 2.0
    assert config.fault.start_step == 50


def test_every_shipped_config_is_valid():
    for config_path in CONFIG_DIRECTORY.glob("*.yaml"):
        load_config(config_path)


def test_defaults_are_valid():
    Config().validate()


def test_unquoted_off_in_yaml_means_off(tmp_path):
    config_path = tmp_path / "off.yaml"
    config_path.write_text("detector: {mode: off}\n")  # YAML reads a bare `off` as False
    assert load_config(config_path).detector.mode == "off"


def test_misspelt_field_is_rejected(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("cluster: {world_sze: 4}\n")
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_all_problems_are_reported_together():
    config = Config()
    config.fault.kind = "compute"
    config.fault.workers = [99]
    config.detector.factor = 1.0
    with pytest.raises(ConfigError, match="fault.workers.*detector.factor"):
        config.validate()


def test_spike_needs_a_duration_and_two_needs_two_workers():
    config = Config()
    config.fault.kind = "spike"
    config.fault.workers = [1]
    with pytest.raises(ConfigError, match="spike"):
        config.validate()
    config.fault.kind = "two"
    config.fault.duration = 3
    with pytest.raises(ConfigError, match="two"):
        config.validate()


def test_command_line_trains_and_reports_bad_config(tmp_path, capsys):
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(
        f"name: tiny\nrun_directory: {tmp_path / 'runs'}\n"
        "cluster: {world_size: 2, backend: pipes}\n"
        f"train: {{steps: 6, batch_size: 16, dataset: synthetic, data_directory: {tmp_path / 'data'}}}\n"
        "detector: {mode: off}\n"
    )
    assert main(["train", "--config", str(config_path)]) == 0
    assert "event log" in capsys.readouterr().out
    assert main(["train", "--config", str(tmp_path / "missing.yaml")]) == 1
