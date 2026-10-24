from pathlib import Path

import pytest

from swim_straggler.cli import main
from swim_straggler.config import Config, ConfigError, load_config

DEMO = Path(__file__).resolve().parents[1] / "experiments" / "configs" / "demo.yaml"


def test_demo_loads():
    cfg = load_config(DEMO)
    assert cfg.cluster.world_size == 8
    assert (cfg.fault.workers, cfg.fault.multiplier, cfg.fault.start_step) == ([4], 2.0, 50)


def test_defaults_are_valid():
    Config().validate()


def test_misspelt_field_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("cluster: {world_sze: 4}\n")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_bad_values_are_reported():
    cfg = Config()
    cfg.fault.kind, cfg.fault.workers, cfg.detector.factor = "compute", [99], 1.0
    with pytest.raises(ConfigError, match="fault.workers.*detector.factor"):
        cfg.validate()


def test_cli(capsys):
    assert main(["train", "--config", str(DEMO)]) == 2  # config fine, training not built yet
    assert "config OK" in capsys.readouterr().out
    assert main(["train", "--config", "/nope.yaml"]) == 1
