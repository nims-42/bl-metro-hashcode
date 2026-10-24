"""Config: plain dataclasses loaded from YAML. Add fields here as later milestones need them."""

from dataclasses import dataclass, field

import yaml


class ConfigError(ValueError):
    pass


@dataclass
class Cluster:
    world_size: int = 8
    backend: str = "gloo"  # "gloo" or "pipes"


@dataclass
class Train:
    steps: int = 150
    batch_size: int = 64
    lr: float = 0.05
    seed: int = 0
    dataset: str = "mnist"  # "mnist" or "synthetic"


@dataclass
class Fault:
    kind: str = "none"  # none, compute, link, intermittent, spike, two
    workers: list = field(default_factory=list)
    multiplier: float = 2.0
    start_step: int = 50
    duration: int = 0  # 0 = until the end of the run


@dataclass
class Detector:
    mode: str = "swim"  # swim, central, steptime, off
    factor: float = 1.3  # suspect when compute > factor x median
    consecutive_steps: int = 5
    k_probes: int = 3
    recovery_steps: int = 20


@dataclass
class Config:
    name: str = "run"
    cluster: Cluster = field(default_factory=Cluster)
    train: Train = field(default_factory=Train)
    fault: Fault = field(default_factory=Fault)
    detector: Detector = field(default_factory=Detector)

    def validate(self):
        c, t, f, d = self.cluster, self.train, self.fault, self.detector
        checks = [
            (c.world_size >= 2, "cluster.world_size must be >= 2"),
            (c.backend in ("gloo", "pipes"), "cluster.backend must be gloo or pipes"),
            (t.steps >= 1 and t.batch_size >= 1 and t.lr > 0, "train.steps/batch_size/lr must be positive"),
            (t.dataset in ("mnist", "synthetic"), "train.dataset must be mnist or synthetic"),
            (f.kind in ("none", "compute", "link", "intermittent", "spike", "two"), "unknown fault.kind"),
            (f.kind == "none" or f.workers, "fault.workers is required when fault.kind is set"),
            (all(0 <= w < c.world_size for w in f.workers), "fault.workers must be in 0..world_size-1"),
            (f.multiplier >= 1.0 and f.start_step >= 0 and f.duration >= 0, "bad fault multiplier/start_step/duration"),
            (d.mode in ("swim", "central", "steptime", "off"), "unknown detector.mode"),
            (d.factor > 1.0, "detector.factor must be > 1"),
            (d.consecutive_steps >= 1 and d.recovery_steps >= 1, "detector step counts must be >= 1"),
            (1 <= d.k_probes <= max(1, c.world_size - 2), "detector.k_probes must be in 1..world_size-2"),
        ]
        problems = [msg for ok, msg in checks if not ok]
        if problems:
            raise ConfigError("invalid config: " + "; ".join(problems))


_SECTIONS = {"cluster": Cluster, "train": Train, "fault": Fault, "detector": Detector}


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    unknown = set(raw) - set(_SECTIONS) - {"name"}
    if unknown:
        raise ConfigError(f"unknown top-level key(s): {sorted(unknown)}")
    parts = {}
    for key, cls in _SECTIONS.items():
        try:
            parts[key] = cls(**(raw.get(key) or {}))
        except TypeError as exc:  # e.g. a misspelt field name
            raise ConfigError(f"{key}: {exc}") from exc
    cfg = Config(name=raw.get("name", "run"), **parts)
    cfg.validate()
    return cfg
