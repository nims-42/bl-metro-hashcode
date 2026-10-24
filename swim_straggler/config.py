"""Configuration: plain dataclasses loaded from a YAML file.

Add a field here only when some code actually uses it.
"""

from dataclasses import asdict, dataclass, field

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is missing, misspelt or contains invalid values."""


@dataclass
class ClusterConfig:
    world_size: int = 8
    backend: str = "gloo"  # "gloo" (one process per worker) or "pipes" (one thread per worker)
    master_address: str = "127.0.0.1"
    master_port: int = 0  # 0 means "choose any free port"


@dataclass
class TrainConfig:
    steps: int = 150
    batch_size: int = 64  # per worker
    learning_rate: float = 0.05
    seed: int = 0
    dataset: str = "mnist"  # "mnist" or "synthetic"
    data_directory: str = "data"


@dataclass
class FaultConfig:
    # none         : no fault
    # compute      : the listed workers compute `multiplier` times slower
    # two          : same as compute, but exactly two workers
    # intermittent : compute slowdown that switches on and off every `period` steps
    # spike        : compute slowdown that lasts only `duration` steps
    # link         : every message the listed workers send is delayed by `link_delay_seconds`
    kind: str = "none"
    workers: list = field(default_factory=list)
    multiplier: float = 2.0
    start_step: int = 50
    duration: int = 0  # number of steps the fault lasts; 0 means until the end of the run
    period: int = 10  # intermittent faults: steps per on or off phase
    slowdown_style: str = "sleep"  # "sleep" frees the processor, "busy" keeps it busy like a throttled chip
    link_delay_seconds: float = 0.005  # extra delay per message for link faults


@dataclass
class DetectorConfig:
    mode: str = "swim"  # swim, central, steptime or off
    factor: float = 1.3  # suspect a worker whose compute time exceeds factor x the median
    consecutive_steps: int = 5
    k_probes: int = 3
    recovery_steps: int = 20
    ewma_alpha: float = 0.3  # weight of the newest measurement when smoothing (1.0 means no smoothing)
    warmup_steps: int = 10  # steps ignored at the start; the first steps are slow and noisy
    health_factor: float = 1.5  # a worker whose own compute exceeds this x the median does not accuse others
    wait_floor_seconds: float = 0.002  # smaller differences in wait time are treated as noise
    gossip_rows: int = 8  # table rows carried in every message (rotating through the table if larger)
    verdict_slots: int = 4  # suspicion and verdict notices carried in every message
    verdict_repeat_steps: int = 5  # steps a worker keeps passing a verdict on after learning it

    def __post_init__(self):
        # YAML reads an unquoted `off` as the boolean False, so `mode: off` would otherwise be rejected.
        if self.mode is False:
            self.mode = "off"


@dataclass
class Config:
    name: str = "run"
    run_directory: str = "runs"
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    fault: FaultConfig = field(default_factory=FaultConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    def to_dict(self):
        """The whole configuration as plain dictionaries, for writing into the event log."""
        return asdict(self)

    def validate(self):
        """Raise ConfigError listing every problem found."""
        cluster, train, fault, detector = self.cluster, self.train, self.fault, self.detector
        compute_kinds = ("compute", "two", "intermittent", "spike")
        checks = [
            (cluster.world_size >= 2, "cluster.world_size must be at least 2"),
            (cluster.backend in ("gloo", "pipes"), "cluster.backend must be gloo or pipes"),
            (0 <= cluster.master_port <= 65535, "cluster.master_port must be between 0 and 65535"),
            (train.steps >= 1, "train.steps must be at least 1"),
            (train.batch_size >= 1, "train.batch_size must be at least 1"),
            (train.learning_rate > 0, "train.learning_rate must be positive"),
            (train.dataset in ("mnist", "synthetic"), "train.dataset must be mnist or synthetic"),
            (fault.kind in ("none", "compute", "two", "intermittent", "spike", "link"), "unknown fault.kind"),
            (fault.kind == "none" or len(fault.workers) > 0, "fault.workers is required when fault.kind is not none"),
            (all(0 <= worker < cluster.world_size for worker in fault.workers),
             "every fault.workers entry must be between 0 and world_size - 1"),
            (fault.kind != "two" or len(set(fault.workers)) == 2, "fault.kind two needs exactly two distinct fault.workers"),
            (fault.kind not in compute_kinds or fault.multiplier >= 1.0, "fault.multiplier must be at least 1.0"),
            (fault.start_step >= 0, "fault.start_step must be 0 or more"),
            (fault.duration >= 0, "fault.duration must be 0 or more"),
            (fault.kind != "spike" or fault.duration >= 1, "a spike needs fault.duration of at least 1"),
            (fault.period >= 1, "fault.period must be at least 1"),
            (fault.slowdown_style in ("sleep", "busy"), "fault.slowdown_style must be sleep or busy"),
            (fault.link_delay_seconds >= 0, "fault.link_delay_seconds must be 0 or more"),
            (detector.mode in ("swim", "central", "steptime", "off"), "unknown detector.mode"),
            (detector.factor > 1.0, "detector.factor must be greater than 1"),
            (detector.consecutive_steps >= 1, "detector.consecutive_steps must be at least 1"),
            (detector.recovery_steps >= 1, "detector.recovery_steps must be at least 1"),
            (detector.mode != "swim" or 1 <= detector.k_probes <= cluster.world_size - 2,
             "detector.k_probes must be between 1 and world_size - 2"),
            (0.0 < detector.ewma_alpha <= 1.0, "detector.ewma_alpha must be above 0 and at most 1"),
            (detector.warmup_steps >= 0, "detector.warmup_steps must be 0 or more"),
            (detector.health_factor > 1.0, "detector.health_factor must be greater than 1"),
            (detector.wait_floor_seconds >= 0, "detector.wait_floor_seconds must be 0 or more"),
            (detector.gossip_rows >= 1 and detector.verdict_slots >= 1,
             "detector.gossip_rows and detector.verdict_slots must be at least 1"),
            (detector.verdict_repeat_steps >= 1, "detector.verdict_repeat_steps must be at least 1"),
        ]
        problems = [message for condition_holds, message in checks if not condition_holds]
        if problems:
            raise ConfigError("invalid config: " + "; ".join(problems))


SECTION_CLASSES = {
    "cluster": ClusterConfig,
    "train": TrainConfig,
    "fault": FaultConfig,
    "detector": DetectorConfig,
}
TOP_LEVEL_FIELDS = ("name", "run_directory")


def load_config(file_path):
    """Read a YAML file and return a validated Config."""
    with open(file_path, encoding="utf-8") as config_file:
        raw_values = yaml.safe_load(config_file) or {}

    allowed_keys = set(SECTION_CLASSES) | set(TOP_LEVEL_FIELDS)
    unknown_keys = set(raw_values) - allowed_keys
    if unknown_keys:
        raise ConfigError(f"unknown top-level key(s): {sorted(unknown_keys)}")

    sections = {}
    for section_name, section_class in SECTION_CLASSES.items():
        try:
            sections[section_name] = section_class(**(raw_values.get(section_name) or {}))
        except TypeError as error:  # for example a misspelt field name
            raise ConfigError(f"{section_name}: {error}") from error

    top_level = {key: raw_values[key] for key in TOP_LEVEL_FIELDS if key in raw_values}
    config = Config(**top_level, **sections)
    config.validate()
    return config
