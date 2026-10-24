"""The detector of one worker, chosen by detector.mode. Plain Python (used by training and by the simulator)."""

from swim_straggler.config import ConfigError
from swim_straggler.detector.base import Detector
from swim_straggler.detector.central import CentralMonitor
from swim_straggler.detector.steptime import StepTimeDetector
from swim_straggler.detector.swim import StragglerDetector


def create_detector(config, rank, world_size):
    """The detector of this worker. Mode "off" gives a detector that does nothing."""
    mode = config.detector.mode
    if mode == "off":
        return Detector()
    if mode == "swim":
        return StragglerDetector(rank, world_size, config.detector, seed=config.train.seed)
    if mode == "central":
        return CentralMonitor(rank, world_size, config.detector)
    if mode == "steptime":
        return StepTimeDetector(rank, world_size, config.detector)
    raise ConfigError(f"unknown detector.mode '{mode}'")
