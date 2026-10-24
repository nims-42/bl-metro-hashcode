"""Running docker commands, and the exact commands that create each container fault.

Every fault is a list of plain command lines (`start` and `stop`), so they are easy to read,
to test without Docker, and to copy into a terminal when something does not work.
"""

import subprocess
from dataclasses import dataclass, field

PUMBA_IMAGE = "ghcr.io/alexei-led/pumba:latest"
PUMBA_NETWORK_TOOLS_IMAGE = "ghcr.io/alexei-led/pumba-alpine-nettools:latest"
FAULT_DURATION_FOR_TOOLS = "24h"  # we end faults ourselves; this is only an upper limit for the tools
NETWORK_INTERFACE = "eth0"
NORMAL_CPU_PERIOD_MICROSECONDS = 100_000
SMALLEST_CPU_PERIOD_MICROSECONDS = 2_000  # Linux allows quotas down to 1 ms, so slices of 1-2 ms are the finest possible


def normal_cpu_limit(cluster):
    """(period, quota) in microseconds for the normal processor limit, for example 100000 / 100000 for 1 processor."""
    return NORMAL_CPU_PERIOD_MICROSECONDS, int(round(cluster.container_cpus * NORMAL_CPU_PERIOD_MICROSECONDS))


def throttled_cpu_limit(cluster, multiplier):
    """(period, quota) giving the container 1/multiplier of its normal share, in very short slices.

    A plain `--cpus 0.5` is a quota per 100 ms. A worker that computes for 12 ms per 30 ms step never
    uses it up, so it is never slowed. With slices of about 1 ms every burst of computing is cut into
    pieces and really takes about `multiplier` times longer.
    """
    period = max(SMALLEST_CPU_PERIOD_MICROSECONDS, int(1000 * multiplier / cluster.container_cpus))
    quota = int(period * cluster.container_cpus / multiplier)
    return period, quota


def cpu_update_command(container_name, period, quota):
    return ["docker", "update", "--cpu-period", str(period), "--cpu-quota", str(quota), container_name]


class ContainerError(RuntimeError):
    """Raised when Docker is missing, an image is missing or a container fails."""


class CommandRunner:
    """Runs command lines. Tests replace this with a recording fake."""

    def run(self, command, check=True):
        """Run a command to completion and return its output text."""
        try:
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        except FileNotFoundError as error:
            raise ContainerError(f"cannot run '{command[0]}': is it installed and on the PATH?") from error
        if check and completed.returncode != 0:
            raise ContainerError(f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr.strip()}")
        return completed.stdout


@dataclass
class FaultAction:
    """How to start one fault on one container, and how to end it again."""

    start: list = field(default_factory=list)  # command lines, run in order
    stop: list = field(default_factory=list)


def number_text(value):
    return f"{value:.3f}".rstrip("0").rstrip(".")


def pumba_command(helper_name, pumba_arguments):
    """Pumba runs as its own container and controls the others through the Docker socket."""
    return ["docker", "run", "-d", "--rm", "--name", helper_name,
            "-v", "/var/run/docker.sock:/var/run/docker.sock", PUMBA_IMAGE, "--log-level", "info", *pumba_arguments]


def build_fault_action(fault, cluster, container_name):
    """The commands that apply (and later remove) the configured fault on one container."""
    kind = fault.kind
    helper_name = f"{container_name}_pumba"
    use_pumba = fault.fault_tool == "pumba"

    if kind == "container_cpu":  # always the docker tool: pumba cannot throttle a processor limit
        return FaultAction(
            start=[cpu_update_command(container_name, *throttled_cpu_limit(cluster, fault.multiplier))],
            stop=[cpu_update_command(container_name, *normal_cpu_limit(cluster))],
        )

    if kind == "container_stress":
        if use_pumba:
            arguments = ["stress", "--duration", FAULT_DURATION_FOR_TOOLS,
                         "--stressors", f"--cpu {fault.stress_workers} --timeout {FAULT_DURATION_FOR_TOOLS}", container_name]
            return FaultAction(start=[pumba_command(helper_name, arguments)], stop=[["docker", "stop", helper_name]])
        return FaultAction(
            start=[["docker", "exec", "-d", container_name, "stress-ng", "--cpu", str(fault.stress_workers),
                    "--timeout", FAULT_DURATION_FOR_TOOLS]],
            stop=[["docker", "exec", container_name, "pkill", "stress-ng"]],
        )

    if kind == "container_network_delay":
        if use_pumba:
            arguments = ["netem", "--duration", FAULT_DURATION_FOR_TOOLS, "--tc-image", PUMBA_NETWORK_TOOLS_IMAGE,
                         "delay", "--time", str(fault.network_delay_ms), container_name]
            return FaultAction(start=[pumba_command(helper_name, arguments)], stop=[["docker", "stop", helper_name]])
        return FaultAction(
            start=[["docker", "exec", container_name, "tc", "qdisc", "add", "dev", NETWORK_INTERFACE,
                    "root", "netem", "delay", f"{fault.network_delay_ms}ms"]],
            stop=[["docker", "exec", container_name, "tc", "qdisc", "del", "dev", NETWORK_INTERFACE, "root"]],
        )

    if kind == "container_pause":
        if use_pumba:
            arguments = ["pause", "--duration", FAULT_DURATION_FOR_TOOLS, container_name]
            return FaultAction(start=[pumba_command(helper_name, arguments)], stop=[["docker", "stop", helper_name]])
        return FaultAction(start=[["docker", "pause", container_name]], stop=[["docker", "unpause", container_name]])

    raise ContainerError(f"'{kind}' is not a container fault kind")
