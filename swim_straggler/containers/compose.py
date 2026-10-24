"""Writing the files the containers need: the config they read and the Docker Compose file."""

import copy
from pathlib import Path

import yaml

import swim_straggler

from swim_straggler.containers.commands import normal_cpu_limit

from swim_straggler.config import CONTAINER_FAULT_KINDS

CONTAINER_MASTER_PORT = 29500
CONTAINER_RUN_DIRECTORY = "/run"
CONTAINER_DATA_DIRECTORY = "/app/data"
CONTAINER_SOURCE_DIRECTORY = "/app/swim_straggler"
CONTAINER_CONFIG_PATH = f"{CONTAINER_RUN_DIRECTORY}/config.yaml"


def project_name(config, run_directory):
    return "swim_" + "".join(character if character.isalnum() else "_" for character in Path(run_directory).name.lower())


def container_name(project, rank):
    return f"{project}_worker{rank}"


def config_for_containers(config):
    """The config every container reads: worker 0 is the meeting point, and the data sits under /app/data."""
    container_config = copy.deepcopy(config)
    container_config.cluster.master_address = "worker0"  # the Compose service name of rank 0
    container_config.cluster.master_port = config.cluster.master_port or CONTAINER_MASTER_PORT
    container_config.train.data_directory = "data"
    return container_config


def write_container_config(config, config_path):
    Path(config_path).write_text(yaml.safe_dump(config_for_containers(config).to_dict(), sort_keys=False), encoding="utf-8")


def build_compose(config, run_directory, data_directory, dataset_source):
    """The Compose file as a dictionary: one service (= one container) per worker."""
    project = project_name(config, run_directory)
    cpu_period, cpu_quota = normal_cpu_limit(config.cluster)
    volumes = [
        {"type": "bind", "source": str(Path(run_directory).resolve()), "target": CONTAINER_RUN_DIRECTORY},
        {"type": "bind", "source": str(Path(data_directory).resolve()), "target": CONTAINER_DATA_DIRECTORY},
    ]
    environment = {"OMP_NUM_THREADS": "1", "GLOO_SOCKET_IFNAME": "eth0"}
    if config.cluster.mount_source_code:
        # The image was built once; without this, containers would silently run the code from build time.
        source_directory = Path(swim_straggler.__file__).resolve().parent
        volumes.append({"type": "bind", "source": str(source_directory), "target": CONTAINER_SOURCE_DIRECTORY,
                        "read_only": True})
        environment["PYTHONPATH"] = "/app"  # found before the copy installed in the image
    services = {}
    for rank in range(config.cluster.world_size):
        services[f"worker{rank}"] = {
            "image": config.cluster.container_image,
            "container_name": container_name(project, rank),
            "command": ["worker", "--config", CONTAINER_CONFIG_PATH, "--rank", str(rank),
                        "--dataset-source", dataset_source, "--run-directory", CONTAINER_RUN_DIRECTORY],
            "environment": environment,
            "volumes": volumes,
            # Not `cpus:`; docker refuses to change period and quota later (the cpu fault) on a container made with `cpus`.
            "cpu_period": cpu_period,
            "cpu_quota": cpu_quota,
            "cap_add": ["NET_ADMIN"],  # lets the network-delay fault run `tc` inside the container
        }
    return {"name": project, "services": services}


def write_compose_file(compose, compose_path):
    Path(compose_path).write_text(yaml.safe_dump(compose, sort_keys=False), encoding="utf-8")


def uses_container_faults(config):
    return config.fault.kind in CONTAINER_FAULT_KINDS
