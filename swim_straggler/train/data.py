"""Training data: MNIST when it can be downloaded, otherwise a synthetic stand-in.

The launcher decides once which source is used (`prepare_dataset`) and tells every worker,
so all workers always train on the same kind of data.
"""

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy
import torch

MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_IMAGE_FILE = "train-images-idx3-ubyte.gz"
MNIST_LABEL_FILE = "train-labels-idx1-ubyte.gz"
MNIST_IMAGE_MAGIC_NUMBER = 2051
MNIST_LABEL_MAGIC_NUMBER = 2049
MNIST_MEAN = 0.1307
MNIST_STANDARD_DEVIATION = 0.3081

IMAGE_SIZE = 28
NUMBER_OF_CLASSES = 10
SYNTHETIC_SAMPLE_COUNT = 6000
DOWNLOAD_TIMEOUT_SECONDS = 20


def download_file(url, destination):
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        destination.write_bytes(response.read())


def parse_idx_images(raw_bytes):
    """Parse an MNIST image file (after decompression) into an array of shape (count, 28, 28)."""
    magic_number, count, rows, columns = struct.unpack(">IIII", raw_bytes[:16])
    if magic_number != MNIST_IMAGE_MAGIC_NUMBER:
        raise ValueError(f"not an MNIST image file (magic number {magic_number})")
    pixels = numpy.frombuffer(raw_bytes, dtype=numpy.uint8, offset=16)
    return pixels.reshape(count, rows, columns)


def parse_idx_labels(raw_bytes):
    """Parse an MNIST label file (after decompression) into an array of shape (count,)."""
    magic_number, count = struct.unpack(">II", raw_bytes[:8])
    if magic_number != MNIST_LABEL_MAGIC_NUMBER:
        raise ValueError(f"not an MNIST label file (magic number {magic_number})")
    return numpy.frombuffer(raw_bytes, dtype=numpy.uint8, offset=8)[:count]


def read_mnist_files(data_directory):
    """Read the downloaded MNIST files. Raises an error if a file is missing or malformed."""
    mnist_directory = Path(data_directory) / "mnist"
    image_bytes = gzip.decompress((mnist_directory / MNIST_IMAGE_FILE).read_bytes())
    label_bytes = gzip.decompress((mnist_directory / MNIST_LABEL_FILE).read_bytes())
    images = parse_idx_images(image_bytes)
    labels = parse_idx_labels(label_bytes)
    if len(images) != len(labels):
        raise ValueError("MNIST image and label counts differ")
    return images, labels


def prepare_dataset(train_config):
    """Make sure the data exists and return the source that will be used: "mnist" or "synthetic"."""
    if train_config.dataset == "synthetic":
        return "synthetic"
    mnist_directory = Path(train_config.data_directory) / "mnist"
    try:
        mnist_directory.mkdir(parents=True, exist_ok=True)
        for file_name in (MNIST_IMAGE_FILE, MNIST_LABEL_FILE):
            destination = mnist_directory / file_name
            if not destination.exists():
                download_file(MNIST_BASE_URL + file_name, destination)
        read_mnist_files(train_config.data_directory)  # check that the files are usable
        return "mnist"
    except Exception as error:  # any failure (no network, corrupt file) falls back to synthetic data
        print(f"WARNING: could not use MNIST ({error!r}); falling back to synthetic data.")
        return "synthetic"


def make_synthetic_dataset(seed):
    """Images that are one fixed random pattern per class plus noise, so a network can learn them."""
    generator = torch.Generator().manual_seed(seed)
    class_patterns = torch.randn(NUMBER_OF_CLASSES, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    labels = torch.randint(0, NUMBER_OF_CLASSES, (SYNTHETIC_SAMPLE_COUNT,), generator=generator)
    noise = torch.randn(SYNTHETIC_SAMPLE_COUNT, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    images = class_patterns[labels] + noise
    return images.unsqueeze(1), labels


def load_rank_shard(dataset_source, train_config, rank, world_size):
    """Return only this worker's share of the data: every world_size-th sample, starting at rank."""
    if dataset_source == "mnist":
        raw_images, raw_labels = read_mnist_files(train_config.data_directory)
        shard_images = torch.from_numpy(raw_images[rank::world_size].copy()).float() / 255.0
        shard_images = (shard_images - MNIST_MEAN) / MNIST_STANDARD_DEVIATION
        shard_images = shard_images.unsqueeze(1)
        shard_labels = torch.from_numpy(raw_labels[rank::world_size].astype(numpy.int64))
        return shard_images, shard_labels
    images, labels = make_synthetic_dataset(train_config.seed)
    return images[rank::world_size].clone(), labels[rank::world_size].clone()


class BatchProvider:
    """Hands out shuffled batches from one worker's shard, one batch per training step."""

    def __init__(self, images, labels, batch_size, seed, rank):
        if batch_size > len(images):
            raise ValueError(f"batch size {batch_size} is larger than this worker's {len(images)} samples")
        self._images = images
        self._labels = labels
        self._batch_size = batch_size
        self._seed = seed
        self._rank = rank
        self._batches_per_epoch = len(images) // batch_size
        self._cached_epoch = None
        self._cached_order = None

    def batch_for_step(self, step):
        epoch, batch_within_epoch = divmod(step, self._batches_per_epoch)
        if epoch != self._cached_epoch:
            generator = torch.Generator().manual_seed(self._seed * 1_000_003 + epoch * 1_009 + self._rank)
            self._cached_order = torch.randperm(len(self._images), generator=generator)
            self._cached_epoch = epoch
        start = batch_within_epoch * self._batch_size
        chosen_samples = self._cached_order[start : start + self._batch_size]
        return self._images[chosen_samples], self._labels[chosen_samples]
