import gzip
import struct

import numpy
import pytest
import torch

from swim_straggler.config import TrainConfig
from swim_straggler.train import data
from swim_straggler.train.data import BatchProvider, load_rank_shard, make_synthetic_dataset, prepare_dataset


def build_idx_image_bytes(images):
    count, rows, columns = images.shape
    return struct.pack(">IIII", data.MNIST_IMAGE_MAGIC_NUMBER, count, rows, columns) + images.tobytes()


def build_idx_label_bytes(labels):
    return struct.pack(">II", data.MNIST_LABEL_MAGIC_NUMBER, len(labels)) + labels.tobytes()


def test_idx_parsers_read_what_was_written():
    images = numpy.arange(3 * 28 * 28, dtype=numpy.uint8).reshape(3, 28, 28)
    labels = numpy.array([4, 1, 7], dtype=numpy.uint8)
    numpy.testing.assert_array_equal(data.parse_idx_images(build_idx_image_bytes(images)), images)
    numpy.testing.assert_array_equal(data.parse_idx_labels(build_idx_label_bytes(labels)), labels)


def test_parser_rejects_wrong_file_type():
    with pytest.raises(ValueError):
        data.parse_idx_images(build_idx_label_bytes(numpy.zeros(5, dtype=numpy.uint8)) + bytes(8))


def test_mnist_shards_are_disjoint_and_normalised(tmp_path):
    images = numpy.random.default_rng(0).integers(0, 256, size=(8, 28, 28), dtype=numpy.uint8)
    labels = numpy.arange(8, dtype=numpy.uint8)
    mnist_directory = tmp_path / "mnist"
    mnist_directory.mkdir()
    (mnist_directory / data.MNIST_IMAGE_FILE).write_bytes(gzip.compress(build_idx_image_bytes(images)))
    (mnist_directory / data.MNIST_LABEL_FILE).write_bytes(gzip.compress(build_idx_label_bytes(labels)))
    train_config = TrainConfig(dataset="mnist", data_directory=str(tmp_path))

    assert prepare_dataset(train_config) == "mnist"
    shard_zero_images, shard_zero_labels = load_rank_shard("mnist", train_config, rank=0, world_size=2)
    _, shard_one_labels = load_rank_shard("mnist", train_config, rank=1, world_size=2)
    assert shard_zero_labels.tolist() == [0, 2, 4, 6]
    assert shard_one_labels.tolist() == [1, 3, 5, 7]
    assert shard_zero_images.shape == (4, 1, 28, 28)


def test_download_failure_falls_back_to_synthetic(tmp_path, monkeypatch, capsys):
    def failing_download(url, destination):
        raise OSError("no network")

    monkeypatch.setattr(data, "download_file", failing_download)
    train_config = TrainConfig(dataset="mnist", data_directory=str(tmp_path))
    assert prepare_dataset(train_config) == "synthetic"
    assert "falling back to synthetic" in capsys.readouterr().out


def test_synthetic_data_is_reproducible_and_batches_cover_the_shard():
    first_images, first_labels = make_synthetic_dataset(seed=3)
    second_images, second_labels = make_synthetic_dataset(seed=3)
    assert torch.equal(first_images, second_images) and torch.equal(first_labels, second_labels)

    shard_images = torch.arange(10, dtype=torch.float32).reshape(10, 1, 1, 1)
    shard_labels = torch.arange(10)
    provider = BatchProvider(shard_images, shard_labels, batch_size=5, seed=0, rank=0)
    first_epoch = torch.cat([provider.batch_for_step(0)[1], provider.batch_for_step(1)[1]])
    assert sorted(first_epoch.tolist()) == list(range(10))  # every sample exactly once per epoch
