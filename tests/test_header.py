import pytest
import torch

from swim_straggler.allreduce.header import Header, HeaderCodec


def test_round_trip():
    codec = HeaderCodec(gossip_length=3)
    header = Header(sender=5, step=1234, compute_time=0.0123, wait_time=0.5, transfer_time=1e-4, gossip=(1.0, 2.5))
    decoded = codec.decode(codec.encode(header))
    assert decoded.sender == 5
    assert decoded.step == 1234
    assert decoded.compute_time == pytest.approx(0.0123, rel=1e-6)
    assert decoded.wait_time == pytest.approx(0.5, rel=1e-6)
    assert decoded.gossip == (1.0, 2.5, 0.0)  # a short gossip block is padded with zeros


def test_wrong_sizes_are_rejected():
    codec = HeaderCodec(gossip_length=2)
    with pytest.raises(ValueError):
        codec.encode(Header(0, 0, gossip=(1.0, 2.0, 3.0)))
    with pytest.raises(ValueError):
        codec.decode(torch.zeros(codec.size + 1))
