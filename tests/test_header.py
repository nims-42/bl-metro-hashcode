import pytest
import torch

from swim_straggler.allreduce.header import Header, HeaderCodec


def test_round_trip():
    codec = HeaderCodec(gossip_len=3)
    h = Header(sender=5, step=1234, compute_time=0.0123, wait_time=0.5, transfer_time=1e-4, gossip=(1.0, 2.5))
    d = codec.decode(codec.encode(h))
    assert (d.sender, d.step) == (5, 1234)
    assert d.compute_time == pytest.approx(0.0123, rel=1e-6)
    assert d.wait_time == pytest.approx(0.5, rel=1e-6)
    assert d.gossip == (1.0, 2.5, 0.0)  # short gossip is zero-padded


def test_rejects_bad_sizes():
    codec = HeaderCodec(gossip_len=2)
    with pytest.raises(ValueError):
        codec.encode(Header(0, 0, gossip=(1.0, 2.0, 3.0)))
    with pytest.raises(ValueError):
        codec.decode(torch.zeros(codec.size + 1))
