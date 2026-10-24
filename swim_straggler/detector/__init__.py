"""The straggler detector. Pure Python: it knows nothing about torch, sockets or processes.

It is fed numbers (durations measured locally, plus gossip received from the previous worker
in the ring) and returns numbers (gossip to send) and events (what it concluded).
"""
