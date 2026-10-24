# One image for every worker. Build it once from the project folder:  swim-straggler build-image
FROM python:3.11-slim

# iproute2: `tc` for the network-delay fault. stress-ng: the processor-hungry neighbour. procps: `pkill`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends iproute2 stress-ng procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The CPU-only build of torch is much smaller than the default one.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY swim_straggler ./swim_straggler
RUN pip install --no-cache-dir .

ENV OMP_NUM_THREADS=1
ENTRYPOINT ["swim-straggler"]
