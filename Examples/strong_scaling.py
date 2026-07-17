"""Measure one fixed workload with different JAX device counts.

For example::

    JAX_NUM_CPU_DEVICES=4 python Examples/strong_scaling.py
    JAX_NUM_CPU_DEVICES=8 python Examples/strong_scaling.py
    CUDA_VISIBLE_DEVICES=0,1 python Examples/strong_scaling.py
"""

from time import perf_counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

import jax
from diffrax import Bosh3

from spectrax import simulation


def run():
    return simulation({"t_max": 0.01}, Nx=8192, Nn=20, timesteps=2,
                      dt=1e-3, solver=Bosh3(), shard_axis="x", adaptive=False)


for label in ("cold", "warm"):
    start = perf_counter()
    output = run()
    jax.block_until_ready(output["Ck"])
    print(label, perf_counter() - start)

print("devices", jax.device_count())
print("local Ck shapes", [shard.data.shape for shard in output["Ck"].addressable_shards])
