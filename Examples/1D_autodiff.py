"""Differentiate and solve an inverse problem through a SPECTRAX simulation.

The first calculation verifies forward- and reverse-mode derivatives of the
final electric-field energy. The second infers the counter-streaming drift
from a synthetic electric-energy time trace. Diffrax's discrete checkpoint
adjoint follows the same discretize-then-differentiate principle emphasized by
Skene and Burns, https://arxiv.org/abs/2506.14792.
"""

import json

import jax
import jax.numpy as jnp
from diffrax import Euler, ForwardMode, RecursiveCheckpointAdjoint

from spectrax import simulation


REVERSE = RecursiveCheckpointAdjoint(checkpoints=8)
FORWARD = ForwardMode()
SOLVER = dict(
    Nx=17, Ny=1, Nz=1, Nn=12, Nm=1, Np=1, Ns=2,
    timesteps=21, dt=0.005, solver=Euler(), adaptive_time_step=False,
)


def electric_energy_trace(drift, adjoint=REVERSE):
    output = simulation(
        {"u_s": jnp.array((drift, 0.0, 0.0, -drift, 0.0, 0.0)), "t_max": 5.0},
        adjoint=adjoint, **SOLVER,
    )
    energy = jnp.real(jnp.sum(abs(output["Fk"]) ** 2, axis=(1, 2, 3, 4)))
    return jnp.log(energy)


def main():
    drift = jnp.array(1.1)
    final_energy = lambda value, adjoint: electric_energy_trace(value, adjoint)[-1]
    reverse = jax.grad(lambda value: final_energy(value, REVERSE))(drift)
    forward = jax.jacfwd(lambda value: final_energy(value, FORWARD))(drift)
    step = 1e-4
    finite_difference = (
        final_energy(drift + step, REVERSE) - final_energy(drift - step, REVERSE)
    ) / (2 * step)

    target = jax.lax.stop_gradient(electric_energy_trace(jnp.array(1.0)))

    def loss(value):
        return jnp.mean((electric_energy_trace(value) - target) ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    history = []
    for _ in range(12):
        value, gradient = value_and_grad(drift)
        drift -= 0.005 * gradient
        history.append(float(value))

    print(json.dumps({
        "final_log_energy_derivative": {
            "reverse": float(reverse),
            "forward": float(forward),
            "finite_difference": float(finite_difference),
        },
        "inferred_drift": float(drift),
        "target_drift": 1.0,
        "loss": history,
    }, indent=2))


if __name__ == "__main__":
    main()
