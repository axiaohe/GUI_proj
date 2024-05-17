from functools import partial

import jax
import jax.numpy as jnp


class RezendeModel:
    def __init__(self):
        self.eps = 1e-7

    def evaluate(self, samples):
        # Construct z
        samples = samples.reshape(-1, 2)
        z_1, z_2 = samples[:, 0], samples[:, 1]
        norm = jnp.sqrt(z_1 ** 2 + z_2 ** 2)

        # First term
        outer_term_1 = 0.5 * ((norm - 2) / 0.4) ** 2

        # Second term
        inner_term_1 = jnp.exp((-0.5 * ((z_1 - 2) / 0.6) ** 2))
        inner_term_2 = jnp.exp((-0.5 * ((z_1 + 2) / 0.6) ** 2))
        outer_term_2 = jnp.log(inner_term_1 + inner_term_2 + self.eps)

        # Potential
        potential = outer_term_1 - outer_term_2

        return -potential

    @partial(jax.jit, static_argnums=0)
    def evaluate_and_gradient(self, samples):
        def eval_squeeze(sample): return self.evaluate(sample).reshape()

        value, gradient = jax.vmap(jax.jit(jax.value_and_grad(eval_squeeze)), in_axes=0)(samples)

        return value, gradient
