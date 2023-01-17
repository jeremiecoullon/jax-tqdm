import jax.numpy as jnp
from jax import lax

from jax_tqdm import progress_bar_scan


def test_readme_example():
    """Just test that README example runs correctly"""

    n = 10_000

    @progress_bar_scan(n)
    def step(carry, x):
        return carry + 1, carry + 1

    _, all_numbers = lax.scan(step, 0, jnp.arange(n))
