import jax.numpy as jnp
from jax import lax

from jax_tqdm import progress_bar_scan, progress_bar_fori_loop


def test_readme_scan_example():
    """Just test that README scan example runs correctly"""

    n = 10_000

    @progress_bar_scan(n)
    def step(carry, x):
        return carry + 1, carry + 1

    last_number, all_numbers = lax.scan(step, 0, jnp.arange(n))


def test_readme_fori_loop_example():
    """Just test that README fori_loop example runs correctly"""

    n = 10_000

    @progress_bar_fori_loop(n)
    def step(i, val): 
        return val + 1

    last_number = lax.fori_loop(0, n, step, 0)
