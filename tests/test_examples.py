import jax.numpy as jnp
from jax import lax

from jax_tqdm import loop_tqdm, scan_tqdm


def test_readme_scan_example():
    """Just test that README scan example runs correctly"""

    n = 10_000

    @scan_tqdm(n)
    def step(carry, x):
        return carry + 1, carry + 1

    last_number, all_numbers = lax.scan(step, 0, jnp.arange(n))

    assert int(last_number) == n
    assert jnp.array_equal(all_numbers, 1 + jnp.arange(n))


def test_readme_fori_loop_example():
    """Just test that README loop example runs correctly"""

    n = 10_000

    @loop_tqdm(n)
    def step(i, val):
        return val + 1

    last_number = lax.fori_loop(0, n, step, 0)

    assert int(last_number) == n
