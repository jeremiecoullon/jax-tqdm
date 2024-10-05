import jax
import jax.numpy as jnp
import pytest

from jax_tqdm import BarId, loop_tqdm, scan_tqdm


@pytest.mark.parametrize("print_rate", [None, 1, 10])
def test_readme_scan_example(print_rate):
    """Just test that README scan example runs correctly"""

    n = 10_000

    @scan_tqdm(n, print_rate=print_rate)
    def step(carry, x):
        return carry + 1, carry + 1

    last_number, all_numbers = jax.lax.scan(step, 0, jnp.arange(n))

    assert int(last_number) == n
    assert jnp.array_equal(all_numbers, 1 + jnp.arange(n))


@pytest.mark.parametrize("print_rate", [None, 1, 10])
def test_readme_fori_loop_example(print_rate):
    """Just test that README loop example runs correctly"""

    n = 10_000

    @loop_tqdm(n, print_rate=print_rate)
    def step(i, val):
        return val + 1

    last_number = jax.lax.fori_loop(0, n, step, 0)

    assert int(last_number) == n


def test_vmap_w_scan():
    n = 10_000

    @scan_tqdm(n, print_rate=10)
    def step(carry, _):
        return carry + 1, carry + 1

    @jax.jit
    def inner(i):
        init = BarId(i=i, carry=0)
        final, _all_numbers = jax.lax.scan(step, init, jnp.arange(n))
        return (
            final.carry,
            _all_numbers,
        )

    last_numbers, all_numbers = jax.vmap(inner)(jax.numpy.arange(5))

    assert int(last_numbers[0]) == n
    assert jnp.array_equal(all_numbers[0], 1 + jnp.arange(n))


def test_vmap_w_loop():
    n = 10_000

    @loop_tqdm(n, print_rate=10)
    def step(i, val):
        return val + 1

    @jax.jit
    def inner(i):
        init = BarId(i=i, carry=0)
        result = jax.lax.fori_loop(0, n, step, init)
        return result.carry

    last_numbers = jax.vmap(inner)(jax.numpy.arange(5))

    assert int(last_numbers[0]) == n
