import jax
import jax.numpy as jnp
import pytest

from jax_tqdm import PBar, loop_tqdm, scan_tqdm


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


@pytest.mark.parametrize("print_rate", [None, 1, 10])
def test_vmap_w_scan(print_rate):
    n = 10_000
    n_maps = 5

    @scan_tqdm(n, print_rate=print_rate)
    def step(carry, _):
        return carry + 1, carry + 1

    @jax.jit
    def inner(i):
        init = PBar(id=i, carry=0)
        final, _all_numbers = jax.lax.scan(step, init, jnp.arange(n))
        return (
            final.carry,
            _all_numbers,
        )

    last_numbers, all_numbers = jax.vmap(inner)(jax.numpy.arange(n_maps))

    assert jnp.array_equal(last_numbers, jnp.full((n_maps,), n))
    assert all_numbers.shape == (n_maps, 10_000)
    assert jnp.array_equal(all_numbers, jnp.tile(1 + jnp.arange(n), (n_maps, 1)))


@pytest.mark.parametrize("print_rate", [None, 1, 10])
def test_vmap_w_loop(print_rate):
    n = 10_000
    n_maps = 5

    @loop_tqdm(n, print_rate=10)
    def step(i, val):
        return val + 1

    @jax.jit
    def inner(i):
        init = PBar(id=i, carry=0)
        result = jax.lax.fori_loop(0, n, step, init)
        return result.carry

    last_numbers = jax.vmap(inner)(jax.numpy.arange(n_maps))

    assert jnp.array_equal(last_numbers, jnp.full((n_maps,), n))


def test_vmap_w_position_keyword():
    n = 10_000
    n_maps = 5

    @scan_tqdm(n, position=2)
    def step(carry, _):
        return carry + 1, carry + 1

    @jax.jit
    def inner(i):
        init = PBar(id=i, carry=0)
        final, _all_numbers = jax.lax.scan(step, init, jnp.arange(n))
        return (
            final.carry,
            _all_numbers,
        )

    last_numbers, all_numbers = jax.vmap(inner)(jax.numpy.arange(n_maps))

    assert jnp.array_equal(last_numbers, jnp.full((n_maps,), n))
    assert all_numbers.shape == (n_maps, 10_000)
    assert jnp.array_equal(all_numbers, jnp.tile(1 + jnp.arange(n), (n_maps, 1)))
