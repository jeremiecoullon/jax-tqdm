# JAX-tqdm

Add a tqdm progress bar to your JAX scans and loops.

## Installation

Install with pip:

```bash
pip install jax-tqdm
```

## Example usage

### in `jax.lax.scan`

```python
from jax_tqdm import scan_tqdm
from jax import lax
import jax.numpy as jnp

n = 10_000

@scan_tqdm(n)
def step(carry, x):
    return carry + 1, carry + 1

last_number, all_numbers = lax.scan(step, 0, jnp.arange(n))
```

### in `jax.lax.fori_loop`

```python
from jax_tqdm import loop_tqdm
from jax import lax

n = 10_000

@loop_tqdm(n)
def step(i, val):
    return val + 1

last_number = lax.fori_loop(0, n, step, 0)
```

## Why JAX-tqdm?

JAX functions are [purely functional](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions), so side effects such as printing progress when running scans and loops are not allowed. However, the [host_callback module](https://jax.readthedocs.io/en/latest/jax.experimental.host_callback.html) has primitives for calling Python functions on the host from JAX code. This can be used to update a Python tqdm progress bar regularly during the computation. JAX-tqdm implements this for JAX scans and loops and is used by simply adding a decorator to the body of your update function (see [example usage](@example-usage)).

Furthermore, as the tqdm progress bar is only updated 20 times during the scan or loop, there is no performance penalty.

The code is explained in more detail in this [blog post](https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/).