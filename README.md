# JAX-tqdm

Add a tqdm progress bar to your JAX scans and loops.

The code is explained in this [blog post](https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/).

## Example usage

### in `jax.lax.scan`

```python
from jax_tqdm import progress_bar_scan
from jax import lax
import jax.numpy as jnp

n = 10_000

@progress_bar_scan(n)
def step(carry, x):
    return carry + 1, carry + 1

last_number, all_numbers = lax.scan(step, 0, jnp.arange(n))
```


### in `jax.lax.fori_loop`

```python
from jax_tqdm import progress_bar_fori_loop
from jax import lax

n = 10_000

@progress_bar_fori_loop(n)
def step(i, val): 
    return val + 1

last_number = lax.fori_loop(0, n, step, 0)
```
