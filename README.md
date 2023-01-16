# JAX-tqdm

Add a progress bar to your JAX scans and loops.

The code is explained in this [blog post](https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/).

## Example usage

```python
from jax_tqdm import progress_bar_scan
from jax import lax
import jax.numpy as jnp

n = 10_000

@progress_bar_scan(n)
def step(carry, x):
    return carry + 1, carry + 1

_, all_numbers = lax.scan(step, 0, jnp.arange(n))
```
