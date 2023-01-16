import typing

import jax
from jax.experimental import host_callback
from tqdm.auto import tqdm


def progress_bar_scan(n: int, message: typing.Optional[str] = None) -> typing.Callable:
    """
    Progress bar for a JAX scan

    Parameters
    ----------
    n : int
        Number of scan steps/iterations.
    message : str
        Optional string to prepend to tqdm progress bar.

    Returns
    -------
    typing.Callable:
        Progress bar wrapping function.
    """
    if message is None:
        message = f"Running for {n:,} iterations"
    tqdm_bars = {}

    if n > 20:
        print_rate = int(n / 20)
    else:
        print_rate = 1
    remainder = n % print_rate

    def _define_tqdm(arg, transform):
        tqdm_bars[0] = tqdm(range(n))
        tqdm_bars[0].set_description(message, refresh=False)

    def _update_tqdm(arg, transform):
        tqdm_bars[0].update(arg)

    def _update_progress_bar(iter_num):
        "Updates tqdm progress bar of a JAX scan or loop"
        _ = jax.jax.lax.cond(
            iter_num == 0,
            lambda _: host_callback.id_tap(_define_tqdm, None, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            (iter_num % print_rate == 0) & (iter_num != n - remainder),
            lambda _: host_callback.id_tap(_update_tqdm, print_rate, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm by `remainder`
            iter_num == n - remainder,
            lambda _: host_callback.id_tap(_update_tqdm, remainder, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

    def _close_tqdm(arg, transform):
        tqdm_bars[0].close()

    def close_tqdm(result, iter_num):
        return jax.lax.cond(
            iter_num == n - 1,
            lambda _: host_callback.id_tap(_close_tqdm, None, result=result),
            lambda _: result,
            operand=None,
        )

    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `jax.lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(n)`,
        or be looping over a tuple who's first element is `np.arange(n)`
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x
            _update_progress_bar(iter_num)
            result = func(carry, x)
            return close_tqdm(result, iter_num)

        return wrapper_progress_bar

    return _progress_bar_scan
