import typing

import jax
import tqdm.auto
import tqdm.notebook
import tqdm.std
from jax.debug import callback


def scan_tqdm(
    n: int,
    print_rate: typing.Optional[int] = None,
    tqdm_type: str = "auto",
    **kwargs,
) -> typing.Callable:
    """
    tqdm progress bar for a JAX scan

    Parameters
    ----------
    n : int
        Number of scan steps/iterations.
    print_rate : int
        Optional integer rate at which the progress bar will be updated,
        by default the print rate will 1/20th of the total number of steps.
    **kwargs
        Extra keyword arguments to pass to tqdm.

    Returns
    -------
    typing.Callable:
        Progress bar wrapping function.
    """

    _update_progress_bar, close_tqdm = build_tqdm(n, print_rate, tqdm_type, **kwargs)

    def _scan_tqdm(func):
        """Decorator that adds a tqdm progress bar to `body_fun` used in `jax.lax.scan`.
        Note that `body_fun` must either be looping over `jnp.arange(n)`,
        or be looping over a tuple who's first element is `jnp.arange(n)`
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

    return _scan_tqdm


def loop_tqdm(
    n: int,
    print_rate: typing.Optional[int] = None,
    tqdm_type: str = "auto",
    **kwargs,
) -> typing.Callable:
    """
    tqdm progress bar for a JAX fori_loop

    Parameters
    ----------
    n : int
        Number of iterations.
    print_rate: int
        Optional integer rate at which the progress bar will be updated,
        by default the print rate will 1/20th of the total number of steps.
    **kwargs
        Extra keyword arguments to pass to tqdm.

    Returns
    -------
    typing.Callable:
        Progress bar wrapping function.
    """

    _update_progress_bar, close_tqdm = build_tqdm(n, print_rate, tqdm_type, **kwargs)

    def _loop_tqdm(func):
        """
        Decorator that adds a tqdm progress bar to `body_fun`
        used in `jax.lax.fori_loop`.
        """

        def wrapper_progress_bar(i, val):
            _update_progress_bar(i)
            result = func(i, val)
            return close_tqdm(result, i)

        return wrapper_progress_bar

    return _loop_tqdm


def build_tqdm(
    n: int,
    print_rate: typing.Optional[int],
    tqdm_type: str,
    **kwargs,
) -> typing.Tuple[typing.Callable, typing.Callable]:
    """
    Build the tqdm progress bar on the host
    """

    if tqdm_type not in ("auto", "std", "notebook"):
        raise ValueError(
            'tqdm_type should be one of "auto", "std", or "notebook" '
            f'but got "{tqdm_type}"'
        )
    pbar = getattr(tqdm, tqdm_type).tqdm

    desc = kwargs.pop("desc", f"Running for {n:,} iterations")
    message = kwargs.pop("message", desc)
    for kwarg in ("total", "mininterval", "maxinterval", "miniters"):
        kwargs.pop(kwarg, None)

    tqdm_bars = {}

    if print_rate is None:
        if n > 20:
            print_rate = int(n / 20)
        else:
            print_rate = 1
    else:
        if print_rate < 1:
            raise ValueError(f"Print rate should be > 0 got {print_rate}")
        elif print_rate > n:
            raise ValueError(
                "Print rate should be less than the "
                f"number of steps {n}, got {print_rate}"
            )

    remainder = n % print_rate

    def _define_tqdm(arg, transform):
        tqdm_bars[0] = pbar(range(n), **kwargs)
        tqdm_bars[0].set_description(message, refresh=False)

    def _update_tqdm(arg, transform):
        tqdm_bars[0].update(int(arg))

    def _update_progress_bar(iter_num):
        "Updates tqdm from a JAX scan or loop"
        _ = jax.lax.cond(
            iter_num == 0,
            lambda _: callback(_define_tqdm, None, None, ordered=True),
            lambda _: None,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            (iter_num % print_rate == 0) & (iter_num != n - remainder),
            lambda _: callback(_update_tqdm, print_rate, None, ordered=True),
            lambda _: None,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm by `remainder`
            iter_num == n - remainder,
            lambda _: callback(_update_tqdm, remainder, None, ordered=True),
            lambda _: None,
            operand=None,
        )

    def _close_tqdm(arg, transform):
        tqdm_bars[0].close()

    def close_tqdm(result, iter_num):
        _ = jax.lax.cond(
            iter_num == n - 1,
            lambda _: callback(_close_tqdm, None, None, ordered=True),
            lambda _: None,
            operand=None,
        )
        return result

    return _update_progress_bar, close_tqdm
