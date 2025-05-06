from typing import Any, Callable, Optional, TypeVar

from .base import PBar, build_tqdm

C = TypeVar("C")
X = TypeVar("X", int, tuple[int, Any])
Y = TypeVar("Y")
Z = TypeVar("Z")

ScanFn = Callable[[C, X], tuple[C, Y]]
WrappedScanFn = ScanFn | Callable[[PBar[C], X], tuple[PBar[C], Y]]


def scan_tqdm(
    n: int,
    print_rate: Optional[int] = None,
    tqdm_type: str = "auto",
    **kwargs: Any,
) -> Callable[[ScanFn], WrappedScanFn]:
    """
    tqdm progress bar for a JAX scan

    Parameters
    ----------
    n : int
        Number of scan steps/iterations.
    print_rate : int
        Optional integer rate at which the progress bar will be updated,
        by default the print rate will 1/20th of the total number of steps.
    tqdm_type: str
        Type of progress-bar, should be one of "auto", "std", or "notebook".
    **kwargs
        Extra keyword arguments to pass to tqdm.

    Returns
    -------
    typing.Callable:
        Progress bar wrapping function.
    """

    update_progress_bar, close_tqdm = build_tqdm(n, print_rate, tqdm_type, **kwargs)

    def _scan_tqdm(func: ScanFn) -> WrappedScanFn:
        """Decorator that adds a tqdm progress bar to `body_fun` used in `jax.lax.scan`.
        Note that `body_fun` must either be looping over `jnp.arange(n)`,
        or be looping over a tuple who's first element is `jnp.arange(n)`
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry: Z, x: X) -> tuple[Z, Y]:
            if isinstance(x, tuple):
                iter_num, *_ = x
            else:
                iter_num = x

            if isinstance(carry, PBar):
                bar_id = carry.id
                carry_ = carry.carry
                carry_, x = update_progress_bar((carry_, x), iter_num, bar_id)
                result = func(carry_, x)
                result = (PBar(id=bar_id, carry=result[0]), result[1])
                return close_tqdm(result, iter_num, bar_id)
            else:
                carry, x = update_progress_bar((carry, x), iter_num, 0)
                result = func(carry, x)
                return close_tqdm(result, iter_num, 0)

        return wrapper_progress_bar

    return _scan_tqdm
