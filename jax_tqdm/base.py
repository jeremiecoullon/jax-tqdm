from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, TypeVar

import jax
import tqdm.auto
import tqdm.notebook
import tqdm.std
from jax.debug import callback

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

C = TypeVar("C")
A = TypeVar("A")
B = TypeVar("B")

UpdateProgressBar = Callable[[A, int, int], A]
CloseTQDM = Callable[[B, int, int], B]


@dataclass
class PBar(Generic[C]):
    id: int
    carry: C


def build_tqdm(
    n: int,
    print_rate: Optional[int],
    tqdm_type: str,
    **kwargs: Any,
) -> tuple[UpdateProgressBar, CloseTQDM]:
    """
    Build the tqdm progress bar on the host

    Parameters
    ----------
    n: int
        Number of updates
    print_rate: int
        Optional integer rate at which the progress bar will be updated,
        If ``None`` the print rate will 1/20th of the total number of steps.
    tqdm_type: str
        Type of progress-bar, should be one of "auto", "std", or "notebook".
    **kwargs
        Extra keyword arguments to pass to tqdm.
    """

    if tqdm_type not in ("auto", "std", "notebook"):
        raise ValueError(
            'tqdm_type should be one of "auto", "std", or "notebook" '
            f'but got "{tqdm_type}"'
        )
    pbar = getattr(tqdm, tqdm_type).tqdm

    desc = kwargs.pop("desc", f"Running for {n:,} iterations")
    message = kwargs.pop("message", desc)
    position_offset = kwargs.pop("position", 0)

    for kwarg in ("total", "mininterval", "maxinterval", "miniters"):
        kwargs.pop(kwarg, None)

    tqdm_bars = dict()

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
    remainder = remainder if remainder > 0 else print_rate

    def _define_tqdm(bar_id: int) -> None:
        bar_id = int(bar_id)
        tqdm_bars[bar_id] = pbar(
            total=n,
            position=bar_id + position_offset,
            desc=message,
            **kwargs,
        )

    def _update_tqdm(bar_id: int) -> None:
        tqdm_bars[int(bar_id)].update(print_rate)

    def _close_tqdm(bar_id: int) -> None:
        _pbar = tqdm_bars.pop(int(bar_id))
        _pbar.update(remainder)
        _pbar.clear()
        _pbar.close()

    def update_progress_bar(carry: A, iter_num: int, bar_id: int) -> A:
        """Updates tqdm from a JAX scan or loop"""

        def _inner_init(_i: int, _carry: A) -> A:
            callback(_define_tqdm, bar_id, ordered=True)
            return _carry

        def _inner_update(i: int, _carry: A) -> A:
            _ = jax.lax.cond(
                i % print_rate == 0,
                lambda: callback(_update_tqdm, bar_id, ordered=True),
                lambda: None,
            )
            return _carry

        carry = jax.lax.cond(
            iter_num == 0,
            _inner_init,
            _inner_update,
            iter_num,
            carry,
        )

        return carry

    def close_tqdm(result: B, iter_num: int, bar_id: int) -> B:
        def _inner_close(_result: B) -> B:
            callback(_close_tqdm, bar_id, ordered=True)
            return _result

        result = jax.lax.cond(iter_num + 1 == n, _inner_close, lambda r: r, result)
        return result

    return update_progress_bar, close_tqdm
