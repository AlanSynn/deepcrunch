import logging
from time import perf_counter
from typing import Any, Callable, Optional


def log_elapsed_time(customized_msg: Optional[str] = "") -> Callable:
    """Get the elapsed time for decorated functions.

    Args:
        customized_msg (string, optional): The parameter passed to decorator. Defaults to None.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = perf_counter()
            # raise ValueError(*args, **kwargs)
            result = func(*args, **kwargs)
            end = perf_counter()
            elapsed_time = round((end - start) * 1000, 2)
            message = customized_msg if customized_msg else func.__qualname__
            logging.getLogger("deepcrunch").info(
                f"{message} elapsed time: {elapsed_time} ms"
            )
            return result

        return wrapper

    return decorator
