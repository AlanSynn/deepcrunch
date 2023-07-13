import importlib
from pathlib import Path


def is_valid_path(path: str) -> bool:
    """Check if the given string is path and valid."""
    if Path(path).is_file():
        return True
    else:
        return False


def singleton(cls):
    """Singleton in decorater"""
    instances = {}

    def _singleton(*args, **kw):
        """Create a singleton object."""
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return _singleton


class LazyImport:
    """Lazy import python module until used."""

    def __init__(self, module_name: str):
        """Initialize LazyImport object.

        Args:
           module_name (str): The name of the module to be imported later.
        """
        self.module_name = module_name
        self.module = None

    def __getattr__(self, name: str):
        """Get the attributes of the module by name."""
        if not self.module:
            self.module = self._load_module(self.module_name)
        try:
            return getattr(self.module, name)
        except AttributeError:
            raise ImportError(f"Cannot import {name} from {self.module_name}")

    def _load_module(self, module_name: str):
        """Load and return a module."""
        if importlib.util.find_spec(module_name) is not None:
            return importlib.import_module(module_name)
        else:
            raise ImportError(f"Cannot find module {module_name}")

    def __call__(self, *args, **kwargs):
        """Call the function in that module."""
        function_name = self.module_name.split(".")[-1]
        try:
            function = getattr(self.module, function_name)
            return function(*args, **kwargs)
        except AttributeError:
            raise ImportError(f"Cannot import {function_name} from {self.module_name}")
