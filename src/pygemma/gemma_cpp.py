import ctypes
import functools
import os
import pathlib
import sys
from typing import Any, Callable, List, NewType, Optional, TypeVar


def _load_shared_library(lib_base_name: str):
    # Construct the paths to the possible shared library names
    _base_path = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

    print(_base_path)

    # Searching for the library in the current directory under the name "lib{lib_base_name}.{ext}"
    _lib_paths: List[pathlib.Path] = []
    # Determine the file extension based on the platform
    if sys.platform.startswith("linux"):
        _lib_paths += [
            _base_path / f"lib{lib_base_name}.so",
        ]
    elif sys.platform == "darwin":
        _lib_paths += [
            _base_path / f"lib{lib_base_name}.so",
            _base_path / f"lib{lib_base_name}.dylib",
        ]
    elif sys.platform == "win32":
        _lib_paths += [
            _base_path / f"{lib_base_name}.dll",
            _base_path / f"lib{lib_base_name}.dll",
        ]
    else:
        raise RuntimeError("Unsupported platform")

    cdll_args = dict()  # type: ignore

    # Try to load the shared library, handling potential errors
    for _lib_path in _lib_paths:
        if _lib_path.exists():
            try:
                return ctypes.CDLL(str(_lib_path), **cdll_args)  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Failed to load shared library '{_lib_path}': {e}")

    raise FileNotFoundError(
        f"Shared library with base name '{lib_base_name}' not found at paths: {_lib_paths}"
    )


# Specify the base name of the shared library to load
_lib_base_name = "pygemma"

# Load the library
_lib = _load_shared_library(_lib_base_name)


F = TypeVar("F", bound=Callable[..., Any])


def ctypes_function_for_shared_library(lib: ctypes.CDLL):
    def ctypes_function(
        name: str, argtypes: List[Any], restype: Any, enabled: bool = True
    ):
        def decorator(f: F) -> F:
            if enabled:
                func = getattr(lib, name)
                func.argtypes = argtypes
                func.restype = restype
                functools.wraps(f)(func)
                return func
            else:
                return f

        return decorator

    return ctypes_function


ctypes_function = ctypes_function_for_shared_library(_lib)

# struct llama_model;
gemma_model_p = NewType("gemma_model_p", int)
gemma_model_p_ctypes = ctypes.c_void_p

# struct gemma_tokenizer;
gemma_tokenizer_p = NewType("gemma_tokenizer_p", int)
gemma_tokenizer_p_ctypes = ctypes.c_void_p


@ctypes_function(
    "loadGemmaModel",
    [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p],
    gemma_model_p_ctypes,
)
def load_gemma_model(
    tokenizer_path: bytes,
    compressed_weights_path: bytes,
    model_type: bytes,
) -> Optional[gemma_model_p]: ...
