"""Alien Ink package identity."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any


try:
    __version__ = version("alien-ink")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Multi-line brand banner (starfield + wordmark). Logged at run start.
stars = rf"""
-------------------------------------------------------------------------------
 ..   . *      .     .   * *. .        . .  .     .  *           `         .
*         . .         ` `  *   . . .            . **              .*    * .  `
     .           . .            ..     *  .               *    .   .
. ..` .              *.  **    . * *  . . .     *   ` . `  .     .. .  . . ..
    _    _ _              ___       _
   / \  | (_) ___ _ __   |_ _|_ __ | | __
  / _ \ | | |/ _ \ '_ \   | || '_ \| |/ /
 / ___ \| | |  __/ | | |  | || | | |   <
/_/   \_\_|_|\___|_| |_| |___|_| |_|_|\_\

*         . .         ` `  *   . . .            . **              .*    * .  `
                    . ..` .              *.  **    . * *  . . .     *   `
. `  .     .. .  . . ..     .       `.              . *          *   .      .
          *         ..  .  `    `*   `    *   `.   .    `  .. .    . *   ..
Version: {__version__}
-------------------------------------------------------------------------------
"""

# Alias for the run header
HEADER = stars

# Lazily expose selected submodules so ``import alien_ink; alien_ink.device``
# works without pulling torch into every import of the package root.
_LAZY_SUBMODULES = frozenset({"device"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_SUBMODULES})
