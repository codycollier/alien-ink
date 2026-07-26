"""Alien Ink package identity."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("alien-ink")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Multi-line brand banner (starfield + wordmark). Logged at experiment start.
HEADER = r"""
 ..   . *      .     .   * *. .        . .  .     .  *           `         .
*         . .         ` `  *   . . .            . **              .*    * .  `
     .           . .            ..     *  .               *    .   .

     _    _     _              ___       _
    / \  | |   (_) ___ _ __   |_ _|_ __ | | __
   / _ \ | |   | |/ _ \ '_ \   | || '_ \| |/ /
  / ___ \| |___| |  __/ | | |  | || | | |   <
 /_/   \_\_____|_|\___|_| |_| |___|_| |_|_|\_\

                    . ..` .              *.  **    . * *  . . .     *   `
. `  .     .. .  . . ..     .       `.              . *          *   .      .
          *         ..  .  `    `*   `    *   `.   .    `  .. .    . *   ..
"""

# Backward-compatible alias for the starfield banner.
stars = HEADER
