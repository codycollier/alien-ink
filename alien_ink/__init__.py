"""Alien Ink package identity."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("alien-ink")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Multi-line brand banner (starfield + wordmark). Logged at experiment start.
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

# Alias for the experiment header
HEADER = stars
