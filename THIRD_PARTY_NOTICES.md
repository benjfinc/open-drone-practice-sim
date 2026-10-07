# Third-party notices

## Betaflight

`fpvsim/rates.py` adapts the Betaflight and Actual rate-curve equations,
deadband behavior, and throttle-curve equations from Betaflight's
`src/main/fc/rc.c`. The corresponding reference tests in
`tests/test_rates.py` independently transcribe those equations for comparison.

Betaflight is copyright its contributors and distributed under the GNU General
Public License, version 3 or (at the recipient's option) any later version.

- Project: <https://github.com/betaflight/betaflight>
- Referenced source: <https://github.com/betaflight/betaflight/blob/master/src/main/fc/rc.c>
- License notice: <https://github.com/betaflight/betaflight/blob/master/DEFAULT_LICENSE.md>
