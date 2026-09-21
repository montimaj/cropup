"""Small constructors the unit tests share.

``fact()`` fills in the provenance every :class:`~cropup.evidence.Fact`
requires, so a test that is about something else does not have to restate an
instrument, a date and a resolution.
"""

from __future__ import annotations

import datetime as dt

from cropup.evidence import Fact

__all__ = ["fact", "OBSERVED_ON"]

OBSERVED_ON = dt.date(2025, 8, 1)


def fact(quantity: str, value, unit: str = "index", **kwargs) -> Fact:
    kwargs.setdefault("source_asset", "TEST/ASSET")
    kwargs.setdefault("observed_on", OBSERVED_ON)
    kwargs.setdefault("resolution_m", 10.0)
    return Fact(quantity=quantity, value=value, unit=unit, **kwargs)
