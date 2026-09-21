"""The field radius is the one geometry input a client and an operator can both
set, so it is bounded in two places that must not drift apart.

``cropup.config`` sits below every other package and cannot import
``cropup.dialog.slots`` without a cycle (SPEC section 2's import rule), so the
ceiling is declared twice. These tests are what keeps the two spellings honest.
"""

from __future__ import annotations

import pytest

from cropup import config as config_mod
from cropup.dialog import slots as slots_mod
from cropup.errors import ConfigError


def test_the_two_ceilings_are_the_same_number():
    """If this fails, one of the two was raised and the other was forgotten."""
    assert config_mod.MAX_FIELD_RADIUS_M == slots_mod.MAX_FIELD_RADIUS_M


def test_the_default_is_inside_the_bounds_both_layers_enforce():
    assert slots_mod.MIN_FIELD_RADIUS_M <= config_mod.DEFAULT_FIELD_RADIUS_M
    assert config_mod.DEFAULT_FIELD_RADIUS_M <= config_mod.MAX_FIELD_RADIUS_M


@pytest.mark.parametrize(
    "radius",
    [0.0, -1.0, float("inf"), float("nan"), config_mod.MAX_FIELD_RADIUS_M + 1.0],
    ids=["zero", "negative", "infinite", "nan", "above-ceiling"],
)
def test_a_bad_configured_radius_fails_at_startup(monkeypatch, radius):
    """A mistyped deployment variable must fail loudly, not measure a district.

    The value goes straight into ``ee.Geometry.Point(...).buffer()``, so an
    unbounded one would quietly report a whole district as the farmer's field.
    """
    monkeypatch.setenv("CROPUP_FIELD_RADIUS_M", repr(radius))
    with pytest.raises(ConfigError, match="CROPUP_FIELD_RADIUS_M"):
        config_mod.Settings.from_env()


def test_the_ceiling_itself_is_accepted():
    """The bound is inclusive; only *past* it is an error."""
    settings = config_mod.Settings(field_radius_m=config_mod.MAX_FIELD_RADIUS_M)
    assert settings.field_radius_m == config_mod.MAX_FIELD_RADIUS_M
