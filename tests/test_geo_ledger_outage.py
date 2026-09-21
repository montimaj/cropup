"""SPEC section 4.1 gap accounting on SPEC section 10's null-adapter path.

The coverage hole this file closes was real and survived a green suite.
``geo/soil.get_soil`` and ``geo/climate.get_climate`` each caught
``EarthEngineUnavailable``, built the full set of ``Missing`` values -- and
returned them to the caller **without putting them in the Ledger**. So with
Earth Engine down, all 20 soil quantities and all 32 climate quantities
(including ``koppen_code``, the exact value SPEC section 2.1 bug 1 says must
never be fabricated) were reported to the farmer as absent while the turn's
accounting said it had measured nothing and had no gaps either.

A quantity that is missing from the ledger reads as *never asked for* rather
than *could not be measured*, and SPEC section 4.1 asks a turn to name what it
failed to measure. That distinction is the whole of the null-adapter path.

Why it was not caught: ``tests/test_null_adapter.py`` drives the outage through
HTTP, where an Earth-Engine intent is refused with a 503 **before** any geo leg
runs, so nothing there ever sees the ledger the geo layer produces. These tests
call the geo entrypoints directly.

Every test here takes ``ee_tripwire``, so "accidentally reached Earth Engine"
is an immediate failure rather than a silent network call on a machine that
happens to have credentials.
"""

from __future__ import annotations

from collections import Counter

import pytest

from cropup.evidence import Fact, Ledger, Missing
from cropup.geo import climate, context, soil, suitability, thermal, vegetation, water


ARUSHA = (-3.38, 36.68)


def _ledger_of(result, ledger: Ledger) -> Ledger:
    """The geo layer has two return shapes; the ledger is the same either way."""
    return result if isinstance(result, Ledger) else ledger


#: Every geo entrypoint SPEC section 2 lists, called the way an orchestrator
#: calls it. ``use_cache=False`` where it exists: a cached answer is a
#: different code path (it has its own ``ledger.extend``) and would hide this
#: one behind whatever an earlier test left behind.
GEO_ENTRYPOINTS = {
    "soil.get_soil": lambda ledger: soil.get_soil(*ARUSHA, ledger=ledger, use_cache=False),
    "climate.get_climate": lambda ledger: climate.get_climate(
        *ARUSHA, ledger=ledger, use_cache=False
    ),
    "context.get_field_context": lambda ledger: context.get_field_context(*ARUSHA, ledger=ledger),
    "thermal.get_land_surface_temperature": lambda ledger: (
        thermal.get_land_surface_temperature(*ARUSHA, ledger=ledger)
    ),
    "vegetation.get_vegetation": lambda ledger: vegetation.get_vegetation(*ARUSHA, ledger=ledger),
    "water.get_water_status": lambda ledger: water.get_water_status(*ARUSHA, ledger=ledger),
    "suitability.get_crop_suitability": lambda ledger: suitability.get_crop_suitability(
        *ARUSHA, "Maize", ledger=ledger
    ),
}


# --------------------------------------------------------------------------
# the invariant that would have caught this, and catches the next sibling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(GEO_ENTRYPOINTS))
def test_every_geo_leg_names_its_gaps_in_the_ledger_when_earth_engine_is_down(
    name, ee_tripwire
):
    """One test over all seven legs. The bug was a missing ``ledger.extend`` in
    one ``except`` branch; the identical omission sat in a second module, and
    nothing structural stops a third."""
    ledger = Ledger(turn=name)
    result = _ledger_of(GEO_ENTRYPOINTS[name](ledger), ledger)

    assert len(result.gaps) > 0, f"{name} measured nothing and reported no gap either"
    assert result.facts == (), f"{name} produced a Fact with Earth Engine down"
    quantities = [gap.quantity for gap in result.gaps]
    duplicated = sorted(q for q, n in Counter(quantities).items() if n > 1)
    assert duplicated == [], f"{name} ledgered {duplicated} more than once"


@pytest.mark.parametrize("name", sorted(GEO_ENTRYPOINTS))
def test_every_ledgered_gap_carries_a_named_reason_not_a_bare_absence(name, ee_tripwire):
    """SPEC section 4.1: a value that is absent is *named*, with the reason and
    the chain that was tried. "Nothing here" is not an answer a farmer can act
    on; "Earth Engine unavailable" is."""
    ledger = Ledger(turn=name)
    result = _ledger_of(GEO_ENTRYPOINTS[name](ledger), ledger)
    for gap in result.gaps:
        assert isinstance(gap, Missing)
        assert gap.quantity
        assert gap.reason, f"{name}/{gap.quantity} has no reason"
        assert gap.measured is False


@pytest.mark.parametrize("name", sorted(GEO_ENTRYPOINTS))
def test_no_geo_leg_invents_a_value_while_the_instrument_is_down(name, ee_tripwire):
    """The firewall, stated at the layer that reads the instruments."""
    ledger = Ledger(turn=name)
    result = _ledger_of(GEO_ENTRYPOINTS[name](ledger), ledger)
    assert not any(isinstance(entry, Fact) for entry in result.entries)
    assert result.degradation()["degraded"] is True


# --------------------------------------------------------------------------
# the two legs the bug was actually in, checked against what they RETURN
# --------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", ["soil", "climate"])
def test_everything_handed_back_to_the_caller_also_reached_the_ledger(
    module_name, ee_tripwire
):
    """The precise shape of the bug: 20 soil (and 32 climate) quantities came
    back to the caller as ``Missing`` while the ledger held **zero** gaps, so
    the turn's own accounting disagreed with the answer it was rendering."""
    ledger = Ledger(turn=module_name)
    if module_name == "soil":
        returned = soil.get_soil(*ARUSHA, ledger=ledger, use_cache=False)
    else:
        returned = climate.get_climate(*ARUSHA, ledger=ledger, use_cache=False)

    assert returned, "the leg returned nothing at all"
    assert all(isinstance(value, Missing) for value in returned.values())
    assert {gap.quantity for gap in ledger.gaps} == set(returned)
    assert ledger.facts == ()
    assert len(ledger.gaps) == len(returned)


def test_the_koppen_class_is_a_named_gap_and_never_a_default(ee_tripwire):
    """SPEC section 2.1 bug 1 by name: the vendored ``climate.py`` swallowed the
    missing-raster error and returned a fabricated ``Cfb``/"Temperate" for
    Arusha. Unknown must be UNKNOWN -- and must be *in the ledger* saying so."""
    ledger = Ledger(turn="koppen")
    returned = climate.get_climate(*ARUSHA, ledger=ledger, use_cache=False)

    for quantity in ("koppen_code", "koppen_name"):
        assert isinstance(returned[quantity], Missing)
        assert ledger.get(quantity) is not None, f"{quantity} never reached the ledger"
        with pytest.raises(Exception):
            ledger.require(quantity)

    blob = str(ledger.as_dict())
    assert "Cfb" not in blob
    assert "Temperate" not in blob


def test_a_degraded_turn_can_say_how_much_of_it_was_degraded(ee_tripwire):
    """What the accounting is *for*: the capability strip and SPEC section 4.1's
    "name what you failed to measure" both read this, and with the gaps absent
    it read as a clean run."""
    ledger = Ledger(turn="whole-field")
    soil.get_soil(*ARUSHA, ledger=ledger, use_cache=False)
    climate.get_climate(*ARUSHA, ledger=ledger, use_cache=False)
    context.get_field_context(*ARUSHA, ledger=ledger)

    degradation = ledger.degradation()
    assert degradation["degraded"] is True
    assert degradation["fact_count"] == 0
    assert degradation["gap_count"] == len(ledger.gaps) > 50
    # Named, one by one, with the reason each one failed for -- not a count.
    by_reason = degradation["gaps_by_reason"]
    assert set(by_reason) == {"source_failed"}
    assert "soil_ph" in by_reason["source_failed"]
    assert "koppen_code" in by_reason["source_failed"]
    assert "is_cropland" in by_reason["source_failed"]
