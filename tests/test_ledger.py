"""SPEC section 4.1: ``Ledger`` gap accounting.

A value with no ``Fact`` cannot reach the renderer, and a value that is absent is
*named* rather than defaulted. ``require()`` is the boundary: it raises, it never
hands back a stand-in.
"""

from __future__ import annotations

import datetime as dt

import pytest

from cropup.errors import FabricationError, InvalidFactError, MissingProvenanceError
from cropup.evidence import (
    Fact,
    Ledger,
    Missing,
    MissingReason,
    SourceDescriptor,
    demote_if_stale,
    fact_from_reduce_region,
)

from helpers import fact


# --------------------------------------------------------------------------
# facts versus named gaps
# --------------------------------------------------------------------------


def test_facts_and_gaps_are_counted_separately(ledger):
    ledger.record(fact("ndvi", 0.62))
    ledger.record(fact("ndmi", 0.31))
    ledger.gap("soil_ph", MissingReason.OUT_OF_COVERAGE, ("polaris",), "POLARIS is US-only")

    assert [f.quantity for f in ledger.facts] == ["ndvi", "ndmi"]
    assert [m.quantity for m in ledger.gaps] == ["soil_ph"]
    assert len(ledger) == 3
    assert ledger.quantities() == ("ndvi", "ndmi", "soil_ph")
    assert ledger.has("ndvi") is True
    assert ledger.has("soil_ph") is False  # a gap is not a measurement


def test_a_gap_names_the_quantity_the_reason_and_the_chain_tried(ledger):
    missing = ledger.gap(
        "et0_forecast", MissingReason.OUT_OF_COVERAGE, ("fret/forecast/eto",), "CONUS only"
    )
    assert missing.quantity == "et0_forecast"
    assert missing.reason is MissingReason.OUT_OF_COVERAGE
    assert missing.chain_tried == ("fret/forecast/eto",)
    rendered = missing.render()
    assert "not available" in rendered
    assert "outside the data's coverage" in rendered
    assert "fret/forecast/eto" in rendered
    assert "CONUS only" in rendered
    # A named absence, not a hedged number: nothing here reads as a measurement.
    assert "0" not in rendered.replace("et0", "")


def test_an_undiagnosed_reason_is_refused(ledger):
    with pytest.raises(InvalidFactError) as caught:
        ledger.gap("soil_ph", "because")
    assert "reason must be one of" in str(caught.value)
    assert ledger.gaps == ()


def test_a_ledger_holds_only_fact_or_missing(ledger):
    for offender in (7.0, None, "pH 7.0", {"value": 7.0}):
        with pytest.raises(FabricationError):
            ledger.add(offender)
    with pytest.raises(FabricationError):
        ledger.record(Missing("soil_ph", MissingReason.MASKED))
    assert len(ledger) == 0


def test_a_measurement_is_never_undone_by_a_later_blanket_absence(ledger):
    """A leg that raised writes Missing for every quantity it owed, including
    ones another leg already measured. The measurement must survive."""
    measured = ledger.record(fact("ndvi", 0.62))
    ledger.gap("ndvi", MissingReason.SOURCE_FAILED, ("s2",), "the leg raised")
    assert ledger.get("ndvi") is measured
    assert ledger.fact("ndvi") is measured
    assert ledger.require("ndvi") is measured
    # ...and the gap is still on the ledger, so the degradation report sees it.
    assert [m.quantity for m in ledger.gaps] == ["ndvi"]


def test_get_falls_back_to_the_latest_missing_when_nothing_was_measured(ledger):
    ledger.gap("soil_ph", MissingReason.MASKED)
    ledger.gap("soil_ph", MissingReason.SOURCE_FAILED)
    entry = ledger.get("soil_ph")
    assert isinstance(entry, Missing)
    assert entry.reason is MissingReason.SOURCE_FAILED
    assert ledger.fact("soil_ph") is None


def test_get_returns_none_for_a_quantity_nobody_ever_mentioned(ledger):
    assert ledger.get("soil_ph") is None
    assert ledger.fact("soil_ph") is None


# --------------------------------------------------------------------------
# require(): raises rather than returning a default
# --------------------------------------------------------------------------


def test_require_raises_for_a_quantity_that_was_never_recorded(ledger):
    with pytest.raises(MissingProvenanceError) as caught:
        ledger.require("soil_ph", template="plant_health.soil")
    error = caught.value
    assert error.slot == "soil_ph"
    assert error.template == "plant_health.soil"
    assert "nothing was recorded" in error.detail
    assert "refusing to render an unmeasured value" in str(error)


def test_require_raises_for_a_named_gap_and_quotes_it(ledger):
    ledger.gap("soil_ph", MissingReason.MASKED, ("isda",), "no unmasked pixel")
    with pytest.raises(MissingProvenanceError) as caught:
        ledger.require("soil_ph")
    assert "the pixel is masked" in caught.value.detail


def test_require_has_no_default_parameter_at_all():
    """There is nowhere to put a fallback: the signature does not take one."""
    import inspect

    parameters = inspect.signature(Ledger.require).parameters
    assert list(parameters) == ["self", "quantity", "template"]
    assert parameters["template"].default == ""


# --------------------------------------------------------------------------
# Fact construction refuses non-measurements
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf"), "", "   "])
def test_a_non_measurement_cannot_become_a_fact(value):
    with pytest.raises(InvalidFactError):
        fact("soil_ph", value)


def test_a_fact_needs_the_instrument_that_produced_it():
    with pytest.raises(InvalidFactError):
        Fact("soil_ph", 6.2, "pH", "", dt.date(2025, 8, 1), 30.0)


def test_a_fact_needs_a_positive_resolution_or_none():
    with pytest.raises(InvalidFactError):
        fact("ndvi", 0.5, resolution_m=0.0)
    with pytest.raises(InvalidFactError):
        fact("ndvi", 0.5, resolution_m=-10.0)
    assert fact("ndvi", 0.5, resolution_m=None).resolution_m is None


def test_a_fact_carries_its_whole_instrument_into_provenance():
    measured = fact(
        "ndvi",
        0.62,
        source_asset="COPERNICUS/S2_SR_HARMONIZED",
        band="B8",
        scaling_applied="raw/10000",
    )
    provenance = measured.provenance()
    assert provenance["source_asset"] == "COPERNICUS/S2_SR_HARMONIZED"
    assert provenance["band"] == "B8"
    assert provenance["observed_on"] == "2025-08-01"
    assert provenance["resolution_m"] == 10.0
    assert provenance["scaling_applied"] == "raw/10000"
    assert provenance["chain_label"] == "primary"
    assert provenance["measured"] is True


def test_freshness_is_tri_state_and_unknown_is_not_fresh():
    undated = fact("soil_ph", 6.2, observed_on=None)
    assert undated.age_days is None
    assert undated.is_stale is None  # not False
    old = fact(
        "evaporative_stress_index",
        -1.2,
        observed_on=dt.date.today() - dt.timedelta(days=500),
        stale_after_days=45,
    )
    assert old.is_stale is True


def test_a_reading_past_its_freshness_limit_is_demoted_to_a_named_absence():
    """SPEC 3.3: ESI at Arusha came back over a year old."""
    old = fact(
        "evaporative_stress_index",
        -1.2,
        observed_on=dt.date.today() - dt.timedelta(days=500),
        stale_after_days=45,
        source_asset="projects/climate-engine/esi/4wk",
    )
    demoted = demote_if_stale(old)
    assert isinstance(demoted, Missing)
    assert demoted.reason is MissingReason.STALE
    assert demoted.age_days == 500
    assert "projects/climate-engine/esi/4wk" in demoted.chain_tried


def test_a_source_with_no_declared_limit_is_not_demoted_on_a_guess():
    old = fact("ndvi", 0.4, observed_on=dt.date.today() - dt.timedelta(days=500))
    assert demote_if_stale(old) is old
    assert "no freshness limit is defined" in (old.staleness_note() or "")


# --------------------------------------------------------------------------
# the reduceRegion boundary
# --------------------------------------------------------------------------


STATIC = SourceDescriptor(
    quantity="soil_ph",
    asset_id="ISDASOIL/Africa/v1/ph",
    unit="pH",
    band="mean_0_20",
    resolution_m=30.0,
    transform=lambda x: x / 10.0,
    valid_range=(3.0, 10.0),
    is_static=True,
)


@pytest.mark.parametrize(
    "result,reason",
    [
        ({"mean_0_20": None}, MissingReason.MASKED),
        ({}, MissingReason.MASKED),
        ({"mean_0_20": 250.0}, MissingReason.MASKED),
        (None, MissingReason.SOURCE_FAILED),
        ("not a dictionary", MissingReason.SOURCE_FAILED),
    ],
)
def test_a_masked_pixel_is_never_zero(result, reason):
    out = fact_from_reduce_region(result, STATIC, observed_on=None)
    assert isinstance(out, Missing)
    assert out.reason is reason
    assert out.render()


def test_a_real_pixel_becomes_a_fact_with_the_transform_applied():
    out = fact_from_reduce_region({"mean_0_20": 62}, STATIC, observed_on=None)
    assert isinstance(out, Fact)
    assert out.value == pytest.approx(6.2)
    assert out.unit == "pH"
    assert out.source_asset == "ISDASOIL/Africa/v1/ph"


def test_a_missing_observation_date_is_a_decision_not_an_omission():
    dated = SourceDescriptor(quantity="q", asset_id="A", unit="", band="b", is_static=False)
    with pytest.raises(InvalidFactError) as caught:
        fact_from_reduce_region({"b": 1.0}, dated, observed_on=None)
    assert "observed_on is required" in str(caught.value)


# --------------------------------------------------------------------------
# the honesty report
# --------------------------------------------------------------------------


def test_degradation_reports_facts_gaps_fallbacks_and_stale(ledger):
    ledger.record(fact("ndvi", 0.62))
    ledger.record(fact("soil_ph", 6.2, chain_position=2, source_asset="soilgrids"))
    ledger.record(
        fact(
            "evaporative_stress_index",
            -1.2,
            observed_on=dt.date.today() - dt.timedelta(days=500),
            stale_after_days=45,
        )
    )
    ledger.gap("et0_forecast", MissingReason.OUT_OF_COVERAGE, ("fret",), "CONUS only")

    report = ledger.degradation()
    assert report["degraded"] is True
    assert report["fact_count"] == 3
    assert report["gap_count"] == 1
    assert report["gaps_by_reason"] == {"out_of_coverage": ["et0_forecast"]}
    assert [f["quantity"] for f in report["fallbacks"]] == ["soil_ph"]
    assert report["fallbacks"][0]["chain_label"] == "fallback #2"
    assert [s["quantity"] for s in report["stale"]] == ["evaporative_stress_index"]
    assert "et0_forecast (out_of_coverage)" in report["summary"]


def test_an_empty_ledger_is_not_degraded(ledger):
    report = ledger.degradation()
    assert report["degraded"] is False
    assert report["fact_count"] == 0 and report["gap_count"] == 0


def test_merging_a_sub_analysis_keeps_both_ledgers_accounting(ledger):
    other = Ledger("soil")
    other.record(fact("soil_ph", 6.2))
    other.gap("soil_om_pct", MissingReason.OUT_OF_COVERAGE, ("polaris",))
    ledger.record(fact("ndvi", 0.62))
    ledger.merge(other)
    assert len(ledger.facts) == 2
    assert len(ledger.gaps) == 1
    with pytest.raises(FabricationError):
        ledger.merge({"soil_ph": 6.2})


def test_a_derived_fact_inherits_the_worst_provenance_of_its_inputs():
    coarse = fact("climate_tavg_01", 21.0, resolution_m=4638.31, observed_on=dt.date(2020, 1, 1))
    fine = fact("climate_prec_01", 90.0, resolution_m=927.66, observed_on=dt.date(2025, 1, 1),
                chain_position=1)
    derived = Fact.derive("koppen_code", "Aw", "", [coarse, fine])
    assert derived.resolution_m == pytest.approx(4638.31)
    assert derived.observed_on == dt.date(2020, 1, 1)
    assert derived.chain_position == 1
    assert derived.derived_from == ("climate_tavg_01", "climate_prec_01")
    with pytest.raises(FabricationError):
        Fact.derive("koppen_code", "Aw", "", [])
    with pytest.raises(FabricationError):
        Fact.derive("koppen_code", "Aw", "", [21.0])
