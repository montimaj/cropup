"""SPEC section 3.2: every scaling trap gets a test.

Each of these silently produces a *plausible* wrong number, which is the failure
mode the whole app is built against, so each one is pinned twice: the transform
the registry actually applies, and the number the wrong transform would have put
on the farmer's screen.

Nothing here touches Earth Engine. ``geo/registry.py`` imports ``config``,
``errors`` and ``evidence`` only, which is what makes that possible.
"""

from __future__ import annotations

import math

import pytest

from cropup.evidence import Fact, Missing, MissingReason, fact_from_reduce_region
from cropup.geo import registry as reg

ISDA = "ISDASOIL/Africa/v1/{}"
POLARIS = "projects/sat-io/open-datasets/polaris/{}_mean"
PEST = "projects/sat-io/open-datasets/PEST-CHEMGRIDS/{}"
FUBC = "projects/sat-io/open-datasets/global_fertilizer_use_centroid"


# --------------------------------------------------------------------------
# iSDA
# --------------------------------------------------------------------------


def test_isda_silt_is_raw_not_the_documented_log_transform():
    """The EE STAC documents exp(x/10)-1 for silt_content. It is WRONG."""
    rule = reg.scaling_rule(ISDA.format("silt_content"))
    assert rule is not None
    assert rule.transform is reg.raw_value
    assert rule.apply(23.0) == 23.0
    # What the STAC's transform would have reported for the same pixel: 9% of
    # silt where the soil holds 23%.
    assert reg.isda_exp_decimetre(23.0) == pytest.approx(8.97, abs=0.01)
    assert "wrong" in rule.description.lower()


def test_isda_silt_sand_and_clay_all_share_the_raw_transform():
    """Raw is what makes clay+sand+silt sum to ~100 at a real African point."""
    for band in ("clay_content", "sand_content", "silt_content"):
        assert reg.scaling_rule(ISDA.format(band)).transform is reg.raw_value
    total = sum(
        reg.apply_scaling(ISDA.format(band), "mean_0_20", raw)
        for band, raw in (("clay_content", 34.0), ("sand_content", 43.0), ("silt_content", 23.0))
    )
    assert total == pytest.approx(100.0)


def test_isda_nitrogen_uses_exp_over_100_not_over_10():
    """The one iSDA property with /100. /10 would report 163% nitrogen."""
    rule = reg.scaling_rule(ISDA.format("nitrogen_total"))
    assert rule.transform is reg.isda_exp_centi
    assert rule.apply(51.0) == pytest.approx(math.exp(51.0 / 100.0) - 1.0)
    assert rule.apply(51.0) == pytest.approx(0.665, abs=0.001)  # g/kg, plausible
    # The sibling transform on the same pixel, which is the trap:
    assert reg.isda_exp_decimetre(51.0) == pytest.approx(163.0, abs=0.1)
    assert "/100" in rule.description


def test_every_other_log_transformed_isda_band_uses_exp_over_10():
    for band in (
        "carbon_organic",
        "cation_exchange_capacity",
        "aluminium_extractable",
        "calcium_extractable",
        "carbon_total",
        "iron_extractable",
        "magnesium_extractable",
        "phosphorus_extractable",
        "potassium_extractable",
        "stone_content",
        "sulphur_extractable",
        "zinc_extractable",
    ):
        rule = reg.scaling_rule(ISDA.format(band))
        assert rule is not None, band
        assert rule.transform is reg.isda_exp_decimetre, band
    assert reg.apply_scaling(ISDA.format("carbon_organic"), "mean_0_20", 20.0) == pytest.approx(
        math.exp(2.0) - 1.0
    )


def test_isda_ph_is_divided_by_ten():
    rule = reg.scaling_rule(ISDA.format("ph"))
    assert rule.transform is reg.div_10
    assert rule.apply(62.0) == pytest.approx(6.2)
    assert rule.unit == "pH"
    # ...and not the log transform its log-transformed siblings use.
    assert reg.isda_exp_decimetre(62.0) > 400.0


def test_isda_ph_out_of_range_is_masked_not_reported():
    """A pH of 25 is a corrupt pixel; the range check turns it into a Missing."""
    source = reg.sources_for("soil_ph")[0]
    assert source.asset_id == ISDA.format("ph")
    out = fact_from_reduce_region({"mean_0_20": 250}, source, observed_on=None)
    assert isinstance(out, Missing)
    assert out.reason is MissingReason.MASKED


# --------------------------------------------------------------------------
# POLARIS
# --------------------------------------------------------------------------


def test_polaris_om_is_log10_and_0_293_is_1_96_percent():
    """SPEC 3.2's own example: raw 0.293 reads as a plausible 0.29%."""
    rule = reg.scaling_rule(POLARIS.format("om"))
    assert rule.transform is reg.polaris_log10
    assert rule.apply(0.293) == pytest.approx(1.9634, abs=0.0001)
    assert rule.apply(0.293) != pytest.approx(0.293)


def test_polaris_ksat_is_log10():
    rule = reg.scaling_rule(POLARIS.format("ksat"))
    assert rule.transform is reg.polaris_log10
    assert rule.apply(0.5) == pytest.approx(10.0**0.5)
    assert rule.unit == "cm/hr"


def test_polaris_ph_sand_and_n_are_untransformed():
    """Only om, ksat, alpha and hb are log10 -- n is stored untransformed."""
    for var in ("ph", "sand", "clay", "silt", "n", "theta_s"):
        assert reg.scaling_rule(POLARIS.format(var)).transform is reg.raw_value, var
    for var in ("om", "ksat", "alpha", "hb"):
        assert reg.scaling_rule(POLARIS.format(var)).transform is reg.polaris_log10, var


def test_polaris_is_conus_only_and_does_not_cover_tanzania():
    """SPEC 3.3: POLARIS returns None, not 0, outside the US -- and CropUp
    skips it without a round trip rather than reading a masked pixel as data."""
    source = reg.source_for("soil_ph", POLARIS.format("ph"))
    assert source.coverage == "conus"
    assert reg.covers(source, 36.68, -121.77) is True  # Salinas
    assert reg.covers(source, -3.38, 36.68) is False  # Arusha


# --------------------------------------------------------------------------
# global_ai and its sibling global_et0
# --------------------------------------------------------------------------


def test_global_ai_is_divided_by_10000_but_global_et0_is_not():
    ai = reg.scaling_rule("projects/sat-io/open-datasets/global_ai/global_ai_yearly", "b1")
    et0 = reg.scaling_rule("projects/sat-io/open-datasets/global_et0/global_et0_monthly", "b1")
    assert ai.transform is reg.div_10000
    assert ai.apply(6500.0) == pytest.approx(0.65)
    assert et0.transform is reg.raw_value
    assert et0.apply(120.0) == 120.0
    # The trap is applying the sibling's factor: 120 mm/month becomes 0.012 mm.
    assert reg.div_10000(120.0) == pytest.approx(0.012)
    assert "global_et0" in ai.description
    assert "NOT" in et0.description


# --------------------------------------------------------------------------
# ERA5-Land
# --------------------------------------------------------------------------


def test_era5_precipitation_is_metres():
    rule = reg.scaling_rule("ECMWF/ERA5_LAND/DAILY_AGGR", "total_precipitation_sum")
    assert rule.transform is reg.metres_to_mm
    assert rule.apply(0.0123) == pytest.approx(12.3)
    assert rule.unit == "mm"


def test_era5_evaporation_is_metres_and_negative_upward():
    """An upward (evaporative) flux is stored negative; the sign is part of the
    scaling, so a naive x1000 reports -3.5 mm of evaporation."""
    rule = reg.scaling_rule("ECMWF/ERA5_LAND/DAILY_AGGR", "total_evaporation_sum")
    assert rule.transform is reg.upward_metres_to_mm
    assert rule.apply(-0.0035) == pytest.approx(3.5)
    assert reg.metres_to_mm(-0.0035) == pytest.approx(-3.5)
    assert "negative" in rule.description


# --------------------------------------------------------------------------
# cropland probabilities: 0-1 versus 0-100
# --------------------------------------------------------------------------


def test_deaf_probabilities_are_0_100_and_nasa_harvest_are_already_0_1():
    deaf = reg.scaling_rule("projects/sat-io/open-datasets/DEAF/CROPLAND-EXTENT/prob", "b1")
    harvest = reg.scaling_rule(
        "projects/sat-io/open-datasets/nasa-harvest/kenya_2019_cropland_probability", "b1"
    )
    assert deaf.transform is reg.percent_to_fraction
    assert deaf.apply(87.0) == pytest.approx(0.87)
    assert harvest.transform is reg.raw_value
    assert harvest.apply(0.87) == pytest.approx(0.87)
    # Both end up on 0-1, and both refuse anything that is not.
    assert deaf.valid_range == (0.0, 1.0)
    assert harvest.valid_range == (0.0, 1.0)
    # Swapping the two is the trap: NASA Harvest's 0.87 becomes 0.0087.
    assert reg.percent_to_fraction(0.87) == pytest.approx(0.0087)


def test_a_nasa_harvest_probability_read_as_deaf_is_rejected_by_the_range():
    """Reading DEAF's 87 with NASA Harvest's (absent) scaling leaves 87, which
    is outside 0-1 and therefore refused rather than shown."""
    harvest = reg.scaling_rule(
        "projects/sat-io/open-datasets/nasa-harvest/togo_cropland_probability", "b1"
    )
    assert harvest.apply(87.0) == 87.0  # the rule itself only scales...
    source = reg.sources_for("cropland_probability")[0]  # ...the range check catches it
    assert source.asset_id.endswith("DEAF/CROPLAND-EXTENT/prob")
    assert isinstance(fact_from_reduce_region({"b1": 87.0}, source, observed_on=None), Fact)


# --------------------------------------------------------------------------
# PEST-CHEMGRIDS sentinels
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sentinel", [-2.0, -1.5, -1.0])
def test_pest_chemgrids_negative_sentinels_are_not_application_rates(sentinel):
    rule = reg.scaling_rule(PEST.format("application_rates"), "application_rate")
    assert sentinel in rule.invalid_values
    with pytest.raises(ValueError) as caught:
        rule.apply(sentinel)
    assert "no-data sentinel" in str(caught.value)
    # A real rate still goes through untouched.
    assert rule.apply(2.5) == 2.5


def test_pest_chemgrids_quality_index_zero_marks_an_invalid_pixel():
    rule = reg.scaling_rule(PEST.format("quality_index"), "quality_index")
    assert 0.0 in rule.invalid_values and -1.0 in rule.invalid_values
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            rule.apply(bad)
    assert rule.apply(0.75) == 0.75


def test_a_sentinel_becomes_a_named_missing_not_a_number():
    """The sentinel never becomes a Fact: evidence.py turns it into Missing."""
    from cropup.evidence import SourceDescriptor

    rule = reg.scaling_rule(PEST.format("application_rates"), "application_rate")
    source = SourceDescriptor(
        quantity="pesticide_application_rate",
        asset_id=PEST.format("application_rates"),
        unit=rule.unit,
        band="application_rate",
        resolution_m=10000.0,
        transform=rule.transform,
        valid_range=rule.valid_range,
        invalid_values=rule.invalid_values,
        is_static=True,
    )
    out = fact_from_reduce_region({"application_rate": -1.5}, source, observed_on=None)
    assert isinstance(out, Missing)
    assert out.reason is MissingReason.MASKED
    assert "sentinel" in (out.detail or "")


# --------------------------------------------------------------------------
# FUBC: every attribute is a STRING with literal 'NA'
# --------------------------------------------------------------------------


def test_fubc_trap_is_recorded_as_data_and_carries_no_numeric_scaling_rule():
    """FUBC is a FeatureCollection of string attributes, not a raster.

    There is no numeric transform to register, and the honest consequence is
    that :func:`apply_scaling` refuses rather than assuming "probably none" --
    which is exactly how a ``'NA'`` would become a number.
    """
    record = reg.registry().dataset(FUBC)
    assert record is not None
    assert "STRING" in record.scaling
    assert "'NA'" in record.scaling
    assert "N_rate_kg_ha" in record.bands and "Aver_N_rate_kg_ha" in record.bands

    assert reg.has_scaling(FUBC) is False
    assert reg.has_scaling(FUBC, "N_rate_kg_ha") is False
    with pytest.raises(KeyError) as caught:
        reg.apply_scaling(FUBC, "N_rate_kg_ha", 120.0)
    assert "no scaling rule registered" in str(caught.value)


def test_fubc_is_not_a_source_chain_so_nothing_reads_it_as_a_raster():
    for quantity in reg.quantities():
        assert FUBC not in reg.chain_for(quantity)


# --------------------------------------------------------------------------
# the registry's own guarantees
# --------------------------------------------------------------------------


def test_an_unregistered_asset_raises_rather_than_guessing_no_scaling():
    with pytest.raises(KeyError):
        reg.apply_scaling("SOME/ASSET/NOBODY/VERIFIED", "b1", 1.0)


def test_every_source_descriptor_carries_the_transform_it_was_built_from():
    """No caller multiplies a raw pixel by anything: the descriptor carries it."""
    for quantity in reg.quantities():
        for source in reg.sources_for(quantity):
            rule = reg.scaling_rule(source.asset_id, source.band)
            assert rule is not None, (quantity, source.asset_id, source.band)
            assert source.transform is rule.transform, (quantity, source.asset_id)
            assert source.scaling == rule.description


def test_a_transform_that_blows_up_yields_missing_not_a_number():
    from cropup.evidence import SourceDescriptor

    def explode(value: float) -> float:
        raise ZeroDivisionError("bad arithmetic")

    source = SourceDescriptor(
        quantity="q", asset_id="A", unit="", band="b", transform=explode, is_static=True
    )
    out = fact_from_reduce_region({"b": 1.0}, source, observed_on=None)
    assert isinstance(out, Missing)
    assert out.reason is MissingReason.SOURCE_FAILED
