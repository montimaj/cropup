"""Water: actual ET, reference ET0, evaporative stress, soil moisture, rain,
and whether this place is mapped as irrigated.

Every asset here was verified live against project ``irrigation-status-474718``
at Arusha, Morogoro and Salinas, and the traps are the ones SPEC section 3 names:

===============  ==========================================================
quantity         source, and what will bite you
===============  ==========================================================
actual ET        ``projects/usgs-ssebop/viirs_et_v6_dekadal`` (global, 1 km,
                 mm per dekad, scale factor 1.0) falling back to
                 ``IDAHO_EPSCOR/TERRACLIMATE`` ``aet`` (x 0.1, mm/month, ends
                 2024-12). **Not** MODIS ``MOD16A2GF``: it is permanently
                 masked at the Arusha point (SPEC 3.3).
reference ET0    ``.../global_et0/global_et0_monthly`` -- mm/month, **no**
                 scaling (its sibling ``global_ai`` needs 1/10000, this does
                 not), and a 1970-2000 climatology, so it is dated ``None``
                 and marked static rather than passed off as current.
                 Its ``system:index`` is uppercase for months 05-09 and
                 lowercase for the rest; a naive lookup fails half the year.
stress           ``projects/climate-engine/esi/4wk`` -- global, but the asset
                 stopped at 2025-08-20 while its catalog page still claims a
                 weekly update. Returned with its real date and a freshness
                 limit, so the age is on screen.
soil moisture    ``NASA/SMAP/SPL4SMGP/008`` (007 is dead) falling back to
                 ``NASA/FLDAS/NOAH01/C/GL/M/V001``.
precipitation    ``UCSB-CHG/CHIRPS/DAILY`` -- mm/day already, 50S-50N.
irrigated        ``.../GFSAD/LGRIP30`` -- 264 tiles, so ``.mosaic()``, and
                 point sampling returns "non-cropland" at all three test
                 points: only buffered class fractions mean anything.
                 :mod:`cropup.geo.context` reads this asset too and publishes
                 its own ``irrigation_regime``, so pass
                 ``irrigation_mapping=False`` when both run over one ledger.
CONUS only       ``projects/climate-engine/fret/forecast/eto`` (the only
                 forward-looking ET anywhere in the catalog) and OpenET.
                 Outside the lower 48 these are ``Missing(out_of_coverage)``
                 -- a Tanzanian farmer is never shown a US-only forecast.
===============  ==========================================================
"""

from __future__ import annotations

import calendar
import datetime as dt
from dataclasses import replace
from typing import Any, Sequence

from .. import bootstrap
from ..config import Settings, get_settings
from ..errors import EarthEngineUnavailable
from ..evidence import (
    Fact,
    Ledger,
    Missing,
    MissingReason,
    SourceDescriptor,
    fact_from_reduce_region,
    facts_from_reduce_region,
)

__all__ = [
    "SSEBOP_DEKADAL",
    "TERRACLIMATE",
    "GLOBAL_ET0_MONTHLY",
    "FRET_ETO",
    "OPENET_MONTHLY",
    "ESI_4WK",
    "SMAP_L4",
    "FLDAS_MONTHLY",
    "CHIRPS_DAILY",
    "LGRIP30",
    "CONUS_ONLY_QUANTITIES",
    "QUANTITY_CHAINS",
    "LGRIP_CLASSES",
    "water_quantities",
    "in_conus",
    "get_water_status",
]

SSEBOP_DEKADAL = "projects/usgs-ssebop/viirs_et_v6_dekadal"
TERRACLIMATE = "IDAHO_EPSCOR/TERRACLIMATE"
GLOBAL_ET0_MONTHLY = "projects/sat-io/open-datasets/global_et0/global_et0_monthly"
FRET_ETO = "projects/climate-engine/fret/forecast/eto"
# The registry records OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0, which the EE
# client now warns is deprecated in favour of this path. Same data, no warning.
OPENET_MONTHLY = "projects/openet/assets/ensemble/conus/gridmet/monthly/v2_0"
ESI_4WK = "projects/climate-engine/esi/4wk"
SMAP_L4 = "NASA/SMAP/SPL4SMGP/008"
FLDAS_MONTHLY = "NASA/FLDAS/NOAH01/C/GL/M/V001"
CHIRPS_DAILY = "UCSB-CHG/CHIRPS/DAILY"
LGRIP30 = "projects/sat-io/open-datasets/GFSAD/LGRIP30"

DEFAULT_RADIUS_M = 5000.0  # SMAP is 11 km and ESI 5.5 km: a point sample means nothing here

# Conterminous US. Alaska, Hawaii and the territories are outside FRET and
# OpenET too, which is why the test is CONUS and not "the United States".
CONUS_BBOX = (24.4, -124.9, 49.4, -66.9)  # south, west, north, east

CONUS_ONLY_QUANTITIES: tuple[str, ...] = ("et0_forecast_day1", "et0_forecast_7d", "openet_actual_et")

# FRET publishes a 7-day run; the window is a little wider so a short or late
# run is reported at its real length rather than silently dropped.
FORECAST_HORIZON_DAYS = 10

LGRIP_CLASSES = {0: "water", 1: "non-cropland", 2: "irrigated", 3: "rainfed"}

# global_et0_monthly ships months 05-09 uppercase and the rest lowercase.
_ET0_INDEX_BY_MONTH = {
    1: "et0_v3_01", 2: "et0_v3_02", 3: "et0_v3_03", 4: "et0_v3_04",
    5: "et0_V3_05", 6: "et0_V3_06", 7: "et0_V3_07", 8: "et0_V3_08",
    9: "et0_V3_09", 10: "et0_v3_10", 11: "et0_v3_11", 12: "et0_v3_12",
}

# et_actual, et0, evaporative_stress_index, soil_moisture_surface and
# soil_moisture_rootzone are spelled exactly as geo/registry.py spells them, so
# analysis/ sees one vocabulary. The rest are aggregates the registry has no
# name for: it lists one 'precipitation' and one 'et0_forecast', while a 7-day
# sum and a 30-day sum are two different measurements and must not share a name.
# The descriptors are built here rather than read from the registry because the
# arithmetic depends on the observation: a SSEBop dekad is 8-11 days long, a
# TerraClimate month 28-31, and both are only known once the image is chosen.
WATER_QUANTITIES: tuple[str, ...] = (
    "et_actual",
    "et_actual_dekad_median",
    "et_actual_anomaly_pct",
    "et0",
    "et_actual_over_et0",
    "et0_forecast_day1",
    "et0_forecast_7d",
    "openet_actual_et",
    "evaporative_stress_index",
    "soil_moisture_surface",
    "soil_moisture_rootzone",
    "soil_moisture_rootzone_wetness",
    "precipitation_7d",
    "precipitation_30d",
    "cropland_pct",
    "irrigated_cropland_pct",
    "rainfed_cropland_pct",
    "irrigation_regime",
)


# Which assets stand behind each quantity, so a degraded run can still say what
# it would have read.
QUANTITY_CHAINS: dict[str, tuple[str, ...]] = {
    "et_actual": (SSEBOP_DEKADAL, TERRACLIMATE),
    "et_actual_dekad_median": (SSEBOP_DEKADAL,),
    "et_actual_anomaly_pct": (SSEBOP_DEKADAL,),
    "et0": (GLOBAL_ET0_MONTHLY,),
    "et_actual_over_et0": (SSEBOP_DEKADAL, GLOBAL_ET0_MONTHLY),
    "et0_forecast_day1": (FRET_ETO,),
    "et0_forecast_7d": (FRET_ETO,),
    "openet_actual_et": (OPENET_MONTHLY,),
    "evaporative_stress_index": (ESI_4WK,),
    "soil_moisture_surface": (SMAP_L4, FLDAS_MONTHLY),
    "soil_moisture_rootzone": (SMAP_L4, FLDAS_MONTHLY),
    "soil_moisture_rootzone_wetness": (SMAP_L4,),
    "precipitation_7d": (CHIRPS_DAILY,),
    "precipitation_30d": (CHIRPS_DAILY,),
    "cropland_pct": (LGRIP30,),
    "irrigated_cropland_pct": (LGRIP30,),
    "rainfed_cropland_pct": (LGRIP30,),
    "irrigation_regime": (LGRIP30,),
}


def water_quantities(*, irrigation_mapping: bool = True) -> tuple[str, ...]:
    """Everything :func:`get_water_status` records when every source answers."""
    if irrigation_mapping:
        return WATER_QUANTITIES
    return tuple(q for q in WATER_QUANTITIES if QUANTITY_CHAINS.get(q) != (LGRIP30,))


def in_conus(lat: float, lon: float) -> bool:
    """Inside the conterminous US bounding box, where FRET and OpenET exist."""
    south, west, north, east = CONUS_BBOX
    return south <= float(lat) <= north and west <= float(lon) <= east


# --------------------------------------------------------------------------- helpers


def _geometry(ee: Any, lat: float, lon: float, radius_m: float) -> Any:
    return ee.Geometry.Point(float(lon), float(lat)).buffer(float(radius_m))


def _mean(ee: Any, image: Any, geom: Any, scale: float) -> Any:
    """Buffered mean as an unevaluated ``ee.Dictionary``.

    ``bestEffort`` lets Earth Engine coarsen the grid instead of failing on a
    large field; for the class fractions below that changes the sampling grid,
    not the proportions.
    """
    return image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=float(scale),
        bestEffort=True,
        maxPixels=int(1e10),
    )


def _newest(ee: Any, asset_id: str, end: dt.date, lookback_days: int) -> Any:
    """Newest image of a collection at or before ``end``, within a window."""
    return ee.Image(
        ee.ImageCollection(asset_id)
        .filterDate((end - dt.timedelta(days=lookback_days)).isoformat(), (end + dt.timedelta(days=1)).isoformat())
        .sort("system:time_start", False)
        .first()
    )


def _fail(ledger: Ledger, quantities: Sequence[str], chain: Sequence[str], exc: BaseException) -> list[Missing]:
    """Record one gap per quantity the failed call was supposed to produce."""
    detail = f"{type(exc).__name__}: {exc}"
    return [ledger.gap(quantity, MissingReason.SOURCE_FAILED, tuple(chain), detail) for quantity in quantities]


# How far back to look for the newest image of each source, in order. Finding
# the newest image by sorting a whole collection is what made the water call
# slow: SMAP L4 is 3-hourly and has 33,000 images, and sorting all of them cost
# 11 s of a 22 s call against 0.3 s for the last three weeks. The wider window
# is only paid for when a source has fallen further behind than it should.
LOOKBACK_DAYS = {
    SSEBOP_DEKADAL: (60, 400),
    TERRACLIMATE: (800, 4000),  # the archive ends 2024-12
    OPENET_MONTHLY: (800, 4000),  # also ends 2024-12
    ESI_4WK: (500, 4000),  # stopped at 2025-08-20
    SMAP_L4: (21, 400),
    FLDAS_MONTHLY: (120, 800),  # monthly, published ~2.5 months in arrears
    CHIRPS_DAILY: (45, 400),
}


def _try_windows(read: Any, lookbacks: Sequence[int]) -> tuple[Any, BaseException | None]:
    """Call ``read(lookback_days)`` over progressively wider windows.

    An empty window makes Earth Engine raise on the null image, which is the
    signal to widen. The result is either a payload or the last exception, and
    the caller decides between a fallback source and a recorded gap.
    """
    failure: BaseException | None = None
    for days in lookbacks:
        try:
            return read(days), None
        except Exception as exc:
            failure = exc
    return None, failure


def _days_in_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _dekad_days(observed_on: dt.date, dekad: int) -> int:
    """Length of a SSEBop dekad: 10, 10, then whatever is left of the month."""
    if dekad >= 3:
        return _days_in_month(observed_on.year, observed_on.month) - 20
    return 10


def _as_date(value: dt.date | str | None) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _note(item: Fact | Missing, note: str) -> Fact | Missing:
    if isinstance(item, Fact):
        return replace(item, note=note if not item.note else f"{item.note}; {note}")
    return replace(item, detail=note if not item.detail else f"{item.detail}. {note}")


# --------------------------------------------------------------------------- actual ET


def _actual_et(ee: Any, ledger: Ledger, geom: Any, end: dt.date, stale_after_days: int) -> Fact | Missing:
    """SSEBop VIIRS dekadal ET, with its same-dekad median for the anomaly."""
    chain = (SSEBOP_DEKADAL, TERRACLIMATE)
    quantities = ("et_actual", "et_actual_dekad_median", "et_actual_anomaly_pct")
    def read(lookback_days: int) -> dict[str, Any]:
        archive = ee.ImageCollection(SSEBOP_DEKADAL).filterDate("2012-01-01", (end + dt.timedelta(days=1)).isoformat())
        recent = archive.filterDate(
            (end - dt.timedelta(days=lookback_days)).isoformat(), (end + dt.timedelta(days=1)).isoformat()
        )
        latest = ee.Image(recent.sort("system:time_start", False).first())
        # The anomaly baseline is the same dekad-of-year across the whole archive.
        same_dekad = archive.filter(
            ee.Filter.And(
                ee.Filter.eq("month", latest.get("month")),
                ee.Filter.eq("dekad", latest.get("dekad")),
            )
        )
        stack = latest.select("et").rename("et_actual").addBands(
            same_dekad.select("et").median().rename("et_actual_dekad_median")
        )
        return ee.Dictionary(
            {
                "stats": _mean(ee, stack, geom, 1000),
                "observed_on": ee.Date(latest.get("system:time_start")).format("YYYY-MM-dd"),
                "dekad": latest.get("dekad"),
                "n_years": same_dekad.size(),
            }
        ).getInfo()

    payload, failure = _try_windows(read, LOOKBACK_DAYS[SSEBOP_DEKADAL])
    if payload is None:
        exc = failure or RuntimeError("SSEBop returned nothing")
        # Only the SSEBop-only quantities are recorded here. et_actual belongs to
        # the fallback now, and one quantity must not appear twice in the ledger:
        # a capability strip listing "soil moisture missing" twice, or once as
        # masked and once as source_failed, reads as two different failures.
        _fail(ledger, quantities[1:], chain, exc)
        return _terraclimate_et(
            ee, ledger, geom, end, stale_after_days, chain, primary_detail=f"SSEBop was unavailable ({exc})"
        )

    observed_on = dt.date.fromisoformat(payload["observed_on"])
    dekad = int(payload["dekad"])
    days = _dekad_days(observed_on, dekad)
    scaling = (
        f"SSEBop band 'et' is a total in mm over dekad {dekad} of "
        f"{observed_on.strftime('%B %Y')} ({days} days), scale factor 1.0; divided by {days} for mm/day"
    )
    current_source = SourceDescriptor(
        quantity="et_actual",
        asset_id=SSEBOP_DEKADAL,
        unit="mm/day",
        band="et",
        result_key="et_actual",
        resolution_m=1074.0,
        chain=chain,
        chain_position=0,
        scaling=scaling,
        transform=lambda value, days=days: value / days,
        valid_range=(0.0, 25.0),
        stale_after_days=stale_after_days,
        coverage="global",
    )
    median_source = replace(
        current_source,
        quantity="et_actual_dekad_median",
        result_key="et_actual_dekad_median",
        scaling=(
            f"median of the same dekad ({dekad}) of {observed_on.strftime('%B')} across "
            f"{payload['n_years']} years of SSEBop VIIRS v6, then divided by {days} for mm/day"
        ),
        stale_after_days=None,
        is_static=True,  # a multi-year median has no single observation date
    )

    stats = payload.get("stats")
    current = fact_from_reduce_region(stats, current_source, observed_on=observed_on)
    baseline = fact_from_reduce_region(stats, median_source, observed_on=None)
    ledger.add(baseline)

    if isinstance(current, Missing):
        # SSEBop is global, so an empty pixel here is worth falling back on, and
        # TerraClimate records the one et_actual entry either way.
        ledger.gap(
            "et_actual_anomaly_pct",
            MissingReason.MASKED,
            chain,
            "no current SSEBop ET at this location, so there is nothing to compare",
        )
        return _terraclimate_et(
            ee,
            ledger,
            geom,
            end,
            stale_after_days,
            chain,
            primary_detail=f"SSEBop had no unmasked pixel here ({current.detail or 'masked'})",
        )
    ledger.add(current)

    if isinstance(baseline, Fact) and baseline.value > 0:
        ledger.record(
            Fact.derive(
                "et_actual_anomaly_pct",
                100.0 * current.value / baseline.value,
                "%",
                [current, baseline],
                scaling_applied="current dekad ET as a percentage of the median ET for the same dekad-of-year",
                precision=0,
                note=(
                    "below 75% means the crop is using much less water than normal for the time of year; "
                    "around 100% is normal"
                ),
            )
        )
    else:
        ledger.gap(
            "et_actual_anomaly_pct",
            MissingReason.MASKED,
            chain,
            "the same-dekad median ET is missing or zero, so a percentage of normal is undefined",
        )
    return current


def _terraclimate_et(
    ee: Any,
    ledger: Ledger,
    geom: Any,
    end: dt.date,
    stale_after_days: int,
    chain: Sequence[str],
    *,
    primary_detail: str | None = None,
) -> Fact | Missing:
    """Fallback actual ET. Monthly, and the archive stops in 2024-12.

    ``primary_detail`` is why SSEBop did not answer; it travels into whatever
    this records so a single ledger entry tells the whole story of the chain.
    """
    def read(lookback_days: int) -> dict[str, Any]:
        latest = _newest(ee, TERRACLIMATE, end, lookback_days)
        return ee.Dictionary(
            {
                "stats": _mean(ee, latest.select("aet").rename("et_actual"), geom, 4638),
                "observed_on": ee.Date(latest.get("system:time_start")).format("YYYY-MM-dd"),
            }
        ).getInfo()

    prefix = f"{primary_detail}. " if primary_detail else ""
    payload, failure = _try_windows(read, LOOKBACK_DAYS[TERRACLIMATE])
    if payload is None:
        exc = failure or RuntimeError("TerraClimate returned nothing")
        return ledger.gap(
            "et_actual",
            MissingReason.SOURCE_FAILED,
            tuple(chain),
            f"{prefix}TerraClimate also failed: {type(exc).__name__}: {exc}",
        )

    observed_on = dt.date.fromisoformat(payload["observed_on"])
    days = _days_in_month(observed_on.year, observed_on.month)
    source = SourceDescriptor(
        quantity="et_actual",
        asset_id=TERRACLIMATE,
        unit="mm/day",
        band="aet",
        result_key="et_actual",
        resolution_m=4638.0,
        chain=tuple(chain),
        chain_position=1,
        scaling=f"TerraClimate 'aet' x 0.1 for mm/month, then divided by {days} days for mm/day",
        transform=lambda value, days=days: value * 0.1 / days,
        valid_range=(0.0, 25.0),
        stale_after_days=stale_after_days,
        coverage="global land",
    )
    item = fact_from_reduce_region(payload.get("stats"), source, observed_on=observed_on)
    if isinstance(item, Missing):
        # Saying "TerraClimate is monthly" about a gap would describe a reading
        # that does not exist; the gap has to name the second empty source.
        return ledger.gap(
            "et_actual", item.reason, tuple(chain), f"{prefix}TerraClimate had nothing here either: {item.detail}"
        )
    return ledger.add(_note(item, f"{prefix}TerraClimate is monthly and its archive ends 2024-12"))


# --------------------------------------------------------------------------- reference ET0


def _reference_et0(ee: Any, ledger: Ledger, geom: Any, month: int) -> Fact | Missing:
    """FAO-56 reference ET, as a 1970-2000 monthly normal. Never "current"."""
    try:
        image = ee.Image(
            ee.ImageCollection(GLOBAL_ET0_MONTHLY)
            .filter(ee.Filter.eq("system:index", _ET0_INDEX_BY_MONTH[month]))
            .first()
        )
        stats = _mean(ee, image.select("b1").rename("et0"), geom, 1000).getInfo()
    except Exception as exc:
        return _fail(ledger, ("et0",), (GLOBAL_ET0_MONTHLY,), exc)[0]

    days = _days_in_month(2001, month)  # a non-leap year: this is a climatology, not a real month
    source = SourceDescriptor(
        quantity="et0",
        asset_id=GLOBAL_ET0_MONTHLY,
        unit="mm/day",
        band="b1",
        result_key="et0",
        resolution_m=1000.0,
        scaling=(
            f"Global-ET0 v3 is mm/month with NO scale factor (unlike its sibling global_ai); "
            f"divided by {days} days for mm/day"
        ),
        transform=lambda value, days=days: value / days,
        valid_range=(0.0, 30.0),
        is_static=True,
        coverage="global",
    )
    item = fact_from_reduce_region(stats, source, observed_on=None)
    return ledger.add(
        _note(
            item,
            f"1970-2000 climatological normal for month {month:02d}; it captures the season, not this year's weather",
        )
    )


def _et_fraction(ledger: Ledger, actual: Fact | Missing, reference: Fact | Missing) -> None:
    """ETa / ETo. Useful, but the denominator is a normal, so say so."""
    if not isinstance(actual, Fact) or not isinstance(reference, Fact) or reference.value <= 0:
        ledger.gap(
            "et_actual_over_et0",
            MissingReason.MASKED,
            (SSEBOP_DEKADAL, GLOBAL_ET0_MONTHLY),
            "needs both actual ET and reference ET0; one of them is missing",
        )
        return
    ledger.record(
        Fact.derive(
            "et_actual_over_et0",
            actual.value / reference.value,
            "ratio",
            [actual, reference],
            scaling_applied="actual ET divided by the reference ET0 normal for the same month",
            note=(
                "crop coefficient x stress coefficient combined; naturally well below 1 over sparse or bare "
                "land, and the denominator is a 1970-2000 normal rather than today's demand"
            ),
        )
    )


# --------------------------------------------------------------------------- CONUS-only


def _forecast_et0(ee: Any, ledger: Ledger, geom: Any, lat: float, lon: float, end: dt.date) -> None:
    """The only forward-looking ET in the catalog, and it stops at the border."""
    quantities = ("et0_forecast_day1", "et0_forecast_7d")
    if not in_conus(lat, lon):
        for quantity in quantities:
            ledger.gap(
                quantity,
                MissingReason.OUT_OF_COVERAGE,
                (FRET_ETO,),
                "NOAA FRET covers the conterminous US only; CropUp has no forward-looking "
                "reference ET at this location, so irrigation advice here can only be retrospective",
            )
        return

    try:
        # The asset holds one forecast run -- measured 2026-09-17: exactly 7
        # images, today through today+6. The date filter is not redundant: an
        # unbounded sum over whatever the collection happens to hold would
        # quietly stop being a 7-day total, and asking for a past date must
        # return nothing rather than a forecast that was never for that day.
        collection = (
            ee.ImageCollection(FRET_ETO)
            .filterDate(end.isoformat(), (end + dt.timedelta(days=FORECAST_HORIZON_DAYS)).isoformat())
            .sort("system:time_start")
        )
        series = collection.map(
            lambda image: ee.Feature(
                None,
                {
                    "observed_on": ee.Image(image).date().format("YYYY-MM-dd"),
                    "eto": _mean(ee, ee.Image(image).select("eto"), geom, 2540).get("eto"),
                },
            )
        )
        rows = ee.FeatureCollection(series).getInfo().get("features", [])
    except Exception as exc:
        _fail(ledger, quantities, (FRET_ETO,), exc)
        return

    days = [(row["properties"]["observed_on"], row["properties"].get("eto")) for row in rows]
    values = [value for _, value in days if value is not None]
    if not days or not values:
        detail = (
            f"FRET published no forecast covering {end.isoformat()} at this location, "
            "although it is inside the CONUS box"
            if not days
            else "FRET returned no value at this location although it is inside the CONUS box"
        )
        for quantity in quantities:
            ledger.gap(quantity, MissingReason.MASKED, (FRET_ETO,), detail)
        return

    issued_for = dt.date.fromisoformat(days[0][0])
    horizon = f"{days[0][0]} to {days[-1][0]}"
    common = {
        "source_asset": FRET_ETO,
        "observed_on": issued_for,
        "resolution_m": 2540.0,
        "band": "eto",
        "scaling_applied": "none (scale factor 1.0); ASCE grass reference ET",
        # A forecast is only about the days it covers, so its first day is also
        # its expiry date. Without this the freshness of a forecast would read
        # as UNKNOWN, and a run served from yesterday's cache would look current.
        "stale_after_days": 1,
    }
    ledger.record(
        Fact(
            quantity="et0_forecast_day1",
            value=values[0],
            unit="mm/day",
            note=f"NOAA FRET forecast for {days[0][0]} (CONUS only)",
            **common,
        )
    )
    ledger.record(
        Fact(
            quantity="et0_forecast_7d",
            value=sum(values),
            unit="mm",
            note=f"sum of the {len(values)}-day NOAA FRET forecast, {horizon} (CONUS only)",
            **common,
        )
    )


def _openet(ee: Any, ledger: Ledger, geom: Any, lat: float, lon: float, end: dt.date, stale_after_days: int) -> None:
    """30 m ensemble ET. CONUS only, and roughly two years behind."""
    if not in_conus(lat, lon):
        ledger.gap(
            "openet_actual_et",
            MissingReason.OUT_OF_COVERAGE,
            (OPENET_MONTHLY,),
            "OpenET covers the conterminous US only",
        )
        return
    def read(lookback_days: int) -> dict[str, Any]:
        latest = _newest(ee, OPENET_MONTHLY, end, lookback_days)
        return ee.Dictionary(
            {
                "stats": _mean(ee, latest.select("et_ensemble_mad").rename("openet_actual_et"), geom, 30),
                "observed_on": ee.Date(latest.get("system:time_start")).format("YYYY-MM-dd"),
            }
        ).getInfo()

    payload, failure = _try_windows(read, LOOKBACK_DAYS[OPENET_MONTHLY])
    if payload is None:
        _fail(ledger, ("openet_actual_et",), (OPENET_MONTHLY,), failure or RuntimeError("OpenET returned nothing"))
        return

    observed_on = dt.date.fromisoformat(payload["observed_on"])
    source = SourceDescriptor(
        quantity="openet_actual_et",
        asset_id=OPENET_MONTHLY,
        unit="mm/month",
        band="et_ensemble_mad",
        result_key="openet_actual_et",
        resolution_m=30.0,
        scaling="none (already mm/month); ensemble median-absolute-deviation member",
        valid_range=(0.0, 600.0),
        stale_after_days=stale_after_days,
        coverage="CONUS",
    )
    item = fact_from_reduce_region(payload.get("stats"), source, observed_on=observed_on)
    ledger.add(_note(item, "OpenET is a monthly total for the month shown, not a current rate"))


# --------------------------------------------------------------------------- stress


def _esi(ee: Any, ledger: Ledger, geom: Any, end: dt.date, stale_after_days: int) -> None:
    """NOAA 4-week Evaporative Stress Index: global, but it stopped in 2025."""
    def read(lookback_days: int) -> dict[str, Any]:
        latest = _newest(ee, ESI_4WK, end, lookback_days)
        return ee.Dictionary(
            {
                "stats": _mean(ee, latest.select("ESI").rename("evaporative_stress_index"), geom, 5566),
                "observed_on": ee.Date(latest.get("system:time_start")).format("YYYY-MM-dd"),
            }
        ).getInfo()

    payload, failure = _try_windows(read, LOOKBACK_DAYS[ESI_4WK])
    if payload is None:
        _fail(ledger, ("evaporative_stress_index",), (ESI_4WK,), failure or RuntimeError("ESI returned nothing"))
        return

    observed_on = dt.date.fromisoformat(payload["observed_on"])
    source = SourceDescriptor(
        quantity="evaporative_stress_index",
        asset_id=ESI_4WK,
        unit="index",
        band="ESI",
        result_key="evaporative_stress_index",
        resolution_m=5566.0,
        scaling="none (scale factor 1.0); standardised anomaly, negative is more stressed than normal",
        valid_range=(-6.0, 6.0),
        stale_after_days=stale_after_days,
        coverage="global",
    )
    item = fact_from_reduce_region(payload.get("stats"), source, observed_on=observed_on)
    ledger.add(
        _note(
            item,
            "the ESI asset stopped updating at 2025-08-20 although its catalog page advertises weekly "
            "updates, so check the observation date before reading this as current stress",
        )
    )


# --------------------------------------------------------------------------- soil moisture


def _soil_moisture(ee: Any, ledger: Ledger, geom: Any, end: dt.date, stale_after_days: int) -> None:
    chain = (SMAP_L4, FLDAS_MONTHLY)
    quantities = ("soil_moisture_surface", "soil_moisture_rootzone", "soil_moisture_rootzone_wetness")
    def read(lookback_days: int) -> dict[str, Any]:
        latest = _newest(ee, SMAP_L4, end, lookback_days)
        return ee.Dictionary(
            {
                "stats": _mean(
                    ee, latest.select(["sm_surface", "sm_rootzone", "sm_rootzone_wetness"]), geom, 11000
                ),
                "observed_on": ee.Date(latest.get("system:time_start")).format("YYYY-MM-dd"),
            }
        ).getInfo()

    payload, failure = _try_windows(read, LOOKBACK_DAYS[SMAP_L4])
    if payload is None:
        exc = failure or RuntimeError("SMAP returned nothing")
        ledger.gap(
            "soil_moisture_rootzone_wetness",
            MissingReason.SOURCE_FAILED,
            chain,
            f"only SMAP reports wetness as a fraction of saturation, and it was unavailable: "
            f"{type(exc).__name__}: {exc}",
        )
        _fldas_soil_moisture(
            ee,
            ledger,
            geom,
            end,
            stale_after_days,
            chain,
            wanted=quantities[:2],
            primary_detail=f"SMAP was unavailable ({type(exc).__name__}: {exc})",
        )
        return

    observed_on = dt.date.fromisoformat(payload["observed_on"])
    common = {
        "asset_id": SMAP_L4,
        "resolution_m": 10593.0,
        "chain": chain,
        "chain_position": 0,
        "scaling": "none (scale factor 1.0)",
        "stale_after_days": stale_after_days,
        "coverage": "global",
    }
    sources = [
        SourceDescriptor(
            quantity="soil_moisture_surface", unit="m3/m3", band="sm_surface",
            result_key="sm_surface", valid_range=(0.0, 1.0), **common,
        ),
        SourceDescriptor(
            quantity="soil_moisture_rootzone", unit="m3/m3", band="sm_rootzone",
            result_key="sm_rootzone", valid_range=(0.0, 1.0), **common,
        ),
        SourceDescriptor(
            quantity="soil_moisture_rootzone_wetness", unit="ratio", band="sm_rootzone_wetness",
            result_key="sm_rootzone_wetness", valid_range=(0.0, 1.0), **common,
        ),
    ]
    evidence = facts_from_reduce_region(payload.get("stats"), sources, observed_on=observed_on)
    notes = {
        "soil_moisture_surface": "SMAP L4 surface layer, 0-5 cm",
        "soil_moisture_rootzone": "SMAP L4 root zone, 0-100 cm: the depth irrigation decisions turn on",
        "soil_moisture_rootzone_wetness": "fraction of saturation in the root zone (0-1)",
    }
    # Whatever SMAP could not measure is handed to FLDAS, which then owns the
    # single ledger entry for it; recording SMAP's Missing here as well would
    # report one gap twice, under two different reasons.
    handed_over = tuple(q for q in quantities[:2] if isinstance(evidence[q], Missing))
    for quantity, item in evidence.items():
        if quantity in handed_over:
            continue
        if quantity == "soil_moisture_rootzone_wetness" and isinstance(item, Missing):
            ledger.add(_note(item, "no other source reports wetness as a fraction of saturation"))
            continue
        ledger.add(_note(item, notes[quantity]))

    if handed_over:
        detail = evidence[handed_over[0]].detail or "masked"
        _fldas_soil_moisture(
            ee,
            ledger,
            geom,
            end,
            stale_after_days,
            chain,
            wanted=handed_over,
            primary_detail=f"SMAP had no unmasked pixel here ({detail})",
        )


def _fldas_soil_moisture(
    ee: Any,
    ledger: Ledger,
    geom: Any,
    end: dt.date,
    stale_after_days: int,
    chain: Sequence[str],
    *,
    wanted: Sequence[str],
    primary_detail: str | None = None,
) -> None:
    """FLDAS Noah, monthly, four named depths. Built for FEWS NET East Africa.

    Records exactly the quantities in ``wanted`` -- the ones SMAP could not
    supply -- so that each quantity appears in the ledger once. Wetness is never
    among them: only SMAP expresses moisture as a fraction of saturation.
    """
    layers = ("SoilMoi00_10cm_tavg", "SoilMoi10_40cm_tavg", "SoilMoi40_100cm_tavg")
    prefix = f"{primary_detail}. " if primary_detail else ""
    def read(lookback_days: int) -> dict[str, Any]:
        latest = _newest(ee, FLDAS_MONTHLY, end, lookback_days)
        return ee.Dictionary(
            {
                "stats": _mean(ee, latest.select(list(layers)), geom, 11132),
                "observed_on": ee.Date(latest.get("system:time_start")).format("YYYY-MM-dd"),
            }
        ).getInfo()

    payload, failure = _try_windows(read, LOOKBACK_DAYS[FLDAS_MONTHLY])
    if payload is None:
        exc = failure or RuntimeError("FLDAS returned nothing")
        for quantity in wanted:
            ledger.gap(
                quantity,
                MissingReason.SOURCE_FAILED,
                tuple(chain),
                f"{prefix}FLDAS also failed: {type(exc).__name__}: {exc}",
            )
        return

    observed_on = dt.date.fromisoformat(payload["observed_on"])
    common = {
        "asset_id": FLDAS_MONTHLY,
        "unit": "m3/m3",
        "resolution_m": 11132.0,
        "chain": tuple(chain),
        "chain_position": 1,
        "scaling": "none (already m3/m3)",
        "valid_range": (0.0, 1.0),
        "stale_after_days": stale_after_days,
        "coverage": "global",
    }
    stats = payload.get("stats")
    if "soil_moisture_surface" in wanted:
        surface = fact_from_reduce_region(
            stats,
            SourceDescriptor(quantity="soil_moisture_surface", band=layers[0], result_key=layers[0], **common),
            observed_on=observed_on,
        )
        if isinstance(surface, Missing):
            ledger.gap(
                "soil_moisture_surface",
                surface.reason,
                tuple(chain),
                f"{prefix}the FLDAS 0-10 cm layer is masked here too",
            )
        else:
            ledger.add(_note(surface, f"{prefix}FLDAS 0-10 cm monthly mean instead"))

    if "soil_moisture_rootzone" not in wanted:
        return

    # FLDAS has no 0-100 cm band, so build one from the three layers that span it.
    depths = ((layers[0], 0.10), (layers[1], 0.30), (layers[2], 0.60))
    parts: list[tuple[Fact, float]] = []
    for band, weight in depths:
        item = fact_from_reduce_region(
            stats,
            SourceDescriptor(quantity=f"fldas_{band}", band=band, result_key=band, **common),
            observed_on=observed_on,
        )
        if isinstance(item, Fact):
            parts.append((item, weight))
    if len(parts) == len(depths):
        ledger.record(
            Fact.derive(
                "soil_moisture_rootzone",
                sum(fact.value * weight for fact, weight in parts),
                "m3/m3",
                [fact for fact, _ in parts],
                scaling_applied="depth-weighted mean of the FLDAS 0-10, 10-40 and 40-100 cm layers (0.1/0.3/0.6)",
                note=f"{prefix}this is a 0-100 cm average built from the FLDAS layers",
            )
        )
    else:
        ledger.gap(
            "soil_moisture_rootzone",
            MissingReason.MASKED,
            tuple(chain),
            f"{prefix}the FLDAS depth layers needed for a 0-100 cm average are masked here too",
        )


# --------------------------------------------------------------------------- precipitation


def _precipitation(ee: Any, ledger: Ledger, geom: Any, end: dt.date, stale_after_days: int) -> None:
    quantities = ("precipitation_7d", "precipitation_30d")
    def read(lookback_days: int) -> dict[str, Any]:
        collection = ee.ImageCollection(CHIRPS_DAILY).filterDate(
            "1981-01-01", (end + dt.timedelta(days=1)).isoformat()
        )
        last = ee.Date(_newest(ee, CHIRPS_DAILY, end, lookback_days).get("system:time_start"))
        window_end = last.advance(1, "day")
        stack = (
            collection.filterDate(window_end.advance(-7, "day"), window_end)
            .select("precipitation")
            .sum()
            .rename("precipitation_7d")
            .addBands(
                collection.filterDate(window_end.advance(-30, "day"), window_end)
                .select("precipitation")
                .sum()
                .rename("precipitation_30d")
            )
        )
        return ee.Dictionary(
            {"stats": _mean(ee, stack, geom, 5566), "observed_on": last.format("YYYY-MM-dd")}
        ).getInfo()

    payload, failure = _try_windows(read, LOOKBACK_DAYS[CHIRPS_DAILY])
    if payload is None:
        _fail(ledger, quantities, (CHIRPS_DAILY,), failure or RuntimeError("CHIRPS returned nothing"))
        return

    observed_on = dt.date.fromisoformat(payload["observed_on"])
    common = {
        "asset_id": CHIRPS_DAILY,
        "unit": "mm",
        "band": "precipitation",
        "resolution_m": 5566.0,
        "stale_after_days": stale_after_days,
        "coverage": "50S-50N",
        "valid_range": (0.0, 5000.0),
    }
    sources = [
        SourceDescriptor(
            quantity="precipitation_7d", result_key="precipitation_7d",
            scaling=f"sum of CHIRPS daily mm over the 7 days ending {observed_on.isoformat()}", **common,
        ),
        SourceDescriptor(
            quantity="precipitation_30d", result_key="precipitation_30d",
            scaling=f"sum of CHIRPS daily mm over the 30 days ending {observed_on.isoformat()}", **common,
        ),
    ]
    for item in facts_from_reduce_region(payload.get("stats"), sources, observed_on=observed_on).values():
        ledger.add(_note(item, f"CHIRPS final runs about two weeks behind; window ends {observed_on.isoformat()}"))


# --------------------------------------------------------------------------- irrigated vs rainfed


def _irrigation_mapping(ee: Any, ledger: Ledger, geom: Any) -> None:
    """LGRIP30 class shares around the field. Point sampling is useless here."""
    quantities = ("cropland_pct", "irrigated_cropland_pct", "rainfed_cropland_pct", "irrigation_regime")
    try:
        mosaic = ee.ImageCollection(LGRIP30).mosaic().rename("lgrip")
        histogram = mosaic.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=geom,
            scale=30,
            bestEffort=True,
            maxPixels=int(1e10),
        ).getInfo()
    except Exception as exc:
        _fail(ledger, quantities, (LGRIP30,), exc)
        return

    counts = (histogram or {}).get("lgrip") or {}
    total = sum(counts.values())
    if not total:
        for quantity in quantities:
            ledger.gap(quantity, MissingReason.MASKED, (LGRIP30,), "LGRIP30 returned no pixels over this field")
        return

    shares = {LGRIP_CLASSES.get(int(float(code)), str(code)): count / total for code, count in counts.items()}
    irrigated = shares.get("irrigated", 0.0)
    rainfed = shares.get("rainfed", 0.0)
    cropland = irrigated + rainfed

    caveat = (
        "LGRIP30 is nominal 2015 and commission errors are visible where it can be ground-checked "
        "(it calls much of the irrigated Salinas Valley rainfed), so treat it as a hint, not a verdict"
    )
    common = {
        "source_asset": LGRIP30,
        "observed_on": None,  # 264 tiles, no system:time_start; nominal 2015
        "resolution_m": 30.0,
        "band": "b1",
        "note": f"share of the sampled field area; classes {LGRIP_CLASSES}. {caveat}",
    }
    for quantity, value, rule in (
        ("cropland_pct", cropland * 100.0, "classes 2 + 3"),
        ("irrigated_cropland_pct", irrigated * 100.0, "class 2"),
        ("rainfed_cropland_pct", rainfed * 100.0, "class 3"),
    ):
        ledger.record(
            Fact(
                quantity=quantity,
                value=value,
                unit="%",
                scaling_applied=f"{rule} as a percentage of all LGRIP30 pixels in the field",
                **common,
            )
        )

    if cropland < 0.05:
        regime = "not mapped as cropland"
    elif irrigated > rainfed:
        regime = "irrigated"
    else:
        regime = "rainfed"
    ledger.record(
        Fact(
            quantity="irrigation_regime",
            value=regime,
            unit="",
            scaling_applied="the larger of the irrigated and rainfed shares, or 'not mapped as cropland' below 5% cropland",
            **common,
        )
    )


# --------------------------------------------------------------------------- public API


def get_water_status(
    lat: float,
    lon: float,
    *,
    geometry: Any | None = None,
    radius_m: float = DEFAULT_RADIUS_M,
    end_date: dt.date | str | None = None,
    irrigation_mapping: bool = True,
    settings: Settings | None = None,
    ledger: Ledger | None = None,
) -> Ledger:
    """Water balance evidence for one field: ET, ET0, stress, moisture, rain.

    Returns a :class:`~cropup.evidence.Ledger`. Sources that do not reach this
    location -- the CONUS-only forecast above all -- are recorded as
    ``Missing(out_of_coverage)`` rather than omitted, because the difference
    between "no forecast exists here" and "no forecast was requested" is the
    difference between honest and misleading irrigation advice.

    ``irrigation_mapping=False`` drops the four LGRIP30 quantities, including
    ``irrigation_regime``. :mod:`cropup.geo.context` reads the same asset a
    different way and publishes a quantity of that name too, so an orchestrator
    that calls both must switch one of them off: two different verdicts on
    whether a field is irrigated, under one name in one ledger, is worse than
    either answer alone.
    """
    settings = settings or get_settings()
    ledger = ledger if ledger is not None else Ledger(turn="water")
    end = _as_date(end_date) or dt.date.today()
    stale_after_days = settings.default_stale_after_days

    expected = WATER_QUANTITIES if irrigation_mapping else tuple(
        q for q in WATER_QUANTITIES if QUANTITY_CHAINS.get(q) != (LGRIP30,)
    )

    try:
        ee = bootstrap.require_ee()
    except EarthEngineUnavailable as exc:
        for quantity in expected:
            ledger.gap(quantity, MissingReason.SOURCE_FAILED, QUANTITY_CHAINS.get(quantity, ()), str(exc))
        return ledger

    geom = geometry if geometry is not None else _geometry(ee, lat, lon, radius_m)

    actual = _actual_et(ee, ledger, geom, end, stale_after_days)
    month = actual.observed_on.month if isinstance(actual, Fact) and actual.observed_on else end.month
    reference = _reference_et0(ee, ledger, geom, month)
    _et_fraction(ledger, actual, reference)
    _forecast_et0(ee, ledger, geom, lat, lon, end)
    _openet(ee, ledger, geom, lat, lon, end, stale_after_days)
    _esi(ee, ledger, geom, end, stale_after_days)
    _soil_moisture(ee, ledger, geom, end, stale_after_days)
    _precipitation(ee, ledger, geom, end, stale_after_days)
    if irrigation_mapping:
        _irrigation_mapping(ee, ledger, geom)
    return ledger
