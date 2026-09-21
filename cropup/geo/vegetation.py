"""Sentinel-2 vegetation indices, NDVI zones and the year-ago comparison.

Ported from ``Data/googlebuildathonfarmers-main/app/earth_engine/vegetation.py``
with its three vegetation bugs fixed (SPEC 2.1):

1. **Zone percentages were shifted by one histogram bin.** The vendored code
   binned NDVI with ``ee.Reducer.fixedHistogram(-0.2, 1.0, 12)`` and then
   classified each bin by comparing its *lower edge* against 0.2 / 0.4 / 0.6.
   Earth Engine returns those edges as ``0.19999999999999996``,
   ``0.39999999999999997`` and ``0.5999999999999999`` (read back live at Arusha
   on 2026-09-17), so ``0.6 <= edge`` is False for the 0.6-0.7 bin and every
   zone absorbed the bin below it: a pixel at NDVI 0.65 was reported as "fair"
   and one at 0.25 as "severe stress". The same field, same composite, both
   ways -- vendored ``healthy 0.0 / fair 1.8 / stressed 7.9 / severe 90.3``
   against ``0.6 / 3.8 / 16.6 / 79.0`` here. Both sum to 100%, which is why the
   bug is invisible without a second implementation. Zones are the mean of a
   boolean mask instead: no bin edges to get wrong, and no silent discard of the
   pixels below -0.2 that fall outside the fixed histogram altogether.
2. **``reduceRegion`` can return nulls and the vendored code did arithmetic on
   them** (``mean_ndvi < 0.2`` raises on None, ``round(mean_ndvi - early_val)``
   invents a change). Every value here goes through
   :func:`cropup.evidence.fact_from_reduce_region`, so a masked region becomes
   ``Missing(MASKED)``.
3. **The composite reported the date of only the most recent scene.** The value
   is a median over every scene in the window, so this module measures the real
   window -- first scene, last scene, scene count -- and dates the Fact with the
   *median* scene date, with the full window in the note.

Per-pixel cloud masking with the SCL band is new: the vendored code filtered on
the scene-level ``CLOUDY_PIXEL_PERCENTAGE`` only, which leaves cloud inside an
otherwise clear scene in the median.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
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
    "S2_ASSET",
    "CORE_INDICES",
    "ALL_INDICES",
    "ZONE_BOUNDS",
    "CompositeWindow",
    "vegetation_quantities",
    "get_vegetation",
]

S2_ASSET = "COPERNICUS/S2_SR_HARMONIZED"
S2_RESOLUTION_M = 10.0

DEFAULT_WINDOW_DAYS = 60
DEFAULT_MAX_CLOUD_PCT = 60.0  # per-pixel SCL masking does the real work
DEFAULT_RADIUS_M = 300.0

# Scene-classification classes dropped before compositing. 4 (vegetation),
# 5 (bare), 6 (water), 7 (unclassified) and 2 (dark area) are kept: water is a
# real surface and NDWI is supposed to see it.
SCL_DROPPED: tuple[tuple[int, str], ...] = (
    (1, "saturated or defective"),
    (3, "cloud shadow"),
    (8, "cloud, medium probability"),
    (9, "cloud, high probability"),
    (10, "thin cirrus"),
    (11, "snow or ice"),
)

# quantity -> (bands used, formula as shown to the farmer, plausible range)
_INDEX_META: dict[str, tuple[str, str, tuple[float, float]]] = {
    "ndvi": ("B8/B4", "(B8 - B4) / (B8 + B4)", (-1.0, 1.0)),
    "ndmi": ("B8/B11", "(B8 - B11) / (B8 + B11)", (-1.0, 1.0)),
    "ndre": ("B8A/B5", "(B8A - B5) / (B8A + B5)", (-1.0, 1.0)),
    "ndwi": ("B3/B8", "(B3 - B8) / (B3 + B8)", (-1.0, 1.0)),
    "psri": ("B4/B2/B6", "(B4 - B2) / B6 on reflectance (DN / 10000)", (-5.0, 5.0)),
    "reci": ("B7/B5", "B7 / B5 - 1 on reflectance (DN / 10000)", (-1.0, 100.0)),
    "gci": ("B8A/B3", "B8A / B3 - 1 on reflectance (DN / 10000)", (-1.0, 100.0)),
    "evi": ("B8/B4/B2", "2.5 (B8 - B4) / (B8 + 6 B4 - 7.5 B2 + 1) on reflectance", (-1.0, 3.0)),
    "savi": ("B8/B4", "1.5 (B8 - B4) / (B8 + B4 + 0.5) on reflectance", (-1.5, 1.5)),
    "msavi": ("B8/B4", "(2 B8 + 1 - sqrt((2 B8 + 1)^2 - 8 (B8 - B4))) / 2 on reflectance", (-1.5, 1.5)),
    "gndvi": ("B8/B3", "(B8 - B3) / (B8 + B3)", (-1.0, 1.0)),
}

# These quantities have no chain in geo/registry.py: an index is computed from
# a composite, not read from a band, so the arithmetic lives here with the
# composite it belongs to. The reflectance scaling is the one registry.py
# records for COPERNICUS/S2_SR_HARMONIZED (raw / 10000).
CORE_INDICES: tuple[str, ...] = ("ndvi", "ndmi", "ndre", "ndwi", "psri", "reci", "gci")
ALL_INDICES: tuple[str, ...] = tuple(_INDEX_META)

# NDVI zone edges, lower bound inclusive. The names are deliberately neutral --
# turning "0.19 of the field below NDVI 0.2" into "severe stress" is an
# interpretation and belongs to analysis/, not to the data layer.
ZONE_BOUNDS: tuple[tuple[str, float | None, float | None], ...] = (
    ("healthy", 0.6, None),
    ("fair", 0.4, 0.6),
    ("stressed", 0.2, 0.4),
    ("severe", None, 0.2),
)

_YEAR_AGO_INDICES: tuple[str, ...] = ("ndvi", "ndmi", "ndre")


@dataclass(frozen=True)
class CompositeWindow:
    """The observations that actually went into a median composite.

    The vendored code kept only ``date_str`` of the newest scene; a median over
    24 scenes is not an observation from the newest of them.
    """

    asset_id: str
    scene_count: int
    first_date: dt.date
    last_date: dt.date
    median_date: dt.date
    requested_start: dt.date
    requested_end: dt.date
    max_cloud_pct: float

    @property
    def label(self) -> str:
        return (
            f"median of {self.scene_count} {self.asset_id} scenes observed "
            f"{self.first_date.isoformat()} to {self.last_date.isoformat()} "
            f"(scene cloud cover < {self.max_cloud_pct:g}%, per-pixel SCL cloud mask); "
            f"dated by the median scene date {self.median_date.isoformat()}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "scene_count": self.scene_count,
            "first_date": self.first_date.isoformat(),
            "last_date": self.last_date.isoformat(),
            "median_date": self.median_date.isoformat(),
            "requested_start": self.requested_start.isoformat(),
            "requested_end": self.requested_end.isoformat(),
            "max_cloud_pct": self.max_cloud_pct,
        }


def vegetation_quantities(
    indices: Sequence[str] = CORE_INDICES,
    *,
    zones: bool = True,
    compare_year_ago: bool = False,
) -> tuple[str, ...]:
    """Every quantity :func:`get_vegetation` would record for these options.

    Used to report the whole set as missing when Earth Engine is unavailable,
    so a degraded run names what it could not measure instead of going quiet.
    """
    out: list[str] = ["s2_scene_count", "s2_observations_per_pixel"]
    for name in indices:
        out += [name, f"{name}_p10", f"{name}_p90"]
    if "ndvi" in indices:
        out.append("ndvi_spread")
    if zones:
        out += [f"ndvi_zone_{zone}_pct" for zone, _, _ in ZONE_BOUNDS]
    if compare_year_ago:
        for name in _YEAR_AGO_INDICES:
            if name in indices:
                out += [f"{name}_year_ago", f"{name}_change_1y"]
    return tuple(out)


# --------------------------------------------------------------------------- EE helpers


def _geometry(ee: Any, lat: float, lon: float, radius_m: float) -> Any:
    """A circle around the field centre, not a single pixel.

    One 10 m pixel is smaller than any real field and is the difference between
    a number and a coincidence.
    """
    return ee.Geometry.Point(float(lon), float(lat)).buffer(float(radius_m))


def _masked_collection(ee: Any, geom: Any, start: dt.date, end: dt.date, max_cloud_pct: float) -> Any:
    """Sentinel-2 SR over the window, per-pixel cloud/shadow/cirrus masked."""
    collection = (
        ee.ImageCollection(S2_ASSET)
        .filterBounds(geom)
        .filterDate(start.isoformat(), end.isoformat())
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", float(max_cloud_pct)))
    )

    def mask(image: Any) -> Any:
        scl = image.select("SCL")
        keep = scl.neq(SCL_DROPPED[0][0])
        for code, _ in SCL_DROPPED[1:]:
            keep = keep.And(scl.neq(code))
        return image.updateMask(keep)

    return collection.map(mask)


def _window(ee: Any, collection: Any, start: dt.date, end: dt.date, max_cloud_pct: float) -> CompositeWindow | None:
    """The real observation window, read back from the scenes themselves."""
    stamps = collection.aggregate_array("system:time_start").getInfo() or []
    if not stamps:
        return None
    dates = sorted(dt.datetime.fromtimestamp(ms / 1000.0, dt.timezone.utc).date() for ms in stamps)
    return CompositeWindow(
        asset_id=S2_ASSET,
        scene_count=len(dates),
        first_date=dates[0],
        last_date=dates[-1],
        median_date=dates[len(dates) // 2],
        requested_start=start,
        requested_end=end,
        max_cloud_pct=float(max_cloud_pct),
    )


def _index_images(ee: Any, image: Any) -> dict[str, Any]:
    """Every index this module knows, computed from one composite image.

    ``S2_SR_HARMONIZED`` already removes the +1000 offset that ESA added with
    processing baseline 04.00, so reflectance is simply DN / 10000. Normalised
    differences are scale-free; the ratio indices are not, which is why they use
    the divided bands.
    """
    r = {band: image.select(band).divide(10000) for band in ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11")}
    two_nir_plus_one = r["B8"].multiply(2).add(1)
    return {
        "ndvi": image.normalizedDifference(["B8", "B4"]),
        "ndmi": image.normalizedDifference(["B8", "B11"]),
        "ndre": image.normalizedDifference(["B8A", "B5"]),
        "ndwi": image.normalizedDifference(["B3", "B8"]),
        "psri": r["B4"].subtract(r["B2"]).divide(r["B6"]),
        "reci": r["B7"].divide(r["B5"]).subtract(1),
        "gci": r["B8A"].divide(r["B3"]).subtract(1),
        "evi": r["B8"].subtract(r["B4"]).multiply(2.5).divide(
            r["B8"].add(r["B4"].multiply(6)).subtract(r["B2"].multiply(7.5)).add(1)
        ),
        "savi": r["B8"].subtract(r["B4"]).divide(r["B8"].add(r["B4"]).add(0.5)).multiply(1.5),
        "msavi": two_nir_plus_one.subtract(
            two_nir_plus_one.pow(2).subtract(r["B8"].subtract(r["B4"]).multiply(8)).sqrt()
        ).divide(2),
        "gndvi": image.normalizedDifference(["B8", "B3"]),
    }


def _index_sources(
    indices: Sequence[str], stale_after_days: int | None, window: CompositeWindow
) -> list[SourceDescriptor]:
    """One descriptor per index mean and per percentile, all from one reduce."""
    sources: list[SourceDescriptor] = []
    for name in indices:
        bands, formula, valid = _INDEX_META[name]
        for suffix, key_suffix, quantity_suffix in (("mean", "_mean", ""), ("10th percentile", "_p10", "_p10"), ("90th percentile", "_p90", "_p90")):
            sources.append(
                SourceDescriptor(
                    quantity=f"{name}{quantity_suffix}",
                    asset_id=S2_ASSET,
                    unit="index",
                    band=bands,
                    result_key=f"{name}{key_suffix}",
                    resolution_m=S2_RESOLUTION_M,
                    scaling=f"{formula}; {suffix} over the field of a {window.scene_count}-scene median composite",
                    valid_range=valid,
                    stale_after_days=stale_after_days,
                )
            )
    return sources


def _zone_sources(stale_after_days: int, window: CompositeWindow) -> list[SourceDescriptor]:
    """NDVI zone shares, as a percentage of the unmasked field."""
    sources: list[SourceDescriptor] = []
    for zone, low, high in ZONE_BOUNDS:
        if low is None:
            rule = f"NDVI < {high:g}"
        elif high is None:
            rule = f"NDVI >= {low:g}"
        else:
            rule = f"{low:g} <= NDVI < {high:g}"
        sources.append(
            SourceDescriptor(
                quantity=f"ndvi_zone_{zone}_pct",
                asset_id=S2_ASSET,
                unit="%",
                band="B8/B4",
                result_key=f"zone_{zone}",
                resolution_m=S2_RESOLUTION_M,
                scaling=f"share of unmasked pixels with {rule}, as a percentage (mean of the 0/1 mask x 100)",
                transform=lambda value: value * 100.0,
                valid_range=(0.0, 100.0),
                stale_after_days=stale_after_days,
            )
        )
    return sources


def _zone_images(ee: Any, ndvi: Any) -> list[Any]:
    """A 0/1 band per zone; the mean of each is the zone's share of the field."""
    bands = []
    for zone, low, high in ZONE_BOUNDS:
        if low is None:
            mask = ndvi.lt(high)
        elif high is None:
            mask = ndvi.gte(low)
        else:
            mask = ndvi.gte(low).And(ndvi.lt(high))
        bands.append(mask.rename(f"zone_{zone}"))
    return bands


def _composite_stats(
    ee: Any,
    collection: Any,
    geom: Any,
    indices: Sequence[str],
    *,
    zones: bool,
) -> dict[str, Any] | None:
    """One batched ``reduceRegion`` for every index, percentile and zone.

    SPEC 3.4: one round trip per geometry, not one per number.
    """
    composite = collection.median()
    images = _index_images(ee, composite)
    stack = images[indices[0]].rename(indices[0])
    for name in indices[1:]:
        stack = stack.addBands(images[name].rename(name))

    stats = stack.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.percentile([10, 90]), sharedInputs=True),
        geometry=geom,
        scale=S2_RESOLUTION_M,
        bestEffort=True,
        maxPixels=int(1e9),
    )

    # Counts and zone shares need a plain mean, so they travel in a second
    # reducer merged into the same request rather than a second round trip.
    extra = collection.select("B8").count().rename("s2_observations_per_pixel")
    if zones:
        for band in _zone_images(ee, images["ndvi"]):
            extra = extra.addBands(band)
    shares = extra.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=S2_RESOLUTION_M,
        bestEffort=True,
        maxPixels=int(1e9),
    )
    return ee.Dictionary(stats).combine(shares).getInfo()


# --------------------------------------------------------------------------- public API


def get_vegetation(
    lat: float,
    lon: float,
    *,
    geometry: Any | None = None,
    radius_m: float | None = None,
    end_date: dt.date | str | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    indices: Sequence[str] = CORE_INDICES,
    zones: bool = True,
    compare_year_ago: bool = False,
    max_cloud_pct: float = DEFAULT_MAX_CLOUD_PCT,
    settings: Settings | None = None,
    ledger: Ledger | None = None,
) -> Ledger:
    """Vegetation indices for one field, as Facts with their observation window.

    ``geometry`` overrides the buffered point when the farmer has drawn a field.
    Everything that could not be measured is recorded as a :class:`Missing`, so
    the caller never has to guess whether a quantity is absent or forgotten.
    """
    settings = settings or get_settings()
    ledger = ledger if ledger is not None else Ledger(turn="vegetation")
    unknown = [name for name in indices if name not in _INDEX_META]
    if unknown:
        raise ValueError(f"unknown vegetation index/indices: {', '.join(sorted(unknown))}")
    if not indices:
        raise ValueError("at least one index must be requested")

    radius_m = float(radius_m if radius_m is not None else settings.ee_neighbourhood_m or DEFAULT_RADIUS_M)
    end = _as_date(end_date) or dt.date.today()
    start = end - dt.timedelta(days=int(window_days))
    stale_after_days = settings.default_stale_after_days
    expected = vegetation_quantities(indices, zones=zones, compare_year_ago=compare_year_ago)

    try:
        ee = bootstrap.require_ee()
    except EarthEngineUnavailable as exc:
        for quantity in expected:
            ledger.gap(quantity, MissingReason.SOURCE_FAILED, (S2_ASSET,), str(exc))
        return ledger

    geom = geometry if geometry is not None else _geometry(ee, lat, lon, radius_m)

    _add_window(
        ee,
        ledger,
        geom,
        start,
        end,
        indices=indices,
        zones=zones,
        max_cloud_pct=max_cloud_pct,
        stale_after_days=stale_after_days,
        suffix="",
    )

    if compare_year_ago:
        wanted = [name for name in _YEAR_AGO_INDICES if name in indices]
        if wanted:
            _add_year_ago(
                ee,
                ledger,
                geom,
                start - dt.timedelta(days=365),
                end - dt.timedelta(days=365),
                indices=wanted,
                max_cloud_pct=max_cloud_pct,
            )
    return ledger


def _add_window(
    ee: Any,
    ledger: Ledger,
    geom: Any,
    start: dt.date,
    end: dt.date,
    *,
    indices: Sequence[str],
    zones: bool,
    max_cloud_pct: float,
    stale_after_days: int | None,
    suffix: str,
) -> CompositeWindow | None:
    """Measure one composite window and record everything it yields."""
    # Everything this call would have produced, so that a window with no usable
    # scene names each quantity as missing. ndvi_spread belongs here even though
    # it is derived further down: leaving it out would drop it from the ledger
    # entirely, which reads as "never asked for" rather than "could not measure".
    quantities = [f"{name}{suffix}" for name in indices]
    if not suffix:
        quantities += [f"{name}_p10" for name in indices] + [f"{name}_p90" for name in indices]
        quantities += ["s2_scene_count", "s2_observations_per_pixel"]
        if "ndvi" in indices:
            quantities.append("ndvi_spread")
        if zones:
            quantities += [f"ndvi_zone_{zone}_pct" for zone, _, _ in ZONE_BOUNDS]

    try:
        collection = _masked_collection(ee, geom, start, end, max_cloud_pct)
        window = _window(ee, collection, start, end, max_cloud_pct)
    except Exception as exc:  # network, quota, or a bad asset: never a number
        for quantity in quantities:
            ledger.gap(quantity, MissingReason.SOURCE_FAILED, (S2_ASSET,), f"{type(exc).__name__}: {exc}")
        return None

    if window is None:
        detail = (
            f"no Sentinel-2 scene with cloud cover < {max_cloud_pct:g}% between "
            f"{start.isoformat()} and {end.isoformat()}"
        )
        for quantity in quantities:
            ledger.gap(quantity, MissingReason.MASKED, (S2_ASSET,), detail)
        return None

    try:
        result = _composite_stats(ee, collection, geom, indices, zones=zones and not suffix)
    except Exception as exc:
        for quantity in quantities:
            ledger.gap(quantity, MissingReason.SOURCE_FAILED, (S2_ASSET,), f"{type(exc).__name__}: {exc}")
        return None

    sources = _index_sources(indices, stale_after_days, window)
    if suffix:
        sources = [
            replace(source, quantity=f"{source.quantity}{suffix}")
            for source in sources
            if source.result_key.endswith("_mean")
        ]
    evidence = facts_from_reduce_region(result, sources, observed_on=window.median_date)

    for item in evidence.values():
        ledger.add(_with_note(item, window.label))

    if not suffix:
        _add_scene_facts(ledger, result, window, stale_after_days)
        if zones:
            zone_evidence = facts_from_reduce_region(
                result, _zone_sources(stale_after_days, window), observed_on=window.median_date
            )
            for item in zone_evidence.values():
                ledger.add(_with_note(item, window.label))
        _add_spread(ledger, indices, window)
    return window


def _add_scene_facts(ledger: Ledger, result: dict[str, Any], window: CompositeWindow, stale_after_days: int) -> None:
    """How much imagery the composite actually rests on."""
    ledger.record(
        Fact(
            quantity="s2_scene_count",
            value=window.scene_count,
            unit="scenes",
            source_asset=S2_ASSET,
            observed_on=window.median_date,
            resolution_m=S2_RESOLUTION_M,
            scaling_applied="count of scenes in the filtered collection",
            stale_after_days=stale_after_days,
            note=window.label,
            precision=0,
        )
    )
    source = SourceDescriptor(
        quantity="s2_observations_per_pixel",
        asset_id=S2_ASSET,
        unit="observations",
        band="B8",
        result_key="s2_observations_per_pixel",
        resolution_m=S2_RESOLUTION_M,
        scaling="mean number of cloud-free observations per pixel behind the median",
        valid_range=(0.0, 1000.0),
        stale_after_days=stale_after_days,
    )
    ledger.add(_with_note(fact_from_reduce_region(result, source, observed_on=window.median_date), window.label))


def _add_spread(ledger: Ledger, indices: Sequence[str], window: CompositeWindow) -> None:
    """NDVI p90 - p10: how uneven the field is. Derived, so it carries both."""
    if "ndvi" not in indices:
        return
    high = ledger.fact("ndvi_p90")
    low = ledger.fact("ndvi_p10")
    if high is None or low is None:
        ledger.gap(
            "ndvi_spread",
            MissingReason.MASKED,
            (S2_ASSET,),
            "needs both the 10th and the 90th NDVI percentile; one of them is missing",
        )
        return
    ledger.record(
        Fact.derive(
            "ndvi_spread",
            high.value - low.value,
            "index",
            [high, low],
            scaling_applied="90th percentile minus 10th percentile of NDVI across the field",
            note="within-field NDVI spread; " + window.label,
        )
    )


def _add_year_ago(
    ee: Any,
    ledger: Ledger,
    geom: Any,
    start: dt.date,
    end: dt.date,
    *,
    indices: Sequence[str],
    max_cloud_pct: float,
) -> None:
    """The same indices one year earlier, plus the change between the two.

    The year-ago facts carry no freshness limit: they are a deliberate
    historical reference, and flagging them stale would report a working
    comparison as a degraded one.
    """
    _add_window(
        ee,
        ledger,
        geom,
        start,
        end,
        indices=indices,
        zones=False,
        max_cloud_pct=max_cloud_pct,
        stale_after_days=None,
        suffix="_year_ago",
    )
    for name in indices:
        now = ledger.fact(name)
        then = ledger.fact(f"{name}_year_ago")
        if now is None or then is None:
            ledger.gap(
                f"{name}_change_1y",
                MissingReason.MASKED,
                (S2_ASSET,),
                f"needs {name} for both windows; "
                + ("this year is missing" if now is None else "the year-ago window is missing"),
            )
            continue
        change = Fact.derive(
            f"{name}_change_1y",
            now.value - then.value,
            "index",
            [now, then],
            scaling_applied=f"{name} now minus {name} one year earlier",
            note=(
                f"{now.note or 'current composite'} MINUS {then.note or 'year-ago composite'}"
            ),
        )
        # derive() inherits the oldest ingredient's date, which is the point of a
        # year-on-year change; a 45-day freshness limit would call it stale.
        ledger.record(replace(change, stale_after_days=None))


def _with_note(item: Fact | Missing, note: str) -> Fact | Missing:
    """Attach the composite window to whatever came back."""
    if isinstance(item, Fact):
        return item if item.note else replace(item, note=note)
    return item if item.detail else replace(item, detail=note)


def _as_date(value: dt.date | str | None) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])
