"""Landsat 8/9 land surface temperature, QA-masked.

Ported from ``Data/googlebuildathonfarmers-main/app/earth_engine/thermal.py``
(``get_ecostress_data``, which never touched ECOSTRESS). What changed:

* **QA masking, which the vendored version did not do at all.** It took the most
  recent scene with scene-level ``CLOUD_COVER < 30`` and averaged ``ST_B10`` over
  the field, clouds included. Measured over a 300 m field at Arusha on
  2026-09-17: the newest scene (2026-09-08) is 96.3% cloud by ``QA_PIXEL`` and
  averages 24.0 degC unmasked against 34.7 degC over its clear pixels. The
  vendored rule skips it on ``CLOUD_COVER`` (77.9) and lands on 2026-08-07, whose
  scene-level cloud is an innocuous 23.9% but whose clear fraction *over this
  field* is zero: it would report 21.7 degC computed entirely from cloud tops.
  This module reports 32.9 degC from 2026-06-28, the newest scene with the field
  actually visible, dated so the farmer can see it is 81 days old.
* **The MODIS ET/PET block is gone.** ``MOD16A2GF`` is permanently masked at
  Arusha (SPEC 3.3); evapotranspiration now lives in :mod:`cropup.geo.water`,
  which uses SSEBop VIIRS.
* Scene selection is honest: the clear-sky fraction of the field is measured for
  each candidate scene in one round trip, and a scene is used only if enough of
  the field is actually visible. If none is, this returns ``Missing(MASKED)``
  naming the scenes tried -- it does not fall back to a cloudy average.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from typing import Any

from .. import bootstrap
from ..config import Settings, get_settings
from ..errors import EarthEngineUnavailable
from ..evidence import (
    Fact,
    Ledger,
    MissingReason,
    SourceDescriptor,
    facts_from_reduce_region,
)

__all__ = [
    "LANDSAT_COLLECTIONS",
    "QA_PIXEL_BITS",
    "ThermalScene",
    "thermal_quantities",
    "get_land_surface_temperature",
]

# LC09 first: it is the newer satellite, but both are searched and the Fact
# names whichever scene was actually used.
LANDSAT_COLLECTIONS: tuple[str, ...] = ("LANDSAT/LC09/C02/T1_L2", "LANDSAT/LC08/C02/T1_L2")
LST_RESOLUTION_M = 30.0  # the ST_B10 product is served on the 30 m L2 grid (100 m native)

# Collection 2 Level-2 QA_PIXEL bit flags; a pixel is used only if all are clear.
QA_PIXEL_BITS: tuple[tuple[int, str], ...] = (
    (0, "fill"),
    (1, "dilated cloud"),
    (2, "cirrus"),
    (3, "cloud"),
    (4, "cloud shadow"),
)

# Collection 2 Level-2 surface temperature: Kelvin = DN * 0.00341802 + 149.0.
ST_SCALE = 0.00341802
ST_OFFSET = 149.0
ST_QA_SCALE = 0.01  # ST_QA is uncertainty in Kelvin * 100

DEFAULT_WINDOW_DAYS = 90
DEFAULT_MAX_SCENES = 12
DEFAULT_MIN_CLEAR_FRACTION = 0.20
DEFAULT_RADIUS_M = 300.0

# Quantity names follow the chain names in geo/registry.py so that analysis/
# sees one vocabulary; the descriptors are built here because the mask and the
# scene choice are per-call facts that a static table cannot hold.
QUANTITIES: tuple[str, ...] = (
    "land_surface_temperature",
    "land_surface_temperature_p10",
    "land_surface_temperature_p90",
    "land_surface_temperature_uncertainty_k",
    "land_surface_temperature_clear_fraction",
)


@dataclass(frozen=True)
class ThermalScene:
    """One candidate scene and what it offered over this field."""

    scene_id: str
    asset_id: str
    spacecraft: str
    observed_on: dt.date
    scene_cloud_pct: float | None
    clear_fraction: float | None
    lst_c: float | None

    @property
    def usable(self) -> bool:
        return self.lst_c is not None and self.clear_fraction is not None

    def describe(self) -> str:
        clear = "no clear pixel" if not self.clear_fraction else f"{self.clear_fraction * 100:.0f}% clear"
        return f"{self.scene_id} ({self.observed_on.isoformat()}, {clear})"


def thermal_quantities() -> tuple[str, ...]:
    """Everything :func:`get_land_surface_temperature` records when it works."""
    return QUANTITIES


def _geometry(ee: Any, lat: float, lon: float, radius_m: float) -> Any:
    return ee.Geometry.Point(float(lon), float(lat)).buffer(float(radius_m))


def _clear_mask(ee: Any, image: Any) -> Any:
    """1 where every QA_PIXEL cloud flag is off, 0 where any is set."""
    qa = image.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << QA_PIXEL_BITS[0][0]).eq(0)
    for bit, _ in QA_PIXEL_BITS[1:]:
        mask = mask.And(qa.bitwiseAnd(1 << bit).eq(0))
    return mask


def _scene_stats(ee: Any, geom: Any, start: dt.date, end: dt.date, max_scenes: int) -> list[dict[str, Any]]:
    """Clear fraction and masked LST for each recent scene, in ONE round trip.

    Trying scenes one at a time would be one ``getInfo`` per scene; mapping the
    reduction over the collection costs a single request for all of them (SPEC
    3.4), and gives the farmer-facing "we tried these scenes" list for free.
    """
    collection = ee.ImageCollection(LANDSAT_COLLECTIONS[0])
    for asset_id in LANDSAT_COLLECTIONS[1:]:
        collection = collection.merge(ee.ImageCollection(asset_id))
    collection = (
        collection.filterBounds(geom)
        .filterDate(start.isoformat(), end.isoformat())
        .sort("system:time_start", False)
        .limit(int(max_scenes))
    )

    def per_scene(image: Any) -> Any:
        image = ee.Image(image)
        clear = _clear_mask(ee, image)
        lst = (
            image.select("ST_B10")
            .multiply(ST_SCALE)
            .add(ST_OFFSET)
            .subtract(273.15)
            .updateMask(clear)
            .rename("land_surface_temperature")
        )
        uncertainty = image.select("ST_QA").multiply(ST_QA_SCALE).updateMask(clear).rename("land_surface_temperature_uncertainty_k")
        stats = (
            lst.addBands(uncertainty)
            .reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.percentile([10, 90]), sharedInputs=True),
                geometry=geom,
                scale=LST_RESOLUTION_M,
                bestEffort=True,
                maxPixels=int(1e9),
            )
            .combine(
                # Not masked by `clear`, so its mean is the clear-sky share of the field.
                clear.rename("land_surface_temperature_clear_fraction").reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geom,
                    scale=LST_RESOLUTION_M,
                    bestEffort=True,
                    maxPixels=int(1e9),
                )
            )
        )
        return ee.Feature(
            None,
            stats.combine(
                {
                    "scene_id": image.get("system:id"),
                    "observed_on": image.date().format("YYYY-MM-dd"),
                    "scene_cloud_pct": image.get("CLOUD_COVER"),
                    "spacecraft": image.get("SPACECRAFT_ID"),
                }
            ),
        )

    features = ee.FeatureCollection(collection.map(per_scene)).getInfo()
    return [feature["properties"] for feature in features.get("features", [])]


def _to_scene(properties: dict[str, Any]) -> ThermalScene:
    scene_id = str(properties.get("scene_id") or "")
    asset_id = scene_id.rsplit("/", 1)[0] if "/" in scene_id else LANDSAT_COLLECTIONS[0]
    return ThermalScene(
        scene_id=scene_id.rsplit("/", 1)[-1] or "unknown",
        asset_id=asset_id,
        spacecraft=str(properties.get("spacecraft") or "unknown"),
        observed_on=dt.date.fromisoformat(str(properties["observed_on"])[:10]),
        scene_cloud_pct=properties.get("scene_cloud_pct"),
        clear_fraction=properties.get("land_surface_temperature_clear_fraction"),
        lst_c=properties.get("land_surface_temperature_mean"),
    )


def _sources(scene: ThermalScene, stale_after_days: int) -> list[SourceDescriptor]:
    common = {
        "asset_id": scene.asset_id,
        "resolution_m": LST_RESOLUTION_M,
        "stale_after_days": stale_after_days,
        "coverage": "global land, 8-day repeat per satellite (16-day per platform)",
    }
    scaling = (
        f"ST_B10 x {ST_SCALE} + {ST_OFFSET} K - 273.15; pixels with any QA_PIXEL "
        f"cloud flag ({', '.join(name for _, name in QA_PIXEL_BITS)}) removed"
    )
    return [
        SourceDescriptor(
            quantity="land_surface_temperature",
            unit="degC",
            band="ST_B10",
            result_key="land_surface_temperature_mean",
            scaling=f"field mean of {scaling}",
            valid_range=(-40.0, 75.0),
            **common,
        ),
        SourceDescriptor(
            quantity="land_surface_temperature_p10",
            unit="degC",
            band="ST_B10",
            result_key="land_surface_temperature_p10",
            scaling=f"10th percentile of {scaling}",
            valid_range=(-40.0, 75.0),
            **common,
        ),
        SourceDescriptor(
            quantity="land_surface_temperature_p90",
            unit="degC",
            band="ST_B10",
            result_key="land_surface_temperature_p90",
            scaling=f"90th percentile of {scaling}",
            valid_range=(-40.0, 75.0),
            **common,
        ),
        SourceDescriptor(
            quantity="land_surface_temperature_uncertainty_k",
            unit="K",
            band="ST_QA",
            result_key="land_surface_temperature_uncertainty_k_mean",
            scaling=f"ST_QA x {ST_QA_SCALE}: the product's own 1-sigma uncertainty, same mask",
            valid_range=(0.0, 30.0),
            **common,
        ),
        SourceDescriptor(
            quantity="land_surface_temperature_clear_fraction",
            unit="ratio",
            band="QA_PIXEL",
            result_key="land_surface_temperature_clear_fraction",
            scaling="share of the field with no QA_PIXEL cloud flag set",
            valid_range=(0.0, 1.0),
            **common,
        ),
    ]


def get_land_surface_temperature(
    lat: float,
    lon: float,
    *,
    geometry: Any | None = None,
    radius_m: float | None = None,
    end_date: dt.date | str | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    max_scenes: int = DEFAULT_MAX_SCENES,
    min_clear_fraction: float = DEFAULT_MIN_CLEAR_FRACTION,
    settings: Settings | None = None,
    ledger: Ledger | None = None,
) -> Ledger:
    """Land surface temperature over one field from the freshest usable scene.

    "Usable" means at least ``min_clear_fraction`` of the field is cloud-free in
    that scene. The Fact carries the scene date, so an old but clear scene is
    reported as old rather than quietly presented as today's temperature.
    """
    settings = settings or get_settings()
    ledger = ledger if ledger is not None else Ledger(turn="thermal")
    radius_m = float(radius_m if radius_m is not None else settings.ee_neighbourhood_m or DEFAULT_RADIUS_M)
    end = _as_date(end_date) or dt.date.today()
    start = end - dt.timedelta(days=int(window_days))
    stale_after_days = settings.default_stale_after_days

    try:
        ee = bootstrap.require_ee()
    except EarthEngineUnavailable as exc:
        for quantity in QUANTITIES:
            ledger.gap(quantity, MissingReason.SOURCE_FAILED, LANDSAT_COLLECTIONS, str(exc))
        return ledger

    geom = geometry if geometry is not None else _geometry(ee, lat, lon, radius_m)

    try:
        raw = _scene_stats(ee, geom, start, end, max_scenes)
    except Exception as exc:  # network, quota, missing asset
        for quantity in QUANTITIES:
            ledger.gap(quantity, MissingReason.SOURCE_FAILED, LANDSAT_COLLECTIONS, f"{type(exc).__name__}: {exc}")
        return ledger

    # The row travels with the scene: two Landsat paths can cover one field on
    # the same day, and picking the row back out by date alone would then read
    # the temperatures of one scene while reporting the clear-sky fraction and
    # the date of the other.
    candidates = sorted(
        ((_to_scene(row), row) for row in raw),
        key=lambda pair: pair[0].observed_on,
        reverse=True,
    )
    scenes = [scene for scene, _ in candidates]
    if not scenes:
        detail = f"no Landsat 8/9 Level-2 scene covers this field between {start.isoformat()} and {end.isoformat()}"
        for quantity in QUANTITIES:
            ledger.gap(quantity, MissingReason.MASKED, LANDSAT_COLLECTIONS, detail)
        return ledger

    chosen, row = next(
        ((s, r) for s, r in candidates if s.usable and (s.clear_fraction or 0.0) >= min_clear_fraction),
        (None, None),
    )
    if chosen is None:
        best = max(scenes, key=lambda s: s.clear_fraction or 0.0)
        detail = (
            f"none of the {len(scenes)} Landsat scenes since {scenes[-1].observed_on.isoformat()} has "
            f"{min_clear_fraction * 100:.0f}% of the field cloud-free; the best was {best.describe()}"
        )
        for quantity in QUANTITIES:
            ledger.gap(quantity, MissingReason.MASKED, LANDSAT_COLLECTIONS, detail)
        return ledger

    skipped = [s for s in scenes if s.observed_on > chosen.observed_on]
    note = (
        f"scene {chosen.scene_id} from {chosen.spacecraft} on {chosen.observed_on.isoformat()}, "
        f"{(chosen.clear_fraction or 0.0) * 100:.0f}% of the field cloud-free"
    )
    if skipped:
        note += (
            f"; {len(skipped)} more recent scene(s) skipped below the "
            f"{min_clear_fraction * 100:.0f}% clear-sky threshold: "
            + ", ".join(s.describe() for s in skipped)
        )

    evidence = facts_from_reduce_region(row, _sources(chosen, stale_after_days), observed_on=chosen.observed_on)
    for item in evidence.values():
        ledger.add(replace(item, note=note) if isinstance(item, Fact) else replace(item, detail=note))
    return ledger


def _as_date(value: dt.date | str | None) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])
