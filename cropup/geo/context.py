"""Field context: is this cropland, what is growing around it, irrigated or rainfed.

Everything here answers in one batched ``reduceRegion`` per point (SPEC 3.4) and
every answer comes back as a ``Fact`` or a ``Missing``.

Source chains (SPEC 3.1), and the traps each one carries:

* **is_cropland** -- ``DEAF/CROPLAND-EXTENT/prob`` (10 m, Africa only, the only
  per-pixel crop *probability* available in Tanzania) -> ``GFSAD/GCEP30`` (30 m,
  global) -> ``ESA/WorldCover/v200`` (10 m, global).
  DEAF probabilities are **0-100 and need dividing by 100**; the NASA Harvest
  probability layers in the same catalog family are already 0-1 and must not be
  divided, which is why the scaling lives on the descriptor and not in a shared
  "probability" helper. DEAF's NoData/Fill is **0**, indistinguishable from a
  genuine "no trees voted crop", so a raw 0 is treated as no data and the chain
  moves on rather than reporting a confident 0%.
  GCEP30's class table is inverted relative to DEAF: **1 means NOT cropland**
  there, 2 means cropland.

* **land_cover** -- ``ESA/WorldCover/v200``, a one-image collection, global, and
  the only layer here that never returns null on land.

* **irrigated vs rainfed** -- ``GFSAD/LGRIP30`` (0 water, 1 non-cropland,
  2 irrigated, 3 rainfed). Point sampling returns class 1 at all three CropUp
  test points, so the exact pixel alone answers nothing; the class *fractions*
  over a 5 km neighbourhood are measured as well. LGRIP's accuracy is visibly
  poor where it can be ground-checked -- it calls much of the drip-irrigated
  Salinas Valley rainfed and calls rural Arusha irrigated -- so every irrigation
  Fact carries that hedge in its note.

Neither GCEP30 nor LGRIP30 carries a ``system:time_start``: their Facts are
therefore undated (``observed_on`` is None, ``is_stale`` is None = unknown) with
the catalog's nominal epoch in the note. Inventing 2015-01-01 for them would be
a fabricated observation date.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Sequence

from .. import bootstrap
from ..config import get_settings
from ..errors import EarthEngineUnavailable
from ..evidence import (
    Fact,
    Ledger,
    Missing,
    MissingReason,
    SourceDescriptor,
    fact_from_reduce_region,
)

__all__ = [
    "DEAF_CROPLAND_PROB",
    "GFSAD_GCEP30",
    "GFSAD_LGRIP30",
    "ESA_WORLDCOVER",
    "WORLDCOVER_CLASSES",
    "GCEP30_CLASSES",
    "LGRIP30_CLASSES",
    "get_field_context",
    "is_cropland",
    "get_irrigation_regime",
    "clear_cache",
]

DEAF_CROPLAND_PROB = "projects/sat-io/open-datasets/DEAF/CROPLAND-EXTENT/prob"
GFSAD_GCEP30 = "projects/sat-io/open-datasets/GFSAD/GCEP30"
GFSAD_LGRIP30 = "projects/sat-io/open-datasets/GFSAD/LGRIP30"
ESA_WORLDCOVER = "ESA/WorldCover/v200"

# gee:classes from the Earth Engine STAC entry.
WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare or sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}
WORLDCOVER_CROPLAND_CLASS = 40

# GCEP30 docs page: note that 1 is NOT cropland here, unlike DEAF.
GCEP30_CLASSES = {0: "Ocean or water body", 1: "Non-cropland", 2: "Cropland"}
GCEP30_CROPLAND_CLASS = 2

LGRIP30_CLASSES = {
    0: "Ocean or water body",
    1: "Non-cropland",
    2: "Irrigated cropland",
    3: "Rainfed cropland",
}
LGRIP_IRRIGATED_CLASS = 2
LGRIP_RAINFED_CLASS = 3

# DEAF publishes prob > 50 as its mask band; verified pixel by pixel (65 -> 1,
# 47 -> 0, 42 -> 0), so this is the dataset's own threshold, not one invented here.
CROPLAND_PROBABILITY_THRESHOLD = 0.50

# LGRIP is 30 m and returns class 1 at a dropped pin far more often than not, so
# the irrigation verdict is read from class fractions over this radius.
IRRIGATION_NEIGHBOURHOOD_M = 5000.0
# Below this share of mapped cropland in the neighbourhood there is no irrigation
# signal to report, and above it one class has to lead the other by this factor
# before the neighbourhood is allowed to speak for the field.
MIN_CROPLAND_FRACTION = 0.05
MIN_REGIME_RATIO = 2.0

_LGRIP_HEDGE = (
    "LGRIP30 is a weak signal: where it can be ground-checked it calls much of the "
    "drip-irrigated Salinas Valley rainfed and calls rural Arusha irrigated"
)
_NO_TIMESTAMP_NOTE = (
    "the asset carries no system:time_start, so this value cannot be dated; the catalog "
    "documents a nominal 2015 epoch built from Landsat 8 imagery of 2014-2017"
)

_DEAF_SCALING = (
    "divide by 100: DEAF probabilities are 0-100 (the NASA Harvest layers in the same "
    "family are already 0-1 and must not be divided)"
)
_NO_SCALING = "none: raw integer class code"


# ---------------------------------------------------------------------------
# in-process cache
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_cache: dict[tuple, tuple[float, dict[str, Any]]] = {}


def clear_cache() -> None:
    """Drop the in-process reduceRegion cache. For tests and long-lived servers."""
    with _cache_lock:
        _cache.clear()


def _cache_get(key: tuple) -> dict[str, Any] | None:
    ttl = get_settings().ee_cache_ttl_s
    with _cache_lock:
        hit = _cache.get(key)
        if hit is None:
            return None
        stored_at, payload = hit
        if time.monotonic() - stored_at > ttl:
            del _cache[key]
            return None
        return payload


def _cache_put(key: tuple, payload: dict[str, Any]) -> None:
    limit = max(1, get_settings().ee_cache_max_entries)
    with _cache_lock:
        while len(_cache) >= limit:
            _cache.pop(next(iter(_cache)))
        _cache[key] = (time.monotonic(), payload)


# ---------------------------------------------------------------------------
# the batched Earth Engine call
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Layer:
    """One tiled collection sampled at the point."""

    asset: str
    key: str
    scale: float
    dated: bool  # does the asset carry system:time_start?


_LAYERS = (
    _Layer(DEAF_CROPLAND_PROB, "deaf_prob", 10.0, True),
    _Layer(GFSAD_GCEP30, "gcep_class", 30.0, False),
    _Layer(GFSAD_LGRIP30, "lgrip_class", 30.0, False),
)


def _fetch(lat: float, lon: float) -> dict[str, Any]:
    """One getInfo with every context layer in it. Raises on Earth Engine trouble."""
    key = (round(lat, 5), round(lon, 5), IRRIGATION_NEIGHBOURHOOD_M)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    ee = bootstrap.require_ee()
    point = ee.Geometry.Point([lon, lat])

    def sampled(asset: str, name: str, geometry: Any) -> Any:
        """A mosaic that always carries a band called ``name``.

        Outside a collection's coverage ``filterBounds`` returns nothing and
        ``mosaic()`` then has no bands at all, which makes ``select`` raise and
        takes the whole batch down with it. A fully masked constant keeps the
        band present; the tile count below says whether the absence was coverage
        or a masked pixel.
        """
        collection = ee.ImageCollection(asset).filterBounds(geometry)
        placeholder = ee.Image.constant(0).selfMask().rename([name]).toFloat()
        return ee.ImageCollection([placeholder]).merge(
            collection.map(lambda image: image.toFloat().rename([name]))
        ).mosaic()

    out = ee.Dictionary({})
    for layer in _LAYERS:
        scoped = ee.ImageCollection(layer.asset).filterBounds(point)
        out = out.set(layer.key + "__tiles", scoped.size())
        if layer.dated:
            out = out.set(layer.key + "__time_start", scoped.aggregate_array("system:time_start").slice(0, 1))
        out = out.combine(
            sampled(layer.asset, layer.key, point).reduceRegion(ee.Reducer.first(), point, layer.scale)
        )

    world = ee.ImageCollection(ESA_WORLDCOVER).first()  # a one-image collection
    out = out.set("worldcover__time_start", world.get("system:time_start"))
    out = out.combine(world.select(["Map"], ["worldcover_class"]).reduceRegion(ee.Reducer.first(), point, 10.0))

    neighbourhood = point.buffer(IRRIGATION_NEIGHBOURHOOD_M)
    out = out.combine(
        sampled(GFSAD_LGRIP30, "lgrip_histogram", neighbourhood).reduceRegion(
            ee.Reducer.frequencyHistogram(), neighbourhood, 30.0, maxPixels=int(1e8)
        )
    )

    result = out.getInfo()
    _cache_put(key, result)
    return result


def _epoch_to_date(millis: Any) -> str | None:
    """system:time_start (ms) as an ISO date, or None when the asset has none."""
    if isinstance(millis, list):
        millis = millis[0] if millis else None
    if not isinstance(millis, (int, float)):
        return None
    return time.strftime("%Y-%m-%d", time.gmtime(millis / 1000.0))


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------

_CROPLAND_CHAIN = (DEAF_CROPLAND_PROB, GFSAD_GCEP30, ESA_WORLDCOVER)


def _class_label(value: float, table: dict[int, str]) -> str | None:
    return table.get(int(value))


def _cropland_probability(result: dict[str, Any]) -> Fact | Missing:
    quantity = "cropland_probability"
    if not result.get("deaf_prob__tiles"):
        return Missing(
            quantity,
            MissingReason.OUT_OF_COVERAGE,
            (DEAF_CROPLAND_PROB,),
            "DEAF covers Africa only and no tile intersects this point",
        )
    source = SourceDescriptor(
        quantity=quantity,
        asset_id=DEAF_CROPLAND_PROB,
        unit="",
        band="b1",
        result_key="deaf_prob",
        resolution_m=10.0,
        chain=_CROPLAND_CHAIN,
        chain_position=0,
        scaling=_DEAF_SCALING,
        transform=lambda raw: raw / 100.0,
        valid_range=(0.0, 1.0),
        # NoData/Fill is 0 in all three DEAF bands, so a raw 0 cannot be told
        # apart from "no random-forest tree voted crop". Treated as no data.
        invalid_values=(0.0,),
        coverage="Africa only",
    )
    observed_on = _epoch_to_date(result.get("deaf_prob__time_start"))
    if observed_on is None:
        return Missing(
            quantity,
            MissingReason.SOURCE_FAILED,
            (DEAF_CROPLAND_PROB,),
            "the DEAF tile carries no system:time_start, so the value could not be dated",
        )
    evidence = fact_from_reduce_region(result, source, observed_on=observed_on)
    if isinstance(evidence, Fact):
        return replace(
            evidence,
            note="share of random-forest trees that voted 'cropland' in 2019 (raw 0 is "
            "indistinguishable from this layer's NoData fill and is not reported)",
        )
    return evidence


def _cropland_class(result: dict[str, Any]) -> Fact | Missing:
    quantity = "cropland_class"
    if not result.get("gcep_class__tiles"):
        return Missing(
            quantity, MissingReason.OUT_OF_COVERAGE, (GFSAD_GCEP30,), "no GCEP30 tile intersects this point"
        )
    source = SourceDescriptor(
        quantity=quantity,
        asset_id=GFSAD_GCEP30,
        unit="class",
        band="b1",
        result_key="gcep_class",
        resolution_m=30.0,
        chain=_CROPLAND_CHAIN,
        chain_position=1,
        scaling=_NO_SCALING,
        valid_range=(0.0, float(max(GCEP30_CLASSES))),
        is_static=True,  # the asset carries no timestamp at all
        coverage="global",
    )
    evidence = fact_from_reduce_region(result, source, observed_on=None)
    if not isinstance(evidence, Fact):
        return evidence
    label = _class_label(evidence.value, GCEP30_CLASSES)
    if label is None:
        return Missing(
            quantity,
            MissingReason.SOURCE_FAILED,
            (GFSAD_GCEP30,),
            f"GCEP30 returned class {evidence.value!r}, which is not in its published class table",
        )
    return Fact.derive(
        quantity,
        label,
        "",
        [evidence],
        scaling_applied="GCEP30 class code (0 water, 1 NON-cropland, 2 cropland; the 1 is "
        "inverted relative to DEAF and the NASA Harvest binaries)",
        note=_NO_TIMESTAMP_NOTE,
    )


def _land_cover(result: dict[str, Any]) -> Fact | Missing:
    quantity = "land_cover"
    source = SourceDescriptor(
        quantity=quantity,
        asset_id=ESA_WORLDCOVER,
        unit="class",
        band="Map",
        result_key="worldcover_class",
        resolution_m=10.0,
        chain=(ESA_WORLDCOVER,),
        chain_position=0,
        scaling=_NO_SCALING,
        valid_range=(0.0, float(max(WORLDCOVER_CLASSES))),
        coverage="global",
    )
    observed_on = _epoch_to_date(result.get("worldcover__time_start"))
    if observed_on is None:
        return Missing(
            quantity,
            MissingReason.SOURCE_FAILED,
            (ESA_WORLDCOVER,),
            "the WorldCover image carries no system:time_start, so the value could not be dated",
        )
    evidence = fact_from_reduce_region(result, source, observed_on=observed_on)
    if not isinstance(evidence, Fact):
        return evidence
    label = _class_label(evidence.value, WORLDCOVER_CLASSES)
    if label is None:
        return Missing(
            quantity,
            MissingReason.SOURCE_FAILED,
            (ESA_WORLDCOVER,),
            f"WorldCover returned class {evidence.value!r}, which is not in its published class table",
        )
    return Fact.derive(
        quantity,
        label,
        "",
        [evidence],
        scaling_applied="ESA WorldCover class code mapped through the STAC class table",
        note="the 2021 epoch of the 10 m global map",
    )


def _is_cropland(probability: Fact | Missing, land_class: Fact | Missing, cover: Fact | Missing) -> Fact | Missing:
    """Cropland or not, from whichever link of the chain answered first."""
    quantity = "is_cropland"
    if isinstance(probability, Fact):
        return Fact.derive(
            quantity,
            float(probability.value) > CROPLAND_PROBABILITY_THRESHOLD,
            "",
            [probability],
            scaling_applied=f"DEAF crop probability > {CROPLAND_PROBABILITY_THRESHOLD:.2f}, the "
            "threshold that reproduces DEAF's own published mask band",
            note=f"from a crop probability of {probability.render()}",
        )
    if isinstance(land_class, Fact):
        return Fact.derive(
            quantity,
            land_class.value == GCEP30_CLASSES[GCEP30_CROPLAND_CLASS],
            "",
            [land_class],
            scaling_applied=f"GCEP30 class == {GCEP30_CROPLAND_CLASS} (cropland)",
            note=f"DEAF gave no usable probability here, so this is GCEP30 at 30 m: {land_class.render()}",
        )
    if isinstance(cover, Fact):
        return Fact.derive(
            quantity,
            cover.value == WORLDCOVER_CLASSES[WORLDCOVER_CROPLAND_CLASS],
            "",
            [cover],
            scaling_applied=f"ESA WorldCover class == {WORLDCOVER_CROPLAND_CLASS} (cropland)",
            note=f"neither DEAF nor GCEP30 answered here, so this is land cover alone: {cover.render()}",
        )
    return Missing(
        quantity,
        MissingReason.SOURCE_FAILED,
        _CROPLAND_CHAIN,
        "no layer in the cropland chain returned a value at this point",
    )


def _irrigation_regime(result: dict[str, Any]) -> Fact | Missing:
    quantity = "irrigation_regime"
    if not result.get("lgrip_class__tiles"):
        return Missing(
            quantity, MissingReason.OUT_OF_COVERAGE, (GFSAD_LGRIP30,), "no LGRIP30 tile intersects this point"
        )
    source = SourceDescriptor(
        quantity=quantity,
        asset_id=GFSAD_LGRIP30,
        unit="class",
        band="b1",
        result_key="lgrip_class",
        resolution_m=30.0,
        chain=(GFSAD_LGRIP30,),
        chain_position=0,
        scaling=_NO_SCALING,
        valid_range=(0.0, float(max(LGRIP30_CLASSES))),
        is_static=True,  # the asset carries no timestamp at all
        coverage="global",
    )
    evidence = fact_from_reduce_region(result, source, observed_on=None)
    if not isinstance(evidence, Fact):
        return evidence
    label = _class_label(evidence.value, LGRIP30_CLASSES)
    if label is None:
        return Missing(
            quantity,
            MissingReason.SOURCE_FAILED,
            (GFSAD_LGRIP30,),
            f"LGRIP30 returned class {evidence.value!r}, which is not in its published class table",
        )
    note = f"{_NO_TIMESTAMP_NOTE}; {_LGRIP_HEDGE}"
    if int(evidence.value) not in (LGRIP_IRRIGATED_CLASS, LGRIP_RAINFED_CLASS):
        note += (
            "; LGRIP only assigns an irrigation class to pixels it maps as cropland, so this "
            "pixel carries no irrigation answer -- see the neighbourhood fractions"
        )
    return Fact.derive(
        quantity,
        label,
        "",
        [evidence],
        scaling_applied="LGRIP30 class code (0 water, 1 non-cropland, 2 irrigated, 3 rainfed)",
        note=note,
    )


def _fraction_facts(result: dict[str, Any]) -> list[Fact | Missing]:
    """Irrigated / rainfed / cropland share of the 5 km neighbourhood."""
    chain = (GFSAD_LGRIP30,)
    histogram = result.get("lgrip_histogram")
    quantities = ("irrigated_cropland_fraction", "rainfed_cropland_fraction", "cropland_fraction")
    if not isinstance(histogram, dict) or not histogram:
        return [
            Missing(
                quantity,
                MissingReason.MASKED,
                chain,
                f"LGRIP30 returned no class histogram over the {IRRIGATION_NEIGHBOURHOOD_M / 1000:g} km "
                "neighbourhood",
            )
            for quantity in quantities
        ]

    counts: dict[int, float] = {}
    for raw_class, count in histogram.items():
        try:
            counts[int(float(raw_class))] = float(count)
        except (TypeError, ValueError):
            continue
    total = sum(counts.values())
    if total <= 0:
        return [
            Missing(quantity, MissingReason.MASKED, chain, "the LGRIP30 class histogram is empty")
            for quantity in quantities
        ]

    irrigated = counts.get(LGRIP_IRRIGATED_CLASS, 0.0) / total
    rainfed = counts.get(LGRIP_RAINFED_CLASS, 0.0) / total
    shared = (
        f"share of the {IRRIGATION_NEIGHBOURHOOD_M / 1000:g} km neighbourhood "
        f"({total:.0f} LGRIP30 pixels); {_LGRIP_HEDGE}; {_NO_TIMESTAMP_NOTE}"
    )
    facts: list[Fact | Missing] = []
    for quantity, value, what in (
        ("irrigated_cropland_fraction", irrigated, "mapped as irrigated cropland"),
        ("rainfed_cropland_fraction", rainfed, "mapped as rainfed cropland"),
        ("cropland_fraction", irrigated + rainfed, "mapped as cropland of either kind"),
    ):
        facts.append(
            Fact(
                quantity=quantity,
                value=value,
                unit="",
                source_asset=GFSAD_LGRIP30,
                observed_on=None,  # the asset carries no timestamp
                resolution_m=2.0 * IRRIGATION_NEIGHBOURHOOD_M,
                band="b1",
                scaling_applied="class-frequency histogram at 30 m, divided by the total pixel count",
                note=f"{what}; {shared}",
                precision=3,
            )
        )
    return facts


def _neighbourhood_regime(fractions: Sequence[Fact | Missing]) -> Fact | Missing:
    """Does the neighbourhood lean irrigated or rainfed? Often neither."""
    quantity = "irrigation_regime_neighbourhood"
    chain = (GFSAD_LGRIP30,)
    measured = {f.quantity: f for f in fractions if isinstance(f, Fact)}
    irrigated = measured.get("irrigated_cropland_fraction")
    rainfed = measured.get("rainfed_cropland_fraction")
    cropland = measured.get("cropland_fraction")
    if irrigated is None or rainfed is None or cropland is None:
        return Missing(
            quantity, MissingReason.MASKED, chain, "the LGRIP30 neighbourhood fractions are not available"
        )

    if float(cropland.value) < MIN_CROPLAND_FRACTION:
        return Missing(
            quantity,
            MissingReason.MASKED,
            chain,
            f"only {float(cropland.value) * 100:.1f}% of the "
            f"{IRRIGATION_NEIGHBOURHOOD_M / 1000:g} km neighbourhood is mapped as cropland at all, "
            f"which is below the {MIN_CROPLAND_FRACTION * 100:.0f}% needed to read an irrigation "
            "signal from it",
        )

    high, low, label = (
        (irrigated, rainfed, "irrigated")
        if float(irrigated.value) >= float(rainfed.value)
        else (rainfed, irrigated, "rainfed")
    )
    if float(low.value) > 0 and float(high.value) / float(low.value) < MIN_REGIME_RATIO:
        return Missing(
            quantity,
            MissingReason.MASKED,
            chain,
            f"the neighbourhood is mixed ({float(irrigated.value) * 100:.1f}% irrigated vs "
            f"{float(rainfed.value) * 100:.1f}% rainfed), too close to call at "
            f"{MIN_REGIME_RATIO:g}:1",
        )
    return Fact.derive(
        quantity,
        label,
        "",
        [high, low],
        scaling_applied=f"the larger LGRIP30 class share over {IRRIGATION_NEIGHBOURHOOD_M / 1000:g} km, "
        f"required to lead the other by {MIN_REGIME_RATIO:g}:1",
        note=f"{float(irrigated.value) * 100:.1f}% irrigated vs {float(rainfed.value) * 100:.1f}% "
        f"rainfed around the field, not a reading at the field itself; {_LGRIP_HEDGE}",
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def get_field_context(lat: float, lon: float, *, ledger: Ledger | None = None) -> dict[str, Fact | Missing]:
    """Everything the land-cover layers know about this point, in one EE call.

    Keys: ``cropland_probability``, ``cropland_class``, ``is_cropland``,
    ``land_cover``, ``irrigation_regime``, ``irrigated_cropland_fraction``,
    ``rainfed_cropland_fraction``, ``cropland_fraction`` and
    ``irrigation_regime_neighbourhood``. Every key is always present, holding a
    Fact or the Missing that says why not.
    """
    quantities = (
        "cropland_probability",
        "cropland_class",
        "is_cropland",
        "land_cover",
        "irrigation_regime",
        "irrigated_cropland_fraction",
        "rainfed_cropland_fraction",
        "cropland_fraction",
        "irrigation_regime_neighbourhood",
    )
    chain = _CROPLAND_CHAIN + (GFSAD_LGRIP30,)

    try:
        result = _fetch(lat, lon)
    except EarthEngineUnavailable as exc:
        return _all_failed(quantities, chain, exc.reason, ledger)
    except Exception as exc:  # an Earth Engine error must not crash a turn
        return _all_failed(quantities, chain, f"{type(exc).__name__}: {exc}", ledger)

    probability = _cropland_probability(result)
    land_class = _cropland_class(result)
    cover = _land_cover(result)
    fractions = _fraction_facts(result)
    context: dict[str, Fact | Missing] = {
        "cropland_probability": probability,
        "cropland_class": land_class,
        "is_cropland": _is_cropland(probability, land_class, cover),
        "land_cover": cover,
        "irrigation_regime": _irrigation_regime(result),
    }
    for item in fractions:
        context[item.quantity] = item
    context["irrigation_regime_neighbourhood"] = _neighbourhood_regime(fractions)

    if ledger is not None:
        for quantity in quantities:
            ledger.add(context[quantity])
    return context


def _all_failed(
    quantities: Sequence[str], chain: Sequence[str], detail: str, ledger: Ledger | None
) -> dict[str, Fact | Missing]:
    context: dict[str, Fact | Missing] = {
        quantity: Missing(quantity, MissingReason.SOURCE_FAILED, tuple(chain), detail) for quantity in quantities
    }
    if ledger is not None:
        for quantity in quantities:
            ledger.add(context[quantity])
    return context


def is_cropland(lat: float, lon: float, *, ledger: Ledger | None = None) -> Fact | Missing:
    """Is this point cropland? Chain: DEAF probability -> GCEP30 -> WorldCover."""
    evidence = get_field_context(lat, lon)["is_cropland"]
    if ledger is not None:
        ledger.add(evidence)
    return evidence


def get_irrigation_regime(lat: float, lon: float, *, ledger: Ledger | None = None) -> Fact | Missing:
    """Irrigated or rainfed at this pixel, per LGRIP30.

    Reports what the pixel says, including "Non-cropland", which is LGRIP's
    answer at all three CropUp test points; ``get_field_context`` additionally
    returns the neighbourhood fractions and the verdict they support.
    """
    evidence = get_field_context(lat, lon)["irrigation_regime"]
    if ledger is not None:
        ledger.add(evidence)
    return evidence
