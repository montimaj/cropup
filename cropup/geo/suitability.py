"""Crop suitability from CropSuite v1.0 (Zabel, Knuettel & Poschlod 2025).

This is the replacement for the 80 MB MaxEnt suitability database the vendored
backend expected and never shipped (SPEC section 2.1, bug 2). CropSuite lives in
the GEE Community Catalog under ``projects/sat-io/open-datasets/CROP_SUITE/``.

Three things about this dataset decide the shape of this module.

1. **Crop is a band, not a filter.** Each of the five collections used here holds
   six images -- one per *scenario*, ``historical_1991-2010_{rf|ir|rfir}_{novar|var}``,
   filterable on the ``scenario`` property -- and each image carries 48 bands, one
   per crop. Selecting a crop CropSuite does not model raises
   "Band pattern did not match any bands", so the band list is guarded here.

2. **Africa only.** The grid is EPSG:4326, origin (-25, 39), 0.008333 deg,
   9600 x 9000 px, i.e. lon -25..55, lat -36..39 (read off the asset's
   ``crs_transform``). Salinas, California is nowhere near it, so every function
   returns ``Missing(out_of_coverage)`` there without spending an EE call.

3. **The exact pixel is often masked.** ``crop_suitability`` is the Liebig
   minimum of climate and nine soil/terrain parameters, and the soil inputs have
   holes: about 8% of Tanzania is masked. At the Arusha test point the exact
   pixel is masked for 47 of the 48 crops in all six scenarios, and measurement
   shows the hole is wider than 1 km -- a 300 m and a 1 km neighbourhood are both
   fully masked there; only 3 km returns (19 of ~33 pixels unmasked).

   So each query walks a ladder of legs -- exact pixel, then widening
   neighbourhood means, then the ``climate_suitability`` ceiling -- and every
   fallback is disclosed on the Fact: ``chain_position`` (the UI renders
   "fallback #2"), ``resolution_m`` widened to the neighbourhood diameter, and a
   ``note`` that says in words that the number is a neighbourhood estimate, not a
   reading at the field.

Raw values need **no scaling**: the int8 pixel already is the 0-100 percentage.
That is recorded on every Fact as ``scaling_applied`` so the UI can say so.
"""

from __future__ import annotations

import datetime as dt
import json
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
    "SCENARIOS",
    "DEFAULT_SCENARIO",
    "CROPSUITE_BANDS",
    "PERENNIAL_BANDS",
    "ANOMALOUS_BANDS",
    "LIMITING_FACTORS",
    "in_coverage",
    "band_for_crop",
    "crop_for_band",
    "crop_coverage",
    "get_crop_suitability",
    "rank_crops",
    "get_limiting_factor",
    "get_optimal_sowing",
    "clear_cache",
]

_ROOT = "projects/sat-io/open-datasets/CROP_SUITE/"
CROP_SUITABILITY = _ROOT + "crop_suitability"
CLIMATE_SUITABILITY = _ROOT + "climate_suitability"
CROP_LIMITING_FACTOR = _ROOT + "crop_limiting_factor"
OPTIMAL_SOWING_DATE = _ROOT + "optimal_sowing_date"
SUITABLE_SOWING_DAYS = _ROOT + "suitable_sowing_days"

# Short per-asset tag used in the reduceRegion keys (see _suffix).
_ASSET_TAG = {
    CROP_SUITABILITY: "cs",
    CLIMATE_SUITABILITY: "cl",
    CROP_LIMITING_FACTOR: "lf",
    OPTIMAL_SOWING_DATE: "sd",
    SUITABLE_SOWING_DAYS: "ss",
}

# projection().nominalScale() of every CROP_SUITE band (30 arcsec). The asset
# property spatial_resolution says "0.5 degrees" and is wrong.
NATIVE_SCALE_M = 927.6624232772797

# lon_min, lon_max, lat_min, lat_max, from the asset crs_transform + dimensions.
AFRICA_GRID = (-25.0, 55.0, -36.0, 39.0)

SCENARIOS = (
    "historical_1991-2010_rf_novar",
    "historical_1991-2010_rf_var",
    "historical_1991-2010_ir_novar",
    "historical_1991-2010_ir_var",
    "historical_1991-2010_rfir_novar",
    "historical_1991-2010_rfir_var",
)
# Rainfed without the climate-variability penalty. `_var` additionally charges
# each crop for its crop-failure recurrence and is far more pessimistic (maize at
# Arusha: 83 under rf_novar, 0 under rf_var), which is a judgement call the
# caller should make explicitly rather than inherit from a default.
DEFAULT_SCENARIO = "historical_1991-2010_rf_novar"

CROPSUITE_BANDS = (
    "alfalfa", "avocado", "banana", "barley", "beans", "cabbage", "carrots",
    "cashew", "cassava", "castorbeans", "chickpea", "citrus", "cocoa", "coconut",
    "coffeearabica", "coffeerobusta", "cotton", "cowpea", "greenpepper",
    "groundnut", "guava", "maize", "mango", "millet", "oilpalm", "olives",
    "onion", "papaya", "pea", "pineapple", "potato", "rapeseed", "rice",
    "rubber", "rye", "safflower", "sesame", "sorghum", "soy", "sugarcane",
    "sunflower", "sweetpotato", "tea", "tobacco", "tomato", "watermelon",
    "wheat", "yam",
)

# The 20 perennial / continuously established crops carry no sowing date: their
# optimal_sowing_date band is identically 0 across Africa. Independently visible
# in the asset metadata -- Earth Engine types exactly these 20 bands int8 in
# optimal_sowing_date and the other 28 int16, because min == max == 0. Reporting
# "1 January" for them would be a fabricated planting date.
PERENNIAL_BANDS = frozenset({
    "alfalfa", "avocado", "banana", "cashew", "cassava", "citrus", "cocoa",
    "coconut", "coffeearabica", "coffeerobusta", "guava", "mango", "oilpalm",
    "olives", "papaya", "pineapple", "rubber", "sugarcane", "tea", "yam",
})

# rapeseed is int16 in every CROP_SUITE collection while the other 47 bands are
# int8, its limiting-factor codes reach 13 (outside the derived class table) and
# its sowing DOY reaches 365 (outside the 0-364 range of every other crop). It
# also scores 72 at Morogoro, which is not a plausible rapeseed site. Excluded
# from ranking by default, and the exclusion is recorded as a gap, not hidden.
ANOMALOUS_BANDS = ("rapeseed",)

# CropSuite publishes no class table for crop_limiting_factor. This one was
# derived from the CropSuite v1.0 source and cross-validated against SoilGrids
# and SRTM in Earth Engine (see cropup/data/ee_registry.json for the evidence).
LIMITING_FACTORS = {
    0: "temperature",
    1: "precipitation",
    2: "climate variability (crop-failure recurrence)",
    3: "photoperiod",
    4: "soil base saturation",
    5: "coarse fragments",
    6: "soil pH",
    7: "salinity",
    8: "soil texture",
    9: "soil organic carbon",
    10: "sodicity",
    11: "soil depth",
    12: "slope",
}
# Codes 0-3, 6, 8, 9, 11 and 12 are supported by independent cross-validation.
# These four follow the source ordering only, so they are labelled as inferred.
_UNCONFIRMED_LIMITING_CODES = frozenset({4, 5, 7, 10})

_LIMITING_TABLE_NOTE = (
    "the data provider publishes no class table for this layer; the codes were "
    "derived from the CropSuite v1.0 source and cross-validated against SoilGrids "
    "and SRTM"
)

_SUITABILITY_SCALING = "none: the raw int8 pixel is already the 0-100 percentage"
_CLIMATOLOGY_NOTE = "a 1991-2010 climatology, not an observation of this season"
_NO_SCALING = "none: raw integer class code"

# CropSuite band -> cropup/data/crops.json key. All 48 bands map onto a crop in
# the 134-crop vocabulary; crop_coverage() reports the mapping (and any breakage)
# from the committed file rather than asserting it here.
_BAND_TO_CROP_KEY = {
    "alfalfa": "alfalfa",
    "avocado": "avocado",
    "banana": "banana",
    "barley": "barley",
    "beans": "beans",
    "cabbage": "cabbage",
    "carrots": "carrot",
    "cashew": "cashew",
    "cassava": "cassava",
    "castorbeans": "castor bean",
    "chickpea": "chickpea",
    "citrus": "citrus",
    "cocoa": "cocoa",
    "coconut": "coconut",
    "coffeearabica": "coffee",
    "coffeerobusta": "coffee robusta",
    "cotton": "cotton",
    "cowpea": "cowpea",
    "greenpepper": "pepper",
    "groundnut": "groundnut",
    "guava": "guava",
    "maize": "maize",
    "mango": "mango",
    "millet": "millet",
    "oilpalm": "oil palm",
    "olives": "olive",
    "onion": "onion",
    "papaya": "papaya",
    "pea": "pea",
    "pineapple": "pineapple",
    "potato": "potato",
    "rapeseed": "rapeseed",
    "rice": "rice",
    "rubber": "rubber",
    "rye": "rye",
    "safflower": "safflower",
    "sesame": "sesame",
    "sorghum": "sorghum",
    "soy": "soybean",
    "sugarcane": "sugarcane",
    "sunflower": "sunflower",
    "sweetpotato": "sweet potato",
    "tea": "tea",
    "tobacco": "tobacco",
    "tomato": "tomato",
    "watermelon": "watermelon",
    "wheat": "wheat",
    "yam": "yam",
}

# Widening neighbourhood legs, in metres. The settings value (300 m by default)
# is tried first; 1 km and 3 km follow because the masked hole at Arusha is
# wider than 1 km -- measured, not guessed.
_EXTRA_RADII_M = (1000.0, 3000.0)

# A non-leap year: CropSuite day-of-year runs 0-364, i.e. 365 days.
_DOY_YEAR = 2001


# ---------------------------------------------------------------------------
# in-process cache: one getInfo per (point, scenario, layer set), SPEC 3.4
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
# crop vocabulary
# ---------------------------------------------------------------------------

_vocab_lock = threading.Lock()
_alias_to_band: dict[str, str] | None = None
_crop_names: dict[str, str] | None = None  # crops.json key -> display name


def _normalise(text: str) -> str:
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    return " ".join(cleaned.split())


def _load_vocabulary() -> tuple[dict[str, str], dict[str, str]]:
    """alias -> CropSuite band, and crops.json key -> display name."""
    global _alias_to_band, _crop_names
    with _vocab_lock:
        if _alias_to_band is not None and _crop_names is not None:
            return _alias_to_band, _crop_names
        path = get_settings().require_data_file("crops")
        payload = json.loads(path.read_text(encoding="utf-8"))
        names: dict[str, str] = {}
        by_key: dict[str, list[str]] = {}
        for entry in payload["crops"]:
            key = entry["key"]
            names[key] = entry.get("name", key)
            by_key[key] = list(entry.get("aliases", ()))

        aliases: dict[str, str] = {}
        for band, key in _BAND_TO_CROP_KEY.items():
            candidates = [band, key, names.get(key, key)]
            candidates.extend(by_key.get(key, ()))
            for candidate in candidates:
                token = _normalise(str(candidate))
                if token:
                    aliases.setdefault(token, band)
        _alias_to_band, _crop_names = aliases, names
        return aliases, names


def band_for_crop(crop: str) -> str | None:
    """The CropSuite band for a crop name, alias or band name, or None.

    Exact lookup over the committed vocabulary only -- fuzzy matching belongs to
    ``nlu/slots.py``, which resolves free text to a canonical crop first.
    """
    if not crop or not isinstance(crop, str):
        return None
    aliases, _ = _load_vocabulary()
    return aliases.get(_normalise(crop))


def crop_for_band(band: str) -> str:
    """The canonical display name for a CropSuite band ("coffeearabica" -> "Coffee")."""
    _, names = _load_vocabulary()
    key = _BAND_TO_CROP_KEY.get(band)
    if key is None:
        return band
    return names.get(key, key)


def crop_coverage() -> dict[str, Any]:
    """How the 48 CropSuite bands line up with the 134-crop vocabulary.

    Measured against the committed crops.json, so a change to either side shows
    up here instead of silently mapping a crop onto the wrong band.
    """
    _, names = _load_vocabulary()
    mapped, broken = [], []
    for band, key in sorted(_BAND_TO_CROP_KEY.items()):
        if key in names:
            mapped.append({"band": band, "crop_key": key, "crop": names[key]})
        else:
            broken.append({"band": band, "crop_key": key})
    modelled_keys = set(_BAND_TO_CROP_KEY.values())
    unmodelled = sorted(names[k] for k in names if k not in modelled_keys)
    return {
        "cropsuite_bands": len(CROPSUITE_BANDS),
        "vocabulary_crops": len(names),
        "mapped": mapped,
        "mapped_count": len(mapped),
        "unresolved_mappings": broken,
        "crops_without_a_cropsuite_band": unmodelled,
        "crops_without_a_cropsuite_band_count": len(unmodelled),
        "perennials_without_a_sowing_date": sorted(crop_for_band(b) for b in PERENNIAL_BANDS),
        "anomalous_bands": list(ANOMALOUS_BANDS),
    }


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------


def in_coverage(lat: float, lon: float) -> bool:
    """True when the point falls inside the CropSuite Africa grid."""
    lon_min, lon_max, lat_min, lat_max = AFRICA_GRID
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


def _out_of_africa(quantity: str, lat: float, lon: float, chain: Sequence[str]) -> Missing:
    lon_min, lon_max, lat_min, lat_max = AFRICA_GRID
    return Missing(
        quantity,
        MissingReason.OUT_OF_COVERAGE,
        tuple(chain),
        f"CropSuite v1.0 covers Africa only (lon {lon_min:g}..{lon_max:g}, "
        f"lat {lat_min:g}..{lat_max:g}); ({lat:g}, {lon:g}) is outside the grid",
    )


# ---------------------------------------------------------------------------
# legs: one reduceRegion each, all of them in a single getInfo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Leg:
    asset: str
    suffix: str  # keeps the keys of the different legs apart in one dictionary
    reducer: str  # "first" (exact pixel), "mean" or "mode" (neighbourhood)
    radius_m: float | None  # None: the exact pixel
    chain_position: int

    @property
    def label(self) -> str:
        where = "exact pixel" if self.radius_m is None else f"{_km(self.radius_m)} neighbourhood {self.reducer}"
        return f"{self.asset} ({where})"

    @property
    def resolution_m(self) -> float:
        # A neighbourhood reduction is honestly coarser than the native pixel:
        # report the diameter of the region it actually summarised.
        return NATIVE_SCALE_M if self.radius_m is None else 2.0 * self.radius_m


def _km(metres: float) -> str:
    return f"{metres / 1000:g} km" if metres >= 1000 else f"{metres:g} m"


def _radii() -> tuple[float, ...]:
    settings = get_settings()
    radii = {float(settings.ee_neighbourhood_m)} | set(_EXTRA_RADII_M)
    return tuple(sorted(r for r in radii if r > 0))


def _suffix(asset: str, radius_m: float | None) -> str:
    """Key suffix for one leg. The asset tag keeps two collections that share a
    band name from overwriting each other inside one ee.Dictionary."""
    where = "x" if radius_m is None else f"n{int(radius_m)}"
    return f"{_ASSET_TAG[asset]}{where}"


def _legs(asset: str, reducer: str, *, start: int = 0) -> list[_Leg]:
    legs = [_Leg(asset, _suffix(asset, None), "first", None, start)]
    for radius in _radii():
        legs.append(_Leg(asset, _suffix(asset, radius), reducer, radius, start + len(legs)))
    return legs


def _suitability_legs() -> tuple[_Leg, ...]:
    legs = _legs(CROP_SUITABILITY, "mean")
    radius = _radii()[-1]
    legs.append(_Leg(CLIMATE_SUITABILITY, _suffix(CLIMATE_SUITABILITY, None), "first", None, len(legs)))
    legs.append(_Leg(CLIMATE_SUITABILITY, _suffix(CLIMATE_SUITABILITY, radius), "mean", radius, len(legs)))
    return tuple(legs)


def _categorical_legs(asset: str) -> tuple[_Leg, ...]:
    """Exact pixel, then the neighbourhood *mode* -- averaging a class code or a
    day-of-year would invent a value that no pixel holds."""
    return tuple(_legs(asset, "mode"))


def _count_legs(asset: str) -> tuple[_Leg, ...]:
    return tuple(_legs(asset, "mean"))


def _value_key(band: str, leg: _Leg) -> str:
    base = f"{band}__{leg.suffix}"
    return base if leg.reducer == "first" else f"{base}_{leg.reducer}"


def _count_key(band: str, leg: _Leg) -> str:
    return f"{band}__{leg.suffix}_count"


def _histogram_key(band: str, leg: _Leg) -> str:
    return f"{band}__{leg.suffix}_histogram"


def _class_votes(result: dict[str, Any], band: str, leg: _Leg) -> list[tuple[float, float]]:
    """(value, weighted pixel count) for a neighbourhood mode, commonest first.

    A mode is only as strong as its margin: at Arusha the wheat limiting factor
    is 4.24 pixels for one class against 4.05 for another, which is a coin toss
    and has to be visible rather than rounded into a verdict.
    """
    histogram = result.get(_histogram_key(band, leg))
    if not isinstance(histogram, dict):
        return []
    votes: list[tuple[float, float]] = []
    for raw_value, weight in histogram.items():
        try:
            votes.append((float(raw_value), float(weight)))
        except (TypeError, ValueError):
            continue
    votes.sort(key=lambda pair: pair[1], reverse=True)
    return votes


def _leg_at(legs: Sequence[_Leg], chain_position: int) -> _Leg | None:
    """The leg a Fact came from. chain_position is the leg's index by construction."""
    if 0 <= chain_position < len(legs) and legs[chain_position].chain_position == chain_position:
        return legs[chain_position]
    return None


def _date_key(asset: str) -> str:
    return "observed_on__" + asset.rsplit("/", 1)[-1]


def _fetch(lat: float, lon: float, bands: Sequence[str], legs: Sequence[_Leg], scenario: str) -> dict[str, Any]:
    """One batched getInfo covering every leg. Raises on any Earth Engine problem."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown CropSuite scenario {scenario!r}; expected one of {', '.join(SCENARIOS)}")

    key = (scenario, round(lat, 5), round(lon, 5), tuple(bands), tuple(legs))
    cached = _cache_get(key)
    if cached is not None:
        return cached

    ee = bootstrap.require_ee()
    point = ee.Geometry.Point([lon, lat])
    images: dict[str, Any] = {}
    out = ee.Dictionary({})
    for leg in legs:
        image = images.get(leg.asset)
        if image is None:
            image = ee.ImageCollection(leg.asset).filter(ee.Filter.eq("scenario", scenario)).first()
            images[leg.asset] = image
            # The observation date is read from the asset, never assumed: these
            # are 1991-2010 model runs, and system:time_end is the end of that run.
            out = out.set(_date_key(leg.asset), ee.Date(image.get("system:time_end")).format("YYYY-MM-dd"))
        names = [f"{band}__{leg.suffix}" for band in bands]
        selected = image.select(list(bands), names)
        if leg.radius_m is None:
            out = out.combine(selected.reduceRegion(ee.Reducer.first(), point, NATIVE_SCALE_M))
            continue
        if leg.reducer == "mean":
            reducer = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)
        else:
            # The class histogram comes back with the mode so the margin of the
            # winning class can be reported instead of just its name.
            reducer = (
                ee.Reducer.mode()
                .combine(ee.Reducer.count(), sharedInputs=True)
                .combine(ee.Reducer.frequencyHistogram(), sharedInputs=True)
            )
        out = out.combine(selected.reduceRegion(reducer, point.buffer(leg.radius_m), NATIVE_SCALE_M))
    result = out.getInfo()
    _cache_put(key, result)
    return result


def _fetch_or_gap(
    quantity: str,
    lat: float,
    lon: float,
    bands: Sequence[str],
    legs: Sequence[_Leg],
    scenario: str,
) -> tuple[dict[str, Any] | None, Missing | None]:
    tried = tuple(leg.label for leg in legs)
    try:
        return _fetch(lat, lon, bands, legs, scenario), None
    except EarthEngineUnavailable as exc:
        return None, Missing(quantity, MissingReason.SOURCE_FAILED, tried, exc.reason)
    except Exception as exc:  # an Earth Engine error must not crash a turn
        return None, Missing(quantity, MissingReason.SOURCE_FAILED, tried, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# turning one leg into evidence
# ---------------------------------------------------------------------------


def _descriptor(
    quantity: str,
    band: str,
    leg: _Leg,
    legs: Sequence[_Leg],
    *,
    unit: str,
    scaling: str,
    valid_range: tuple[float, float] | None,
    invalid_values: tuple[float, ...] = (),
) -> SourceDescriptor:
    return SourceDescriptor(
        quantity=quantity,
        asset_id=leg.asset,
        unit=unit,
        band=band,
        result_key=_value_key(band, leg),
        resolution_m=leg.resolution_m,
        chain=tuple(other.label for other in legs),
        chain_position=leg.chain_position,
        scaling=scaling,
        valid_range=valid_range,
        invalid_values=invalid_values,
        coverage="Africa only",
    )


def _leg_note(leg: _Leg, result: dict[str, Any], band: str) -> str | None:
    parts: list[str] = []
    if leg.radius_m is not None:
        count = result.get(_count_key(band, leg))
        pixels = f"{int(count)} unmasked pixels" if isinstance(count, (int, float)) else "the unmasked pixels"
        estimate = (
            f"neighbourhood estimate, not a reading at the field: {leg.reducer} of {pixels} "
            f"within {_km(leg.radius_m)} of the point, because the {NATIVE_SCALE_M:.0f} m "
            "CropSuite pixel at the point is masked"
        )
        votes = _class_votes(result, band, leg)
        if votes:
            total = sum(weight for _, weight in votes)
            if total > 0:
                estimate += f"; the commonest value holds {votes[0][1] / total * 100:.0f}% of them"
        parts.append(estimate)
    if leg.asset == CLIMATE_SUITABILITY:
        parts.append(
            "climate-only ceiling: the full crop_suitability layer is masked here, and CropSuite "
            "takes the minimum over climate and nine soil/terrain parameters, so the true "
            "suitability is at most this value"
        )
    return "; ".join(parts) if parts else None


def _evidence(
    result: dict[str, Any],
    band: str,
    quantity: str,
    legs: Sequence[_Leg],
    *,
    unit: str,
    scaling: str,
    valid_range: tuple[float, float] | None,
    invalid_values: tuple[float, ...] = (),
    only_leg: _Leg | None = None,
) -> Fact | Missing:
    """Walk the legs in order and return the first measured value, or a Missing
    that names every leg that was tried."""
    tried = tuple(leg.label for leg in legs)
    last: Missing | None = None
    for leg in legs:
        if only_leg is not None and leg is not only_leg:
            continue
        observed_on = result.get(_date_key(leg.asset))
        if not observed_on:
            last = Missing(
                quantity,
                MissingReason.SOURCE_FAILED,
                tried,
                f"{leg.asset} returned no system:time_end, so the value could not be dated",
            )
            continue
        source = _descriptor(
            quantity,
            band,
            leg,
            legs,
            unit=unit,
            scaling=scaling,
            valid_range=valid_range,
            invalid_values=invalid_values,
        )
        evidence = fact_from_reduce_region(result, source, observed_on=observed_on)
        if isinstance(evidence, Fact):
            note = _leg_note(leg, result, band)
            return replace(evidence, note=note) if note else evidence
        last = evidence
    if last is None:
        return Missing(quantity, MissingReason.MASKED, tried, "no leg was attempted")
    # Keep the reason of the deepest leg but report the whole chain that was tried.
    return replace(last, chain_tried=tried)


def _record(ledger: Ledger | None, *items: Fact | Missing) -> None:
    if ledger is not None:
        for item in items:
            ledger.add(item)


def _resolve(
    quantity: str, lat: float, lon: float, crop: str, chain: Sequence[str]
) -> tuple[str | None, Missing | None]:
    """The CropSuite band to query, or the gap that stops the query.

    The location is checked before the crop so that the reason reported first is
    the one that is actually true: outside Africa nothing is available, and
    inside it the crop list is the only thing that can be missing.
    """
    if not in_coverage(lat, lon):
        return None, _out_of_africa(quantity, lat, lon, chain)
    band = band_for_crop(crop)
    if band is None:
        return None, Missing(
            quantity,
            MissingReason.OUT_OF_COVERAGE,
            tuple(chain),
            f"the point is inside the CropSuite grid, but CropSuite v1.0 models 48 crops "
            f"and {crop!r} is not one of them",
        )
    return band, None


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def get_crop_suitability(
    lat: float,
    lon: float,
    crop: str | None = None,
    *,
    scenario: str = DEFAULT_SCENARIO,
    ledger: Ledger | None = None,
) -> Fact | Missing:
    """CropSuite suitability (0-100) for one crop at one point.

    Walks exact pixel -> widening neighbourhood means -> climate-only ceiling and
    labels whichever leg answered. Returns ``Missing`` outside Africa, for a crop
    CropSuite does not model, and when every leg is masked.
    """
    quantity = "crop_suitability"
    legs = _suitability_legs()
    chain = tuple(leg.label for leg in legs)

    if crop is None:
        gap = Missing(
            quantity,
            MissingReason.NOT_REQUESTED,
            chain,
            "no crop was named; rank_crops() lists the crops CropSuite rates highest here",
        )
        _record(ledger, gap)
        return gap

    band, gap = _resolve(quantity, lat, lon, crop, chain)
    if band is None:
        _record(ledger, gap)
        return gap

    result, failure = _fetch_or_gap(quantity, lat, lon, [band], legs, scenario)
    if failure is not None:
        _record(ledger, failure)
        return failure

    evidence = _evidence(
        result,
        band,
        quantity,
        legs,
        unit="%",
        scaling=_SUITABILITY_SCALING,
        valid_range=(0.0, 100.0),
    )
    if isinstance(evidence, Fact):
        crop_name = crop_for_band(band)
        extra = f"{crop_name}, CropSuite scenario {scenario}, {_CLIMATOLOGY_NOTE}"
        if band in ANOMALOUS_BANDS:
            extra += "; this band is int16 while the other 47 are int8 and its values are anomalous"
        evidence = replace(evidence, note=f"{evidence.note}; {extra}" if evidence.note else extra)
    _record(ledger, evidence)
    return evidence


def rank_crops(
    lat: float,
    lon: float,
    n: int = 10,
    *,
    scenario: str = DEFAULT_SCENARIO,
    ledger: Ledger | None = None,
    include_anomalous: bool = False,
) -> list[Fact]:
    """The ``n`` crops CropSuite rates highest at this point, best first.

    Every returned Fact comes from the *same* leg of the ladder, so the scores are
    comparable; crops masked on that leg are recorded as gaps on the ledger rather
    than dropped silently. Returns an empty list outside Africa or when every leg
    is masked -- the reason is on the ledger.
    """
    quantity_all = "crop_suitability"
    legs = _suitability_legs()
    chain = tuple(leg.label for leg in legs)
    bands = [b for b in CROPSUITE_BANDS if include_anomalous or b not in ANOMALOUS_BANDS]

    if not in_coverage(lat, lon):
        _record(ledger, _out_of_africa(quantity_all, lat, lon, chain))
        return []

    if not include_anomalous:
        for band in ANOMALOUS_BANDS:
            _record(
                ledger,
                Missing(
                    _quantity_for(band),
                    MissingReason.SOURCE_FAILED,
                    (CROP_SUITABILITY,),
                    f"the {band} band is int16 where the other 47 are int8 and its values are "
                    "anomalous in this mirror; excluded from the ranking "
                    "(pass include_anomalous=True to see it)",
                ),
            )

    result, failure = _fetch_or_gap(quantity_all, lat, lon, bands, legs, scenario)
    if failure is not None:
        _record(ledger, failure)
        return []

    chosen = _first_leg_with_values(result, bands, legs)
    if chosen is None:
        _record(
            ledger,
            Missing(
                quantity_all,
                MissingReason.MASKED,
                chain,
                "every CropSuite leg is masked at this point, for all 48 crops",
            ),
        )
        return []

    facts: list[Fact] = []
    gaps: list[Missing] = []
    for band in bands:
        evidence = _evidence(
            result,
            band,
            _quantity_for(band),
            legs,
            unit="%",
            scaling=_SUITABILITY_SCALING,
            valid_range=(0.0, 100.0),
            only_leg=chosen,
        )
        if isinstance(evidence, Fact):
            crop_name = crop_for_band(band)
            note = f"{evidence.note}; {crop_name}" if evidence.note else crop_name
            facts.append(
                replace(evidence, note=f"{note}, CropSuite scenario {scenario}, {_CLIMATOLOGY_NOTE}")
            )
        else:
            gaps.append(evidence)

    facts.sort(key=lambda f: (-float(f.value), f.quantity))
    top = facts[: max(0, n)]
    _record(ledger, *top)
    _record(ledger, *gaps)
    return top


def _quantity_for(band: str) -> str:
    return "crop_suitability_" + _BAND_TO_CROP_KEY.get(band, band).replace(" ", "_")


def _first_leg_with_values(result: dict[str, Any], bands: Sequence[str], legs: Sequence[_Leg]) -> _Leg | None:
    """The shallowest leg that returned a value for at least one crop. Ranking
    across different legs would compare a soil-limited score with a climate
    ceiling, so one leg has to serve the whole list."""
    for leg in legs:
        for band in bands:
            if result.get(_value_key(band, leg)) is not None:
                return leg
    return None


def _contested_limiting_factor(result: dict[str, Any], band: str, leg: _Leg | None) -> str | None:
    """A sentence naming the runner-up class when the neighbourhood barely chose.

    Telling a farmer their limit is base saturation when the window voted 22% to
    21% for soil pH would be a false certainty, so the second place is named.
    """
    if leg is None:
        return None
    votes = _class_votes(result, band, leg)
    if len(votes) < 2:
        return None
    total = sum(weight for _, weight in votes)
    (_, winner_weight), (second, second_weight) = votes[0], votes[1]
    if total <= 0 or second_weight < 0.8 * winner_weight:
        return None
    other = LIMITING_FACTORS.get(int(second), f"class code {int(second)}")
    return (
        f"the window barely chose it: {other} (class code {int(second)}) took almost as many "
        f"pixels, {second_weight / total * 100:.0f}%"
    )


def get_limiting_factor(
    lat: float,
    lon: float,
    crop: str,
    *,
    scenario: str = DEFAULT_SCENARIO,
    ledger: Ledger | None = None,
) -> Fact | Missing:
    """What holds this crop back here: "precipitation", "soil pH", "slope", ...

    Returns a Fact whose value is the factor name, derived from the integer class
    Fact (also written to the ledger). Codes 4, 5, 7 and 10 are labelled as
    inferred, because only the source ordering supports them.
    """
    quantity = "crop_limiting_factor"
    legs = _categorical_legs(CROP_LIMITING_FACTOR)
    chain = tuple(leg.label for leg in legs)

    band, gap = _resolve(quantity, lat, lon, crop, chain)
    if band is None:
        _record(ledger, gap)
        return gap

    result, failure = _fetch_or_gap(quantity, lat, lon, [band], legs, scenario)
    if failure is not None:
        _record(ledger, failure)
        return failure

    code_fact = _evidence(
        result,
        band,
        "crop_limiting_factor_code",
        legs,
        unit="class",
        scaling=_NO_SCALING,
        valid_range=(0.0, float(max(LIMITING_FACTORS))),
        invalid_values=(-1.0,),  # -1 is CropSuite's nodata
    )
    _record(ledger, code_fact)
    if not isinstance(code_fact, Fact):
        return code_fact

    code = int(code_fact.value)
    label = LIMITING_FACTORS[code]
    note = f"{crop_for_band(band)}, CropSuite scenario {scenario}; class code {code}; {_LIMITING_TABLE_NOTE}"
    if code in _UNCONFIRMED_LIMITING_CODES:
        note += "; this particular code follows the source ordering and could not be independently confirmed"
    runner_up = _contested_limiting_factor(result, band, _leg_at(legs, code_fact.chain_position))
    if runner_up:
        note += f"; {runner_up}"
    if code_fact.note:
        note = f"{code_fact.note}; {note}"
    fact = Fact.derive(
        quantity,
        label,
        "",
        [code_fact],
        scaling_applied="integer class code mapped through the derived CropSuite limiting-factor table",
        note=note,
    )
    _record(ledger, fact)
    return fact


def get_optimal_sowing(
    lat: float,
    lon: float,
    crop: str,
    *,
    scenario: str = DEFAULT_SCENARIO,
    ledger: Ledger | None = None,
) -> Fact | Missing:
    """The climatologically best sowing date, e.g. "8 February".

    The band is a **zero-based** day of year (the catalog's "1-365" is wrong), so
    1 is added before the date is formed; both the raw day-of-year Fact and the
    number of suitable sowing days per year are written to the ledger.

    The 20 perennial crops carry no sowing date at all -- their band is zero
    everywhere -- so they return Missing instead of a fabricated "1 January".
    """
    quantity = "optimal_sowing_date"
    legs = _categorical_legs(OPTIMAL_SOWING_DATE)
    window_legs = _count_legs(SUITABLE_SOWING_DAYS)
    chain = tuple(leg.label for leg in legs)

    band, gap = _resolve(quantity, lat, lon, crop, chain)
    if band is None:
        _record(ledger, gap)
        return gap

    all_legs = tuple(legs) + tuple(window_legs)
    result, failure = _fetch_or_gap(quantity, lat, lon, [band], all_legs, scenario)
    if failure is not None:
        _record(ledger, failure)
        return failure

    window = _evidence(
        result,
        band,
        "sowing_window_days",
        window_legs,
        unit="days per year",
        scaling=_NO_SCALING,
        valid_range=(0.0, 365.0),
    )
    _record(ledger, window)

    if band in PERENNIAL_BANDS:
        gap = Missing(
            quantity,
            MissingReason.OUT_OF_COVERAGE,
            chain,
            f"the point is inside the CropSuite grid, but CropSuite models no sowing date for "
            f"{crop_for_band(band)}: it is perennial or continuously established and its band is "
            "zero across the whole of Africa; see sowing_window_days for how much of the year "
            "establishment is possible",
        )
        _record(ledger, gap)
        return gap

    doy_fact = _evidence(
        result,
        band,
        "optimal_sowing_doy",
        legs,
        unit="day of year",
        scaling="none: raw zero-based day of year (the catalog's '1-365' is wrong)",
        valid_range=(0.0, 364.0),
    )
    _record(ledger, doy_fact)
    if not isinstance(doy_fact, Fact):
        return doy_fact

    doy = int(doy_fact.value) + 1  # zero-based band -> 1-based day of year
    when = dt.date(_DOY_YEAR, 1, 1) + dt.timedelta(days=doy - 1)
    label = f"{when.day} {when.strftime('%B')}"
    note = (
        f"{crop_for_band(band)}, CropSuite scenario {scenario}; day {doy} of the year, "
        "a 1991-2010 climatological average rather than advice for this season"
    )
    if doy_fact.note:
        note = f"{doy_fact.note}; {note}"
    fact = Fact.derive(
        quantity,
        label,
        "",
        [doy_fact],
        scaling_applied="zero-based day of year + 1, rendered as a calendar day",
        note=note,
    )
    _record(ledger, fact)
    return fact
