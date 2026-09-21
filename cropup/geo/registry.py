"""The Earth Engine source registry: which asset, which band, which arithmetic.

Two things live here and nowhere else.

**The catalogue.** ``cropup/data/ee_registry.json`` holds 92 live-probed datasets
with their coverage, resolution and the prose a human wrote after querying them.
:func:`registry` gives typed read-only access to it.

**The scaling traps of SPEC section 3.2.** Every one of them is a *named*
transform registered against ``(asset_id, band)``. No caller multiplies a raw
pixel by anything: it asks for a :class:`~cropup.evidence.SourceDescriptor` and
hands it to ``evidence.fact_from_reduce_region``, which applies the transform,
rejects the no-data sentinels and range-checks the result. That is the whole
point -- a scaling mistake here produces a *plausible* wrong number, which is
the failure mode this app is built to prevent, so the arithmetic is written down
once, next to the evidence that fixed it.

The traps, and how each was settled:

* ``ISDASOIL/.../silt_content`` -- the EE STAC documents ``exp(x/10)-1``. It is
  **wrong**; the raw value is already percent. 313 random African points sum
  clay+sand+silt to 98.0 on raw versus 86.9 with the log transform, and the raw
  fractions reproduce iSDA's own ``texture_class`` at both Tanzanian test points.
* ``ISDASOIL/.../nitrogen_total`` -- ``exp(x/100)-1``, the only iSDA property
  that uses ``/100``. ``/10`` would report 163% nitrogen.
* ``ISDASOIL/.../carbon_organic``, ``cation_exchange_capacity`` and the other
  log-transformed properties -- ``exp(x/10)-1``. ``ph`` is ``x/10``,
  ``bulk_density`` ``x/100``, ``clay_content``/``sand_content`` raw.
* ``projects/soilgrids-isric/nitrogen_mean`` -- ``/1000``, not the documented
  ``/100``; the EE copy is stored ten times the ISRIC mapped unit.
* POLARIS ``om`` and ``ksat`` are ``log10``: ``10**raw``. Raw ``om=0.293`` reads
  as a plausible 0.29% and is actually 1.96%.
* POLARIS ``alpha`` and ``hb`` are ``log10`` in kPa units, and ``n`` is stored
  **untransformed** -- verified here over six US points of contrasting texture
  (see :mod:`cropup.geo.soil`).
* ``global_ai`` needs ``/10000``; its sibling ``global_et0``, same provider and
  same docs page, needs **no** scaling.
* ERA5-Land precipitation and evaporation are in metres, and evaporation is
  negative for an upward flux.
* NASA Harvest probabilities are already 0-1; DEAF probabilities are 0-100.
* PEST-CHEMGRIDS carries negative no-data sentinels (-2, -1.5, -1) that a naive
  read reports as a real application rate.
* CropSuite ``optimal_sowing_date`` is a **zero-based** day of year although the
  docs say 1-365.

Imports ``config``, ``errors`` and ``evidence`` only: no ``ee`` and no other
``geo`` module, so it can be unit-tested with Earth Engine switched off.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterator, Mapping, Sequence

from ..config import Settings, get_settings
from ..errors import DataFileError
from ..evidence import Evidence, Fact, Missing, MissingReason, SourceDescriptor

__all__ = [
    "DatasetRecord",
    "Registry",
    "registry",
    "reload_registry",
    "Coverage",
    "COVERAGES",
    "coverage",
    "covers",
    "ScalingRule",
    "SCALING_RULES",
    "scaling_rule",
    "has_scaling",
    "apply_scaling",
    "sources_for",
    "source_for",
    "chain_for",
    "quantities",
    "resolve_chain",
    "TTLCache",
]


# ---------------------------------------------------------------------------
# 1. The catalogue: cropup/data/ee_registry.json
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetRecord:
    """One live-probed dataset, as written down in ``ee_registry.json``.

    The prose fields (``scaling``, ``coverage``, ``notes``) are the probe
    author's findings. They are documentation for humans; the machine-readable
    version of ``scaling`` is :data:`SCALING_RULES`.
    """

    asset_id: str
    ee_type: str = ""
    family: str = ""
    bands: tuple[str, ...] = ()
    scaling: str = ""
    units: str = ""
    coverage: str = ""
    resolution: str = ""
    temporal_range: str = ""
    license: str = ""
    notes: str = ""
    values: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "DatasetRecord":
        values = {
            key[len("value_") :]: str(value)
            for key, value in payload.items()
            if key.startswith("value_") and value is not None
        }
        return cls(
            asset_id=str(payload["asset_id"]),
            ee_type=str(payload.get("ee_type") or ""),
            family=str(payload.get("family") or ""),
            bands=tuple(payload.get("bands") or ()),
            scaling=str(payload.get("scaling") or ""),
            units=str(payload.get("units") or ""),
            coverage=str(payload.get("coverage") or ""),
            resolution=str(payload.get("resolution") or ""),
            temporal_range=str(payload.get("temporal_range") or ""),
            license=str(payload.get("license") or ""),
            notes=str(payload.get("notes") or ""),
            values=values,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "ee_type": self.ee_type,
            "family": self.family,
            "bands": list(self.bands),
            "scaling": self.scaling,
            "units": self.units,
            "coverage": self.coverage,
            "resolution": self.resolution,
            "temporal_range": self.temporal_range,
            "license": self.license,
            "notes": self.notes,
            "values": dict(self.values),
        }


class Registry:
    """Read-only view of ``ee_registry.json``."""

    def __init__(self, datasets: Sequence[DatasetRecord], test_points: Mapping[str, Sequence[float]], generated_from: str = "") -> None:
        self._datasets = tuple(datasets)
        self._by_id = {d.asset_id: d for d in self._datasets}
        self.test_points = {name: (float(v[0]), float(v[1])) for name, v in test_points.items()}
        self.generated_from = generated_from

    def datasets(self) -> tuple[DatasetRecord, ...]:
        return self._datasets

    def dataset(self, asset_id: str) -> DatasetRecord | None:
        """The record for an asset, or None. Several records in the file cover a
        group of assets under one heading, so a miss is normal and not an error."""
        return self._by_id.get(asset_id)

    def families(self) -> tuple[str, ...]:
        seen: list[str] = []
        for d in self._datasets:
            if d.family not in seen:
                seen.append(d.family)
        return tuple(seen)

    def by_family(self, prefix: str) -> tuple[DatasetRecord, ...]:
        lowered = prefix.lower()
        return tuple(d for d in self._datasets if d.family.lower().startswith(lowered))

    def search(self, text: str) -> tuple[DatasetRecord, ...]:
        lowered = text.lower()
        return tuple(d for d in self._datasets if lowered in d.asset_id.lower() or lowered in d.notes.lower())

    def __len__(self) -> int:
        return len(self._datasets)

    def __iter__(self) -> Iterator[DatasetRecord]:
        return iter(self._datasets)

    def __contains__(self, asset_id: object) -> bool:
        return asset_id in self._by_id

    def __repr__(self) -> str:
        return f"Registry({len(self._datasets)} datasets, {len(self.test_points)} test points)"


_registry: Registry | None = None
_registry_lock = threading.Lock()


def _load(settings: Settings | None = None) -> Registry:
    settings = settings or get_settings()
    path = settings.require_data_file("ee_registry")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataFileError(path, f"not valid JSON: {exc}") from exc
    try:
        raw_datasets = payload["datasets"]
    except (KeyError, TypeError) as exc:
        raise DataFileError(path, "expected a top-level 'datasets' list") from exc
    records = [DatasetRecord.from_json(entry) for entry in raw_datasets]
    return Registry(records, payload.get("test_points") or {}, str(payload.get("generated_from") or ""))


def registry(settings: Settings | None = None) -> Registry:
    """The process-wide catalogue, parsed once."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = _load(settings)
    return _registry


def reload_registry(settings: Settings | None = None) -> Registry:
    """Re-read the file. For tests and for tools/."""
    global _registry
    with _registry_lock:
        _registry = _load(settings)
    return _registry


# ---------------------------------------------------------------------------
# 2. Coverage: the cliffs of SPEC section 3.3, as data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Coverage:
    """A bounding box a source is known to cover, plus what it means outside it.

    The box is a cheap pre-filter so an out-of-coverage point is reported as
    ``OUT_OF_COVERAGE`` rather than burning a round trip to learn it is masked.
    Inside the box a pixel can still be masked -- that is a different, and also
    honest, answer.
    """

    name: str
    west: float
    south: float
    east: float
    north: float
    detail: str = ""

    def contains(self, lat: float, lon: float) -> bool:
        return self.south <= lat <= self.north and self.west <= lon <= self.east


COVERAGES: dict[str, Coverage] = {
    "global": Coverage("global", -180.0, -90.0, 180.0, 90.0, "global land"),
    # iSDA STAC footprint, quoted in ee_registry.json.
    "africa": Coverage("africa", -31.46, -35.22, 57.08, 37.98, "Africa only; masked elsewhere"),
    # POLARIS/OpenET/FRET/CDL. Verified masked (None, not 0) at both Tanzanian points.
    "conus": Coverage("conus", -125.0, 24.0, -66.0, 53.0, "conterminous United States only"),
    # CHIRPS grid.
    "chirps": Coverage("chirps", -180.0, -50.0, 180.0, 50.0, "50S-50N"),
    # CropSuite grid: lon -25..55, lat -36..39.
    "cropsuite": Coverage("cropsuite", -25.0, -36.0, 55.0, 39.0, "Africa only (CropSuite grid)"),
    "kenya": Coverage("kenya", 33.907, -4.670, 41.905, 4.623, "Kenya only"),
    "togo": Coverage("togo", -0.147, 6.104, 1.807, 11.139, "Togo only"),
}


def coverage(name: str | None) -> Coverage:
    """The named coverage, defaulting to global for sources that declare none."""
    if not name:
        return COVERAGES["global"]
    try:
        return COVERAGES[name]
    except KeyError as exc:
        known = ", ".join(sorted(COVERAGES))
        raise KeyError(f"unknown coverage {name!r} (known: {known})") from exc


def covers(source: SourceDescriptor, lat: float, lon: float) -> bool:
    """True when this source's footprint contains the point."""
    return coverage(source.coverage).contains(lat, lon)


# ---------------------------------------------------------------------------
# 3. Named transforms. One function per trap, so a stack trace names the bug.
# ---------------------------------------------------------------------------


def raw_value(value: float) -> float:
    """No transform: the stored number is already in the stated unit."""
    return value


def div_10(value: float) -> float:
    """x / 10 -- the ``gee:scale 0.1`` and ``d_factor 10`` case."""
    return value / 10.0


def div_100(value: float) -> float:
    """x / 100."""
    return value / 100.0


def div_1000(value: float) -> float:
    """x / 1000 -- SoilGrids nitrogen on EE, which is stored 10x the mapped unit."""
    return value / 1000.0


def div_10000(value: float) -> float:
    """x / 10000 -- global_ai and HiHydroSoil, but never global_et0."""
    return value / 10000.0


def isda_exp_decimetre(value: float) -> float:
    """exp(x/10) - 1 -- the iSDA natural-log back-transform."""
    return math.exp(value / 10.0) - 1.0


def isda_exp_centi(value: float) -> float:
    """exp(x/100) - 1 -- iSDA nitrogen_total only. Every sibling uses /10."""
    return math.exp(value / 100.0) - 1.0


def polaris_log10(value: float) -> float:
    """10 ** x -- POLARIS om, ksat, alpha and hb are stored as log10."""
    return 10.0**value


def kelvin_to_celsius(value: float) -> float:
    """x - 273.15 -- ERA5-Land and FLDAS store real Kelvin, not a scaled integer."""
    return value - 273.15


def metres_to_mm(value: float) -> float:
    """x * 1000 -- ERA5-Land precipitation is in metres."""
    return value * 1000.0


def upward_metres_to_mm(value: float) -> float:
    """x * -1000 -- ERA5-Land evaporation is metres and negative for an upward flux."""
    return value * -1000.0


def kg_m2_s_to_mm_day(value: float) -> float:
    """x * 86400 -- FLDAS fluxes are per second."""
    return value * 86400.0


def s2_reflectance(value: float) -> float:
    """x / 10000 -- Sentinel-2 L2A surface reflectance."""
    return value / 10000.0


def landsat_sr_reflectance(value: float) -> float:
    """x * 2.75e-05 - 0.2 -- Landsat Collection 2 Level 2 surface reflectance."""
    return value * 2.75e-05 - 0.2


def landsat_st_to_celsius(value: float) -> float:
    """x * 0.00341802 + 149.0 - 273.15 -- Landsat C2 L2 surface temperature to degC."""
    return value * 0.00341802 + 149.0 - 273.15


def percent_to_fraction(value: float) -> float:
    """x / 100 -- DEAF and WorldCereal probabilities are 0-100; NASA Harvest's are not."""
    return value / 100.0


def zero_based_doy_to_one_based(value: float) -> float:
    """x + 1 -- CropSuite sowing dates are 0-364 although the docs say 1-365."""
    return value + 1.0


# ---------------------------------------------------------------------------
# 4. Scaling rules, keyed by (asset_id, band)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalingRule:
    """The arithmetic between a stored pixel and a number a farmer can read."""

    name: str
    transform: Callable[[float], float]
    description: str
    unit: str = ""
    valid_range: tuple[float, float] | None = None
    invalid_values: tuple[float, ...] = ()

    def apply(self, raw: float) -> float:
        """Scale one raw pixel value.

        Raises ``ValueError`` on a no-data sentinel or a non-finite result
        rather than returning a number that looks like a measurement.
        """
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise TypeError(f"{self.name}: expected a number, got {type(raw).__name__}")
        if any(math.isclose(raw, s, rel_tol=1e-9, abs_tol=1e-9) for s in self.invalid_values):
            raise ValueError(f"{self.name}: raw {raw!r} is a no-data sentinel, not a measurement")
        out = float(self.transform(float(raw)))
        if not math.isfinite(out):
            raise ValueError(f"{self.name}: scaling {raw!r} produced {out!r}")
        return out


def _rule(name: str, transform: Callable[[float], float], description: str, unit: str = "", *,
          valid_range: tuple[float, float] | None = None,
          invalid_values: tuple[float, ...] = ()) -> ScalingRule:
    return ScalingRule(name, transform, description, unit, valid_range, invalid_values)


# Physically plausible envelopes. A value outside one is a masked or corrupt
# pixel, not a soil; evidence.fact_from_reduce_region turns it into Missing.
_PH_RANGE = (3.0, 10.0)
_PCT_RANGE = (0.0, 100.0)

_ISDA_EXP10 = _rule(
    "isda_exp_decimetre",
    isda_exp_decimetre,
    "exp(raw/10) - 1, the iSDA natural-log back-transform",
)

SCALING_RULES: dict[tuple[str, str | None], ScalingRule] = {
    # -- iSDA Africa, 30 m. Band is None: the transform is a property of the
    # asset and applies to mean_0_20 and mean_20_50 alike.
    ("ISDASOIL/Africa/v1/ph", None): _rule(
        "isda_ph", div_10, "raw/10 (iSDA stores pH x 10)", "pH", valid_range=_PH_RANGE),
    ("ISDASOIL/Africa/v1/clay_content", None): _rule(
        "isda_clay_raw", raw_value, "none: the raw value is already percent", "%", valid_range=_PCT_RANGE),
    ("ISDASOIL/Africa/v1/sand_content", None): _rule(
        "isda_sand_raw", raw_value, "none: the raw value is already percent", "%", valid_range=_PCT_RANGE),
    ("ISDASOIL/Africa/v1/silt_content", None): _rule(
        "isda_silt_raw",
        raw_value,
        "none: the raw value is already percent. The EE STAC documents exp(x/10)-1; "
        "that is wrong and would report 9% where the soil holds 23%",
        "%",
        valid_range=_PCT_RANGE,
    ),
    ("ISDASOIL/Africa/v1/carbon_organic", None): _rule(
        "isda_soc", isda_exp_decimetre, "exp(raw/10) - 1", "g/kg", valid_range=(0.0, 600.0)),
    ("ISDASOIL/Africa/v1/nitrogen_total", None): _rule(
        "isda_nitrogen",
        isda_exp_centi,
        "exp(raw/100) - 1. Note /100: every other log-transformed iSDA band uses /10, "
        "and /10 here would report 163% nitrogen",
        "g/kg",
        valid_range=(0.0, 50.0),
    ),
    ("ISDASOIL/Africa/v1/cation_exchange_capacity", None): _rule(
        "isda_cec", isda_exp_decimetre, "exp(raw/10) - 1", "cmol(+)/kg", valid_range=(0.0, 200.0)),
    ("ISDASOIL/Africa/v1/bulk_density", None): _rule(
        "isda_bulk_density", div_100, "raw/100", "g/cm3", valid_range=(0.1, 2.2)),
    ("ISDASOIL/Africa/v1/texture_class", None): _rule(
        "isda_texture_code", raw_value, "none: a class code, 1-12", "class", valid_range=(1.0, 12.0)),
    ("ISDASOIL/Africa/v1/bedrock_depth", None): _rule(
        "isda_bedrock_depth",
        raw_value,
        "none: centimetres. 200 is a censored maximum meaning '>= 200 cm'",
        "cm",
        valid_range=(0.0, 200.0),
    ),
    ("ISDASOIL/Africa/v1/aluminium_extractable", None): _ISDA_EXP10,
    ("ISDASOIL/Africa/v1/calcium_extractable", None): _ISDA_EXP10,
    ("ISDASOIL/Africa/v1/carbon_total", None): _ISDA_EXP10,
    ("ISDASOIL/Africa/v1/iron_extractable", None): _ISDA_EXP10,
    ("ISDASOIL/Africa/v1/magnesium_extractable", None): _ISDA_EXP10,
    ("ISDASOIL/Africa/v1/phosphorus_extractable", None): _ISDA_EXP10,
    ("ISDASOIL/Africa/v1/potassium_extractable", None): _ISDA_EXP10,
    ("ISDASOIL/Africa/v1/stone_content", None): _ISDA_EXP10,
    ("ISDASOIL/Africa/v1/sulphur_extractable", None): _ISDA_EXP10,
    ("ISDASOIL/Africa/v1/zinc_extractable", None): _ISDA_EXP10,
    # -- SoilGrids 2.0, 250 m, global.
    ("projects/soilgrids-isric/phh2o_mean", None): _rule(
        "soilgrids_ph", div_10, "raw/10 (d_factor 10)", "pH", valid_range=_PH_RANGE),
    ("projects/soilgrids-isric/clay_mean", None): _rule(
        "soilgrids_clay", div_10, "raw/10 (g/kg -> %)", "%", valid_range=_PCT_RANGE),
    ("projects/soilgrids-isric/sand_mean", None): _rule(
        "soilgrids_sand", div_10, "raw/10 (g/kg -> %)", "%", valid_range=_PCT_RANGE),
    ("projects/soilgrids-isric/silt_mean", None): _rule(
        "soilgrids_silt", div_10, "raw/10 (g/kg -> %)", "%", valid_range=_PCT_RANGE),
    ("projects/soilgrids-isric/soc_mean", None): _rule(
        "soilgrids_soc", div_10, "raw/10 (dg/kg -> g/kg)", "g/kg", valid_range=(0.0, 600.0)),
    ("projects/soilgrids-isric/nitrogen_mean", None): _rule(
        "soilgrids_nitrogen",
        div_1000,
        "raw/1000, not the documented /100: the EE copy is stored ten times the ISRIC "
        "mapped unit (verified EE/REST = 9.989 while all eight sibling properties are 1.000)",
        "g/kg",
        valid_range=(0.0, 50.0),
    ),
    ("projects/soilgrids-isric/cec_mean", None): _rule(
        "soilgrids_cec", div_10, "raw/10", "cmol(+)/kg", valid_range=(0.0, 200.0)),
    ("projects/soilgrids-isric/bdod_mean", None): _rule(
        "soilgrids_bdod", div_100, "raw/100", "g/cm3", valid_range=(0.1, 2.2)),
    ("projects/soilgrids-isric/cfvo_mean", None): _rule(
        "soilgrids_cfvo", div_10, "raw/10, not the /100 the community catalog page states", "vol%",
        valid_range=_PCT_RANGE),
    # -- POLARIS, 30 m, United States only.
    ("projects/sat-io/open-datasets/polaris/ph_mean", None): _rule(
        "polaris_ph", raw_value, "none: pH units", "pH", valid_range=_PH_RANGE),
    ("projects/sat-io/open-datasets/polaris/clay_mean", None): _rule(
        "polaris_clay", raw_value, "none: already percent", "%", valid_range=_PCT_RANGE),
    ("projects/sat-io/open-datasets/polaris/sand_mean", None): _rule(
        "polaris_sand", raw_value, "none: already percent", "%", valid_range=_PCT_RANGE),
    ("projects/sat-io/open-datasets/polaris/silt_mean", None): _rule(
        "polaris_silt", raw_value, "none: already percent", "%", valid_range=_PCT_RANGE),
    ("projects/sat-io/open-datasets/polaris/bd_mean", None): _rule(
        "polaris_bd", raw_value, "none: g/cm3", "g/cm3", valid_range=(0.1, 2.2)),
    ("projects/sat-io/open-datasets/polaris/om_mean", None): _rule(
        "polaris_om",
        polaris_log10,
        "10**raw: POLARIS stores organic matter as log10(%). Raw 0.293 reads as a "
        "plausible 0.29% and is actually 1.96%",
        "%",
        valid_range=(0.0, 100.0),
    ),
    ("projects/sat-io/open-datasets/polaris/ksat_mean", None): _rule(
        "polaris_ksat", polaris_log10, "10**raw: stored as log10(cm/hr)", "cm/hr",
        valid_range=(1e-4, 1000.0)),
    ("projects/sat-io/open-datasets/polaris/theta_s_mean", None): _rule(
        "polaris_theta_s", raw_value, "none: m3/m3 at saturation", "m3/m3", valid_range=(0.1, 0.9)),
    ("projects/sat-io/open-datasets/polaris/theta_r_mean", None): _rule(
        "polaris_theta_r", raw_value, "none: residual m3/m3", "m3/m3", valid_range=(0.0, 0.3)),
    ("projects/sat-io/open-datasets/polaris/alpha_mean", None): _rule(
        "polaris_vg_alpha",
        polaris_log10,
        "10**raw: van Genuchten alpha, stored as log10(1/kPa). Confirmed here by "
        "10**alpha == 1/(10**hb) at every probed point",
        "1/kPa",
        valid_range=(0.001, 10.0),
    ),
    ("projects/sat-io/open-datasets/polaris/n_mean", None): _rule(
        "polaris_vg_n",
        raw_value,
        "none: van Genuchten n is stored untransformed despite the POLARIS README. "
        "Probed values run 1.29-1.52 across sand, loam and clay; 10**raw would give 20-30",
        "",
        valid_range=(1.01, 5.0),
    ),
    ("projects/sat-io/open-datasets/polaris/hb_mean", None): _rule(
        "polaris_hb", polaris_log10, "10**raw: bubbling pressure, stored as log10(kPa)", "kPa",
        valid_range=(0.01, 1000.0)),
    ("projects/sat-io/open-datasets/polaris/lambda_mean", None): _rule(
        "polaris_lambda", raw_value, "none: pore-size distribution index", "", valid_range=(0.0, 5.0)),
    # -- HiHydroSoil v2.0, 250 m, global.
    ("projects/sat-io/open-datasets/HiHydroSoilv2_0/wcavail", None): _rule(
        "hihydrosoil_wcavail", div_10000, "raw/10000", "m3/m3", valid_range=(0.0, 0.7)),
    ("projects/sat-io/open-datasets/HiHydroSoilv2_0/ksat", None): _rule(
        "hihydrosoil_ksat", div_10000, "raw/10000", "cm/d", valid_range=(0.0, 100000.0)),
    # -- Climate normals.
    ("WORLDCLIM/V1/MONTHLY", "tavg"): _rule(
        "worldclim_temp", div_10, "raw x 0.1 (gee:scale 0.1)", "degC", valid_range=(-70.0, 60.0)),
    ("WORLDCLIM/V1/MONTHLY", "tmin"): _rule(
        "worldclim_temp", div_10, "raw x 0.1 (gee:scale 0.1)", "degC", valid_range=(-90.0, 60.0)),
    ("WORLDCLIM/V1/MONTHLY", "tmax"): _rule(
        "worldclim_temp", div_10, "raw x 0.1 (gee:scale 0.1)", "degC", valid_range=(-70.0, 70.0)),
    ("WORLDCLIM/V1/MONTHLY", "prec"): _rule(
        "worldclim_prec", raw_value, "none: already mm/month (gee:scale null)", "mm",
        valid_range=(0.0, 12000.0)),
    ("IDAHO_EPSCOR/TERRACLIMATE", "tmmn"): _rule(
        "terraclimate_temp", div_10, "raw x 0.1", "degC", valid_range=(-90.0, 60.0)),
    ("IDAHO_EPSCOR/TERRACLIMATE", "tmmx"): _rule(
        "terraclimate_temp", div_10, "raw x 0.1", "degC", valid_range=(-70.0, 70.0)),
    ("IDAHO_EPSCOR/TERRACLIMATE", "tavg"): _rule(
        "terraclimate_tavg",
        div_10,
        "(tmmn + tmmx)/2 then x 0.1: TerraClimate ships no tavg band",
        "degC",
        valid_range=(-70.0, 60.0),
    ),
    ("IDAHO_EPSCOR/TERRACLIMATE", "pr"): _rule(
        "terraclimate_prec", raw_value, "none: already mm/month", "mm", valid_range=(0.0, 12000.0)),
    ("IDAHO_EPSCOR/TERRACLIMATE", "aet"): _rule(
        "terraclimate_aet", div_10, "raw x 0.1", "mm", valid_range=(0.0, 2000.0)),
    ("IDAHO_EPSCOR/TERRACLIMATE", "pet"): _rule(
        "terraclimate_pet", div_10, "raw x 0.1", "mm", valid_range=(0.0, 2000.0)),
    ("IDAHO_EPSCOR/TERRACLIMATE", "def"): _rule(
        "terraclimate_def", div_10, "raw x 0.1", "mm", valid_range=(0.0, 2000.0)),
    ("IDAHO_EPSCOR/TERRACLIMATE", "pdsi"): _rule(
        "terraclimate_pdsi", div_100, "raw x 0.01", "index", valid_range=(-20.0, 20.0)),
    ("ECMWF/ERA5_LAND/MONTHLY_AGGR", "temperature_2m"): _rule(
        "era5_temperature", kelvin_to_celsius, "raw - 273.15: ERA5-Land stores real Kelvin", "degC",
        valid_range=(-90.0, 60.0)),
    ("ECMWF/ERA5_LAND/MONTHLY_AGGR", "total_precipitation_sum"): _rule(
        "era5_precipitation", metres_to_mm, "raw x 1000: ERA5-Land precipitation is in metres", "mm",
        valid_range=(0.0, 5000.0)),
    ("ECMWF/ERA5_LAND/DAILY_AGGR", "total_precipitation_sum"): _rule(
        "era5_precipitation", metres_to_mm, "raw x 1000: metres to mm", "mm", valid_range=(0.0, 2000.0)),
    ("ECMWF/ERA5_LAND/DAILY_AGGR", "total_evaporation_sum"): _rule(
        "era5_evaporation",
        upward_metres_to_mm,
        "raw x -1000: metres, and negative for an upward (evaporative) flux",
        "mm",
        valid_range=(-50.0, 50.0),
    ),
    ("ECMWF/ERA5_LAND/DAILY_AGGR", "potential_evaporation_sum"): _rule(
        "era5_evaporation", upward_metres_to_mm, "raw x -1000: metres, upward flux is negative", "mm",
        valid_range=(-50.0, 50.0)),
    ("projects/sat-io/open-datasets/global_ai/global_ai_yearly", "b1"): _rule(
        "global_ai",
        div_10000,
        "raw/10000. Its sibling global_et0 shares a docs page and needs no scaling at all",
        "index",
        valid_range=(0.0, 10.0),
    ),
    # -- Water and evapotranspiration.
    ("projects/usgs-ssebop/viirs_et_v6_dekadal", "et"): _rule(
        "ssebop_et", raw_value, "none: already mm per 10-day dekad", "mm", valid_range=(0.0, 400.0)),
    ("projects/usgs-ssebop/viirs_et_v6_monthly", "et"): _rule(
        "ssebop_et_monthly", raw_value, "none: already mm/month", "mm", valid_range=(0.0, 1000.0)),
    ("MODIS/061/MOD16A2GF", "ET"): _rule(
        "modis_et", div_10, "raw x 0.1", "mm", valid_range=(0.0, 400.0)),
    ("MODIS/061/MOD16A2GF", "PET"): _rule(
        "modis_pet", div_10, "raw x 0.1", "mm", valid_range=(0.0, 600.0)),
    ("projects/sat-io/open-datasets/global_et0/global_et0_monthly", "b1"): _rule(
        "global_et0_monthly", raw_value, "none: already mm/month. Do NOT apply the 1e-4 factor "
        "that its sibling global_ai uses", "mm", valid_range=(0.0, 1000.0)),
    ("projects/sat-io/open-datasets/global_et0/global_et0_yearly", "b1"): _rule(
        "global_et0_yearly", raw_value, "none: already mm/year", "mm", valid_range=(0.0, 6000.0)),
    ("projects/climate-engine/esi/4wk", "ESI"): _rule(
        "esi", raw_value, "none: a standardised anomaly (z-score)", "index", valid_range=(-6.0, 6.0)),
    ("projects/climate-engine/esi/12wk", "ESI"): _rule(
        "esi", raw_value, "none: a standardised anomaly (z-score)", "index", valid_range=(-6.0, 6.0)),
    ("projects/climate-engine/fret/forecast/eto", "eto"): _rule(
        "fret_eto", raw_value, "none: mm/day", "mm/day", valid_range=(0.0, 30.0)),
    ("NASA/SMAP/SPL4SMGP/008", "sm_surface"): _rule(
        "smap_sm", raw_value, "none: volumetric m3/m3", "m3/m3", valid_range=(0.0, 0.8)),
    ("NASA/SMAP/SPL4SMGP/008", "sm_rootzone"): _rule(
        "smap_sm", raw_value, "none: volumetric m3/m3", "m3/m3", valid_range=(0.0, 0.8)),
    ("NASA/FLDAS/NOAH01/C/GL/M/V001", "SoilMoi00_10cm_tavg"): _rule(
        "fldas_sm", raw_value, "none: already m3/m3", "m3/m3", valid_range=(0.0, 0.8)),
    ("NASA/FLDAS/NOAH01/C/GL/M/V001", "Rainf_f_tavg"): _rule(
        "fldas_flux", kg_m2_s_to_mm_day, "raw x 86400: kg/m2/s to mm/day", "mm/day",
        valid_range=(0.0, 2000.0)),
    ("NASA/FLDAS/NOAH01/C/GL/M/V001", "Tair_f_tavg"): _rule(
        "fldas_tair", kelvin_to_celsius, "raw - 273.15", "degC", valid_range=(-90.0, 60.0)),
    ("UCSB-CHG/CHIRPS/DAILY", "precipitation"): _rule(
        "chirps_precipitation", raw_value, "none: already mm/day", "mm", valid_range=(0.0, 1000.0)),
    ("projects/climate-engine-pro/assets/ce-chirps-prelim-pentad", "precipitation"): _rule(
        "chirps_prelim", raw_value, "none: already mm/day", "mm", valid_range=(0.0, 1000.0)),
    # -- Vegetation and thermal. Not in ee_registry.json; scale factors read from
    # the EE STAC and checked here against real pixels at Salinas 2025-08.
    ("COPERNICUS/S2_SR_HARMONIZED", None): _rule(
        "s2_reflectance", s2_reflectance, "raw/10000: Sentinel-2 L2A surface reflectance", "reflectance",
        valid_range=(-0.2, 1.6)),
    ("LANDSAT/LC08/C02/T1_L2", "ST_B10"): _rule(
        "landsat_surface_temperature", landsat_st_to_celsius,
        "raw x 0.00341802 + 149.0 K, then to degC", "degC", valid_range=(-90.0, 80.0)),
    ("LANDSAT/LC09/C02/T1_L2", "ST_B10"): _rule(
        "landsat_surface_temperature", landsat_st_to_celsius,
        "raw x 0.00341802 + 149.0 K, then to degC", "degC", valid_range=(-90.0, 80.0)),
    ("LANDSAT/LC08/C02/T1_L2", None): _rule(
        "landsat_sr_reflectance", landsat_sr_reflectance, "raw x 2.75e-05 - 0.2", "reflectance",
        valid_range=(-0.2, 1.6)),
    ("LANDSAT/LC09/C02/T1_L2", None): _rule(
        "landsat_sr_reflectance", landsat_sr_reflectance, "raw x 2.75e-05 - 0.2", "reflectance",
        valid_range=(-0.2, 1.6)),
    # -- Cropland context.
    ("projects/sat-io/open-datasets/DEAF/CROPLAND-EXTENT/prob", "b1"): _rule(
        "deaf_cropland_probability", percent_to_fraction,
        "raw/100: DEAF probabilities are 0-100, unlike NASA Harvest's which are already 0-1",
        "ratio", valid_range=(0.0, 1.0)),
    ("projects/sat-io/open-datasets/nasa-harvest/kenya_2019_cropland_probability", "b1"): _rule(
        "nasa_harvest_probability", raw_value, "none: already a 0-1 float", "ratio",
        valid_range=(0.0, 1.0)),
    ("projects/sat-io/open-datasets/nasa-harvest/togo_cropland_probability", "b1"): _rule(
        "nasa_harvest_probability", raw_value, "none: already a 0-1 float", "ratio",
        valid_range=(0.0, 1.0)),
    ("ESA/WorldCereal/2021/MODELS/v100", "confidence"): _rule(
        "worldcereal_confidence", percent_to_fraction, "raw/100", "ratio", valid_range=(0.0, 1.0)),
    ("ESA/WorldCereal/2021/MODELS/v100", "classification"): _rule(
        "worldcereal_classification", raw_value,
        "none: literally 0 or 100 meaning absent/present, NOT a percentage", "class"),
    ("projects/sat-io/open-datasets/GFSAD/GCEP30", "b1"): _rule(
        "gcep30_class", raw_value,
        "none: 0 water, 1 NON-cropland, 2 cropland. Note 1 means cropland in DEAF", "class",
        valid_range=(0.0, 2.0)),
    ("projects/sat-io/open-datasets/GFSAD/LGRIP30", "b1"): _rule(
        "lgrip30_class", raw_value, "none: 0 water, 1 non-cropland, 2 irrigated, 3 rainfed", "class",
        valid_range=(0.0, 3.0)),
    ("ESA/WorldCover/v200", "Map"): _rule(
        "worldcover_class", raw_value, "none: 10/20/30/40/... land-cover codes (40 = cropland)", "class",
        valid_range=(10.0, 100.0)),
    # -- Farm inputs.
    ("projects/sat-io/open-datasets/NPKGRIDS", None): _rule(
        "npkgrids_rate", raw_value, "none: already kg/ha/yr (float32)", "kg/ha/yr",
        valid_range=(0.0, 2000.0), invalid_values=(-1.0,)),
    ("projects/sat-io/open-datasets/PEST-CHEMGRIDS/application_rates", "application_rate"): _rule(
        "pest_chemgrids_rate",
        raw_value,
        "none: kg active ingredient/ha/yr, but -2, -1.5 and -1 are no-data sentinels that a "
        "naive read reports as a real application rate",
        "kg/ha/yr",
        valid_range=(0.0, 500.0),
        invalid_values=(-2.0, -1.5, -1.0),
    ),
    ("projects/sat-io/open-datasets/PEST-CHEMGRIDS/quality_index", "quality_index"): _rule(
        "pest_chemgrids_quality",
        raw_value,
        "none: 0-1, but -1 is no-data and 0 marks a pixel with no estimate",
        "index",
        valid_range=(0.0, 1.0),
        invalid_values=(-1.0, 0.0),
    ),
    # -- CropSuite, Africa only.
    ("projects/sat-io/open-datasets/CROP_SUITE/crop_suitability", None): _rule(
        "cropsuite_suitability", raw_value, "none: the int8 value is already 0-100 percent", "%",
        valid_range=(0.0, 100.0), invalid_values=(-1.0,)),
    ("projects/sat-io/open-datasets/CROP_SUITE/climate_suitability", None): _rule(
        "cropsuite_suitability", raw_value, "none: already 0-100 percent", "%",
        valid_range=(0.0, 100.0), invalid_values=(-1.0,)),
    ("projects/sat-io/open-datasets/CROP_SUITE/optimal_sowing_date", None): _rule(
        "cropsuite_sowing_date",
        zero_based_doy_to_one_based,
        "raw + 1: the stored day of year is zero-based (0-364) although the docs say 1-365",
        "day of year",
        valid_range=(1.0, 366.0),
        invalid_values=(-1.0,),
    ),
    ("projects/sat-io/open-datasets/CROP_SUITE/suitable_sowing_days", None): _rule(
        "cropsuite_sowing_days", raw_value, "none: a day count", "days", valid_range=(0.0, 366.0),
        invalid_values=(-1.0,)),
    ("projects/sat-io/open-datasets/CROP_SUITE/multiple_cropping", None): _rule(
        "cropsuite_multiple_cropping", raw_value, "none: harvests per year, 0 means not growable",
        "harvests/yr", valid_range=(0.0, 3.0), invalid_values=(-1.0,)),
}


def scaling_rule(asset_id: str, band: str | None = None) -> ScalingRule | None:
    """The rule for a band, falling back to the asset-wide rule. None if unknown."""
    if (asset_id, band) in SCALING_RULES:
        return SCALING_RULES[(asset_id, band)]
    return SCALING_RULES.get((asset_id, None))


def has_scaling(asset_id: str, band: str | None = None) -> bool:
    return scaling_rule(asset_id, band) is not None


def apply_scaling(asset_id: str, band: str | None, raw: float) -> float:
    """Scale one raw pixel value from a named asset and band.

    Raises ``KeyError`` when no rule is registered -- an unregistered asset is a
    transform nobody has verified, and guessing "probably no scaling" is exactly
    how a plausible wrong number gets onto a farmer's screen.
    """
    rule = scaling_rule(asset_id, band)
    if rule is None:
        raise KeyError(
            f"no scaling rule registered for {asset_id!r} band {band!r}; "
            "add one to cropup.geo.registry.SCALING_RULES after verifying it against real pixels"
        )
    return rule.apply(raw)


# ---------------------------------------------------------------------------
# 5. Source descriptors and the chains of SPEC section 3.1
# ---------------------------------------------------------------------------


def _source(
    quantity: str,
    asset_id: str,
    band: str | None,
    *,
    resolution_m: float,
    coverage_name: str,
    unit: str | None = None,
    result_key: str | None = None,
    is_static: bool = False,
    stale_after_days: int | None = None,
    valid_range: tuple[float, float] | None = None,
) -> SourceDescriptor:
    """Build a descriptor, taking the transform and unit from the scaling rule."""
    rule = scaling_rule(asset_id, band)
    if rule is None:
        raise KeyError(f"no scaling rule for {asset_id!r} band {band!r} (quantity {quantity!r})")
    return SourceDescriptor(
        quantity=quantity,
        asset_id=asset_id,
        unit=unit if unit is not None else rule.unit,
        band=band,
        result_key=result_key,
        resolution_m=resolution_m,
        scaling=rule.description,
        transform=rule.transform,
        valid_range=valid_range if valid_range is not None else rule.valid_range,
        invalid_values=rule.invalid_values,
        stale_after_days=stale_after_days,
        is_static=is_static,
        coverage=coverage_name,
    )


def _chain(*sources: SourceDescriptor) -> tuple[SourceDescriptor, ...]:
    """Stamp a fallback order onto a group of sources for one quantity."""
    assets = tuple(s.asset_id for s in sources)
    return tuple(replace(s, chain=assets, chain_position=i) for i, s in enumerate(sources))


_ISDA_TOPSOIL = "mean_0_20"
_SG_TOPSOIL = "_0-5cm_mean"
_ISDA_RES = 30.0
_POLARIS_RES = 30.0
_SOILGRIDS_RES = 250.0
_WORLDCLIM_RES = 927.66
_TERRACLIMATE_RES = 4638.31


def _isda(quantity: str, asset: str, band: str = _ISDA_TOPSOIL) -> SourceDescriptor:
    # 18 of the 21 iSDA assets publish mean_0_20; texture_class and bedrock_depth
    # name their bands differently, which is why the band is a parameter.
    return _source(
        quantity,
        f"ISDASOIL/Africa/v1/{asset}",
        band,
        resolution_m=_ISDA_RES,
        coverage_name="africa",
        is_static=True,
    )


def _polaris(quantity: str, var: str) -> SourceDescriptor:
    # Every POLARIS asset is a 6-image collection, one image per depth, band b1.
    return _source(
        quantity,
        f"projects/sat-io/open-datasets/polaris/{var}_mean",
        "b1",
        resolution_m=_POLARIS_RES,
        coverage_name="conus",
        is_static=True,
    )


def _soilgrids(quantity: str, asset: str, prop: str) -> SourceDescriptor:
    return _source(
        quantity,
        f"projects/soilgrids-isric/{asset}",
        f"{prop}{_SG_TOPSOIL}",
        resolution_m=_SOILGRIDS_RES,
        coverage_name="global",
        is_static=True,
    )


def _worldclim(quantity: str, band: str, month: int) -> SourceDescriptor:
    # toBands() on the 12-image collection yields '01_tavg' ... '12_prec'.
    return _source(
        quantity,
        "WORLDCLIM/V1/MONTHLY",
        band,
        result_key=f"{month:02d}_{band}",
        resolution_m=_WORLDCLIM_RES,
        coverage_name="global",
        is_static=True,
    )


def _terraclimate_normal(quantity: str, band: str, month: int) -> SourceDescriptor:
    # climate.py builds one band per month named '<band>_01' ... '<band>_12'.
    return _source(
        quantity,
        "IDAHO_EPSCOR/TERRACLIMATE",
        band,
        result_key=f"{band}_{month:02d}",
        resolution_m=_TERRACLIMATE_RES,
        coverage_name="global",
        is_static=True,
    )


def _soil_chains() -> dict[str, tuple[SourceDescriptor, ...]]:
    """SPEC 3.1: iSDA (Africa) -> POLARIS (US) -> SoilGrids (global)."""
    chains: dict[str, tuple[SourceDescriptor, ...]] = {
        "soil_ph": _chain(
            _isda("soil_ph", "ph"),
            _polaris("soil_ph", "ph"),
            _soilgrids("soil_ph", "phh2o_mean", "phh2o"),
        ),
        "soil_clay_pct": _chain(
            _isda("soil_clay_pct", "clay_content"),
            _polaris("soil_clay_pct", "clay"),
            _soilgrids("soil_clay_pct", "clay_mean", "clay"),
        ),
        "soil_sand_pct": _chain(
            _isda("soil_sand_pct", "sand_content"),
            _polaris("soil_sand_pct", "sand"),
            _soilgrids("soil_sand_pct", "sand_mean", "sand"),
        ),
        "soil_silt_pct": _chain(
            _isda("soil_silt_pct", "silt_content"),
            _polaris("soil_silt_pct", "silt"),
            _soilgrids("soil_silt_pct", "silt_mean", "silt"),
        ),
        # POLARIS maps organic matter, not organic carbon, so a US point falls
        # through to SoilGrids rather than converting OM with an assumed factor.
        "soil_soc": _chain(
            _isda("soil_soc", "carbon_organic"),
            _soilgrids("soil_soc", "soc_mean", "soc"),
        ),
        "soil_nitrogen": _chain(
            _isda("soil_nitrogen", "nitrogen_total"),
            _soilgrids("soil_nitrogen", "nitrogen_mean", "nitrogen"),
        ),
        "soil_cec": _chain(
            _isda("soil_cec", "cation_exchange_capacity"),
            _soilgrids("soil_cec", "cec_mean", "cec"),
        ),
        "soil_bulk_density": _chain(
            _isda("soil_bulk_density", "bulk_density"),
            _polaris("soil_bulk_density", "bd"),
            _soilgrids("soil_bulk_density", "bdod_mean", "bdod"),
        ),
        "soil_texture_code": _chain(_isda("soil_texture_code", "texture_class", "texture_0_20")),
        "soil_bedrock_depth": _chain(_isda("soil_bedrock_depth", "bedrock_depth", "mean_0_200")),
        "soil_om_pct": _chain(_polaris("soil_om_pct", "om")),
        "soil_theta_s": _chain(_polaris("soil_theta_s", "theta_s")),
        "soil_theta_r": _chain(_polaris("soil_theta_r", "theta_r")),
        "soil_ksat": _chain(_polaris("soil_ksat", "ksat")),
        "soil_vg_alpha": _chain(_polaris("soil_vg_alpha", "alpha")),
        "soil_vg_n": _chain(_polaris("soil_vg_n", "n")),
        # Measured available water content, for points POLARIS does not reach.
        "soil_available_water": _chain(
            _source(
                "soil_available_water",
                "projects/sat-io/open-datasets/HiHydroSoilv2_0/wcavail",
                "b1",
                resolution_m=253.38,
                coverage_name="global",
                is_static=True,
            )
        ),
    }
    return chains


def _climate_chains() -> dict[str, tuple[SourceDescriptor, ...]]:
    """Monthly climate normals, one descriptor per month per variable.

    WorldClim (1 km, 1960-1990) is primary; TerraClimate (4.6 km, from which
    climate.py builds 1991-2020 normals) is the fallback. They agree on the
    Koppen code at all three CropUp test points.
    """
    chains: dict[str, tuple[SourceDescriptor, ...]] = {}
    for month in range(1, 13):
        chains[f"climate_tavg_{month:02d}"] = _chain(
            _worldclim(f"climate_tavg_{month:02d}", "tavg", month),
            _terraclimate_normal(f"climate_tavg_{month:02d}", "tavg", month),
        )
        chains[f"climate_prec_{month:02d}"] = _chain(
            _worldclim(f"climate_prec_{month:02d}", "prec", month),
            _terraclimate_normal(f"climate_prec_{month:02d}", "pr", month),
        )
    chains["aridity_index"] = _chain(
        _source(
            "aridity_index",
            "projects/sat-io/open-datasets/global_ai/global_ai_yearly",
            "b1",
            resolution_m=1000.0,
            coverage_name="global",
            is_static=True,
        )
    )
    return chains


def _field_chains() -> dict[str, tuple[SourceDescriptor, ...]]:
    """The remaining SPEC 3.1 rows, for the geo modules other packages own.

    Nothing in this module fetches them; they are here so that the scaling and
    the fallback order are written down once, in data, for vegetation.py,
    thermal.py, water.py, context.py and suitability.py.
    """
    stale = 45  # SPEC default freshness limit for an in-season observation
    return {
        "et_actual": _chain(
            _source("et_actual", "projects/usgs-ssebop/viirs_et_v6_dekadal", "et",
                    resolution_m=1074.0, coverage_name="global", stale_after_days=30),
            _source("et_actual", "IDAHO_EPSCOR/TERRACLIMATE", "aet",
                    resolution_m=_TERRACLIMATE_RES, coverage_name="global", stale_after_days=400),
        ),
        "et0": _chain(
            _source("et0", "projects/sat-io/open-datasets/global_et0/global_et0_monthly", "b1",
                    resolution_m=1000.0, coverage_name="global", is_static=True),
        ),
        "et0_forecast": _chain(
            _source("et0_forecast", "projects/climate-engine/fret/forecast/eto", "eto",
                    resolution_m=2540.0, coverage_name="conus", stale_after_days=2),
        ),
        "evaporative_stress_index": _chain(
            _source("evaporative_stress_index", "projects/climate-engine/esi/4wk", "ESI",
                    resolution_m=5566.0, coverage_name="global", stale_after_days=stale),
            _source("evaporative_stress_index", "projects/climate-engine/esi/12wk", "ESI",
                    resolution_m=5566.0, coverage_name="global", stale_after_days=stale),
        ),
        "soil_moisture_surface": _chain(
            _source("soil_moisture_surface", "NASA/SMAP/SPL4SMGP/008", "sm_surface",
                    resolution_m=10593.0, coverage_name="global", stale_after_days=10),
            _source("soil_moisture_surface", "NASA/FLDAS/NOAH01/C/GL/M/V001", "SoilMoi00_10cm_tavg",
                    resolution_m=11132.0, coverage_name="global", stale_after_days=120),
        ),
        "soil_moisture_rootzone": _chain(
            _source("soil_moisture_rootzone", "NASA/SMAP/SPL4SMGP/008", "sm_rootzone",
                    resolution_m=10593.0, coverage_name="global", stale_after_days=10),
        ),
        "precipitation": _chain(
            _source("precipitation", "UCSB-CHG/CHIRPS/DAILY", "precipitation",
                    resolution_m=5566.0, coverage_name="chirps", stale_after_days=stale),
            _source("precipitation", "projects/climate-engine-pro/assets/ce-chirps-prelim-pentad",
                    "precipitation", resolution_m=5566.0, coverage_name="chirps", stale_after_days=stale),
        ),
        "land_surface_temperature": _chain(
            _source("land_surface_temperature", "LANDSAT/LC08/C02/T1_L2", "ST_B10",
                    resolution_m=30.0, coverage_name="global", stale_after_days=stale),
            _source("land_surface_temperature", "LANDSAT/LC09/C02/T1_L2", "ST_B10",
                    resolution_m=30.0, coverage_name="global", stale_after_days=stale),
        ),
        "cropland_probability": _chain(
            _source("cropland_probability", "projects/sat-io/open-datasets/DEAF/CROPLAND-EXTENT/prob",
                    "b1", resolution_m=10.0, coverage_name="africa", is_static=True),
        ),
        "cropland_class": _chain(
            _source("cropland_class", "projects/sat-io/open-datasets/GFSAD/GCEP30", "b1",
                    resolution_m=30.0, coverage_name="global", is_static=True),
            _source("cropland_class", "ESA/WorldCover/v200", "Map",
                    resolution_m=10.0, coverage_name="global", is_static=True),
        ),
        "irrigation_class": _chain(
            _source("irrigation_class", "projects/sat-io/open-datasets/GFSAD/LGRIP30", "b1",
                    resolution_m=30.0, coverage_name="global", is_static=True),
        ),
    }


SOURCES: dict[str, tuple[SourceDescriptor, ...]] = {}
SOURCES.update(_soil_chains())
SOURCES.update(_climate_chains())
SOURCES.update(_field_chains())


def quantities() -> tuple[str, ...]:
    """Every quantity with a registered source chain."""
    return tuple(SOURCES)


def sources_for(quantity: str) -> tuple[SourceDescriptor, ...]:
    """The chain for a quantity, primary first."""
    try:
        return SOURCES[quantity]
    except KeyError as exc:
        raise KeyError(f"no source chain registered for quantity {quantity!r}") from exc


def source_for(quantity: str, asset_id: str) -> SourceDescriptor:
    """One named link of a chain."""
    for source in sources_for(quantity):
        if source.asset_id == asset_id:
            return source
    raise KeyError(f"{asset_id!r} is not in the chain for {quantity!r}")


def chain_for(quantity: str) -> tuple[str, ...]:
    """The asset ids of a chain, in fallback order."""
    return tuple(s.asset_id for s in sources_for(quantity))


# ---------------------------------------------------------------------------
# 6. The chain resolver
# ---------------------------------------------------------------------------

# Which diagnosis survives when several sources fail differently. A source that
# blew up tells us less than one that was simply masked, so it wins.
_REASON_PRIORITY = {
    MissingReason.SOURCE_FAILED: 3,
    MissingReason.STALE: 2,
    MissingReason.MASKED: 1,
    MissingReason.OUT_OF_COVERAGE: 0,
    MissingReason.NOT_REQUESTED: 0,
}


def resolve_chain(
    quantity: str,
    fetch: Callable[[SourceDescriptor], Evidence | None],
    *,
    sources: Sequence[SourceDescriptor] | None = None,
    lat: float | None = None,
    lon: float | None = None,
    detail: str | None = None,
) -> Fact | Missing:
    """Walk a source chain until one produces a measurement.

    ``fetch`` does the Earth Engine work for a single source and returns the
    ``Fact`` or ``Missing`` it produced (``None`` counts as a failure). It is
    called lazily and in order, so a point covered by the primary source costs
    one call; ``soil.py`` uses that to batch one ``reduceRegion`` per tier and
    only reach for the next tier when the first came back masked.

    Sources whose coverage box excludes the point are skipped without a call and
    named in the resulting ``Missing``, so "POLARIS does not cover Tanzania" and
    "POLARIS had no pixel here" stay different answers.
    """
    chain = tuple(sources) if sources is not None else sources_for(quantity)
    attempted: list[str] = []
    skipped: list[str] = []
    reasons: list[Missing] = []

    for source in chain:
        if lat is not None and lon is not None and not covers(source, lat, lon):
            skipped.append(f"{source.asset_id} ({coverage(source.coverage).detail})")
            continue
        attempted.append(source.asset_id)
        evidence = fetch(source)
        if isinstance(evidence, Fact):
            return evidence
        if isinstance(evidence, Missing):
            reasons.append(evidence)
        else:
            reasons.append(
                Missing(quantity, MissingReason.SOURCE_FAILED, (source.asset_id,),
                        f"{source.asset_id} returned no evidence")
            )

    if not chain:
        return Missing(quantity, MissingReason.NOT_REQUESTED, (), detail or "no sources are registered")

    if attempted:
        reason = max((m.reason for m in reasons), key=lambda r: _REASON_PRIORITY.get(r, 0))
        parts = [f"{asset}: {m.detail}" for asset, m in zip(attempted, reasons) if m.detail]
    else:
        reason = MissingReason.OUT_OF_COVERAGE
        parts = []
    if skipped:
        parts.append("not attempted (outside coverage): " + ", ".join(skipped))
    if detail:
        parts.append(detail)
    return Missing(quantity, reason, tuple(attempted), "; ".join(parts) or None)


# ---------------------------------------------------------------------------
# 7. A small per-point cache (SPEC 3.4)
# ---------------------------------------------------------------------------


class TTLCache:
    """Bounded time-to-live cache for Earth Engine results.

    Soil and climate normals do not change between two questions about the same
    field, and the vendored pipeline's 28.7 s is mostly round trips. Keys are
    built by the caller; :meth:`point_key` rounds coordinates to ~10 m so two
    taps on the same field hit the same entry.
    """

    def __init__(self, ttl_s: float, max_entries: int) -> None:
        self.ttl_s = float(ttl_s)
        self.max_entries = int(max_entries)
        self._store: dict[Any, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def point_key(prefix: str, lat: float, lon: float, *extra: Any) -> tuple[Any, ...]:
        return (prefix, round(float(lat), 4), round(float(lon), 4)) + tuple(extra)

    def get(self, key: Any) -> Any | None:
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            stored_at, value = entry
            if now - stored_at > self.ttl_s:
                self._store.pop(key, None)
                return None
            return value

    def put(self, key: Any, value: Any) -> Any:
        now = time.monotonic()
        with self._lock:
            if len(self._store) >= self.max_entries:
                # Drop the oldest entry; insertion order is good enough here.
                oldest = min(self._store, key=lambda k: self._store[k][0])
                self._store.pop(oldest, None)
            self._store[key] = (now, value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
