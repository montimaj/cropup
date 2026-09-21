"""Soil properties from Earth Engine: iSDA -> POLARIS -> SoilGrids.

SPEC section 3.1 gives the chain and section 3.2 gives the traps. The transforms
themselves live in :mod:`cropup.geo.registry`; this module decides *which* source
to ask, batches one ``reduceRegion`` per tier, and turns what comes back into
``Fact`` or ``Missing``.

Three things it deliberately does not do:

* it never converts a masked pixel to 0 -- POLARIS returns ``None``, not 0,
  outside the United States, and that becomes ``Missing``, not a soil;
* it never fills a gap with a default. The vendored backend interpolated pH 7.0
  and sand 30% into user-facing prose (SPEC 2.1 bug 4); here a missing pH is a
  ``Missing`` and the renderer refuses to print it;
* it does not silently mix depths. iSDA is 0-20 cm, POLARIS and SoilGrids are
  0-5 cm, and every Fact says which in its note.

**Hydraulics.** Where POLARIS covers the point, plant-available water capacity is
computed from the van Genuchten curve rather than guessed from texture. The
POLARIS README is ambiguous about which parameters are log-transformed, so the
interpretation used here was checked against six US points of contrasting
texture (0-5 cm, probed 2026-09-17):

    point               clay  sand   alpha(1/kPa)    n     ksat cm/h   PAWC
    Sand Hills NE        3.5  88.6      0.788      1.515     25.34     0.063
    Ames IA             20.0  46.1      0.545      1.357      1.61     0.112
    Salinas CA task pt  15.2  46.2      0.476      1.360      2.81     0.119
    Salinas city        30.2  34.0      0.315      1.289      0.27     0.125
    Mississippi delta   12.6  56.3*     0.323      1.361      2.78     0.128
    Imperial CA         30.0  23.7      0.326      1.293      0.93     0.138
    (* silt)

``alpha`` and ``hb`` are log10 and reciprocal to each other at every point, so
they share a pressure unit (kPa); ``n`` is stored untransformed -- ``10**raw``
would give 20-30, which is not a van Genuchten n. The resulting PAWC ranks sand
below loam below clay-rich, at textbook magnitudes, which is the check that
matters.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Sequence

from .. import bootstrap
from ..config import Settings, get_settings
from ..errors import EarthEngineUnavailable
from ..evidence import (
    Evidence,
    Fact,
    Ledger,
    Missing,
    MissingReason,
    SourceDescriptor,
    fact_from_reduce_region,
)
from . import registry

__all__ = [
    "get_soil",
    "SOIL_QUANTITIES",
    "ISDA_TEXTURE_CLASSES",
    "usda_texture_class",
    "van_genuchten_theta",
    "plant_available_water",
    "clear_cache",
]

# The properties SPEC asks for, plus the POLARIS hydraulics. Order is the order
# they are resolved in, and the order they appear in the returned dict.
SOIL_QUANTITIES: tuple[str, ...] = (
    "soil_ph",
    "soil_clay_pct",
    "soil_sand_pct",
    "soil_silt_pct",
    "soil_soc",
    "soil_nitrogen",
    "soil_cec",
    "soil_bulk_density",
    "soil_texture_code",
    "soil_om_pct",
    "soil_theta_s",
    "soil_theta_r",
    "soil_ksat",
    "soil_vg_alpha",
    "soil_vg_n",
)

# The four van Genuchten parameters needed for a water-retention curve.
_VG_QUANTITIES = ("soil_theta_r", "soil_theta_s", "soil_vg_alpha", "soil_vg_n")

# Matric potentials, kPa. Field capacity is taken at 33 kPa (1/3 bar), the
# convention for medium-textured soils; permanent wilting point at 1500 kPa.
FIELD_CAPACITY_KPA = 33.0
WILTING_POINT_KPA = 1500.0

# iSDA ships this legend on the asset itself (texture_0_20_class_names).
ISDA_TEXTURE_CLASSES: dict[int, str] = {
    1: "Clay",
    2: "Silty Clay",
    3: "Sandy Clay",
    4: "Clay Loam",
    5: "Silty Clay Loam",
    6: "Sandy Clay Loam",
    7: "Loam",
    8: "Silt Loam",
    9: "Sandy Loam",
    10: "Silt",
    11: "Loamy Sand",
    12: "Sand",
}

# One batched reduceRegion per tier: its member assets, the depth its topsoil
# band covers, and the scale to reduce at.
_TIERS: dict[str, tuple[str, float]] = {
    "isda": ("0-20 cm", 30.0),
    "polaris": ("0-5 cm", 30.0),
    "soilgrids": ("0-5 cm", 250.0),
    "hihydrosoil": ("0-5 cm", 253.38),
}

_POLARIS_DEPTH = "0_5"
_HIHYDROSOIL_IMAGE = "WCavail_0-5cm_M_250m"

_cache = registry.TTLCache(get_settings().ee_cache_ttl_s, get_settings().ee_cache_max_entries)


def clear_cache() -> None:
    """Forget cached point results. For tests and for tools/."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Pure helpers: no Earth Engine, unit-testable on their own
# ---------------------------------------------------------------------------


def usda_texture_class(clay_pct: float, sand_pct: float, silt_pct: float) -> str | None:
    """USDA soil texture triangle. Returns None if the fractions classify nowhere.

    The three fractions are normalised to 100 first, because iSDA models each
    property independently and its fractions sum to 95-98 rather than exactly
    100.
    """
    total = float(clay_pct) + float(sand_pct) + float(silt_pct)
    if total <= 0 or not math.isfinite(total):
        return None
    clay = 100.0 * float(clay_pct) / total
    sand = 100.0 * float(sand_pct) / total
    silt = 100.0 * float(silt_pct) / total

    if silt + 1.5 * clay < 15:
        return "Sand"
    if silt + 1.5 * clay >= 15 and silt + 2.0 * clay < 30:
        return "Loamy Sand"
    if (7 <= clay < 20 and sand > 52 and silt + 2.0 * clay >= 30) or (clay < 7 and silt < 50 and silt + 2.0 * clay >= 30):
        return "Sandy Loam"
    if 7 <= clay < 27 and 28 <= silt < 50 and sand <= 52:
        return "Loam"
    if (silt >= 50 and 12 <= clay < 27) or (50 <= silt < 80 and clay < 12):
        return "Silt Loam"
    if silt >= 80 and clay < 12:
        return "Silt"
    if 20 <= clay < 35 and silt < 28 and sand > 45:
        return "Sandy Clay Loam"
    if 27 <= clay < 40 and 20 < sand <= 45:
        return "Clay Loam"
    if 27 <= clay < 40 and sand <= 20:
        return "Silty Clay Loam"
    if clay >= 35 and sand > 45:
        return "Sandy Clay"
    if clay >= 40 and silt >= 40:
        return "Silty Clay"
    if clay >= 40 and sand <= 45 and silt < 40:
        return "Clay"
    return None


def van_genuchten_theta(theta_r: float, theta_s: float, alpha_per_kpa: float, n: float, head_kpa: float) -> float:
    """Volumetric water content at a matric potential, m3/m3.

    theta(h) = theta_r + (theta_s - theta_r) / [1 + (alpha*h)^n]^(1 - 1/n)
    """
    if n <= 1.0:
        raise ValueError(f"van Genuchten n must exceed 1, got {n!r}")
    if theta_s <= theta_r:
        raise ValueError(f"theta_s ({theta_s!r}) must exceed theta_r ({theta_r!r})")
    m = 1.0 - 1.0 / n
    saturation = (1.0 + (alpha_per_kpa * head_kpa) ** n) ** (-m)
    return theta_r + (theta_s - theta_r) * saturation


def plant_available_water(
    theta_r: float,
    theta_s: float,
    alpha_per_kpa: float,
    n: float,
    *,
    field_capacity_kpa: float = FIELD_CAPACITY_KPA,
    wilting_point_kpa: float = WILTING_POINT_KPA,
) -> tuple[float, float, float]:
    """(field capacity, wilting point, plant-available water), all m3/m3."""
    fc = van_genuchten_theta(theta_r, theta_s, alpha_per_kpa, n, field_capacity_kpa)
    wp = van_genuchten_theta(theta_r, theta_s, alpha_per_kpa, n, wilting_point_kpa)
    return fc, wp, fc - wp


# ---------------------------------------------------------------------------
# Earth Engine plumbing
# ---------------------------------------------------------------------------


def _tier_of(source: SourceDescriptor) -> str:
    asset = source.asset_id
    if asset.startswith("ISDASOIL/"):
        return "isda"
    if "/polaris/" in asset:
        return "polaris"
    if asset.startswith("projects/soilgrids-isric/"):
        return "soilgrids"
    if "HiHydroSoilv2_0" in asset:
        return "hihydrosoil"
    raise KeyError(f"{asset!r} belongs to no soil tier")


def _tier_members() -> dict[str, tuple[SourceDescriptor, ...]]:
    """Every soil source, grouped by the batch it is read in."""
    grouped: dict[str, list[SourceDescriptor]] = {name: [] for name in _TIERS}
    quantities = SOIL_QUANTITIES + ("soil_available_water",)
    for quantity in quantities:
        for source in registry.sources_for(quantity):
            grouped[_tier_of(source)].append(source)
    return {name: tuple(members) for name, members in grouped.items()}


_TIER_MEMBERS = _tier_members()


def _band_image(ee: Any, source: SourceDescriptor) -> Any:
    """One EE image holding this source's band, renamed to its quantity.

    Renaming is what makes batching possible: nine iSDA assets all publish a band
    called ``mean_0_20`` and thirteen POLARIS collections all publish ``b1``.
    """
    tier = _tier_of(source)
    if tier == "polaris":
        variable = source.asset_id.rsplit("/", 1)[-1][: -len("_mean")]
        collection = ee.ImageCollection(source.asset_id)
        image = ee.Image(collection.filter(ee.Filter.eq("system:index", f"{variable}_{_POLARIS_DEPTH}")).first())
        return image.rename(source.quantity)
    if tier == "hihydrosoil":
        collection = ee.ImageCollection(source.asset_id)
        image = ee.Image(collection.filter(ee.Filter.eq("system:index", _HIHYDROSOIL_IMAGE)).first())
        return image.rename(source.quantity)
    return ee.Image(source.asset_id).select(source.band).rename(source.quantity)


def _read_tier(ee: Any, tier: str, lat: float, lon: float) -> tuple[Mapping[str, Any] | None, str | None]:
    """One batched reduceRegion for a whole tier. Returns (result, error)."""
    sources = _TIER_MEMBERS[tier]
    if not sources:
        return {}, None
    _, scale = _TIERS[tier]
    try:
        image = ee.Image.cat([_band_image(ee, source) for source in sources])
        point = ee.Geometry.Point([lon, lat])
        result = image.reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=scale)
        return result.getInfo(), None
    except Exception as exc:  # an EE failure is a gap, never a fabricated soil
        return None, f"{type(exc).__name__}: {exc}"


def _all_missing(reason: MissingReason, detail: str) -> dict[str, Fact | Missing]:
    """Every soil quantity as a gap, for a degraded run."""
    out: dict[str, Fact | Missing] = {}
    for quantity in SOIL_QUANTITIES + ("soil_available_water",):
        out[quantity] = Missing(quantity, reason, registry.chain_for(quantity), detail)
    for quantity in ("soil_texture_class", "soil_field_capacity", "soil_wilting_point", "soil_pawc"):
        out[quantity] = Missing(quantity, reason, (), detail)
    return out


def _inherited_reason(inputs: Sequence[Evidence | None], default: MissingReason) -> MissingReason:
    """A derived value is missing for the same reason its ingredients are.

    Without this, "POLARIS is masked at this Iowa pixel" would be reported as
    "POLARIS does not cover Iowa", which is a different and false statement.
    """
    for item in inputs:
        if isinstance(item, Missing):
            return item.reason
    return default


def _derive_texture_class(measured: Mapping[str, Evidence]) -> Fact | Missing:
    """iSDA's own class where it exists, else the USDA triangle over the fractions."""
    code = measured.get("soil_texture_code")
    if isinstance(code, Fact):
        name = ISDA_TEXTURE_CLASSES.get(int(round(float(code.value))))
        if name:
            return Fact.derive(
                "soil_texture_class",
                name,
                "",
                [code],
                scaling_applied=f"iSDA texture_class legend, code {int(round(float(code.value)))}",
            )

    fractions = [measured.get(q) for q in ("soil_clay_pct", "soil_sand_pct", "soil_silt_pct")]
    if all(isinstance(f, Fact) for f in fractions):
        clay, sand, silt = (float(f.value) for f in fractions)  # type: ignore[union-attr]
        name = usda_texture_class(clay, sand, silt)
        if name:
            return Fact.derive(
                "soil_texture_class",
                name,
                "",
                [f for f in fractions if isinstance(f, Fact)],
                scaling_applied=(
                    f"USDA texture triangle on clay {clay:.1f}%, sand {sand:.1f}%, silt {silt:.1f}% "
                    f"normalised to 100 (measured sum {clay + sand + silt:.1f}%)"
                ),
            )
        return Missing(
            "soil_texture_class",
            MissingReason.SOURCE_FAILED,
            (),
            f"clay {clay:.1f}%, sand {sand:.1f}%, silt {silt:.1f}% fall in no USDA texture class",
        )

    absent = [q for q, f in zip(("clay", "sand", "silt"), fractions) if not isinstance(f, Fact)]
    return Missing(
        "soil_texture_class",
        _inherited_reason(fractions + [code], MissingReason.MASKED),
        (),
        "no texture class was measured and " + ", ".join(absent) + " could not be measured either",
    )


def _derive_hydraulics(measured: Mapping[str, Evidence]) -> dict[str, Fact | Missing]:
    """Field capacity, wilting point and PAWC from the van Genuchten curve."""
    inputs = [measured.get(q) for q in _VG_QUANTITIES]
    names = ("soil_field_capacity", "soil_wilting_point", "soil_pawc")
    if not all(isinstance(f, Fact) for f in inputs):
        absent = [q for q, f in zip(_VG_QUANTITIES, inputs) if not isinstance(f, Fact)]
        gap = Missing(
            "soil_pawc",
            _inherited_reason(inputs, MissingReason.OUT_OF_COVERAGE),
            registry.chain_for("soil_theta_s"),
            "van Genuchten parameters unavailable here (" + ", ".join(absent) + ")",
        )
        return {name: replace(gap, quantity=name) for name in names}

    theta_r, theta_s, alpha, n = (float(f.value) for f in inputs)  # type: ignore[union-attr]
    facts = [f for f in inputs if isinstance(f, Fact)]
    try:
        fc, wp, pawc = plant_available_water(theta_r, theta_s, alpha, n)
    except ValueError as exc:
        gap = Missing("soil_pawc", MissingReason.SOURCE_FAILED, registry.chain_for("soil_theta_s"), str(exc))
        return {name: replace(gap, quantity=name) for name in names}

    curve = (
        f"van Genuchten theta(h) with theta_r={theta_r:.3f}, theta_s={theta_s:.3f}, "
        f"alpha={alpha:.3f}/kPa, n={n:.3f}"
    )
    return {
        "soil_field_capacity": Fact.derive(
            "soil_field_capacity", fc, "m3/m3", facts,
            scaling_applied=f"{curve} at {FIELD_CAPACITY_KPA:g} kPa", precision=3),
        "soil_wilting_point": Fact.derive(
            "soil_wilting_point", wp, "m3/m3", facts,
            scaling_applied=f"{curve} at {WILTING_POINT_KPA:g} kPa", precision=3),
        "soil_pawc": Fact.derive(
            "soil_pawc", pawc, "m3/m3", facts,
            scaling_applied=(
                f"{curve}, water held between {FIELD_CAPACITY_KPA:g} kPa and {WILTING_POINT_KPA:g} kPa"
            ),
            precision=3,
        ),
    }


def get_soil(
    lat: float,
    lon: float,
    *,
    ledger: Ledger | None = None,
    settings: Settings | None = None,
    use_cache: bool = True,
) -> dict[str, Fact | Missing]:
    """Soil properties at a point, each one a Fact or a named gap.

    Returns a dict keyed by quantity: ``soil_ph``, ``soil_clay_pct``,
    ``soil_sand_pct``, ``soil_silt_pct``, ``soil_soc``, ``soil_nitrogen``,
    ``soil_cec``, ``soil_bulk_density``, ``soil_texture_code``,
    ``soil_om_pct``, the POLARIS hydraulics (``soil_theta_s``,
    ``soil_theta_r``, ``soil_ksat``, ``soil_vg_alpha``, ``soil_vg_n``,
    ``soil_available_water``) and the derived ``soil_texture_class``,
    ``soil_field_capacity``, ``soil_wilting_point`` and ``soil_pawc``.

    Chains are walked lazily, one batched ``reduceRegion`` per tier, so an
    African point costs one round trip and a US point two.
    """
    settings = settings or get_settings()
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise ValueError(f"({lat}, {lon}) is not a valid latitude/longitude pair")

    key = registry.TTLCache.point_key("soil", lat, lon)
    if use_cache:
        cached = _cache.get(key)
        if cached is not None:
            # A cached answer is still this turn's evidence: it has to reach the
            # ledger, or the turn reports no provenance at all.
            if ledger is not None:
                ledger.extend(cached.values())
            return dict(cached)

    try:
        ee = bootstrap.require_ee()
    except EarthEngineUnavailable as exc:
        # A degraded run is still a run: SPEC 4.1 asks the turn to name everything
        # it failed to measure, and SPEC 10's null-adapter run is entirely this
        # path. Returning the gaps without ledgering them would leave that turn
        # reporting neither a soil nor a reason for not having one.
        gaps = _all_missing(MissingReason.SOURCE_FAILED, f"Earth Engine unavailable: {exc.reason}")
        if ledger is not None:
            ledger.extend(gaps.values())
        return gaps

    tier_results: dict[str, Mapping[str, Any] | None] = {}
    tier_errors: dict[str, str | None] = {}

    def read(tier: str) -> Mapping[str, Any] | None:
        if tier not in tier_results:
            result, error = _read_tier(ee, tier, lat, lon)
            tier_results[tier] = result
            tier_errors[tier] = error
        return tier_results[tier]

    def fetch(source: SourceDescriptor) -> Evidence:
        tier = _tier_of(source)
        result = read(tier)
        depth, _ = _TIERS[tier]
        evidence = fact_from_reduce_region(
            result,
            source,
            observed_on=None,  # every soil grid here is a static layer
            key=source.quantity,
            detail=tier_errors.get(tier),
        )
        if isinstance(evidence, Fact):
            # A class code is an integer label, not a measurement to two decimals.
            precision = 0 if source.unit == "class" else evidence.precision
            return replace(evidence, note=f"{depth} depth", precision=precision)
        return evidence

    measured: dict[str, Fact | Missing] = {}
    for quantity in SOIL_QUANTITIES:
        measured[quantity] = registry.resolve_chain(quantity, fetch, lat=lat, lon=lon)

    hydraulics = _derive_hydraulics(measured)
    if isinstance(hydraulics["soil_pawc"], Fact):
        # A site-specific retention curve beats a 250 m global grid, so the
        # extra round trip for HiHydroSoil is not worth making.
        measured["soil_available_water"] = Missing(
            "soil_available_water",
            MissingReason.NOT_REQUESTED,
            (),
            "POLARIS van Genuchten parameters gave a site-specific water capacity, "
            "so the 250 m HiHydroSoil layer was not read",
        )
    else:
        measured["soil_available_water"] = registry.resolve_chain(
            "soil_available_water", fetch, lat=lat, lon=lon
        )
        awc = measured["soil_available_water"]
        if isinstance(awc, Fact):
            hydraulics["soil_pawc"] = Fact.derive(
                "soil_pawc",
                awc.value,
                awc.unit,
                [awc],
                scaling_applied="HiHydroSoil v2.0 available water content (field capacity minus wilting point)",
                precision=3,
                note="no van Genuchten parameters here; the measured available water content is used directly",
            )

    measured["soil_texture_class"] = _derive_texture_class(measured)
    measured.update(hydraulics)

    ordered = {q: measured[q] for q in SOIL_QUANTITIES}
    for extra in ("soil_available_water", "soil_texture_class", "soil_field_capacity", "soil_wilting_point", "soil_pawc"):
        ordered[extra] = measured[extra]

    if use_cache:
        _cache.put(key, dict(ordered))
    if ledger is not None:
        ledger.extend(ordered.values())
    return ordered
