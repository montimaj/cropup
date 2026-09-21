"""Koppen-Geiger climate classification, computed from Earth Engine normals.

There is no Koppen asset in Earth Engine. The vendored backend sampled a local
GeoTIFF, swallowed the "file not found" and returned a fabricated ``Cfb`` /
"Temperate" for Arusha (SPEC 2.1, bug 1) -- a wrong answer that looks like a
right one, which is the exact failure this package exists to stop.

So the class is computed. Twelve monthly temperature normals and twelve monthly
precipitation normals are read from Earth Engine in **one** ``reduceRegion``,
and the Peel/Finlayson/McMahon (2007) decision rules are then applied to those
measured numbers. Every normal is a :class:`~cropup.evidence.Fact`, and the code
is a :meth:`Fact.derive` over all 24, so the letter on screen can be traced back
to the pixels that produced it. When the normals cannot be read the code is
:data:`UNKNOWN` and the quantity is recorded as a gap -- never a guess.

Sources (SPEC 3.1):

* ``WORLDCLIM/V1/MONTHLY`` -- 927 m, 1960-1990 baseline. Primary, because it is
  the only global monthly normal fine enough to resolve Arusha's 1401 m plateau
  or the floor of the Salinas Valley. ``tavg``/``tmin``/``tmax`` are scaled 0.1;
  ``prec`` is already mm.
* ``IDAHO_EPSCOR/TERRACLIMATE`` -- 4.6 km, from which a 1991-2020 baseline is
  built here with ``calendarRange`` means. Fallback. It ships no ``tavg`` band,
  so monthly mean temperature is ``(tmmn + tmmx) / 2`` before the 0.1 scaling.

A point takes one baseline or the other, never a mixture: if WorldClim is
incomplete at the pixel the whole month set falls through to TerraClimate, so
the twelve months a classification rests on always share a baseline.

Validated against the live-probed values in ``ee_registry.json``: Arusha Cwb,
Morogoro Aw, Salinas Csa at the task point (36.6, -121.1) and Csb at the city
(36.68, -121.66) -- the a/b split is the 22 degC warmest-month line, and the
task point's warmest month is 22.4 degC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    "get_climate",
    "classify_koppen",
    "KoppenResult",
    "KOPPEN_NAMES",
    "UNKNOWN",
    "CLIMATE_QUANTITIES",
    "clear_cache",
]

# What the classifier returns when it cannot classify. It is a sentinel string,
# never a climate: get_climate turns it into a Missing rather than a Fact.
UNKNOWN = "UNKNOWN"

KOPPEN_NAMES: dict[str, str] = {
    "Af": "Tropical rainforest",
    "Am": "Tropical monsoon",
    "Aw": "Tropical savanna",
    "BWh": "Hot desert",
    "BWk": "Cold desert",
    "BSh": "Hot semi-arid steppe",
    "BSk": "Cold semi-arid steppe",
    "Csa": "Mediterranean, hot summer",
    "Csb": "Mediterranean, warm summer",
    "Csc": "Mediterranean, cold summer",
    "Cwa": "Humid subtropical, dry winter",
    "Cwb": "Subtropical highland, dry winter",
    "Cwc": "Subtropical highland, short cool summer",
    "Cfa": "Humid subtropical",
    "Cfb": "Temperate oceanic",
    "Cfc": "Subpolar oceanic",
    "Dsa": "Continental, dry hot summer",
    "Dsb": "Continental, dry warm summer",
    "Dsc": "Continental, dry cool summer",
    "Dsd": "Continental, dry summer and very cold winter",
    "Dwa": "Continental, dry winter and hot summer",
    "Dwb": "Continental, dry winter and warm summer",
    "Dwc": "Subarctic, dry winter",
    "Dwd": "Subarctic, dry and very cold winter",
    "Dfa": "Humid continental, hot summer",
    "Dfb": "Humid continental, warm summer",
    "Dfc": "Subarctic",
    "Dfd": "Subarctic, very cold winter",
    "ET": "Tundra",
    "EF": "Ice cap",
}

MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

# Peel et al. define the summer half as the warmer of these two six-month
# blocks, which is what makes the rules hemisphere-agnostic.
_APR_SEP = (3, 4, 5, 6, 7, 8)
_OCT_MAR = (9, 10, 11, 0, 1, 2)

CLIMATE_QUANTITIES: tuple[str, ...] = tuple(
    [f"climate_tavg_{m:02d}" for m in range(1, 13)] + [f"climate_prec_{m:02d}" for m in range(1, 13)]
)

_DERIVED_QUANTITIES: tuple[str, ...] = (
    "koppen_code",
    "koppen_name",
    "annual_temp_c",
    "annual_precip_mm",
    "warmest_month_temp_c",
    "coldest_month_temp_c",
    "wettest_month_precip_mm",
    "driest_month_precip_mm",
)

_TIER_SCALE = {"worldclim": 927.66, "terraclimate": 4638.31}
_TERRACLIMATE_BASELINE = ("1991-01-01", "2021-01-01")

_cache = registry.TTLCache(get_settings().ee_cache_ttl_s, get_settings().ee_cache_max_entries)


def clear_cache() -> None:
    """Forget cached point results. For tests and for tools/."""
    _cache.clear()


# ---------------------------------------------------------------------------
# The classifier: pure arithmetic, no Earth Engine, unit-testable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KoppenResult:
    """A Koppen-Geiger verdict and the trail that produced it."""

    code: str
    name: str
    main_class: str = ""
    summer_months: tuple[int, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def known(self) -> bool:
        return self.code != UNKNOWN

    def explain(self) -> str:
        return "; ".join(self.reasons)


def _unknown(reason: str) -> KoppenResult:
    return KoppenResult(UNKNOWN, "Unknown", "", (), (reason,))


def _usable(values: Sequence[Any]) -> bool:
    return len(values) == 12 and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values
    )


def classify_koppen(monthly_temp_c: Sequence[float], monthly_precip_mm: Sequence[float]) -> KoppenResult:
    """Koppen-Geiger class from twelve monthly normals, January first.

    Implements Peel, Finlayson and McMahon (2007), Table 1, including the
    seasonal-precipitation sub-classes. Where a location satisfies both the
    dry-summer and the dry-winter criterion -- Arusha does, because its wettest
    month falls in the cooler half of the year -- the drier half of the year
    decides, which is how the reference implementations break the tie and what
    makes Arusha come out Cwb rather than Cs.

    Two boundaries follow Peel rather than Koppen's originals, and shift a few
    marginal places by one letter: the C/D split is at a coldest month of 0 degC
    (not -3), so Reykjavik comes out Dfc rather than Cfc; and a warmest month of
    exactly 10 degC is read as polar, a case Peel's table leaves undefined.

    Returns ``code == UNKNOWN`` if the twelve values are not all finite. It
    never guesses.
    """
    temps = list(monthly_temp_c)
    precs = list(monthly_precip_mm)
    if not _usable(temps):
        return _unknown("twelve finite monthly temperature normals are required")
    if not _usable(precs):
        return _unknown("twelve finite monthly precipitation normals are required")
    temps = [float(t) for t in temps]
    precs = [max(0.0, float(p)) for p in precs]

    mat = sum(temps) / 12.0
    map_mm = sum(precs)
    t_hot = max(temps)
    t_cold = min(temps)
    months_above_10 = sum(1 for t in temps if t >= 10.0)
    p_dry = min(precs)

    summer = _APR_SEP if sum(temps[i] for i in _APR_SEP) >= sum(temps[i] for i in _OCT_MAR) else _OCT_MAR
    winter = _OCT_MAR if summer is _APR_SEP else _APR_SEP
    p_summer = sum(precs[i] for i in summer)
    p_winter = sum(precs[i] for i in winter)
    p_summer_dry = min(precs[i] for i in summer)
    p_summer_wet = max(precs[i] for i in summer)
    p_winter_dry = min(precs[i] for i in winter)
    p_winter_wet = max(precs[i] for i in winter)

    reasons = [
        f"MAT {mat:.1f} degC, MAP {map_mm:.0f} mm, warmest month {t_hot:.1f}, coldest month {t_cold:.1f}",
        f"summer half = {MONTH_NAMES[summer[0]]}-{MONTH_NAMES[summer[-1]]} "
        f"({p_summer:.0f} mm vs {p_winter:.0f} mm in the winter half)",
    ]

    # -- B, the arid classes, are tested first: they override A, C and D.
    if map_mm > 0 and p_winter >= 0.7 * map_mm:
        p_threshold = 2.0 * mat
        threshold_why = "at least 70% of rain falls in the winter half"
    elif map_mm > 0 and p_summer >= 0.7 * map_mm:
        p_threshold = 2.0 * mat + 28.0
        threshold_why = "at least 70% of rain falls in the summer half"
    else:
        p_threshold = 2.0 * mat + 14.0
        threshold_why = "rain is spread across the year"
    reasons.append(f"aridity threshold {p_threshold:.0f} mm ({threshold_why})")

    if map_mm < 10.0 * p_threshold:
        second = "W" if map_mm < 5.0 * p_threshold else "S"
        third = "h" if mat >= 18.0 else "k"
        reasons.append(
            f"MAP {map_mm:.0f} < 10 x threshold, so arid; "
            f"{'desert' if second == 'W' else 'steppe'} and {'hot' if third == 'h' else 'cold'}"
        )
        return _result("B" + second + third, "B", summer, reasons)

    # -- A, tropical.
    if t_cold >= 18.0:
        if p_dry >= 60.0:
            reasons.append(f"coldest month {t_cold:.1f} >= 18 and driest month {p_dry:.0f} mm >= 60: rainforest")
            return _result("Af", "A", summer, reasons)
        monsoon_floor = 100.0 - map_mm / 25.0
        if p_dry >= monsoon_floor:
            reasons.append(f"driest month {p_dry:.0f} mm >= 100 - MAP/25 ({monsoon_floor:.0f}): monsoon")
            return _result("Am", "A", summer, reasons)
        # Peel et al. use a single savanna class, Aw, whichever half is dry; the
        # half is recorded in the trail rather than split into Aw/As, so the code
        # matches the reference implementations the registry was verified against.
        dry_half = "summer" if p_summer < p_winter else "winter"
        reasons.append(
            f"driest month {p_dry:.0f} mm < 100 - MAP/25 ({monsoon_floor:.0f}): savanna, dry {dry_half}"
        )
        return _result("Aw", "A", summer, reasons)

    # -- E, polar: no month warm enough for C or D.
    if t_hot <= 10.0:
        code = "ET" if t_hot > 0.0 else "EF"
        reasons.append(f"warmest month {t_hot:.1f} <= 10: {'tundra' if code == 'ET' else 'ice cap'}")
        return _result(code, "E", summer, reasons)

    main = "C" if t_cold > 0.0 else "D"
    reasons.append(
        f"warmest month above 10 and coldest month {t_cold:.1f} "
        f"{'above' if main == 'C' else 'at or below'} 0: {main}"
    )

    dry_summer = p_summer_dry < 40.0 and p_summer_dry < p_winter_wet / 3.0
    dry_winter = p_winter_dry < p_summer_wet / 10.0
    if dry_summer and dry_winter:
        # Both criteria fire where the wettest month sits in the cooler half.
        # The half that actually receives less rain decides.
        dry_summer = p_summer < p_winter
        dry_winter = not dry_summer
        reasons.append(
            "both the dry-summer and dry-winter criteria are met; the drier half of the year decides"
        )
    if dry_summer:
        second = "s"
        reasons.append(f"driest summer month {p_summer_dry:.0f} mm < 40 and < wettest winter month / 3: dry summer")
    elif dry_winter:
        second = "w"
        reasons.append(f"driest winter month {p_winter_dry:.0f} mm < wettest summer month / 10: dry winter")
    else:
        second = "f"
        reasons.append("no dry season")

    if t_hot >= 22.0:
        third = "a"
        reasons.append(f"warmest month {t_hot:.1f} >= 22: hot summer")
    elif months_above_10 >= 4:
        third = "b"
        reasons.append(f"{months_above_10} months at or above 10 degC: warm summer")
    elif main == "D" and t_cold <= -38.0:
        third = "d"
        reasons.append(f"coldest month {t_cold:.1f} <= -38: very cold winter")
    else:
        third = "c"
        reasons.append(f"only {months_above_10} months at or above 10 degC: cold summer")

    return _result(main + second + third, main, summer, reasons)


def _result(code: str, main: str, summer: Sequence[int], reasons: Sequence[str]) -> KoppenResult:
    return KoppenResult(
        code=code,
        name=KOPPEN_NAMES.get(code, "Unknown"),
        main_class=main,
        summer_months=tuple(i + 1 for i in summer),
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# Earth Engine plumbing
# ---------------------------------------------------------------------------


def _tier_of(source: SourceDescriptor) -> str:
    if source.asset_id == "WORLDCLIM/V1/MONTHLY":
        return "worldclim"
    if source.asset_id == "IDAHO_EPSCOR/TERRACLIMATE":
        return "terraclimate"
    raise KeyError(f"{source.asset_id!r} belongs to no climate tier")


def _tier_keys(tier: str) -> tuple[str, ...]:
    """The reduceRegion keys a complete read of this tier must contain."""
    keys: list[str] = []
    for quantity in CLIMATE_QUANTITIES:
        for source in registry.sources_for(quantity):
            if _tier_of(source) == tier:
                keys.append(source.key)
    return tuple(keys)


def _worldclim_image(ee: Any) -> Any:
    # toBands() on the 12-image collection yields '01_tavg' ... '12_prec',
    # which is exactly what the registry's result_keys expect.
    return ee.ImageCollection("WORLDCLIM/V1/MONTHLY").select(["tavg", "prec"]).toBands()


def _terraclimate_image(ee: Any) -> Any:
    """1991-2020 monthly normals, built here because TerraClimate ships none.

    TerraClimate has no ``tavg`` band, so monthly mean temperature is the mean of
    ``tmmn`` and ``tmmx`` -- still on the raw 0.1 scale, which the registry's
    transform applies afterwards.
    """
    start, end = _TERRACLIMATE_BASELINE
    base = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterDate(start, end)
    bands = []
    for month in range(1, 13):
        monthly = base.filter(ee.Filter.calendarRange(month, month, "month"))
        tmin = monthly.select("tmmn").mean()
        tmax = monthly.select("tmmx").mean()
        bands.append(tmin.add(tmax).divide(2).rename(f"tavg_{month:02d}"))
        bands.append(monthly.select("pr").mean().rename(f"pr_{month:02d}"))
    return ee.Image.cat(bands)


def _read_tier(ee: Any, tier: str, lat: float, lon: float) -> tuple[Mapping[str, Any] | None, str | None]:
    """One batched reduceRegion for a whole baseline. Returns (result, error)."""
    try:
        image = _worldclim_image(ee) if tier == "worldclim" else _terraclimate_image(ee)
        point = ee.Geometry.Point([lon, lat])
        result = image.reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=_TIER_SCALE[tier])
        return result.getInfo(), None
    except Exception as exc:  # a failed read is a gap, never a fabricated climate
        return None, f"{type(exc).__name__}: {exc}"


def _complete(result: Mapping[str, Any] | None, tier: str) -> bool:
    if not isinstance(result, Mapping):
        return False
    return all(result.get(key) is not None for key in _tier_keys(tier))


def _all_missing(reason: MissingReason, detail: str) -> dict[str, Fact | Missing]:
    out: dict[str, Fact | Missing] = {}
    for quantity in _DERIVED_QUANTITIES:
        out[quantity] = Missing(quantity, reason, (), detail)
    for quantity in CLIMATE_QUANTITIES:
        out[quantity] = Missing(quantity, reason, registry.chain_for(quantity), detail)
    return out


def _derive_summary(temps: Sequence[Fact], precs: Sequence[Fact]) -> dict[str, Fact]:
    """Annual and extreme-month figures, each carrying all twelve ingredients."""
    temp_values = [float(f.value) for f in temps]
    prec_values = [float(f.value) for f in precs]
    warmest = temp_values.index(max(temp_values))
    coldest = temp_values.index(min(temp_values))
    wettest = prec_values.index(max(prec_values))
    driest = prec_values.index(min(prec_values))
    return {
        "annual_temp_c": Fact.derive(
            "annual_temp_c", sum(temp_values) / 12.0, "degC", list(temps),
            scaling_applied="mean of the twelve monthly temperature normals", precision=1),
        "annual_precip_mm": Fact.derive(
            "annual_precip_mm", sum(prec_values), "mm", list(precs),
            scaling_applied="sum of the twelve monthly precipitation normals", precision=0),
        "warmest_month_temp_c": Fact.derive(
            "warmest_month_temp_c", temp_values[warmest], "degC", list(temps),
            scaling_applied="warmest of the twelve monthly normals", precision=1,
            note=f"warmest month is {MONTH_NAMES[warmest]}"),
        "coldest_month_temp_c": Fact.derive(
            "coldest_month_temp_c", temp_values[coldest], "degC", list(temps),
            scaling_applied="coldest of the twelve monthly normals", precision=1,
            note=f"coldest month is {MONTH_NAMES[coldest]}"),
        "wettest_month_precip_mm": Fact.derive(
            "wettest_month_precip_mm", prec_values[wettest], "mm", list(precs),
            scaling_applied="wettest of the twelve monthly normals", precision=0,
            note=f"wettest month is {MONTH_NAMES[wettest]}"),
        "driest_month_precip_mm": Fact.derive(
            "driest_month_precip_mm", prec_values[driest], "mm", list(precs),
            scaling_applied="driest of the twelve monthly normals", precision=0,
            note=f"driest month is {MONTH_NAMES[driest]}"),
    }


def get_climate(
    lat: float,
    lon: float,
    *,
    ledger: Ledger | None = None,
    settings: Settings | None = None,
    use_cache: bool = True,
) -> dict[str, Fact | Missing]:
    """Climate normals and the Koppen-Geiger class at a point.

    Returns a dict keyed by quantity: ``koppen_code``, ``koppen_name``,
    ``annual_temp_c``, ``annual_precip_mm``, the warmest/coldest/wettest/driest
    month figures, and the 24 monthly normals ``climate_tavg_01`` ...
    ``climate_prec_12``.

    ``koppen_code`` is a ``Missing`` when the normals could not be read. There is
    no default class: an unclassifiable point is reported as unclassified.
    """
    settings = settings or get_settings()
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise ValueError(f"({lat}, {lon}) is not a valid latitude/longitude pair")

    key = registry.TTLCache.point_key("climate", lat, lon)
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
        # reporting neither a climate class nor a reason for not having one.
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

    # A classification must rest on one baseline, so pick the first tier that
    # returns all 24 normals rather than letting the chain mix them month by month.
    chosen: str | None = None
    for tier in ("worldclim", "terraclimate"):
        if _complete(read(tier), tier):
            chosen = tier
            break

    def fetch(source: SourceDescriptor) -> Evidence:
        tier = _tier_of(source)
        if chosen is not None and tier != chosen:
            return Missing(
                source.quantity,
                MissingReason.MASKED,
                (source.asset_id,),
                f"the {tier} normals are incomplete at this point, so the {chosen} baseline was used throughout",
            )
        return fact_from_reduce_region(
            read(tier), source, observed_on=None, key=source.key, detail=tier_errors.get(tier)
        )

    measured: dict[str, Fact | Missing] = {}
    for quantity in CLIMATE_QUANTITIES:
        measured[quantity] = registry.resolve_chain(quantity, fetch, lat=lat, lon=lon)

    temps = [measured[f"climate_tavg_{m:02d}"] for m in range(1, 13)]
    precs = [measured[f"climate_prec_{m:02d}"] for m in range(1, 13)]
    out: dict[str, Fact | Missing] = {}

    if all(isinstance(f, Fact) for f in temps) and all(isinstance(f, Fact) for f in precs):
        temp_facts = [f for f in temps if isinstance(f, Fact)]
        prec_facts = [f for f in precs if isinstance(f, Fact)]
        verdict = classify_koppen([float(f.value) for f in temp_facts], [float(f.value) for f in prec_facts])
        out.update(_derive_summary(temp_facts, prec_facts))
        if verdict.known:
            code_fact = Fact.derive(
                "koppen_code", verdict.code, "", temp_facts + prec_facts,
                scaling_applied="Koppen-Geiger (Peel et al. 2007) over the monthly normals: " + verdict.explain(),
                note=f"baseline: {chosen}",
            )
            out["koppen_code"] = code_fact
            out["koppen_name"] = Fact.derive(
                "koppen_name", verdict.name, "", [code_fact],
                scaling_applied=f"Koppen-Geiger name for code {verdict.code}",
            )
        else:
            gap = Missing(
                "koppen_code", MissingReason.SOURCE_FAILED, registry.chain_for("climate_tavg_01"),
                f"the normals could not be classified: {verdict.explain()}",
            )
            out["koppen_code"] = gap
            out["koppen_name"] = Missing("koppen_name", MissingReason.SOURCE_FAILED, (), gap.detail)
    else:
        absent = [q for q in CLIMATE_QUANTITIES if not isinstance(measured[q], Fact)]
        detail = f"{len(absent)} of the 24 monthly normals could not be read here"
        for quantity in _DERIVED_QUANTITIES:
            out[quantity] = Missing(
                quantity, MissingReason.MASKED, registry.chain_for("climate_tavg_01"), detail
            )

    ordered: dict[str, Fact | Missing] = {q: out[q] for q in _DERIVED_QUANTITIES}
    ordered.update({q: measured[q] for q in CLIMATE_QUANTITIES})

    if use_cache:
        _cache.put(key, dict(ordered))
    if ledger is not None:
        ledger.extend(ordered.values())
    return ordered
