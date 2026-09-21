"""Crop selection: what CropSuite rates highest at this point, what holds the
named crop back, and when the climatology says to sow it.

Composed from the ``geo/`` leaves, not from the vendored
``analyze_agronomic_recommendations`` (SPEC section 2.1). The differences that
matter:

* The vendored suitability came from ``maxent.py``, whose ``_get_maxent_conn()``
  has no ``return`` statement. Replaced by CropSuite.
* Its Köppen class came from a raster read that swallowed its own missing-file
  error and returned ``Cfb``/"Temperate" for Arusha, which is wrong. Replaced by
  ``geo/climate.py``, which computes the class from the monthly normals and
  reports ``UNKNOWN`` when it cannot.
* Its ``get_climate_matched_crops`` interpolated pH 7.0 and sand 30% into the
  prose when the soil call failed. Here a missing soil property blocks the line
  that needed it and is named.
* **CropSuite is Africa-only** (SPEC 3.3). Outside Africa this analysis returns
  no ranking at all and says why. It does not fall back to a different model and
  present the answer as if it were the same thing.
* At the exact Arusha pixel CropSuite is masked for 47 of 48 crops, so
  ``geo/suitability.py`` widens to a neighbourhood mean and labels the leg it
  used. Those labels travel through to the findings here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..config import Settings, get_settings
from ..evidence import Fact, Ledger, MissingReason
from ..geo import climate as geo_climate
from ..geo import context as geo_context
from ..geo import soil as geo_soil
from ..geo import suitability as geo_suitability
from .rules import (
    CLIMATE_DERIVED_QUANTITIES,
    SOIL_DERIVED_QUANTITIES,
    Finding,
    Leg,
    absorb,
    evidence_for,
    gather,
    timed,
)

__all__ = [
    "CropSelectionResult",
    "CropRanking",
    "analyze_crop_selection",
    "DEFAULT_RANK_COUNT",
]

DEFAULT_RANK_COUNT = 10

_CONTEXT_QUANTITIES: tuple[str, ...] = (
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

_LEG_ASSETS: dict[str, tuple[str, ...]] = {
    "suitability": (geo_suitability.CROP_SUITABILITY, geo_suitability.CLIMATE_SUITABILITY),
    "climate": ("WORLDCLIM/V1/MONTHLY", "IDAHO_EPSCOR/TERRACLIMATE"),
    "soil": ("ISDASOIL/Africa/v1", "projects/sat-io/open-datasets/polaris", "projects/soilgrids-isric"),
    "context": (geo_context.DEAF_CROPLAND_PROB, geo_context.GFSAD_GCEP30, geo_context.ESA_WORLDCOVER),
}


@dataclass(frozen=True)
class CropRanking:
    """One crop CropSuite scored here, with the leg the score came from."""

    crop: str
    fact: Fact

    @property
    def score(self) -> float:
        return float(self.fact.value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "crop": self.crop,
            "rendered": self.fact.render(),
            "score": self.score,
            "chain_label": self.fact.chain_label,
            "note": self.fact.note,
            "provenance": self.fact.provenance(),
        }


@dataclass(frozen=True)
class CropSelectionResult:
    """The ranking, the named crop's own reading, and the evidence behind both."""

    lat: float
    lon: float
    crop: str | None
    ledger: Ledger
    findings: tuple[Finding, ...]
    ranking: tuple[CropRanking, ...]
    scenario: str
    in_coverage: bool
    legs: tuple[Leg, ...]
    elapsed_s: float

    def finding(self, key: str) -> Finding | None:
        for item in self.findings:
            if item.key == key:
                return item
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "analysis": "crop_selection",
            "lat": self.lat,
            "lon": self.lon,
            "crop": self.crop,
            "scenario": self.scenario,
            "in_coverage": self.in_coverage,
            "ranking": [r.as_dict() for r in self.ranking],
            "findings": [f.as_dict() for f in self.findings],
            "legs": [leg.as_dict() for leg in self.legs],
            "elapsed_s": round(self.elapsed_s, 2),
            "ledger": self.ledger.as_dict(),
        }


# ---------------------------------------------------------------------------
# The CropSuite leg
# ---------------------------------------------------------------------------


def _crop_name(fact: Fact) -> str:
    """The crop a ranking Fact is about, from its band or its quantity name."""
    if fact.band:
        return geo_suitability.crop_for_band(fact.band)
    return fact.quantity.replace("crop_suitability_", "").replace("_", " ")


def _suitability_leg(
    lat: float,
    lon: float,
    crop: str | None,
    *,
    n: int,
    scenario: str,
) -> tuple[Ledger, list[CropRanking]]:
    """Rank the crops, then read the named crop's suitability, limiting factor
    and sowing date. All four calls write to one sub-ledger.

    ``rank_crops`` and the three per-crop reads are sequential on purpose: they
    share ``geo/suitability.py``'s per-point cache, so running them in parallel
    would issue the same ``getInfo`` several times over.
    """
    sub = Ledger(turn="suitability")
    ranked = geo_suitability.rank_crops(lat, lon, n, scenario=scenario, ledger=sub)
    rankings = [CropRanking(_crop_name(f), f) for f in ranked]

    if crop is not None:
        geo_suitability.get_crop_suitability(lat, lon, crop, scenario=scenario, ledger=sub)
        geo_suitability.get_limiting_factor(lat, lon, crop, scenario=scenario, ledger=sub)
        geo_suitability.get_optimal_sowing(lat, lon, crop, scenario=scenario, ledger=sub)
    else:
        for quantity in ("crop_suitability", "crop_limiting_factor", "optimal_sowing_date"):
            sub.gap(
                quantity,
                MissingReason.NOT_REQUESTED,
                (geo_suitability.CROP_SUITABILITY,),
                "no crop was named, so there is nothing to score against",
            )
    return sub, rankings


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


def _ranking_reading(rankings: list[CropRanking]) -> str:
    if not rankings:
        return ""
    best = rankings[0]
    names = ", ".join(f"{r.crop} {r.fact.render()}" for r in rankings[:5])
    text = f"CropSuite rates {names} highest here."
    if best.fact.chain_position > 0:
        text += (
            f" Every score is a {best.fact.chain_label} reading: {best.fact.note or 'the exact pixel is masked'}."
        )
    return text


def _findings(
    ledger: Ledger,
    lat: float,
    lon: float,
    crop: str | None,
    rankings: list[CropRanking],
) -> tuple[Finding, ...]:
    out: list[Finding] = []
    chain = (geo_suitability.CROP_SUITABILITY, geo_suitability.CLIMATE_SUITABILITY)

    if rankings:
        best = rankings[0]
        out.append(
            Finding(
                "best_crops",
                "Best-rated crops here",
                best.fact,
                _ranking_reading(rankings),
                "info",
                tuple(r.fact for r in rankings[1:5]),
            )
        )
    else:
        out.append(
            Finding(
                "best_crops",
                "Best-rated crops here",
                evidence_for(
                    ledger,
                    "crop_suitability",
                    chain,
                    "CropSuite models Africa only; there is no ranking for this point",
                ),
                (
                    "No crop ranking exists for this location. CropSuite, the only crop-suitability "
                    "model in the catalog, covers Africa only, and nothing else in the catalog ranks "
                    "crops against a site."
                    if not geo_suitability.in_coverage(lat, lon)
                    else "CropSuite covers this point but returned no unmasked score on any leg."
                ),
                "watch",
            )
        )

    current = evidence_for(
        ledger,
        "crop_suitability",
        chain,
        "no crop was named" if crop is None else None,
    )
    reading = ""
    importance = "info"
    if isinstance(current, Fact):
        reading = f"CropSuite rates {crop} at {current.render()} here"
        if current.chain_position > 0:
            reading += f" ({current.chain_label})"
        reading += "."
        if current.note:
            reading += f" {current.note}."
        if float(current.value) < 50.0:
            importance = "watch"
    out.append(Finding("current_crop_suitability", f"Suitability of {crop or 'the named crop'}", current, reading, importance))

    limiting = evidence_for(ledger, "crop_limiting_factor", (geo_suitability.CROP_LIMITING_FACTOR,))
    reading = ""
    if isinstance(limiting, Fact):
        reading = f"What holds {crop} back here is {limiting.render()}."
        if limiting.note:
            reading += f" {limiting.note}."
    out.append(Finding("limiting_factor", "Limiting factor", limiting, reading))

    sowing = evidence_for(ledger, "optimal_sowing_date", (geo_suitability.OPTIMAL_SOWING_DATE,))
    reading = ""
    if isinstance(sowing, Fact):
        window = ledger.fact("sowing_window_days")
        reading = f"The climatologically best sowing date for {crop} here is {sowing.render()}"
        if window is not None:
            reading += f", inside a window of {window.render()}"
        reading += ". It is a 1991-2010 climatology, not a forecast for this season."
    out.append(
        Finding(
            "sowing_date",
            "Optimal sowing date",
            sowing,
            reading,
            "info",
            tuple(e for e in (ledger.get("sowing_window_days"),) if e),
        )
    )

    koppen = evidence_for(ledger, "koppen_code", _LEG_ASSETS["climate"])
    reading = ""
    if isinstance(koppen, Fact):
        name = ledger.fact("koppen_name")
        temp = ledger.fact("annual_temp_c")
        precip = ledger.fact("annual_precip_mm")
        reading = f"The climate class is {koppen.render()}"
        if name is not None:
            reading += f", {name.render()}"
        if temp is not None and precip is not None:
            reading += f": {temp.render()} mean annual temperature and {precip.render()} of rain a year"
        reading += "."
    out.append(
        Finding(
            "climate_class",
            "Climate class",
            koppen,
            reading,
            "info",
            tuple(
                e
                for e in (
                    ledger.get("koppen_name"),
                    ledger.get("annual_temp_c"),
                    ledger.get("annual_precip_mm"),
                    ledger.get("driest_month_precip_mm"),
                )
                if e
            ),
        )
    )

    ph = evidence_for(ledger, "soil_ph")
    reading = ""
    if isinstance(ph, Fact):
        value = float(ph.value)
        where = "acid" if value < 5.5 else "alkaline" if value > 7.5 else "in the range most crops tolerate"
        reading = f"Topsoil pH {ph.render()} is {where}."
    out.append(
        Finding(
            "soil_reaction",
            "Soil pH",
            ph,
            reading,
            "info",
            tuple(e for e in (ledger.get("soil_texture_class"), ledger.get("soil_soc")) if e),
        )
    )

    cropland = evidence_for(ledger, "is_cropland", _LEG_ASSETS["context"])
    reading = ""
    importance = "info"
    if isinstance(cropland, Fact):
        cover = ledger.fact("land_cover")
        reading = "This point is mapped as cropland" if cropland.value else "This point is not mapped as cropland"
        if cover is not None:
            reading += f"; the land-cover map calls it {cover.render()}"
        reading += "."
        if cropland.value is False:
            importance = "watch"
    out.append(Finding("is_cropland", "Is this cropland", cropland, reading, importance))

    return tuple(out)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze_crop_selection(
    lat: float,
    lon: float,
    crop: str | None = None,
    *,
    n: int = DEFAULT_RANK_COUNT,
    scenario: str = geo_suitability.DEFAULT_SCENARIO,
    settings: Settings | None = None,
    ledger: Ledger | None = None,
) -> CropSelectionResult:
    """What to grow here, scored, with the coverage cliff stated when it bites.

    Runs the CropSuite, climate, soil and context legs concurrently. Outside
    Africa the ranking is empty and every suitability quantity is a named gap:
    there is no second model to fall back to, and presenting one would answer a
    different question than the farmer asked.
    """
    settings = settings or get_settings()
    started = time.perf_counter()
    ledger = ledger if ledger is not None else Ledger(turn="crop_selection")
    rankings: list[CropRanking] = []

    def suitability() -> Ledger:
        nonlocal rankings
        sub, rankings = _suitability_leg(lat, lon, crop, n=n, scenario=scenario)
        return sub

    jobs = {
        "suitability": timed("suitability", suitability),
        "climate": timed("climate", lambda: geo_climate.get_climate(lat, lon, settings=settings)),
        "soil": timed("soil", lambda: geo_soil.get_soil(lat, lon, settings=settings)),
        "context": timed("context", lambda: geo_context.get_field_context(lat, lon)),
    }

    expected = {
        "suitability": (
            "crop_suitability",
            "crop_limiting_factor",
            "crop_limiting_factor_code",
            "optimal_sowing_date",
            "optimal_sowing_doy",
            "sowing_window_days",
        ),
        "climate": geo_climate.CLIMATE_QUANTITIES + CLIMATE_DERIVED_QUANTITIES,
        "soil": geo_soil.SOIL_QUANTITIES + SOIL_DERIVED_QUANTITIES,
        "context": _CONTEXT_QUANTITIES,
    }

    results = gather(jobs, max_workers=settings.ee_max_workers)

    legs: list[Leg] = []
    for name in jobs:
        outcome = results.get(name)
        if isinstance(outcome, BaseException):
            absorb(ledger, legs, name, 0.0, outcome, expected[name], _LEG_ASSETS[name])
            continue
        _, elapsed, payload = outcome
        absorb(ledger, legs, name, elapsed, payload, expected[name], _LEG_ASSETS[name])
    legs.sort(key=lambda leg: leg.name)

    return CropSelectionResult(
        lat=float(lat),
        lon=float(lon),
        crop=crop,
        ledger=ledger,
        findings=_findings(ledger, lat, lon, crop, rankings),
        ranking=tuple(rankings),
        scenario=scenario,
        in_coverage=geo_suitability.in_coverage(lat, lon),
        legs=tuple(legs),
        elapsed_s=time.perf_counter() - started,
    )
