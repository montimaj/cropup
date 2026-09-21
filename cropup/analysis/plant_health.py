"""Field health: what the canopy, the surface temperature, the water balance and
the soil say about one field right now.

Composed from the ``geo/`` leaves, not from the vendored ``analyze_plant_health``
(SPEC section 2.1). The differences that matter:

* The vendored orchestrator returned ``{"error": ...}`` and nothing else when no
  cloud-free scene was found, so a thermal or soil answer that *was* available
  was thrown away with it. Here every leg runs, every leg reports, and a dead
  leg is a set of named gaps on the ledger.
* It interpolated soil defaults (pH 7.0, sand 30%) into prose. Here a value with
  no Fact behind it cannot be rendered at all.
* Its risk list came out of a rules engine that read a missing measurement as a
  passing one; see :mod:`cropup.analysis.rules`.
* Its "satellite date" was the date of the newest scene in a 12-month median.
  ``geo/vegetation.py`` dates the composite by its median scene and says how
  many scenes went in.

Latency: the six legs run concurrently, each one batching its own
``reduceRegion`` calls, against the vendored 28.7 s.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from typing import Any

from ..config import Settings, get_settings
from ..evidence import Fact, Ledger, MissingReason
from ..geo import context as geo_context
from ..geo import soil as geo_soil
from ..geo import thermal as geo_thermal
from ..geo import vegetation as geo_vegetation
from ..geo import water as geo_water
from .rules import (
    SOIL_DERIVED_QUANTITIES,
    Finding,
    Leg,
    RiskAssessment,
    absorb,
    derive_drainage,
    evidence_for,
    gather,
    run_rules,
    timed,
)

__all__ = [
    "PlantHealthResult",
    "analyze_plant_health",
    "CANOPY_BANDS",
    "canopy_status",
]

# The NDVI bands the zone percentages are cut at in geo/vegetation.py, named.
# They are an interpretation, so they live in analysis/ and are stated on screen
# alongside the reading rather than applied silently.
CANOPY_BANDS: tuple[tuple[str, float | None, float | None], ...] = (
    ("vigorous", 0.6, None),
    ("fair", 0.4, 0.6),
    ("stressed", 0.2, 0.4),
    ("severe", None, 0.2),
)

# Vegetation indices this analysis reads. PSRI, ReCI and GCI are wanted by the
# disease library, so they are fetched even though nothing else displays them.
_INDICES = ("ndvi", "ndmi", "ndre", "ndwi", "psri", "reci", "gci")

# Everything geo/context.py returns, so a failed context leg leaves nine named
# gaps rather than four.
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
    "vegetation": (geo_vegetation.S2_ASSET,),
    "thermal": geo_thermal.LANDSAT_COLLECTIONS,
    "soil": ("ISDASOIL/Africa/v1", "projects/sat-io/open-datasets/polaris", "projects/soilgrids-isric"),
    "water": (geo_water.SSEBOP_DEKADAL, geo_water.ESI_4WK, geo_water.SMAP_L4, geo_water.CHIRPS_DAILY),
    "context": (geo_context.DEAF_CROPLAND_PROB, geo_context.GFSAD_GCEP30, geo_context.ESA_WORLDCOVER),
}


def canopy_status(ndvi: float) -> str:
    """The band an NDVI reading falls in: vigorous / fair / stressed / severe."""
    for name, low, high in CANOPY_BANDS:
        if (low is None or ndvi >= low) and (high is None or ndvi < high):
            return name
    return "severe"


@dataclass(frozen=True)
class PlantHealthResult:
    """The turn's evidence, its findings and its risk list, in one object."""

    lat: float
    lon: float
    crop: str | None
    ledger: Ledger
    findings: tuple[Finding, ...]
    risks: RiskAssessment
    legs: tuple[Leg, ...]
    elapsed_s: float

    def finding(self, key: str) -> Finding | None:
        for item in self.findings:
            if item.key == key:
                return item
        return None

    def measured(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.measured)

    def as_dict(self) -> dict[str, Any]:
        return {
            "analysis": "plant_health",
            "lat": self.lat,
            "lon": self.lon,
            "crop": self.crop,
            "findings": [f.as_dict() for f in self.findings],
            "risks": self.risks.as_dict(),
            "legs": [leg.as_dict() for leg in self.legs],
            "elapsed_s": round(self.elapsed_s, 2),
            "ledger": self.ledger.as_dict(),
        }


def _derive_canopy(ledger: Ledger) -> None:
    """Canopy status, stressed area share and the year-on-year direction."""
    ndvi = ledger.fact("ndvi")
    if ndvi is not None:
        status = canopy_status(float(ndvi.value))
        ledger.record(
            Fact.derive(
                "canopy_status",
                status,
                "",
                [ndvi],
                scaling_applied="NDVI cut at 0.6 / 0.4 / 0.2 into vigorous / fair / stressed / severe",
                note="a canopy greenness band, not a diagnosis: a bare, a young and a dying canopy can share it",
            )
        )

    stressed = ledger.fact("ndvi_zone_stressed_pct")
    severe = ledger.fact("ndvi_zone_severe_pct")
    if stressed is not None and severe is not None:
        ledger.record(
            Fact.derive(
                "stressed_area_pct",
                float(stressed.value) + float(severe.value),
                "%",
                [stressed, severe],
                scaling_applied="share of the field below NDVI 0.4: the stressed and severe zones added",
            )
        )
    else:
        ledger.gap(
            "stressed_area_pct",
            MissingReason.MASKED,
            (geo_vegetation.S2_ASSET,),
            "needs both the stressed and the severe NDVI zone shares",
        )

    change = ledger.fact("ndvi_change_1y")
    if change is not None:
        value = float(change.value)
        direction = "greener" if value > 0.02 else "browner" if value < -0.02 else "unchanged"
        ledger.record(
            Fact.derive(
                "canopy_trend_1y",
                direction,
                "",
                [change],
                scaling_applied="NDVI change against the same window a year ago, called flat inside +/- 0.02",
                note="a year-on-year comparison of two medians, not a within-season trend",
            )
        )


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


def _band_text(value: float) -> str:
    for name, low, high in CANOPY_BANDS:
        if (low is None or value >= low) and (high is None or value < high):
            if low is None:
                return f"below {high:g}, which this build calls {name}"
            if high is None:
                return f"at or above {low:g}, which this build calls {name}"
            return f"in the {low:g} to {high:g} band, which this build calls {name}"
    return ""


def _findings(ledger: Ledger) -> tuple[Finding, ...]:
    out: list[Finding] = []

    ndvi = evidence_for(ledger, "ndvi", (geo_vegetation.S2_ASSET,))
    reading = ""
    if isinstance(ndvi, Fact):
        reading = f"NDVI {ndvi.render()} is {_band_text(float(ndvi.value))}."
        spread = ledger.fact("ndvi_spread")
        if spread is not None:
            reading += f" Spread across the field (p90 - p10) is {spread.render()}."
    out.append(
        Finding(
            "canopy_vigour",
            "Canopy vigour",
            ndvi,
            reading,
            "act" if isinstance(ndvi, Fact) and float(ndvi.value) < 0.4 else "info",
            tuple(e for e in (ledger.get("ndvi_p10"), ledger.get("ndvi_p90"), ledger.get("s2_scene_count")) if e),
        )
    )

    stressed = evidence_for(ledger, "stressed_area_pct", (geo_vegetation.S2_ASSET,))
    reading = ""
    if isinstance(stressed, Fact):
        reading = f"{stressed.render()} of the field reads below NDVI 0.4."
    out.append(
        Finding(
            "stressed_area",
            "Stressed area",
            stressed,
            reading,
            "act" if isinstance(stressed, Fact) and float(stressed.value) > 25.0 else "info",
            tuple(
                e
                for e in (
                    ledger.get("ndvi_zone_healthy_pct"),
                    ledger.get("ndvi_zone_fair_pct"),
                    ledger.get("ndvi_zone_stressed_pct"),
                    ledger.get("ndvi_zone_severe_pct"),
                )
                if e
            ),
        )
    )

    trend = evidence_for(ledger, "canopy_trend_1y", (geo_vegetation.S2_ASSET,), "the year-ago comparison was not requested")
    change = ledger.get("ndvi_change_1y")
    reading = ""
    if isinstance(trend, Fact) and isinstance(change, Fact):
        moved = "unchanged from" if trend.value == "unchanged" else f"{trend.value} than"
        reading = f"The canopy is {moved} a year ago: NDVI moved {change.render()}."
    out.append(
        Finding(
            "canopy_trend",
            "Year-on-year change",
            trend,
            reading,
            "watch" if isinstance(change, Fact) and float(change.value) < -0.05 else "info",
            (change,) if change else (),
        )
    )

    moisture = evidence_for(ledger, "ndmi", (geo_vegetation.S2_ASSET,))
    reading = ""
    if isinstance(moisture, Fact):
        reading = f"Canopy moisture index {moisture.render()}; negative values mean a dry canopy."
    out.append(
        Finding(
            "canopy_moisture",
            "Canopy moisture",
            moisture,
            reading,
            "act" if isinstance(moisture, Fact) and float(moisture.value) < -0.1 else "info",
        )
    )

    senescence = evidence_for(ledger, "psri", (geo_vegetation.S2_ASSET,))
    reading = ""
    if isinstance(senescence, Fact):
        reading = f"Senescence index {senescence.render()}; above 0.1 means chlorophyll is breaking down."
    out.append(Finding("senescence", "Canopy senescence", senescence, reading))

    lst = evidence_for(ledger, "land_surface_temperature", geo_thermal.LANDSAT_COLLECTIONS)
    reading = ""
    importance = "info"
    if isinstance(lst, Fact):
        clear = ledger.fact("land_surface_temperature_clear_fraction")
        reading = f"Surface temperature {lst.render()} on the last usable Landsat scene"
        if clear is not None:
            reading += f", {clear.render()} of the field cloud-free"
        reading += "."
        note = lst.staleness_note()
        if note:
            reading += f" It was {note}."
        if float(lst.value) > 38.0:
            importance = "act"
    out.append(
        Finding(
            "surface_temperature",
            "Land surface temperature",
            lst,
            reading,
            importance,
            tuple(
                e
                for e in (
                    ledger.get("land_surface_temperature_uncertainty_k"),
                    ledger.get("land_surface_temperature_clear_fraction"),
                )
                if e
            ),
        )
    )

    esi = evidence_for(ledger, "evaporative_stress_index", (geo_water.ESI_4WK,))
    reading = ""
    importance = "info"
    if isinstance(esi, Fact):
        reading = f"Evaporative stress index {esi.render()}."
        note = esi.staleness_note()
        if note:
            reading += f" This observation was {note}, so it describes an earlier month, not today."
        elif float(esi.value) < 0.5:
            importance = "watch"
    out.append(Finding("evaporative_stress", "Evaporative stress", esi, reading, importance))

    rain = evidence_for(ledger, "precipitation_30d", (geo_water.CHIRPS_DAILY,))
    reading = ""
    if isinstance(rain, Fact):
        week = ledger.fact("precipitation_7d")
        reading = f"{rain.render()} of rain in the 30 days to the last CHIRPS day"
        if week is not None:
            reading += f", {week.render()} of it in the last 7"
        reading += "."
    out.append(Finding("rainfall", "Recent rainfall", rain, reading))

    root = evidence_for(ledger, "soil_moisture_rootzone", (geo_water.SMAP_L4, geo_water.FLDAS_MONTHLY))
    reading = ""
    if isinstance(root, Fact):
        reading = f"Root-zone soil moisture {root.render()} over an 11 km SMAP cell, not this field alone."
    out.append(Finding("root_zone_moisture", "Root-zone soil moisture", root, reading))

    ph = evidence_for(ledger, "soil_ph")
    reading = ""
    importance = "info"
    if isinstance(ph, Fact):
        value = float(ph.value)
        where = "acid" if value < 5.5 else "alkaline" if value > 7.5 else "in the range most crops tolerate"
        reading = f"Topsoil pH {ph.render()} is {where}."
        if value < 5.5 or value > 7.5:
            importance = "watch"
    out.append(Finding("soil_reaction", "Soil pH", ph, reading, importance))

    texture = evidence_for(ledger, "soil_texture_class")
    reading = ""
    if isinstance(texture, Fact):
        drainage = ledger.fact("soil_drainage")
        reading = f"Texture reads as {texture.render()}"
        if drainage is not None:
            reading += f", which drains {str(drainage.value).lower()}"
        reading += "."
    out.append(
        Finding(
            "soil_texture",
            "Soil texture",
            texture,
            reading,
            "info",
            tuple(
                e
                for e in (
                    ledger.get("soil_clay_pct"),
                    ledger.get("soil_sand_pct"),
                    ledger.get("soil_silt_pct"),
                    ledger.get("soil_drainage"),
                )
                if e
            ),
        )
    )

    carbon = evidence_for(ledger, "soil_soc")
    reading = ""
    if isinstance(carbon, Fact):
        reading = f"Soil organic carbon {carbon.render()}."
    out.append(Finding("soil_carbon", "Soil organic carbon", carbon, reading))

    cropland = evidence_for(ledger, "is_cropland", _LEG_ASSETS["context"])
    reading = ""
    importance = "info"
    if isinstance(cropland, Fact):
        cover = ledger.fact("land_cover")
        if cropland.value is False:
            reading = "This point is not mapped as cropland"
            importance = "watch"
        else:
            reading = "This point is mapped as cropland"
        if cover is not None:
            reading += f"; the land-cover map calls it {cover.render()}"
        reading += ". Everything above describes whatever is growing there."
    out.append(Finding("is_cropland", "Is this cropland", cropland, reading, importance))

    return tuple(out)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze_plant_health(
    lat: float,
    lon: float,
    crop: str | None = None,
    *,
    radius_m: float | None = None,
    end_date: dt.date | str | None = None,
    window_days: int = geo_vegetation.DEFAULT_WINDOW_DAYS,
    settings: Settings | None = None,
    ledger: Ledger | None = None,
) -> PlantHealthResult:
    """How this field is doing, with the instrument behind every number.

    Runs the vegetation, thermal, soil, water and context legs concurrently,
    folds them into one :class:`~cropup.evidence.Ledger`, derives the canopy
    status and the drainage class the disease library needs, and evaluates the
    library against what was actually measured.

    ``crop=None`` is allowed: the crop-specific half of the library is then
    skipped and the count of skipped rules is reported. Nothing is defaulted.
    """
    settings = settings or get_settings()
    started = time.perf_counter()
    ledger = ledger if ledger is not None else Ledger(turn="plant_health")
    jobs = {
        "vegetation": timed(
            "vegetation",
            lambda: geo_vegetation.get_vegetation(
                lat,
                lon,
                radius_m=radius_m,
                end_date=end_date,
                window_days=window_days,
                indices=_INDICES,
                zones=True,
                compare_year_ago=True,
                settings=settings,
            ),
        ),
        "thermal": timed(
            "thermal",
            lambda: geo_thermal.get_land_surface_temperature(
                lat, lon, radius_m=radius_m, end_date=end_date, settings=settings
            ),
        ),
        "soil": timed("soil", lambda: geo_soil.get_soil(lat, lon, settings=settings)),
        # irrigation_mapping=False: geo/context.py publishes irrigation_regime
        # from the same LGRIP30 asset a different way, and two verdicts under
        # one quantity name in one ledger is worse than either alone.
        "water": timed(
            "water",
            lambda: geo_water.get_water_status(
                lat, lon, end_date=end_date, irrigation_mapping=False, settings=settings
            ),
        ),
        "context": timed("context", lambda: geo_context.get_field_context(lat, lon)),
    }

    results = gather(jobs, max_workers=settings.ee_max_workers)

    expected = {
        "vegetation": geo_vegetation.vegetation_quantities(_INDICES, zones=True, compare_year_ago=True),
        "thermal": geo_thermal.thermal_quantities(),
        "soil": geo_soil.SOIL_QUANTITIES + SOIL_DERIVED_QUANTITIES,
        "water": geo_water.water_quantities(irrigation_mapping=False),
        "context": _CONTEXT_QUANTITIES,
    }

    legs: list[Leg] = []
    for name in jobs:
        outcome = results.get(name)
        if isinstance(outcome, BaseException):  # gather() itself failed, not the leg
            absorb(ledger, legs, name, 0.0, outcome, expected[name], _LEG_ASSETS[name])
            continue
        _, elapsed, payload = outcome
        absorb(ledger, legs, name, elapsed, payload, expected[name], _LEG_ASSETS[name])
    legs.sort(key=lambda leg: leg.name)

    derive_drainage(ledger)
    _derive_canopy(ledger)

    risks = run_rules(ledger, crop=crop, settings=settings, ledger=ledger)
    findings = _findings(ledger)

    return PlantHealthResult(
        lat=float(lat),
        lon=float(lon),
        crop=crop,
        ledger=ledger,
        findings=findings,
        risks=risks,
        legs=tuple(legs),
        elapsed_s=time.perf_counter() - started,
    )
