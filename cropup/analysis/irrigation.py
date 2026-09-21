"""Irrigation advice: the measured water balance of one field, and what is
missing before anything can be advised.

Composed from the ``geo/`` leaves, not from the vendored ``analyze_irrigation``
(SPEC section 2.1). The differences that matter:

* The vendored version returned ``{"error": ...}`` when no cloud-free scene was
  found, discarding the soil, ET and rainfall answers it already had.
* Its recommendations came from ``get_soil_informed_recommendations``, which
  interpolates pH 7.0 and sand 30% into user-facing prose when the soil call
  fails. Here an advice line is built only from Facts, and the reason it could
  not be built is named instead.
* Its ET came from MODIS ``MOD16A2``, which is permanently masked at Arusha
  (SPEC 3.3). ``geo/water.py`` uses SSEBop VIIRS with a TerraClimate fallback.
* The forward-looking part of irrigation -- FRET reference-ET forecast and
  OpenET -- is **CONUS-only**. A Tanzanian field gets
  ``Missing(out_of_coverage)`` for it, on screen, rather than a US answer
  presented as if it applied.

Nothing here prescribes a depth to apply. It reports the measured balance --
what left the field as ET, what arrived as rain, how wet the root zone is --
and says which of those it could not measure.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from typing import Any

from ..config import Settings, get_settings
from ..evidence import Fact, Ledger, Missing, MissingReason
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
    inherited_reason,
    run_rules,
    timed,
)

__all__ = [
    "IrrigationResult",
    "Advice",
    "analyze_irrigation",
    "WATER_RISK_CATEGORIES",
    "BALANCE_WINDOW_DAYS",
]

# The part of the disease library that is about water. Running the whole library
# here would put leaf blights in an irrigation answer and push the drought rules
# past the cap.
WATER_RISK_CATEGORIES: tuple[str, ...] = ("Drought", "Waterlogging", "Salinity")

# The window the water balance is struck over: CHIRPS publishes a 7-day sum and
# SSEBop a dekad, so 7 days is the shortest window both sides can honour.
BALANCE_WINDOW_DAYS = 7

# Indices that say something about water. The disease library's drought rules
# key on NDMI and NDWI; NDVI comes along because the zone shares need it.
_INDICES = ("ndvi", "ndmi", "ndwi")

_LEG_ASSETS: dict[str, tuple[str, ...]] = {
    "water": (
        geo_water.SSEBOP_DEKADAL,
        geo_water.GLOBAL_ET0_MONTHLY,
        geo_water.ESI_4WK,
        geo_water.SMAP_L4,
        geo_water.CHIRPS_DAILY,
        geo_water.LGRIP30,
    ),
    "soil": ("ISDASOIL/Africa/v1", "projects/sat-io/open-datasets/polaris", "projects/soilgrids-isric"),
    "vegetation": (geo_vegetation.S2_ASSET,),
    "thermal": geo_thermal.LANDSAT_COLLECTIONS,
}


@dataclass(frozen=True)
class Advice:
    """One thing the farmer can act on, and the measurements it rests on.

    ``blocked_by`` is non-empty when the advice could not be formed: it names
    the quantities that were missing. A blocked line is displayed as a question,
    never as a softened recommendation.
    """

    key: str
    text: str
    urgency: str
    evidence: tuple[Fact, ...] = ()
    blocked_by: tuple[str, ...] = ()

    @property
    def actionable(self) -> bool:
        return not self.blocked_by

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "text": self.text,
            "urgency": self.urgency,
            "actionable": self.actionable,
            "blocked_by": list(self.blocked_by),
            "evidence": [f.provenance() for f in self.evidence],
        }


@dataclass(frozen=True)
class IrrigationResult:
    """The turn's water evidence, its findings, its advice and its risks."""

    lat: float
    lon: float
    crop: str | None
    ledger: Ledger
    findings: tuple[Finding, ...]
    advice: tuple[Advice, ...]
    risks: RiskAssessment
    legs: tuple[Leg, ...]
    in_conus: bool
    elapsed_s: float

    def finding(self, key: str) -> Finding | None:
        for item in self.findings:
            if item.key == key:
                return item
        return None

    def actionable(self) -> tuple[Advice, ...]:
        return tuple(a for a in self.advice if a.actionable)

    def as_dict(self) -> dict[str, Any]:
        return {
            "analysis": "irrigation",
            "lat": self.lat,
            "lon": self.lon,
            "crop": self.crop,
            "in_conus": self.in_conus,
            "forecast_available": self.in_conus,
            "findings": [f.as_dict() for f in self.findings],
            "advice": [a.as_dict() for a in self.advice],
            "risks": self.risks.as_dict(),
            "legs": [leg.as_dict() for leg in self.legs],
            "elapsed_s": round(self.elapsed_s, 2),
            "ledger": self.ledger.as_dict(),
        }


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------


def _derive_balance(ledger: Ledger) -> None:
    """The 7-day water balance, and the plant-available water the soil can hold.

    Both are ``Fact.derive`` calls, so each one carries the coarsest resolution
    and the oldest observation of its ingredients -- an 11 km SMAP cell and a
    two-week-old CHIRPS run do not become a field measurement by being added up.
    """
    et = ledger.fact("et_actual")
    rain = ledger.fact("precipitation_7d")
    if et is not None and rain is not None:
        used = float(et.value) * BALANCE_WINDOW_DAYS
        ledger.record(
            Fact.derive(
                "water_balance_7d",
                float(rain.value) - used,
                "mm",
                [et, rain],
                scaling_applied=(
                    f"{BALANCE_WINDOW_DAYS}-day CHIRPS rainfall minus actual ET held at its "
                    f"latest daily rate for {BALANCE_WINDOW_DAYS} days"
                ),
                note=(
                    "negative means the field lost more water than it received; the ET rate is a "
                    "dekad average carried forward, not a daily measurement"
                ),
            )
        )
    else:
        # Whichever ingredient is absent already carries a diagnosed reason;
        # inherit it rather than calling a failed water leg a masked pixel.
        absent = ledger.get("et_actual") if et is None else ledger.get("precipitation_7d")
        ledger.gap(
            "water_balance_7d",
            inherited_reason(absent, MissingReason.MASKED),
            (geo_water.SSEBOP_DEKADAL, geo_water.CHIRPS_DAILY),
            "needs both actual ET and the 7-day rainfall total; one of them is missing",
        )

    pawc = ledger.fact("soil_pawc")
    if pawc is not None:
        ledger.record(
            Fact.derive(
                "soil_pawc_mm_per_m",
                float(pawc.value) * 1000.0,
                "mm/m",
                [pawc],
                scaling_applied="volumetric plant-available water x 1000: mm of water per metre of rooting depth",
                note="per metre of root depth; CropUp does not measure this crop's rooting depth",
            )
        )
    else:
        # POLARIS being US-only is the usual reason, not the only one: a soil
        # leg that raised owes ``soil_pawc`` a ``source_failed``, and reporting
        # that as "POLARIS does not cover this point" is a false statement about
        # a point POLARIS may well cover.
        entry = ledger.get("soil_pawc")
        detail = "the van Genuchten parameters behind plant-available water are POLARIS, which is US-only"
        if isinstance(entry, Missing) and entry.reason is not MissingReason.OUT_OF_COVERAGE:
            detail = f"plant-available water could not be computed here: {entry.reason.describe()}"
        ledger.gap(
            "soil_pawc_mm_per_m",
            inherited_reason(entry, MissingReason.OUT_OF_COVERAGE),
            ("projects/sat-io/open-datasets/polaris",),
            detail,
        )


# ---------------------------------------------------------------------------
# Findings and advice
# ---------------------------------------------------------------------------


def _findings(ledger: Ledger, lat: float, lon: float) -> tuple[Finding, ...]:
    out: list[Finding] = []
    conus = geo_water.in_conus(lat, lon)

    demand = evidence_for(ledger, "et_actual", (geo_water.SSEBOP_DEKADAL, geo_water.TERRACLIMATE))
    reading = ""
    if isinstance(demand, Fact):
        ratio = ledger.fact("et_actual_over_et0")
        reading = f"The field is using {demand.render()}"
        if ratio is not None:
            reading += f", {ratio.render()} of the reference demand for this month"
        reading += "."
        anomaly = ledger.fact("et_actual_anomaly_pct")
        if anomaly is not None:
            reading += f" That is {anomaly.render()} against the median for this dekad."
    out.append(
        Finding(
            "water_use",
            "Actual water use",
            demand,
            reading,
            "info",
            tuple(e for e in (ledger.get("et0"), ledger.get("et_actual_over_et0")) if e),
        )
    )

    balance = evidence_for(ledger, "water_balance_7d")
    reading = ""
    importance = "info"
    if isinstance(balance, Fact):
        rain = ledger.fact("precipitation_7d")
        value = float(balance.value)
        direction = "it lost more water than it received" if value < 0 else "rain covered what the field used"
        reading = f"The {BALANCE_WINDOW_DAYS}-day balance is {balance.render()}: {direction}"
        if rain is not None:
            reading += f", on {rain.render()} of rain"
        reading += "."
        if value < 0:
            importance = "act"
    out.append(
        Finding(
            "water_balance",
            f"{BALANCE_WINDOW_DAYS}-day water balance",
            balance,
            reading,
            importance,
            tuple(e for e in (ledger.get("precipitation_7d"), ledger.get("et_actual")) if e),
        )
    )

    root = evidence_for(ledger, "soil_moisture_rootzone", (geo_water.SMAP_L4, geo_water.FLDAS_MONTHLY))
    reading = ""
    if isinstance(root, Fact):
        wetness = ledger.fact("soil_moisture_rootzone_wetness")
        reading = f"Root-zone soil moisture {root.render()}"
        if wetness is not None:
            reading += f", {wetness.render()} of the way from wilting point to saturation"
        reading += ". The SMAP cell is 11 km across, so this is the neighbourhood, not the field."
    out.append(Finding("root_zone_moisture", "Root-zone soil moisture", root, reading))

    canopy = evidence_for(ledger, "ndmi", (geo_vegetation.S2_ASSET,))
    reading = ""
    importance = "info"
    if isinstance(canopy, Fact):
        reading = f"Canopy moisture index {canopy.render()}."
        if float(canopy.value) < -0.1:
            importance = "watch"
            reading += " Below -0.1 the canopy is reading dry."
    out.append(
        Finding(
            "canopy_moisture",
            "Canopy moisture",
            canopy,
            reading,
            importance,
            tuple(e for e in (ledger.get("ndwi"), ledger.get("ndvi")) if e),
        )
    )

    stress = evidence_for(ledger, "evaporative_stress_index", (geo_water.ESI_4WK,))
    reading = ""
    if isinstance(stress, Fact):
        reading = f"Evaporative stress index {stress.render()}."
        note = stress.staleness_note()
        if note:
            reading += f" It was {note}, so it describes an earlier month rather than this week."
    out.append(Finding("evaporative_stress", "Evaporative stress", stress, reading))

    forecast = evidence_for(ledger, "et0_forecast_7d", (geo_water.FRET_ETO,))
    if isinstance(forecast, Fact):
        day1 = ledger.fact("et0_forecast_day1")
        reading = f"Reference ET over the next week is forecast at {forecast.render()}"
        if day1 is not None:
            reading += f", {day1.render()} tomorrow"
        reading += "."
    else:
        reading = (
            "There is no ET forecast for this field. The only forecast source in the catalog, "
            "NOAA FRET, covers the conterminous United States only, so forward-looking irrigation "
            "advice does not exist here -- everything above is what has already happened."
        )
    out.append(Finding("et_forecast", "Reference ET forecast", forecast, reading, "info" if conus else "watch"))

    regime = evidence_for(ledger, "irrigation_regime", (geo_water.LGRIP30,))
    reading = ""
    if isinstance(regime, Fact):
        irrigated = ledger.fact("irrigated_cropland_pct")
        # LGRIP30's fourth verdict is "not mapped as cropland", which does not
        # read as a regime; it is the absence of one.
        if str(regime.value) == "not mapped as cropland":
            reading = "LGRIP30 does not map this pixel as cropland at all, so it assigns no regime"
        else:
            reading = f"The 30 m LGRIP30 pixel here is mapped as {regime.render()}"
        if irrigated is not None:
            reading += f"; {irrigated.render()} of the surrounding window is mapped irrigated"
        reading += "."
    out.append(
        Finding(
            "irrigation_regime",
            "Irrigated or rainfed",
            regime,
            reading,
            "info",
            tuple(
                e
                for e in (
                    ledger.get("cropland_pct"),
                    ledger.get("irrigated_cropland_pct"),
                    ledger.get("rainfed_cropland_pct"),
                )
                if e
            ),
        )
    )

    holding = evidence_for(ledger, "soil_pawc_mm_per_m")
    reading = ""
    if isinstance(holding, Fact):
        texture = ledger.fact("soil_texture_class")
        reading = f"The topsoil holds {holding.render()} of plant-available water"
        if texture is not None:
            reading += f" at a {texture.render()} texture"
        reading += ". Turning that into a depth needs this crop's rooting depth, which CropUp does not measure."
    out.append(
        Finding(
            "soil_water_holding",
            "Soil water holding capacity",
            holding,
            reading,
            "info",
            tuple(e for e in (ledger.get("soil_texture_class"), ledger.get("soil_field_capacity")) if e),
        )
    )

    return tuple(out)


def _advice(ledger: Ledger, lat: float, lon: float) -> tuple[Advice, ...]:
    """Deterministic advice lines, each one gated on the Facts it needs."""
    out: list[Advice] = []

    balance = ledger.get("water_balance_7d")
    if isinstance(balance, Fact):
        et = ledger.require("et_actual", template="irrigation.balance")
        rain = ledger.require("precipitation_7d", template="irrigation.balance")
        value = float(balance.value)
        if value < 0:
            out.append(
                Advice(
                    "replace_deficit",
                    (
                        f"The {BALANCE_WINDOW_DAYS}-day water balance is {balance.render()}: the field used "
                        f"{et.render()} and received {rain.render()} of rain, so it lost more than it "
                        "received. Replacing that shortfall is the measured need; the depth to apply also "
                        "depends on what is already in the root zone, which is measured here only at 11 km."
                    ),
                    "Within 3 days",
                    (balance, et, rain),
                )
            )
        else:
            out.append(
                Advice(
                    "rain_covered_demand",
                    (
                        f"Rain covered demand over the last {BALANCE_WINDOW_DAYS} days: {rain.render()} "
                        f"against {et.render()} of use. No shortfall was measured."
                    ),
                    "Ongoing",
                    (balance, et, rain),
                )
            )
    else:
        missing = [q for q in ("et_actual", "precipitation_7d") if not ledger.has(q)]
        out.append(
            Advice(
                "replace_deficit",
                (
                    "The water balance could not be struck for this field, so there is no measured "
                    "shortfall to act on. What is missing: " + ", ".join(missing) + "."
                ),
                "",
                (),
                tuple(missing),
            )
        )

    if not geo_water.in_conus(lat, lon):
        out.append(
            Advice(
                "schedule_ahead",
                (
                    "Scheduling the next irrigation from a forecast is not possible at this location: "
                    "the reference-ET forecast (NOAA FRET) and OpenET both stop at the United States "
                    "border. Use the measured balance above and a field check instead."
                ),
                "",
                (),
                ("et0_forecast_7d",),
            )
        )
    else:
        forecast = ledger.get("et0_forecast_7d")
        if isinstance(forecast, Fact):
            out.append(
                Advice(
                    "schedule_ahead",
                    (
                        f"Reference ET over the coming week is forecast at {forecast.render()}. Plan the "
                        "next application against that demand and the balance above."
                    ),
                    "Within 1 week",
                    (forecast,),
                )
            )
        else:
            out.append(
                Advice(
                    "schedule_ahead",
                    "The forecast source covers this location but returned nothing for this field.",
                    "",
                    (),
                    ("et0_forecast_7d",),
                )
            )

    regime = ledger.get("irrigation_regime")
    if isinstance(regime, Fact) and str(regime.value) == "rainfed":
        out.append(
            Advice(
                "regime_check",
                (
                    "LGRIP30 maps this pixel as rainfed. If the field is in fact irrigated, say so: "
                    "the map is a 30 m classification from 2015, not a record of this field."
                ),
                "Ongoing",
                (regime,),
            )
        )

    return tuple(out)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze_irrigation(
    lat: float,
    lon: float,
    crop: str | None = None,
    *,
    radius_m: float | None = None,
    end_date: dt.date | str | None = None,
    settings: Settings | None = None,
    ledger: Ledger | None = None,
) -> IrrigationResult:
    """What this field's water balance measures, and what cannot be measured here.

    Runs the water, soil, vegetation and thermal legs concurrently. The
    CONUS-only forecast quantities are requested everywhere so that a Tanzanian
    field reports them as out of coverage rather than omitting them: the farmer
    should be able to see that forward-looking advice does not exist for them.
    """
    settings = settings or get_settings()
    started = time.perf_counter()
    ledger = ledger if ledger is not None else Ledger(turn="irrigation")

    jobs = {
        # irrigation_mapping=True here: this is the analysis that needs LGRIP30,
        # and geo/context.py is not called, so nothing else claims the name.
        "water": timed(
            "water",
            lambda: geo_water.get_water_status(
                lat, lon, end_date=end_date, irrigation_mapping=True, settings=settings
            ),
        ),
        "soil": timed("soil", lambda: geo_soil.get_soil(lat, lon, settings=settings)),
        "vegetation": timed(
            "vegetation",
            lambda: geo_vegetation.get_vegetation(
                lat, lon, radius_m=radius_m, end_date=end_date, indices=_INDICES, zones=True, settings=settings
            ),
        ),
        "thermal": timed(
            "thermal",
            lambda: geo_thermal.get_land_surface_temperature(
                lat, lon, radius_m=radius_m, end_date=end_date, settings=settings
            ),
        ),
    }

    expected = {
        "water": geo_water.water_quantities(irrigation_mapping=True),
        "soil": geo_soil.SOIL_QUANTITIES + SOIL_DERIVED_QUANTITIES,
        "vegetation": geo_vegetation.vegetation_quantities(_INDICES, zones=True),
        "thermal": geo_thermal.thermal_quantities(),
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

    derive_drainage(ledger)
    _derive_balance(ledger)

    risks = run_rules(
        ledger,
        crop=crop,
        categories=WATER_RISK_CATEGORIES,
        settings=settings,
        ledger=ledger,
    )

    return IrrigationResult(
        lat=float(lat),
        lon=float(lon),
        crop=crop,
        ledger=ledger,
        findings=_findings(ledger, lat, lon),
        advice=_advice(ledger, lat, lon),
        risks=risks,
        legs=tuple(legs),
        in_conus=geo_water.in_conus(lat, lon),
        elapsed_s=time.perf_counter() - started,
    )
