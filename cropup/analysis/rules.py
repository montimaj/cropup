"""The disease-library rules engine, ported from the vendored backend with its
bugs fixed, plus the small vocabulary the three orchestrators share.

What the port changes (SPEC section 2.1, bugs 7 and 8):

* The vendored engine called ``obs.get("crop_type", "").lower()`` and raised
  ``AttributeError`` whenever the crop was ``None``. Here the crop is optional:
  with no crop only the 46 ``Generic`` rules are evaluated, and the number of
  crop-specific rules that were therefore skipped is reported rather than
  silently lost.
* The vendored ``_eval_condition`` returned ``False`` for any observation it did
  not have. A missing measurement therefore *shortened* the risk list, which is
  the opposite of honest: "we did not look" was displayed as "you are fine".
  Here a condition is tri-state -- True, False or unknown -- and a rule that
  cannot be decided lands in :attr:`RiskAssessment.unevaluated` with the fields
  it was waiting for.
* 288 of the 384 rules key on a single NDVI threshold, so one low NDVI fired
  about eighteen risks at once, "Maize Lethal Necrosis" and "Heavy Metal
  Contamination" among them. Risks are deduplicated by ``issue_id``, ranked by
  severity *within* their evidence tier, and capped at ``CROPUP_RISK_CAP``. A
  rule that only canopy indices satisfied is a candidate to scout for, and is
  ranked below and labelled differently from one something off the canopy also
  confirmed. What the cap removed is kept in
  :attr:`RiskAssessment.suppressed`, so "and 11 more" is sayable and nothing is
  quietly dropped.
* A threshold is compared only when the measurement is in the unit the rule was
  written in. ``cec_mmol_kg`` against a Fact in cmol(+)/kg is a factor-of-ten
  error that reads as a plausible number; the conversion is declared here or the
  condition is unknown.

The engine never sees a bare float. Observations arrive as ``Fact`` objects, so
every triggered risk can name the measurement, the instrument and the date that
triggered it.

This module also holds the pieces all three orchestrators need -- ``Finding``,
``Leg``, the severity ranking, the thread fan-out and the drainage inference --
because ``analysis/`` has four modules and no separate types module, and each of
them starts from a vocabulary this file owns.
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..config import Settings, get_settings
from ..evidence import Evidence, Fact, Ledger, Missing, MissingReason

__all__ = [
    "SEVERITY_ORDER",
    "severity_rank",
    "SOIL_DERIVED_QUANTITIES",
    "CLIMATE_DERIVED_QUANTITIES",
    "Finding",
    "Leg",
    "gather",
    "timed",
    "absorb",
    "evidence_for",
    "inherited_reason",
    "FieldSpec",
    "FIELD_SPECS",
    "RULE_FIELDS",
    "Condition",
    "Rule",
    "Trigger",
    "Risk",
    "Unevaluated",
    "RiskAssessment",
    "load_rules",
    "rules_for_crop",
    "run_rules",
    "drainage_from_texture",
    "derive_drainage",
    "DRAINAGE_BY_TEXTURE",
    "clear_cache",
]


# ---------------------------------------------------------------------------
# Shared analysis vocabulary
# ---------------------------------------------------------------------------

# Most severe first. "" is the severity of the 50 "Perfect Conditions" rows,
# which never reach the risk list at all.
SEVERITY_ORDER: tuple[str, ...] = ("Severe", "High", "Medium", "Low")

_SEVERITY_RANK = {name.lower(): i for i, name in enumerate(SEVERITY_ORDER)}


def severity_rank(severity: str | None) -> int:
    """Sort key for a severity label. An unrecognised label sorts last rather
    than being promoted to a level nobody wrote down."""
    return _SEVERITY_RANK.get((severity or "").strip().lower(), len(SEVERITY_ORDER))


# The quantities ``geo/soil.py`` *derives* and publishes alongside its measured
# ``SOIL_QUANTITIES``. An orchestrator has to list them among the quantities its
# soil leg owes, because a leg that raised owes everything it would have
# returned: leave them out and a soil failure records nothing for them, and then
# the first caller to notice names them under a reason nobody diagnosed --
# ``not_requested`` from :func:`run_rules`, ``masked`` from
# :func:`derive_drainage`, ``out_of_coverage`` ("POLARIS is US-only") from the
# irrigation water balance. The reason was ``source_failed``.
SOIL_DERIVED_QUANTITIES: tuple[str, ...] = (
    "soil_texture_class",
    "soil_field_capacity",
    "soil_wilting_point",
    "soil_pawc",
)

# The same for ``geo/climate.py``: ``CLIMATE_QUANTITIES`` names the 24 monthly
# normals only, and the Köppen class and the annual aggregates the orchestrators
# actually read are derived from them.
CLIMATE_DERIVED_QUANTITIES: tuple[str, ...] = (
    "koppen_code",
    "koppen_name",
    "annual_temp_c",
    "annual_precip_mm",
    "warmest_month_temp_c",
    "coldest_month_temp_c",
    "wettest_month_precip_mm",
    "driest_month_precip_mm",
)


@dataclass(frozen=True)
class Finding:
    """One thing an orchestrator has to say, tied to the evidence for it.

    ``reading`` is the interpretation in words. Every number inside it comes
    from ``evidence.render()`` or a supporting Fact's ``render()``: the
    orchestrators have no other way to put a digit on screen.
    """

    key: str
    label: str
    evidence: Evidence
    reading: str = ""
    importance: str = "info"  # info | watch | act
    supporting: tuple[Evidence, ...] = ()

    @property
    def measured(self) -> bool:
        return self.evidence.measured

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "measured": self.measured,
            "rendered": self.evidence.render(),
            "reading": self.reading,
            "importance": self.importance,
            "provenance": self.evidence.provenance(),
            "supporting": [e.provenance() for e in self.supporting],
        }


@dataclass(frozen=True)
class Leg:
    """One data leg of an orchestrator's fan-out: whether it answered, and how
    long it took. The wall time is per leg, so a slow source is nameable."""

    name: str
    ok: bool
    elapsed_s: float
    facts: int
    gaps: int
    detail: str = ""
    assets: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "elapsed_s": round(self.elapsed_s, 2),
            "facts": self.facts,
            "gaps": self.gaps,
            "detail": self.detail,
            "assets": list(self.assets),
        }


def timed(name: str, fn: Callable[[], Any]) -> Callable[[], tuple[str, float, Any]]:
    """Wrap a leg so its wall time and its failure travel with its result."""

    def run() -> tuple[str, float, Any]:
        started = time.perf_counter()
        try:
            payload: Any = fn()
        except BaseException as exc:  # noqa: BLE001 - recorded by absorb(), never swallowed
            payload = exc
        return name, time.perf_counter() - started, payload

    return run


def absorb(
    ledger: Ledger,
    legs: list[Leg],
    name: str,
    elapsed: float,
    payload: Any,
    quantities: Sequence[str],
    assets: Sequence[str] = (),
) -> None:
    """Fold one leg's result into the turn ledger and record how the leg went.

    A leg that raised is not a silent absence: every quantity it owed is written
    as ``Missing(source_failed)`` carrying the exception text, so the turn can
    still say what it would have measured.
    """
    assets = tuple(assets)
    before_facts, before_gaps = len(ledger.facts), len(ledger.gaps)
    if isinstance(payload, BaseException):
        for quantity in quantities:
            ledger.gap(quantity, MissingReason.SOURCE_FAILED, assets, f"{type(payload).__name__}: {payload}")
        legs.append(
            Leg(name, False, elapsed, 0, len(ledger.gaps) - before_gaps, f"{type(payload).__name__}: {payload}", assets)
        )
        return
    if isinstance(payload, Ledger):
        ledger.merge(payload)
    elif isinstance(payload, Mapping):
        ledger.extend(payload.values())
    elif isinstance(payload, (Fact, Missing)):
        ledger.add(payload)
    facts = len(ledger.facts) - before_facts
    gaps = len(ledger.gaps) - before_gaps
    legs.append(Leg(name, facts > 0, elapsed, facts, gaps, "", assets))


def evidence_for(ledger: Ledger, quantity: str, chain: Sequence[str] = (), detail: str | None = None) -> Evidence:
    """The ledger's entry for a quantity, or a Missing saying nothing asked for it.

    Returning a ``Missing`` rather than ``None`` keeps "we never looked" on
    screen next to "we looked and the pixel was masked".
    """
    entry = ledger.get(quantity)
    if entry is not None:
        return entry
    return Missing(quantity, MissingReason.NOT_REQUESTED, tuple(chain), detail)


def inherited_reason(entry: Evidence | None, default: MissingReason) -> MissingReason:
    """The reason a derived quantity is missing: its ingredient's, not a guess.

    ``geo/soil.py`` does this for its own derivations, and the orchestrators need
    it for theirs. Without it a leg that raised is reported as "the pixel is
    masked" or "POLARIS is US-only" -- different, and false, statements about
    what happened.
    """
    return entry.reason if isinstance(entry, Missing) else default


def gather(jobs: Mapping[str, Callable[[], Any]], *, max_workers: int | None = None) -> dict[str, Any]:
    """Run the data legs concurrently and return ``{name: result_or_exception}``.

    An exception is returned rather than raised: one dead source must not take
    the other five down with it. The caller decides what the failure means and
    records it as a gap -- there is no result that means "nothing happened".
    """
    if not jobs:
        return {}
    workers = max_workers or get_settings().ee_max_workers
    out: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
        futures = {pool.submit(fn): name for name, fn in jobs.items()}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                out[name] = future.result()
            except BaseException as exc:  # noqa: BLE001 - handed to the caller intact
                out[name] = exc
    return out


# ---------------------------------------------------------------------------
# The observation vocabulary the CSV was written against
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    """One observation name used by ``rule_logic``, and what has to be measured
    for it: which CropUp quantity carries it and in which unit."""

    field: str
    quantity: str
    unit: str | None  # None: categorical, no unit to check
    label: str
    note: str = ""

    @property
    def categorical(self) -> bool:
        return self.unit is None


def _spec(field: str, quantity: str, unit: str | None, label: str, note: str = "") -> tuple[str, FieldSpec]:
    return field, FieldSpec(field, quantity, unit, label, note)


# The 21 fields the 384 rules actually key on, mapped onto the quantity names
# geo/ publishes. Four of them have no source in CropUp at all; they are listed
# anyway, because a rule waiting on an unmeasurable field must be reported as
# undecided rather than quietly counted as "no risk".
FIELD_SPECS: dict[str, FieldSpec] = dict(
    (
        _spec("ndvi", "ndvi", "index", "NDVI"),
        _spec("ndmi", "ndmi", "index", "NDMI"),
        _spec("ndre", "ndre", "index", "NDRE"),
        _spec("ndwi", "ndwi", "index", "NDWI"),
        _spec("psri", "psri", "index", "PSRI"),
        _spec("reci", "reci", "index", "ReCI"),
        _spec("gci", "gci", "index", "GCI"),
        _spec("ph", "soil_ph", "pH", "soil pH"),
        _spec("clay_pct", "soil_clay_pct", "%", "clay content"),
        _spec("sand_pct", "soil_sand_pct", "%", "sand content"),
        _spec("soc_g_kg", "soil_soc", "g/kg", "soil organic carbon"),
        # The library names this field ``cec_mmol_kg``, but its single threshold
        # -- GEN_K_DEF, "Potassium Deficiency", ``cec_mmol_kg < 10`` -- is the
        # textbook low-CEC cut-off, which is written in cmol(+)/kg. Read as
        # mmol(+)/kg it says "below 1 cmol(+)/kg", which no soil reaches: iSDA
        # and SoilGrids both publish cmol(+)/kg, so multiplying by ten to reach
        # the field's *name* put every real soil above the threshold and made the
        # rule unfirable. The unit a threshold is compared in is the unit that
        # threshold was written in, not the one the column heading claims.
        _spec("cec_mmol_kg", "soil_cec", "cmol(+)/kg", "cation exchange capacity"),
        _spec("bulk_density", "soil_bulk_density", "g/cm3", "bulk density"),
        _spec("soil_texture", "soil_texture_class", None, "soil texture class"),
        _spec(
            "soil_drainage",
            "soil_drainage",
            None,
            "soil drainage",
            "inferred from the texture class; no drainage layer is measured directly",
        ),
        _spec(
            "soil_ec_ds_m",
            "soil_electrical_conductivity",
            "dS/m",
            "soil electrical conductivity",
            "no CropUp source measures field EC; salinity rules stay undecided until a farmer enters a reading",
        ),
        _spec(
            "avg_temp_7d",
            "air_temp_7d_mean",
            "degC",
            "7-day mean air temperature",
            "air temperature, not the Landsat land surface temperature; no CropUp source measures it",
        ),
        _spec(
            "max_temp_7d",
            "air_temp_7d_max",
            "degC",
            "7-day maximum air temperature",
            "air temperature, not the Landsat land surface temperature; no CropUp source measures it",
        ),
        _spec(
            "min_temp_7d",
            "air_temp_7d_min",
            "degC",
            "7-day minimum air temperature",
            "air temperature, not the Landsat land surface temperature; no CropUp source measures it",
        ),
        _spec(
            "consecutive_dry_days",
            "consecutive_dry_days",
            "days",
            "consecutive dry days",
            "needs a daily rainfall series; CropUp reads CHIRPS as 7-day and 30-day totals only",
        ),
        _spec(
            "consecutive_wet_days",
            "consecutive_wet_days",
            "days",
            "consecutive wet days",
            "needs a daily rainfall series; CropUp reads CHIRPS as 7-day and 30-day totals only",
        ),
        _spec(
            "growth_stage",
            "growth_stage",
            None,
            "growth stage",
            "the farmer tells us this; nothing observes it",
        ),
    )
)

RULE_FIELDS: tuple[str, ...] = tuple(FIELD_SPECS)

# Exact unit conversions, declared rather than assumed. Anything not on this
# list and not already in the expected unit leaves the condition undecided.
_UNIT_CONVERSIONS: dict[tuple[str, str], tuple[float, str]] = {
    ("cmol(+)/kg", "mmol(+)/kg"): (10.0, "1 cmol(+)/kg = 10 mmol(+)/kg"),
    ("mmol(+)/kg", "cmol(+)/kg"): (0.1, "10 mmol(+)/kg = 1 cmol(+)/kg"),
}

# Drainage is not measured by any asset in the registry. The rules need the
# word, so it is inferred from the USDA texture class by the standard
# coarse/medium/fine ordering, and every Fact built from it says so.
DRAINAGE_BY_TEXTURE: dict[str, str] = {
    "sand": "Well",
    "loamy sand": "Well",
    "sandy loam": "Well",
    "loam": "Well",
    "sandy clay loam": "Moderate",
    "silt loam": "Moderate",
    "silt": "Moderate",
    "clay loam": "Moderate",
    "silty clay loam": "Moderate",
    "sandy clay": "Poor",
    "silty clay": "Poor",
    "clay": "Poor",
}


def drainage_from_texture(texture_class: str) -> str | None:
    """The drainage word for a USDA texture class: Well, Moderate or Poor.

    ``None`` when the class is not one of the twelve.

    An inference, not a measurement: it is only ever used to build a Fact whose
    note names the texture class it came from.
    """
    return DRAINAGE_BY_TEXTURE.get(str(texture_class).strip().lower())


def derive_drainage(ledger: Ledger) -> None:
    """Add the drainage class the disease library asks for, from the texture class.

    Nothing in the registry measures drainage. Deriving it from texture is an
    inference, so it is a derived Fact whose note says what it was inferred from,
    and a Missing when the texture class itself is missing.
    """
    texture = ledger.fact("soil_texture_class")
    if texture is None:
        # The texture class is missing for a reason the soil leg already
        # diagnosed. Inherit it: a leg that raised is not a masked pixel.
        entry = ledger.get("soil_texture_class")
        detail = "drainage is inferred from the USDA texture class, which is itself missing here"
        if isinstance(entry, Missing):
            detail += f": {entry.reason.describe()}"
        ledger.gap(
            "soil_drainage",
            inherited_reason(entry, MissingReason.MASKED),
            ("derived from soil_texture_class",),
            detail,
        )
        return
    drainage = drainage_from_texture(str(texture.value))
    if drainage is None:
        ledger.gap(
            "soil_drainage",
            MissingReason.MASKED,
            ("derived from soil_texture_class",),
            f"no drainage class is defined for texture {texture.value!r}",
        )
        return
    ledger.record(
        Fact.derive(
            "soil_drainage",
            drainage,
            "",
            [texture],
            scaling_applied="USDA texture class mapped to Well / Moderate / Poor by the coarse-to-fine ordering",
            note=f"inferred from the {texture.value} texture class, not measured; no drainage layer exists in the registry",
        )
    )


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------

_NUMERIC_OPS = frozenset({">=", "<=", ">", "<"})
_SKIPPED_LOGIC = frozenset({"EXTERNAL", "CROP_SPECIFIC"})
_SKIPPED_CATEGORY = "Perfect Conditions"

# A canopy index is a symptom, not a cause. Rules whose only satisfied
# conditions are vegetation indices are labelled and ranked below rules of the
# same severity that something off the canopy also had to confirm.
_INDEX_FIELDS = frozenset({"ndvi", "ndmi", "ndre", "ndwi", "psri", "reci", "gci"})


@dataclass(frozen=True)
class Condition:
    """One clause of a rule, resolved against the quantity that can answer it."""

    field: str
    op: str
    value: Any
    spec: FieldSpec | None  # None: the CSV names a field this port does not know

    @property
    def quantity(self) -> str | None:
        return self.spec.quantity if self.spec else None

    @property
    def label(self) -> str:
        return self.spec.label if self.spec else self.field

    def describe(self) -> str:
        """The threshold in words, with no observation in it."""
        words = {">=": "at least", "<=": "at most", ">": "above", "<": "below", "==": "is"}
        if self.op == "in":
            return f"{self.label} is one of {', '.join(str(v) for v in self.value)}"
        if self.op == "contains":
            return f"{self.label} mentions {self.value}"
        if self.op == "between":
            return f"{self.label} between {self.value[0]} and {self.value[1]}"
        return f"{self.label} {words.get(self.op, self.op)} {self.value}"

    def as_dict(self) -> dict[str, Any]:
        return {"field": self.field, "op": self.op, "value": self.value, "quantity": self.quantity}


@dataclass(frozen=True)
class Rule:
    """One row of ``disease_library.csv`` that can fire."""

    issue_id: str
    crop: str
    name: str
    category: str
    causal_agent: str
    severity: str
    urgency: str
    notes: str
    recommendation: str
    logic: str
    conditions: tuple[Condition, ...]

    @property
    def is_generic(self) -> bool:
        return self.crop.strip().lower() == "generic"

    @property
    def rank(self) -> int:
        return severity_rank(self.severity)

    @property
    def fields(self) -> tuple[str, ...]:
        seen: list[str] = []
        for cond in self.conditions:
            if cond.field not in seen:
                seen.append(cond.field)
        return tuple(seen)

    def as_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "crop": self.crop,
            "name": self.name,
            "category": self.category,
            "severity": self.severity,
            "logic": self.logic,
            "conditions": [c.as_dict() for c in self.conditions],
        }


_RULES_LOCK = threading.Lock()
_RULES_CACHE: dict[tuple[str, float, int], tuple[Rule, ...]] = {}


def clear_cache() -> None:
    """Forget the parsed rules. For tests and for tools/."""
    with _RULES_LOCK:
        _RULES_CACHE.clear()


def load_rules(path: Path | str | None = None, settings: Settings | None = None) -> tuple[Rule, ...]:
    """Parse ``disease_library.csv`` into rules that can actually be evaluated.

    Rows that can never fire are dropped here rather than filtered at every
    call: the 50 "Perfect Conditions" rows, the 3 ``EXTERNAL`` /
    ``CROP_SPECIFIC`` rows whose logic lives somewhere else, and the rows whose
    condition list is empty.
    """
    settings = settings or get_settings()
    csv_path = Path(path) if path is not None else settings.require_data_file("disease_library")
    stat = csv_path.stat()
    key = (str(csv_path), stat.st_mtime, stat.st_size)
    with _RULES_LOCK:
        cached = _RULES_CACHE.get(key)
    if cached is not None:
        return cached

    rules: list[Rule] = []
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row.get("category") or "").strip() == _SKIPPED_CATEGORY:
                continue
            try:
                logic = json.loads(row.get("rule_logic") or "null")
            except (TypeError, ValueError):
                logic = None
            if not isinstance(logic, dict):
                continue
            kind = str(logic.get("logic", "AND")).upper()
            if kind in _SKIPPED_LOGIC:
                continue
            raw_conditions = logic.get("conditions") or []
            if not raw_conditions:
                continue
            conditions = tuple(
                Condition(
                    field=str(c.get("field")),
                    op=str(c.get("op")),
                    value=c.get("value"),
                    spec=FIELD_SPECS.get(str(c.get("field"))),
                )
                for c in raw_conditions
                if isinstance(c, dict)
            )
            if not conditions:
                continue
            rules.append(
                Rule(
                    issue_id=(row.get("issue_id") or "").strip(),
                    crop=(row.get("crop") or "Generic").strip(),
                    name=(row.get("issue_name") or "").strip(),
                    category=(row.get("category") or "").strip(),
                    causal_agent=(row.get("causal_agent") or "").strip(),
                    severity=(row.get("severity") or "").strip(),
                    urgency=(row.get("urgency") or "").strip(),
                    notes=(row.get("notes") or "").strip(),
                    recommendation=(row.get("recommendation") or "").strip(),
                    logic=kind,
                    conditions=conditions,
                )
            )

    out = tuple(rules)
    with _RULES_LOCK:
        _RULES_CACHE[key] = out
    return out


def rules_for_crop(crop: str | None, rules: Sequence[Rule] | None = None) -> tuple[Rule, ...]:
    """The rules that apply to a crop: the generic ones plus that crop's own.

    ``crop=None`` yields the generic rules only. The vendored engine crashed
    here instead.
    """
    pool = rules if rules is not None else load_rules()
    if crop is None or not str(crop).strip():
        return tuple(r for r in pool if r.is_generic)
    wanted = str(crop).strip().lower()
    return tuple(r for r in pool if r.is_generic or r.crop.lower() == wanted)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trigger:
    """A condition that was satisfied, and the measurement that satisfied it."""

    condition: Condition
    fact: Fact
    comparison_value: float | str | bool
    margin: float  # how far past the threshold, relative to the threshold
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.condition.field,
            "quantity": self.fact.quantity,
            "op": self.condition.op,
            "threshold": self.condition.value,
            "text": self.text,
            "margin": round(self.margin, 4),
            "provenance": self.fact.provenance(),
        }


@dataclass(frozen=True)
class Risk:
    """A rule that fired, with the observations that fired it."""

    issue_id: str
    name: str
    category: str
    severity: str
    urgency: str
    causal_agent: str
    notes: str
    recommendation: str
    crop: str
    crop_specific: bool
    triggers: tuple[Trigger, ...]

    @property
    def rank(self) -> int:
        return severity_rank(self.severity)

    @property
    def corroboration(self) -> int:
        """How many distinct measurements had to be true for this to fire."""
        return len({t.fact.quantity for t in self.triggers})

    @property
    def index_only(self) -> bool:
        """True when nothing but canopy indices support this risk.

        Two indices are not two independent measurements: they come off the same
        Sentinel-2 composite and they all fall when the canopy thins. Maize
        Lethal Necrosis, fall armyworm, drought and heavy-metal contamination
        depress NDVI alike, which is how the vendored build put all four on one
        screen off one reading.
        """
        return bool(self.triggers) and all(t.condition.field in _INDEX_FIELDS for t in self.triggers)

    @property
    def evidence_strength(self) -> str:
        """``"corroborated"`` once something other than the canopy had to be
        true as well, otherwise ``"canopy_only"``."""
        return "canopy_only" if self.index_only else "corroborated"

    @property
    def confirmed(self) -> bool:
        return not self.index_only

    @property
    def margin(self) -> float:
        """The weakest link: how far past its threshold the least extreme
        satisfied condition sits."""
        return min((t.margin for t in self.triggers), default=0.0)

    def evidence_line(self) -> str:
        """One sentence naming every measurement behind this risk."""
        return "; ".join(t.text for t in self.triggers)

    def sort_key(self) -> tuple[Any, ...]:
        """Severity, but inside the evidence tier first.

        SPEC 4.3 asks for a severity ranking. Applied to this library alone it
        reproduces the bug it is meant to fix: 288 of the 331 live rules key on
        one NDVI threshold, so a single low reading puts "Maize Lethal Necrosis
        (Severe)" at the top of a list whose evidence is one number. Ranking by
        severity *within* the evidence tier keeps the severity ordering the spec
        asks for and stops a canopy-only candidate outranking a soil measurement.
        """
        return (
            0 if self.confirmed else 1,
            self.rank,
            -self.corroboration,
            0 if self.crop_specific else 1,
            -self.margin,
            self.issue_id,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "name": self.name,
            "category": self.category,
            "severity": self.severity,
            "urgency": self.urgency,
            "causal_agent": self.causal_agent,
            "detail": self.notes,
            "recommendation": self.recommendation,
            "crop": self.crop,
            "crop_specific": self.crop_specific,
            "evidence_strength": self.evidence_strength,
            "evidence": self.evidence_line(),
            "triggers": [t.as_dict() for t in self.triggers],
        }


@dataclass(frozen=True)
class Unevaluated:
    """A rule that could not be decided, and what it was waiting for.

    The vendored engine answered ``False`` here, so a field with no soil data
    looked healthier than a field with bad soil data.
    """

    issue_id: str
    name: str
    category: str
    severity: str
    crop: str
    waiting_for: tuple[str, ...]  # rule field names
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "name": self.name,
            "category": self.category,
            "severity": self.severity,
            "crop": self.crop,
            "waiting_for": list(self.waiting_for),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class RiskAssessment:
    """What the rules engine concluded, including what it could not conclude."""

    crop: str | None
    risks: tuple[Risk, ...]
    suppressed: tuple[Risk, ...]
    unevaluated: tuple[Unevaluated, ...]
    considered: int
    decided: int
    fired: int
    cap: int
    crop_rules_skipped: int
    fields_used: tuple[str, ...]
    fields_missing: dict[str, str]  # rule field -> why it had no value
    categories: tuple[str, ...] = ()

    @property
    def capped(self) -> bool:
        return bool(self.suppressed)

    @property
    def all_unconfirmed(self) -> bool:
        """Every displayed risk rests on canopy indices alone.

        The caller must say so: a list of named diseases that only a low NDVI
        supports is a list of candidates, and reads as a diagnosis if nobody
        writes the caveat down.
        """
        return bool(self.risks) and not any(r.confirmed for r in self.risks)

    def caveat(self) -> str | None:
        """The sentence that has to travel with the risk list, or ``None``."""
        if self.all_unconfirmed:
            return (
                "Every risk below rests on canopy indices alone. A thin canopy looks the same "
                "whatever thinned it, so these are candidates to scout for, not diagnoses."
            )
        if any(not r.confirmed for r in self.risks):
            return "Risks marked 'canopy_only' rest on canopy indices alone and need scouting to confirm."
        return None

    def actions(self) -> tuple[dict[str, str], ...]:
        """The recommendations of the displayed risks, most urgent first, one
        per distinct wording."""
        seen: set[str] = set()
        out: list[dict[str, str]] = []
        for risk in self.risks:
            text = risk.recommendation
            if not text or text in seen:
                continue
            seen.add(text)
            out.append({"action": text, "urgency": risk.urgency, "issue_id": risk.issue_id})
        return tuple(out)

    def summary(self) -> str:
        parts = [f"{self.fired} of {self.considered} rules fired", f"{len(self.risks)} shown (cap {self.cap})"]
        if self.suppressed:
            parts.append(f"{len(self.suppressed)} below the cap")
        if self.unevaluated:
            parts.append(f"{len(self.unevaluated)} undecided for want of data")
        if self.crop_rules_skipped:
            parts.append(f"{self.crop_rules_skipped} crop-specific rules skipped: no crop named")
        confirmed = sum(1 for r in self.risks if r.confirmed)
        if self.risks:
            parts.append(f"{confirmed} of {len(self.risks)} shown rest on more than the canopy")
        return ", ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "crop": self.crop,
            "risks": [r.as_dict() for r in self.risks],
            "suppressed": [{"issue_id": r.issue_id, "name": r.name, "severity": r.severity} for r in self.suppressed],
            "unevaluated": [u.as_dict() for u in self.unevaluated],
            "actions": list(self.actions()),
            "considered": self.considered,
            "decided": self.decided,
            "fired": self.fired,
            "cap": self.cap,
            "capped": self.capped,
            "crop_rules_skipped": self.crop_rules_skipped,
            "fields_used": list(self.fields_used),
            "fields_missing": dict(self.fields_missing),
            "categories": list(self.categories),
            "caveat": self.caveat(),
            "summary": self.summary(),
        }


def _observations(evidence: Ledger | Mapping[str, Evidence]) -> dict[str, Evidence]:
    """Index the turn's evidence by quantity name, newest entry winning."""
    if isinstance(evidence, Ledger):
        out: dict[str, Evidence] = {}
        for entry in evidence:
            out[entry.quantity] = entry
        return out
    return {str(k): v for k, v in evidence.items()}


def _convert(fact: Fact, spec: FieldSpec) -> tuple[float | None, str | None]:
    """The fact's value in the unit the rule was written in, or a reason why not."""
    have = (fact.unit or "").strip()
    want = spec.unit or ""
    if have.lower() == want.lower():
        return float(fact.value), None
    conversion = _UNIT_CONVERSIONS.get((have.lower(), want.lower()))
    if conversion is None:
        return None, f"{spec.label} is measured in {have or 'no unit'}, the rule is written in {want}"
    factor, _ = conversion
    return float(fact.value) * factor, None


def _margin(op: str, value: float, threshold: float) -> float:
    """How far past the threshold, scaled so indices and temperatures compare."""
    scale = max(abs(threshold), 1.0)
    if op in (">", ">="):
        return max(0.0, (value - threshold) / scale)
    if op in ("<", "<="):
        return max(0.0, (threshold - value) / scale)
    return 1.0


def _evaluate(cond: Condition, observations: Mapping[str, Evidence]) -> tuple[bool | None, Trigger | None, str]:
    """Tri-state condition evaluation.

    Returns ``(verdict, trigger, reason)``. ``verdict`` is ``None`` when the
    condition could not be decided; ``reason`` then says why, in words a farmer
    could read.
    """
    spec = cond.spec
    if spec is None:
        return None, None, f"'{cond.field}' is not a measurement this build knows how to take"

    entry = observations.get(spec.quantity)
    if entry is None:
        detail = f" ({spec.note})" if spec.note else ""
        return None, None, f"{spec.label} was not measured{detail}"
    if isinstance(entry, Missing):
        return None, None, f"{spec.label}: {entry.reason.describe()}"
    if not isinstance(entry, Fact):
        return None, None, f"{spec.label} carries no provenance"

    if cond.op in _NUMERIC_OPS or cond.op == "between":
        if not entry.is_numeric:
            return None, None, f"{spec.label} is not a number here ({entry.render()})"
        if spec.categorical:
            return None, None, f"{spec.label} is a category, and the rule compares it as a number"
        value, problem = _convert(entry, spec)
        if value is None:
            return None, None, problem or f"{spec.label} is in the wrong unit"
        try:
            if cond.op == "between":
                low, high = float(cond.value[0]), float(cond.value[1])
                verdict = low <= value <= high
                margin = 1.0 if verdict else 0.0
                threshold_text = f"between {low:g} and {high:g}"
            else:
                threshold = float(cond.value)
                verdict = {
                    ">=": value >= threshold,
                    "<=": value <= threshold,
                    ">": value > threshold,
                    "<": value < threshold,
                }[cond.op]
                margin = _margin(cond.op, value, threshold)
                words = {">=": "at or above", "<=": "at or below", ">": "above", "<": "below"}
                threshold_text = f"{words[cond.op]} {threshold:g}"
        except (TypeError, ValueError, IndexError):
            return None, None, f"the rule's threshold for {spec.label} is not a number ({cond.value!r})"
        if not math.isfinite(value):
            return None, None, f"{spec.label} came back non-finite"
        if not verdict:
            return False, None, ""
        text = f"{spec.label} {entry.render()} is {threshold_text}"
        return True, Trigger(cond, entry, value, margin, text), ""

    text_value = str(entry.value).strip()
    if cond.op == "==":
        verdict = text_value.lower() == str(cond.value).strip().lower()
    elif cond.op == "in":
        options = cond.value if isinstance(cond.value, (list, tuple)) else [cond.value]
        verdict = text_value.lower() in {str(v).strip().lower() for v in options}
    elif cond.op == "contains":
        verdict = str(cond.value).strip().lower() in text_value.lower()
    else:
        return None, None, f"the rule uses an operator this build does not implement ({cond.op})"
    if not verdict:
        return False, None, ""
    text = f"{spec.label} is {entry.render()}"
    return True, Trigger(cond, entry, text_value, 1.0, text), ""


def run_rules(
    evidence: Ledger | Mapping[str, Evidence],
    *,
    crop: str | None = None,
    cap: int | None = None,
    categories: Sequence[str] | None = None,
    rules: Sequence[Rule] | None = None,
    settings: Settings | None = None,
    ledger: Ledger | None = None,
) -> RiskAssessment:
    """Evaluate the disease library against one turn's measurements.

    ``evidence`` is a :class:`~cropup.evidence.Ledger` or a mapping of quantity
    name to ``Fact``/``Missing``; no path accepts a bare number. ``categories``
    restricts the library to a subset (irrigation only wants the water ones).

    Passing ``ledger`` records a gap for each rule field that nothing measured,
    so ``/api/capabilities`` shows that the risk list was computed with part of
    the library undecided.
    """
    settings = settings or get_settings()
    cap = int(cap if cap is not None else settings.risk_cap)
    pool = load_rules(settings=settings) if rules is None else tuple(rules)
    wanted_categories = {c.strip().lower() for c in categories} if categories else None

    observations = _observations(evidence)
    applicable = rules_for_crop(crop, pool)
    if wanted_categories is not None:
        applicable = tuple(r for r in applicable if r.category.strip().lower() in wanted_categories)
    crop_rules_skipped = 0
    if crop is None or not str(crop).strip():
        crop_rules_skipped = sum(
            1
            for r in pool
            if not r.is_generic and (wanted_categories is None or r.category.strip().lower() in wanted_categories)
        )

    fired: list[Risk] = []
    undecided: list[Unevaluated] = []
    fields_used: list[str] = []
    fields_missing: dict[str, str] = {}
    decided = 0

    for rule in applicable:
        verdicts: list[bool | None] = []
        triggers: list[Trigger] = []
        reasons: list[str] = []
        waiting: list[str] = []
        for cond in rule.conditions:
            verdict, trigger, reason = _evaluate(cond, observations)
            verdicts.append(verdict)
            if trigger is not None:
                triggers.append(trigger)
                if cond.field not in fields_used:
                    fields_used.append(cond.field)
            if verdict is None:
                if cond.field not in waiting:
                    waiting.append(cond.field)
                if reason and reason not in reasons:
                    reasons.append(reason)
                fields_missing.setdefault(cond.field, reason)

        if rule.logic == "OR":
            outcome: bool | None
            if any(v is True for v in verdicts):
                outcome = True
            elif any(v is None for v in verdicts):
                outcome = None
            else:
                outcome = False
        else:  # AND, and anything the CSV spells differently
            if any(v is False for v in verdicts):
                outcome = False
            elif any(v is None for v in verdicts):
                outcome = None
            else:
                outcome = True

        if outcome is None:
            undecided.append(
                Unevaluated(
                    issue_id=rule.issue_id,
                    name=rule.name,
                    category=rule.category,
                    severity=rule.severity,
                    crop=rule.crop,
                    waiting_for=tuple(waiting),
                    reasons=tuple(reasons),
                )
            )
            continue

        decided += 1
        if outcome is False:
            continue
        fired.append(
            Risk(
                issue_id=rule.issue_id,
                name=rule.name,
                category=rule.category,
                severity=rule.severity,
                urgency=rule.urgency,
                causal_agent=rule.causal_agent,
                notes=rule.notes,
                recommendation=rule.recommendation,
                crop=rule.crop,
                crop_specific=not rule.is_generic,
                triggers=tuple(triggers),
            )
        )

    # Deduplicate by issue_id, keeping the best-supported reading of each.
    by_id: dict[str, Risk] = {}
    for risk in fired:
        current = by_id.get(risk.issue_id)
        if current is None or risk.sort_key() < current.sort_key():
            by_id[risk.issue_id] = risk
    ranked = sorted(by_id.values(), key=Risk.sort_key)

    if ledger is not None:
        for name, reason in fields_missing.items():
            spec = FIELD_SPECS.get(name)
            quantity = spec.quantity if spec else name
            if quantity in observations:
                continue  # the leaf already recorded why; do not double-count it
            ledger.gap(
                quantity,
                MissingReason.NOT_REQUESTED,
                (),
                f"needed by the disease library: {reason}",
            )

    return RiskAssessment(
        crop=crop,
        risks=tuple(ranked[:cap]),
        suppressed=tuple(ranked[cap:]),
        unevaluated=tuple(undecided),
        considered=len(applicable),
        decided=decided,
        fired=len(ranked),
        cap=cap,
        crop_rules_skipped=crop_rules_skipped,
        fields_used=tuple(fields_used),
        fields_missing=fields_missing,
        categories=tuple(categories or ()),
    )
