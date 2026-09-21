"""The citable knowledge cards (SPEC section 6).

Four kinds of card, all built from files that are committed in this repo:

``disease``
    One card per rule in ``cropup/data/disease_library.csv`` (384 rules,
    102 crops): the observable signs, the numeric thresholds that fire the
    rule, the recommendation and its urgency.
``practice``
    The decision branches of the vendored ``crop_recommendations.py`` and
    ``soil_recommendations.py``, rewritten as documents with their conditions
    stated. The agronomy is the reference implementation's; what has been
    removed is the interpolation of *measured* values into the prose, because
    the vendored code filled those with silent defaults (pH 7.0, sand 30%) --
    bug 4 of SPEC section 2.1. A card states the threshold; the measured value
    comes from a :class:`cropup.evidence.Fact` or it is not shown at all.
``dataset``
    One card per entry in ``cropup/data/ee_registry.json`` (92 datasets), so
    "where does this number come from" is answerable without leaving the app.
``topic``
    An index over the other cards: every verbatim mention of a named practice
    (mulching, composting, liming, ...) with a citation back to the rule it
    came from. These cards define nothing; they quote.

Nothing here paraphrases an agronomic claim, and nothing here is authored
agronomy: every sentence is either quoted from a committed data file, ported
from the vendored logic, or literal scaffolding such as "Recommendation:".

Imports ``config`` and ``errors`` only (SPEC section 2 import rule).
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from ..config import Settings, get_settings
from ..errors import DataFileError

__all__ = [
    "KIND_DISEASE",
    "KIND_PRACTICE",
    "KIND_DATASET",
    "KIND_TOPIC",
    "CARDS_FILE",
    "Condition",
    "Card",
    "build_cards",
    "build_disease_cards",
    "build_practice_cards",
    "build_dataset_cards",
    "build_topic_cards",
    "write_corpus",
    "load_cards",
    "cards_path",
    "corpus_digest",
]

KIND_DISEASE = "disease"
KIND_PRACTICE = "practice"
KIND_DATASET = "dataset"
KIND_TOPIC = "topic"

CARDS_FILE = "cards.jsonl"

# The 12 intents of SPEC section 5.2. Cards declare the ones they can serve so
# retrieval can prefer, e.g., an irrigation card for an irrigation question.
INTENTS = (
    "field_health_check",
    "crop_problem_diagnosis",
    "irrigation_advice",
    "crop_selection",
    "fertilizer_advice",
    "soil_fertility_management",
    "seed_variety_selection",
    "crop_management_practice",
    "market_and_inputs_supply",
    "livestock_and_adjacent",
    "agronomy_concept_explainer",
    "out_of_scope_or_unclear",
)

_DISEASE_CITATION = "CropUp disease library (cropup/data/disease_library.csv), rule {issue_id}"
_REGISTRY_CITATION = "CropUp Earth Engine registry (cropup/data/ee_registry.json), entry {asset_id}"

# Values that mean "this column was left blank" rather than "no constraint".
_BLANK = frozenset({"", "none", "n/a", "na", "-"})


def _clean(value: Any) -> str:
    """Trim a CSV cell; return "" for the library's several ways of saying blank."""
    text = ("" if value is None else str(value)).strip()
    return "" if text.lower() in _BLANK else text


def _clean_constraint(value: Any) -> str:
    """Like :func:`_clean` but also drops "Any", which is not a constraint."""
    text = _clean(value)
    return "" if text.lower() == "any" else text


def _number(value: Any) -> float | None:
    text = _clean(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _trim_number(value: float) -> str:
    return f"{value:g}"


# --------------------------------------------------------------------------
# cards
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """One numeric or categorical condition that a card's advice depends on.

    Kept structured rather than baked into the prose so the caller can show
    *why* a card was offered next to the Facts it actually measured.
    """

    field: str
    op: str
    value: Any
    unit: str = ""

    def render(self) -> str:
        if isinstance(self.value, (list, tuple)):
            shown = ", ".join(str(v) for v in self.value)
            shown = f"[{shown}]"
        elif isinstance(self.value, float):
            shown = _trim_number(self.value)
        else:
            shown = str(self.value)
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.field} {self.op} {shown}{unit}"

    def as_dict(self) -> dict[str, Any]:
        return {"field": self.field, "op": self.op, "value": self.value, "unit": self.unit}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Condition":
        return cls(
            field=str(payload["field"]),
            op=str(payload["op"]),
            value=payload.get("value"),
            unit=str(payload.get("unit") or ""),
        )


@dataclass(frozen=True)
class Card:
    """One retrievable, citable document.

    ``body`` is what the farmer is shown -- verbatim, never summarised.
    ``embed_text`` is what the index embeds; it is a shorter view of the same
    card so that a 3,000-character dataset card is not truncated down to its
    licence line by the encoder's 256-token window.
    """

    card_id: str
    kind: str
    title: str
    body: str
    source: str
    embed_text: str = ""
    crops: tuple[str, ...] = ()
    intents: tuple[str, ...] = ()
    conditions: tuple[Condition, ...] = ()
    keywords: tuple[str, ...] = ()
    category: str | None = None
    severity: str | None = None
    urgency: str | None = None
    origin: str | None = None

    def __post_init__(self) -> None:
        for name in ("card_id", "kind", "title", "body", "source"):
            if not str(getattr(self, name) or "").strip():
                raise ValueError(f"card {self.card_id!r}: {name} must not be empty")
        unknown = [i for i in self.intents if i not in INTENTS]
        if unknown:
            raise ValueError(f"card {self.card_id!r}: unknown intents {unknown}")
        if not self.embed_text.strip():
            object.__setattr__(self, "embed_text", f"{self.title}\n{self.body}")

    @property
    def citation(self) -> str:
        return self.source

    def matches_crop(self, crop: str | None) -> bool:
        """A card with no crops is general advice and matches every crop."""
        if not crop or not self.crops:
            return True
        return crop.strip().lower() in self.crops

    def matches_intent(self, intent: str | None) -> bool:
        if not intent or not self.intents:
            return True
        return intent in self.intents

    def as_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "kind": self.kind,
            "title": self.title,
            "body": self.body,
            "source": self.source,
            "embed_text": self.embed_text,
            "crops": list(self.crops),
            "intents": list(self.intents),
            "conditions": [c.as_dict() for c in self.conditions],
            "keywords": list(self.keywords),
            "category": self.category,
            "severity": self.severity,
            "urgency": self.urgency,
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Card":
        return cls(
            card_id=payload["card_id"],
            kind=payload["kind"],
            title=payload["title"],
            body=payload["body"],
            source=payload["source"],
            embed_text=payload.get("embed_text", ""),
            crops=tuple(payload.get("crops") or ()),
            intents=tuple(payload.get("intents") or ()),
            conditions=tuple(Condition.from_dict(c) for c in payload.get("conditions") or ()),
            keywords=tuple(payload.get("keywords") or ()),
            category=payload.get("category"),
            severity=payload.get("severity"),
            urgency=payload.get("urgency"),
            origin=payload.get("origin"),
        )

    def __repr__(self) -> str:  # keeps test failures readable
        return f"Card({self.card_id!r}, {self.kind}, {self.title!r})"


# --------------------------------------------------------------------------
# disease cards
# --------------------------------------------------------------------------

# A "Perfect Conditions" row is not a problem report: it records where the crop
# grows well, so it carries none of the problem intents.
_IDEAL_CATEGORY = "Perfect Conditions"
_IDEAL_INTENTS = ("crop_selection", "seed_variety_selection", "field_health_check")

# Every other rule answers "what is wrong with my crop" and "what do I do".
_PROBLEM_INTENTS = ("crop_problem_diagnosis", "crop_management_practice")

# Category component -> the intents the *subject* of the rule adds to
# _PROBLEM_INTENTS. Keyed by component rather than by whole category so the
# library's compound categories ("Fungal/Oomycete", "Abiotic/Fungal",
# "Viral+Pest") resolve to the union of their parts. Every component the
# library uses is listed, and build_disease_cards refuses to build when one is
# not: a silent default is how 292 of the 384 rules -- every fungal, oomycete,
# bacterial, viral, pest, nematode and phytoplasma rule -- became unreachable
# under any intent but those two.
_CATEGORY_INTENTS: dict[str, tuple[str, ...]] = {
    # Biotic agents. Naming the organism says nothing about which advice the
    # rule gives, so these add nothing here; the widening comes from what the
    # rule recommends (_RECOMMENDATION_INTENTS) and from whether it is visible
    # from orbit (_SATELLITE_COLUMNS).
    "Fungal": (),
    "Oomycete": (),
    "Protist": (),
    "Bacterial": (),
    "Viral": (),
    "Phytoplasma": (),
    "Pest": (),
    "Nematode": (),
    # Abiotic categories, where the subject of the rule is itself the answer to
    # a nutrient, soil or water question.
    "Deficiency": ("fertilizer_advice", "soil_fertility_management"),
    "Toxicity": ("soil_fertility_management",),
    "Salinity": ("soil_fertility_management", "irrigation_advice"),
    "Drought": ("irrigation_advice",),
    "Waterlogging": ("irrigation_advice",),
    "Heat": (),
    "Cold": (),
    "Abiotic": (),
    "Physiological": (),
}

# What a rule *recommends* is the evidence for which advice question it can
# answer, and a much better one than its category: 27 of the 29 viral rules
# name a resistant variety but only 5 of the 152 fungal ones do, while 42 of
# the 44 oomycete rules are about drainage. Matched against the recommendation
# column alone -- the action, not the symptom description -- so a rule earns an
# intent only by prescribing that kind of action.
_RECOMMENDATION_INTENTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        (
            "resistant variet", "tolerant variet", "resistant hybrid", "resistant cultivar",
            "resistant rootstock", "certified seed", "certified disease-free",
            "certified virus-free", "virus-free planting", "clean planting material",
            "indexed clean planting", "clean seed", "healthy seed", "seed treatment",
            "treated seed",
        ),
        ("seed_variety_selection",),
    ),
    (
        (
            "fertiliser", "fertilizer", "urea", " npk", " dap", "top dress", "topdress",
            "nitrogen", "potassium", "phosphate", "boron", "micronutrient", "kieserite",
            "zinc sulphate", "zinc sulfate", "magnesium sulphate", "manganese sulphate",
            "ammonium sulphate", "copper sulphate foliar", "calcium nitrate",
            "sodium molybdate", "chelated iron", "chelated micronutrient",
            "elemental sulphur",
        ),
        ("fertilizer_advice", "soil_fertility_management"),
    ),
    (
        (
            "lime", "liming", "gypsum", "apply compost", "compost at", "manure",
            "organic matter", "soil ph", "raise ph", "lower soil ph", "acidify soil",
            "leach",
        ),
        ("soil_fertility_management",),
    ),
    (("irrigat", "drain", "waterlog", "flood"), ("irrigation_advice",)),
)

# What a field-health question actually looks at. A rule that declares an
# NDVI/NDMI/NDRE signature or threshold can answer one; a rule that declares
# none cannot, and does not claim to.
_SATELLITE_COLUMNS = (
    "ndvi_signal",
    "ndmi_signal",
    "ndre_signal",
    "ndvi_threshold",
    "ndmi_threshold",
    "ndre_threshold",
)

_CATEGORY_SPLIT = re.compile(r"[/+]")


def _category_components(category: str) -> tuple[str, ...]:
    """Split a compound category ("Fungal/Oomycete", "Viral+Pest") into its parts."""
    return tuple(part.strip() for part in _CATEGORY_SPLIT.split(category) if part.strip())


def _unmapped_components(category: str) -> tuple[str, ...]:
    """Components of ``category`` that :data:`_CATEGORY_INTENTS` does not map."""
    if _clean(category) == _IDEAL_CATEGORY:
        return ()
    return tuple(c for c in _category_components(category) if c not in _CATEGORY_INTENTS)


def _disease_intents(row: dict[str, str]) -> tuple[str, ...]:
    """The SPEC section 5.2 intents one rule can answer, in a stable order.

    Three sources, each a column of the row itself: the category says what kind
    of problem it is, the recommendation says what kind of action it
    prescribes, and the satellite columns say whether it is visible from orbit.
    A rule never earns an intent that no column of it supports.
    """
    category = _clean(row.get("category"))
    if category == _IDEAL_CATEGORY:
        return _IDEAL_INTENTS

    out = list(_PROBLEM_INTENTS)
    for component in _category_components(category):
        out.extend(_CATEGORY_INTENTS.get(component, ()))

    recommendation = _clean(row.get("recommendation")).lower()
    for terms, intents in _RECOMMENDATION_INTENTS:
        if any(term in recommendation for term in terms):
            out.extend(intents)

    if any(_clean(row.get(column)) for column in _SATELLITE_COLUMNS):
        out.append("field_health_check")

    return tuple(dict.fromkeys(out))


def _range_phrase(low: float | None, high: float | None, opt: float | None, unit: str) -> str:
    """Render whichever of min/optimum/max the library actually supplies.

    ``unit`` is appended with no separator, so the caller decides between "%"
    and " degC".
    """
    if low is not None and high is not None:
        text = f"{_trim_number(low)}-{_trim_number(high)}{unit}"
    elif low is not None:
        text = f"{_trim_number(low)}{unit} or above"
    elif high is not None:
        text = f"up to {_trim_number(high)}{unit}"
    else:
        text = ""
    if opt is not None:
        opt_text = f"optimum {_trim_number(opt)}{unit}"
        text = f"{text} ({opt_text})" if text else opt_text
    return text


# Each ``perfect_*`` quantity, paired with the columns on the same row that
# record the conditions *favouring the problem*, plus how it is rendered.
#
# The pairing is the point. On 218 of the 384 rules the library fills a
# ``perfect_*`` cell with a byte-for-byte copy of its trigger cell -- 213 soil
# pH ranges and 72 drainage classes -- so labelling that cell "this crop's
# ideal range" told the farmer that the conditions which cause the disease are
# the conditions to aim for, "Poor drainage" on 47 rules and "Flooded" on 7.
# Where the optimum only restates the trigger the library records no optimum,
# and SPEC section 4 forbids presenting a value the data does not carry, so the
# quantity is dropped. The trigger itself stays on the card under "Conditions
# that favour it", labelled as what it is.
_IDEAL_QUANTITIES: tuple[tuple[tuple[str, ...], tuple[str, ...], str, str], ...] = (
    (
        ("perfect_temp_min_c", "perfect_temp_max_c"),
        ("temp_min_c", "temp_max_c"),
        " degC",
        "temperature {}",
    ),
    (
        ("perfect_rh_min_pct", "perfect_rh_max_pct"),
        ("rh_min_pct", "rh_opt_pct"),
        "%",
        "relative humidity {}",
    ),
    (
        ("perfect_soil_ph_min", "perfect_soil_ph_max"),
        ("soil_ph_min", "soil_ph_max"),
        "",
        "soil pH {}",
    ),
    (("perfect_soil_drainage",), ("soil_drainage",), "", "{} drainage"),
    (("perfect_rainfall_mm_week",), ("rain_trigger",), "", "rainfall {} mm/week"),
)

# Named, not defaulted: a card that states no ideal range says so, so it cannot
# be read as a crop for which every condition is acceptable.
_NO_IDEAL_RANGE = (
    "This crop's ideal range: the disease library records none for it apart from this "
    "rule's own thresholds, so none is stated here."
)


def _same_cell(left: Any, right: Any) -> bool:
    """True when two cells state the same thing; numbers compare as numbers."""
    first, second = _clean_constraint(left), _clean_constraint(right)
    if not first or not second:
        return False
    first_number, second_number = _number(first), _number(second)
    if first_number is not None and second_number is not None:
        return first_number == second_number
    return first.casefold() == second.casefold()


def _restates_trigger(
    row: dict[str, str], ideal_columns: tuple[str, ...], trigger_columns: tuple[str, ...]
) -> bool:
    """True when the filled ``perfect_*`` cells only repeat the rule's triggers."""
    filled = [
        (ideal, trigger)
        for ideal, trigger in zip(ideal_columns, trigger_columns)
        if _clean_constraint(row.get(ideal))
    ]
    if not filled:
        return False
    return all(_same_cell(row.get(ideal), row.get(trigger)) for ideal, trigger in filled)


def _ideal_conditions(row: dict[str, str], *, on_problem_rule: bool) -> list[str]:
    """The crop's ideal envelope, from the row's ``perfect_*`` columns.

    On a problem rule a quantity whose ``perfect_*`` cells only restate the
    rule's own trigger cells is left out -- see :data:`_IDEAL_QUANTITIES`. On a
    "Perfect Conditions" row there is no trigger to restate: those rows fill
    the ``perfect_*`` columns and leave every trigger column blank.
    """
    out: list[str] = []
    for ideal_columns, trigger_columns, unit, template in _IDEAL_QUANTITIES:
        if on_problem_rule and _restates_trigger(row, ideal_columns, trigger_columns):
            continue
        if len(ideal_columns) == 2:
            low, high = (_number(row.get(column)) for column in ideal_columns)
            text = _range_phrase(low, high, None, unit)
        else:
            text = _clean_constraint(row.get(ideal_columns[0]))
        if text:
            out.append(template.format(text))
    return out


def _disease_conditions(row: dict[str, str]) -> tuple[Condition, ...]:
    """The machine-checkable trigger, taken from the row's ``rule_logic`` JSON."""
    raw = _clean(row.get("rule_logic"))
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return ()
    out: list[Condition] = []
    for item in parsed.get("conditions") or ():
        if not isinstance(item, dict) or "field" not in item or "op" not in item:
            continue
        out.append(Condition(str(item["field"]), str(item["op"]), item.get("value")))
    return tuple(out)


def _disease_body(row: dict[str, str]) -> tuple[str, str]:
    """Return (body, embed_text) for one disease-library rule."""
    crop = _clean(row["crop"]) or "Generic"
    issue = _clean(row["issue_name"])
    category = _clean(row["category"])
    agent = _clean(row["causal_agent"])
    notes = _clean(row["notes"])
    recommendation = _clean(row["recommendation"])
    urgency = _clean(row["urgency"])
    severity = _clean(row["severity"])

    # A "Perfect Conditions" row is not a problem: it records the crop's ideal
    # envelope, and its causal_agent/notes columns both read "Optimal
    # environment", which would make a nonsense card if treated as symptoms.
    is_ideal = category == "Perfect Conditions"

    if is_ideal:
        lines = [f"Crop: {crop}", "This card records where the crop grows best; it is not a problem report."]
    else:
        lines = [f"Crop: {crop}", f"Issue: {issue}" + (f" ({category})" if category else "")]
        if agent:
            lines.append(f"Causal agent: {agent}")
        if notes:
            lines.append(f"Field signs and context: {notes}")
    stage = _clean_constraint(row.get("growth_stage"))
    if stage:
        lines.append(f"Growth stage at risk: {stage}")

    favouring: list[str] = []
    temp = _range_phrase(
        _number(row.get("temp_min_c")), _number(row.get("temp_max_c")), _number(row.get("temp_opt_c")), " degC"
    )
    if temp:
        favouring.append(f"temperature {temp}")
    rh = _range_phrase(_number(row.get("rh_min_pct")), None, _number(row.get("rh_opt_pct")), "%")
    if rh:
        favouring.append(f"relative humidity {rh}")
    wetness = _number(row.get("leaf_wetness_hrs"))
    if wetness is not None:
        favouring.append(f"leaf wetness {_trim_number(wetness)} h or more")
    rain = _clean_constraint(row.get("rain_trigger"))
    if rain:
        favouring.append(f"rainfall {rain}")
    ph = _range_phrase(_number(row.get("soil_ph_min")), _number(row.get("soil_ph_max")), None, "")
    if ph:
        favouring.append(f"soil pH {ph}")
    drainage = _clean_constraint(row.get("soil_drainage"))
    if drainage:
        favouring.append(f"{drainage} drainage")
    texture = _clean_constraint(row.get("soil_texture"))
    if texture:
        favouring.append(f"{texture} texture")
    if favouring:
        lines.append("Conditions that favour it: " + "; ".join(favouring) + ".")

    signals = [
        _clean_constraint(row.get("ndvi_signal")),
        _clean_constraint(row.get("ndmi_signal")),
        _clean_constraint(row.get("ndre_signal")),
    ]
    signals = [s for s in signals if s]
    if signals:
        lines.append("Satellite signature: " + "; ".join(signals) + ".")

    triggers: list[str] = []
    for column, phrase in (
        ("consecutive_dry_days_trigger", "{v} consecutive dry days"),
        ("consecutive_wet_days_trigger", "{v} consecutive wet days"),
        ("temp_stress_threshold_c", "air temperature past {v} degC"),
        ("soil_ec_threshold_ds_m", "soil EC past {v} dS/m"),
        ("ndvi_threshold", "NDVI change of {v}"),
        ("ndmi_threshold", "NDMI change of {v}"),
        ("ndre_threshold", "NDRE change of {v}"),
    ):
        value = _number(row.get(column))
        if value is not None:
            triggers.append(phrase.format(v=_trim_number(value)))
    if triggers:
        lines.append("Numeric triggers: " + "; ".join(triggers) + ".")

    ideal = _ideal_conditions(row, on_problem_rule=not is_ideal)
    if is_ideal and ideal:
        lines.append("Ideal growing conditions: " + "; ".join(ideal) + ".")
    elif is_ideal:
        lines.append("Ideal growing conditions: the disease library records none for this crop.")
    elif ideal:
        lines.append("This crop's ideal range, for comparison: " + "; ".join(ideal) + ".")
    else:
        lines.append(_NO_IDEAL_RANGE)

    if severity:
        lines.append(f"Severity: {severity}.")
    if recommendation:
        lines.append(f"Recommendation: {recommendation.rstrip('. ')}.")
    if urgency:
        lines.append(f"Act: {urgency}.")

    body = "\n".join(lines)

    # The encoder sees a sentence-shaped view: the words a farmer would use
    # (crop, symptom, action) rather than the tabular thresholds.
    if is_ideal:
        embed_parts = [f"Ideal growing conditions for {crop}"]
        if ideal:
            embed_parts.append("; ".join(ideal))
        embed_parts.append(f"Is {crop} suited to this place")
    else:
        embed_parts = [f"{crop} {issue}"]
        if category:
            embed_parts.append(f"a {category.lower()} problem")
        if agent:
            embed_parts.append(f"caused by {agent}")
        if notes:
            embed_parts.append(notes)
        if signals:
            embed_parts.append("; ".join(signals))
    if recommendation:
        embed_parts.append(f"What to do: {recommendation}")
    return body, ". ".join(p.rstrip(". ") for p in embed_parts)


def build_disease_cards(path: Path) -> list[Card]:
    """One card per row of the disease library."""
    if not path.is_file():
        raise DataFileError(path, "disease library is required to build the corpus")
    cards: list[Card] = []
    seen: set[str] = set()
    unmapped: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            issue_id = _clean(row.get("issue_id"))
            if not issue_id:
                continue
            if issue_id in seen:
                raise DataFileError(path, f"duplicate issue_id {issue_id!r}; card ids would collide")
            seen.add(issue_id)
            crop = _clean(row["crop"]) or "Generic"
            category = _clean(row["category"])
            unmapped.update(_unmapped_components(category))
            body, embed_text = _disease_body(row)
            title = f"{crop}: {_clean(row['issue_name'])}"
            crops = () if crop.lower() == "generic" else (crop.lower(),)
            cards.append(
                Card(
                    card_id=f"{KIND_DISEASE}:{issue_id}",
                    kind=KIND_DISEASE,
                    title=title,
                    body=body,
                    source=_DISEASE_CITATION.format(issue_id=issue_id),
                    embed_text=embed_text,
                    crops=crops,
                    intents=_disease_intents(row),
                    conditions=_disease_conditions(row),
                    keywords=tuple(k for k in (category.lower(), crop.lower()) if k),
                    category=category or None,
                    severity=_clean(row.get("severity")) or None,
                    urgency=_clean(row.get("urgency")) or None,
                    origin="cropup/data/disease_library.csv",
                )
            )
    if unmapped:
        raise DataFileError(
            path,
            "unmapped rule category component(s) "
            + ", ".join(sorted(repr(c) for c in unmapped))
            + "; add each to _CATEGORY_INTENTS, because a rule that falls back to "
            f"{_PROBLEM_INTENTS} alone cannot be retrieved under any other intent",
        )
    return cards


# --------------------------------------------------------------------------
# practice cards
# --------------------------------------------------------------------------

_CROP_REC_SOURCE = (
    "CropUp practice card, ported from the decision branches of "
    "crop_recommendations.py::{func} in the vendored reference implementation"
)
_SOIL_REC_SOURCE = (
    "CropUp practice card, ported from the decision branches of "
    "soil_recommendations.py::get_soil_informed_recommendations in the vendored "
    "reference implementation"
)

# Each entry is one branch of the vendored code. ``conditions`` is that
# branch's guard, verbatim from the source; ``body`` is its prose with the
# interpolated measurements taken out (SPEC 2.1 bug 4) and nothing added.
_PRACTICE_CARDS: tuple[dict[str, Any], ...] = (
    {
        "id": "aridification_drought_tolerant_crops",
        "title": "Transition toward drought-tolerant crops before aridification bites",
        "horizon": "5-15 years",
        "conditions": [("climate_shift_direction", "==", "aridification", "")],
        "source": _CROP_REC_SOURCE.format(func="get_future_crop_recommendations"),
        "intents": ("crop_selection", "seed_variety_selection", "crop_management_practice"),
        "keywords": ("drought", "aridification", "variety", "crop switch"),
        "body": (
            "Horizon: 5-15 years.\n"
            "Where the climate is projected to shift toward drier conditions, begin introducing "
            "drought-tolerant varieties or crops now rather than waiting for the shift to complete.\n"
            "Crop by crop:\n"
            "- Maize: drought-tolerant maize hybrids (DT maize), or replace with sorghum in the driest years.\n"
            "- Wheat: CIMMYT drought-tolerant wheat varieties, or dual-purpose barley.\n"
            "- Rice: aerobic or upland rice varieties; reduce paddy area in favour of sorghum or cowpea.\n"
            "- Potato: move potato to cooler and wetter seasons only; introduce sweet potato as a "
            "dry-season alternative.\n"
            "- Beans: cowpea or tepary bean, which tolerate 30% less rainfall than common bean.\n"
            "- Coffee: shade-grown Robusta or Liberica; Arabica will face severe heat stress.\n"
            "- Banana: plantain varieties with higher drought tolerance; reduce overall area.\n"
            "- Any other crop: explore sorghum, millet, cowpea or cassava as more resilient alternatives."
        ),
    },
    {
        "id": "build_soil_water_storage",
        "title": "Build soil water storage before aridification accelerates",
        "conditions": [("climate_shift_direction", "==", "aridification", "")],
        "source": _CROP_REC_SOURCE.format(func="get_future_crop_recommendations"),
        "intents": ("soil_fertility_management", "crop_management_practice", "agronomy_concept_explainer"),
        "keywords": ("compost", "cover crop", "residue", "organic matter", "water holding capacity"),
        "body": (
            "Horizon: now.\n"
            "Building soil water storage is the single highest-return investment for drought resilience. "
            "Every 1% increase in soil organic matter adds about 20,000 L/ha of water holding capacity.\n"
            "Apply compost at 3-5 t/ha annually, retain all crop residues, and introduce a legume cover "
            "crop in the off-season.\n"
            "Soil texture sets the starting point: a soil above about 55% sand holds limited water, "
            "loam and clay soils hold moderate amounts."
        ),
    },
    {
        "id": "water_harvesting_on_sand",
        "title": "Install water harvesting infrastructure on sandy soil",
        "conditions": [
            ("climate_shift_direction", "==", "aridification", ""),
            ("sand_pct", ">", 55, "%"),
        ],
        "source": _CROP_REC_SOURCE.format(func="get_future_crop_recommendations"),
        "intents": ("irrigation_advice", "crop_management_practice"),
        "keywords": ("water harvesting", "zai pits", "tied ridges", "drip"),
        "body": (
            "Horizon: 2-5 years.\n"
            "Sandy soil (above 55% sand) drains rapidly and will lose water faster under aridification. "
            "Consider tied ridges, half-moon catchments, or zai pits to capture rainfall.\n"
            "Micro-irrigation (drip or pitcher) will become economically necessary within 10 years at "
            "current climate trajectories."
        ),
    },
    {
        "id": "moistening_drainage_and_disease",
        "title": "Prepare drainage and disease management for wetter conditions",
        "conditions": [("climate_shift_direction", "==", "moistening", "")],
        "source": _CROP_REC_SOURCE.format(func="get_future_crop_recommendations"),
        "intents": ("crop_management_practice", "crop_problem_diagnosis", "seed_variety_selection"),
        "keywords": ("drainage", "raised beds", "disease pressure", "resistant varieties"),
        "body": (
            "Horizon: 5-15 years.\n"
            "A projected shift toward wetter conditions means fungal and bacterial disease pressure will "
            "increase significantly.\n"
            "Invest in field drainage now - raised beds or ridge-furrow systems. Select disease-resistant "
            "varieties.\n"
            "Crop by crop:\n"
            "- Maize: grey leaf spot and northern blight will become more frequent.\n"
            "- Wheat: Fusarium head blight and septoria are the primary risks.\n"
            "- Potato: late blight pressure will intensify - resistant varieties are critical.\n"
            "- Coffee: coffee leaf rust will spread to higher altitudes.\n"
            "- Banana: black Sigatoka will require more frequent fungicide cycles.\n"
            "- Any other crop: scout regularly for new fungal and bacterial diseases."
        ),
    },
    {
        "id": "continentalization_temperature_extremes",
        "title": "Adapt to hotter summers and colder winters",
        "conditions": [("climate_shift_direction", "==", "continentalization", "")],
        "source": _CROP_REC_SOURCE.format(func="get_future_crop_recommendations"),
        "intents": ("crop_selection", "seed_variety_selection"),
        "keywords": ("frost", "heat stress", "continental"),
        "body": (
            "Horizon: 5-15 years.\n"
            "A shift toward a more continental climate means more extreme temperatures in both "
            "directions. Select varieties with wide temperature tolerance.\n"
            "Winter frost risk increases - protect perennial crops. Summer heat stress at flowering will "
            "become a more frequent yield constraint.\n"
            "Barley, rye and winter wheat are better adapted to this trajectory than maize."
        ),
    },
    {
        "id": "warming_new_crop_opportunities",
        "title": "A warming climate opens new crop opportunities",
        "conditions": [("climate_shift_direction", "in", ["warming", "tropicalization"], "")],
        "source": _CROP_REC_SOURCE.format(func="get_future_crop_recommendations"),
        "intents": ("crop_selection", "crop_problem_diagnosis"),
        "keywords": ("warming", "pest pressure", "new crops"),
        "body": (
            "Horizon: 5-15 years.\n"
            "Under a shift toward warmer conditions by 2041-2070, crops currently limited by temperature "
            "(sugarcane, banana, cassava) become viable.\n"
            "Existing crops face higher pest and disease pressure year-round as winters become milder. "
            "Pollinators and beneficial insects will also shift - monitor carefully."
        ),
    },
    {
        "id": "koppen_zone_boundary_moving",
        "title": "A climate zone boundary is moving through the farm",
        "conditions": [("climate_shift_direction", "==", "zone_shift", "")],
        "source": _CROP_REC_SOURCE.format(func="get_future_crop_recommendations"),
        "intents": ("crop_selection", "agronomy_concept_explainer"),
        "keywords": ("koppen", "climate zone", "diversify"),
        "body": (
            "Horizon: 5-15 years.\n"
            "A location sitting near a Koppen-Geiger zone boundary will change zone under all emissions "
            "scenarios, including the low-emissions one.\n"
            "Diversify the crop mix now to hedge across possible futures."
        ),
    },
    {
        "id": "low_soc_climate_vulnerability",
        "title": "Low organic matter is the biggest climate vulnerability",
        "conditions": [
            ("soc_g_kg", "<", 8, "g/kg"),
            ("climate_shifting", "==", True, ""),
        ],
        "source": _CROP_REC_SOURCE.format(func="get_future_crop_recommendations"),
        "intents": ("soil_fertility_management", "crop_management_practice"),
        "keywords": ("soc", "organic matter", "compost", "cover crop", "minimum tillage"),
        "body": (
            "Horizon: now, urgent.\n"
            "Soil organic carbon below 8 g/kg gives a soil almost no buffer against climate extremes. "
            "Under any climate trajectory, low SOC means faster moisture loss, weaker nutrient cycling "
            "and lower yield stability.\n"
            "Target 15+ g/kg SOC through compost, cover crops and minimum tillage. This is the most "
            "cost-effective climate adaptation available."
        ),
    },
    {
        "id": "lime_before_aridification",
        "title": "Lime acid soil before aridification worsens aluminium toxicity",
        "conditions": [
            ("ph", "<", 5.5, "pH"),
            ("climate_shift_direction", "==", "aridification", ""),
        ],
        "source": _CROP_REC_SOURCE.format(func="get_future_crop_recommendations"),
        "intents": ("soil_fertility_management", "fertilizer_advice"),
        "keywords": ("lime", "aluminium", "acidity", "phosphorus availability"),
        "body": (
            "Horizon: this season.\n"
            "Acid soils below pH 5.5 concentrate toxic Al3+ and Mn2+ as they dry. Under aridification, "
            "drying-rewetting cycles intensify this effect.\n"
            "Apply agricultural lime at 2-3 t/ha now; target pH 6.0-6.5. This also improves phosphorus "
            "availability and reduces crop sensitivity to drought."
        ),
    },
    {
        "id": "clay_drainage_under_wetter_futures",
        "title": "Clay soil drainage is critical under any wetter scenario",
        "conditions": [
            ("clay_pct", ">", 40, "%"),
            ("climate_shift_direction", "in", ["moistening", "unknown"], ""),
        ],
        "source": _CROP_REC_SOURCE.format(func="get_future_crop_recommendations"),
        "intents": ("crop_management_practice", "irrigation_advice"),
        "keywords": ("clay", "drainage", "gypsum", "raised beds"),
        "body": (
            "Horizon: 2-5 years.\n"
            "Heavy clay (above 40% clay) will waterlog rapidly under increased rainfall. Install "
            "subsurface drainage or switch to raised-bed cropping systems.\n"
            "Gypsum at 1-2 t/ha improves clay structure and drainage capacity."
        ),
    },
    {
        "id": "crops_at_risk_under_aridification",
        "title": "Crops that may not stay viable under aridification",
        "conditions": [("climate_shift_direction", "==", "aridification", "")],
        "source": _CROP_REC_SOURCE.format(func="get_future_crop_recommendations"),
        "intents": ("crop_selection", "crop_management_practice"),
        "keywords": ("drought", "crop switch", "transition"),
        "body": (
            "Horizon: 10-20 years.\n"
            "These crops require reliable moisture that an aridifying location will increasingly lack: "
            "rice, potato, wheat, lettuce, spinach, strawberry, barley, oats, tea, coffee, apple, grape.\n"
            "Where one of them is the main crop, plan now for a 10-20 year transition: identify an "
            "alternative primary crop, diversify income streams, and grow the at-risk crop only in the "
            "wettest years or with full irrigation."
        ),
    },
    {
        "id": "crops_resilient_under_aridification",
        "title": "Crops that are relatively well-positioned under aridification",
        "conditions": [("climate_shift_direction", "==", "aridification", "")],
        "source": _CROP_REC_SOURCE.format(func="get_future_crop_recommendations"),
        "intents": ("crop_selection", "seed_variety_selection"),
        "keywords": ("drought tolerant", "sorghum", "millet", "cassava", "cowpea"),
        "body": (
            "Horizon: 10-20 years.\n"
            "Sorghum, millet, cassava, cowpea, groundnut, cotton, sesame, pigeon pea and okra have "
            "moderate to good drought tolerance.\n"
            "For a crop in this group, focus on variety selection within the crop rather than crop "
            "substitution - drought-tolerant varieties can maintain 70-80% of yield under 30% less "
            "rainfall."
        ),
    },
    {
        "id": "koppen_climate_envelopes",
        "title": "Temperature and rainfall envelopes of the Koppen-Geiger zones",
        "conditions": [],
        "source": _CROP_REC_SOURCE.format(func="_koppen_to_climate_envelope"),
        "intents": ("agronomy_concept_explainer", "crop_selection"),
        "keywords": ("koppen", "climate zone", "envelope", "rainfall"),
        "body": (
            "Each Koppen-Geiger code implies a coldest-month minimum temperature, a warmest-month "
            "maximum temperature and a weekly rainfall figure, given here as "
            "min degC / max degC / mm per week:\n"
            "Af 18/32/35, Am 18/32/22, Aw 18/36/8.\n"
            "BWh 18/42/2, BWk -5/28/2, BSh 18/38/5, BSk -5/32/5.\n"
            "Csa 2/34/8, Csb 2/24/10, Csc 2/18/10, Cwa 5/34/14, Cwb 5/24/14, Cwc 5/18/12, "
            "Cfa 2/34/20, Cfb 2/22/18, Cfc -3/18/18.\n"
            "Dsa -10/34/10, Dsb -10/22/10, Dsc -20/18/10, Dsd -30/18/10, Dwa -10/34/12, Dwb -10/22/12, "
            "Dwc -20/18/10, Dwd -30/18/8, Dfa -10/34/14, Dfb -15/22/14, Dfc -20/18/12, Dfd -30/18/10.\n"
            "ET -20/10/8, EF -40/0/4.\n"
            "A code outside this table has no envelope and must be treated as unknown rather than "
            "substituted with a default."
        ),
    },
    {
        "id": "climate_matched_crop_scoring",
        "title": "How a crop is scored against a climate and a soil",
        "conditions": [],
        "source": _CROP_REC_SOURCE.format(func="get_climate_matched_crops"),
        "intents": ("crop_selection", "agronomy_concept_explainer"),
        "keywords": ("crop matching", "score", "ideal conditions"),
        "body": (
            "A candidate crop is scored against the ideal conditions recorded for it in the disease "
            "library, weighted as follows: temperature fit 2 points, rainfall fit 2 points, soil pH fit "
            "2 points, humidity fit 1 point, drainage fit 1 point. The score is the points earned as a "
            "percentage of the points available.\n"
            "Temperature outside the crop's ideal band costs a further 1.5 points, in either direction.\n"
            "Rainfall is scored against a window from 0.45 to 1.9 times the crop's ideal weekly rainfall; "
            "below the bottom of that window the crop needs irrigation.\n"
            "Drainage scoring: a crop wanting flooded conditions scores fully on poorly drained soil; a "
            "crop wanting well-drained soil scores fully on well-drained soil and about 60% on moderately "
            "drained soil.\n"
            "Candidates below 40% are dropped outright; where the climate is not shifting the bar rises "
            "to 70%."
        ),
    },
    {
        "id": "sandy_soil_irrigate_frequently",
        "title": "Sandy soil with a moisture deficit: irrigate little and often",
        "conditions": [("sand_pct", ">", 65, "%"), ("ndmi", "<", 0.0, "index")],
        "source": _SOIL_REC_SOURCE,
        "intents": ("irrigation_advice", "crop_management_practice"),
        "keywords": ("irrigation", "sandy", "drip", "ndmi"),
        "body": (
            "Priority: high.\n"
            "Sandy soil (above 65% sand) has very low water retention capacity - it loses moisture 2-3 "
            "times faster than loam. An NDMI below 0.0 confirms an active moisture deficit.\n"
            "Apply smaller, more frequent irrigations (every 2-3 days) rather than large infrequent "
            "doses. Drip irrigation is strongly preferred over flood irrigation on this soil type."
        ),
    },
    {
        "id": "clay_soil_do_not_irrigate",
        "title": "Clay soil with elevated surface moisture: do not irrigate",
        "conditions": [("clay_pct", ">", 35, "%"), ("ndwi", ">", 0.0, "index")],
        "source": _SOIL_REC_SOURCE,
        "intents": ("irrigation_advice", "crop_problem_diagnosis"),
        "keywords": ("waterlogging", "clay", "drainage", "ndwi"),
        "body": (
            "Priority: low - the correct action is to wait.\n"
            "Clay-dominant soil (above 35% clay) drains very slowly. An NDWI above 0.0 indicates surface "
            "moisture is already elevated.\n"
            "Adding water now risks waterlogging, root anoxia and fungal disease. Allow the soil to dry "
            "before any irrigation. Check field drainage first."
        ),
    },
    {
        "id": "clay_loam_moderate_irrigation",
        "title": "Clay loam under mild moisture stress: one moderate irrigation",
        "conditions": [
            ("clay_pct", ">", 25, "%"),
            ("clay_pct", "<=", 40, "%"),
            ("ndmi", "<", 0.05, "index"),
        ],
        "source": _SOIL_REC_SOURCE,
        "intents": ("irrigation_advice",),
        "keywords": ("irrigation", "clay loam", "compaction"),
        "body": (
            "Priority: medium.\n"
            "Clay loam soil (25-40% clay) has good water retention. An NDMI below 0.05 suggests mild "
            "moisture stress.\n"
            "Apply a single moderate irrigation and wait 5-7 days before reassessing. Over-irrigation on "
            "this soil causes compaction and reduces aeration."
        ),
    },
    {
        "id": "loam_approaching_stress_threshold",
        "title": "Loam approaching its stress threshold: irrigate to field capacity",
        "conditions": [
            ("sand_pct", "<=", 65, "%"),
            ("clay_pct", "<=", 25, "%"),
            ("ndmi", "<", -0.05, "index"),
        ],
        "source": _SOIL_REC_SOURCE,
        "intents": ("irrigation_advice",),
        "keywords": ("irrigation", "loam", "et0", "dry days"),
        "body": (
            "Priority: medium.\n"
            "Loam and silt loam soils have balanced drainage - good water holding without waterlogging "
            "risk. An NDMI below -0.05 together with a run of consecutive dry days indicates a growing "
            "moisture deficit, and reference ET0 above the week's rainfall confirms that water demand "
            "exceeds supply.\n"
            "Irrigate to field capacity: within 2 days if more than 7 consecutive dry days have passed, "
            "otherwise within 4 days."
        ),
    },
    {
        "id": "no_irrigation_needed",
        "title": "Adequate canopy moisture: no irrigation needed",
        "conditions": [("ndmi", ">=", 0.15, "index")],
        "source": _SOIL_REC_SOURCE,
        "intents": ("irrigation_advice", "field_health_check"),
        "keywords": ("irrigation", "ndmi", "monitor"),
        "body": (
            "Priority: none.\n"
            "An NDMI of 0.15 or above indicates adequate canopy moisture; the soil is currently holding "
            "sufficient water. Monitor again in 5 days."
        ),
    },
    {
        "id": "low_soc_improve_organic_matter",
        "title": "Low soil organic carbon: build organic matter",
        "conditions": [("soc_g_kg", "<", 8, "g/kg")],
        "source": _SOIL_REC_SOURCE,
        "intents": ("soil_fertility_management", "fertilizer_advice", "crop_management_practice"),
        "keywords": ("compost", "soc", "organic matter", "residue", "cover crop"),
        "body": (
            "Priority: medium.\n"
            "Soil organic carbon below 8 g/kg is below the threshold for good soil health. On a sandy "
            "soil that significantly reduces water retention; on a heavier soil it mainly limits nutrient "
            "cycling.\n"
            "Incorporate crop residues rather than burning them, apply compost at 2-4 t/ha, or introduce "
            "a legume cover crop in the off-season.\n"
            "Even a 1% increase in soil organic matter can increase water holding capacity by about "
            "20,000 L/ha."
        ),
    },
    {
        "id": "acid_soil_apply_lime",
        "title": "Acid soil: apply lime to correct pH",
        "conditions": [("ph", "<", 5.5, "pH")],
        "source": _SOIL_REC_SOURCE,
        "intents": ("soil_fertility_management", "fertilizer_advice", "crop_problem_diagnosis"),
        "keywords": ("lime", "acidity", "aluminium", "ph"),
        "body": (
            "Priority: high below pH 5.0, medium between pH 5.0 and 5.5.\n"
            "A soil pH below 5.5 is under the optimal range (6.0-7.0) for most crops. At this pH, "
            "phosphorus, calcium and magnesium availability is reduced, and aluminium and manganese "
            "toxicity become risks.\n"
            "Apply agricultural lime at 1-3 t/ha depending on buffer pH. Re-test the soil after 6 months.\n"
            "Crop sensitivity to acidity:\n"
            "- Maize is moderately sensitive to acidity - yields drop significantly below pH 5.5.\n"
            "- Wheat is sensitive to aluminium toxicity at low pH; performance will be poor below 5.5.\n"
            "- Soybean rhizobium nitrogen fixation fails below pH 5.8; it is critical to lime before "
            "planting.\n"
            "- Potato tolerates mild acidity but scab risk increases above pH 5.5 - lime with caution.\n"
            "- Common beans are highly sensitive to acidity; liming is strongly recommended.\n"
            "- Flooded rice is more tolerant of acidity than upland crops; monitor, but less urgent.\n"
            "- Any other crop may experience reduced nutrient uptake at this pH."
        ),
    },
    {
        "id": "alkaline_soil_micronutrients",
        "title": "Alkaline soil: watch for micronutrient deficiency",
        "conditions": [("ph", ">", 7.8, "pH")],
        "source": _SOIL_REC_SOURCE,
        "intents": ("soil_fertility_management", "fertilizer_advice", "crop_problem_diagnosis"),
        "keywords": ("alkaline", "micronutrient", "iron", "zinc", "sulfur"),
        "body": (
            "Priority: medium.\n"
            "Above pH 7.8, iron, zinc, manganese and boron availability decreases sharply. If NDRE is "
            "low, this may be contributing to apparent nitrogen deficiency symptoms.\n"
            "Apply chelated micronutrients (EDTA-Fe, zinc sulfate), use ammonium-based fertilizers over "
            "nitrate-based ones, and consider elemental sulfur at 200-500 kg/ha to gradually lower pH."
        ),
    },
    {
        "id": "high_bulk_density_compaction",
        "title": "High bulk density: relieve compaction",
        "conditions": [("bulk_density", ">", 1.5, "g/cm3")],
        "source": _SOIL_REC_SOURCE,
        "intents": ("soil_fertility_management", "crop_management_practice"),
        "keywords": ("compaction", "bulk density", "tillage", "cover crop"),
        "body": (
            "Priority: medium.\n"
            "A bulk density above 1.5 g/cm3 indicates potential compaction. Compaction restricts root "
            "penetration, reduces water infiltration and limits nutrient uptake.\n"
            "Avoid heavy machinery on wet soil, subsoil till if compaction is confirmed, and plant "
            "deep-rooted cover crops (radish, sunflower) to break up compaction layers."
        ),
    },
    {
        "id": "low_cec_split_fertiliser",
        "title": "Low CEC on sandy soil: split the fertilizer applications",
        "conditions": [("sand_pct", ">", 60, "%"), ("cec_mmol_kg", "<", 10, "mmol/kg")],
        "source": _SOIL_REC_SOURCE,
        "intents": ("fertilizer_advice", "soil_fertility_management"),
        "keywords": ("fertilizer", "cec", "leaching", "fertigation", "slow release"),
        "body": (
            "Priority: medium.\n"
            "Sandy soil with a cation exchange capacity below 10 mmol/kg cannot hold nutrients "
            "effectively. Standard broadcast fertilizer applications will result in significant "
            "leaching, especially of nitrogen and potassium.\n"
            "Split all fertilizer applications into 3-4 smaller doses through the season. Fertigation "
            "(fertilizer through the irrigation water) is ideal for this soil type. Consider "
            "polymer-coated slow-release fertilizers."
        ),
    },
    {
        "id": "satellite_stress_linked_to_soil",
        "title": "Satellite stress linked to soil constraints",
        "conditions": [
            ("ndvi", "<", 0.4, "index"),
            ("soc_g_kg|bulk_density|ph", "any_of", "soc<8 g/kg, bulk_density>1.5 g/cm3, ph<5.5", ""),
        ],
        "source": _SOIL_REC_SOURCE,
        "intents": ("field_health_check", "crop_problem_diagnosis", "soil_fertility_management"),
        "keywords": ("ndvi", "stress", "soil constraint"),
        "body": (
            "Priority: high.\n"
            "An NDVI below 0.4 indicates a stressed crop. Where that coincides with low organic carbon "
            "(below 8 g/kg), high bulk density (above 1.5 g/cm3) or acidic pH (below 5.5), those soil "
            "conditions are likely contributing to or compounding the satellite-detected stress.\n"
            "Addressing the soil constraints will improve the crop's resilience to future stress events."
        ),
    },
)


def build_practice_cards() -> list[Card]:
    cards: list[Card] = []
    for entry in _PRACTICE_CARDS:
        conditions = tuple(Condition(f, op, value, unit) for f, op, value, unit in entry["conditions"])
        body = entry["body"]
        if conditions:
            stated = "; ".join(c.render() for c in conditions)
            body = f"{body}\n\nApplies when: {stated}."
        cards.append(
            Card(
                card_id=f"{KIND_PRACTICE}:{entry['id']}",
                kind=KIND_PRACTICE,
                title=entry["title"],
                body=body,
                source=entry["source"],
                embed_text=f"{entry['title']}. {entry['body']}",
                intents=tuple(entry["intents"]),
                conditions=conditions,
                keywords=tuple(entry["keywords"]),
                category="Practice",
                origin="vendored crop_recommendations.py / soil_recommendations.py",
            )
        )
    return cards


# --------------------------------------------------------------------------
# dataset provenance cards
# --------------------------------------------------------------------------

# Registry families -> the intents a question about that data would carry.
_FAMILY_INTENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("soil", ("soil_fertility_management", "agronomy_concept_explainer")),
    ("water", ("irrigation_advice", "agronomy_concept_explainer")),
    ("suitability", ("crop_selection", "agronomy_concept_explainer")),
    ("crop_suite", ("crop_selection", "agronomy_concept_explainer")),
    ("cropland", ("field_health_check", "agronomy_concept_explainer")),
    ("farm inputs", ("fertilizer_advice", "market_and_inputs_supply", "agronomy_concept_explainer")),
    ("climate", ("crop_selection", "agronomy_concept_explainer")),
)
_DEFAULT_DATASET_INTENTS = ("agronomy_concept_explainer",)


def _dataset_intents(family: str) -> tuple[str, ...]:
    lowered = family.lower()
    for needle, intents in _FAMILY_INTENTS:
        if needle in lowered:
            return intents
    return _DEFAULT_DATASET_INTENTS


def _asset_slug(asset_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", asset_id.lower()).strip("_")


def _family_short(family: str) -> str:
    """The family name without its parenthetical or em-dash commentary."""
    head = re.split(r"\s+\(|\s+[-—]{1,2}\s+", family, maxsplit=1)[0]
    return head.strip().rstrip(",")


def build_dataset_cards(path: Path) -> list[Card]:
    """One card per Earth Engine dataset, so a number's origin is answerable."""
    if not path.is_file():
        raise DataFileError(path, "ee_registry is required to build the corpus")
    payload = json.loads(path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets") or []
    cards: list[Card] = []
    # Four assets are registered twice, once per family they serve (TERRACLIMATE,
    # CHIRPS, global_ai_yearly, LGRIP30), with different notes each time. Both
    # entries are kept; the second gets a suffixed id so neither is lost.
    seen: dict[str, int] = {}
    for entry in datasets:
        asset_id = _clean(entry.get("asset_id"))
        if not asset_id:
            continue
        slug = _asset_slug(asset_id)
        seen[slug] = seen.get(slug, 0) + 1
        if seen[slug] > 1:
            slug = f"{slug}__{seen[slug]}"
        bands = [b for b in (entry.get("bands") or []) if b]
        family = _clean(entry.get("family"))
        lines = [
            f"Earth Engine asset: {asset_id} ({_clean(entry.get('ee_type')) or 'unknown type'})",
            f"Used for: {family}" if family else "",
            "Bands: " + ", ".join(bands) if bands else "",
            f"Units: {_clean(entry.get('units'))}" if _clean(entry.get("units")) else "",
            f"Scaling CropUp applies: {_clean(entry.get('scaling'))}" if _clean(entry.get("scaling")) else "",
            f"Coverage: {_clean(entry.get('coverage'))}" if _clean(entry.get("coverage")) else "",
            f"Native resolution: {_clean(entry.get('resolution'))}" if _clean(entry.get("resolution")) else "",
            f"Temporal range: {_clean(entry.get('temporal_range'))}" if _clean(entry.get("temporal_range")) else "",
            f"Licence: {_clean(entry.get('license'))}" if _clean(entry.get("license")) else "",
            f"Notes: {_clean(entry.get('notes'))}" if _clean(entry.get("notes")) else "",
        ]
        probes = [
            (label, _clean(entry.get(key)))
            for label, key in (
                ("Arusha, Tanzania", "value_arusha"),
                ("Morogoro, Tanzania", "value_morogoro"),
                ("Salinas, United States", "value_salinas"),
            )
        ]
        probes = [(label, value) for label, value in probes if value]
        if probes:
            lines.append(
                "Readings observed at the three verification points when this registry was built "
                "(they verify the asset, they are not a measurement of your field):"
            )
            lines.extend(f"- {label}: {value}" for label, value in probes)
        body = "\n".join(line for line in lines if line)

        embed_bits = [f"{asset_id}", family, "bands " + ", ".join(bands) if bands else ""]
        for key in ("units", "coverage", "resolution", "temporal_range"):
            value = _clean(entry.get(key))
            if value:
                embed_bits.append(value)
        notes = _clean(entry.get("notes"))
        if notes:
            embed_bits.append(notes[:400])
        cards.append(
            Card(
                card_id=f"{KIND_DATASET}:{slug}",
                kind=KIND_DATASET,
                title=f"Data source: {asset_id}" + (f" - {_family_short(family)}" if family else ""),
                body=body,
                source=_REGISTRY_CITATION.format(asset_id=asset_id),
                embed_text=". ".join(b for b in embed_bits if b),
                intents=_dataset_intents(family),
                keywords=tuple(b.lower() for b in bands[:8]),
                category="Data source",
                origin="cropup/data/ee_registry.json",
            )
        )
    return cards


# --------------------------------------------------------------------------
# topic cards
# --------------------------------------------------------------------------

# A topic card is an index, not a definition: it collects the sentences that
# already exist elsewhere in the corpus and keeps each one's citation. A topic
# with fewer than _TOPIC_MIN_QUOTES supporting sentences is not emitted, so an
# empty heading can never look like knowledge.
_TOPIC_MIN_QUOTES = 2
_TOPIC_MAX_QUOTES = 10

_TOPICS: tuple[dict[str, Any], ...] = (
    {"id": "mulching", "title": "Mulching", "terms": ("mulch",),
     "intents": ("agronomy_concept_explainer", "crop_management_practice")},
    {"id": "composting", "title": "Compost and manure", "terms": ("compost", "manure"),
     "intents": ("agronomy_concept_explainer", "soil_fertility_management")},
    {"id": "liming", "title": "Liming acid soil", "terms": ("lime", "liming"),
     "intents": ("agronomy_concept_explainer", "soil_fertility_management", "fertilizer_advice")},
    {"id": "crop_rotation", "title": "Crop rotation", "terms": ("rotat",),
     "intents": ("agronomy_concept_explainer", "crop_management_practice")},
    {"id": "cover_crops_and_residues", "title": "Cover crops and crop residues",
     "terms": ("cover crop", "residue"),
     "intents": ("agronomy_concept_explainer", "soil_fertility_management")},
    {"id": "field_drainage", "title": "Field drainage", "terms": ("drainage", "drain ", "drains"),
     "intents": ("agronomy_concept_explainer", "crop_management_practice", "irrigation_advice")},
    {"id": "seed_treatment", "title": "Seed treatment and certified seed",
     "terms": ("seed treat", "certified seed", "treated seed"),
     "intents": ("agronomy_concept_explainer", "seed_variety_selection")},
    {"id": "resistant_varieties", "title": "Resistant and tolerant varieties",
     "terms": ("resistant variet", "tolerant variet", "resistant hybrid"),
     "intents": ("seed_variety_selection", "agronomy_concept_explainer")},
    {"id": "scouting", "title": "Scouting and field inspection", "terms": ("scout",),
     "intents": ("agronomy_concept_explainer", "crop_management_practice")},
    {"id": "canopy_airflow", "title": "Canopy airflow, spacing and pruning",
     "terms": ("airflow", "prune", "pruning", "spacing"),
     "intents": ("agronomy_concept_explainer", "crop_management_practice")},
    {"id": "foliar_feeding", "title": "Foliar nutrient sprays", "terms": ("foliar",),
     "intents": ("fertilizer_advice", "agronomy_concept_explainer")},
    {"id": "nitrogen_fertiliser", "title": "Nitrogen fertilizer",
     "terms": ("nitrogen", "urea", "ammonium", " n/ha"),
     "intents": ("fertilizer_advice", "soil_fertility_management")},
    {"id": "gypsum_and_sodicity", "title": "Gypsum, sodicity and soil structure",
     "terms": ("gypsum", "sodic"),
     "intents": ("soil_fertility_management", "agronomy_concept_explainer")},
    {"id": "irrigation_scheduling", "title": "Irrigation scheduling",
     "terms": ("irrigate", "irrigation"),
     "intents": ("irrigation_advice", "agronomy_concept_explainer")},
)

_SENTENCE_SPLIT = re.compile(r"[;.]\s+|\n")


def _quotable_sentences(card: Card) -> Iterator[str]:
    """The sentences of a card that are advice, not tabular scaffolding."""
    if card.kind == KIND_DISEASE:
        parts = [line for line in card.body.splitlines() if line.startswith("Recommendation: ")]
        parts = [p[len("Recommendation: ") :] for p in parts]
    elif card.kind == KIND_PRACTICE:
        parts = [line for line in card.body.splitlines() if line and not line.startswith("Applies when:")]
    else:
        return
    for part in parts:
        for sentence in _SENTENCE_SPLIT.split(part):
            # The split already dropped the separator; strip any list bullet and
            # trailing stop so the quote gets exactly one closing period.
            sentence = sentence.strip().lstrip("-").strip().rstrip(". ")
            if len(sentence) >= 25:
                yield sentence


def build_topic_cards(cards: Sequence[Card]) -> list[Card]:
    """Index cards: every verbatim mention of a named practice, with citations."""
    out: list[Card] = []
    for topic in _TOPICS:
        terms = tuple(t.lower() for t in topic["terms"])
        quotes: list[tuple[str, str]] = []
        seen: set[str] = set()
        for card in cards:
            if card.kind == KIND_TOPIC:
                continue
            for sentence in _quotable_sentences(card):
                lowered = sentence.lower()
                if not any(term in lowered for term in terms):
                    continue
                key = lowered
                if key in seen:
                    continue
                seen.add(key)
                quotes.append((sentence, card.citation))
                break  # at most one quote per card, so one crop cannot dominate
            if len(quotes) >= _TOPIC_MAX_QUOTES:
                break
        if len(quotes) < _TOPIC_MIN_QUOTES:
            continue
        title = topic["title"]
        lines = [
            f"What the CropUp knowledge base says about {title.lower()}.",
            "Each line below is quoted exactly as it appears in the cited rule or practice card; "
            "this card adds no advice of its own.",
        ]
        lines.extend(f'- "{sentence}." [{citation}]' for sentence, citation in quotes)
        out.append(
            Card(
                card_id=f"{KIND_TOPIC}:{topic['id']}",
                kind=KIND_TOPIC,
                title=title,
                body="\n".join(lines),
                source=(
                    f"CropUp topic index for {title.lower()}: {len(quotes)} verbatim quotations, "
                    "each carrying its own citation"
                ),
                embed_text=f"{title}. " + " ".join(f"{sentence}." for sentence, _ in quotes),
                intents=tuple(topic["intents"]),
                keywords=terms,
                category="Topic index",
                origin="assembled from the disease and practice cards",
            )
        )
    return out


# --------------------------------------------------------------------------
# build / write / load
# --------------------------------------------------------------------------


def build_cards(settings: Settings | None = None) -> list[Card]:
    """Every card, in a stable order: disease, practice, dataset, topic."""
    settings = settings or get_settings()
    cards: list[Card] = []
    cards.extend(build_disease_cards(settings.require_data_file("disease_library")))
    cards.extend(build_practice_cards())
    cards.extend(build_dataset_cards(settings.require_data_file("ee_registry")))
    cards.extend(build_topic_cards(cards))

    ids = [c.card_id for c in cards]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate card ids: {sorted(duplicates)}")
    return cards


def corpus_digest(cards: Sequence[Card]) -> str:
    """Identifies the exact card set an index was built from."""
    digest = hashlib.sha256()
    for card in cards:
        digest.update(card.card_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(card.embed_text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def cards_path(settings: Settings | None = None) -> Path:
    return (settings or get_settings()).corpus_dir / CARDS_FILE


def write_corpus(cards: Sequence[Card], corpus_dir: Path) -> dict[str, Any]:
    """Write ``cards.jsonl``; return a summary of what was written.

    The summary is returned rather than written beside the cards. A manifest
    file here was read by nothing: the index sidecar already carries the digest
    that decides whether the built index is current, and a second copy of it
    could only go stale against ``cards.jsonl`` without anything noticing.
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)
    target = corpus_dir / CARDS_FILE
    with target.open("w", encoding="utf-8") as handle:
        for card in cards:
            handle.write(json.dumps(card.as_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    by_kind: dict[str, int] = {}
    for card in cards:
        by_kind[card.kind] = by_kind.get(card.kind, 0) + 1
    return {
        "card_count": len(cards),
        "by_kind": dict(sorted(by_kind.items())),
        "crops_covered": len({c for card in cards for c in card.crops}),
        "digest": corpus_digest(cards),
    }


def load_cards(corpus_dir: Path | None = None, settings: Settings | None = None) -> list[Card]:
    """Read the written corpus. Raises DataFileError when it has not been built."""
    settings = settings or get_settings()
    directory = corpus_dir or settings.corpus_dir
    target = directory / CARDS_FILE
    if not target.is_file():
        raise DataFileError(target, "corpus has not been built; run tools/build_corpus.py")
    cards: list[Card] = []
    with target.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cards.append(Card.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise DataFileError(target, f"line {line_no} is not a valid card: {exc}") from exc
    if not cards:
        raise DataFileError(target, "corpus file is empty")
    return cards
