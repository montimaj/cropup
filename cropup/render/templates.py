"""The renderer. Deterministic templates over Facts, and nothing else.

SPEC section 4.2 asks for *renderer-level impossibility*: not a rule that says
"do not print 7.0 when pH is missing", but a code path where that sentence
cannot be produced. This module is that code path.

* A template slot is written ``{soil_ph}`` and can only be filled by a
  :class:`~cropup.evidence.Fact`. A float, an int, a string, ``None`` or a
  :class:`~cropup.evidence.Missing` all raise instead of rendering.
* A slot with no Fact raises :class:`~cropup.errors.MissingProvenanceError`.
  There is no default branch, no ``or 0``, no ``.get(..., 7.0)``.
* Literal text supplied by the caller -- the crop the farmer named, the place
  they confirmed -- is written ``{@crop}`` so that the two namespaces are
  visibly different in the template itself, and a literal must be a ``str``: a
  number describing the world has to arrive as a Fact.
* There is no generative model here and no free text. Every sentence in
  :data:`TEMPLATES` is written out in full, and retrieved knowledge is quoted
  verbatim with its citation (SPEC section 4.3).

Absence is rendered, never skipped. A section whose inputs came back masked
prints the :class:`~cropup.evidence.Missing` sentence -- "rootzone soil
moisture: not available -- the pixel is masked ..." -- and a quantity the
analysis never attempted is listed under ``not_measured`` rather than quietly
vanishing from the page.

Output is structured, not a string: every number comes back as its own
:class:`Segment` carrying ``fact.provenance()``, which is what the UI hover of
SPEC section 9 needs (instrument, date, resolution, chain position, scaling).

The analysis layer's own output -- ``Finding``, ``Advice``, ``Risk``,
``CropRanking`` -- reaches the farmer through here and not around it (SPEC
sections 4.2 and 4.3). Each of those types is read *structurally*, through an
``adapt()`` classmethod, because ``render`` imports ``evidence`` and nothing
else: it must never import ``analysis``. Their prose is routed verbatim, never
rewritten, and it is routed next to the Fact it was written from, so the number
on screen still carries its instrument.

Imports: :mod:`cropup.evidence` and :mod:`cropup.errors`. The one setting this
module reads is the risk cap of SPEC section 4.3, imported lazily inside the
function that needs it so that importing this module pulls in nothing else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..errors import FabricationError, MissingProvenanceError
from ..evidence import Fact, Ledger, Missing

__all__ = [
    "SEGMENT_TEXT",
    "SEGMENT_FACT",
    "SEGMENT_MISSING",
    "SEGMENT_QUOTE",
    "SEGMENT_DIAGNOSTIC",
    "SEGMENT_KINDS",
    "FIELD_RENDERERS",
    "TEMPLATES",
    "Segment",
    "RenderedText",
    "Section",
    "Answer",
    "TriggerView",
    "RiskView",
    "CropRankingView",
    "FindingView",
    "AdviceView",
    "AnswerBuilder",
    "template_slots",
    "template_literals",
    "render_template",
    "render_gap",
    "risk_rows",
    "render_risks",
    "render_finding",
    "render_findings",
    "render_advice",
    "render_advice_list",
    "plant_health_answer",
    "irrigation_answer",
    "crop_selection_answer",
    "analysis_answer",
    "rag_answer",
]

# -- segment kinds ---------------------------------------------------------

SEGMENT_TEXT = "text"  # literal template prose
SEGMENT_FACT = "fact"  # a measured number, with its provenance attached
SEGMENT_MISSING = "missing"  # a named absence
SEGMENT_QUOTE = "quote"  # retrieved knowledge, verbatim
SEGMENT_DIAGNOSTIC = "diagnostic"  # a number about the system, not about the field

SEGMENT_KINDS = (SEGMENT_TEXT, SEGMENT_FACT, SEGMENT_MISSING, SEGMENT_QUOTE, SEGMENT_DIAGNOSTIC)

# ``{{`` / ``}}`` escape a brace; ``{@name}`` is literal text; ``{name}`` and
# ``{name:spec}`` are Facts.
_SLOT_RE = re.compile(r"\{\{|\}\}|\{(@?)([A-Za-z_][A-Za-z0-9_]*)(?::([A-Za-z0-9_]+))?\}")


def _no_unit_value(fact: Fact) -> str:
    """The bare number, for sentences that already carry the unit in words."""
    if isinstance(fact.value, bool):
        return "yes" if fact.value else "no"
    if fact.is_numeric:
        return fact.render().removesuffix(fact.unit).strip() if fact.unit else fact.render()
    return str(fact.value)


#: ``{slot:spec}`` handlers. A bare ``{slot}`` renders value + unit; a numeric
#: spec (``{ndvi:3}``) sets the decimal places. Each of these answers with a
#: sentence rather than an empty string when the Fact does not carry the detail,
#: because "" reads as "nothing to report" and that would be a quiet default.
FIELD_RENDERERS = {
    "value": _no_unit_value,
    "unit": lambda f: f.unit or "no unit",
    "source": lambda f: f.source_asset,
    "band": lambda f: f.band or "no single band",
    "date": lambda f: (
        f.observed_on.isoformat() if f.observed_on else "no observation date (a static layer)"
    ),
    "age": lambda f: (
        f"{f.age_days} days ago" if f.age_days is not None else "with no observation date"
    ),
    "resolution": lambda f: (
        f"{f.resolution_m:g} m" if f.resolution_m else "an unrecorded resolution"
    ),
    "chain": lambda f: f.chain_label,
    "scaling": lambda f: f.scaling_applied or "no scaling recorded",
    "note": lambda f: f.note or "no note recorded",
}


@dataclass(frozen=True)
class Segment:
    """One piece of a rendered line, and where it came from.

    The UI draws ``kind == "fact"`` segments as hoverable numbers using
    ``provenance``; SPEC section 9 wants instrument, date and resolution behind
    every number on screen, and they are all in there.
    """

    kind: str
    text: str
    slot: str | None = None
    provenance: Mapping[str, Any] | None = None
    citation: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in SEGMENT_KINDS:
            raise ValueError(f"unknown segment kind {self.kind!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "slot": self.slot,
            "provenance": dict(self.provenance) if self.provenance else None,
            "citation": self.citation,
        }


@dataclass(frozen=True)
class RenderedText:
    """One rendered line: the text, and the evidence behind each number."""

    template: str  # the template's name, for the error messages and the tests
    source: str  # the literal template text, so a reader can audit the wording
    segments: tuple[Segment, ...]

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.segments)

    def facts_used(self) -> tuple[str, ...]:
        """The template *slots* that rendered a Fact, in order.

        A slot is not a quantity: ``line_with`` binds ``crop_suitability_maize``
        to a ``{suitability}`` slot. Use :meth:`quantities_used` for the names
        the ledger knows.
        """
        return tuple(s.slot for s in self.segments if s.kind == SEGMENT_FACT and s.slot)

    def quantities_used(self) -> tuple[str, ...]:
        """The quantities behind this line, read off each Fact's provenance."""
        out: list[str] = []
        for segment in self.segments:
            if segment.kind != SEGMENT_FACT or not segment.provenance:
                continue
            quantity = str(segment.provenance.get("quantity") or segment.slot or "")
            if quantity and quantity not in out:
                out.append(quantity)
        return tuple(out)

    def gaps_named(self) -> tuple[str, ...]:
        return tuple(s.slot for s in self.segments if s.kind == SEGMENT_MISSING and s.slot)

    def provenance(self) -> dict[str, Any]:
        return {
            s.slot: dict(s.provenance)
            for s in self.segments
            if s.slot and s.provenance is not None
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "text": self.text,
            "segments": [s.as_dict() for s in self.segments],
        }

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class Section:
    """A headed block of an answer."""

    key: str
    heading: str
    lines: tuple[RenderedText, ...] = ()
    #: quantities this section would have used, that the analysis never attempted
    not_measured: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "heading": self.heading,
            "lines": [line.as_dict() for line in self.lines],
            "not_measured": list(self.not_measured),
        }


@dataclass(frozen=True)
class Answer:
    """A whole rendered response: sections, citations and what was degraded."""

    kind: str
    title: str
    sections: tuple[Section, ...] = ()
    citations: tuple[str, ...] = ()
    not_measured: tuple[str, ...] = ()
    degradation: Mapping[str, Any] | None = None

    @property
    def text(self) -> str:
        blocks = [self.title]
        for section in self.sections:
            body = section.text
            if not body:
                continue
            blocks.append(f"{section.heading}\n{body}" if section.heading else body)
        if self.citations:
            blocks.append("Sources\n" + "\n".join(self.citations))
        return "\n\n".join(blocks)

    def facts_used(self) -> tuple[str, ...]:
        """Every quantity this answer put on screen, by the ledger's name for it."""
        seen: list[str] = []
        for section in self.sections:
            for line in section.lines:
                for quantity in line.quantities_used():
                    if quantity not in seen:
                        seen.append(quantity)
        return tuple(seen)

    def gaps_named(self) -> tuple[str, ...]:
        seen: list[str] = []
        for section in self.sections:
            for line in section.lines:
                for quantity in line.gaps_named():
                    if quantity not in seen:
                        seen.append(quantity)
        return tuple(seen)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "sections": [s.as_dict() for s in self.sections],
            "citations": list(self.citations),
            "facts_used": list(self.facts_used()),
            "gaps_named": list(self.gaps_named()),
            "not_measured": list(self.not_measured),
            "degradation": dict(self.degradation) if self.degradation else None,
            "text": self.text,
        }


# --------------------------------------------------------------------------
# the template engine
# --------------------------------------------------------------------------


def template_slots(template: str) -> tuple[str, ...]:
    """The Fact slots a template needs, in order, without duplicates."""
    out: list[str] = []
    for match in _SLOT_RE.finditer(template):
        if match.group(0) in ("{{", "}}") or match.group(1) == "@":
            continue
        name = match.group(2)
        if name not in out:
            out.append(name)
    return tuple(out)


def template_literals(template: str) -> tuple[str, ...]:
    """The ``{@name}`` literals a template needs."""
    out: list[str] = []
    for match in _SLOT_RE.finditer(template):
        if match.group(0) in ("{{", "}}") or match.group(1) != "@":
            continue
        name = match.group(2)
        if name not in out:
            out.append(name)
    return tuple(out)


def _require_fact(
    facts: Ledger | Mapping[str, Any], slot: str, template: str
) -> Fact:
    """The one lookup. Every path out of here is a Fact or an exception."""
    if isinstance(facts, Ledger):
        return facts.require(slot, template=template)  # raises MissingProvenanceError
    try:
        value = facts[slot]
    except (KeyError, TypeError):
        raise MissingProvenanceError(slot, template, "nothing was recorded for this quantity") from None
    if isinstance(value, Fact):
        return value
    if isinstance(value, Missing):
        raise MissingProvenanceError(slot, template, value.render())
    raise FabricationError(
        f"template {template!r} slot {slot!r} was given a {type(value).__name__}; "
        "the renderer accepts Fact objects only, so wrap the value with its source first",
        value,
    )


def _render_field(fact: Fact, spec: str | None, slot: str, template: str) -> str:
    if not spec:
        return fact.render()
    if spec.isdigit():
        return fact.render(precision=int(spec))
    handler = FIELD_RENDERERS.get(spec)
    if handler is None:
        known = ", ".join(sorted(FIELD_RENDERERS))
        raise ValueError(
            f"template {template!r} slot {slot!r}: unknown field {spec!r} (known: {known}, or a digit)"
        )
    return handler(fact)


def render_template(
    template: str,
    facts: Ledger | Mapping[str, Any] | None = None,
    *,
    name: str = "",
    literals: Mapping[str, str] | None = None,
) -> RenderedText:
    """Fill one template from Facts. Raises rather than defaulting.

    ``facts`` is a :class:`~cropup.evidence.Ledger` or a mapping of slot name to
    Fact. ``literals`` fills the ``{@name}`` slots and must contain strings: a
    number that describes the world belongs in a Fact, where its instrument
    travels with it.
    """
    name = name or "<inline>"
    facts = {} if facts is None else facts
    literals = literals or {}
    segments: list[Segment] = []
    cursor = 0
    pending: list[str] = []  # literal run, flushed as one text segment

    def flush() -> None:
        if pending:
            segments.append(Segment(SEGMENT_TEXT, "".join(pending)))
            pending.clear()

    for match in _SLOT_RE.finditer(template):
        pending.append(template[cursor : match.start()])
        cursor = match.end()
        token = match.group(0)
        if token == "{{":
            pending.append("{")
            continue
        if token == "}}":
            pending.append("}")
            continue
        is_literal, slot, spec = match.group(1) == "@", match.group(2), match.group(3)
        if is_literal:
            if slot not in literals:
                raise ValueError(f"template {name!r}: literal {{@{slot}}} was not supplied")
            value = literals[slot]
            if not isinstance(value, str):
                raise FabricationError(
                    f"template {name!r} literal {{@{slot}}} is a {type(value).__name__}; "
                    "literals are text, and a measured number must arrive as a Fact",
                    value,
                )
            pending.append(value)
            continue
        fact = _require_fact(facts, slot, name)
        flush()
        segments.append(
            Segment(
                kind=SEGMENT_FACT,
                text=_render_field(fact, spec, slot, name),
                slot=slot,
                provenance=fact.provenance(),
            )
        )

    pending.append(template[cursor:])
    flush()
    return RenderedText(template=name, source=template, segments=tuple(segments))


def render_gap(missing: Missing, *, name: str = "gap") -> RenderedText:
    """Render a named absence as its own line.

    This is the only way a quantity with no Fact reaches the page, and it says
    what was tried and why it failed instead of showing a number.
    """
    if not isinstance(missing, Missing):
        raise FabricationError(
            f"render_gap() takes a Missing, not {type(missing).__name__}", missing
        )
    return RenderedText(
        template=name,
        source="<missing>",
        segments=(
            Segment(
                kind=SEGMENT_MISSING,
                text=missing.render(),
                slot=missing.quantity,
                provenance=missing.provenance(),
            ),
        ),
    )


# --------------------------------------------------------------------------
# the templates
# --------------------------------------------------------------------------

#: Every sentence the app can produce about a measurement. Written out in full
#: on purpose: this table *is* the language model (SPEC section 4.3).
TEMPLATES: dict[str, str] = {
    # -- plant health ------------------------------------------------------
    "plant_health.ndvi": (
        "Canopy greenness (NDVI) is {ndvi}, measured by {ndvi:source} on {ndvi:date} "
        "at {ndvi:resolution} resolution."
    ),
    "plant_health.spread": (
        "Within the field NDVI runs from {ndvi_p10} at the 10th percentile to {ndvi_p90} "
        "at the 90th, a spread of {ndvi_spread}."
    ),
    "plant_health.zones": (
        "By area the canopy splits into {ndvi_zone_healthy_pct} healthy, {ndvi_zone_fair_pct} fair, "
        "{ndvi_zone_stressed_pct} stressed and {ndvi_zone_severe_pct} severe."
    ),
    "plant_health.change": (
        "A year ago the same field read {ndvi_year_ago}; the change since then is {ndvi_change_1y}."
    ),
    "plant_health.scenes": (
        "The composite behind these numbers used {s2_scene_count}, "
        "{s2_observations_per_pixel} per pixel."
    ),
    "plant_health.moisture": (
        "Canopy moisture (NDMI) is {ndmi} and canopy chlorophyll (NDRE) is {ndre}."
    ),
    "plant_health.rootzone": (
        "Rootzone soil moisture is {soil_moisture_rootzone}, from {soil_moisture_rootzone:source} "
        "on {soil_moisture_rootzone:date}."
    ),
    "plant_health.rain": "Rain over the last 7 days totalled {precipitation_7d}.",
    "plant_health.thermal": (
        "Land surface temperature is {land_surface_temperature}, from "
        "{land_surface_temperature:source} on {land_surface_temperature:date}; that scene was "
        "{land_surface_temperature_clear_fraction} clear."
    ),
    "plant_health.soil": (
        "The soil is {soil_texture_class}, pH {soil_ph:value}, with organic carbon at {soil_soc}."
    ),
    "plant_health.climate": "The climate here is {koppen_code}, {koppen_name}.",
    # -- irrigation --------------------------------------------------------
    "irrigation.et": (
        "The crop used {et_actual} of water against a reference demand of {et0}, "
        "measured by {et_actual:source} on {et_actual:date}."
    ),
    "irrigation.ratio": (
        "That is a crop-to-reference ratio of {et_actual_over_et0}; at 1.0 the canopy is "
        "transpiring at the reference rate for this climate."
    ),
    "irrigation.anomaly": (
        "Against the dekadal median for this date, actual ET is {et_actual_anomaly_pct} "
        "(median {et_actual_dekad_median})."
    ),
    "irrigation.esi": (
        "The evaporative stress index is {evaporative_stress_index}, observed on "
        "{evaporative_stress_index:date}, {evaporative_stress_index:age}."
    ),
    "irrigation.rootzone": (
        "Rootzone soil moisture is {soil_moisture_rootzone}, which is "
        "{soil_moisture_rootzone_wetness} of saturation, from {soil_moisture_rootzone:source} "
        "on {soil_moisture_rootzone:date}."
    ),
    "irrigation.rain": (
        "Rain in the last 7 days came to {precipitation_7d}, and in the last 30 days "
        "{precipitation_30d}."
    ),
    "irrigation.forecast": (
        "Reference ET is forecast at {et0_forecast_day1} for tomorrow and "
        "{et0_forecast_7d} over the next 7 days."
    ),
    "irrigation.pawc": (
        "This soil holds {soil_pawc} of plant-available water between field capacity "
        "({soil_field_capacity}) and wilting point ({soil_wilting_point})."
    ),
    "irrigation.regime": (
        "The cropland around this field is mapped as {irrigation_regime}: "
        "{irrigated_cropland_pct} of it irrigated and {rainfed_cropland_pct} rainfed."
    ),
    # -- crop selection ----------------------------------------------------
    "crop_selection.rank": (
        "{@rank}. {@crop}: {suitability}, from {suitability:source} "
        "({suitability:chain}, {suitability:resolution})."
    ),
    "crop_selection.crop": (
        "{@crop} scores {crop_suitability} here, from {crop_suitability:source} "
        "({crop_suitability:chain})."
    ),
    "crop_selection.limiting": "The factor limiting {@crop} here is {crop_limiting_factor}.",
    "crop_selection.sowing": (
        "The climatological optimum sowing day is {optimal_sowing_date}; the suitable "
        "sowing window is {sowing_window_days}."
    ),
    "crop_selection.context": (
        "The land cover at this point is {land_cover}, with a cropland probability of "
        "{cropland_probability}."
    ),
    # -- freshness (SPEC 3.3) ----------------------------------------------
    "staleness.note": (
        "{@quantity}: observed {stale:date}, {stale:age}, from {stale:source}. {@note}"
    ),
    # -- risks (SPEC 4.3) --------------------------------------------------
    "risk.headline": "{@severity} — {@name} ({@issue_id}).",
    "risk.detail": "{@detail}",
    "risk.evidence_strength": "Evidence strength: {@evidence_strength}.",
    "risk.action": "Recommended: {@recommendation} ({@urgency}).",
    "risk.trigger": "Triggered by {@quantity} = {trigger}, from {trigger:source} on {trigger:date}.",
    # -- findings: the analysis layer's readings, routed (SPEC 4.2) --------
    # The measurement is rendered from the Fact, so it keeps its hover; the
    # reading beside it is the orchestrator's own deterministic prose, quoted
    # as written. Nothing here rewrites, summarises or softens it.
    "finding.measured": "{@label}: {finding}, measured by {finding:source} {finding:age}.",
    "finding.reading": "{@reading}",
    "finding.supporting": (
        "Alongside it, {@quantity} is {supporting}, measured by {supporting:source} {supporting:age}."
    ),
    # -- advice ------------------------------------------------------------
    "advice.text": "{@text}",
    "advice.urgency": "Timing: {@urgency}.",
    "advice.evidence": (
        "That rests on {@quantity} = {evidence}, measured by {evidence:source} {evidence:age}."
    ),
    "advice.blocked": "This cannot be advised here until these are measured: {@missing}.",
}


# --------------------------------------------------------------------------
# adapters: what the renderer needs from what analysis/ produces
# --------------------------------------------------------------------------


def _reader(item: Any) -> Any:
    """A ``get(key, default)`` over a Mapping or over an object's attributes.

    The renderer never imports ``analysis``, so every producer type is read
    structurally: the rules engine's own ``Risk`` and the dict its ``as_dict()``
    returns both arrive here, and neither is named.
    """
    if isinstance(item, Mapping):
        return item.get
    return lambda k, d=None: getattr(item, k, d)


@dataclass(frozen=True)
class TriggerView:
    """One observation that fired a risk, and the Fact behind it.

    ``analysis/rules.py`` hands over its own ``Trigger``, which already holds
    the ``Fact`` that satisfied the condition. A caller with only a quantity
    name may hand that over instead, and the Fact is then looked up on the
    ledger. Either way the risk reaches the page with the measurement that
    fired it attached (SPEC section 4.3).

    The producer's own word for the field is read from ``label`` and from
    nowhere else. ``Trigger`` keeps it on ``condition.label``, and that does not
    survive ``Trigger.as_dict()``: reading the nested object here and falling
    back to the raw ``field`` name rendered one trigger as "cation exchange
    capacity" from the object and as "cec_mmol_kg" from that same object's dict,
    so a server that serialised before rendering showed the farmer different
    text for the same measurement. One key both forms can carry, and otherwise
    the quantity the Fact names itself, is what makes the two agree.
    """

    quantity: str
    fact: Fact | None = None
    name: str = ""  # the rule's own word for the field, when it has one

    @property
    def label(self) -> str:
        return self.name or self.quantity.replace("_", " ")

    @classmethod
    def adapt(cls, trigger: Any) -> "TriggerView":
        if isinstance(trigger, TriggerView):
            return trigger
        if isinstance(trigger, str):
            return cls(trigger)
        if isinstance(trigger, Fact):
            return cls(trigger.quantity, trigger)
        get = _reader(trigger)
        fact = get("fact")
        name = str(get("label", "") or "")
        if isinstance(fact, Fact):
            return cls(fact.quantity, fact, name)
        quantity = str(get("quantity", "") or "")
        if not quantity:
            raise FabricationError(
                "a risk trigger must name the quantity that fired it or carry its Fact; "
                f"got a {type(trigger).__name__} with neither",
                trigger,
            )
        return cls(quantity, None, name)


@dataclass(frozen=True)
class RiskView:
    """What the renderer needs from one entry of the rules engine.

    Kept structural on purpose: ``analysis/rules.py`` owns the ranking, the
    deduplication by ``issue_id`` and the ordering, and hands rows here as
    dicts or as its own objects.

    ``triggered_by`` is normalised to :class:`TriggerView` on construction, so a
    caller may build one from bare quantity names and the rules engine's own
    ``Risk.triggers`` lands in the same place. The producer's name for the field
    is ``triggers``; missing it is how the trigger evidence SPEC section 4.3
    requires used to become an empty tuple without anything being raised.

    ``evidence_strength`` is the rules engine's own verdict on those triggers --
    ``"canopy_only"`` when nothing but the Sentinel-2 indices fired the rule,
    ``"corroborated"`` once something off the canopy had to be true as well. It
    is carried because ``RiskAssessment.caveat()`` names it: "risks marked
    'canopy_only' ... need scouting to confirm" is a caveat about a mark, and
    dropping the field left that caveat on the page pointing at nothing.
    """

    issue_id: str
    name: str
    severity: str
    detail: str = ""
    recommendation: str = ""
    urgency: str = ""
    category: str = ""
    evidence_strength: str = ""
    triggered_by: tuple[TriggerView, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "triggered_by", tuple(TriggerView.adapt(t) for t in self.triggered_by)
        )

    @classmethod
    def adapt(cls, risk: Any) -> "RiskView":
        if isinstance(risk, RiskView):
            return risk
        get = _reader(risk)
        return cls(
            issue_id=str(get("issue_id", "") or ""),
            name=str(get("name", "") or get("issue_name", "") or ""),
            severity=str(get("severity", "") or get("level", "") or ""),
            detail=str(get("detail", "") or get("notes", "") or ""),
            recommendation=str(get("recommendation", "") or ""),
            urgency=str(get("urgency", "") or ""),
            category=str(get("category", "") or ""),
            evidence_strength=str(get("evidence_strength", "") or ""),
            triggered_by=tuple(
                get("triggers", ()) or get("triggered_by", ()) or get("observations", ()) or ()
            ),
        )


@dataclass(frozen=True)
class CropRankingView:
    """One row of a crop ranking: the crop, and the Fact that scored it.

    ``analysis/crop_selection.py`` produces ``CropRanking(crop, fact)``;
    ``geo/suitability.rank_crops`` produces bare Facts named
    ``crop_suitability_<crop>``. Both arrive here, and either way it is the Fact
    that the row template binds, so the score keeps the instrument, the date and
    the resolution that SPEC section 4.2 requires of it.
    """

    crop: str
    fact: Fact
    key: str = ""  # the quantity's crop suffix, which the ``names`` override keys on

    def __post_init__(self) -> None:
        if not isinstance(self.fact, Fact):
            raise FabricationError(
                "a crop ranking row is rendered from the Fact that scored it, not from a "
                f"{type(self.fact).__name__}",
                self.fact,
            )
        if not self.key:
            object.__setattr__(self, "key", self.fact.quantity.removeprefix("crop_suitability_"))
        if not self.crop:
            object.__setattr__(self, "crop", self.key.replace("_", " ").title())

    @classmethod
    def adapt(cls, row: Any) -> "CropRankingView":
        if isinstance(row, CropRankingView):
            return row
        if isinstance(row, Fact):
            return cls("", row)
        get = _reader(row)
        fact = get("fact")
        if not isinstance(fact, Fact):
            raise FabricationError(
                f"crop rankings must carry the Fact that scored them; a {type(row).__name__} "
                "does not, and a provenance dictionary is not a substitute for one",
                row,
            )
        return cls(str(get("crop", "") or ""), fact)


@dataclass(frozen=True)
class FindingView:
    """What the renderer needs from one ``analysis.rules.Finding``.

    ``reading`` is the orchestrator's own deterministic prose (SPEC section 4.3:
    no generative model wrote it, and every number in it came out of
    ``Fact.render()``). It is routed, not rewritten -- but it is routed *beside*
    its evidence, which is rendered from the Fact itself, so the measurement on
    screen still carries its instrument.

    ``evidence`` must be a Fact or a Missing. A provenance dictionary is neither:
    it has already lost the firewall's guarantee, so it raises here rather than
    being printed as though it were a measurement.
    """

    key: str
    label: str
    evidence: Fact | Missing
    reading: str = ""
    importance: str = "info"
    supporting: tuple[Fact | Missing, ...] = ()

    @property
    def measured(self) -> bool:
        return isinstance(self.evidence, Fact)

    @property
    def heading(self) -> str:
        return (
            self.label
            or self.key.replace("_", " ").capitalize()
            or self.evidence.quantity.replace("_", " ")
        )

    @classmethod
    def adapt(cls, finding: Any) -> "FindingView":
        if isinstance(finding, FindingView):
            return finding
        get = _reader(finding)
        evidence = get("evidence")
        if not isinstance(evidence, (Fact, Missing)):
            raise FabricationError(
                "a finding reaches the renderer with its Fact or its named Missing, not a "
                f"{type(evidence).__name__}",
                evidence,
            )
        return cls(
            key=str(get("key", "") or ""),
            label=str(get("label", "") or ""),
            evidence=evidence,
            reading=str(get("reading", "") or ""),
            importance=str(get("importance", "") or "info"),
            supporting=tuple(get("supporting", ()) or ()),
        )


@dataclass(frozen=True)
class AdviceView:
    """What the renderer needs from one ``analysis.irrigation.Advice``.

    A blocked line -- one whose ``blocked_by`` names quantities that were never
    measured -- is rendered as the absence it is. It is never softened into a
    recommendation, which is the vendored backend's pH 7.0 failure in a
    different costume (SPEC section 2.1, bug 4).
    """

    key: str
    text: str
    urgency: str = ""
    evidence: tuple[Fact, ...] = ()
    blocked_by: tuple[str, ...] = ()

    @property
    def actionable(self) -> bool:
        return not self.blocked_by

    @classmethod
    def adapt(cls, advice: Any) -> "AdviceView":
        if isinstance(advice, AdviceView):
            return advice
        get = _reader(advice)
        evidence = tuple(get("evidence", ()) or ())
        bad = [e for e in evidence if not isinstance(e, Fact)]
        if bad:
            raise FabricationError(
                f"advice is rendered from the Facts it rests on, not from a {type(bad[0]).__name__}",
                bad[0],
            )
        return cls(
            key=str(get("key", "") or ""),
            text=str(get("text", "") or ""),
            urgency=str(get("urgency", "") or ""),
            evidence=evidence,
            blocked_by=tuple(str(q) for q in (get("blocked_by", ()) or ())),
        )


# --------------------------------------------------------------------------
# building an answer
# --------------------------------------------------------------------------


class AnswerBuilder:
    """Assembles an :class:`Answer` from a Ledger, one section at a time.

    ``line`` is the strict primitive: it renders or it raises. ``optional`` is
    the honest one: it renders when every quantity the template names was
    measured, prints the absence when one came back :class:`Missing`, and lists
    the quantity under ``not_measured`` when the analysis never attempted it.
    Nothing disappears silently in either case.
    """

    def __init__(
        self,
        kind: str,
        title: str,
        ledger: Ledger,
        *,
        literals: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(ledger, Ledger):
            raise FabricationError(
                f"an answer is built from a Ledger, not {type(ledger).__name__}", ledger
            )
        self.kind = kind
        self.title = title
        self.ledger = ledger
        self.literals = dict(literals or {})
        self._sections: list[Section] = []
        self._key: str | None = None
        self._heading = ""
        self._lines: list[RenderedText] = []
        self._not_measured: list[str] = []
        self._section_not_measured: list[str] = []
        self._gaps_emitted: set[str] = set()
        #: every Fact this answer has actually put on screen, by its own
        #: quantity, in the order it first appeared
        self._rendered_facts: list[tuple[str, Fact]] = []
        self._citations: list[str] = []

    # -- sections ----------------------------------------------------------

    def section(self, key: str, heading: str) -> "AnswerBuilder":
        self._close()
        self._key, self._heading = key, heading
        return self

    def _close(self) -> None:
        if self._key is not None and (self._lines or self._section_not_measured):
            self._sections.append(
                Section(
                    key=self._key,
                    heading=self._heading,
                    lines=tuple(self._lines),
                    not_measured=tuple(self._section_not_measured),
                )
            )
        self._key, self._heading = None, ""
        self._lines, self._section_not_measured = [], []

    def _emit(self, line: RenderedText, facts: Mapping[str, Fact] | None = None) -> RenderedText:
        if self._key is None:
            self.section("body", "")
        self._lines.append(line)
        self._register(line, facts)
        return line

    def _register(self, line: RenderedText, facts: Mapping[str, Fact] | None) -> None:
        """Remember the Facts this line put on screen, under their own names.

        A slot name is not a quantity name. ``line_with`` binds a Fact to a slot
        the row template can reuse -- ``{suitability}`` for
        ``crop_suitability_maize``, ``{trigger}`` for ``ndvi`` -- so resolving
        the slot against the ledger afterwards finds nothing at all. Following
        the Fact that was actually bound is what lets :meth:`stale_section` see a
        ranking row or a risk trigger; looking the slot up instead is how SPEC
        section 3.3's staleness silently stopped being shown on those paths.
        """
        for slot in line.facts_used():
            fact = facts.get(slot) if facts is not None else self.ledger.fact(slot)
            if not isinstance(fact, Fact):
                continue
            if any(quantity == fact.quantity for quantity, _ in self._rendered_facts):
                continue
            self._rendered_facts.append((fact.quantity, fact))

    # -- lines -------------------------------------------------------------

    def line(self, template_name: str, *, literals: Mapping[str, str] | None = None) -> RenderedText:
        """Render a named template. Raises on any slot with no Fact."""
        return self.line_with(template_name, None, literals=literals)

    def line_with(
        self,
        template_name: str,
        facts: Mapping[str, Fact] | None,
        *,
        literals: Mapping[str, str] | None = None,
    ) -> RenderedText:
        """Render a named template against explicit Facts.

        Used where the quantity name varies per row -- a crop suitability
        ranking binds ``crop_suitability_maize`` to the template's
        ``{suitability}`` slot -- so the row templates stay readable.
        """
        template = TEMPLATES[template_name]
        return self._emit(
            render_template(
                template,
                facts if facts is not None else self.ledger,
                name=template_name,
                literals={**self.literals, **(literals or {})},
            ),
            facts,
        )

    def text(self, literal: str) -> RenderedText:
        """A literal sentence with no measurement in it."""
        return self._emit(
            RenderedText(
                template="<literal>",
                source=literal,
                segments=(Segment(SEGMENT_TEXT, literal),),
            )
        )

    def optional(
        self,
        template_name: str,
        *,
        requires: Sequence[str] | None = None,
        literals: Mapping[str, str] | None = None,
    ) -> bool:
        """Render the line if it can be measured; otherwise name what is absent."""
        template = TEMPLATES[template_name]
        needed = tuple(requires) if requires is not None else template_slots(template)
        gaps: list[Missing] = []
        absent: list[str] = []
        for quantity in needed:
            entry = self.ledger.get(quantity)
            if entry is None:
                absent.append(quantity)
            elif isinstance(entry, Missing):
                gaps.append(entry)
        if gaps or absent:
            for missing in gaps:
                self.gap(missing)
            for quantity in absent:
                if quantity not in self._not_measured:
                    self._not_measured.append(quantity)
                if quantity not in self._section_not_measured:
                    self._section_not_measured.append(quantity)
            return False
        self.line(template_name, literals=literals)
        return True

    def note_not_measured(self, *quantities: str) -> "AnswerBuilder":
        """Name a quantity this answer wanted and the ledger never recorded.

        The counterpart to :meth:`gap`: a Missing says what was tried and failed,
        this says nothing was tried. Both are printed; neither is a default. A
        quantity the ledger has any entry for -- Fact or Missing -- is skipped,
        because it *was* attempted and saying otherwise is its own small lie.
        """
        if self._key is None:
            self.section("body", "")
        for quantity in quantities:
            if self.ledger.get(quantity) is not None:
                continue
            if quantity not in self._not_measured:
                self._not_measured.append(quantity)
            if quantity not in self._section_not_measured:
                self._section_not_measured.append(quantity)
        return self

    def gap(self, missing: Missing | str) -> RenderedText | None:
        """Print a named absence once per quantity."""
        if isinstance(missing, str):
            entry = self.ledger.get(missing)
            if not isinstance(entry, Missing):
                return None
            missing = entry
        if missing.quantity in self._gaps_emitted:
            return None
        self._gaps_emitted.add(missing.quantity)
        return self._emit(render_gap(missing, name=f"gap.{missing.quantity}"))

    def quote(self, text: str, citation: str) -> RenderedText:
        """Retrieved knowledge, verbatim, with its citation (SPEC section 6)."""
        if citation not in self._citations:
            self._citations.append(citation)
        return self._emit(
            RenderedText(
                template="<quote>",
                source=text,
                segments=(Segment(SEGMENT_QUOTE, text, citation=citation),),
            )
        )

    def diagnostic(self, text: str, provenance: Mapping[str, Any]) -> RenderedText:
        """A number about the system rather than about the field -- a retrieval
        score, a card count. It still names the instrument that produced it."""
        return self._emit(
            RenderedText(
                template="<diagnostic>",
                source=text,
                segments=(Segment(SEGMENT_DIAGNOSTIC, text, provenance=dict(provenance)),),
            )
        )

    # -- standard closing sections ----------------------------------------

    def stale_section(self, heading: str = "How old these numbers are") -> "AnswerBuilder":
        """Name every rendered Fact whose observation is old.

        SPEC section 3.3: ESI at Arusha came back over a year old, and a number
        that old must not sit on the page looking like today's reading. The
        source Fact is bound to the line, so the hover still carries everything.
        """
        aged = []
        for quantity, fact in self._rendered_facts:
            note = fact.staleness_note()
            if note:
                aged.append((quantity, fact, note))
        if aged:
            self.section("staleness", heading)
            for quantity, fact, note in aged:
                self.line_with(
                    "staleness.note",
                    {"stale": fact},
                    literals={
                        "quantity": quantity.replace("_", " "),
                        "note": note[0].upper() + note[1:] + ".",
                    },
                )
        return self

    def gaps_section(
        self, heading: str = "What I could not measure", *, only: Sequence[str] | None = None
    ) -> "AnswerBuilder":
        """Every Missing in the ledger that has not already been printed.

        ``only`` narrows the list to quantities the caller cares about; the full
        set always stays in ``Answer.degradation`` for the capability strip.
        """
        wanted = None if only is None else set(only)
        remaining = [
            m
            for m in self.ledger.gaps
            if m.quantity not in self._gaps_emitted and (wanted is None or m.quantity in wanted)
        ]
        if remaining:
            self.section("gaps", heading)
            for missing in remaining:
                self.gap(missing)
        return self

    def _outstanding(self) -> list[str]:
        """Quantities this answer wanted and nothing ever said anything about.

        A quantity is off this list once *either* record of an attempt exists:
        an entry on the ledger, or a :class:`~cropup.evidence.Missing` this
        answer has already printed. A supporting Missing carried on a Finding is
        the second case without being the first, and listing it as "never
        attempted" next to the masked-pixel sentence it just printed is the
        small lie :meth:`note_not_measured` is written to avoid.
        """
        return [
            q
            for q in self._not_measured
            if not self.ledger.has(q) and q not in self._gaps_emitted
        ]

    def not_measured_section(self, heading: str = "Not measured for this answer") -> "AnswerBuilder":
        outstanding = self._outstanding()
        if outstanding:
            self.section("not_measured", heading)
            self.text(
                "These were never attempted for this question, so there is nothing to report: "
                + ", ".join(outstanding)
                + "."
            )
        return self

    def build(self) -> Answer:
        self._close()
        return Answer(
            kind=self.kind,
            title=self.title,
            sections=tuple(self._sections),
            citations=tuple(self._citations),
            not_measured=tuple(self._outstanding()),
            degradation=self.ledger.degradation(),
        )


def _subject(crop: str | None, place: str | None) -> str:
    if crop and place:
        return f"{crop} at {place}"
    if crop:
        return crop
    if place:
        return f"the field at {place}"
    return "the confirmed field"


def _risk_cap(settings: Any | None) -> int:
    # The only setting the renderer reads (SPEC 4.3). Imported here so that
    # importing this module costs nothing but evidence + errors.
    if settings is not None:
        return int(settings.risk_cap)
    from ..config import get_settings

    return int(get_settings().risk_cap)


def _declared_cap(value: Any) -> int | None:
    """A producer's own cap, or ``None`` when it did not state a usable one."""
    try:
        cap = int(value)
    except (TypeError, ValueError):
        return None
    return cap if cap > 0 else None


def _risk_decision(risks: Any) -> tuple[tuple[Any, ...], tuple[Any, ...], int | None]:
    """What the producer decided: (to display, suppressed, the cap it applied).

    ``analysis/rules.py`` returns a ``RiskAssessment`` that has already ranked,
    deduplicated by ``issue_id`` and capped. Its ``risks`` are the rows it chose
    to show, its ``suppressed`` are the ones its cap took off the page, and its
    own ``cap`` travels with them: re-deriving a cap from settings here is how
    the renderer could put a suppressed row back (SPEC section 4.3 -- an 18-risk
    dump is a bug, and the engine's suppression is not the renderer's to undo).
    A bare sequence carries no such decision, so the cap comes back ``None`` and
    the caller applies the spec's own.
    """
    if risks is None:
        return (), (), None
    if isinstance(risks, Mapping):
        if "risks" in risks:
            return (
                tuple(risks.get("risks") or ()),
                tuple(risks.get("suppressed") or ()),
                _declared_cap(risks.get("cap")),
            )
        return (), (), None
    ranked = getattr(risks, "risks", None)
    if ranked is not None:
        return (
            tuple(ranked),
            tuple(getattr(risks, "suppressed", ()) or ()),
            _declared_cap(getattr(risks, "cap", None)),
        )
    return tuple(risks), (), None


def _risk_caveat(risks: Any) -> str:
    """The sentence that has to travel above a risk list, from either form.

    ``RiskAssessment`` carries it as the method ``caveat()`` and as the string
    under ``"caveat"`` in ``as_dict()``. Reading only the method meant a server
    that serialised before rendering printed the marked risks with nothing on
    the page to explain the mark.
    """
    if isinstance(risks, Mapping):
        caveat = risks.get("caveat")
    else:
        caveat = getattr(risks, "caveat", None)
        if callable(caveat):
            caveat = caveat()
    return str(caveat) if caveat else ""


def risk_rows(risks: Any) -> tuple[Any, ...]:
    """Every row an assessment holds, displayed and suppressed alike.

    This is the "is there anything to say about risk" test the answer builders
    use to decide whether to open the section. It is not the render list:
    :func:`render_risks` follows :func:`_risk_decision`, leaves a suppressed row
    off the page and counts it in the "N further risks" line, so the count never
    silently reads zero.
    """
    ranked, suppressed, _ = _risk_decision(risks)
    return ranked + suppressed


def render_risks(
    builder: AnswerBuilder,
    risks: Any,
    *,
    settings: Any | None = None,
) -> int:
    """Render a ranked risk list, capped, with the observation behind each one.

    SPEC section 4.3: ranked, deduplicated by ``issue_id``, capped at
    ``CROPUP_RISK_CAP``, and *each carries the observation that triggered it*.
    What is cut is stated, because an answer that quietly drops 13 of 18 risks
    is as dishonest as one that dumps all 18.

    ``risks`` is a sequence of rows or a whole ``RiskAssessment``. An assessment
    has already decided: its ``risks`` are rendered, its ``suppressed`` are
    counted and never shown, and its own ``cap`` is the one quoted. ``settings``
    is consulted only for a bare sequence, which arrives with no decision on it.
    The caveat that has to travel with a list of canopy-only candidates is
    printed above the list, and each risk then carries its own
    ``evidence_strength``, so the caveat has the mark it refers to beside it.

    A risk claiming to be triggered by a quantity with no Fact -- neither one it
    carries nor one on the ledger -- raises ``MissingProvenanceError``: the rules
    engine only fires on a measurement, so a trigger with no measurement behind
    it is a fabrication, not a display problem.
    """
    ranked, suppressed, cap = _risk_decision(risks)
    if cap is None:
        cap = _risk_cap(settings)
    views: list[RiskView] = []
    seen: set[str] = set()
    for risk in ranked:
        view = RiskView.adapt(risk)
        if view.issue_id and view.issue_id in seen:
            continue
        seen.add(view.issue_id)
        views.append(view)
    shown = views[:cap]
    if shown:
        caveat = _risk_caveat(risks)
        if caveat:
            builder.text(caveat)
    for view in shown:
        builder.line(
            "risk.headline",
            literals={
                "severity": view.severity or "Unrated",
                "name": view.name or "unnamed issue",
                "issue_id": view.issue_id or "no id",
            },
        )
        if view.detail:
            builder.line("risk.detail", literals={"detail": view.detail})
        for trigger in view.triggered_by:
            fact = trigger.fact if trigger.fact is not None else builder.ledger.fact(trigger.quantity)
            if fact is None:
                raise MissingProvenanceError(
                    trigger.quantity,
                    "risk.trigger",
                    f"risk {view.issue_id or view.name!r} claims a trigger that was never measured",
                )
            builder.line_with(
                "risk.trigger", {"trigger": fact}, literals={"quantity": trigger.label}
            )
        builder.line(
            "risk.evidence_strength",
            literals={"evidence_strength": view.evidence_strength or "not recorded"},
        )
        if view.recommendation:
            builder.line(
                "risk.action",
                literals={
                    "recommendation": view.recommendation,
                    "urgency": view.urgency or "no urgency recorded",
                },
            )
    cut = (len(views) - len(shown)) + len(suppressed)
    if cut:
        builder.text(
            f"{cut} further risks scored below these and are not shown; "
            f"the list is capped at {cap}."
        )
    return len(shown)


# --------------------------------------------------------------------------
# findings and advice
# --------------------------------------------------------------------------


def render_finding(
    builder: AnswerBuilder, finding: Any, *, supporting: bool = True
) -> bool:
    """Render one finding: its measurement, then the reading written from it.

    Returns True when the measurement itself was rendered. A finding whose
    evidence is a :class:`~cropup.evidence.Missing` prints the named absence
    instead -- and still prints its reading, because "there is no ET forecast
    for this field, and here is why" *is* the answer, not the lack of one.

    ``supporting`` adds a line per supporting Fact, and is **on** by default.
    Those numbers are usually already inside the reading -- the p10/p90 spread
    behind a canopy reading, the dekadal median behind an ET anomaly -- but a
    reading is routed as one literal text segment, so in there they are plain
    characters with no ``provenance`` attached. SPEC section 9 asks that *every*
    rendered number be hoverable for its instrument, date and resolution, and a
    Fact segment is the only thing in this module that carries those. Saying it
    twice is the price of the hover; printing it once with nothing behind it is
    not on offer. A supporting :class:`~cropup.evidence.Missing` prints as the
    named absence it is, so a number that could not be measured is named rather
    than quietly dropped. Pass ``supporting=False`` only where the reading has
    no number in it at all.
    """
    view = FindingView.adapt(finding)
    if isinstance(view.evidence, Fact):
        builder.line_with(
            "finding.measured", {"finding": view.evidence}, literals={"label": view.heading}
        )
    else:
        builder.gap(view.evidence)
    if view.reading:
        builder.line("finding.reading", literals={"reading": view.reading})
    if supporting:
        for item in view.supporting:
            if isinstance(item, Fact):
                builder.line_with(
                    "finding.supporting",
                    {"supporting": item},
                    literals={"quantity": item.quantity.replace("_", " ")},
                )
            elif isinstance(item, Missing):
                builder.gap(item)
    return view.measured


def render_findings(
    builder: AnswerBuilder,
    findings: Iterable[Any],
    *,
    supporting: bool = True,
    measured_only: bool = False,
) -> int:
    """Render a list of findings. Returns how many carried a measurement."""
    measured = 0
    for finding in findings or ():
        view = FindingView.adapt(finding)
        if measured_only and not view.measured:
            continue
        if render_finding(builder, view, supporting=supporting):
            measured += 1
    return measured


def render_advice(builder: AnswerBuilder, advice: Any) -> bool:
    """Render one advice line and the Facts it rests on.

    Returns True when the advice was actionable. A blocked line prints what it
    is waiting for and then hands each of those quantities to the machinery that
    names absences -- ``gap`` when the ledger diagnosed one, ``not_measured``
    when nothing was ever recorded. It is never rendered as advice with the
    uncertainty written into the prose.
    """
    view = AdviceView.adapt(advice)
    if view.text:
        builder.line("advice.text", literals={"text": view.text})
    if view.blocked_by:
        builder.line(
            "advice.blocked",
            literals={"missing": ", ".join(q.replace("_", " ") for q in view.blocked_by)},
        )
        for quantity in view.blocked_by:
            builder.gap(quantity)
            builder.note_not_measured(quantity)
        return False
    if view.urgency:
        builder.line("advice.urgency", literals={"urgency": view.urgency})
    for fact in view.evidence:
        builder.line_with(
            "advice.evidence",
            {"evidence": fact},
            literals={"quantity": fact.quantity.replace("_", " ")},
        )
    return True


def render_advice_list(builder: AnswerBuilder, advice: Iterable[Any]) -> int:
    """Render a list of advice lines. Returns how many were actionable."""
    actionable = 0
    for item in advice or ():
        if render_advice(builder, item):
            actionable += 1
    return actionable


# --------------------------------------------------------------------------
# the four answers
# --------------------------------------------------------------------------


def plant_health_answer(
    ledger: Ledger,
    *,
    crop: str | None = None,
    place: str | None = None,
    risks: Any = (),
    findings: Iterable[Any] = (),
    supporting: bool = True,
    settings: Any | None = None,
) -> Answer:
    """The field-health report (intent ``field_health_check``).

    ``risks`` is a ``RiskAssessment`` or a sequence of risk rows; ``findings``
    is ``PlantHealthResult.findings``. Both are optional: the measured sections
    are rendered from the ledger either way.
    """
    builder = AnswerBuilder(
        "plant_health",
        f"Field health: {_subject(crop, place)}",
        ledger,
        literals={"crop": crop or "the crop", "place": place or "this field"},
    )

    builder.section("canopy", "Canopy")
    builder.optional("plant_health.ndvi")
    builder.optional("plant_health.spread")
    builder.optional("plant_health.zones")
    builder.optional("plant_health.change")
    builder.optional("plant_health.scenes")

    builder.section("water", "Water and heat")
    builder.optional("plant_health.moisture")
    builder.optional("plant_health.rootzone")
    builder.optional("plant_health.rain")
    builder.optional("plant_health.thermal")

    builder.section("ground", "Soil and climate")
    builder.optional("plant_health.soil")
    builder.optional("plant_health.climate")

    findings = tuple(findings or ())
    if findings:
        builder.section("readings", "What this means")
        render_findings(builder, findings, supporting=supporting)

    if risk_rows(risks):
        builder.section("risks", "What to watch")
        render_risks(builder, risks, settings=settings)

    builder.stale_section()
    builder.gaps_section()
    builder.not_measured_section()
    return builder.build()


def irrigation_answer(
    ledger: Ledger,
    *,
    crop: str | None = None,
    place: str | None = None,
    findings: Iterable[Any] = (),
    advice: Iterable[Any] = (),
    risks: Any = (),
    supporting: bool = True,
    settings: Any | None = None,
) -> Answer:
    """The water-balance report (intent ``irrigation_advice``).

    The forecast section is where SPEC section 3.3 bites: ``fret/forecast/eto``
    and OpenET are CONUS-only, so outside the US the forecast line renders as a
    named absence. A US-only answer is never shown to a Tanzanian farmer -- and
    an ``Advice`` blocked on the missing forecast renders as the block, not as a
    hedged recommendation.
    """
    builder = AnswerBuilder(
        "irrigation",
        f"Water status: {_subject(crop, place)}",
        ledger,
        literals={"crop": crop or "the crop", "place": place or "this field"},
    )

    builder.section("balance", "Water balance")
    builder.optional("irrigation.et")
    builder.optional("irrigation.ratio")
    builder.optional("irrigation.anomaly")

    builder.section("store", "What the soil is holding")
    builder.optional("irrigation.rootzone")
    builder.optional("irrigation.pawc")
    builder.optional("irrigation.rain")

    builder.section("stress", "Stress")
    builder.optional("irrigation.esi")

    builder.section("forecast", "The next few days")
    builder.optional("irrigation.forecast")

    builder.section("context", "How this land is farmed")
    builder.optional("irrigation.regime")

    findings = tuple(findings or ())
    if findings:
        builder.section("readings", "What this means")
        render_findings(builder, findings, supporting=supporting)

    advice = tuple(advice or ())
    if advice:
        builder.section("advice", "What to do")
        render_advice_list(builder, advice)

    if risk_rows(risks):
        builder.section("risks", "What to watch")
        render_risks(builder, risks, settings=settings)

    builder.stale_section()
    builder.gaps_section()
    builder.not_measured_section()
    return builder.build()


def crop_selection_answer(
    ledger: Ledger,
    *,
    place: str | None = None,
    ranked: Sequence[Any] | None = None,
    crop: str | None = None,
    names: Mapping[str, str] | None = None,
    findings: Iterable[Any] = (),
    supporting: bool = True,
    settings: Any | None = None,
) -> Answer:
    """The crop-suitability report (intent ``crop_selection``).

    ``ranked`` is a ranking from either producer: ``CropSelectionResult.ranking``
    (rows carrying a crop name and its Fact) or the bare Facts
    ``geo.suitability.rank_crops`` returns. Both are read through
    :class:`CropRankingView`, and either way the Fact is what the row template
    binds, so a rendered rank still carries its instrument, date and resolution.
    When ``ranked`` is not given, the ledger's ``crop_suitability_*`` Facts are
    used in the order they were recorded, which is the order the ranker produced
    them.

    ``names`` overrides the displayed crop label, keyed either by the quantity's
    crop suffix (``coffee_arabica``) or by the crop name the producer supplied.
    """
    builder = AnswerBuilder(
        "crop_selection",
        f"Crop suitability at {place}" if place else "Crop suitability",
        ledger,
        literals={"place": place or "this field", "crop": crop or "the crop"},
    )
    names = dict(names or {})

    if ranked is None:
        ranked = [f for f in ledger.facts if f.quantity.startswith("crop_suitability_")]
    rows = [CropRankingView.adapt(row) for row in ranked]

    if rows:
        builder.section("ranking", "Best-suited crops here")
        for position, row in enumerate(rows, start=1):
            label = names.get(row.key) or names.get(row.crop) or row.crop
            builder.line_with(
                "crop_selection.rank",
                {"suitability": row.fact},
                literals={"rank": str(position), "crop": label},
            )

    if crop:
        builder.section("crop", f"{crop} in particular")
        builder.optional("crop_selection.crop")
        builder.optional("crop_selection.limiting")
        builder.optional("crop_selection.sowing")

    builder.section("context", "The land itself")
    builder.optional("crop_selection.context")

    findings = tuple(findings or ())
    if findings:
        builder.section("readings", "What this means")
        render_findings(builder, findings, supporting=supporting)

    builder.stale_section()
    builder.gaps_section()
    builder.not_measured_section()
    return builder.build()


def analysis_answer(
    result: Any,
    *,
    crop: str | None = None,
    place: str | None = None,
    names: Mapping[str, str] | None = None,
    supporting: bool = True,
    settings: Any | None = None,
) -> Answer:
    """Render whatever ``analysis/`` returned, as an :class:`Answer`.

    This is the one call a web handler needs: hand it a ``PlantHealthResult``,
    an ``IrrigationResult`` or a ``CropSelectionResult`` and it routes that
    result's ledger, findings, advice, ranking and risks through the templates
    above. The three are told apart by what they carry rather than by their
    type, because ``render`` does not import ``analysis`` (SPEC section 2).

    ``crop`` and ``place`` override what the result carries; ``place`` has no
    counterpart on the result at all, since the analysis works in coordinates
    and only the dialog layer knows what the farmer confirmed the field is
    called.
    """
    ledger = getattr(result, "ledger", None)
    if not isinstance(ledger, Ledger):
        raise FabricationError(
            "an analysis result is rendered from its Ledger; a "
            f"{type(result).__name__} offered a {type(ledger).__name__}",
            result,
        )
    crop = crop if crop is not None else getattr(result, "crop", None)
    findings = tuple(getattr(result, "findings", ()) or ())
    risks = getattr(result, "risks", ())

    if hasattr(result, "ranking"):
        return crop_selection_answer(
            ledger,
            place=place,
            ranked=tuple(result.ranking or ()),
            crop=crop,
            names=names,
            findings=findings,
            supporting=supporting,
            settings=settings,
        )
    if hasattr(result, "advice"):
        return irrigation_answer(
            ledger,
            crop=crop,
            place=place,
            findings=findings,
            advice=tuple(result.advice or ()),
            risks=risks,
            supporting=supporting,
            settings=settings,
        )
    return plant_health_answer(
        ledger,
        crop=crop,
        place=place,
        risks=risks,
        findings=findings,
        supporting=supporting,
        settings=settings,
    )


def rag_answer(
    result: Any,
    *,
    question: str | None = None,
    ledger: Ledger | None = None,
) -> Answer:
    """A knowledge answer: retrieved cards, quoted verbatim, with citations.

    ``result`` is a ``cropup.rag.retrieve.RetrievalResult`` (duck-typed, because
    ``render`` does not import ``rag``). Nothing here paraphrases, merges or
    shortens a retrieved agronomic claim -- the text is the card's body exactly
    as it was written (SPEC section 6).
    """
    query = question or getattr(result, "query", "") or ""
    snippets = tuple(getattr(result, "snippets", ()) or ())
    floor = float(getattr(result, "floor", 0.0) or 0.0)
    best = float(getattr(result, "best_score", 0.0) or 0.0)
    model_id = str(getattr(result, "model_id", "") or "unrecorded model")
    note = str(getattr(result, "note", "") or "")
    builder = AnswerBuilder(
        "rag",
        f"About: {query}" if query else "From the knowledge base",
        ledger if ledger is not None else Ledger(turn="rag"),
    )

    if not snippets:
        builder.section("nothing", "No cited answer")
        builder.text(
            "I have nothing in the knowledge base that is close enough to this question "
            "to quote, and I will not paraphrase something I did not retrieve."
        )
        if note:
            builder.text(note)
        builder.diagnostic(
            f"The closest card scored {best:.3f} against a floor of {floor:.2f}.",
            {
                "quantity": "rag_best_score",
                "value": round(best, 4),
                "floor": floor,
                "source_asset": model_id,
                "measured": True,
                "note": "cosine similarity in the local card index, not a measurement of the field",
            },
        )
        builder.text(
            "Try the questionnaire instead: with a location and a crop I can measure the field directly."
        )
        return builder.build()

    builder.section("answer", "What the knowledge base says")
    for snippet in snippets:
        title = str(getattr(snippet, "title", "") or "")
        text = str(getattr(snippet, "text", "") or "")
        citation = str(getattr(snippet, "citation", "") or "uncited")
        score = float(getattr(snippet, "score", 0.0) or 0.0)
        if title:
            builder.text(title)
        builder.quote(text, citation)
        builder.diagnostic(
            f"— {citation} (similarity {score:.3f}, floor {floor:.2f})",
            {
                "quantity": "rag_similarity",
                "value": round(score, 4),
                "floor": floor,
                "source_asset": model_id,
                "card_id": str(getattr(snippet, "card_id", "") or ""),
                "measured": True,
                "note": "cosine similarity in the local card index, not a measurement of the field",
            },
        )
    return builder.build()
