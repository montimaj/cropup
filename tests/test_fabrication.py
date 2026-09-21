"""SPEC section 10's fabrication tests, and SPEC section 4.2's central claim.

Two things are asserted, adversarially:

1. The renderer **raises** on a missing Fact. Not "logs a warning", not "omits
   the line" -- raises.
2. **No default value ever reaches output.** A raw float, a ``None``, a stale
   Fact and a ``Missing`` are each pushed through every public render entry
   point, and none of them produces a number on the page.

The failure this is written against is the vendored backend's
``crop_recommendations.py``, which interpolated pH 7.0 and sand 30% into
user-facing prose whenever soil was missing (SPEC 2.1 #4).
"""

from __future__ import annotations

import datetime as dt
import re

import pytest

from cropup.errors import FabricationError, MissingProvenanceError, ProvenanceError
from cropup.evidence import Fact, Ledger, Missing, MissingReason
from cropup.render import templates as T

from helpers import fact

#: Everything a caller might hand a slot instead of a Fact.
IMPOSTORS = (
    7.0,  # the vendored default pH
    30,  # the vendored default sand %
    None,
    "7.0",
    True,
    [6.2],
    {"value": 6.2, "unit": "pH"},
    {"quantity": "soil_ph", "value": 6.2, "source_asset": "isda"},  # a provenance dict
)

#: The four whole-answer entry points a web handler can reach.
ANSWER_BUILDERS = (
    ("plant_health", T.plant_health_answer),
    ("irrigation", T.irrigation_answer),
    ("crop_selection", T.crop_selection_answer),
)


def stale_fact(quantity: str = "evaporative_stress_index") -> Fact:
    return fact(
        quantity,
        -1.2,
        observed_on=dt.date.today() - dt.timedelta(days=500),
        stale_after_days=45,
        source_asset="projects/climate-engine/esi/4wk",
    )


# --------------------------------------------------------------------------
# 1. the renderer raises on a missing Fact
# --------------------------------------------------------------------------


def test_rendering_a_slot_with_no_fact_raises():
    with pytest.raises(MissingProvenanceError) as caught:
        T.render_template("Soil pH is {soil_ph}.", Ledger(), name="plant_health.soil")
    assert caught.value.slot == "soil_ph"
    assert caught.value.template == "plant_health.soil"


def test_rendering_a_slot_from_an_empty_mapping_raises():
    with pytest.raises(MissingProvenanceError):
        T.render_template("Soil pH is {soil_ph}.", {}, name="t")
    with pytest.raises(MissingProvenanceError):
        T.render_template("Soil pH is {soil_ph}.", None, name="t")


def test_rendering_a_slot_whose_quantity_is_a_named_gap_raises():
    ledger = Ledger()
    ledger.gap("soil_ph", MissingReason.MASKED, ("isda",), "no unmasked pixel")
    with pytest.raises(MissingProvenanceError) as caught:
        T.render_template("Soil pH is {soil_ph}.", ledger, name="t")
    assert "the pixel is masked" in caught.value.detail


@pytest.mark.parametrize("impostor", IMPOSTORS)
def test_a_slot_given_anything_but_a_fact_raises(impostor):
    with pytest.raises(ProvenanceError) as caught:
        T.render_template("Soil pH is {soil_ph}.", {"soil_ph": impostor}, name="t")
    # Never rendered, never coerced.
    assert isinstance(caught.value, (FabricationError, MissingProvenanceError))


def test_a_stale_fact_that_was_demoted_raises_like_any_other_absence():
    from cropup.evidence import demote_if_stale

    ledger = Ledger()
    ledger.add(demote_if_stale(stale_fact()))
    with pytest.raises(MissingProvenanceError) as caught:
        T.render_template(
            "Stress is {evaporative_stress_index}.", ledger, name="irrigation.esi"
        )
    assert "too old to use" in caught.value.detail


def test_every_template_in_the_table_raises_on_an_empty_ledger():
    """No template has a default branch anywhere in it."""
    empty = Ledger()
    checked = 0
    for name, template in T.TEMPLATES.items():
        if not T.template_slots(template):
            continue  # a literal-only template has no measurement to fake
        literals = {key: "x" for key in T.template_literals(template)}
        with pytest.raises(MissingProvenanceError):
            T.render_template(template, empty, name=name, literals=literals)
        checked += 1
    assert checked > 20


def test_a_literal_slot_will_not_take_a_number():
    """``{@crop}`` is text. A number describing the world must arrive as a Fact."""
    with pytest.raises(FabricationError):
        T.render_template("{@crop} is fine.", {}, literals={"crop": 7.0})
    with pytest.raises(ValueError):
        T.render_template("{@crop} is fine.", {}, name="t")


def test_render_gap_refuses_anything_but_a_missing():
    with pytest.raises(FabricationError):
        T.render_gap(fact("soil_ph", 6.2))  # type: ignore[arg-type]
    with pytest.raises(FabricationError):
        T.render_gap(7.0)  # type: ignore[arg-type]


def test_an_answer_is_built_from_a_ledger_and_nothing_else():
    for offender in ({}, None, [fact("ndvi", 0.5)], "ledger"):
        with pytest.raises(FabricationError):
            T.AnswerBuilder("k", "t", offender)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# 2. no default value ever reaches output
# --------------------------------------------------------------------------


#: Quantity names that carry a digit in the name itself (``ndvi_p10``,
#: ``et0``, ``precipitation_7d``). Naming one is not printing a measurement, so
#: they are removed before the page is searched for numbers.
def _without_quantity_names(text: str, quantities) -> str:
    for quantity in sorted(quantities, key=len, reverse=True):
        text = text.replace(quantity, "").replace(quantity.replace("_", " "), "")
    return text


def numbers_in(text: str, quantities=()) -> list[str]:
    """Every number a reader would take as a measurement."""
    return re.findall(r"\d+(?:\.\d+)?", _without_quantity_names(text, quantities))


def answer_numbers(answer) -> list[str]:
    named = set(answer.not_measured) | set(answer.gaps_named()) | set(answer.facts_used())
    named |= {
        str(gap.get("quantity") or "")
        for gap in (answer.degradation or {}).get("gaps", ())
    }
    return numbers_in(answer.text, named)


@pytest.mark.parametrize("kind,builder", ANSWER_BUILDERS)
def test_an_answer_over_an_empty_ledger_contains_no_number_at_all(kind, builder):
    answer = builder(Ledger(turn=kind), place="Arusha", crop="Maize")
    assert answer.facts_used() == ()
    assert answer_numbers(answer) == []
    assert answer.not_measured, "the quantities it wanted must still be named"
    for section in answer.sections:
        for line in section.lines:
            assert all(segment.kind != T.SEGMENT_FACT for segment in line.segments)


@pytest.mark.parametrize("kind,builder", ANSWER_BUILDERS)
def test_an_answer_over_a_ledger_of_gaps_prints_the_absences_not_numbers(kind, builder):
    ledger = Ledger(turn=kind)
    for quantity in ("ndvi", "soil_ph", "et_actual", "et0", "cropland_probability"):
        ledger.gap(quantity, MissingReason.MASKED, ("some/asset",), "no unmasked pixel")
    answer = builder(ledger, place="Arusha", crop="Maize")
    assert answer.facts_used() == ()
    assert answer_numbers(answer) == []
    assert set(answer.gaps_named()) & {"ndvi", "soil_ph"}
    assert "not available" in answer.text


def test_the_only_numbers_in_an_answer_come_from_facts():
    ledger = Ledger(turn="plant_health")
    ledger.record(
        Fact("ndvi", 0.62, "index", "COPERNICUS/S2_SR_HARMONIZED", dt.date(2025, 8, 1), 10.0)
    )
    ledger.gap("soil_ph", MissingReason.OUT_OF_COVERAGE, ("polaris",), "POLARIS is US-only")
    answer = T.plant_health_answer(ledger, crop="Maize", place="Arusha")

    assert answer.facts_used() == ("ndvi",)
    for section in answer.sections:
        for line in section.lines:
            for segment in line.segments:
                if segment.kind == T.SEGMENT_FACT:
                    assert segment.provenance is not None
                    assert segment.provenance["source_asset"]
                    assert segment.provenance["measured"] is True
    # The gap is printed as a sentence, and carries no number of its own.
    assert "soil ph: not available" in answer.text.lower()
    assert "7.0" not in answer.text


def test_a_missing_quantity_is_named_rather_than_dropped():
    ledger = Ledger()
    ledger.record(Fact("ndvi", 0.62, "index", "S2", dt.date(2025, 8, 1), 10.0))
    answer = T.plant_health_answer(ledger)
    assert "soil_ph" in answer.not_measured
    assert "soil_ph" in answer.as_dict()["not_measured"]


def test_a_rag_answer_with_nothing_retrieved_refuses_to_paraphrase():
    class Empty:
        query = "how much urea for maize"
        snippets = ()
        floor = 0.35
        best_score = 0.11
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        note = ""

    answer = T.rag_answer(Empty())
    assert answer.facts_used() == ()
    assert "will not paraphrase something I did not retrieve" in answer.text
    # The one number is a diagnostic about the retriever, and it says so.
    diagnostics = [
        segment
        for section in answer.sections
        for line in section.lines
        for segment in line.segments
        if segment.kind == T.SEGMENT_DIAGNOSTIC
    ]
    assert diagnostics
    for segment in diagnostics:
        assert "not a measurement of the field" in segment.provenance["note"]


def test_a_retrieved_snippet_is_quoted_verbatim_with_its_citation():
    class Snippet:
        title = "Urea top-dressing"
        text = "Top-dress 50 kg/ha of urea at knee height."
        citation = "disease_library.csv row GEN_N_DEF"
        score = 0.81
        card_id = "card-1"

    class Result:
        query = "urea"
        snippets = (Snippet(),)
        floor = 0.35
        best_score = 0.81
        model_id = "m"
        note = ""

    answer = T.rag_answer(Result())
    quotes = [
        segment
        for section in answer.sections
        for line in section.lines
        for segment in line.segments
        if segment.kind == T.SEGMENT_QUOTE
    ]
    assert len(quotes) == 1
    assert quotes[0].text == Snippet.text  # verbatim, not paraphrased
    assert quotes[0].citation == Snippet.citation
    assert Snippet.citation in answer.citations


# --------------------------------------------------------------------------
# the analysis-layer adapters
# --------------------------------------------------------------------------


def test_a_risk_trigger_with_no_measurement_behind_it_raises():
    """The rules engine only fires on a measurement, so a trigger with none is
    a fabrication, not a display problem."""
    builder = T.AnswerBuilder("plant_health", "t", Ledger())
    risk = {
        "issue_id": "MLN",
        "name": "Maize Lethal Necrosis",
        "severity": "Severe",
        "triggers": [{"quantity": "ndvi"}],
    }
    with pytest.raises(MissingProvenanceError) as caught:
        T.render_risks(builder, [risk])
    assert caught.value.slot == "ndvi"
    assert "never measured" in caught.value.detail


def test_a_risk_trigger_that_names_nothing_at_all_raises():
    with pytest.raises(FabricationError):
        T.TriggerView.adapt({"threshold": 0.3})


def test_a_finding_must_carry_a_fact_or_a_named_missing():
    for evidence in (0.62, None, {"value": 0.62}):
        with pytest.raises(FabricationError):
            T.FindingView.adapt({"key": "ndvi", "label": "NDVI", "evidence": evidence})
    view = T.FindingView.adapt(
        {"key": "ndvi", "label": "NDVI", "evidence": Missing("ndvi", MissingReason.MASKED)}
    )
    assert view.measured is False


def test_a_crop_ranking_row_must_carry_the_fact_that_scored_it():
    with pytest.raises(FabricationError):
        T.CropRankingView.adapt({"crop": "Maize", "fact": 82.0})
    with pytest.raises(FabricationError):
        T.CropRankingView.adapt({"crop": "Maize", "fact": {"value": 82.0}})
    row = T.CropRankingView.adapt(fact("crop_suitability_maize", 82.0, "%"))
    assert row.crop == "Maize" and isinstance(row.fact, Fact)


def test_advice_is_rendered_from_facts_and_a_blocked_line_is_never_softened():
    with pytest.raises(FabricationError):
        T.AdviceView.adapt({"key": "irrigate", "text": "water now", "evidence": [0.3]})

    ledger = Ledger()
    ledger.gap("et0_forecast", MissingReason.OUT_OF_COVERAGE, ("fret",), "CONUS only")
    builder = T.AnswerBuilder("irrigation", "t", ledger)
    actionable = T.render_advice(
        builder,
        {"key": "irrigate", "text": "How much to apply", "blocked_by": ("et0_forecast",)},
    )
    assert actionable is False
    answer = builder.build()
    assert "cannot be advised here until these are measured" in answer.text
    assert answer_numbers(answer) == []


def test_an_analysis_result_without_a_ledger_cannot_be_rendered():
    class NoLedger:
        ledger = {"soil_ph": 7.0}

    with pytest.raises(FabricationError):
        T.analysis_answer(NoLedger())


# --------------------------------------------------------------------------
# staleness is shown, not hidden (SPEC 3.3)
# --------------------------------------------------------------------------


def test_an_old_reading_that_is_rendered_carries_its_age_on_the_page():
    ledger = Ledger("irrigation")
    ledger.record(fact("et_actual", 12.0, "mm", observed_on=dt.date(2020, 1, 1)))
    ledger.record(fact("et0", 40.0, "mm", observed_on=None))
    answer = T.irrigation_answer(ledger, place="Arusha")
    assert "How old these numbers are" in [section.heading for section in answer.sections]
    assert "no freshness limit is defined" in answer.text
