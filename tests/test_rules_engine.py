"""SPEC section 4.3: ranked, deduplicated by ``issue_id``, capped at 5, and each
surfaced risk carries the observation that triggered it.

The bug being tested against is SPEC section 2.1 #8: 288 of the live rules key
on one NDVI threshold, so a single low reading fired about eighteen risks at
once -- "Maize Lethal Necrosis" and "Heavy Metal Contamination" on the same
screen. An 18-risk dump is explicitly a bug.
"""

from __future__ import annotations

import datetime as dt

import pytest

from cropup.analysis import rules as R
from cropup.evidence import Fact, Ledger, Missing, MissingReason

NDVI = R.Condition("ndvi", "<", 0.3, R.FIELD_SPECS["ndvi"])
PH = R.Condition("ph", "<", 5.5, R.FIELD_SPECS["ph"])


def rule(issue_id: str, name: str, severity: str, conditions, crop: str = "Generic") -> R.Rule:
    return R.Rule(
        issue_id=issue_id,
        crop=crop,
        name=name,
        category="Test",
        causal_agent="agent",
        severity=severity,
        urgency="now",
        notes="detail",
        recommendation=f"do something about {name}",
        logic="AND",
        conditions=tuple(conditions),
    )


@pytest.fixture
def observed() -> Ledger:
    led = Ledger(turn="rules")
    led.record(
        Fact("ndvi", 0.18, "index", "COPERNICUS/S2_SR_HARMONIZED", dt.date(2025, 8, 1), 10.0)
    )
    led.record(Fact("soil_ph", 5.1, "pH", "ISDASOIL/Africa/v1/ph", None, 30.0))
    return led


# --------------------------------------------------------------------------
# the cap
# --------------------------------------------------------------------------


def test_the_real_library_on_one_low_ndvi_is_capped_at_five(observed):
    """The regression: one NDVI reading used to fire ~18 risks at once."""
    assessment = R.run_rules(observed, crop="Maize")
    assert assessment.cap == 5
    assert assessment.fired > 5  # the library really does fire a pile of rules
    assert len(assessment.risks) == 5
    assert assessment.capped is True
    # Nothing is quietly dropped: what the cap removed is still counted.
    assert len(assessment.risks) + len(assessment.suppressed) == assessment.fired


def test_the_cap_is_configurable_and_the_remainder_is_kept_not_discarded(observed):
    rules = [rule(f"ID{i}", f"N{i}", "High", (NDVI,)) for i in range(9)]
    assessment = R.run_rules(observed, crop=None, rules=rules, cap=2)
    assert [r.issue_id for r in assessment.risks] == ["ID0", "ID1"]
    assert len(assessment.suppressed) == 7
    assert assessment.fired == 9
    assert "7 below the cap" in assessment.summary()


# --------------------------------------------------------------------------
# deduplication by issue_id
# --------------------------------------------------------------------------


def test_two_rules_with_one_issue_id_collapse_to_the_better_supported_reading(observed):
    thin = rule("DUP", "canopy only", "Severe", (NDVI,))
    corroborated = rule("DUP", "canopy plus soil", "Low", (NDVI, PH))
    assessment = R.run_rules(observed, crop=None, rules=[thin, corroborated])
    assert len(assessment.risks) == 1
    surviving = assessment.risks[0]
    assert surviving.issue_id == "DUP"
    # Severity alone would have kept the Severe one; evidence tier wins first.
    assert surviving.name == "canopy plus soil"
    assert surviving.evidence_strength == "corroborated"


def test_no_issue_id_appears_twice_in_a_real_run(observed):
    assessment = R.run_rules(observed, crop="Maize")
    ids = [r.issue_id for r in assessment.risks]
    assert len(ids) == len(set(ids))
    all_ids = ids + [r.issue_id for r in assessment.suppressed]
    assert len(all_ids) == len(set(all_ids))


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------


def test_a_corroborated_risk_outranks_a_more_severe_canopy_only_one(observed):
    rules = [
        rule("CANOPY", "Maize Lethal Necrosis", "Severe", (NDVI,)),
        rule("SOIL", "Acid soil", "Medium", (NDVI, PH)),
    ]
    assessment = R.run_rules(observed, crop=None, rules=rules)
    assert [r.issue_id for r in assessment.risks] == ["SOIL", "CANOPY"]
    assert assessment.risks[0].evidence_strength == "corroborated"
    assert assessment.risks[1].evidence_strength == "canopy_only"


def test_severity_orders_risks_inside_one_evidence_tier(observed):
    rules = [
        rule("LOW", "low", "Low", (NDVI,)),
        rule("SEV", "severe", "Severe", (NDVI,)),
        rule("MED", "medium", "Medium", (NDVI,)),
    ]
    assessment = R.run_rules(observed, crop=None, rules=rules)
    assert [r.issue_id for r in assessment.risks] == ["SEV", "MED", "LOW"]
    assert R.severity_rank("Severe") < R.severity_rank("Low")
    assert R.severity_rank("something nobody wrote down") == len(R.SEVERITY_ORDER)


def test_a_list_resting_only_on_the_canopy_carries_its_caveat(observed):
    assessment = R.run_rules(observed, crop=None, rules=[rule("A", "a", "Severe", (NDVI,))])
    assert assessment.all_unconfirmed is True
    caveat = assessment.caveat()
    assert caveat and "candidates to scout for, not diagnoses" in caveat


# --------------------------------------------------------------------------
# every surfaced risk carries its triggering observation
# --------------------------------------------------------------------------


def test_every_surfaced_risk_names_the_measurement_that_fired_it(observed):
    assessment = R.run_rules(observed, crop="Maize")
    assert assessment.risks
    for risk in assessment.risks:
        assert risk.triggers, risk.issue_id
        for trigger in risk.triggers:
            assert isinstance(trigger.fact, Fact)
            assert trigger.fact.source_asset  # the instrument
            assert trigger.condition.value is not None  # the threshold
            assert trigger.fact.render() in trigger.text
        assert risk.evidence_line()


def test_a_trigger_carries_the_full_provenance_of_its_fact(observed):
    assessment = R.run_rules(observed, crop=None, rules=[rule("A", "a", "High", (NDVI, PH))])
    payload = assessment.risks[0].as_dict()
    quantities = {t["quantity"] for t in payload["triggers"]}
    assert quantities == {"ndvi", "soil_ph"}
    for trigger in payload["triggers"]:
        assert trigger["provenance"]["source_asset"]
        assert trigger["provenance"]["measured"] is True


# --------------------------------------------------------------------------
# what the port fixed
# --------------------------------------------------------------------------


def test_a_rule_whose_field_was_never_measured_is_undecided_not_false():
    """The vendored engine answered False, so a field with no data looked
    healthier than a field with bad data."""
    led = Ledger()
    led.record(Fact("ndvi", 0.18, "index", "S2", dt.date(2025, 8, 1), 10.0))
    assessment = R.run_rules(led, crop=None, rules=[rule("A", "a", "High", (NDVI, PH))])
    assert assessment.risks == ()
    assert len(assessment.unevaluated) == 1
    undecided = assessment.unevaluated[0]
    assert "ph" in undecided.waiting_for
    assert any("soil pH" in reason for reason in undecided.reasons)


def test_a_missing_measurement_is_undecided_and_says_why():
    led = Ledger()
    led.record(Fact("ndvi", 0.18, "index", "S2", dt.date(2025, 8, 1), 10.0))
    led.gap("soil_ph", MissingReason.OUT_OF_COVERAGE, ("polaris",), "US only")
    assessment = R.run_rules(led, crop=None, rules=[rule("A", "a", "High", (NDVI, PH))])
    assert assessment.risks == ()
    assert "outside the data's coverage" in " ".join(assessment.unevaluated[0].reasons)


def test_crop_none_evaluates_the_generic_rules_and_counts_what_it_skipped(observed):
    """The vendored engine raised AttributeError when crop_type was None."""
    assessment = R.run_rules(observed, crop=None)
    assert assessment.crop is None
    assert assessment.crop_rules_skipped > 0
    assert all(not r.crop_specific for r in assessment.risks)


def test_a_threshold_is_only_compared_in_the_unit_it_was_written_in():
    """cec_mmol_kg against cmol(+)/kg is a factor-of-ten error that reads as a
    plausible number, so an undeclared unit leaves the condition undecided."""
    cec = R.Condition("cec_mmol_kg", "<", 10.0, R.FIELD_SPECS["cec_mmol_kg"])
    led = Ledger()
    led.record(Fact("soil_cec", 6.0, "cmol(+)/kg", "ISDASOIL/Africa/v1/cec", None, 30.0))
    fired = R.run_rules(led, crop=None, rules=[rule("K", "Potassium Deficiency", "High", (cec,))])
    assert [r.issue_id for r in fired.risks] == ["K"]

    wrong_unit = Ledger()
    wrong_unit.record(Fact("soil_cec", 6.0, "furlongs", "X", None, 30.0))
    undecided = R.run_rules(
        wrong_unit, crop=None, rules=[rule("K", "Potassium Deficiency", "High", (cec,))]
    )
    assert undecided.risks == ()
    assert "furlongs" in " ".join(undecided.unevaluated[0].reasons)


def test_the_engine_never_sees_a_bare_float():
    """Observations arrive as Facts. A raw number is simply not an observation."""
    assessment = R.run_rules({"ndvi": 0.18}, crop=None, rules=[rule("A", "a", "High", (NDVI,))])
    assert assessment.risks == ()
    assert assessment.unevaluated
    assert "carries no provenance" in " ".join(assessment.unevaluated[0].reasons)


def test_run_rules_records_its_undecided_fields_as_named_gaps():
    led = Ledger()
    led.record(Fact("ndvi", 0.18, "index", "S2", dt.date(2025, 8, 1), 10.0))
    R.run_rules(led, crop=None, rules=[rule("A", "a", "High", (NDVI, PH))], ledger=led)
    gap = [g for g in led.gaps if g.quantity == "soil_ph"]
    assert len(gap) == 1
    assert gap[0].reason is MissingReason.NOT_REQUESTED
    assert "disease library" in (gap[0].detail or "")


def test_drainage_is_derived_from_texture_and_says_so():
    led = Ledger()
    led.record(Fact("soil_texture_class", "clay loam", "", "ISDASOIL/Africa/v1/texture_class", None, 30.0))
    R.derive_drainage(led)
    derived = led.fact("soil_drainage")
    assert derived is not None and derived.value == "Moderate"
    assert "inferred" in (derived.note or "")
    assert derived.derived_from == ("soil_texture_class",)


def test_drainage_inherits_the_reason_its_ingredient_was_missing():
    led = Ledger()
    led.gap("soil_texture_class", MissingReason.SOURCE_FAILED, ("isda",), "leg raised")
    R.derive_drainage(led)
    missing = led.get("soil_drainage")
    assert isinstance(missing, Missing)
    assert missing.reason is MissingReason.SOURCE_FAILED
