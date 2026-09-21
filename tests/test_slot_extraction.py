"""SPEC section 5.3: closed vocabularies, Swahili aliases, ambiguous short names
and the cross-slot arbitration.

A resolved location is what authorises an Earth Engine run (SPEC 4.4), so a crop
word that quietly becomes a place is an answer about somebody else's field. That
is the arbitration these tests pin.
"""

from __future__ import annotations

import datetime as dt

import pytest

from cropup.nlu import slots as S

TODAY = dt.date(2025, 8, 15)


def extract(text: str) -> S.SlotExtraction:
    return S.extract_slots(text, today=TODAY)


# --------------------------------------------------------------------------
# the Swahili vocabulary
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "swahili,crop",
    [
        ("mahindi", "Maize"),
        ("muhogo", "Cassava"),
        ("mpunga", "Rice"),
        ("nyanya", "Tomato"),
    ],
)
def test_swahili_aliases_resolve_to_the_canonical_crop(swahili, crop):
    best = extract(swahili).best("crop")
    assert best is not None, f"{swahili} left no actionable crop"
    assert best.value == crop
    assert best.exact is True
    assert best.score == 100.0
    assert best.source == S.SOURCE_CROP_VOCAB


def test_the_alias_survives_inside_a_whole_sentence():
    best = extract("mahindi yangu yana njano").best("crop")
    assert best is not None and best.value == "Maize"
    assert best.raw == "mahindi"


def test_find_crop_is_an_exact_lookup_over_the_committed_vocabulary():
    assert S.find_crop("mahindi").name == "Maize"
    assert S.find_crop("MAHINDI").name == "Maize"
    assert S.find_crop("not-a-crop") is None


def test_autocomplete_shows_why_a_row_was_returned():
    rows = S.suggest_crops("mahindi")
    assert rows and rows[0]["name"] == "Maize"
    assert rows[0]["matched_alias"] == "mahindi"


# --------------------------------------------------------------------------
# the seven ambiguous short names
# --------------------------------------------------------------------------


def test_the_gazetteer_flags_exactly_the_documented_ambiguous_names():
    flagged = sorted(p.name for p in S.load_places().entries if p.ambiguous)
    assert flagged == ["Bunda", "Hai", "Kilosa", "Lindi", "Mara", "Mpwapwa", "Same"]


@pytest.mark.parametrize("name", ["Hai", "Same", "Bunda", "Kilosa", "Lindi", "Mpwapwa"])
def test_an_ambiguous_name_alone_is_never_actionable(name):
    extraction = extract(f"my farm in {name}")
    assert extraction.top("location") is not None, name
    assert extraction.top("location").needs_confirmation is True
    assert extraction.best("location") is None, f"{name} was acted on without corroboration"
    assert "location" in extraction.unresolved()
    assert "more than one real place" in (extraction.top("location").note or "")


def test_a_corroborating_parent_region_makes_an_ambiguous_name_usable():
    extraction = extract("my maize in Hai, Kilimanjaro")
    hai = next(c for c in extraction.locations if c.value == "hai")
    assert hai.needs_confirmation is False
    assert hai.detail["corroborated_by"] == "Kilimanjaro Region"
    assert "corroborated by" in (hai.note or "")
    assert extraction.best("location") is hai


def test_a_named_region_around_two_districts_is_not_offered_as_the_field():
    extraction = extract("in Hai and Siha, Kilimanjaro")
    region = next(c for c in extraction.locations if c.value == "kilimanjaro")
    assert region.needs_confirmation is True
    assert "not the field itself" in (region.note or "")


def test_ambiguity_rides_on_every_autocomplete_row():
    rows = S.suggest_places("hai")
    assert rows
    assert any(row["name"] == "Hai" and row["ambiguous"] is True for row in rows)
    assert all("ambiguous" in row for row in rows)


# --------------------------------------------------------------------------
# cross-slot arbitration
# --------------------------------------------------------------------------


def test_fertilizer_for_pamba_leaves_no_actionable_location():
    """*pamba* is Swahili for cotton and also a gazetteer entry. 'for' is not a
    locative preposition, so nothing here authorises a run on a field."""
    extraction = extract("best fertilizer for pamba")
    assert extraction.best("crop").value == "Cotton"
    assert extraction.top("location") is not None  # the reading is kept, flagged
    assert extraction.top("location").needs_confirmation is True
    assert extraction.best("location") is None
    assert "location" not in extraction.filled()
    assert "location" in extraction.unresolved()


def test_my_maize_in_arusha_leaves_both_slots_unflagged():
    extraction = extract("my maize in Arusha is yellow")
    crop, location = extraction.best("crop"), extraction.best("location")
    assert crop is not None and crop.value == "Maize"
    assert location is not None and location.value == "arusha"
    assert crop.needs_confirmation is False
    assert location.needs_confirmation is False
    assert extraction.unresolved() == ()
    assert location.detail["lat"] and location.detail["lon"]


def test_a_crop_word_that_fuzzy_matches_a_village_flags_the_place():
    """*viazi* (potato) fuzzy-matches the real village Vianzi."""
    extraction = extract("I grow viazi")
    assert extraction.best("crop").value == "Potato"
    place = extraction.top("location")
    assert place is not None and place.value == "vianzi"
    assert place.needs_confirmation is True
    assert place.detail["conflicts_with"]["slot"] == "crop"
    assert place.detail["conflicts_with"]["value"] == "Potato"
    assert extraction.best("location") is None


def test_the_single_slot_views_do_not_arbitrate_and_extract_slots_does():
    """``extract_locations`` sees only the gazetteer; the reconciliation that
    stops a crop word becoming a field lives in ``extract_slots``."""
    alone = S.extract_locations("I grow viazi")
    assert alone and alone[0].value == "vianzi"
    assert "conflicts_with" not in alone[0].detail
    assert "conflicts_with" in extract("I grow viazi").top("location").detail


def test_a_locative_cue_lets_a_place_that_is_also_a_crop_win_its_span():
    """*karanga* is groundnut and a ward of Moshi; with two places named the
    dialog is asked which field, rather than one being picked by sort order."""
    extraction = extract("my karanga in Moshi")
    assert extraction.best("crop").value == "Groundnut"
    assert extraction.best("location") is None
    labels = {c.value for c in extraction.locations}
    assert {"karanga", "moshi"} <= labels
    assert all(c.needs_confirmation for c in extraction.locations)


def test_an_ordinary_english_word_is_not_read_as_a_place():
    extraction = extract("the same field needs water")
    assert extraction.best("location") is None


# --------------------------------------------------------------------------
# timeframes: no date is ever invented
# --------------------------------------------------------------------------


def test_a_named_season_resolves_to_a_season_with_no_dates():
    extraction = extract("when should I plant in masika")
    timeframe = extraction.top("timeframe")
    assert timeframe is not None
    assert timeframe.value == "masika (long rains)"
    assert timeframe.detail["start"] is None and timeframe.detail["end"] is None
    assert timeframe.needs_confirmation is True
    assert extraction.best("timeframe") is None


def test_a_real_window_resolves_to_real_dates_against_the_reference_date():
    timeframe = extract("what happened in the last 7 days").best("timeframe")
    assert timeframe is not None
    assert timeframe.detail["start"] == "2025-08-08"
    assert timeframe.detail["end"] == "2025-08-15"
    assert timeframe.detail["reference_date"] == "2025-08-15"


def test_a_bare_month_carries_no_year_and_so_no_dates():
    timeframe = extract("I planted in March").top("timeframe")
    assert timeframe is not None and timeframe.value == "march"
    assert timeframe.detail["start"] is None
    assert "no year given" in (timeframe.note or "")


# --------------------------------------------------------------------------
# the module's two refusals
# --------------------------------------------------------------------------


def test_nothing_is_extracted_from_a_sentence_about_neither():
    extraction = extract("what is the price of diesel today")
    assert extraction.best("crop") is None
    assert extraction.best("location") is None


def test_best_withholds_a_contested_reading_while_top_still_shows_it():
    extraction = extract("my farm in Same")
    assert extraction.top("location") is not None
    assert extraction.best("location") is None
    payload = extraction.as_dict()
    assert payload["filled"] == {}
    assert payload["unresolved"] == ["location"]
    assert payload["slots"]["location"][0]["needs_confirmation"] is True
