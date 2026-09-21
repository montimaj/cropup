"""Crop, location and timeframe extraction from a farmer's sentence.

SPEC section 5.3: closed vocabularies beat NER here. The crop vocabulary is the
committed 134-crop / 379-alias table including Swahili (``mahindi`` -> Maize,
``nyanya`` -> Tomato); the gazetteer is the k-anonymised 1,073-place table.
Matching is rapidfuzz over the aliases, not a model.

Two things this module refuses to do:

* It never returns a single silent guess. :func:`extract_slots` returns ranked
  candidates with the score that produced them; :meth:`SlotExtraction.best`
  hands back a value only when nothing contests it, and ``None`` otherwise, so
  the dialog layer has to ask rather than assume.
* It never invents a date. "this week" resolves to real dates; "masika" or
  "next season" resolve to a named season with ``start``/``end`` of ``None``,
  because the boundaries of a season are not a fact this module holds.

Seven gazetteer entries are flagged ``ambiguous`` (Hai, Same, Bunda, Kilosa,
Lindi, Mara, Mpwapwa): short names that several real places share. The flag is
an editorial list in ``tools/build_gazetteer.py``, not a measurement -- the
artifact publishes no ping dispersion to derive one from (SPEC section 7) --
and it is the only ambiguity signal this module has. Those are only accepted
when the text corroborates them with the parent region, the country, or a
neighbouring place; otherwise they come back needing confirmation.

The two vocabularies also overlap, and the overlap is not symmetric noise: the
Swahili for groundnut (*karanga*) and cotton (*pamba*) are gazetteer entries,
and *viazi* (potato), *wimbi* (finger millet) and *mgomba* (banana) fuzzy-match
the real places Vianzi, Mwimbi and Momba. A resolved location is what authorises
an Earth Engine run (SPEC section 4.4), so a crop word that quietly becomes a
place is an answer about the wrong field. :func:`extract_slots` therefore
reconciles the two slots over the spans they share -- see
:func:`_reconcile_crop_location` -- while :func:`extract_crops` and
:func:`extract_locations` remain single-slot views that each see only their own
vocabulary.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

from rapidfuzz import fuzz, process

from ..config import Settings, get_settings
from ..errors import DataFileError

__all__ = [
    "SLOT_NAMES",
    "CropEntry",
    "Place",
    "SlotCandidate",
    "SlotExtraction",
    "extract_slots",
    "extract_crops",
    "extract_locations",
    "extract_timeframes",
    "load_crops",
    "load_places",
    "find_crop",
    "find_place",
    "suggest_crops",
    "suggest_places",
    "vocab_status",
    "reset",
]

SLOT_NAMES: tuple[str, ...] = ("crop", "location", "timeframe")

SOURCE_CROP_VOCAB = "crop_vocabulary"
SOURCE_GAZETTEER = "gazetteer"
SOURCE_TIME_PATTERN = "time_pattern"

# Words that are also a crop alias or a gazetteer name but are almost always
# being used as ordinary language ("the same field", "cook", "soy" in "soya
# sauce" is not a farm). They are never fuzzy-matched, and an exact hit on one
# is marked weak: it needs a locative cue or a corroborating region.
_COMMON_WORDS = frozenset(
    """
    a an and are as at be been before but by can could did do does for from get give go
    had has have he her here him his how i if in into is it its just know like make many
    me more most my need no not now of on once one only or other our out over please
    same see she should so some still such take tell than that the their them then there
    these they this those to too under until up us use very want was we were what when
    where which while who why will with would you your yes ok okay thanks thank hello hi
    field farm crop crops plant plants planting soil water rain season leaf leaves cook
    """.split()
)

# A name we matched right after one of these is being used as a place. Only
# genuinely locative prepositions belong here: "for" and "of" are the two
# commonest non-locative prepositions in a farmer's question ("fertilizer for
# pamba", "the price of rice"), and reading them as locative hands the span to
# the gazetteer -- which is what authorises an Earth Engine run (SPEC 4.4) on a
# field the farmer never named.
_LOCATIVE_CUES = frozenset(
    {"in", "at", "near", "around", "from", "within", "kwa", "huko", "katika", "mkoa", "wilaya"}
)
# ... and these chain a place list together: "in Hai and Siha, Kilimanjaro".
_LIST_CUES = frozenset({"and", "or", "na", "plus", "also"})

# Suffixes that a gazetteer parent carries but a farmer does not say.
_PARENT_SUFFIXES = ("region", "district", "county", "province", "governorate", "municipality", "city")

_MIN_FUZZY_CROP_LEN = 4
_MIN_FUZZY_PLACE_LEN = 5
_MAX_CROP_NGRAM = 3  # longest alias is 3 tokens
_MAX_PLACE_NGRAM = 5  # longest gazetteer name is 5 tokens

# A name written inside a longer unsegmented token -- "maizefield", "inarusha".
# ``fuzz.ratio`` cannot see it (the surrounding letters sink the score) and an
# unguarded substring search sees far too much, so three guards apply: the name
# must be this long, the token must be at least this much longer than the name,
# and the match must start or end the token, because a compound is a
# concatenation and an interior hit ("disimpassioned" -> passion) is noise.
# Measured over 40,000 rare English words from /usr/share/dict/words the guards
# hold the false-positive rate to 0.22% for crops and 0.08% for places -- and
# every hit is returned needing confirmation regardless, so the worst a survivor
# costs is one question.
_MIN_COMPOUND_NAME_LEN = 5
_MIN_COMPOUND_GAP = 2
_COMPOUND_SCORE_FLOOR = 95.0

# Autocomplete (SPEC section 8). A suggestion is a menu entry, not an assertion
# about the farmer's field: it is never acted on, and the farmer picks. The
# floor is the measured line between a real completion of a half-typed name
# ("maiz" -> maize at 88.9, "arusa" -> Arusha at 90.9) and token noise ("toma"
# -> Dodoma at 77.1, "kilim" -> pilipili manga at 72).
_SUGGEST_LIMIT_MAX = 50
_SUGGEST_SCORE_FLOOR = 85.0
# Below this many characters a fuzzy distance says nothing -- every short name is
# two edits from every other -- so a short query only matches names that contain
# it. "aru" offers Arusha, not Rubber.
_SUGGEST_SUBSTRING_ONLY_LEN = 3
# What rapidfuzz is asked for, so that a substring match below the floor is
# still seen and can be accepted on its own terms.
_SUGGEST_SUBSTRING_FLOOR = 70.0
# rapidfuzz has to return more than one page of hits, or re-ranking a one-letter
# query only re-orders whatever arbitrary slice came back.
_SUGGEST_POOL = 200

_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, Any, float]] = {}  # name -> (loaded_at, value, mtime)


# --------------------------------------------------------------------------
# vocabularies
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CropEntry:
    """One canonical crop plus the backends that actually know it."""

    name: str
    key: str
    aliases: tuple[str, ...]
    in_suitability: bool
    in_disease_library: bool
    rules_fallback: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "key": self.key,
            "aliases": list(self.aliases),
            "in_suitability": self.in_suitability,
            "in_disease_library": self.in_disease_library,
            "rules_fallback": self.rules_fallback,
        }


@dataclass(frozen=True)
class Place:
    """One k-anonymised gazetteer entry (SPEC section 7)."""

    key: str
    name: str
    lat: float
    lon: float
    country: str | None
    parent: str | None
    admin_level: int | None
    # How well attested the entry is, as the coarse band the gazetteer publishes
    # ("5-9", "10-49", ...). ``None`` means the gazetteer published no band,
    # i.e. nothing attests the entry at or above its k-anonymity floor: the case
    # for a seeded town centroid -- public geography, not an unvisited place.
    # The exact count and the ping dispersion are personal data and are
    # deliberately not in the file (SPEC section 7), so they are not modelled
    # here either: there is nothing to default them to.
    support_band: str | None
    ambiguous: bool
    sources: tuple[str, ...]

    @property
    def parent_stem(self) -> str | None:
        """Parent without its administrative suffix: "Kilimanjaro Region" -> "kilimanjaro"."""
        if not self.parent:
            return None
        stem = self.parent.lower().strip()
        for suffix in _PARENT_SUFFIXES:
            if stem.endswith(" " + suffix):
                stem = stem[: -len(suffix) - 1].strip()
                break
        return stem or None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "country": self.country,
            "parent": self.parent,
            "admin_level": self.admin_level,
            "support_band": self.support_band,
            "ambiguous": self.ambiguous,
            "sources": list(self.sources),
        }


@dataclass(frozen=True)
class _CropVocab:
    entries: tuple[CropEntry, ...]
    by_key: Mapping[str, CropEntry]
    by_alias: Mapping[str, CropEntry]
    alias_list: tuple[str, ...]  # rapidfuzz choices, fuzzy-eligible aliases only
    compound_list: tuple[str, ...]  # of those, the ones that can hide in a word


@dataclass(frozen=True)
class _PlaceVocab:
    entries: tuple[Place, ...]
    by_key: Mapping[str, Place]
    name_list: tuple[str, ...]  # rapidfuzz choices, fuzzy-eligible names only
    compound_list: tuple[str, ...]  # of those, the ones that can hide in a word
    parent_stems: frozenset[str]
    band_rank: Mapping[str, int]  # support band -> 1-based ordinal, best last


def _read_json(settings: Settings, name: str) -> Any:
    path = settings.require_data_file(name)
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise DataFileError(path, f"{type(exc).__name__}: {exc}") from exc


def _cached(name: str, builder: Callable[[], Any], settings: Settings) -> Any:
    """TTL cache keyed on CROPUP_VOCAB_CACHE_TTL_S, invalidated by file mtime."""
    path = settings.require_data_file(name)
    mtime = path.stat().st_mtime
    now = time.monotonic()
    hit = _CACHE.get(name)
    if hit is not None:
        loaded_at, value, cached_mtime = hit
        fresh = now - loaded_at < settings.vocab_cache_ttl_s
        if fresh and cached_mtime == mtime:
            return value
    with _LOCK:
        hit = _CACHE.get(name)
        if hit is not None and time.monotonic() - hit[0] < settings.vocab_cache_ttl_s and hit[2] == mtime:
            return hit[1]
        value = builder()
        _CACHE[name] = (now, value, mtime)
        return value


def _build_crop_vocab(settings: Settings) -> _CropVocab:
    raw = _read_json(settings, "crops")
    try:
        rows = raw["crops"]
    except (TypeError, KeyError) as exc:
        raise DataFileError(settings.crops_path, "no 'crops' array") from exc
    entries: list[CropEntry] = []
    by_alias: dict[str, CropEntry] = {}
    fuzzy: set[str] = set()
    for row in rows:
        entry = CropEntry(
            name=row["name"],
            key=row["key"],
            aliases=tuple(row.get("aliases") or ()),
            in_suitability=bool(row.get("in_suitability")),
            in_disease_library=bool(row.get("in_disease_library")),
            rules_fallback=row.get("rules_fallback"),
        )
        entries.append(entry)
        for alias in (entry.name.lower(), *entry.aliases):
            alias = _squash(alias)
            if alias:
                by_alias.setdefault(alias, entry)
                if len(alias) >= _MIN_FUZZY_CROP_LEN and alias not in _COMMON_WORDS:
                    fuzzy.add(alias)
    return _CropVocab(
        entries=tuple(entries),
        by_key={e.key: e for e in entries},
        by_alias=by_alias,
        alias_list=tuple(sorted(fuzzy)),
        compound_list=tuple(sorted(_compound_choices(fuzzy))),
    )


def _build_place_vocab(settings: Settings) -> _PlaceVocab:
    raw = _read_json(settings, "gazetteer")
    try:
        rows = raw["places"]
    except (TypeError, KeyError) as exc:
        raise DataFileError(settings.gazetteer_path, "no 'places' array") from exc
    entries: list[Place] = []
    by_key: dict[str, Place] = {}
    fuzzy: set[str] = set()
    stems: set[str] = set()
    for row in rows:
        place = Place(
            key=_squash(row["key"]),
            name=row["name"],
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            country=row.get("country"),
            parent=row.get("parent"),
            admin_level=row.get("admin_level"),
            support_band=row.get("support_band"),
            ambiguous=bool(row.get("ambiguous")),
            sources=tuple(row.get("sources") or ()),
        )
        entries.append(place)
        by_key.setdefault(place.key, place)
        if len(place.key) >= _MIN_FUZZY_PLACE_LEN and place.key not in _COMMON_WORDS:
            fuzzy.add(place.key)
        stem = place.parent_stem
        if stem:
            stems.add(stem)
    # The file states the ascending order of its own support bands, so the
    # ranking is read from the artifact rather than duplicated here and left to
    # drift when tools/build_gazetteer.py changes the table.
    bands = (raw.get("privacy") or {}).get("support_bands") or ()
    return _PlaceVocab(
        entries=tuple(entries),
        by_key=by_key,
        name_list=tuple(sorted(fuzzy)),
        compound_list=tuple(sorted(_compound_choices(fuzzy))),
        parent_stems=frozenset(stems),
        band_rank={str(label): i for i, label in enumerate(bands, start=1)},
    )


def _band_rank(vocab: _PlaceVocab, place: Place) -> int:
    """Where ``place``'s support band sits in the published order; 0 if none.

    0 is "the gazetteer published no band", not "zero pings": a seeded town
    centroid is public geography that no personal record attests above the
    k-anonymity floor. It sorts last because it is not attestation, not because
    the place is unimportant.
    """
    return vocab.band_rank.get(place.support_band or "", 0)


def _compound_choices(names: Iterable[str]) -> set[str]:
    """The names long enough, and solid enough, to hide inside another word.

    A multi-word name cannot: the tokenizer would have split it already.
    """
    return {
        name for name in names if len(name) >= _MIN_COMPOUND_NAME_LEN and " " not in name
    }


def load_crops(settings: Settings | None = None) -> _CropVocab:
    settings = settings or get_settings()
    return _cached("crops", lambda: _build_crop_vocab(settings), settings)


def load_places(settings: Settings | None = None) -> _PlaceVocab:
    settings = settings or get_settings()
    return _cached("gazetteer", lambda: _build_place_vocab(settings), settings)


def find_crop(name: str, settings: Settings | None = None) -> CropEntry | None:
    """Exact canonical/alias lookup, for a value the farmer already confirmed."""
    vocab = load_crops(settings)
    key = _squash(name)
    return vocab.by_key.get(key) or vocab.by_alias.get(key)


def find_place(key: str, settings: Settings | None = None) -> Place | None:
    return load_places(settings).by_key.get(_squash(key))


def _suggestion_rank(name: str, needle: str, score: float) -> tuple[int, int, int, float]:
    """Sort key for one autocomplete row, biggest first.

    Exact first, then the names the query is a prefix of -- and among those the
    shortest, because "mosh" completing to Moshi is a tighter reading than
    Moshono however rapidfuzz scores the two. Everything else falls back to the
    score. The third element is only ever compared inside one prefix class, so
    mixing a length into it cannot promote a non-prefix match.
    """
    prefix = name.startswith(needle)
    return (int(name == needle), int(prefix), -len(name) if prefix else 0, score)


def _suggestion_accepts(name: str, needle: str, score: float) -> bool:
    """Whether one match is worth offering.

    A name that literally contains what was typed is always worth offering,
    however rapidfuzz scores it -- half of "green gram" scores 82. Anything else
    has to clear the floor, and a query too short to carry a distance has to
    contain the typing outright.
    """
    if needle in name:
        return True
    return score >= _SUGGEST_SCORE_FLOOR and len(needle) > _SUGGEST_SUBSTRING_ONLY_LEN


def suggest_crops(
    query: str, *, limit: int = 10, settings: Settings | None = None
) -> list[dict[str, Any]]:
    """Autocomplete rows for ``GET /api/vocab/crops`` (SPEC section 8).

    Ranks the 379 aliases -- Swahili included -- exact, then prefix, then fuzzy,
    and collapses them to one row per canonical crop. A blank query lists the
    vocabulary alphabetically, which is what an empty search box wants.

    Each row is :meth:`CropEntry.as_dict` plus ``matched_alias`` and ``score``,
    so the UI can show *mahindi -> Maize* instead of an unexplained rename. The
    rows are plain JSON: this is a menu, not an assertion about a field, and
    nothing here fills a slot -- :func:`extract_slots` does that, and the farmer
    confirms it.
    """
    settings = settings or get_settings()
    vocab = load_crops(settings)
    limit = max(1, min(int(limit), _SUGGEST_LIMIT_MAX))
    needle = _squash(query)
    if not needle:
        return [
            {**entry.as_dict(), "matched_alias": entry.name.lower(), "score": None}
            for entry in sorted(vocab.entries, key=lambda e: e.name)[:limit]
        ]

    best: dict[str, tuple[tuple[int, int, int, float], dict[str, Any]]] = {}
    for alias, score, _idx in process.extract(
        needle,
        tuple(vocab.by_alias),
        scorer=fuzz.WRatio,
        limit=max(limit * 5, _SUGGEST_POOL),
        score_cutoff=_SUGGEST_SUBSTRING_FLOOR,
    ):
        if not _suggestion_accepts(alias, needle, float(score)):
            continue
        entry = vocab.by_alias[alias]
        rank = _suggestion_rank(alias, needle, float(score))
        if entry.key in best and best[entry.key][0] >= rank:
            continue
        best[entry.key] = (
            rank,
            {**entry.as_dict(), "matched_alias": alias, "score": round(float(score), 2)},
        )
    ranked = sorted(best.values(), key=lambda item: (*(-v for v in item[0]), item[1]["name"]))
    return [row for _rank, row in ranked[:limit]]


def suggest_places(
    query: str, *, limit: int = 10, settings: Settings | None = None
) -> list[dict[str, Any]]:
    """Autocomplete rows for ``GET /api/vocab/places`` (SPEC section 8).

    Same ranking as :func:`suggest_crops` over the k-anonymised gazetteer, with
    the better-attested place first on a tie. Each row is :meth:`Place.as_dict`
    plus ``label`` (the "Siha, Kilimanjaro Region, Tanzania" form), the
    ``matched_name`` and the ``score``.

    ``ambiguous`` rides along on every row on purpose: seven names belong to more
    than one real place (SPEC section 5.3), and a picker that hides that fact
    would be handing back a coordinate the farmer never chose.
    """
    settings = settings or get_settings()
    vocab = load_places(settings)
    limit = max(1, min(int(limit), _SUGGEST_LIMIT_MAX))
    needle = _squash(query)

    def row(place: Place, matched: str, score: float | None) -> dict[str, Any]:
        return {
            **place.as_dict(),
            "label": _place_label(place),
            "matched_name": matched,
            "score": round(score, 2) if score is not None else None,
        }

    if not needle:
        ordered = sorted(vocab.entries, key=lambda p: (-_band_rank(vocab, p), p.name))
        return [row(place, place.key, None) for place in ordered[:limit]]

    best: dict[str, tuple[tuple[int, int, int, float, int], dict[str, Any]]] = {}
    for key, score, _idx in process.extract(
        needle,
        tuple(vocab.by_key),
        scorer=fuzz.WRatio,
        limit=max(limit * 5, _SUGGEST_POOL),
        score_cutoff=_SUGGEST_SUBSTRING_FLOOR,
    ):
        if not _suggestion_accepts(key, needle, float(score)):
            continue
        place = vocab.by_key[key]
        rank = (*_suggestion_rank(key, needle, float(score)), _band_rank(vocab, place))
        if key in best and best[key][0] >= rank:
            continue
        best[key] = (rank, row(place, key, float(score)))
    ranked = sorted(best.values(), key=lambda item: (*(-v for v in item[0]), item[1]["name"]))
    return [entry for _rank, entry in ranked[:limit]]


def vocab_status(settings: Settings | None = None) -> dict[str, Any]:
    """Counts for /api/capabilities; loads the files if they are not cached."""
    settings = settings or get_settings()
    crops = load_crops(settings)
    places = load_places(settings)
    return {
        "component": "nlu:slots",
        "crops": len(crops.entries),
        "crop_aliases": len(crops.by_alias),
        "places": len(places.entries),
        "ambiguous_places": sum(1 for p in places.entries if p.ambiguous),
    }


def reset() -> None:
    """Drop the vocabulary caches (tests, tools/)."""
    with _LOCK:
        _CACHE.clear()


# --------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotCandidate:
    """One reading of one span of the sentence.

    ``score`` is the rapidfuzz score (0-100) that produced the match, or 100 for
    an exact pattern; ``confidence`` is that score on 0-1. Neither is invented:
    if a candidate cannot be scored it is not returned.
    """

    slot: str
    value: str  # canonical: crop name, gazetteer key, or timeframe expression
    label: str  # how to show it to the farmer
    raw: str  # the text that matched
    score: float
    source: str
    span: tuple[int, int] | None = None
    exact: bool = False
    needs_confirmation: bool = False
    note: str | None = None
    alternatives: tuple[str, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        return round(self.score / 100.0, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "value": self.value,
            "label": self.label,
            "raw": self.raw,
            "score": round(self.score, 2),
            "confidence": self.confidence,
            "source": self.source,
            "span": list(self.span) if self.span else None,
            "exact": self.exact,
            "needs_confirmation": self.needs_confirmation,
            "note": self.note,
            "alternatives": list(self.alternatives),
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class SlotExtraction:
    """Everything one sentence said about crop, location and timeframe."""

    text: str
    reference_date: dt.date
    candidates: Mapping[str, tuple[SlotCandidate, ...]]

    @property
    def crops(self) -> tuple[SlotCandidate, ...]:
        return self.candidates.get("crop", ())

    @property
    def locations(self) -> tuple[SlotCandidate, ...]:
        return self.candidates.get("location", ())

    @property
    def timeframes(self) -> tuple[SlotCandidate, ...]:
        return self.candidates.get("timeframe", ())

    def top(self, slot: str) -> SlotCandidate | None:
        """Highest-ranked reading, contested or not. For display only."""
        found = self.candidates.get(slot, ())
        return found[0] if found else None

    def best(self, slot: str) -> SlotCandidate | None:
        """The reading safe to act on, or ``None`` if the farmer must be asked.

        ``None`` is returned when the top candidate needs confirmation, so a
        caller cannot accidentally treat an ambiguous "Same" as a resolved
        district.
        """
        candidate = self.top(slot)
        if candidate is None or candidate.needs_confirmation:
            return None
        return candidate

    def filled(self) -> dict[str, SlotCandidate]:
        """Slot -> the candidate that can be acted on, for slots that have one."""
        out = {}
        for slot in SLOT_NAMES:
            candidate = self.best(slot)
            if candidate is not None:
                out[slot] = candidate
        return out

    def unresolved(self) -> tuple[str, ...]:
        """Slots with candidates that are all contested; the dialog must ask."""
        return tuple(
            slot
            for slot in SLOT_NAMES
            if self.candidates.get(slot) and self.best(slot) is None
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "reference_date": self.reference_date.isoformat(),
            "slots": {
                slot: [c.as_dict() for c in self.candidates.get(slot, ())] for slot in SLOT_NAMES
            },
            "filled": {slot: c.value for slot, c in self.filled().items()},
            "unresolved": list(self.unresolved()),
        }


# --------------------------------------------------------------------------
# tokenisation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Token:
    text: str  # lowercased
    start: int
    end: int


_TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)
_SQUASH_RE = re.compile(r"[^a-z0-9']+")


def _squash(value: str) -> str:
    """Normalise a vocabulary entry or a span to its comparison form."""
    return _SQUASH_RE.sub(" ", value.lower()).strip()


def _tokenize(text: str) -> list[_Token]:
    return [_Token(m.group(0).lower(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def _ngrams(text: str, tokens: Sequence[_Token], max_n: int) -> list[tuple[str, int, int, int, int]]:
    """(phrase, start_char, end_char, first_token_index, token_count), longest first.

    An n-gram is only formed across whitespace or hyphens, so "Siha, Kilimanjaro"
    never becomes the phrase "siha kilimanjaro".
    """
    out: list[tuple[str, int, int, int, int]] = []
    for n in range(min(max_n, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            window = tokens[i : i + n]
            joinable = True
            for left, right in zip(window, window[1:]):
                gap = text[left.end : right.start]
                if gap.strip(" \t\n\r-"):
                    joinable = False
                    break
            if not joinable:
                continue
            phrase = " ".join(t.text for t in window)
            out.append((phrase, window[0].start, window[-1].end, i, n))
    return out


def _overlaps(span: tuple[int, int], claimed: list[tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in claimed)


def _is_misspelling(phrase: str, name: str) -> bool:
    """Whether a fuzzy hit reads as ``phrase`` misspelt rather than as a
    different word that merely ends in ``name``.

    ``fuzz.ratio`` charges one edit for a leading letter, so "prices" scores
    90.9 against the alias "rices" and "price" 88.9 against "rice" -- both over
    the 88.0 crop floor, and a question about the price of diesel comes back as
    a question about rice. The two readings are told apart by where the extra
    letters sit: a misspelling or an unlisted plural keeps the beginning of the
    word ("tomatos" -> tomato, "mahindii" -> mahindi), while a word that only
    ends in a vocabulary name is a different word. Reading a name out of the
    tail of a longer token is :func:`_compound_match`'s job -- it has the length
    guards for it, and returns every hit needing confirmation, neither of which
    this pass does.
    """
    return name not in phrase or phrase.startswith(name)


def _compound_match(token: str, choices: Sequence[str]) -> tuple[str, float] | None:
    """The best vocabulary name written inside ``token``, or ``None``.

    "maizefield" and "inarusha" are one token to :func:`_tokenize`, so neither
    the exact pass nor the whole-phrase fuzzy pass can see the name in them:
    ``fuzz.ratio`` charges for every surrounding letter. ``fuzz.partial_ratio``
    finds it, and the guards documented at :data:`_MIN_COMPOUND_NAME_LEN` keep
    it from finding much else. Callers flag every hit for confirmation: the
    farmer wrote one word and this is reading two out of it.
    """
    if len(token) < _MIN_COMPOUND_NAME_LEN + _MIN_COMPOUND_GAP:
        return None  # too short to hold a name and a gap; also skips "my", "is"
    for choice, score, _idx in process.extract(
        token,
        choices,
        scorer=fuzz.partial_ratio,
        limit=3,
        score_cutoff=_COMPOUND_SCORE_FLOOR,
    ):
        if len(token) < len(choice) + _MIN_COMPOUND_GAP:
            continue
        aligned = fuzz.partial_ratio_alignment(choice, token)
        if aligned is None or (aligned.dest_start != 0 and aligned.dest_end != len(token)):
            continue
        return choice, float(score)  # process.extract ranks, so the first is the best
    return None


# --------------------------------------------------------------------------
# crops
# --------------------------------------------------------------------------


def extract_crops(
    text: str, settings: Settings | None = None
) -> tuple[SlotCandidate, ...]:
    """Every crop the sentence supports, ranked, against the crop vocabulary only.

    This is a single-slot view: it does not know what the gazetteer would make
    of the same words. :func:`extract_slots` is what reconciles the two.
    """
    settings = settings or get_settings()
    vocab = load_crops(settings)
    tokens = _tokenize(text)
    claimed: list[tuple[int, int]] = []
    found: list[SlotCandidate] = []

    # Pass 1: exact alias hits, longest phrase first so "sweet potato" beats "potato".
    for phrase, start, end, _i, _n in _ngrams(text, tokens, _MAX_CROP_NGRAM):
        entry = vocab.by_alias.get(phrase)
        if entry is None or _overlaps((start, end), claimed):
            continue
        claimed.append((start, end))
        found.append(
            SlotCandidate(
                slot="crop",
                value=entry.name,
                label=entry.name,
                raw=text[start:end],
                score=100.0,
                source=SOURCE_CROP_VOCAB,
                span=(start, end),
                exact=True,
                detail=_crop_detail(entry),
            )
        )

    # Pass 2: fuzzy, for typos and plurals the alias table does not list. The
    # window is the same as pass 1's, so a three-token alias is still reachable
    # once it is misspelt ("ndizi za kupica"); a long window cannot steal a short
    # alias, because fuzz.ratio charges for every character the alias does not
    # cover.
    for phrase, start, end, _i, n in _ngrams(text, tokens, _MAX_CROP_NGRAM):
        if _overlaps((start, end), claimed):
            continue
        if len(phrase) < _MIN_FUZZY_CROP_LEN or (n == 1 and phrase in _COMMON_WORDS):
            continue
        matches = [
            match
            for match in process.extract(
                phrase,
                vocab.alias_list,
                scorer=fuzz.ratio,
                limit=3,
                score_cutoff=settings.crop_match_floor,
            )
            if _is_misspelling(phrase, match[0])
        ]
        if not matches:
            continue
        alias, score, _idx = matches[0]
        entry = vocab.by_alias[alias]
        # place_ambiguous_margin is the only ambiguity margin Settings defines;
        # a crop span this close to a second reading is contested the same way.
        others = tuple(
            sorted(
                {
                    vocab.by_alias[a].name
                    for a, s, _ in matches[1:]
                    if vocab.by_alias[a].name != entry.name
                    and s >= score - settings.place_ambiguous_margin
                }
            )
        )
        claimed.append((start, end))
        found.append(
            SlotCandidate(
                slot="crop",
                value=entry.name,
                label=entry.name,
                raw=text[start:end],
                score=float(score),
                source=SOURCE_CROP_VOCAB,
                span=(start, end),
                exact=False,
                needs_confirmation=bool(others),
                note=(
                    f"'{text[start:end]}' also reads as {', '.join(others)}" if others else
                    f"fuzzy match to alias '{alias}'"
                ),
                alternatives=others,
                detail=_crop_detail(entry),
            )
        )

    # Pass 3: the alias written inside one longer token, "mymaize" or
    # "maizefield". Always contested -- the farmer wrote one word.
    for phrase, start, end, _i, _n in _ngrams(text, tokens, 1):
        if _overlaps((start, end), claimed):
            continue
        match = _compound_match(phrase, vocab.compound_list)
        if match is None:
            continue
        alias, score = match
        entry = vocab.by_alias[alias]
        claimed.append((start, end))
        found.append(
            SlotCandidate(
                slot="crop",
                value=entry.name,
                label=entry.name,
                raw=text[start:end],
                score=score,
                source=SOURCE_CROP_VOCAB,
                span=(start, end),
                exact=False,
                needs_confirmation=True,
                note=f"read '{alias}' inside the single word '{text[start:end]}'",
                detail=_crop_detail(entry),
            )
        )

    return _dedupe(found)


def _crop_detail(entry: CropEntry) -> dict[str, Any]:
    return {
        "key": entry.key,
        "in_suitability": entry.in_suitability,
        "in_disease_library": entry.in_disease_library,
        "rules_fallback": entry.rules_fallback,
    }


# --------------------------------------------------------------------------
# locations
# --------------------------------------------------------------------------


def extract_locations(
    text: str, settings: Settings | None = None
) -> tuple[SlotCandidate, ...]:
    """Every place the sentence supports, ranked, against the gazetteer only.

    This is a single-slot view. It knows that some gazetteer names are also crop
    aliases -- that is the ``weak`` test below -- but it cannot see what the crop
    extractor actually matched, so a place that only *fuzzily* resembles a crop
    word still looks clean here. :func:`extract_slots` reconciles the two.
    """
    settings = settings or get_settings()
    vocab = load_places(settings)
    crops = load_crops(settings)
    tokens = _tokenize(text)
    claimed: list[tuple[int, int]] = []
    # (place, score, start, end, first token index, exact, other readings, compound)
    hits: list[tuple[Place, float, int, int, int, bool, tuple[str, ...], bool]] = []

    for phrase, start, end, i, _n in _ngrams(text, tokens, _MAX_PLACE_NGRAM):
        place = vocab.by_key.get(phrase)
        if place is None or _overlaps((start, end), claimed):
            continue
        claimed.append((start, end))
        hits.append((place, 100.0, start, end, i, True, (), False))

    # The fuzzy window matches the exact one: "dar es salaam" misspelt is three
    # tokens, and no two-token window of it scores anywhere near the floor.
    for phrase, start, end, i, n in _ngrams(text, tokens, _MAX_PLACE_NGRAM):
        if _overlaps((start, end), claimed):
            continue
        if len(phrase) < _MIN_FUZZY_PLACE_LEN or (n == 1 and phrase in _COMMON_WORDS):
            continue
        matches = process.extract(
            phrase,
            vocab.name_list,
            scorer=fuzz.ratio,
            limit=3,
            score_cutoff=settings.place_match_floor,
        )
        if not matches:
            continue
        key, score, _idx = matches[0]
        place = vocab.by_key[key]
        others = tuple(
            sorted(
                {
                    vocab.by_key[k].name
                    for k, s, _ in matches[1:]
                    if k != key and s >= score - settings.place_ambiguous_margin
                }
            )
        )
        claimed.append((start, end))
        hits.append((place, float(score), start, end, i, False, others, False))

    # ... and the name written inside one longer token: "inarusha", "moshitown".
    for phrase, start, end, i, _n in _ngrams(text, tokens, 1):
        if _overlaps((start, end), claimed):
            continue
        match = _compound_match(phrase, vocab.compound_list)
        if match is None:
            continue
        key, score = match
        claimed.append((start, end))
        hits.append((vocab.by_key[key], score, start, end, i, False, (), True))

    # Corroboration is computed over the whole sentence: which regions, countries
    # and sibling places does the text itself mention?
    mentioned = {p.key for p, *_ in hits}
    lowered = " " + _squash(text) + " "
    found: list[SlotCandidate] = []
    for place, score, start, end, token_index, exact, others, compound in hits:
        stem = place.parent_stem
        corroborated_by = None
        if stem and f" {stem} " in lowered and stem != place.key:
            corroborated_by = place.parent
        elif place.country and f" {place.country.lower()} " in lowered:
            corroborated_by = place.country
        else:
            siblings = [
                other.name
                for other in vocab.entries
                if other.key in mentioned
                and other.key != place.key
                and other.parent
                and other.parent == place.parent
            ]
            if siblings:
                corroborated_by = siblings[0]

        weak = (
            len(place.key) <= 4
            or place.key in _COMMON_WORDS
            or place.key in crops.by_alias
        )
        cue = _has_locative_cue(tokens, token_index, mentioned)

        if place.key in _COMMON_WORDS and not cue and not corroborated_by:
            # "the same field", "cook" -- the farmer used an English word, not a
            # place name. Declining to assert it is not a guess.
            continue

        needs_confirmation = False
        notes: list[str] = []
        if place.ambiguous:
            if corroborated_by:
                notes.append(f"ambiguous name, corroborated by {corroborated_by}")
            else:
                needs_confirmation = True
                notes.append(
                    f"'{place.name}' names more than one real place; confirm the region"
                )
        if weak and not cue and not corroborated_by:
            needs_confirmation = True
            notes.append(f"'{place.name}' is also an ordinary word here; no locative cue")
        if others:
            needs_confirmation = True
            notes.append(f"also reads as {', '.join(others)}")
        if compound:
            needs_confirmation = True
            notes.append(f"read '{place.key}' inside the single word '{text[start:end]}'")
        elif not exact:
            notes.append(f"fuzzy match to '{place.name}'")

        found.append(
            SlotCandidate(
                slot="location",
                value=place.key,
                label=_place_label(place),
                raw=text[start:end],
                score=score,
                source=SOURCE_GAZETTEER,
                span=(start, end),
                exact=exact,
                needs_confirmation=needs_confirmation,
                note="; ".join(notes) or None,
                alternatives=others,
                detail={
                    **place.as_dict(),
                    "corroborated_by": corroborated_by,
                    "locative_cue": cue,
                    "inside_word": compound,
                    "support_rank": _band_rank(vocab, place),
                },
            )
        )
    return _mark_competition(_dedupe(found), vocab)


def _mark_competition(
    candidates: Sequence[SlotCandidate], vocab: _PlaceVocab
) -> tuple[SlotCandidate, ...]:
    """Settle several places named in one sentence.

    "in Hai and Siha, Kilimanjaro" names two fields inside one region. The
    region is corroboration, not a third field, and the two districts are a
    genuine choice -- so all three come back needing confirmation and the
    dialog asks, instead of one of them being picked by sort order.
    """
    if len(candidates) < 2:
        return tuple(candidates)
    keys = {c.value for c in candidates}
    children_of: dict[str, list[str]] = {}
    for candidate in candidates:
        place = vocab.by_key.get(candidate.value)
        stem = place.parent_stem if place else None
        if stem and stem in keys and stem != candidate.value:
            children_of.setdefault(stem, []).append(place.name)
    specific = [c for c in candidates if c.value not in children_of]

    out: list[SlotCandidate] = []
    for candidate in candidates:
        note = candidate.note
        if candidate.value in children_of:
            extra = f"names the region around {', '.join(children_of[candidate.value])}, not the field itself"
        elif len(specific) > 1:
            others = ", ".join(c.label for c in specific if c.value != candidate.value)
            extra = f"the message also names {others}; confirm which field"
        else:
            out.append(candidate)
            continue
        out.append(
            replace(
                candidate,
                needs_confirmation=True,
                note="; ".join(part for part in (note, extra) if part),
            )
        )
    return tuple(sorted(out, key=_rank, reverse=True))


def _place_label(place: Place) -> str:
    parts = [place.name]
    if place.parent and place.parent != place.name:
        parts.append(place.parent)
    if place.country:
        parts.append(place.country)
    return ", ".join(parts)


def _has_locative_cue(
    tokens: Sequence[_Token], token_index: int, mentioned: Iterable[str]
) -> bool:
    """True if the tokens before this one mark it as a place.

    Walks back through list conjunctions so that the "in" of "in Hai and Siha"
    still covers Siha.
    """
    mentioned = set(mentioned)
    i = token_index - 1
    steps = 0
    while i >= 0 and steps < 4:
        word = tokens[i].text
        if word in _LOCATIVE_CUES:
            return True
        if word in _LIST_CUES or word in mentioned:
            i -= 1
            steps += 1
            continue
        return False
    return False


# --------------------------------------------------------------------------
# timeframes
# --------------------------------------------------------------------------

_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fourteen": 14, "twenty": 20, "thirty": 30,
}
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
# Named East African seasons. Real calendar boundaries vary by year and by
# district, so no start/end is invented for them.
_SEASONS = {
    "masika": "masika (long rains)",
    "vuli": "vuli (short rains)",
    "kiangazi": "kiangazi (dry season)",
    "long rains": "long rains",
    "short rains": "short rains",
    "dry season": "dry season",
    "rainy season": "rainy season",
    "wet season": "wet season",
    "planting season": "planting season",
    "harvest season": "harvest season",
    "growing season": "growing season",
}
# Only these make a bare "may" a month rather than the modal verb.
_MONTH_CUE = re.compile(r"\b(in|by|since|during|before|after|until|around|from|of)\s*$")
_STAGES = (
    "planting", "sowing", "transplanting", "germination", "flowering", "tasseling",
    "podding", "weeding", "harvest", "harvesting",
)


def _window(days: int, today: dt.date, forward: bool) -> tuple[dt.date, dt.date]:
    return (today, today + dt.timedelta(days=days)) if forward else (today - dt.timedelta(days=days), today)


def _month_bounds(year: int, month: int) -> tuple[dt.date, dt.date]:
    start = dt.date(year, month, 1)
    end = dt.date(year + (month == 12), (month % 12) + 1, 1) - dt.timedelta(days=1)
    return start, end


def extract_timeframes(
    text: str, today: dt.date | None = None, settings: Settings | None = None
) -> tuple[SlotCandidate, ...]:
    """Timeframe expressions with the dates they resolve to.

    ``today`` is the reference date; it defaults to the real current date, which
    is the one thing here that is genuinely "now".
    """
    today = today or dt.date.today()
    lowered = text.lower()
    found: list[SlotCandidate] = []
    claimed: list[tuple[int, int]] = []

    def add(
        match: re.Match,
        value: str,
        kind: str,
        start: dt.date | None,
        end: dt.date | None,
        note: str | None = None,
    ) -> None:
        span = (match.start(), match.end())
        if _overlaps(span, claimed):
            return
        claimed.append(span)
        found.append(
            SlotCandidate(
                slot="timeframe",
                value=value,
                label=value,
                raw=text[span[0] : span[1]],
                score=100.0,
                source=SOURCE_TIME_PATTERN,
                span=span,
                exact=True,
                needs_confirmation=start is None,
                note=note,
                detail={
                    "kind": kind,
                    "start": start.isoformat() if start else None,
                    "end": end.isoformat() if end else None,
                    "reference_date": today.isoformat(),
                },
            )
        )

    for match in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", lowered):
        try:
            day = dt.date(int(match[1]), int(match[2]), int(match[3]))
        except ValueError:
            continue
        add(match, day.isoformat(), "day", day, day)

    for match in re.finditer(
        r"\b(?:the\s+)?(?:last|past|previous)\s+(\d{1,3}|"
        + "|".join(_WORD_NUMBERS)
        + r")\s+(day|days|week|weeks|month|months)\b",
        lowered,
    ):
        count = int(match[1]) if match[1].isdigit() else _WORD_NUMBERS.get(match[1])
        if not count:
            continue
        days = count * {"day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30}[match[2]]
        start, end = _window(days, today, forward=False)
        add(match, f"last {count} {match[2]}", "window", start, end)

    for match in re.finditer(
        r"\b(?:in|after|within|next)\s+(\d{1,3}|"
        + "|".join(_WORD_NUMBERS)
        + r")\s+(day|days|week|weeks|month|months)\b",
        lowered,
    ):
        count = int(match[1]) if match[1].isdigit() else _WORD_NUMBERS.get(match[1])
        if not count:
            continue
        days = count * {"day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30}[match[2]]
        start, end = _window(days, today, forward=True)
        add(match, f"in {count} {match[2]}", "window", start, end)

    for match in re.finditer(r"\b(today|right now|just now|now|this morning|this afternoon|this evening|tonight)\b", lowered):
        add(match, "today", "day", today, today)
    for match in re.finditer(r"\btomorrow\b", lowered):
        add(match, "tomorrow", "day", today + dt.timedelta(days=1), today + dt.timedelta(days=1))
    for match in re.finditer(r"\byesterday\b", lowered):
        add(match, "yesterday", "day", today - dt.timedelta(days=1), today - dt.timedelta(days=1))

    week_start = today - dt.timedelta(days=today.weekday())
    for match in re.finditer(r"\b(this|next|last|past)\s+week\b", lowered):
        shift = {"this": 0, "next": 7, "last": -7, "past": -7}[match[1]]
        start = week_start + dt.timedelta(days=shift)
        add(match, f"{match[1]} week", "window", start, start + dt.timedelta(days=6))

    for match in re.finditer(r"\b(this|next|last|past)\s+month\b", lowered):
        offset = {"this": 0, "next": 1, "last": -1, "past": -1}[match[1]]
        month = today.month + offset
        year = today.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        start, end = _month_bounds(year, month)
        add(match, f"{match[1]} month", "window", start, end)

    for match in re.finditer(r"\b(" + "|".join(_MONTHS) + r")\b(?:\s+(\d{4}))?", lowered):
        month = _MONTHS[match[1]]
        if match[1] == "may" and not match[2] and not _MONTH_CUE.search(lowered[: match.start()]):
            continue  # "may i ask" is the modal verb, not the month
        if match[2]:
            start, end = _month_bounds(int(match[2]), month)
            add(match, f"{match[1]} {match[2]}", "month", start, end)
        else:
            add(
                match,
                match[1],
                "month",
                None,
                None,
                note="no year given, so the dates are unknown",
            )

    for phrase, label in _SEASONS.items():
        for match in re.finditer(r"\b" + re.escape(phrase) + r"\b", lowered):
            add(match, label, "season", None, None, note="season boundaries vary by year and district")
    for match in re.finditer(r"\b(this|next|last|past)\s+season\b", lowered):
        add(
            match,
            f"{match[1]} season",
            "season",
            None,
            None,
            note="season boundaries vary by year and district",
        )

    for match in re.finditer(
        r"\b(?:at|before|after|during)\s+(" + "|".join(_STAGES) + r")\b", lowered
    ):
        add(
            match,
            match.group(0),
            "growth_stage",
            None,
            None,
            note="a crop stage, not a calendar date",
        )

    return tuple(sorted(found, key=lambda c: c.span or (0, 0)))


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def _dedupe(candidates: Sequence[SlotCandidate]) -> tuple[SlotCandidate, ...]:
    """One candidate per canonical value, best span first."""
    best: dict[str, SlotCandidate] = {}
    for candidate in candidates:
        current = best.get(candidate.value)
        if current is None or _rank(candidate) > _rank(current):
            best[candidate.value] = candidate
    return tuple(sorted(best.values(), key=_rank, reverse=True))


def _rank(candidate: SlotCandidate) -> tuple:
    detail = candidate.detail or {}
    return (
        candidate.score,
        not candidate.needs_confirmation,
        len(candidate.raw),
        # Ties go to the better-attested place. A band the gazetteer did not
        # publish is not evidence of anything, so it sorts last rather than
        # being read as "nobody has ever been there".
        float(detail.get("support_rank") or 0),
    )


def _cross_slot_support(candidate: SlotCandidate) -> int:
    """How hard one reading of a span is to argue with. Bigger wins.

    Two things can make a reading strong: the vocabulary matched it exactly, and
    the sentence said what kind of thing it is. Only a place can have the second
    -- a locative cue ("in Karanga") or a corroborating region -- which is the
    asymmetry that settles most collisions correctly.

    The match score is deliberately not part of this. The two readings were
    scored against different vocabularies, so one standing a few points higher
    is not an argument about which the farmer meant -- and if it were allowed to
    break the tie it would leave the winner silently actionable, which is the
    outcome SPEC section 5.1 rules out. Equal support is a tie, and a tie flags
    both readings.
    """
    detail = candidate.detail or {}
    said_so = bool(detail.get("locative_cue") or detail.get("corroborated_by"))
    return int(candidate.exact) + int(said_so)


def _flag_cross_slot(
    candidate: SlotCandidate, winner: SlotCandidate
) -> SlotCandidate:
    """Mark ``candidate`` as contested by a reading in the other slot."""
    kind = "crop" if winner.slot == "crop" else "place"
    extra = (
        f"'{candidate.raw}' also reads as the {kind} {winner.label}; confirm before "
        "using it as a " + ("crop" if candidate.slot == "crop" else "location")
    )
    return replace(
        candidate,
        needs_confirmation=True,
        note="; ".join(part for part in (candidate.note, extra) if part),
        detail={
            **candidate.detail,
            "conflicts_with": {
                "slot": winner.slot,
                "value": winner.value,
                "label": winner.label,
                "raw": winner.raw,
                "score": round(winner.score, 2),
                "exact": winner.exact,
            },
        },
    )


def _reconcile_crop_location(
    crops: Sequence[SlotCandidate], locations: Sequence[SlotCandidate]
) -> tuple[tuple[SlotCandidate, ...], tuple[SlotCandidate, ...]]:
    """Settle the spans that the crop vocabulary and the gazetteer both claim.

    *viazi* is Swahili for potato and Vianzi is a real village; *karanga* is
    groundnut and also a ward of Moshi. Run separately, both extractors answer
    confidently about the same word, and because a resolved location is what
    authorises an Earth Engine run (SPEC section 4.4), the loser of that silence
    is a farmer who gets an answer about someone else's field.

    So for every shared span: whichever reading the sentence supports better
    keeps its standing, the other is flagged; and when neither is better, both
    are flagged, because misrouting silently is worse than asking (SPEC 5.1).
    A reading that already needs confirmation contests nothing -- it is not an
    assertion this module was willing to make in the first place.
    """
    if not crops or not locations:
        return tuple(crops), tuple(locations)

    # Index -> flagged copy. A candidate can lose to more than one reading in
    # the other slot, and the flags accumulate so every reason is on the note.
    crop_flags: dict[int, SlotCandidate] = {}
    place_flags: dict[int, SlotCandidate] = {}
    for ci, crop in enumerate(crops):
        for li, location in enumerate(locations):
            if crop.span is None or location.span is None:
                continue
            if not _overlaps(crop.span, [location.span]):
                continue
            # Decided on what the extractors themselves asserted, so the outcome
            # of one contest never feeds into the next.
            if crop.needs_confirmation or location.needs_confirmation:
                continue
            crop_support = _cross_slot_support(crop)
            place_support = _cross_slot_support(location)
            if crop_support >= place_support:
                place_flags[li] = _flag_cross_slot(place_flags.get(li, location), crop)
            if place_support >= crop_support:
                crop_flags[ci] = _flag_cross_slot(crop_flags.get(ci, crop), location)

    if not crop_flags and not place_flags:
        return tuple(crops), tuple(locations)
    # Re-rank: _rank puts uncontested readings first, and some just stopped
    # being uncontested.
    return (
        tuple(sorted(
            (crop_flags.get(i, c) for i, c in enumerate(crops)), key=_rank, reverse=True
        )),
        tuple(sorted(
            (place_flags.get(i, c) for i, c in enumerate(locations)), key=_rank, reverse=True
        )),
    )


def extract_slots(
    text: str,
    *,
    today: dt.date | None = None,
    settings: Settings | None = None,
) -> SlotExtraction:
    """Extract every crop, location and timeframe the sentence supports.

    Returns ranked candidates. Nothing is dropped for being uncertain -- it is
    returned flagged, because the farmer, not this module, settles a tie. Crop
    and location are reconciled against each other here (see
    :func:`_reconcile_crop_location`); the timeframe patterns share no
    vocabulary with either and need no reconciliation.
    """
    settings = settings or get_settings()
    today = today or dt.date.today()
    crops, locations = _reconcile_crop_location(
        extract_crops(text, settings), extract_locations(text, settings)
    )
    return SlotExtraction(
        text=text,
        reference_date=today,
        candidates={
            "crop": crops,
            "location": locations,
            "timeframe": extract_timeframes(text, today, settings),
        },
    )
