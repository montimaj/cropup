"""Retrieve knowledge cards and cite them (SPEC sections 4.3 and 6).

Two rules shape everything here.

1. A retrieved agronomic claim is returned **verbatim**. ``Snippet.text`` is the
   card's body exactly as it was written, and every snippet carries the
   citation for the rule or file it came from. Nothing in this module rewrites,
   shortens or merges a claim.
2. Below the similarity floor there is no answer. ``retrieve`` returns zero
   snippets and the score that fell short, so the caller can say "I do not have
   anything on that" and offer the questionnaire instead of serving the nearest
   unrelated card.

Thresholds come from :mod:`cropup.config` (``CROPUP_RAG_SIMILARITY_FLOOR``,
``CROPUP_RAG_TOP_K``), never from literals here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from ..config import Settings, get_settings
from ..errors import CropUpError
from . import index as index_module
from .corpus import Card

__all__ = ["Snippet", "RetrievalResult", "retrieve", "is_available"]


@dataclass(frozen=True)
class Snippet:
    """One card, returned exactly as written, with its citation and score."""

    card_id: str
    title: str
    text: str
    citation: str
    score: float
    kind: str
    category: str | None = None
    severity: str | None = None
    urgency: str | None = None
    crops: tuple[str, ...] = ()
    intents: tuple[str, ...] = ()
    conditions: tuple[dict[str, Any], ...] = ()
    origin: str | None = None

    @classmethod
    def from_card(cls, card: Card, score: float) -> "Snippet":
        return cls(
            card_id=card.card_id,
            title=card.title,
            text=card.body,
            citation=card.citation,
            score=round(float(score), 4),
            kind=card.kind,
            category=card.category,
            severity=card.severity,
            urgency=card.urgency,
            crops=card.crops,
            intents=card.intents,
            conditions=tuple(c.as_dict() for c in card.conditions),
            origin=card.origin,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "title": self.title,
            "text": self.text,
            "citation": self.citation,
            "score": self.score,
            "kind": self.kind,
            "category": self.category,
            "severity": self.severity,
            "urgency": self.urgency,
            "crops": list(self.crops),
            "intents": list(self.intents),
            "conditions": [dict(c) for c in self.conditions],
            "origin": self.origin,
        }


@dataclass(frozen=True)
class RetrievalResult:
    """What one query found, and -- when it found nothing -- why."""

    query: str
    snippets: tuple[Snippet, ...]
    floor: float
    k: int
    considered: int
    best_score: float
    model_id: str
    note: str
    intent: str | None = None
    crop: str | None = None
    filtered_by: tuple[str, ...] = ()

    @property
    def found(self) -> bool:
        return bool(self.snippets)

    def citations(self) -> tuple[str, ...]:
        return tuple(s.citation for s in self.snippets)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "found": self.found,
            "snippets": [s.as_dict() for s in self.snippets],
            "floor": self.floor,
            "k": self.k,
            "considered": self.considered,
            "best_score": self.best_score,
            "model_id": self.model_id,
            "note": self.note,
            "intent": self.intent,
            "crop": self.crop,
            "filtered_by": list(self.filtered_by),
        }

    def __repr__(self) -> str:
        return (
            f"RetrievalResult({self.query!r}, {len(self.snippets)} snippets, "
            f"best={self.best_score:.3f}, floor={self.floor:.2f})"
        )


@lru_cache(maxsize=4)
def _alias_to_crop(crops_file: str) -> dict[str, str]:
    """alias -> canonical crop key, from crops.json.

    Only an exact lookup: fuzzy crop matching belongs to ``nlu/slots.py`` and is
    not duplicated here. A missing or unreadable file simply means no aliases,
    because retrieval must not fail over a convenience.
    """
    path = Path(crops_file)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    mapping: dict[str, str] = {}
    for entry in payload.get("crops") or ():
        key = str(entry.get("key") or entry.get("name") or "").strip().lower()
        if not key:
            continue
        mapping[key] = key
        mapping[str(entry.get("name", "")).strip().lower()] = key
        for alias in entry.get("aliases") or ():
            mapping[str(alias).strip().lower()] = key
    mapping.pop("", None)
    return mapping


def _canonical_crop(crop: str | None, settings: Settings) -> str | None:
    if not crop:
        return None
    lowered = crop.strip().lower()
    if not lowered:
        return None
    return _alias_to_crop(str(settings.crops_path)).get(lowered, lowered)


def _candidates(
    cards: Sequence[Card], intent: str | None, crop: str | None
) -> tuple[list[int], tuple[str, ...], bool]:
    """Narrow the pool by crop and intent. Returns (pool, filters, intent_covered).

    Two different empty cases, deliberately treated differently:

    * No card anywhere declares this intent -- livestock questions, for
      instance. The corpus does not cover the subject, so the pool stays empty
      and the caller says so. Searching the whole corpus instead would answer
      "my cow is limping" with a cowpea card.
    * The intent is covered but no card survives alongside the crop filter. The
      filter is dropped, because a metadata mismatch is not the same as an
      absence of knowledge.
    """
    applied: list[str] = []
    pool = list(range(len(cards)))

    if crop:
        narrowed = [i for i in pool if cards[i].matches_crop(crop)]
        if narrowed:
            pool = narrowed
            applied.append(f"crop={crop}")

    if not intent:
        return pool, tuple(applied), True

    intent_covered = any(intent in card.intents for card in cards)
    if not intent_covered:
        return [], tuple(applied + [f"intent={intent}"]), False
    narrowed = [i for i in pool if cards[i].matches_intent(intent)]
    if narrowed:
        pool = narrowed
        applied.append(f"intent={intent}")
    return pool, tuple(applied), True


def is_available(settings: Settings | None = None) -> bool:
    """True when a built index can be loaded. Never raises."""
    try:
        index_module.load(settings=settings)
    except CropUpError:
        return False
    return True


def retrieve(
    query: str,
    k: int | None = None,
    *,
    intent: str | None = None,
    crop: str | None = None,
    floor: float | None = None,
    index: index_module.CardIndex | None = None,
    settings: Settings | None = None,
) -> RetrievalResult:
    """Retrieve up to ``k`` cards for ``query``, verbatim and cited.

    ``intent`` is one of the SPEC 5.2 intents and ``crop`` a crop name or alias;
    each narrows the pool when it can and is ignored when it cannot.

    Raises :class:`cropup.errors.DataFileError` when the index has not been
    built and :class:`cropup.errors.ModelUnavailable` when the encoder cannot
    run -- both are conditions the caller must report, not paper over.
    """
    settings = settings or get_settings()
    k = settings.rag_top_k if k is None else int(k)
    floor = settings.rag_similarity_floor if floor is None else float(floor)
    text = (query or "").strip()

    if not text:
        return RetrievalResult(
            query=query or "",
            snippets=(),
            floor=floor,
            k=k,
            considered=0,
            best_score=0.0,
            model_id=settings.embed_model_id,
            note="no query text was given, so nothing was retrieved",
            intent=intent,
            crop=crop,
        )

    card_index = index or index_module.load(settings=settings)
    resolved_crop = _canonical_crop(crop, settings)
    pool, applied, intent_covered = _candidates(card_index.cards, intent, resolved_crop)

    if not intent_covered:
        return RetrievalResult(
            query=text,
            snippets=(),
            floor=floor,
            k=k,
            considered=0,
            best_score=0.0,
            model_id=card_index.model_id,
            note=f"the knowledge base holds no cards for {intent} questions",
            intent=intent,
            crop=resolved_crop,
            filtered_by=applied,
        )

    query_vector = index_module.encode([text], settings)[0]
    best = card_index.best_score(query_vector, pool)
    hits = card_index.search(query_vector, k, floor=floor, candidates=pool)

    if not hits:
        # "nothing cleared the floor" and "no snippets were asked for" are
        # different answers, and the second must not read as the first.
        if k <= 0:
            note = f"k={k}, so no snippets were requested (closest of {len(pool)} cards: {best:.3f})"
        else:
            note = (
                f"nothing in the knowledge base scored above the {floor:.2f} similarity floor "
                f"(closest of {len(pool)} cards: {best:.3f})"
            )
        return RetrievalResult(
            query=text,
            snippets=(),
            floor=floor,
            k=k,
            considered=len(pool),
            best_score=round(best, 4),
            model_id=card_index.model_id,
            note=note,
            intent=intent,
            crop=resolved_crop,
            filtered_by=applied,
        )

    snippets = tuple(Snippet.from_card(card_index.cards[i], score) for i, score in hits)
    return RetrievalResult(
        query=text,
        snippets=snippets,
        floor=floor,
        k=k,
        considered=len(pool),
        best_score=round(best, 4),
        model_id=card_index.model_id,
        note=f"{len(snippets)} cards above the {floor:.2f} similarity floor",
        intent=intent,
        crop=resolved_crop,
        filtered_by=applied,
    )
