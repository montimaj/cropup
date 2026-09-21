"""The embedding index over the knowledge cards (SPEC section 6).

A numpy matrix and a JSON sidecar. For a corpus of this size an external vector
database would be a dependency bought with nothing: 500-odd cards at 384
dimensions is a 0.8 MB float32 matrix, and an exact dot product over it is
faster than a network round trip to a service that would only approximate it.

The encoder is **not** implemented here. ``cropup.nlu.embed`` owns the ONNX
MiniLM session (SPEC section 5); this module imports it rather than keeping a
second copy of the model loading code that could drift from it.

Layout inside ``cropup/data/corpus/``::

    cards.jsonl   the cards themselves (corpus.py writes it)
    index.npy     float32 (n_cards, dim), L2-normalised, row i = card i
    index.json    model id, dim, card count and digest -- metadata only

``index.json`` carried a second copy of every card payload so that ``load()``
never had to re-parse ``cards.jsonl``. Measured, that bought 0.8 ms -- 6.0 ms
to parse the 515 cards out of one JSON object against 6.8 ms to parse them out
of the JSONL -- in exchange for 884 KB of committed duplication and a way for
the two files to disagree about what a card says. ``cards.jsonl`` is the one
source of truth now; the sidecar keeps the digest that says whether
``index.npy`` was built from it, and ``load()`` refuses an index whose card
count or digest has drifted rather than scoring one card's vector against
another card's text. A cold load is 13 ms in total.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import Settings, get_settings
from ..errors import DataFileError, ModelUnavailable
from .corpus import CARDS_FILE, Card, build_cards, corpus_digest, load_cards

__all__ = [
    "VECTORS_FILE",
    "SIDECAR_FILE",
    "CardIndex",
    "encode",
    "build",
    "load",
    "clear_cache",
]

VECTORS_FILE = "index.npy"
SIDECAR_FILE = "index.json"
INDEX_FORMAT = 1

_CACHE: dict[str, tuple[int, int, "CardIndex"]] = {}


def _embed_module() -> Any:
    """The NLU encoder module, imported late so importing rag stays cheap."""
    try:
        from ..nlu import embed as embed_module
    except ImportError as exc:  # the NLU package is absent or failed to import
        raise ModelUnavailable("cropup.nlu.embed", f"could not import the encoder: {exc}") from exc
    return embed_module


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows so a dot product is a cosine similarity.

    Applied on both sides even though the encoder is expected to normalise
    already: it is idempotent, and it means a score can be read as a cosine
    without trusting an assumption about another module.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)


def encode(texts: Sequence[str], settings: Settings | None = None) -> np.ndarray:
    """Encode text with the NLU encoder; returns (n, dim) float32, L2-normalised.

    ``cropup.nlu.embed.encode`` returns ``None`` when the model is unavailable,
    because intent classification degrades to its rule tier. Retrieval has no
    such tier, so the ``None`` is turned back into a
    :class:`~cropup.errors.ModelUnavailable` carrying the reason.
    """
    settings = settings or get_settings()
    if not texts:
        return np.zeros((0, settings.embed_dim), dtype=np.float32)
    module = _embed_module()
    raw = module.encode(list(texts), settings)
    if raw is None:
        status = module.embed_status(settings)
        raise ModelUnavailable(settings.embed_model_id, status.get("error") or "encoder unavailable")
    vectors = np.asarray(raw, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.shape[0] != len(texts):
        raise ModelUnavailable(
            settings.embed_model_id,
            f"encoder returned {vectors.shape[0]} vectors for {len(texts)} texts",
        )
    if vectors.shape[1] != settings.embed_dim:
        raise ModelUnavailable(
            settings.embed_model_id,
            f"encoder returned dimension {vectors.shape[1]}, expected {settings.embed_dim}",
        )
    return _normalise(vectors)


@dataclass(frozen=True, eq=False)  # eq=False: a numpy matrix has no scalar ==
class CardIndex:
    """Cards plus their vectors. Row ``i`` of ``vectors`` embeds ``cards[i]``."""

    cards: tuple[Card, ...]
    vectors: np.ndarray
    model_id: str
    dim: int
    built_at: str
    digest: str
    path: Path | None = None

    def __post_init__(self) -> None:
        if self.vectors.shape[0] != len(self.cards):
            raise ValueError(
                f"index has {self.vectors.shape[0]} vectors for {len(self.cards)} cards"
            )
        if self.cards and self.vectors.shape[1] != self.dim:
            raise ValueError(f"index dimension {self.vectors.shape[1]} does not match {self.dim}")

    def __len__(self) -> int:
        return len(self.cards)

    def __repr__(self) -> str:
        return f"CardIndex({len(self.cards)} cards, dim={self.dim}, model={self.model_id})"

    def matches(self, cards: Sequence[Card]) -> bool:
        """True when this index was built from exactly these cards."""
        return self.digest == corpus_digest(cards)

    def score(self, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity of one query against every card."""
        if len(self.cards) == 0:
            return np.zeros((0,), dtype=np.float32)
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm > 0.0:
            query = query / norm
        return np.asarray(self.vectors @ query, dtype=np.float32)

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        *,
        floor: float = 0.0,
        candidates: Sequence[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Top ``k`` (card index, score) pairs at or above ``floor``, best first."""
        scores = self.score(query_vector)
        if scores.size == 0 or k <= 0:
            return []
        pool = np.arange(scores.size) if candidates is None else np.asarray(list(candidates), dtype=int)
        if pool.size == 0:
            return []
        pool_scores = scores[pool]
        keep = pool_scores >= floor
        pool, pool_scores = pool[keep], pool_scores[keep]
        if pool.size == 0:
            return []
        order = np.argsort(-pool_scores, kind="stable")[:k]
        return [(int(pool[i]), float(pool_scores[i])) for i in order]

    def best_score(self, query_vector: np.ndarray, candidates: Sequence[int] | None = None) -> float:
        """The highest similarity in the pool, floor or no floor.

        Retrieval reports this even when nothing clears the floor, so the UI can
        say "the closest card scored 0.21" instead of an unexplained silence.
        """
        scores = self.score(query_vector)
        if scores.size == 0:
            return 0.0
        if candidates is not None:
            pool = np.asarray(list(candidates), dtype=int)
            if pool.size == 0:
                return 0.0
            scores = scores[pool]
        return float(scores.max())


def build(
    cards: Sequence[Card] | None = None,
    *,
    settings: Settings | None = None,
    corpus_dir: Path | None = None,
    write: bool = True,
) -> CardIndex:
    """Embed the cards and (by default) write ``index.npy`` + ``index.json``."""
    settings = settings or get_settings()
    directory = corpus_dir or settings.corpus_dir
    if cards is None:
        try:
            cards = load_cards(directory, settings)
        except DataFileError:
            cards = build_cards(settings)
    cards = tuple(cards)

    vectors = encode([card.embed_text for card in cards], settings)
    index = CardIndex(
        cards=cards,
        vectors=vectors,
        model_id=settings.embed_model_id,
        dim=settings.embed_dim,
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        digest=corpus_digest(cards),
        path=directory if write else None,
    )
    if write:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / VECTORS_FILE, vectors)
        sidecar = {
            "format": INDEX_FORMAT,
            "model_id": index.model_id,
            "onnx_file": settings.embed_onnx_file,
            "dim": index.dim,
            "card_count": len(cards),
            "built_at": index.built_at,
            "digest": index.digest,
            "vectors_file": VECTORS_FILE,
            "cards_file": CARDS_FILE,
        }
        (directory / SIDECAR_FILE).write_text(
            json.dumps(sidecar, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
        _CACHE.pop(str(directory / SIDECAR_FILE), None)
    return index


def load(
    *,
    settings: Settings | None = None,
    corpus_dir: Path | None = None,
    refresh: bool = False,
) -> CardIndex:
    """Read a built index. One JSON parse and one mmap, then cached in-process."""
    settings = settings or get_settings()
    directory = corpus_dir or settings.corpus_dir
    sidecar_path = directory / SIDECAR_FILE
    vectors_path = directory / VECTORS_FILE
    for path in (sidecar_path, vectors_path):
        if not path.is_file():
            raise DataFileError(path, "index has not been built; run tools/build_corpus.py")

    stat = sidecar_path.stat()
    key = str(sidecar_path)
    cached = _CACHE.get(key)
    if cached and not refresh and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2]

    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataFileError(sidecar_path, f"index sidecar is unreadable: {exc}") from exc
    if payload.get("format") != INDEX_FORMAT:
        raise DataFileError(sidecar_path, f"unsupported index format {payload.get('format')!r}")
    # mmap: the matrix is read straight from the file, so loading does not cost
    # a copy and a cold /api/message does not wait on 0.8 MB of allocation.
    cards = tuple(load_cards(directory, settings))
    try:
        vectors = np.load(vectors_path, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise DataFileError(vectors_path, f"index is unusable, rebuild it: {exc}") from exc
    # Row i of the matrix must be card i. Two files, written by two steps, so
    # the pairing is checked rather than assumed: a mismatched row would return
    # one card's text under another card's score, cited as if it were the hit.
    if vectors.shape[0] != len(cards):
        raise DataFileError(
            vectors_path,
            f"{len(cards)} cards in {CARDS_FILE} but {vectors.shape[0]} vectors here; "
            "rerun tools/build_corpus.py",
        )
    try:
        index = CardIndex(
            cards=cards,
            vectors=vectors,
            model_id=payload.get("model_id", settings.embed_model_id),
            dim=int(payload.get("dim", settings.embed_dim)),
            built_at=payload.get("built_at", ""),
            digest=str(payload.get("digest") or ""),
            path=directory,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DataFileError(sidecar_path, f"index is unusable, rebuild it: {exc}") from exc
    if not index.matches(cards):
        raise DataFileError(
            sidecar_path,
            f"this index was built from a different {CARDS_FILE}; rerun tools/build_corpus.py",
        )
    _CACHE[key] = (stat.st_mtime_ns, stat.st_size, index)
    return index


def clear_cache() -> None:
    """Drop the in-process index cache (tests, and after a rebuild)."""
    _CACHE.clear()
