"""The honesty endpoint: what CropUp can do *right now* (SPEC 8, SPEC 9).

``Ledger.degradation()`` answers "what did that turn fail to measure". This
module answers the question the farmer has before they type anything: **what is
degraded now**. They are not the same report, and a fresh ``Ledger`` cannot
stand in for this one -- it says ``{"degraded": false, "fact_count": 0}``,
which is not "everything works", it is "nothing has been tried".

So this module is a *pre-turn* producer. It has four properties, and each one is
a requirement rather than a nicety:

* **No network.** It reads ``bootstrap.ee_status()``, the verdict of an
  initialisation that already happened; it never calls :func:`~cropup.bootstrap.ee_ready`,
  because that would initialise Earth Engine, and a capability strip must not
  cost a round trip on every page load. Model files are probed in the local
  cache, never downloaded.
* **Offline-safe and cheap.** Nothing here imports ``ee``, ``numpy``,
  ``onnxruntime`` or ``cropup.rag``. The ``.npy`` header is parsed by hand
  (:func:`_npy_header`) and the corpus digest is recomputed with ``hashlib``,
  so the endpoint still answers when the heavy half of the process is exactly
  what is broken.
* **It degrades instead of raising.** Every probe is wrapped; a probe that
  cannot answer returns ``UNKNOWN`` carrying the exception text. An honesty
  endpoint that 500s during an outage is the one that was needed.
* **JSON-serialisable.** :func:`capability_report` returns plain dicts, lists,
  strings, numbers, bools and ``None``.

The coverage cliffs of SPEC 3.3 are reported as capabilities too, because
"CropSuite is Africa-only" is not an implementation detail -- it is the
difference between an answer and a US-only answer shown to a Tanzanian farmer.
With no confirmed field they report ``UNKNOWN``: the restriction is known, but
whether it bites is not, and inventing a verdict is the same failure as
inventing a number.
"""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from . import bootstrap
from .config import Settings, get_settings

__all__ = [
    "CapabilityStatus",
    "Capability",
    "CoverageCliff",
    "IndexState",
    "COVERAGE_CLIFFS",
    "earth_engine_capability",
    "nlu_capability",
    "data_capabilities",
    "rag_index_state",
    "rag_capability",
    "coverage_capabilities",
    "capability_report",
    "clear_cache",
]


class CapabilityStatus(str, Enum):
    """How a capability stands. ``UNKNOWN`` is a real answer, not a placeholder:
    it means nobody has looked, and saying so beats guessing either way."""

    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"

    def __str__(self) -> str:  # so f"{status}" is the wire value
        return self.value


@dataclass(frozen=True)
class Capability:
    """One thing CropUp can or cannot do, and what the farmer loses when it cannot.

    ``lost`` is the honest half. "the NLU model is missing" means nothing to a
    farmer; "questions are routed by keywords only, so an unusual phrasing will
    be asked about rather than answered" is the same fact in the terms they care
    about.
    """

    name: str
    status: CapabilityStatus
    detail: str
    lost: tuple[str, ...] = ()
    info: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True only for ``READY``. ``DEGRADED`` is usable but not whole."""
        return self.status is CapabilityStatus.READY

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "ok": self.ok,
            "detail": self.detail,
            "lost": list(self.lost),
            "info": dict(self.info),
        }

    def __repr__(self) -> str:
        return f"Capability({self.name}, {self.status.value}: {self.detail})"


# ---------------------------------------------------------------------------
# Coverage cliffs (SPEC 3.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageCliff:
    """A source whose coverage stops at a border, declared rather than probed.

    ``bbox`` is the *declared* extent -- from the asset's own grid definition as
    recorded in ``ee_registry.json`` -- not a mask test. Inside it means "this
    source claims this ground", never "there is a value at this pixel": at the
    exact Arusha pixel CropSuite is inside its grid and still masked for 47 of
    48 crops. Only an Earth Engine read can answer the second question, and this
    module does not make one.
    """

    name: str
    label: str
    assets: tuple[str, ...]
    region: str
    bbox: tuple[float, float, float, float]  # (min_lat, min_lon, max_lat, max_lon)
    spec: str
    note: str
    lost: tuple[str, ...] = ()

    def contains(self, lat: float, lon: float) -> bool:
        min_lat, min_lon, max_lat, max_lon = self.bbox
        return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon

    def as_dict(self) -> dict[str, Any]:
        min_lat, min_lon, max_lat, max_lon = self.bbox
        return {
            "name": self.name,
            "label": self.label,
            "assets": list(self.assets),
            "region": self.region,
            "declared_bbox": {
                "min_lat": min_lat,
                "min_lon": min_lon,
                "max_lat": max_lat,
                "max_lon": max_lon,
            },
            "spec": self.spec,
            "note": self.note,
        }


#: The restrictions SPEC 3.3 records, each verified live against the three test
#: points before it was written down. The boxes come from the grid definitions
#: in ``cropup/data/ee_registry.json`` (CropSuite states its own transform:
#: 9600x9000 px at 1/120 degree from (-25, 39), i.e. lon -25..55, lat -36..39).
COVERAGE_CLIFFS: tuple[CoverageCliff, ...] = (
    CoverageCliff(
        name="coverage:crop_suitability",
        label="Crop suitability (CropSuite)",
        assets=(
            "projects/sat-io/open-datasets/CROP_SUITE/crop_suitability",
            "projects/sat-io/open-datasets/CROP_SUITE/climate_suitability",
            "projects/sat-io/open-datasets/CROP_SUITE/optimal_sowing_date",
        ),
        region="Africa",
        bbox=(-36.0, -25.0, 39.0, 55.0),
        spec="SPEC 3.3",
        note=(
            "Africa-only grid. Being inside it is not a value: at the exact Arusha pixel "
            "crop_suitability is masked for 47 of 48 crops in all 6 scenarios, so the read "
            "uses a neighbourhood reduction and says that it did."
        ),
        lost=("crop suitability scoring; crop selection falls back to climate and soil evidence",),
    ),
    CoverageCliff(
        name="coverage:et0_forecast",
        label="Reference ET forecast (fret)",
        assets=("projects/climate-engine/fret/forecast/eto",),
        region="Conterminous United States",
        bbox=(24.0, -125.0, 50.0, -66.0),
        spec="SPEC 3.3",
        note=(
            "CONUS only; returns null at both Tanzanian test points. Forward-looking "
            "irrigation advice therefore exists only in the US and must never be presented "
            "to a farmer outside it."
        ),
        lost=("the 7-day reference-ET forecast; irrigation advice is retrospective only",),
    ),
    CoverageCliff(
        name="coverage:openet",
        label="OpenET ensemble actual ET",
        assets=("OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0",),
        region="Conterminous United States",
        bbox=(24.0, -125.0, 50.0, -66.0),
        spec="SPEC 3.3",
        note=(
            "CONUS only; null at both Tanzanian points. Actual ET elsewhere comes from "
            "SSEBop VIIRS, which is global."
        ),
        lost=("the OpenET cross-model ensemble; actual ET falls through to SSEBop VIIRS",),
    ),
    CoverageCliff(
        name="coverage:polaris_soil",
        label="POLARIS soil properties",
        assets=(
            "projects/sat-io/open-datasets/polaris/ph_mean",
            "projects/sat-io/open-datasets/polaris/clay_mean",
            "projects/sat-io/open-datasets/polaris/sand_mean",
            "projects/sat-io/open-datasets/polaris/silt_mean",
            "projects/sat-io/open-datasets/polaris/om_mean",
            "projects/sat-io/open-datasets/polaris/ksat_mean",
        ),
        region="United States (CONUS)",
        bbox=(24.0, -125.0, 50.0, -66.0),
        spec="SPEC 3.3",
        note=(
            "US only, and it returns None rather than 0 outside -- a clean boundary, which "
            "is why it is safe as the second link of the soil chain."
        ),
        lost=(),  # the chain continues: iSDA covers Africa, SoilGrids is global
    ),
    CoverageCliff(
        name="coverage:precipitation",
        label="CHIRPS daily rainfall",
        assets=("UCSB-CHG/CHIRPS/DAILY",),
        region="50S to 50N",
        bbox=(-50.0, -180.0, 50.0, 180.0),
        spec="SPEC 3.1",
        note="No data above 50 degrees of latitude in either hemisphere.",
        lost=("rainfall totals; a field above 50 degrees has no precipitation source here",),
    ),
)


# ---------------------------------------------------------------------------
# Earth Engine
# ---------------------------------------------------------------------------


def earth_engine_capability(settings: Settings | None = None) -> Capability:
    """Whether Earth Engine is usable, read from the recorded verdict.

    Never initialises: this is a page-load endpoint and ``ee_ready()`` would
    make it a network call. Before anything has tried -- ``/api/health`` and the
    first field run both do -- the honest answer is ``UNKNOWN``.
    """
    settings = settings or get_settings()
    state = bootstrap.ee_status()
    # Farmer-facing: this string is rendered in the capability strip on first
    # paint, with no gesture, so it carries no spec references and no internal
    # vocabulary. The precise version lives in `detail` for the evidence layer.
    lost = (
        "Checking your field from satellites: field health, watering advice and "
        "which crop to plant. Questions answered from the farming guides still work.",
    )
    info: dict[str, Any] = dict(state)
    info["enabled"] = settings.ee_enabled
    info["deadline"] = bootstrap.deadline_status(settings)

    if not settings.ee_enabled:
        return Capability(
            "earth_engine",
            CapabilityStatus.UNAVAILABLE,
            "disabled by CROPUP_EE_ENABLED=0; this is the null-adapter run of SPEC 10",
            lost,
            info,
        )
    if not state.get("attempted"):
        return Capability(
            "earth_engine",
            CapabilityStatus.UNKNOWN,
            "not initialised yet; nothing has tried, so nothing is known. "
            "GET /api/health initialises it.",
            lost,
            info,
        )
    if not state.get("ready"):
        return Capability(
            "earth_engine",
            CapabilityStatus.UNAVAILABLE,
            state.get("error") or "initialisation failed for an unrecorded reason",
            lost,
            info,
        )
    if not state.get("verified"):
        return Capability(
            "earth_engine",
            CapabilityStatus.DEGRADED,
            f"initialised for project {state.get('project_id')} but never verified with a "
            "round trip, so 'ready' means credentials parsed, not usable",
            lost,
            info,
        )
    return Capability(
        "earth_engine",
        CapabilityStatus.READY,
        f"project {state.get('project_id')}, verified with a round trip",
        (),
        info,
    )


# ---------------------------------------------------------------------------
# NLU encoder
# ---------------------------------------------------------------------------


#: A Hugging Face repo id is ``name`` or ``namespace/name`` over this alphabet.
#: Anything rooted, dotted-relative, backslashed or deeper than one slash is a
#: path somebody typed, not a repo anyone can fetch.
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


def _looks_like_repo_id(model_id: str) -> bool:
    """Is ``CROPUP_EMBED_MODEL_ID`` a hub repo, or a local directory?

    The hub answers "not cached" for both, so without this a typo'd path --
    ``/opt/models/minilm-l6`` with the directory renamed -- reports "a download
    is permitted and will be attempted on first use". No download will ever
    happen, and the operator goes looking at the network for a mistake on disk.
    """
    if model_id.startswith(("~", ".", "/", "\\")) or os.path.isabs(model_id):
        return False
    return bool(_REPO_ID_RE.match(model_id))


def _cached_model_file(settings: Settings, filename: str) -> tuple[str | None, str]:
    """Local path of one model file, found without downloading anything.

    Mirrors the resolution order of ``nlu.embed._resolve_file`` -- a directory
    on disk first, then the Hugging Face cache -- but with
    ``try_to_load_from_cache``, which only ever looks locally.
    """
    local_dir = Path(settings.embed_model_id).expanduser()
    if local_dir.is_dir():
        candidate = local_dir / filename
        return (str(candidate), "local directory") if candidate.is_file() else (None, "local directory")
    if not _looks_like_repo_id(settings.embed_model_id):
        # Not a repo id, so the hub is the wrong place to look and the wrong
        # thing to blame: name the directory that is not there.
        return None, f"local directory {local_dir} does not exist"

    try:
        from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415 -- optional, probed
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return None, f"huggingface_hub unusable: {type(exc).__name__}: {exc}"

    try:
        found = try_to_load_from_cache(
            repo_id=settings.embed_model_id,
            filename=filename,
            cache_dir=str(settings.model_cache_dir) if settings.model_cache_dir else None,
        )
    except Exception as exc:  # noqa: BLE001 - a corrupt cache must not 500 this endpoint
        return None, f"cache unreadable: {type(exc).__name__}: {exc}"
    # The hub returns a sentinel object for "known not to exist"; only a str is a file.
    return (found, "hub cache") if isinstance(found, str) else (None, "hub cache")


def nlu_capability(settings: Settings | None = None) -> Capability:
    """Whether the MiniLM encoder could be loaded, without loading it.

    Checks the two runtimes with ``find_spec`` (which does not execute them) and
    the two model files in the local cache. If the encoder has already been
    loaded this process, its own verdict is preferred -- an attempt that failed
    outranks a file that looks present.
    """
    settings = settings or get_settings()
    lost = (
        "embedding intent routing and RAG retrieval. The keyword tier still routes "
        "(SPEC 5.1), so unusual phrasings are asked about rather than answered.",
    )
    is_repo_id = _looks_like_repo_id(settings.embed_model_id)
    info: dict[str, Any] = {
        "model_id": settings.embed_model_id,
        "model_id_kind": "hub repo id" if is_repo_id else "local path",
        "onnx_file": settings.embed_onnx_file,
        "dim": settings.embed_dim,
        # "a download can happen", not "the flag is set": a local path is never
        # fetched however permissive CROPUP_ALLOW_MODEL_DOWNLOAD is.
        "download_allowed": (
            is_repo_id and settings.allow_model_download and not os.environ.get("HF_HUB_OFFLINE")
        ),
    }

    missing_runtimes = [
        name for name in ("onnxruntime", "tokenizers") if importlib.util.find_spec(name) is None
    ]
    info["missing_runtimes"] = missing_runtimes
    if missing_runtimes:
        return Capability(
            "nlu:encoder",
            CapabilityStatus.UNAVAILABLE,
            f"runtime missing: {', '.join(missing_runtimes)}",
            lost,
            info,
        )

    onnx_path, onnx_where = _cached_model_file(settings, settings.embed_onnx_file)
    tokenizer_path, _ = _cached_model_file(settings, settings.embed_tokenizer_file)
    info["source"] = onnx_where
    info["onnx_path"] = onnx_path
    info["tokenizer_present"] = tokenizer_path is not None
    if onnx_path and os.path.isfile(onnx_path):
        info["onnx_size_mb"] = round(os.path.getsize(onnx_path) / 1e6, 2)

    # A load already attempted this process knows more than the filesystem does.
    # Read it only if the module is imported: importing it here would pull numpy
    # into an endpoint that must answer when numpy is what is broken.
    embed_module = sys.modules.get("cropup.nlu.embed")
    if embed_module is not None:
        try:
            status = embed_module.embed_status(settings)
            info["loaded"] = bool(status.get("available"))
            info["load_attempted"] = bool(status.get("attempted"))
            if status.get("attempted") and not status.get("available"):
                return Capability(
                    "nlu:encoder",
                    CapabilityStatus.UNAVAILABLE,
                    status.get("error") or "the encoder failed to load for an unrecorded reason",
                    lost,
                    info,
                )
        except Exception as exc:  # noqa: BLE001 - the probe is not the product
            info["embed_status_error"] = f"{type(exc).__name__}: {exc}"

    if onnx_path is None or tokenizer_path is None:
        absent = ", ".join(
            name
            for name, path in (
                (settings.embed_onnx_file, onnx_path),
                (settings.embed_tokenizer_file, tokenizer_path),
            )
            if path is None
        )
        if not is_repo_id:
            return Capability(
                "nlu:encoder",
                CapabilityStatus.UNAVAILABLE,
                f"{absent} is not under {Path(settings.embed_model_id).expanduser()}: "
                "CROPUP_EMBED_MODEL_ID names a local path, not a Hugging Face repo, "
                "so nothing will download it",
                lost,
                info,
            )
        if info["download_allowed"]:
            return Capability(
                "nlu:encoder",
                CapabilityStatus.DEGRADED,
                f"{absent} is not in the local cache; a download is permitted and will be "
                "attempted on first use, so the first turn may be slow or may fail",
                lost,
                info,
            )
        return Capability(
            "nlu:encoder",
            CapabilityStatus.UNAVAILABLE,
            f"{absent} is not in the local cache and downloads are disabled",
            lost,
            info,
        )
    return Capability(
        "nlu:encoder",
        CapabilityStatus.READY,
        f"{settings.embed_model_id} present in the {onnx_where}",
        (),
        info,
    )


# ---------------------------------------------------------------------------
# Committed data artifacts
# ---------------------------------------------------------------------------

#: What each artifact carries, in the terms the answer is given in.
_DATA_LOSS = {
    "ee_registry": "every Earth Engine read: the source chains and their scaling live here",
    "crops": "crop name matching, including the Swahili aliases",
    "gazetteer": "place-name resolution; a farmer who names a town cannot be located",
    "disease_library": "the rules engine and the 384 disease knowledge cards",
}


def data_capabilities(settings: Settings | None = None) -> list[Capability]:
    """One capability per committed artifact, with its record count.

    The counts come from :func:`cropup.bootstrap.count_records`, the same
    function ``/api/health`` uses, so the two endpoints cannot disagree about
    how many crops are shipped.
    """
    settings = settings or get_settings()
    out: list[Capability] = []
    for name, path in settings.data_files().items():
        lost = (_DATA_LOSS.get(name, f"whatever reads {name}"),)
        info: dict[str, Any] = {"path": str(path), "expected": bootstrap.EXPECTED_COUNTS.get(name)}
        if not path.is_file():
            out.append(
                Capability(f"data:{name}", CapabilityStatus.UNAVAILABLE, f"missing: {path}", lost, info)
            )
            continue
        try:
            count = bootstrap.count_records(name, path)
        except Exception as exc:  # noqa: BLE001 - an unreadable file is a reportable state
            info["error"] = f"{type(exc).__name__}: {exc}"
            out.append(
                Capability(
                    f"data:{name}",
                    CapabilityStatus.UNAVAILABLE,
                    f"unreadable: {info['error']}",
                    lost,
                    info,
                )
            )
            continue
        info["records"] = count
        expected = bootstrap.EXPECTED_COUNTS.get(name)
        if count == 0:
            out.append(Capability(f"data:{name}", CapabilityStatus.UNAVAILABLE, "0 records", lost, info))
        elif expected is not None and count != expected:
            out.append(
                Capability(
                    f"data:{name}",
                    CapabilityStatus.DEGRADED,
                    f"{count} records, SPEC expects {expected}",
                    lost,
                    info,
                )
            )
        else:
            out.append(Capability(f"data:{name}", CapabilityStatus.READY, f"{count} records", (), info))
    return out


# ---------------------------------------------------------------------------
# RAG index
# ---------------------------------------------------------------------------

# Mirrors cropup.rag.index.{SIDECAR_FILE, VECTORS_FILE, INDEX_FORMAT} and
# cropup.rag.corpus.CARDS_FILE. Named here rather than imported so that this
# module -- which has to answer when the heavy half of the process is broken --
# needs neither numpy nor the 1,200-line corpus builder. If the layout in
# rag/index.py changes, the mismatch shows up here as a reported problem, not as
# a silent pass.
_SIDECAR_FILE = "index.json"
_VECTORS_FILE = "index.npy"
_CARDS_FILE = "cards.jsonl"
_INDEX_FORMAT = 1

_NPY_MAGIC = b"\x93NUMPY"

# (mtime_ns, size, digest, card_count) per cards.jsonl path. Hashing 850 KB of
# cards costs ~10 ms; a capability strip that polls should not pay it twice for
# a file that has not changed.
_digest_cache: dict[str, tuple[int, int, str, int]] = {}


@dataclass(frozen=True)
class IndexState:
    """The RAG index's verdict, shared by /api/capabilities and /api/health."""

    status: CapabilityStatus
    detail: str
    info: Mapping[str, Any] = field(default_factory=dict)
    problems: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status is CapabilityStatus.READY


def _npy_header(path: Path) -> dict[str, Any]:
    """``shape`` and ``dtype`` of a .npy file, without importing numpy.

    The format is a 6-byte magic, a version, a header length and a Python dict
    literal. Reading it costs one 128-byte read instead of a numpy import and an
    mmap, which matters because this endpoint exists for the case where numpy
    itself is the thing that is broken (SPEC 1.1).
    """
    with path.open("rb") as handle:
        if handle.read(6) != _NPY_MAGIC:
            raise ValueError(f"{path.name} is not a .npy file")
        major = handle.read(1)[0]
        handle.read(1)  # minor version: irrelevant to the two header layouts
        if major == 1:
            length = int.from_bytes(handle.read(2), "little")
        elif major in (2, 3):
            length = int.from_bytes(handle.read(4), "little")
        else:
            raise ValueError(f"unsupported .npy major version {major}")
        header = ast.literal_eval(handle.read(length).decode("latin1"))
    if not isinstance(header, dict) or "shape" not in header or "descr" not in header:
        raise ValueError(f"{path.name} has no usable .npy header")
    shape = tuple(int(n) for n in header["shape"])
    return {"shape": shape, "dtype": str(header["descr"]), "fortran_order": bool(header.get("fortran_order"))}


def _corpus_digest(path: Path) -> tuple[str, int]:
    """``(digest, card_count)`` for cards.jsonl, hashed the way corpus.py does.

    ``rag.corpus.corpus_digest`` hashes ``card_id`` and ``embed_text`` with NUL
    separators; ``Card.__post_init__`` fills an empty ``embed_text`` from title
    and body, and that fallback is mirrored here so the two digests agree. A
    digest computed any other way would report a mismatch on a healthy index,
    which is a worse failure than not checking.
    """
    stat = path.stat()
    key = str(path)
    cached = _digest_cache.get(key)
    if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2], cached[3]

    digest = hashlib.sha256()
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            embed_text = str(row.get("embed_text") or "")
            if not embed_text.strip():
                embed_text = f"{row['title']}\n{row['body']}"
            digest.update(str(row["card_id"]).encode("utf-8"))
            digest.update(b"\0")
            digest.update(embed_text.encode("utf-8"))
            digest.update(b"\0")
            count += 1
    value = digest.hexdigest()
    _digest_cache[key] = (stat.st_mtime_ns, stat.st_size, value, count)
    return value, count


def rag_index_state(settings: Settings | None = None) -> IndexState:
    """Is the embedding index built, loadable, and built from *these* cards?

    Three failures are separated because they mean different things:

    * **not built** -- retrieval cannot answer at all;
    * **structurally broken** (row count, dimension, dtype) -- ``CardIndex``
      would refuse to load it;
    * **inconsistent** -- it loads and retrieves, from a corpus or an encoder
      that is no longer the one on disk. That is the dangerous one: SPEC 6
      promises a verbatim snippet with a citation, and an index built from other
      cards cites the wrong document confidently.
    """
    settings = settings or get_settings()
    directory = settings.corpus_dir
    sidecar_path = directory / _SIDECAR_FILE
    vectors_path = directory / _VECTORS_FILE
    cards_path = directory / _CARDS_FILE
    info: dict[str, Any] = {"corpus_dir": str(directory)}

    if not directory.is_dir():
        return IndexState(CapabilityStatus.UNAVAILABLE, f"{directory} does not exist", info)
    absent = [p.name for p in (sidecar_path, vectors_path) if not p.is_file()]
    if absent:
        return IndexState(
            CapabilityStatus.UNAVAILABLE,
            f"{', '.join(absent)} missing; the index has not been built (tools/build_corpus.py)",
            info,
        )

    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a corrupt sidecar is a state, not a crash
        return IndexState(
            CapabilityStatus.UNAVAILABLE, f"{_SIDECAR_FILE} is unreadable: {type(exc).__name__}: {exc}", info
        )
    if not isinstance(payload, Mapping):
        return IndexState(CapabilityStatus.UNAVAILABLE, f"{_SIDECAR_FILE} is not a JSON object", info)

    card_count = payload.get("card_count")
    info.update(
        {
            "format": payload.get("format"),
            "model_id": payload.get("model_id"),
            "dim": payload.get("dim"),
            "card_count": card_count,
            "built_at": payload.get("built_at"),
            "digest": payload.get("digest"),
        }
    )
    if payload.get("format") != _INDEX_FORMAT:
        return IndexState(
            CapabilityStatus.UNAVAILABLE,
            f"index format {payload.get('format')!r} is not the {_INDEX_FORMAT} this build reads",
            info,
        )

    try:
        header = _npy_header(vectors_path)
    except Exception as exc:  # noqa: BLE001
        return IndexState(
            CapabilityStatus.UNAVAILABLE, f"{_VECTORS_FILE} is unusable: {type(exc).__name__}: {exc}", info
        )
    info["vectors"] = header

    structural: list[str] = []
    shape = header["shape"]
    if len(shape) != 2:
        structural.append(f"vectors have shape {shape}, expected two dimensions")
    else:
        rows, dim = shape
        if isinstance(card_count, int) and rows != card_count:
            structural.append(f"{rows} vectors for {card_count} cards")
        if isinstance(payload.get("dim"), int) and dim != payload["dim"]:
            structural.append(f"vectors are {dim}-d, the sidecar says {payload['dim']}")
        if dim != settings.embed_dim:
            structural.append(f"vectors are {dim}-d, CROPUP_EMBED_DIM says {settings.embed_dim}")
    if header["dtype"] not in ("<f4", "|f4", "=f4"):
        structural.append(f"vectors are {header['dtype']}, expected float32")
    if structural:
        return IndexState(
            CapabilityStatus.UNAVAILABLE,
            "the index will not load: " + "; ".join(structural),
            info,
            tuple(structural),
        )

    problems: list[str] = []
    if payload.get("model_id") != settings.embed_model_id:
        problems.append(
            f"built with {payload.get('model_id')!r}, this process encodes with "
            f"{settings.embed_model_id!r}: the query and the cards are in different spaces"
        )
    if not cards_path.is_file():
        problems.append(f"{_CARDS_FILE} is absent, so the index cannot be checked against the corpus")
    else:
        info["cards_file"] = str(cards_path)
        try:
            digest, on_disk = _corpus_digest(cards_path)
            info["corpus_digest"] = digest
            info["corpus_cards"] = on_disk
            if payload.get("digest") and digest != payload["digest"]:
                problems.append(
                    f"built from a different card set ({_CARDS_FILE} now hashes to {digest[:12]}, "
                    f"the index carries {str(payload['digest'])[:12]}); rebuild it"
                )
            if isinstance(card_count, int) and on_disk != card_count:
                problems.append(f"{on_disk} cards on disk, {card_count} in the index")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{_CARDS_FILE} could not be hashed: {type(exc).__name__}: {exc}")
        if cards_path.stat().st_mtime_ns > sidecar_path.stat().st_mtime_ns:
            problems.append(f"{_CARDS_FILE} is newer than the index")

    if problems:
        return IndexState(
            CapabilityStatus.DEGRADED,
            f"{card_count} cards, but: " + "; ".join(problems),
            info,
            tuple(problems),
        )
    return IndexState(
        CapabilityStatus.READY,
        f"{card_count} cards, {shape[1]}-d, built {payload.get('built_at') or 'at an unrecorded time'}",
        info,
    )


def rag_capability(settings: Settings | None = None) -> Capability:
    """The RAG index as a capability. SPEC 1.2: this serves 72% of real traffic."""
    settings = settings or get_settings()
    lost = (
        "knowledge answers, which are 56 of 78 real questions (SPEC 1.2); only the "
        "questionnaire and the measured pipelines remain.",
    )
    try:
        state = rag_index_state(settings)
    except Exception as exc:  # noqa: BLE001 - never 500 the honesty endpoint
        return Capability(
            "rag:index",
            CapabilityStatus.UNKNOWN,
            f"could not be checked: {type(exc).__name__}: {exc}",
            lost,
            {"corpus_dir": str(settings.corpus_dir)},
        )
    return Capability(
        "rag:index",
        state.status,
        state.detail,
        () if state.status is CapabilityStatus.READY else lost,
        state.info,
    )


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def _registered_assets(settings: Settings) -> tuple[frozenset[str], str | None]:
    """Asset ids in ee_registry.json, and why not when it could not be read.

    The two halves are returned separately because an empty set means two very
    different things. A registry that lists no datasets is *readable and empty*:
    every asset below really is unregistered and really cannot be resolved. A
    registry that could not be parsed is *unchecked*: the assets may be fine and
    nobody knows. Collapsing both into ``frozenset()`` made the honesty endpoint
    report "ee_registry.json could not be read" about a file it had just read.
    """
    try:
        payload = json.loads(settings.ee_registry_path.read_text(encoding="utf-8"))
        return frozenset(str(d["asset_id"]) for d in payload["datasets"]), None
    except Exception as exc:  # noqa: BLE001 - data:ee_registry reports it as a capability too
        return frozenset(), f"{type(exc).__name__}: {exc}"


def _echo(value: object) -> str | None:
    """What was given, as a short string that is always safe to put in JSON.

    Never raises and never grows: a ``repr`` can be megabytes long (or can raise
    on the way out), and this endpoint's whole job is to answer when things are
    misbehaving.
    """
    if value is None:
        return None
    try:
        text = repr(value)
    except Exception as exc:  # noqa: BLE001 - an object whose repr raises is still describable
        return f"<{type(value).__name__} with an unusable repr: {type(exc).__name__}>"
    return text if len(text) <= 60 else text[:57] + "..."


def _usable_point(lat: float | None, lon: float | None) -> tuple[bool, str]:
    """Is this a point we can test a bbox against? Never raises on bad input."""
    if lat is None and lon is None:
        return False, "no field location has been confirmed"
    if lat is None or lon is None:
        return False, "a location needs both a latitude and a longitude"
    try:
        lat_f, lon_f = float(lat), float(lon)
    # Not only TypeError/ValueError: ``float(x)`` runs ``x.__float__``, which is
    # the caller's code and may raise anything at all. Whatever it raises, the
    # answer here is the same -- this is not a point we can test -- and it is an
    # answer, not an exception for the endpoint to die of.
    except Exception as exc:  # noqa: BLE001
        why = f"latitude {_echo(lat)} and longitude {_echo(lon)} are not numbers"
        if not isinstance(exc, (TypeError, ValueError)):
            why += f" ({type(exc).__name__}: {_echo(str(exc))})"
        return False, why
    if not (math.isfinite(lat_f) and math.isfinite(lon_f)):
        return False, "latitude and longitude must be finite"
    if not -90.0 <= lat_f <= 90.0:
        return False, f"latitude {lat_f:g} is outside [-90, 90]"
    if not -180.0 <= lon_f <= 180.0:
        return False, f"longitude {lon_f:g} is outside [-180, 180]"
    return True, ""


def coverage_capabilities(
    lat: float | None = None,
    lon: float | None = None,
    settings: Settings | None = None,
) -> list[Capability]:
    """The SPEC 3.3 cliffs, as capabilities -- for this field if one is known.

    With no confirmed location every cliff is ``UNKNOWN`` and names its region,
    which is what the strip should say before the farmer has told us where they
    are. With a location the verdict is a bounding-box test against the declared
    grid: it can say "this field is outside CropSuite", and it deliberately
    cannot say "there is a value here" -- only an Earth Engine read can, and
    this module makes none.
    """
    settings = settings or get_settings()
    registered, registry_error = _registered_assets(settings)
    readable = registry_error is None
    usable, why = _usable_point(lat, lon)
    out: list[Capability] = []

    # Unreadable is "unchecked", and unchecked must not read like "checked and
    # fine": with ee_registry.json unparseable nothing is known about whether
    # these assets can be resolved, so every detail below says so rather than
    # reporting a clean bounding-box verdict on a source nobody has looked at.
    # A registry that *was* read and lists nothing is a different fact entirely
    # -- then the assets genuinely are unregistered -- and it is reported by the
    # unregistered branch below, not by this note.
    registry_note = (
        ""
        if readable
        else f" (ee_registry.json could not be read -- {registry_error} -- so whether these "
        "assets are registered is unchecked, and data:ee_registry says why)"
    )

    for cliff in COVERAGE_CLIFFS:
        info = cliff.as_dict()
        # Only a readable registry can say an asset is absent from it. When it
        # could not be read, "unchecked" is not "registered" and not
        # "unregistered" either, so it is reported as None.
        unregistered = [a for a in cliff.assets if a not in registered] if readable else []
        info["registry_readable"] = readable
        info["registered"] = (not unregistered) if readable else None
        if registry_error is not None:
            info["registry_error"] = registry_error
        if unregistered:
            info["unregistered_assets"] = unregistered

        if unregistered:
            out.append(
                Capability(
                    cliff.name,
                    CapabilityStatus.UNAVAILABLE,
                    "ee_registry.json was read and lists no datasets at all, so nothing can "
                    f"read the {len(cliff.assets)} assets behind this"
                    if not registered
                    else f"{len(unregistered)} of {len(cliff.assets)} assets are not in "
                    "ee_registry.json, so nothing can read them",
                    cliff.lost,
                    info,
                )
            )
            continue
        if not usable:
            out.append(
                Capability(
                    cliff.name,
                    CapabilityStatus.UNKNOWN,
                    f"{cliff.label} covers {cliff.region} only; {why}, so it is not known "
                    f"whether this farmer's field is inside it{registry_note}",
                    cliff.lost,
                    info,
                )
            )
            continue

        lat_f, lon_f = float(lat), float(lon)  # type: ignore[arg-type] -- _usable_point checked
        info["point"] = {"lat": lat_f, "lon": lon_f}
        if cliff.contains(lat_f, lon_f):
            out.append(
                Capability(
                    cliff.name,
                    # READY means whole. With the registry unreadable the bbox test
                    # still holds, but nothing is known about resolving the asset
                    # to read it, so the honest verdict is DEGRADED with the
                    # reason attached.
                    CapabilityStatus.READY if readable else CapabilityStatus.DEGRADED,
                    f"the field at {lat_f:g}, {lon_f:g} is inside the declared {cliff.region} "
                    f"coverage; a pixel there may still be masked{registry_note}",
                    () if readable else cliff.lost,
                    info,
                )
            )
        else:
            out.append(
                Capability(
                    cliff.name,
                    CapabilityStatus.UNAVAILABLE,
                    f"the field at {lat_f:g}, {lon_f:g} is outside {cliff.region}: "
                    f"{cliff.label} has nothing to say about it",
                    cliff.lost,
                    info,
                )
            )
    return out


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _point_report(lat: float | None, lon: float | None) -> dict[str, Any] | None:
    """The field location as the payload should state it, or ``None`` if none
    was given. A location that cannot be used says so rather than disappearing:
    a mistyped coordinate that silently becomes "no location" is how a report
    about the wrong field gets read as a report about no field."""
    if lat is None and lon is None:
        return None
    usable, why = _usable_point(lat, lon)
    if usable:
        return {
            "lat": float(lat),  # type: ignore[arg-type] -- checked above
            "lon": float(lon),  # type: ignore[arg-type]
            "usable": True,
            "detail": why or "inside the range coverage can be tested against",
        }
    # Never echo an unusable coordinate into the numeric fields. ``float("nan")``
    # and ``inf`` serialise to tokens no JSON parser accepts, and an arbitrary
    # object does not serialise at all -- either one turns the endpoint that
    # exists for an outage into the 500 it is supposed to prevent. The value is
    # NAMED as unusable instead (SPEC 4), under a key nothing reads as a
    # coordinate, so a caller that plots ``point.lat`` plots nothing rather than
    # somewhere.
    return {
        "lat": None,
        "lon": None,
        "usable": False,
        "detail": why,
        "given": {"lat": _echo(lat), "lon": _echo(lon)},
    }


#: The three probed parts of understanding a free-text turn (SPEC 5), each with
#: the capability that answers for it and what the farmer loses when it is gone.
#: The consequences are per-part on purpose: "a place the farmer names cannot be
#: resolved" is false when only the encoder is missing, and this endpoint is the
#: last place that may state a consequence which did not happen.
_FREE_TEXT_PARTS: tuple[tuple[str, str, str], ...] = (
    (
        "embedding routing",
        "nlu:encoder",
        "an unusual phrasing is asked about rather than routed",
    ),
    (
        "crop matching",
        "data:crops",
        "a crop the farmer names is asked about rather than resolved",
    ),
    (
        "place resolution",
        "data:gazetteer",
        "a place the farmer names is asked about rather than resolved",
    ),
)


def _safe_point_report(lat: float | None, lon: float | None) -> dict[str, Any] | None:
    """:func:`_point_report`, but it cannot take the endpoint down with it.

    Everything else in the payload is already wrapped; this was the one part
    that was not, and it is the part that touches the caller's objects --
    ``float(lat)`` runs ``lat.__float__`` and ``repr(lat)`` runs ``lat.__repr__``,
    both of which are somebody else's code. A field location that cannot even be
    described is still named as unusable rather than becoming the 500 this
    endpoint exists to prevent.
    """
    try:
        return _point_report(lat, lon)
    except Exception as exc:  # noqa: BLE001 - the honesty endpoint must answer
        return {
            "lat": None,
            "lon": None,
            "usable": False,
            "detail": f"the given field location could not be read: {type(exc).__name__}: "
            f"{_echo(str(exc))}",
            # The type name, not the value: repr() is a plausible thing to have
            # just raised, so it is not called again here.
            "given": {"lat": f"<{type(lat).__name__}>", "lon": f"<{type(lon).__name__}>"},
        }


def _probe(name: str, fn: Callable[..., Capability], *args: Any) -> Capability:
    """Run one probe; a probe that raises becomes an ``UNKNOWN`` capability."""
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 - the strip must render during an outage
        return Capability(
            name, CapabilityStatus.UNKNOWN, f"probe failed: {type(exc).__name__}: {exc}"
        )


def capability_report(
    settings: Settings | None = None,
    *,
    lat: float | None = None,
    lon: float | None = None,
) -> dict[str, Any]:
    """The ``GET /api/capabilities`` payload: the live degradation matrix.

    Pure, offline and JSON-serialisable. Pass the confirmed field location when
    a session has one and the coverage cliffs are answered for that field
    instead of in the abstract.
    """
    settings = settings or get_settings()
    capabilities: list[Capability] = [
        _probe("earth_engine", earth_engine_capability, settings),
        _probe("nlu:encoder", nlu_capability, settings),
    ]
    try:
        capabilities.extend(data_capabilities(settings))
    except Exception as exc:  # noqa: BLE001
        capabilities.append(
            Capability("data", CapabilityStatus.UNKNOWN, f"probe failed: {type(exc).__name__}: {exc}")
        )
    capabilities.append(_probe("rag:index", rag_capability, settings))
    try:
        capabilities.extend(coverage_capabilities(lat, lon, settings))
    except Exception as exc:  # noqa: BLE001
        capabilities.append(
            Capability("coverage", CapabilityStatus.UNKNOWN, f"probe failed: {type(exc).__name__}: {exc}")
        )

    by_status: dict[str, list[str]] = {s.value: [] for s in CapabilityStatus}
    for capability in capabilities:
        by_status[capability.status.value].append(capability.name)

    found = {c.name: c for c in capabilities}

    def status_of(name: str) -> CapabilityStatus:
        capability = found.get(name)
        return capability.status if capability else CapabilityStatus.UNKNOWN

    def detail_of(name: str) -> str:
        capability = found.get(name)
        return capability.detail if capability else f"{name} was not probed"

    ee_ok = status_of("earth_engine") is CapabilityStatus.READY
    nlu_ok = status_of("nlu:encoder") is CapabilityStatus.READY
    # A DEGRADED index still retrieves; it is listed as degraded so the strip can
    # say why, but withholding every knowledge answer over a stale digest would
    # cost more than it protects.
    rag_ok = status_of("rag:index") in (CapabilityStatus.READY, CapabilityStatus.DEGRADED)
    # Retrieval needs both halves: the index to search and the encoder to turn
    # the question into a vector. Name whichever one is the reason.
    knowledge_detail = detail_of("nlu:encoder") if rag_ok and not nlu_ok else detail_of("rag:index")

    # Understanding a free-text turn is three parts (SPEC 5), and this entry
    # claims all three. It is derived from their probes rather than asserted:
    # with the gazetteer missing a farmer who names a town is not understood at
    # all, and an endpoint whose job is to report degradation must not be the
    # one place that hard-codes True. The keyword tier is the only part with no
    # probe -- it is pure in-process code with no model and no artifact -- which
    # is why it is what the degraded detail promises, and all this entry can
    # promise when the other parts are gone.
    #
    # A part that is DEGRADED is not a part that is gone, and the two must not
    # produce the same sentence: a crop vocabulary with an unexpected record
    # count still matches crops, and an encoder that has to download still
    # routes once it has. So a degraded part is named with its reason and the
    # entry stays usable, a part that is UNAVAILABLE or UNKNOWN takes the entry
    # down, and the consequence spelled out is only the consequence of the parts
    # that are actually gone.
    free_text_gone: list[str] = []
    free_text_degraded: list[str] = []
    free_text_costs: list[str] = []
    for part, probe, cost in _FREE_TEXT_PARTS:
        status = status_of(probe)
        if status is CapabilityStatus.READY:
            continue
        if status is CapabilityStatus.DEGRADED:
            free_text_degraded.append(f"{part} is degraded ({detail_of(probe)})")
        else:
            free_text_gone.append(f"{part} is {status.value}")
            free_text_costs.append(cost)

    if free_text_gone:
        free_text_detail = (
            "keyword tier only (SPEC 5.1): it always runs, but "
            + "; ".join(free_text_gone + free_text_degraded)
            + " -- so "
            + "; ".join(free_text_costs)
        )
    elif free_text_degraded:
        free_text_detail = (
            "rules, embeddings and slot filling are all present, but "
            + "; ".join(free_text_degraded)
        )
    else:
        free_text_detail = "rules, embeddings and slot filling are all available"

    can = {
        "understand_free_text": {
            "ok": not free_text_gone,
            "detail": free_text_detail,
        },
        "answer_from_knowledge": {"ok": bool(rag_ok and nlu_ok), "detail": knowledge_detail},
        "resolve_a_place": {
            "ok": status_of("data:gazetteer") is CapabilityStatus.READY,
            "detail": detail_of("data:gazetteer"),
        },
        "resolve_a_crop": {
            "ok": status_of("data:crops") is CapabilityStatus.READY,
            "detail": detail_of("data:crops"),
        },
        "run_field_analysis": {"ok": ee_ok, "detail": detail_of("earth_engine")},
    }

    impaired = [c for c in capabilities if c.status in (CapabilityStatus.DEGRADED, CapabilityStatus.UNAVAILABLE)]
    unresolved = [c for c in capabilities if c.status is CapabilityStatus.UNKNOWN]

    # What the farmer actually loses. A coverage cliff nobody can evaluate yet --
    # no confirmed field -- costs nothing that is known, so it is not listed as a
    # loss; it is listed as unknown, which is what it is.
    lost: list[str] = []
    for capability in impaired + [c for c in unresolved if not c.name.startswith("coverage:")]:
        for item in capability.lost:
            if item not in lost:
                lost.append(item)

    counts = {status: len(names) for status, names in by_status.items() if names}
    summary = ", ".join(f"{n} {status}" for status, n in counts.items()) or "nothing to report"
    if impaired:
        summary += " | not working: " + ", ".join(c.name for c in impaired)
    blocked = [name for name, entry in can.items() if not entry["ok"]]
    if blocked:
        summary += " | cannot: " + ", ".join(blocked)

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        # Degraded means something is known to be broken or something the farmer
        # can ask for is unavailable. An unevaluated coverage cliff is neither.
        "degraded": bool(impaired or blocked),
        "summary": summary,
        "counts": counts,
        "point": _safe_point_report(lat, lon),
        "can": can,
        "lost": lost,
        "unknown": [c.name for c in unresolved],
        "capabilities": [c.as_dict() for c in capabilities],
        "by_status": {status: names for status, names in by_status.items() if names},
        "deadlines": bootstrap.deadline_status(settings),
        "logging": bootstrap.logging_status(),
    }


def clear_cache() -> None:
    """Drop the corpus-digest cache (tests, and after a rebuild)."""
    _digest_cache.clear()
