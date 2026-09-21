"""MiniLM sentence encoder, ONNX only.

SPEC section 5: ``sentence-transformers/all-MiniLM-L6-v2``, the INT8 export
``onnx/model_qint8_avx512.onnx`` (23 MB, 0.41 s cold, 2.5 ms warm). The runtime
is ``onnxruntime`` + ``tokenizers``; **torch is never imported** -- the local
build is compiled against NumPy 1.x and would abort the process.

Two rules shape this module:

* The encoder is optional. Every entry point returns ``None`` rather than
  raising when the model cannot be loaded, so :mod:`cropup.nlu.classify` can
  fall back to the rule tier and the app still boots. :func:`load_encoder` is
  the one function that raises, for callers that want the reason.
* The failure is remembered. A missing model must not cost a download attempt
  on every farmer turn, so the verdict is cached until :func:`reset`. How long
  that one attempt took is remembered with it and reported by
  :func:`embed_status`, so a slow failure -- a hub lookup that sat on a socket
  for thirty seconds -- is visible on the capability strip rather than being
  indistinguishable from a file that was simply not there.

The files are resolved through ``huggingface_hub``, which serves them from the
local cache; with ``HF_HUB_OFFLINE=1`` (or ``CROPUP_ALLOW_MODEL_DOWNLOAD=0``)
nothing goes to the network. ``CROPUP_EMBED_MODEL_ID`` may also be a directory
on disk, in which case the files are read straight from it.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import Settings, get_settings
from ..errors import ModelUnavailable

__all__ = [
    "Encoder",
    "load_encoder",
    "get_encoder",
    "encode",
    "encode_one",
    "cosine_similarity",
    "embed_status",
    "reset",
]

# Candidates for the padding token, in the order a BERT-family tokenizer.json
# is likely to name it. If none is present we refuse to guess an id.
_PAD_TOKENS = ("[PAD]", "<pad>", "<PAD>")

_LOCK = threading.Lock()
_ENCODER: "Encoder | None" = None
_ERROR: str | None = None
_ATTEMPTED = False
# Wall clock of the one load attempt, success or failure. Reported by
# embed_status(); a failure has no Encoder to carry it.
_LOAD_ELAPSED_S: float | None = None

# Small bounded cache of query vectors. Seed centroids are cached by the caller;
# this one exists because a single turn encodes the same farmer sentence for
# intent routing and again for RAG retrieval.
_VECTOR_CACHE: dict[str, np.ndarray] = {}
_VECTOR_CACHE_MAX = 512


@dataclass(frozen=True)
class Encoder:
    """A loaded ONNX MiniLM. Build it with :func:`load_encoder`."""

    session: Any  # onnxruntime.InferenceSession
    tokenizer: Any  # tokenizers.Tokenizer
    model_id: str
    onnx_file: str
    onnx_path: str
    dim: int
    max_tokens: int
    input_names: tuple[str, ...]
    provider: str
    load_elapsed_s: float

    def encode(self, texts: Sequence[str], *, batch_size: int = 16) -> np.ndarray:
        """Embed ``texts`` -> ``(len(texts), dim)`` float32, L2-normalised rows.

        Mean pooling over the attention mask, which is what the
        sentence-transformers checkpoint was trained with; taking the CLS token
        instead silently costs several points of accuracy.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            out[start : start + len(chunk)] = self._encode_batch(list(chunk))
        return out

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feed: dict[str, np.ndarray] = {}
        for name in self.input_names:
            if name == "input_ids":
                feed[name] = input_ids
            elif name == "attention_mask":
                feed[name] = attention_mask
            elif name == "token_type_ids":
                feed[name] = np.array([e.type_ids for e in encodings], dtype=np.int64)
            else:  # an export we have not seen; better to say so than to feed zeros
                raise ModelUnavailable(self.model_id, f"unexpected ONNX input {name!r}")
        hidden = self.session.run(None, feed)[0]  # (batch, seq, dim)
        mask = attention_mask[:, :, None].astype(np.float32)
        summed = (hidden * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        vectors = (summed / counts).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)

    def describe(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "onnx_file": self.onnx_file,
            "onnx_path": self.onnx_path,
            "onnx_size_mb": round(os.path.getsize(self.onnx_path) / 1e6, 2)
            if os.path.isfile(self.onnx_path)
            else None,
            "dim": self.dim,
            "max_tokens": self.max_tokens,
            "provider": self.provider,
            "load_elapsed_s": round(self.load_elapsed_s, 3),
        }


def _resolve_file(settings: Settings, filename: str) -> str:
    """Local path for one model file, from a directory or the HF cache."""
    local_dir = Path(settings.embed_model_id).expanduser()
    if local_dir.is_dir():
        candidate = local_dir / filename
        if not candidate.is_file():
            raise ModelUnavailable(settings.embed_model_id, f"{candidate} not found")
        return str(candidate)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - huggingface_hub is a dependency
        raise ModelUnavailable(settings.embed_model_id, f"huggingface_hub missing: {exc}") from exc

    try:
        return hf_hub_download(
            repo_id=settings.embed_model_id,
            filename=filename,
            cache_dir=str(settings.model_cache_dir) if settings.model_cache_dir else None,
            local_files_only=not settings.allow_model_download,
        )
    except Exception as exc:  # hub raises many types; the caller only needs the reason
        raise ModelUnavailable(
            settings.embed_model_id, f"{filename}: {type(exc).__name__}: {exc}"
        ) from exc


def load_encoder(settings: Settings | None = None) -> Encoder:
    """Load the ONNX encoder, or raise :class:`ModelUnavailable`.

    Does not touch the module cache; :func:`get_encoder` is the cached door.
    """
    settings = settings or get_settings()
    started = time.perf_counter()

    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ModelUnavailable(settings.embed_model_id, f"runtime missing: {exc}") from exc

    onnx_path = _resolve_file(settings, settings.embed_onnx_file)
    tokenizer_path = _resolve_file(settings, settings.embed_tokenizer_file)

    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    except Exception as exc:
        raise ModelUnavailable(settings.embed_model_id, f"tokenizer: {exc}") from exc

    pad_id = None
    pad_token = None
    for token in _PAD_TOKENS:
        found = tokenizer.token_to_id(token)
        if found is not None:
            pad_id, pad_token = found, token
            break
    if pad_id is None:
        raise ModelUnavailable(
            settings.embed_model_id,
            f"tokenizer has no padding token (looked for {', '.join(_PAD_TOKENS)})",
        )
    tokenizer.enable_truncation(max_length=settings.embed_max_tokens)
    tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        session = ort.InferenceSession(
            onnx_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:
        raise ModelUnavailable(settings.embed_model_id, f"onnxruntime: {exc}") from exc

    output = session.get_outputs()[0]
    dim = output.shape[-1] if isinstance(output.shape[-1], int) else settings.embed_dim
    if dim != settings.embed_dim:
        raise ModelUnavailable(
            settings.embed_model_id,
            f"model emits {dim}-d vectors, CROPUP_EMBED_DIM says {settings.embed_dim}",
        )

    return Encoder(
        session=session,
        tokenizer=tokenizer,
        model_id=settings.embed_model_id,
        onnx_file=settings.embed_onnx_file,
        onnx_path=onnx_path,
        dim=dim,
        max_tokens=settings.embed_max_tokens,
        input_names=tuple(i.name for i in session.get_inputs()),
        provider=session.get_providers()[0],
        load_elapsed_s=time.perf_counter() - started,
    )


def get_encoder(settings: Settings | None = None) -> Encoder | None:
    """The process-wide encoder, or ``None`` if it could not be loaded.

    The failure reason is kept in :func:`embed_status` rather than raised, and
    the load is attempted exactly once per process (per :func:`reset`).
    """
    global _ENCODER, _ERROR, _ATTEMPTED, _LOAD_ELAPSED_S
    if _ENCODER is not None:
        return _ENCODER
    with _LOCK:
        if _ENCODER is not None:
            return _ENCODER
        if _ATTEMPTED:
            return None
        _ATTEMPTED = True
        started = time.perf_counter()
        try:
            _ENCODER = load_encoder(settings)
            _ERROR = None
        except ModelUnavailable as exc:
            _ENCODER = None
            _ERROR = str(exc)
        except Exception as exc:  # a broken cache file should degrade, not crash
            _ENCODER = None
            _ERROR = f"{type(exc).__name__}: {exc}"
        # Timed on both paths: the cost of finding out the model is missing is
        # exactly the number a health check needs.
        _LOAD_ELAPSED_S = time.perf_counter() - started
        return _ENCODER


def encode(texts: Sequence[str], settings: Settings | None = None) -> np.ndarray | None:
    """Embed ``texts``; ``None`` when the model is unavailable."""
    encoder = get_encoder(settings)
    if encoder is None:
        return None
    return encoder.encode(texts)


def encode_one(text: str, settings: Settings | None = None) -> np.ndarray | None:
    """Embed one string, with a small cache. The returned array is read-only."""
    cached = _VECTOR_CACHE.get(text)
    if cached is not None:
        return cached
    encoder = get_encoder(settings)
    if encoder is None:
        return None
    vector = encoder.encode_one(text)
    vector.flags.writeable = False
    with _LOCK:
        if len(_VECTOR_CACHE) >= _VECTOR_CACHE_MAX:
            _VECTOR_CACHE.clear()
        _VECTOR_CACHE[text] = vector
    return vector


def cosine_similarity(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine of one vector against each row of ``matrix``.

    Both are re-normalised here: a centroid is a mean of unit vectors and is not
    itself a unit vector.
    """
    vector = np.asarray(vector, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    v_norm = float(np.linalg.norm(vector))
    if v_norm == 0.0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    m_norms = np.clip(np.linalg.norm(matrix, axis=1), 1e-12, None)
    return (matrix @ (vector / v_norm)) / m_norms


def embed_status(settings: Settings | None = None) -> dict[str, Any]:
    """What the capability strip shows for the encoder. Never loads the model."""
    settings = settings or get_settings()
    status: dict[str, Any] = {
        "component": "nlu:embed",
        "model_id": settings.embed_model_id,
        "onnx_file": settings.embed_onnx_file,
        "attempted": _ATTEMPTED,
        "available": _ENCODER is not None,
        "error": _ERROR,
        # Present even when the load failed, where it is the cost of the failure;
        # None until something has asked for the encoder at all.
        "load_elapsed_s": round(_LOAD_ELAPSED_S, 3) if _LOAD_ELAPSED_S is not None else None,
        "offline": bool(os.environ.get("HF_HUB_OFFLINE")) or not settings.allow_model_download,
    }
    if _ENCODER is not None:
        status.update(_ENCODER.describe())
    return status


def reset() -> None:
    """Forget the encoder and the failure verdict (tests, tools/)."""
    global _ENCODER, _ERROR, _ATTEMPTED, _LOAD_ELAPSED_S
    with _LOCK:
        _ENCODER = None
        _ERROR = None
        _ATTEMPTED = False
        _LOAD_ELAPSED_S = None
        _VECTOR_CACHE.clear()
