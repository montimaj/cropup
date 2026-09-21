"""Settings, read from the environment, with defaults that work unconfigured.

Every variable is ``CROPUP_*``. Nothing here reads the vendored ``OPENFARM_*``
or bare ``EE_PROJECT_ID`` names; ``bootstrap`` exports ``EE_PROJECT_ID`` for the
Earth Engine client library, but it is not an input.

A bad value raises :class:`~cropup.errors.ConfigError` instead of falling back
to the default: a misspelled threshold must not quietly become the one CropUp
was shipped with.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .errors import ConfigError, DataFileError

__all__ = [
    "Settings",
    "get_settings",
    "reload_settings",
    "LOG_LEVELS",
    "PACKAGE_DIR",
    "REPO_ROOT",
]

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

# The EE project verified working with the local user credentials (SPEC 1.1).
DEFAULT_EE_PROJECT_ID = "irrigation-status-474718"

# SPEC section 5: MiniLM INT8 ONNX, 23 MB, 2.5 ms warm. Torch is never imported.
DEFAULT_EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBED_ONNX_FILE = "onnx/model_qint8_avx512.onnx"
DEFAULT_EMBED_TOKENIZER_FILE = "tokenizer.json"
DEFAULT_EMBED_DIM = 384

# Optional tie-breaker only, shipped disabled (739 MB, 74 ms, entailment index 0).
DEFAULT_NLI_MODEL_ID = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"

# Radius of the disc a confirmed point is buffered into before it is sent to
# Earth Engine, and therefore the polygon ``GET /api/geo/field`` shows the
# farmer (SPEC 8). Provenance of the 15 m: it is the buffer the sibling project
# settled on after iterating on buffer size, and CropUp matches that number
# rather than inventing a second one. It is a *field* radius -- what the farmer
# is asking about -- and is deliberately not ``ee_neighbourhood_m`` (300 m),
# which is the wider reduction a coarse or masked asset needs (SPEC 3.3).
DEFAULT_FIELD_RADIUS_M = 15.0

#: The largest radius an operator may configure, matching the ceiling
#: ``dialog.slots`` enforces on a per-field radius. Config sits below every
#: other package, so it cannot import that constant without a cycle; the two
#: are pinned together by a test instead.
MAX_FIELD_RADIUS_M = 5_000.0

# CROPUP_LOG_LEVEL takes one of these names. They are spelled the way uvicorn
# spells them, so the one setting configures both uvicorn and
# ``bootstrap.configure_logging``; the values are the stdlib constants.
LOG_LEVELS: dict[str, int] = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def _raw(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_str(name: str, default: str) -> str:
    return _raw(name) or default


def _env_path(name: str, default: Path) -> Path:
    raw = _raw(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_int(name: str, default: int) -> int:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not a number") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _raw(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name}={raw!r} is not a boolean (use 1/0, true/false, yes/no, on/off)")


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the environment. Build it with :meth:`from_env`."""

    # -- paths --------------------------------------------------------------
    package_dir: Path = PACKAGE_DIR
    data_dir: Path = PACKAGE_DIR / "data"
    corpus_dir: Path = PACKAGE_DIR / "data" / "corpus"
    static_dir: Path = PACKAGE_DIR / "web" / "static"
    model_cache_dir: Path | None = None  # None: let huggingface_hub choose

    # -- Earth Engine -------------------------------------------------------
    ee_project_id: str = DEFAULT_EE_PROJECT_ID
    ee_enabled: bool = True  # 0 gives the null_adapter run of SPEC section 10
    ee_verify_on_init: bool = True  # one cheap round trip so ee_ready() is not a guess
    ee_request_timeout_s: float = 60.0  # per round trip; 0 disables the deadline
    ee_max_workers: int = 4
    ee_neighbourhood_m: float = 300.0  # CropSuite masks single pixels (SPEC 3.3)
    # The buffer that turns the confirmed point into the polygon EE is asked
    # about; see DEFAULT_FIELD_RADIUS_M for where 15 m comes from. A session may
    # override it per field (``SlotBag.field_radius_m``); this is what the app
    # uses when the farmer has not chosen one, so that /api/geo/field can state
    # the exact polygon instead of reporting no radius at all.
    field_radius_m: float = DEFAULT_FIELD_RADIUS_M

    # -- models -------------------------------------------------------------
    embed_model_id: str = DEFAULT_EMBED_MODEL_ID
    embed_onnx_file: str = DEFAULT_EMBED_ONNX_FILE
    embed_tokenizer_file: str = DEFAULT_EMBED_TOKENIZER_FILE
    embed_dim: int = DEFAULT_EMBED_DIM
    embed_max_tokens: int = 256
    nli_model_id: str = DEFAULT_NLI_MODEL_ID
    nli_enabled: bool = False
    nli_entailment_index: int = 0  # measured, not assumed (SPEC 5.1)
    allow_model_download: bool = True

    # -- confidence floors --------------------------------------------------
    # Tier 1 rules: a hit needs both of these to short-circuit (SPEC 5.1).
    rule_score_floor: float = 3.0
    rule_margin_floor: float = 2.0
    # Tier 2 centroid cosine: below the floor, route to clarify rather than guess.
    intent_confidence_floor: float = 0.40
    intent_margin_floor: float = 0.05
    # Tier 3 NLI tie-breaker, only consulted when tier 2 is inside the margin.
    nli_confidence_floor: float = 0.50
    # Slot extraction (rapidfuzz scores, 0-100).
    crop_match_floor: float = 88.0
    place_match_floor: float = 90.0
    place_ambiguous_margin: float = 5.0
    # RAG: below this cosine, say nothing was found and offer the questionnaire.
    rag_similarity_floor: float = 0.30
    rag_top_k: int = 4

    # -- display / safety ---------------------------------------------------
    risk_cap: int = 5  # SPEC 4.3: ranked, deduplicated by issue_id, capped at 5
    default_stale_after_days: int = 45  # used only where a source defines none
    require_confirmation_before_ee: bool = True  # SPEC 4.4

    # -- caches (seconds) ---------------------------------------------------
    ee_cache_ttl_s: int = 900
    ee_cache_max_entries: int = 512
    vocab_cache_ttl_s: int = 3600
    session_ttl_s: int = 86400

    # -- server -------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"

    def __post_init__(self) -> None:
        if self.risk_cap < 1:
            raise ConfigError(f"CROPUP_RISK_CAP must be at least 1, got {self.risk_cap}")
        if not 0.0 <= self.intent_confidence_floor <= 1.0:
            raise ConfigError("CROPUP_INTENT_CONFIDENCE_FLOOR must be between 0 and 1")
        if not 0.0 <= self.rag_similarity_floor <= 1.0:
            raise ConfigError("CROPUP_RAG_SIMILARITY_FLOOR must be between 0 and 1")
        if self.embed_dim < 1:
            raise ConfigError("CROPUP_EMBED_DIM must be positive")
        if self.default_stale_after_days < 1:
            raise ConfigError("CROPUP_STALE_AFTER_DAYS must be positive")
        # A zero or negative radius is a degenerate polygon: EE would be asked
        # about nothing at all and /api/geo/field would show the farmer a point
        # they cannot see. There is no sane fallback, so it is an error.
        if not math.isfinite(self.field_radius_m) or self.field_radius_m <= 0.0:
            raise ConfigError(
                "CROPUP_FIELD_RADIUS_M must be a positive number of metres, "
                f"got {self.field_radius_m!r}"
            )
        # An upper bound too, and for the same reason the per-field radius has
        # one: this value goes straight into ee.Geometry.Point(...).buffer(), so
        # a mistyped deployment variable would quietly measure a whole district
        # and report it as the farmer's field. Caught at startup rather than at
        # the Earth Engine door, so a bad deploy fails loudly and immediately.
        # Kept equal to dialog.slots.MAX_FIELD_RADIUS_M, which config cannot
        # import without a cycle; tests/test_field_radius.py binds the two.
        if self.field_radius_m > MAX_FIELD_RADIUS_M:
            raise ConfigError(
                f"CROPUP_FIELD_RADIUS_M must be at most {MAX_FIELD_RADIUS_M:g} metres "
                f"(a farmer's field, not a district), got {self.field_radius_m!r}"
            )
        if not math.isfinite(self.ee_request_timeout_s) or self.ee_request_timeout_s < 0.0:
            raise ConfigError(
                "CROPUP_EE_REQUEST_TIMEOUT_S must be a non-negative number of seconds "
                f"(0 disables the deadline), got {self.ee_request_timeout_s!r}"
            )
        # uvicorn only accepts the lower-case spelling, so normalise rather than
        # hand it a name it will reject three frames later.
        level = str(self.log_level).strip().lower()
        if level not in LOG_LEVELS:
            known = ", ".join(LOG_LEVELS)
            raise ConfigError(f"CROPUP_LOG_LEVEL={self.log_level!r} is not one of: {known}")
        object.__setattr__(self, "log_level", level)

    @property
    def log_level_number(self) -> int:
        """``log_level`` as a :mod:`logging` constant, for ``setLevel``."""
        return LOG_LEVELS[self.log_level]

    # -- committed data artifacts ------------------------------------------

    @property
    def ee_registry_path(self) -> Path:
        return self.data_dir / "ee_registry.json"

    @property
    def crops_path(self) -> Path:
        return self.data_dir / "crops.json"

    @property
    def gazetteer_path(self) -> Path:
        return self.data_dir / "gazetteer.json"

    @property
    def disease_library_path(self) -> Path:
        return self.data_dir / "disease_library.csv"

    def data_files(self) -> dict[str, Path]:
        """Name -> path for every committed artifact the app needs."""
        return {
            "ee_registry": self.ee_registry_path,
            "crops": self.crops_path,
            "gazetteer": self.gazetteer_path,
            "disease_library": self.disease_library_path,
        }

    def missing_data_files(self) -> dict[str, Path]:
        return {name: path for name, path in self.data_files().items() if not path.is_file()}

    def require_data_file(self, name: str) -> Path:
        """Path to a committed artifact, or raise. Never returns a path that
        does not exist, so callers cannot open() their way to a silent empty."""
        try:
            path = self.data_files()[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.data_files()))
            raise ConfigError(f"unknown data file {name!r} (known: {known})") from exc
        if not path.is_file():
            raise DataFileError(path, "file not found")
        return path

    # -- construction -------------------------------------------------------

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = _env_path("CROPUP_DATA_DIR", PACKAGE_DIR / "data")
        model_cache_raw = _raw("CROPUP_MODEL_CACHE_DIR")
        return cls(
            package_dir=PACKAGE_DIR,
            data_dir=data_dir,
            corpus_dir=_env_path("CROPUP_CORPUS_DIR", data_dir / "corpus"),
            static_dir=_env_path("CROPUP_STATIC_DIR", PACKAGE_DIR / "web" / "static"),
            model_cache_dir=Path(model_cache_raw).expanduser().resolve() if model_cache_raw else None,
            ee_project_id=_env_str("CROPUP_EE_PROJECT_ID", DEFAULT_EE_PROJECT_ID),
            ee_enabled=_env_bool("CROPUP_EE_ENABLED", True),
            ee_verify_on_init=_env_bool("CROPUP_EE_VERIFY_ON_INIT", True),
            ee_request_timeout_s=_env_float("CROPUP_EE_REQUEST_TIMEOUT_S", 60.0),
            ee_max_workers=_env_int("CROPUP_EE_MAX_WORKERS", 4),
            ee_neighbourhood_m=_env_float("CROPUP_EE_NEIGHBOURHOOD_M", 300.0),
            field_radius_m=_env_float("CROPUP_FIELD_RADIUS_M", DEFAULT_FIELD_RADIUS_M),
            embed_model_id=_env_str("CROPUP_EMBED_MODEL_ID", DEFAULT_EMBED_MODEL_ID),
            embed_onnx_file=_env_str("CROPUP_EMBED_ONNX_FILE", DEFAULT_EMBED_ONNX_FILE),
            embed_tokenizer_file=_env_str("CROPUP_EMBED_TOKENIZER_FILE", DEFAULT_EMBED_TOKENIZER_FILE),
            embed_dim=_env_int("CROPUP_EMBED_DIM", DEFAULT_EMBED_DIM),
            embed_max_tokens=_env_int("CROPUP_EMBED_MAX_TOKENS", 256),
            nli_model_id=_env_str("CROPUP_NLI_MODEL_ID", DEFAULT_NLI_MODEL_ID),
            nli_enabled=_env_bool("CROPUP_NLI_ENABLED", False),
            nli_entailment_index=_env_int("CROPUP_NLI_ENTAILMENT_INDEX", 0),
            allow_model_download=_env_bool("CROPUP_ALLOW_MODEL_DOWNLOAD", True),
            rule_score_floor=_env_float("CROPUP_RULE_SCORE_FLOOR", 3.0),
            rule_margin_floor=_env_float("CROPUP_RULE_MARGIN_FLOOR", 2.0),
            intent_confidence_floor=_env_float("CROPUP_INTENT_CONFIDENCE_FLOOR", 0.40),
            intent_margin_floor=_env_float("CROPUP_INTENT_MARGIN_FLOOR", 0.05),
            nli_confidence_floor=_env_float("CROPUP_NLI_CONFIDENCE_FLOOR", 0.50),
            crop_match_floor=_env_float("CROPUP_CROP_MATCH_FLOOR", 88.0),
            place_match_floor=_env_float("CROPUP_PLACE_MATCH_FLOOR", 90.0),
            place_ambiguous_margin=_env_float("CROPUP_PLACE_AMBIGUOUS_MARGIN", 5.0),
            rag_similarity_floor=_env_float("CROPUP_RAG_SIMILARITY_FLOOR", 0.30),
            rag_top_k=_env_int("CROPUP_RAG_TOP_K", 4),
            risk_cap=_env_int("CROPUP_RISK_CAP", 5),
            default_stale_after_days=_env_int("CROPUP_STALE_AFTER_DAYS", 45),
            require_confirmation_before_ee=_env_bool("CROPUP_REQUIRE_CONFIRMATION", True),
            ee_cache_ttl_s=_env_int("CROPUP_EE_CACHE_TTL_S", 900),
            ee_cache_max_entries=_env_int("CROPUP_EE_CACHE_MAX_ENTRIES", 512),
            vocab_cache_ttl_s=_env_int("CROPUP_VOCAB_CACHE_TTL_S", 3600),
            session_ttl_s=_env_int("CROPUP_SESSION_TTL_S", 86400),
            host=_env_str("CROPUP_HOST", "127.0.0.1"),
            port=_env_int("CROPUP_PORT", 8000),
            log_level=_env_str("CROPUP_LOG_LEVEL", "info"),
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe view, for /api/health. Contains no secrets by design."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = str(value) if isinstance(value, Path) else value
        out["data_files"] = {name: str(path) for name, path in self.data_files().items()}
        return out


_settings: Settings | None = None


def get_settings() -> Settings:
    """The process-wide Settings, built once from the environment."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reload_settings() -> Settings:
    """Re-read the environment. For tests and for tools/ CLIs."""
    global _settings
    _settings = Settings.from_env()
    return _settings
