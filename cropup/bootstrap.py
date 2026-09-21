"""Earth Engine bootstrap, request deadlines, logging and startup assertions.

Import this before anything touches Earth Engine::

    from cropup import bootstrap
    ee = bootstrap.require_ee()   # raises EarthEngineUnavailable if degraded

Three properties this module has to keep:

* it owns the single ``ee.Initialize()`` call, so no other module calls it;
* it is idempotent -- repeated calls are free and return the same verdict;
* it never raises at import time. If Earth Engine is unreachable the app boots
  degraded, ``/api/health`` says which assertion failed, and the RAG side of the
  product (72% of real traffic, SPEC 1.2) keeps working.

``ee`` is imported lazily inside the functions, so ``import cropup.bootstrap``
alone does not pull the Earth Engine client into the process.

It also owns the two process-wide policies that every entrypoint needs before
anything else happens, and that only make sense next to the code that owns the
round trip:

* **Deadlines.** ``ee.getInfo()`` is a blocking HTTPS call with no client-side
  timeout, so one wedged source hangs the whole turn -- the opposite of the
  latency budget in SPEC 3.4. :func:`call_with_deadline` bounds one round trip
  and :func:`call_or_missing` turns a blown budget into
  ``Missing(source_failed)`` instead of an exception that kills the turn.
  :func:`get_info` is the one-line form for ``geo/``::

      stats = bootstrap.get_info(image.reduceRegion(...), "soil_ph")
* **Logging.** Opt-in, off until :func:`configure_logging` is called, scoped to
  the ``cropup`` logger, and filtered so the SPEC 7 payloads (farmer message
  text, ProfileIds, raw pings) cannot reach a handler. The process that serves
  the app logs from two places, so there are two filters:
  :class:`PrivacyFilter` on the ``cropup`` logger, and :class:`RequestLogFilter`
  on the server's own loggers (``uvicorn``, ``uvicorn.access``,
  ``uvicorn.error``), where a farmer's words arrive inside a request URL.
  :func:`install_privacy_filters` attaches the second set and
  :func:`configure_logging` calls it, so no entrypoint has to remember.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import os
import secrets
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .config import Settings, get_settings
from .errors import ConfigError, EarthEngineUnavailable
from .evidence import Missing, MissingReason

__all__ = [
    "EEState",
    "Assertion",
    "Deadline",
    "PrivacyFilter",
    "RequestLogFilter",
    "LiteralMessage",
    "literal",
    "LOG_NAMESPACE",
    "REQUEST_LOG_NAMESPACES",
    "PRIVATE_FIELD_NAMES",
    "REDACTED",
    "REDACTED_MESSAGE",
    "REDACTED_QUERY",
    "initialize",
    "ee_ready",
    "ee_status",
    "ee_module",
    "require_ee",
    "request_timeout_s",
    "call_with_deadline",
    "call_or_missing",
    "get_info",
    "timeout_missing",
    "deadline_status",
    "configure_logging",
    "install_privacy_filters",
    "get_logger",
    "logging_status",
    "fingerprint",
    "count_records",
    "startup_assertions",
    "health_report",
    "reset",
]

# Counts SPEC records for the committed artifacts; a mismatch is reported, not fatal.
EXPECTED_COUNTS = {
    "ee_registry": 92,
    "crops": 134,
    "gazetteer": 1073,
    "disease_library": 384,
}

_lock = threading.Lock()


@dataclass(frozen=True)
class EEState:
    """What we know about Earth Engine right now. ``ready`` is measured, not hoped."""

    attempted: bool = False
    ready: bool = False
    verified: bool = False  # a real round trip returned
    project_id: str | None = None
    error: str | None = None
    elapsed_s: float | None = None
    attempted_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "ready": self.ready,
            "verified": self.verified,
            "project_id": self.project_id,
            "error": self.error,
            "elapsed_s": round(self.elapsed_s, 3) if self.elapsed_s is not None else None,
            "attempted_at": self.attempted_at,
        }


@dataclass(frozen=True)
class Assertion:
    """One startup check, as reported by /api/health."""

    name: str
    ok: bool
    detail: str
    required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail, "required": self.required}


_state = EEState()


def initialize(project_id: str | None = None, *, force: bool = False, verify: bool | None = None) -> bool:
    """Initialise Earth Engine once. Returns True when it is usable.

    Never raises: a failure is recorded in :func:`ee_status` and the caller
    decides whether to degrade or to raise :class:`EarthEngineUnavailable`.

    The whole boot -- the auth handshake inside ``ee.Initialize()`` and the
    verification round trip together -- shares one
    ``CROPUP_EE_REQUEST_TIMEOUT_S`` budget, so a wedged handshake cannot hold
    ``_lock`` (and therefore ``/api/health``) open forever. What that bound is
    and is not, is in the comment on the call itself.
    """
    global _state
    settings = get_settings()
    project = project_id or settings.ee_project_id
    if verify is None:
        verify = settings.ee_verify_on_init

    with _lock:
        if _state.attempted and not force:
            return _state.ready

        # The EE client library and the vendored reference code both read this.
        os.environ["EE_PROJECT_ID"] = project
        started = time.monotonic()
        now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

        if not settings.ee_enabled:
            _state = EEState(
                attempted=True,
                ready=False,
                project_id=project,
                error="disabled by CROPUP_EE_ENABLED=0 (null-adapter run)",
                elapsed_s=0.0,
                attempted_at=now,
            )
            return False

        try:
            import ee  # noqa: PLC0415 -- lazy on purpose: importing cropup must stay light

            # One budget for the whole boot, not one per leg. Both legs below
            # are blocking HTTPS calls with no client-side timeout, and both run
            # under ``_lock`` on the /api/health path: an unbounded one hangs
            # every health check *and* holds the lock while it does. So the
            # handshake and the verification share a single
            # ``CROPUP_EE_REQUEST_TIMEOUT_S`` and the second gets what the first
            # left. A blown budget raises TimeoutError, which the except below
            # records as the reason Earth Engine is not ready.
            #
            # Honestly, though: a deadline here *abandons*, it does not cancel
            # (see the Deadline section below). The worker thread keeps running
            # the handshake, so after a timeout
            #
            # * ``ee`` may finish initialising itself minutes later, behind a
            #   recorded verdict of ready=False. That verdict is not revised on
            #   its own; ``require_ee()`` gates on it and not on the ee module's
            #   own state, and ``initialize(force=True)`` is what re-attempts;
            # * a second ``ee.Initialize`` may then run concurrently with the
            #   abandoned one. It is the same call with the same project, which
            #   is why re-attempting is safe to offer at all;
            # * the abandoned thread is counted in ``deadline_status()``
            #   ["abandoned_threads"], so this is visible rather than silent.
            #
            # The ``import ee`` above stays outside the budget deliberately: it
            # reads the disk, not the network, and a timeout inside an import
            # would leave a half-initialised module in ``sys.modules``.
            boot = Deadline.for_request(label="earth engine boot", settings=settings)
            call_with_deadline(
                lambda: ee.Initialize(project=project),
                deadline=boot,
                label="ee.Initialize(project=...)",
            )
            verified = False
            if verify:
                # One trivial round trip. Without it "initialised" only means
                # "credentials parsed", which is not the same as "usable".
                answer = call_with_deadline(
                    lambda: ee.Number(1).add(1).getInfo(),
                    deadline=boot,
                    label="earth engine verification (1 + 1)",
                    settings=settings,
                )
                if answer != 2:
                    raise RuntimeError(f"Earth Engine returned {answer!r} for 1 + 1, expected 2")
                verified = True
            _state = EEState(
                attempted=True,
                ready=True,
                verified=verified,
                project_id=project,
                error=None,
                elapsed_s=time.monotonic() - started,
                attempted_at=now,
            )
        except Exception as exc:  # degraded boot is a supported state
            _state = EEState(
                attempted=True,
                ready=False,
                verified=False,
                project_id=project,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_s=time.monotonic() - started,
                attempted_at=now,
            )
        return _state.ready


def ee_ready() -> bool:
    """True when Earth Engine is initialised and usable.

    Initialises on first call, so callers do not have to sequence the bootstrap
    themselves.
    """
    if not _state.attempted:
        initialize()
    return _state.ready


def ee_status() -> dict[str, Any]:
    """The current Earth Engine verdict, JSON-safe, for /api/health."""
    return _state.as_dict()


def ee_module() -> Any | None:
    """The initialised ``ee`` module, or None when unavailable."""
    if not ee_ready():
        return None
    import ee

    return ee


def require_ee() -> Any:
    """The initialised ``ee`` module, or raise. Every geo/ call starts here."""
    module = ee_module()
    if module is None:
        raise EarthEngineUnavailable(_state.error or "Earth Engine has not been initialised")
    return module


# ---------------------------------------------------------------------------
# Request deadlines (SPEC 3.4)
# ---------------------------------------------------------------------------
#
# The Earth Engine client sends a blocking HTTPS request and waits. There is no
# supported per-call timeout argument, and the vendored backend had none either
# -- which is why one masked, retrying or wedged asset could hold a turn open
# indefinitely while the farmer watched an empty panel. The only way to bound a
# call we cannot cancel is to stop *waiting* for it: the work runs on a worker
# thread and the caller gives up when the budget is gone.
#
# What that buys and what it does not:
#
# * it bounds the *turn*, which is what SPEC 3.4 is about;
# * it does NOT cancel the request. The worker thread is a daemon, so it keeps
#   running until Earth Engine answers or the process exits, and its result is
#   discarded. Abandoned threads are counted in :func:`deadline_status` rather
#   than hidden, because a rising count is the honest signal that the budget is
#   too small or a source is sick.
#
# A budget of 0 (``CROPUP_EE_REQUEST_TIMEOUT_S=0``) disables the deadline and is
# the escape hatch for a debugging session; it is not a default anyone should
# ship.

_timeout_lock = threading.Lock()
_timeouts_seen = 0
_threads_abandoned = 0


@dataclass(frozen=True)
class Deadline:
    """A wall-clock budget, optionally shared by several calls.

    One ``Deadline`` per fan-out gives the whole fan-out a single budget: each
    leg gets what is left rather than a fresh 60 s, so six legs cannot take six
    minutes between them.
    """

    budget_s: float
    label: str = ""
    started_at: float = field(default_factory=time.monotonic, compare=False)

    @classmethod
    def for_request(
        cls,
        timeout_s: float | None = None,
        *,
        label: str = "",
        settings: Settings | None = None,
    ) -> "Deadline":
        """A budget from ``CROPUP_EE_REQUEST_TIMEOUT_S``, or from ``timeout_s``."""
        return cls(budget_s=request_timeout_s(timeout_s, settings), label=label)

    @property
    def bounded(self) -> bool:
        """False when this deadline never expires (budget 0)."""
        return self.budget_s > 0.0

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining_s(self) -> float | None:
        """Seconds left, or ``None`` when unbounded. ``0.0`` means expired."""
        if not self.bounded:
            return None
        return max(0.0, self.budget_s - self.elapsed_s)

    @property
    def expired(self) -> bool:
        return self.bounded and self.remaining_s == 0.0

    def describe(self) -> str:
        if not self.bounded:
            return "no deadline"
        return f"{self.elapsed_s:.1f}s of {self.budget_s:g}s used"

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "budget_s": self.budget_s,
            "elapsed_s": round(self.elapsed_s, 3),
            "remaining_s": None if self.remaining_s is None else round(self.remaining_s, 3),
            "expired": self.expired,
        }


def request_timeout_s(override: float | None = None, settings: Settings | None = None) -> float:
    """The per-round-trip budget in seconds; ``0.0`` means no deadline."""
    if override is None:
        return (settings or get_settings()).ee_request_timeout_s
    value = float(override)
    if not math.isfinite(value) or value < 0.0:
        raise ConfigError(f"timeout_s must be a non-negative number of seconds, got {override!r}")
    return value


def call_with_deadline(
    fn: Callable[[], Any],
    *,
    timeout_s: float | None = None,
    deadline: Deadline | None = None,
    label: str | None = None,
    settings: Settings | None = None,
) -> Any:
    """Run ``fn`` on a worker thread and stop waiting when the budget is gone.

    Returns whatever ``fn`` returns. Re-raises whatever ``fn`` raises, so a real
    Earth Engine error still reads as itself. Raises :class:`TimeoutError` when
    the budget expires first -- the call is abandoned, not cancelled.

    ``deadline`` wins over ``timeout_s``: pass a shared :class:`Deadline` to give
    a whole fan-out one budget, or ``timeout_s`` to bound this call alone. An
    already-expired deadline raises without starting the work.
    """
    global _timeouts_seen, _threads_abandoned

    what = label or getattr(fn, "__name__", "earth engine call")
    if deadline is not None:
        budget = deadline.remaining_s
        if deadline.expired:
            with _timeout_lock:
                _timeouts_seen += 1
            raise TimeoutError(f"{what}: {deadline.describe()}, no budget left before the call started")
    else:
        resolved = request_timeout_s(timeout_s, settings)
        budget = resolved if resolved > 0.0 else None

    box: dict[str, Any] = {}
    done = threading.Event()

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below, never swallowed
            box["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=run, name=f"cropup-deadline-{what}", daemon=True)
    worker.start()
    if not done.wait(budget):
        with _timeout_lock:
            _timeouts_seen += 1
            _threads_abandoned += 1
        raise TimeoutError(f"{what}: no answer within {budget:g}s; the request was abandoned, not cancelled")
    if "error" in box:
        raise box["error"]
    return box["value"]


def timeout_missing(
    quantity: str,
    *,
    timeout_s: float,
    chain_tried: Sequence[str] = (),
    label: str | None = None,
    detail: str | None = None,
) -> Missing:
    """The named absence a blown deadline produces.

    ``SOURCE_FAILED`` rather than ``MASKED``: a timeout says nothing at all
    about the pixel, and "we ran out of time" must not read as "there is no
    data here".
    """
    what = f"{label} " if label else ""
    return Missing(
        quantity,
        MissingReason.SOURCE_FAILED,
        tuple(chain_tried),
        detail or f"{what}did not answer within {timeout_s:g}s",
    )


def call_or_missing(
    quantity: str,
    fn: Callable[[], Any],
    *,
    chain_tried: Sequence[str] = (),
    timeout_s: float | None = None,
    deadline: Deadline | None = None,
    label: str | None = None,
    settings: Settings | None = None,
) -> Any:
    """:func:`call_with_deadline`, with the failure already named as a gap.

    Returns ``fn()`` on success, or a :class:`~cropup.evidence.Missing` carrying
    the reason -- a blown deadline or the exception the source raised. Nothing
    is defaulted and nothing is swallowed: the caller records the ``Missing`` in
    the turn ledger and the farmer is told which quantity was not measured.

    ``BaseException`` that is not an ``Exception`` (``KeyboardInterrupt``,
    ``SystemExit``) is left to propagate: those are not source failures.
    """
    budget = deadline.budget_s if deadline is not None else request_timeout_s(timeout_s, settings)
    try:
        return call_with_deadline(
            fn, timeout_s=timeout_s, deadline=deadline, label=label or quantity, settings=settings
        )
    except TimeoutError as exc:
        # str(exc) says which of the two timeouts it was -- the call ran out of
        # time, or a shared deadline had nothing left before it started.
        return timeout_missing(
            quantity, timeout_s=budget, chain_tried=chain_tried, label=label or quantity, detail=str(exc)
        )
    except Exception as exc:  # noqa: BLE001 - recorded as a named gap, not swallowed
        return Missing(
            quantity,
            MissingReason.SOURCE_FAILED,
            tuple(chain_tried),
            f"{type(exc).__name__}: {exc}",
        )


def get_info(
    ee_object: Any,
    quantity: str,
    *,
    chain_tried: Sequence[str] = (),
    timeout_s: float | None = None,
    deadline: Deadline | None = None,
    label: str | None = None,
    settings: Settings | None = None,
) -> Any:
    """``ee_object.getInfo()`` under the deadline, with the failure already named.

    This is the one-line adoption of SPEC 3.4 in ``geo/``::

        stats = image.reduceRegion(...).getInfo()                       # unbounded
        stats = bootstrap.get_info(image.reduceRegion(...), "soil_ph")  # bounded

    Returns exactly what ``getInfo()`` returned, or a
    :class:`~cropup.evidence.Missing` carrying the reason -- a blown budget, or
    the error the source raised -- so the gap reaches the ledger named instead of
    the turn hanging. Callers branch on ``isinstance(result, Missing)``, which is
    the branch they already have for an unreadable source.

    Inside a ``try`` that already turns a failure into a named gap,
    :func:`call_with_deadline` is the smaller change: ``x.getInfo()`` becomes
    ``call_with_deadline(x.getInfo, label="...")`` and the existing ``except``
    catches the :class:`TimeoutError` like any other source error.

    Pass one shared ``deadline`` through a fan-out to give the whole fan-out one
    budget; ``ee`` is never imported here, so this stays a thin wrapper over
    whatever object the caller already built.

    The attribute lookup happens *inside* the guarded call, so an object with no
    ``getInfo`` -- a :class:`~cropup.evidence.Missing` handed straight through
    from the previous link of a source chain is the realistic one -- comes back
    as a named ``Missing`` like any other source failure, instead of raising an
    ``AttributeError`` past every caller's ``isinstance(result, Missing)``
    branch.
    """
    return call_or_missing(
        quantity,
        lambda: ee_object.getInfo(),
        chain_tried=chain_tried,
        timeout_s=timeout_s,
        deadline=deadline,
        label=label or quantity,
        settings=settings,
    )


def deadline_status(settings: Settings | None = None) -> dict[str, Any]:
    """How often the deadline has fired this process, for /api/capabilities.

    ``budget_s`` is the budget of the ``settings`` being reported on: a report
    generated for one Settings must not quote another Settings' timeout. The
    counters are process-wide because the abandoned threads are.

    A non-zero ``abandoned_threads`` is a real cost -- those threads are still
    waiting on Earth Engine -- so it is reported rather than reset.
    """
    budget_s = (settings or get_settings()).ee_request_timeout_s
    with _timeout_lock:
        return {
            "budget_s": budget_s,
            "timeouts": _timeouts_seen,
            "abandoned_threads": _threads_abandoned,
        }


# ---------------------------------------------------------------------------
# Logging (SPEC 7)
# ---------------------------------------------------------------------------
#
# SPEC 7: "No raw ping, ProfileId or farmer message is ever committed or
# logged." The process that serves the app writes two kinds of log line, so
# that sentence is only true if both are covered:
#
# 1. what *cropup* logs -- :class:`PrivacyFilter` on the ``cropup`` logger;
# 2. what the *server* logs -- :class:`RequestLogFilter` on ``uvicorn``,
#    ``uvicorn.access`` and ``uvicorn.error``. ``python -m cropup.web`` runs
#    uvicorn, and uvicorn's access log writes the request target verbatim.
#    Farmer text reaches a URL on the documented run path: the autocomplete
#    endpoints take the farmer's typed prefix as ``?q=`` (static/app.js) and
#    ``/api/capabilities`` and ``/api/geo/field`` take the field's exact
#    ``?lat=&lon=``, which is a raw ping in the SPEC 7 sense. Covering only the
#    ``cropup`` logger left both in the access log in cleartext.
#
# So logging here is opt-in, namespaced, and filtered:
#
# * nothing is configured until :func:`configure_logging` is called, so importing
#   cropup installs no handlers and a library user keeps their own setup;
# * handlers are attached to the ``cropup`` logger only, never to the root, and
#   ``propagate`` is turned off so a farmer-shaped string cannot escape into
#   somebody else's root handler;
# * the server's loggers keep their own handlers -- they are uvicorn's, not ours
#   -- so :func:`install_privacy_filters` attaches the filter to those *loggers*
#   instead, where it runs before any handler of theirs can see the record;
# * :class:`PrivacyFilter` redacts the record attributes that carry SPEC 7
#   payloads, so ``extra={"message_text": text}`` is a redaction rather than a
#   leak. ``extra={"message": ...}`` is *not* the example to copy: stdlib
#   ``Logger.makeRecord`` raises ``KeyError: "Attempt to overwrite 'message' in
#   LogRecord"`` for the reserved names (``message``, ``asctime``, and every
#   attribute a record already has) before any filter runs. ``message`` stays in
#   :data:`PRIVATE_FIELD_NAMES` because a ``%(message)s``-style argument can
#   still arrive under that key;
# * :func:`fingerprint` exists so a turn can still be correlated in the log
#   without its text ever being written.
#
# The filter cannot police a format string somebody interpolated by hand before
# passing arguments with it. The rule for callers is therefore: log identifiers,
# counts, quantities, sources and reasons -- never the farmer's words. In
# practice that is three habits:
#
# 1. numbers may go in positionally; anything that is a string goes in
#    **named** -- ``extra={...}`` or a dict-style argument,
#    ``log.info("%(asset)s failed", {"asset": asset_id})`` -- because a
#    positional string has no name for the filter to check and is therefore
#    redacted on the way out;
# 2. a message with **no arguments at all** is data, not a template. That is the
#    shape ``log.info(text)`` and ``log.info(f"asked {text}")`` arrive in, so it
#    is redacted whole unless the caller marks it with :func:`literal`, which is
#    how a developer says "these are my words";
# 3. ``fingerprint()`` is the escape hatch when "was this the same message" has
#    to be answerable.

LOG_NAMESPACE = "cropup"

#: The loggers of the HTTP server this app is started with. They are not ours,
#: they are configured by uvicorn after we are imported, and they log the
#: request URL -- so they get :class:`RequestLogFilter`. A logger that is not
#: named here is not covered, which is why this is a list of the loggers that
#: actually call ``log()``: a record logged to a *descendant* reaches these
#: loggers' handlers by propagation without passing their filters.
REQUEST_LOG_NAMESPACES = ("uvicorn", "uvicorn.error", "uvicorn.access")

REDACTED = "<redacted>"

#: What the query string of a logged URL becomes. Named separately from
#: :data:`REDACTED` so an access log still shows that there *was* a query.
REDACTED_QUERY = "<redacted-query>"

#: What an unmarked message becomes. It names the mechanism, because the
#: developer whose ``log.info("corpus loaded")`` just vanished needs to know
#: why, and the farmer whose sentence it might have been is no worse off.
REDACTED_MESSAGE = f"{REDACTED} (unmarked log message; mark developer text with bootstrap.literal)"

#: Record attributes whose contents are personal data under SPEC 7. Matched
#: case-insensitively, so ``ProfileId`` and ``profile_id`` are both caught.
PRIVATE_FIELD_NAMES = frozenset(
    {
        "message",
        "message_text",
        "text",
        "utterance",
        "question",
        "query",
        "answer",
        "profile_id",
        "profileid",
        "user_id",
        "userid",
        "phone",
        "ping",
        "pings",
        "gps",
        "lat",
        "lon",
        "latitude",
        "longitude",
        "coordinates",
    }
)

# Per-process salt: a fingerprint identifies repeats inside one run and is
# worthless outside it, so a log file cannot be joined against anything.
_FINGERPRINT_SALT = secrets.token_bytes(16)

_logging_lock = threading.Lock()
_logging_configured = False


def _safe_arg(value: Any) -> Any:
    """One positional ``%`` argument, or :data:`REDACTED` if it could be words.

    ``int``, ``float``, ``bool`` and ``None`` are kept: they cannot carry a
    sentence, and redacting them would break ``%d``/``%.2f`` formatting for the
    counts and quantities callers are asked to log. Everything else -- ``str``,
    ``bytes``, a tuple of coordinates, any object whose ``repr`` might hold a
    message -- is replaced, because the record carries no name to check it
    against and "probably not a farmer's words" is not a privacy guarantee.
    """
    return value if value is None or isinstance(value, (int, float)) else REDACTED


def _safe_request_arg(value: Any) -> Any:
    """One positional ``%`` argument of a *server* log record.

    The shape that matters is uvicorn's access record,
    ``'%s - "%s %s HTTP/%s" %d' % (client_addr, method, target, version,
    status)``: the farmer's words are in ``target``, after the ``?``. So the
    query string goes and the path stays, which keeps the line worth reading --
    ``GET /api/vocab/places?<redacted-query> 200`` still says which endpoint was
    hit and how it answered.

    Keeping the path is a claim about this app, not a general one: no route in
    ``web/server.py`` takes farmer text in a path segment (paths are route
    literals plus server-generated session ids), and farmer text arrives in the
    query string or the request body. A string with whitespace in it is not a
    URL any caller of ours built, so it is redacted whole rather than guessed
    at. Non-strings are returned unchanged -- the status code is an ``int`` and
    uvicorn's ``AccessFormatter`` unpacks exactly five arguments and calls
    ``int()`` on the last, so neither the arity nor that type may change here.
    """
    if not isinstance(value, str):
        return value
    path, sep, _query = value.partition("?")
    if any(ch.isspace() for ch in path):
        return REDACTED
    return f"{path}?{REDACTED_QUERY}" if sep else path


#: Lines of a formatted traceback that are prose rather than an exception.
_TRACEBACK_PROSE = (
    "Traceback (most recent call last)",
    "Stack (most recent call last)",
    "During handling of the above exception",
    "The above exception was the direct cause",
)


def _redact_exception_text(record: logging.LogRecord) -> None:
    """Pre-format ``exc_info`` with every exception *message* redacted.

    A traceback is not in the record when a filter runs: ``exc_info`` holds the
    exception and the *handler's formatter* renders it later, so redacting
    record attributes does nothing about ``raise ValueError(farmer_text)``.
    What a filter can do is fill ``record.exc_text`` in advance --
    ``logging.Formatter.format`` appends that verbatim and skips formatting the
    exception itself when it is already set.

    The stack is kept in full: frame lines are file names, line numbers and
    *source* text, which is code, not data. Each exception's type is kept too.
    Only what follows ``SomeError: `` goes, because that is the part built from
    values. A multi-line exception message is dropped whole with it.
    """
    if not record.exc_info or record.exc_text:
        return
    try:
        chunks = traceback.format_exception(*record.exc_info)
    except Exception:  # noqa: BLE001 - a broken exc_info must not lose the record
        record.exc_text = f"{REDACTED} (exception text could not be formatted)"
        return
    out: list[str] = []
    for chunk in chunks:
        first = chunk.split("\n", 1)[0]
        if not first or first[0].isspace() or first.startswith(_TRACEBACK_PROSE):
            out.append(chunk)
            continue
        kind, sep, message = first.partition(": ")
        out.append(f"{kind}: {REDACTED}\n" if sep and message.strip() else chunk)
    record.exc_text = "".join(out).rstrip("\n")


def _redact_private_attrs(record: logging.LogRecord) -> None:
    """Replace every record attribute named in :data:`PRIVATE_FIELD_NAMES`.

    This is the ``extra={"message_text": text}`` shape, and it is the same in
    both filters: a name on the list carries personal data whoever logged it.
    """
    for name in list(record.__dict__):
        if name.lower() in PRIVATE_FIELD_NAMES:
            record.__dict__[name] = REDACTED


def _redact_private_keys(args: Mapping[Any, Any]) -> dict[Any, Any]:
    """Redact the private keys of a dict-style ``%`` argument."""
    return {
        key: (REDACTED if str(key).lower() in PRIVATE_FIELD_NAMES else value)
        for key, value in args.items()
    }


class LiteralMessage(str):
    """A log message the caller vouches for as *their* words, not a farmer's.

    Build one with :func:`literal`. It exists because nothing in a
    ``LogRecord`` distinguishes ``log.info("corpus loaded")`` from
    ``log.info(farmer_question)``: both arrive as a message with no arguments,
    and a filter that guesses between them guesses wrong on the farmer's turn.
    So the default is redaction and this is the one explicit exception, which
    an auditor can find with ``grep -rn 'literal('``.
    """

    __slots__ = ()


def literal(text: str) -> LiteralMessage:
    """Mark a log message as developer text so it survives :class:`PrivacyFilter`.

    ::

        log.info(bootstrap.literal("session opened"))
        log.warning(bootstrap.literal("%(asset)s is stale"), {"asset": asset_id})

    Only ever wrap a string spelled out in the source. Wrapping a value --
    ``literal(request.text)``, ``literal(f"asked {text}")`` -- is the SPEC 7
    violation this whole mechanism exists to prevent, and the wrapper is
    deliberately explicit so that doing it is a visible choice rather than an
    accident.
    """
    return LiteralMessage(text)


def _message_allowed(record: logging.LogRecord) -> bool:
    """May this record's message text reach a handler at all?

    A :class:`LiteralMessage` may: the caller has said in as many words that it
    is theirs. A plain ``str`` may when it carries ``%`` arguments, because the
    arguments make it a *template* -- the message is the shape and the data is
    in the arguments, where the rest of this filter can check it.

    Everything else is a value somebody logged, and a value is exactly what
    SPEC 7 is about: ``log.info(text)``, ``log.info(f"asked {text}")`` and
    ``log.info(some_object)`` all reach here indistinguishable from each other,
    so none of them is let through.
    """
    if isinstance(record.msg, LiteralMessage):
        return True
    return bool(record.args) and isinstance(record.msg, str)


class PrivacyFilter(logging.Filter):
    """Redacts SPEC 7 payloads out of a **cropup** record before a handler sees it.

    Scope first, because a privacy filter that is trusted beyond its scope is
    worse than none: :func:`configure_logging` installs this on the one handler
    it attaches to the ``cropup`` logger, and that is the only place it runs.
    It says nothing about the root logger, about a handler somebody else
    attached, or about the HTTP server's own loggers -- those are covered
    separately, and only because :class:`RequestLogFilter` is installed on them
    by name (:func:`install_privacy_filters`). :func:`logging_status` reports
    which of the two is in place, so /api/health answers this rather than this
    docstring being the answer.

    Within that scope, five shapes can carry a farmer's words, and all five are
    covered:

    * ``extra={...}`` under a name in :data:`PRIVATE_FIELD_NAMES`;
    * a dict-style ``%`` argument under such a key;
    * a **positional** ``%`` argument -- ``log.info("asked %s", text)``, which
      arrives with no name at all. Nothing in the record says whose words they
      are, so nothing non-numeric is let through (:func:`_safe_arg`); a source
      id or a reason is logged named instead, where the name can be checked;
    * the **message itself** -- ``log.info(text)``, or the f-string that spells
      the same thing, which is the shortest way anyone leaks a farmer's
      question and the one the three rules above never saw, because a message
      with no arguments has no name, key or position to check. It is treated as
      data rather than as a template (:func:`_message_allowed`) and replaced
      whole with :data:`REDACTED_MESSAGE`; developer text says so with
      :func:`literal`;
    * the **traceback** of ``log.exception(...)`` / ``exc_info=``, where
      ``raise ValueError(f"no crop in {text}")`` prints the farmer's sentence
      from the handler rather than from the record. The exception messages are
      redacted and the stack is kept (:func:`_redact_exception_text`).

    What it still cannot police: a format string somebody interpolated by hand
    before passing it (that arrives as a no-argument message, so it is redacted
    whole -- but as data, not because the filter understood it), and anything
    logged to a logger this filter is not on.

    Replacement is :data:`REDACTED` and the record is still emitted: "a message
    arrived" is useful, the message is not ours to keep. :func:`fingerprint` is
    how a string stays correlatable without being kept.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        _redact_private_attrs(record)
        _redact_exception_text(record)
        if not _message_allowed(record):
            record.msg = REDACTED_MESSAGE
            # The template is gone, so its arguments have nowhere to go; leaving
            # them would raise in ``record.getMessage()`` and lose the record.
            record.args = ()
            return True
        if isinstance(record.args, Mapping):
            record.args = _redact_private_keys(record.args)
        elif record.args:
            record.args = tuple(_safe_arg(arg) for arg in record.args)
        return True


class RequestLogFilter(logging.Filter):
    """Redacts SPEC 7 payloads out of the **HTTP server's** records.

    ``python -m cropup.web`` runs uvicorn, and uvicorn logs every request as
    ``'%s - "%s %s HTTP/%s" %d'`` with the request target -- query string and
    all -- as an argument. On the documented run path that target carries the
    farmer's typed autocomplete prefix (``/api/vocab/places?q=...``) and the
    field's exact coordinates (``/api/capabilities?lat=...&lon=...``), so
    without this the access log is a transcript of exactly what SPEC 7 forbids
    keeping.

    :class:`PrivacyFilter` is the wrong instrument here: its rule that an
    unargumented message is data assumes the caller is cropup code following
    the conventions above it, and would erase uvicorn's own constants
    ("Application startup complete."), which are the library's words and carry
    nothing of the farmer's. So this filter does less, on purpose:

    * record attributes named in :data:`PRIVATE_FIELD_NAMES` are redacted;
    * dict-style ``%`` arguments are redacted by key;
    * every **string** positional ``%`` argument loses its query string
      (:func:`_safe_request_arg`); arity and non-string arguments are preserved
      because uvicorn's ``AccessFormatter`` unpacks exactly five arguments and
      calls ``int()`` on the status code;
    * tracebacks logged by ``uvicorn.error`` -- "Exception in ASGI application"
      -- keep their stack and lose their exception messages
      (:func:`_redact_exception_text`), which is where a handler's
      ``ValueError(farmer_text)`` would otherwise surface.

    What it does **not** do, stated plainly: the message template itself is
    passed through, so a library that interpolates a value into its own message
    before logging it would not be caught; and the path of the URL is kept (see
    :func:`_safe_request_arg`). It is installed by name on
    :data:`REQUEST_LOG_NAMESPACES` and covers nothing else.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        _redact_private_attrs(record)
        _redact_exception_text(record)
        if isinstance(record.args, Mapping):
            record.args = _redact_private_keys(record.args)
        elif record.args:
            record.args = tuple(_safe_request_arg(arg) for arg in record.args)
        return True


def install_privacy_filters(namespaces: Sequence[str] = REQUEST_LOG_NAMESPACES) -> list[str]:
    """Attach :class:`RequestLogFilter` to the server's loggers. Idempotent.

    On the *loggers*, not on their handlers, for two reasons. A logger's filters
    run inside ``Logger.handle`` before any handler is called, so the redaction
    holds for every handler uvicorn installs -- including the ones it installs
    *later*, when ``uvicorn.run()`` applies its ``dictConfig``: that replaces a
    logger's handlers but leaves its filters alone (``logging.config``'s
    ``common_logger_config``, checked against this interpreter's stdlib), so
    calling this before or after ``uvicorn.run`` both work. And a handler is
    uvicorn's to replace, while a filter we put on the logger is not.

    Call it as early as the process can: :func:`configure_logging` calls it, and
    ``cropup.web.__main__`` calls it before ``uvicorn.run`` so that the window
    before the app's lifespan runs is covered too.

    Returns the logger names now carrying the filter, which is what
    :func:`logging_status` reports to /api/health.
    """
    installed: list[str] = []
    for name in namespaces:
        logger = logging.getLogger(name)
        if not any(isinstance(f, RequestLogFilter) for f in logger.filters):
            logger.addFilter(RequestLogFilter())
        installed.append(name)
    return installed


def fingerprint(value: object, *, length: int = 8) -> str:
    """A short, salted, non-reversible token for a private string.

    Two identical messages in one process fingerprint the same, so "the farmer
    asked that again" stays answerable without the text ever being written. The
    salt is new on every process, so the token cannot be correlated across runs
    or against any other file. It is a debugging aid, not a pseudonym to build
    on: nothing should key storage on it.
    """
    digest = hashlib.sha256(_FINGERPRINT_SALT + str(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest[:length]}/{len(str(value))}ch"


def configure_logging(
    settings: Settings | None = None,
    *,
    stream: Any = None,
    force: bool = False,
) -> logging.Logger:
    """Attach one stderr handler to the ``cropup`` logger. Opt-in, idempotent.

    Honours ``CROPUP_LOG_LEVEL``. Touches neither the root logger nor anybody
    else's, so importing cropup from another application changes nothing until
    this is called. Calling it again only updates the level unless ``force``.

    It does add one thing outside the ``cropup`` namespace: the SPEC 7 filter on
    the server's loggers (:func:`install_privacy_filters`). Adding a filter is
    not configuring somebody's logging -- it removes nothing, silences nothing
    and changes no level -- and a redaction that has to be remembered separately
    by each entrypoint is the one that gets forgotten.
    """
    global _logging_configured
    settings = settings or get_settings()
    # The server logs too, and its loggers are not ours to configure -- only to
    # filter. Done outside the lock below: this takes no lock of its own and
    # ``_logging_lock`` is not reentrant.
    install_privacy_filters()
    logger = logging.getLogger(LOG_NAMESPACE)
    with _logging_lock:
        if _logging_configured and not force:
            logger.setLevel(settings.log_level_number)
            return logger
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s %(message)s", "%Y-%m-%dT%H:%M:%S")
        )
        handler.addFilter(PrivacyFilter())
        logger.addHandler(handler)
        logger.setLevel(settings.log_level_number)
        # Do not hand cropup records to a root handler somebody else installed:
        # SPEC 7 is only enforceable on handlers this module controls.
        logger.propagate = False
        _logging_configured = True
    return logger


def get_logger(name: str = "") -> logging.Logger:
    """A logger under the ``cropup`` namespace. Silent until configured."""
    if not name or name == LOG_NAMESPACE:
        return logging.getLogger(LOG_NAMESPACE)
    if name.startswith(LOG_NAMESPACE + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOG_NAMESPACE}.{name}")


def logging_status() -> dict[str, Any]:
    """What logging is doing, for /api/health. Contains no log content."""
    logger = logging.getLogger(LOG_NAMESPACE)
    return {
        "configured": _logging_configured,
        "namespace": LOG_NAMESPACE,
        "level": logging.getLevelName(logger.level).lower(),
        "handlers": len(logger.handlers),
        "propagates": logger.propagate,
        "privacy_filter": any(
            isinstance(f, PrivacyFilter) for h in logger.handlers for f in h.filters
        ),
        # SPEC 7 covers the process, not one logger: the access log is where a
        # farmer's words leave through a URL, so whether it is filtered is a
        # health fact and not an implementation detail.
        "request_log_filters": {
            name: any(
                isinstance(f, RequestLogFilter) for f in logging.getLogger(name).filters
            )
            for name in REQUEST_LOG_NAMESPACES
        },
    }


def count_records(name: str, path: Path) -> int:
    """Rows/entries in a committed artifact, counted without pandas.

    Public because ``/api/capabilities`` reports the same counts as
    ``/api/health``: two implementations would eventually disagree about how
    many crops are shipped, and the honesty endpoint cannot be the one that is
    wrong. Raises whatever the file does; callers decide what a bad file means.
    """
    if name == "disease_library":
        import csv

        with path.open(newline="", encoding="utf-8") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if name == "ee_registry":
        return len(payload["datasets"])
    if name == "crops":
        return len(payload["crops"])
    if name == "gazetteer":
        return len(payload["places"])
    return len(payload)


def startup_assertions(settings: Settings | None = None) -> list[Assertion]:
    """Everything /api/health checks, in report order.

    A False on a ``required`` assertion means the app is broken; a False on an
    optional one means it is degraded but honest about it.
    """
    settings = settings or get_settings()
    out: list[Assertion] = []

    for name, path in settings.data_files().items():
        if not path.is_file():
            out.append(Assertion(f"data:{name}", False, f"missing: {path}", required=True))
            continue
        try:
            count = count_records(name, path)
        except Exception as exc:
            out.append(Assertion(f"data:{name}", False, f"unreadable: {type(exc).__name__}: {exc}", required=True))
            continue
        expected = EXPECTED_COUNTS.get(name)
        detail = f"{count} records"
        if expected is not None and count != expected:
            detail += f" (SPEC expects {expected})"
        out.append(Assertion(f"data:{name}", count > 0, detail, required=True))

    # SPEC 1.1: torch is broken against NumPy 2.2.6 here. If anything imported
    # it, the NLU path is one call away from a segfault and we want to know.
    torch_imported = "torch" in sys.modules
    out.append(
        Assertion(
            "no_torch_imported",
            not torch_imported,
            "torch is not in sys.modules" if not torch_imported else "torch has been imported; it is broken in this environment",
            required=True,
        )
    )

    corpus_ready = settings.corpus_dir.is_dir() and any(settings.corpus_dir.iterdir())
    out.append(
        Assertion(
            "rag:corpus",
            corpus_ready,
            f"{settings.corpus_dir}" + ("" if corpus_ready else " is empty or absent; RAG answers unavailable"),
            required=False,
        )
    )

    # An index built from an older corpus, or by a different encoder, retrieves
    # the wrong cards and cites them confidently -- the failure mode SPEC 6 is
    # written against. The probe is in capabilities.py (imported here rather
    # than at module scope: that module imports this one) and reads the sidecar
    # header only, so it costs no network, no numpy and no model load.
    try:
        from .capabilities import rag_index_state  # noqa: PLC0415 -- see above

        index_state = rag_index_state(settings)
        out.append(
            Assertion("rag:index", index_state.ok, index_state.detail, required=False)
        )
    except Exception as exc:  # the honesty endpoint must never break /api/health
        out.append(
            Assertion("rag:index", False, f"could not be checked: {type(exc).__name__}: {exc}", required=False)
        )

    # SPEC 9 is a browser app served by this process. Without the static dir the
    # HTTP API still answers, so this is degraded rather than broken -- but it
    # has to be named, or "GET / returns 404" is discovered by the farmer.
    index_html = settings.static_dir / "index.html"
    static_ready = settings.static_dir.is_dir() and index_html.is_file()
    if static_ready:
        static_detail = f"{settings.static_dir}"
    elif settings.static_dir.is_dir():
        static_detail = f"{index_html} is missing; the API answers but GET / cannot"
    else:
        static_detail = f"{settings.static_dir} is absent; the API answers but GET / cannot"
    out.append(Assertion("web:static", static_ready, static_detail, required=False))

    out.append(
        Assertion(
            "ee:initialized",
            ee_ready(),
            _state.error or f"project {_state.project_id}" + (", verified" if _state.verified else ", not verified"),
            required=False,  # the app must boot and serve RAG without Earth Engine
        )
    )
    return out


def health_report(settings: Settings | None = None) -> dict[str, Any]:
    """The /api/health payload: assertions plus the Earth Engine verdict."""
    settings = settings or get_settings()
    checks = startup_assertions(settings)
    failed_required = [c.name for c in checks if c.required and not c.ok]
    degraded = [c.name for c in checks if not c.required and not c.ok]
    return {
        "status": "ok" if not failed_required else "broken",
        "degraded": degraded,
        "failed": failed_required,
        "checks": [c.as_dict() for c in checks],
        "earth_engine": ee_status(),
        "deadlines": deadline_status(settings),
        "logging": logging_status(),
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


def reset() -> None:
    """Forget the Earth Engine verdict and the deadline counters.

    For tests and for the null-adapter run of SPEC 10. Logging is deliberately
    left alone: tearing a configured handler down mid-process would lose the
    record of why the reset happened.
    """
    global _state, _timeouts_seen, _threads_abandoned
    with _lock:
        _state = EEState()
    with _timeout_lock:
        _timeouts_seen = 0
        _threads_abandoned = 0
