"""The FastAPI app: the module SPEC section 2 calls "the one that imports everything".

Fourteen endpoints (SPEC section 8) over the layers below, and four rules that
are the reason this file exists rather than being glue.

**1. SPEC section 4.4 is enforced here, not observed here.** A natural-language
turn cannot reach Earth Engine, and neither can a direct POST to ``/run``.
Neither is a convention: ``cropup.web.dispatch.run`` refuses anything but a
:class:`~cropup.web.dispatch.RunAuthorization`, and the only function that mints
one routes the decision through ``dialog.policy``. ``/message`` never calls it.
``/run`` does, and gets the same verdict the chat turn got, from the same pure
function.

**2. No number is formatted in this file.** Every farmer-facing sentence leaves
through ``render/templates.py`` -- including the dialog's own questions, via
:mod:`cropup.web.replies` -- so every number on screen arrives as a segment
carrying ``fact.provenance()``: instrument, date, resolution, chain position.
That is what SPEC section 9's hover is built from, and the server never invents
a shape for it.

**3. Errors are named, not 500'd.** ``errors.CropUpError`` has a branch per
failure mode and each maps to a status code here (:data:`ERROR_STATUS`). A
missing index is a 503 that says the index is missing; an unconfirmed field is a
409 that says which slots are unconfirmed. A 500 from this app means a bug.

**4. It must start and serve with Earth Engine completely unavailable.** That
is the normal case: ``bootstrap.initialize`` is attempted once on startup, on a
background thread so a dead endpoint cannot delay the first request, and every
handler reads ``bootstrap.ee_status()`` -- the recorded verdict -- rather than
``ee_ready()``, which would initialise.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Mapping

from fastapi import FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .. import __version__, bootstrap
from ..capabilities import capability_report
from ..config import Settings, get_settings
from ..dialog import policy as policy_mod
from ..dialog.slots import INTENT_SLOT, ORIGIN_USER, SlotBag
from ..errors import (
    ConfigError,
    ConfirmationRequired,
    CropUpError,
    DataFileError,
    EarthEngineUnavailable,
    ModelUnavailable,
    ProvenanceError,
    SourceUnavailable,
)
from ..nlu import classify as nlu_classify
from ..nlu import intents as intents_mod
from ..nlu import slots as nlu_slots
from . import dispatch, geofield, progress, replies
from .session import (
    CLIENT_ORIGINS,
    Session,
    SessionCapacityReached,
    SessionNotFound,
    SessionStore,
    initial_writes,
    resolve_origin,
    sequence_of,
)

__all__ = [
    "app",
    "create_app",
    "ERROR_STATUS",
    "MAX_BODY_BYTES",
    "BodySizeLimit",
    "LifespanRequired",
]


# ---------------------------------------------------------------------------
# request bounds
# ---------------------------------------------------------------------------
#
# Every field a client can grow is bounded, and the whole body is bounded on top
# of that. Without both, one POST of a megabyte of text costs seconds of NLU and
# comes back as a multi-megabyte response, because a turn echoes what it parsed.
# These are farmer-message sizes, not arbitrary round numbers: the longest of the
# 78 real questions in SPEC 1.2 is a couple of hundred characters.

#: The largest request body any endpoint accepts, before parsing. A restored bag
#: (four slots, each with a label, alternatives and a coordinate pair) is a few
#: kilobytes, so this is two orders of magnitude of headroom.
MAX_BODY_BYTES = 64 * 1024

#: One chat message or one ``/api/nlu/parse`` probe.
MAX_MESSAGE_CHARS = 2000

#: One slot value, a crop key, a place key or an intent name -- and every string
#: nested inside a free-form object, which is the part the validator used to
#: miss. ``{"slots": {"crop": {"x": "z" * 5000}}}`` is not a string, so the old
#: check skipped it, and ``initial_writes`` then stored ``str(...)`` of it: a
#: 5,014-character crop. The bound is on the text a value *becomes*, not on
#: whether it arrived already spelled as one. The longest entry in the committed
#: crop and gazetteer vocabularies is well under this.
MAX_VALUE_CHARS = 200

#: Items in a list-of-slot-names field (``clear``, ``unlock``, ``alternatives``).
MAX_LIST_ITEMS = 32

#: Keys in a free-form mapping (``slots``, a restored slot's ``detail``).
MAX_MAPPING_KEYS = 32


# ---------------------------------------------------------------------------
# error mapping (rule 3)
# ---------------------------------------------------------------------------

#: One status code per failure mode, most specific class first. A handler never
#: chooses a code inline: it raises the exception that says what happened.
ERROR_STATUS: tuple[tuple[type[BaseException], int], ...] = (
    (SessionNotFound, 404),
    (SessionCapacityReached, 503),  # every live session is somebody's conversation
    (ConfirmationRequired, 409),  # SPEC 4.4: the gate is shut, and that is not an error
    (EarthEngineUnavailable, 503),
    (ModelUnavailable, 503),
    (DataFileError, 503),
    (SourceUnavailable, 503),
    (ProvenanceError, 500),  # the firewall fired: a bug in a template, not bad input
    (ConfigError, 500),
    (TimeoutError, 504),
    (ValueError, 400),
    (CropUpError, 500),
)

_REMEDIES: dict[type[BaseException], str] = {
    SessionNotFound: "open a new session with POST /api/session",
    SessionCapacityReached: (
        "retry shortly; a session already open is unaffected, because this "
        "process refuses a new one rather than evicting a conversation in use"
    ),
    ConfirmationRequired: (
        "confirm the resolved location and crop with "
        "POST /api/session/{sid}/confirm, then POST /api/session/{sid}/run"
    ),
    EarthEngineUnavailable: (
        "GET /api/capabilities shows what is degraded; knowledge questions are "
        "still answered from the cited corpus"
    ),
    ModelUnavailable: "GET /api/capabilities names the missing model file",
    DataFileError: "GET /api/health names the missing artifact under cropup/data/",
    TimeoutError: (
        "the run outlived its budget and was abandoned, not cancelled; retry, or "
        "raise CROPUP_EE_REQUEST_TIMEOUT_S"
    ),
}


def _status_for(exc: BaseException) -> int:
    for kind, status in ERROR_STATUS:
        if isinstance(exc, kind):
            return status
    return 500


def _remedy_for(exc: BaseException) -> str | None:
    for kind, remedy in _REMEDIES.items():
        if isinstance(exc, kind):
            return remedy
    return None


def _error_detail(exc: BaseException) -> dict[str, Any]:
    detail: dict[str, Any] = {}
    for attribute in (
        "missing_confirmations",
        "quantity",
        "chain_tried",
        "model_id",
        "path",
        "slot",
        "template",
        "reason",
        "live",
        "capacity",
        "retry_after_s",
        # SessionCapacityReached only: the idle time a conversation is actually
        # being protected for, and whether pressure has shortened it. A client
        # told to retry deserves to know the guarantee moved.
        "grace_s",
        "degraded",
    ):
        value = getattr(exc, attribute, None)
        if value is None:
            continue
        detail[attribute] = list(value) if isinstance(value, tuple) else value
    return detail


def _error_response(exc: BaseException) -> JSONResponse:
    status = _status_for(exc)
    payload: dict[str, Any] = {
        "error": {
            "type": type(exc).__name__,
            "status": status,
            "message": str(exc),
            "detail": _error_detail(exc),
            "remedy": _remedy_for(exc),
        },
        "ran_earth_engine": False,
    }
    if isinstance(exc, ProvenanceError):
        payload["error"]["note"] = (
            "the fabrication firewall refused to render an unmeasured value "
            "(SPEC 4.2); this is a template bug, and no defaulted number was shown"
        )
    headers: dict[str, str] | None = None
    if isinstance(exc, SessionCapacityReached):
        headers = {"Retry-After": str(max(1, int(exc.retry_after_s)))}
    # Every endpoint in this app can reach here, unauthenticated, and several of
    # these messages are built from a filesystem path: ``DataFileError`` puts one
    # in ``detail["path"]`` and another in its message. The two endpoints that
    # used to redact were not the leak; this function was.
    return JSONResponse(_redact_host_paths(payload), status_code=status, headers=headers)


#: How many rejected fields one 422 names, and how long each explanation may be.
#: A validation failure is a contract mismatch, so the client needs the *names*
#: it got wrong -- not its own body read back to it.
MAX_REPORTED_FIELDS = 8
MAX_REASON_CHARS = 200


def _validation_response(exc: Exception) -> JSONResponse:
    """A rejected body, answered in this app's envelope and without the body.

    FastAPI's own 422 is two contract breaks in one response. It is not the
    ``{"error": {...}, "ran_earth_engine": ...}`` shape every other failure in
    this app has -- ``errorMessage()`` in ``static/app.js`` carries a second
    code path purely to read it -- and each entry carries ``input``: the exact
    value that was rejected, echoed back at whatever size got through
    :class:`BodySizeLimit`. An unauthenticated POST of 64 kB of junk to
    ``/api/nlu/parse`` came back as 64 kB of junk, which is an amplifier
    pointed at anybody whose address can be spoofed.

    So: the field names and pydantic's own short reason, capped both ways, and
    nothing of the value.
    """
    raw = list(exc.errors()) if hasattr(exc, "errors") else []
    fields: list[dict[str, Any]] = []
    for error in raw[:MAX_REPORTED_FIELDS]:
        location = ".".join(str(part) for part in (error.get("loc") or ()))
        fields.append(
            {
                "field": location[:MAX_VALUE_CHARS] or "body",
                "kind": str(error.get("type") or "")[:64],
                # pydantic's message describes the *rule*, not the value
                # ("Extra inputs are not permitted"); a validator of ours can
                # name a key, so it is truncated rather than trusted.
                "reason": str(error.get("msg") or "")[:MAX_REASON_CHARS],
            }
        )
    payload: dict[str, Any] = {
        "error": {
            "type": "RequestValidationError",
            "status": 422,
            "message": (
                "the request body did not match this endpoint's contract, so "
                "nothing was read and nothing was written"
            ),
            "detail": {
                "fields": fields,
                "rejected_fields": len(raw),
                "reported_fields": len(fields),
                "note": (
                    "the rejected values are not echoed here on purpose; the "
                    "field names and the rule each one broke are the whole "
                    "diagnosis"
                ),
            },
            "remedy": "GET /docs is the request contract for this endpoint",
        },
        "ran_earth_engine": False,
    }
    return JSONResponse(_redact_host_paths(payload), status_code=422)


# ---------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------


def _bound_nested(item: Any, *, what: str, where: str) -> None:
    """Every string anywhere inside ``item`` is bounded, and so is every width.

    A ``detail`` object nests, so checking only its top-level strings left the
    bound to be walked around by one more ``{}``.
    """
    if isinstance(item, str):
        if len(item) > MAX_VALUE_CHARS:
            raise ValueError(
                f"{what}{where} is {len(item)} characters; at most {MAX_VALUE_CHARS} are read"
            )
        return
    if isinstance(item, Mapping):
        if len(item) > MAX_MAPPING_KEYS:
            raise ValueError(
                f"{what}{where} carries {len(item)} keys; at most {MAX_MAPPING_KEYS} are read"
            )
        for key, nested in item.items():
            if len(str(key)) > MAX_VALUE_CHARS:
                raise ValueError(
                    f"{what}{where} has a key longer than {MAX_VALUE_CHARS} characters"
                )
            _bound_nested(nested, what=what, where=f"{where}[{key!r}]")
        return
    if isinstance(item, (list, tuple)):
        if len(item) > MAX_LIST_ITEMS:
            raise ValueError(
                f"{what}{where} carries {len(item)} entries; at most {MAX_LIST_ITEMS} are read"
            )
        for index, nested in enumerate(item):
            _bound_nested(nested, what=what, where=f"{where}[{index}]")


def _bounded_mapping(value: Any, *, what: str, as_slot_values: bool = False) -> Any:
    """Cap a free-form JSON object's width and the length of every value in it.

    ``as_slot_values`` is the ``slots`` mapping, whose values are written into
    the bag by :func:`cropup.web.session.initial_writes` as ``str(raw)``. That
    string is what :data:`MAX_VALUE_CHARS` documents itself as bounding, so it
    is the string measured -- not the JSON shape it arrived in.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):  # pragma: no cover - pydantic types it first
        raise ValueError(f"{what} must be an object of name -> value")
    if len(value) > MAX_MAPPING_KEYS:
        raise ValueError(f"{what} carries {len(value)} keys; at most {MAX_MAPPING_KEYS} are read")
    for key, item in value.items():
        if len(str(key)) > MAX_VALUE_CHARS:
            raise ValueError(f"{what} has a key longer than {MAX_VALUE_CHARS} characters")
        if as_slot_values:
            stored = item if isinstance(item, str) else str(item)
            if len(stored) > MAX_VALUE_CHARS:
                raise ValueError(
                    f"{what}[{key!r}] becomes a {len(stored)}-character slot value; "
                    f"at most {MAX_VALUE_CHARS} are read"
                )
            continue
        _bound_nested(item, what=what, where=f"[{key!r}]")
    return value


def _bounded_list(value: Any, *, what: str) -> Any:
    if value is None:
        return None
    if len(value) > MAX_LIST_ITEMS:
        raise ValueError(f"{what} carries {len(value)} entries; at most {MAX_LIST_ITEMS} are read")
    return value


class PointIn(BaseModel):
    """A map pin. ``label`` is what the farmer calls the field, if anything."""

    model_config = ConfigDict(extra="forbid")

    lat: float = Field(ge=-90.0, le=90.0, allow_inf_nan=False)
    lon: float = Field(ge=-180.0, le=180.0, allow_inf_nan=False)
    label: str | None = Field(default=None, max_length=MAX_VALUE_CHARS)


class RestoreSlotIn(BaseModel):
    """One slot of the snapshot a browser kept, typed.

    Typing it is the point. ``dict[str, Any]`` validates nothing, so a payload
    like ``{"slots": {"crop": {"alternatives": 5}}}`` reached
    ``SlotValue.from_dict`` and died there as an unhandled ``TypeError`` -- a 500
    produced by client input, which SPEC's rule 3 (errors are named, not 500'd)
    does not allow. Every field here is exactly what
    :func:`cropup.web.session.sanitise_bag_payload` documents as travelling; the
    two flags a client may not assert (``confirmed`` and its timestamp) are not
    on the model at all, so they cannot even be spelled.
    """

    model_config = ConfigDict(extra="ignore")

    slot: str | None = Field(default=None, max_length=MAX_VALUE_CHARS)
    #: Absent or blank means "no reading", and the sanitiser drops the slot.
    value: str | None = Field(default=None, max_length=MAX_VALUE_CHARS)
    label: str | None = Field(default=None, max_length=MAX_VALUE_CHARS)
    origin: str | None = Field(default=None, max_length=MAX_VALUE_CHARS)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    needs_confirmation: bool | None = None
    source: str | None = Field(default=None, max_length=MAX_VALUE_CHARS)
    note: str | None = Field(default=None, max_length=MAX_VALUE_CHARS)
    alternatives: list[str] | None = None
    detail: dict[str, Any] | None = None
    set_at: str | None = Field(default=None, max_length=64)

    @field_validator("alternatives")
    @classmethod
    def _check_alternatives(cls, value: list[str] | None) -> list[str] | None:
        value = _bounded_list(value, what="alternatives")
        if value is not None:
            for item in value:
                if len(item) > MAX_VALUE_CHARS:
                    raise ValueError(f"an alternative is longer than {MAX_VALUE_CHARS} characters")
        return value

    @field_validator("detail")
    @classmethod
    def _check_detail(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        return _bounded_mapping(value, what="detail")


class RestoreIn(BaseModel):
    """The whole snapshot. Only these five keys are read; the rest are dropped."""

    model_config = ConfigDict(extra="ignore")

    session_id: str | None = Field(default=None, max_length=MAX_VALUE_CHARS)
    created_at: str | None = Field(default=None, max_length=64)
    updated_at: str | None = Field(default=None, max_length=64)
    field_radius_m: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    slots: dict[str, RestoreSlotIn] | None = None

    @field_validator("slots")
    @classmethod
    def _check_slots(cls, value: dict[str, RestoreSlotIn] | None) -> dict[str, RestoreSlotIn] | None:
        if value is not None and len(value) > MAX_MAPPING_KEYS:
            raise ValueError(f"slots carries {len(value)} entries; at most {MAX_MAPPING_KEYS}")
        return value

    def as_payload(self) -> dict[str, Any]:
        """The mapping :func:`sanitise_bag_payload` expects, with nothing absent."""
        return self.model_dump(exclude_none=True)


class SessionOpenIn(BaseModel):
    """Open a session, optionally restoring a snapshot the browser kept.

    ``restore`` is typed by :class:`RestoreIn` and then sanitised before it
    touches a bag: see :func:`cropup.web.session.sanitise_bag_payload`. Nothing
    in it can produce a confirmed slot.
    """

    model_config = ConfigDict(extra="forbid")

    restore: RestoreIn | None = None
    slots: dict[str, Any] | None = None
    intent: str | None = Field(default=None, max_length=MAX_VALUE_CHARS)
    field_radius_m: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)

    @field_validator("slots")
    @classmethod
    def _check_slots(cls, value: Any) -> Any:
        return _bounded_mapping(value, what="slots", as_slot_values=True)


class MessageIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        default="",
        max_length=MAX_MESSAGE_CHARS,
        description="the farmer's message, verbatim",
    )


class SlotsIn(BaseModel):
    """Questionnaire and map writes. Every value here locks the slot.

    ``extra="forbid"``: a top-level ``{"lat": -3.38, "lon": 36.68}`` -- the
    obvious spelling, and the wrong one, since a pin travels as ``point`` --
    used to be dropped by pydantic and answered with a 200 and an empty
    ``slot_outcomes``. A client typo that silently writes nothing is the same
    class of failure as a silent default: the farmer is shown a field they did
    not set. Naming a key this endpoint does not read is now a 422 that says
    which key.
    """

    model_config = ConfigDict(extra="forbid")

    slots: dict[str, Any] | None = None
    intent: str | None = Field(default=None, max_length=MAX_VALUE_CHARS)
    crop_key: str | None = Field(default=None, max_length=MAX_VALUE_CHARS)
    place_key: str | None = Field(default=None, max_length=MAX_VALUE_CHARS)
    point: PointIn | None = None
    field_radius_m: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    clear: list[str] | None = None
    unlock: list[str] | None = None
    origin: str | None = Field(
        default=None,
        max_length=MAX_VALUE_CHARS,
        description=f"one of {', '.join(CLIENT_ORIGINS)}; defaults to {ORIGIN_USER}",
    )

    @field_validator("slots")
    @classmethod
    def _check_slots(cls, value: Any) -> Any:
        return _bounded_mapping(value, what="slots", as_slot_values=True)

    @field_validator("clear", "unlock")
    @classmethod
    def _check_names(cls, value: list[str] | None) -> list[str] | None:
        return _bounded_list(value, what="a list of slot names")


class ConfirmIn(BaseModel):
    """SPEC 4.4's gate. Confirming also locks: an endorsed value is the farmer's."""

    model_config = ConfigDict(extra="forbid")

    slots: list[str] | None = None
    unconfirm: list[str] | None = None

    @field_validator("slots", "unconfirm")
    @classmethod
    def _check_names(cls, value: list[str] | None) -> list[str] | None:
        return _bounded_list(value, what="a list of slot names")


class RunIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str | None = Field(default=None, max_length=MAX_VALUE_CHARS)


class ParseIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(default="", max_length=MAX_MESSAGE_CHARS)


# ---------------------------------------------------------------------------
# the body-size bound
# ---------------------------------------------------------------------------


class BodySizeLimit:
    """Refuse a request body larger than ``max_bytes``, before anything parses it.

    Per-field ``max_length`` bounds what each string may be, but only after
    pydantic has decoded the whole body, and a client can send fields nobody
    declared. This is the bound underneath: it is ASGI middleware rather than a
    ``@app.middleware("http")`` function so that it sees the raw
    ``http.request`` messages and can stop a chunked body -- one with no
    ``Content-Length`` to check -- as it arrives, instead of after the fact.

    A body that declares its length is waved through or refused on the header
    alone, so the normal path buffers nothing extra.
    """

    #: Methods that carry no body worth bounding; SSE is a GET and must not be
    #: touched, because this would otherwise wait for a body that never comes.
    BODILESS = frozenset({"GET", "HEAD", "OPTIONS", "DELETE", "TRACE"})

    def __init__(self, app: Any, *, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = int(max_bytes)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("method", "").upper() in self.BODILESS:
            await self.app(scope, receive, send)
            return

        declared = _content_length(scope)
        if declared is not None:
            if declared > self.max_bytes:
                await self._refuse(send, declared)
                return
            await self.app(scope, receive, send)
            return

        # No Content-Length: read the body ourselves, stopping at the limit, and
        # replay what we read so the handler still sees a normal request.
        chunks: list[bytes] = []
        total = 0
        more = True
        while more:
            message = await receive()
            if message["type"] != "http.request":
                chunks.append(b"")  # a disconnect: let the app see it and unwind
                break
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                await self._refuse(send, None)
                return
            chunks.append(message.get("body", b""))
            more = bool(message.get("more_body"))

        body = b"".join(chunks)
        replayed = False

        async def replay() -> Any:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)

    async def _refuse(self, send: Any, declared: int | None) -> None:
        payload = {
            "error": {
                "type": "RequestTooLarge",
                "status": 413,
                "message": (
                    f"request body is larger than {self.max_bytes} bytes and was not read"
                ),
                "detail": {
                    "max_bytes": self.max_bytes,
                    "content_length": declared,
                },
                "remedy": (
                    "a farmer's question is a sentence, not a file: send at most "
                    f"{MAX_MESSAGE_CHARS} characters of text per turn"
                ),
            },
            "ran_earth_engine": False,
        }
        body = json.dumps(payload).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})


class LifespanRequired:
    """Refuse every request until this app's ASGI lifespan has run.

    The app behaved differently depending on whether it was constructed inside
    its lifespan or not, and the difference was not cosmetic.
    ``bootstrap.initialize`` runs *in* the lifespan, so without it
    ``bootstrap.ee_status()["attempted"]`` stays ``False``,
    :func:`_ee_available` returns ``None`` -- "nobody has looked yet", which the
    policy must not read as "unavailable" -- and ``POST /run`` on a confirmed
    field sailed past the gate and **started a run**, returning 200 where the
    same app under its lifespan returns a named 503. An app that has not
    checked whether the instrument is up is not entitled to an opinion about
    it, so it answers that instead of guessing.

    ``TestClient(create_app(...))`` used as a plain object rather than a context
    manager is exactly this case, which is why the failure says so by name.
    """

    def __init__(self, app: Any, *, state: Any) -> None:
        self.app = app
        self.state = state

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or getattr(self.state, "started", False):
            await self.app(scope, receive, send)
            return
        payload = {
            "error": {
                "type": "LifespanNotRun",
                "status": 503,
                "message": (
                    "this app was constructed without running its ASGI lifespan, "
                    "so startup never happened: Earth Engine has not been "
                    "attempted and its availability is unknown. No request is "
                    "served in that state, because a run authorised against an "
                    "unknown instrument is exactly what SPEC 4.4 forbids"
                ),
                "detail": {"started": False},
                "remedy": (
                    "serve it with an ASGI server (python -m cropup.web), or in "
                    "tests use TestClient as a context manager: "
                    "with TestClient(create_app(settings)) as client: ..."
                ),
            },
            "ran_earth_engine": False,
        }
        body = json.dumps(payload).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 503,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})


def _content_length(scope: Mapping[str, Any]) -> int | None:
    for name, value in scope.get("headers") or ():
        if name.lower() == b"content-length":
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


# ---------------------------------------------------------------------------
# host paths are the operator's, not the client's
# ---------------------------------------------------------------------------

#: The directory holding the ``cropup`` package, and the account's home. Both are
#: substituted in anything a client is shown: ``/api/health`` is a liveness
#: probe, and "which artifacts are present" is the answer it owes a caller --
#: "/Users/<account>/VSCode/cropup/..." is not. Which file under the app is
#: missing *is* the answer, so ``<app>/cropup/data/crops.json`` survives.
_APP_ROOT = str(Path(__file__).resolve().parents[2])
_HOME = str(Path.home())

#: Placeholders for the two roots above, kept out of the regex's way while it
#: works. They are opaque to :data:`_PATH_TOKEN`'s trailing character class, so
#: the whole of ``<app>/cropup/data/x.csv`` matches as one token and is kept.
_APP_MARK = "\x00app\x00"
_HOME_MARK = "\x00home\x00"

#: The **allowlist**: the only absolute paths a client may be shown are this
#: app's own URL space, which is where a remedy sends them ("POST
#: /api/session/{sid}/confirm"). Everything else shaped like a filesystem path
#: is redacted, whatever directory it names.
#:
#: This replaced a denylist of five prefixes. That denylist failed open the
#: moment ``CROPUP_DATA_DIR`` or ``CROPUP_STATIC_DIR`` pointed at ``/srv``,
#: ``/opt``, ``/mnt`` or a container's ``/data`` -- none of which start with
#: ``/Users/`` -- and the redaction was only applied to two endpoints anyway,
#: while :func:`_error_response` answers every one of them, unauthenticated.
_ALLOWED_ABSOLUTE = re.compile(
    r"^/$|^/(?:api|static|docs|redoc|openapi\.json|favicon\.ico)(?:[/?#].*)?$"
)

#: One filesystem-path-shaped token: a POSIX absolute path, a Windows drive
#: path, or one of the two markers above. The lookbehind is what keeps ``mm/day``
#: and ``ISDASOIL/Africa/v1`` -- relative, and not paths on this host -- intact:
#: a token must start at a boundary, not mid-word.
_PATH_TOKEN = re.compile(
    r"(?<![\w.~-])(?:\x00app\x00|\x00home\x00|[A-Za-z]:[\\/]|/)[^\s,;'\"`<>()\[\]]*"
)


def _redact_token(match: re.Match[str]) -> str:
    token = match.group(0)
    if token.startswith(_APP_MARK) or token.startswith(_HOME_MARK):
        return token  # already named relative to a root the client may know
    if _ALLOWED_ABSOLUTE.match(token):
        return token  # a route on this app, not a directory on this machine
    return "<path hidden>"


def _redact_host_paths(value: Any) -> Any:
    """``value`` with every filesystem path that names this host taken out.

    Fail-closed: an absolute path survives only by being on
    :data:`_ALLOWED_ABSOLUTE` or by lying under the app root or the home
    directory, where it has already been renamed ``<app>``/``<home>``. A path
    nobody anticipated is redacted rather than printed.
    """
    if isinstance(value, str):
        text = value
        if _APP_ROOT:
            text = text.replace(_APP_ROOT, _APP_MARK)
        if _HOME and _HOME != "/":
            text = text.replace(_HOME, _HOME_MARK)
        # A caller that already substituted gets the same treatment, so the two
        # spellings cannot disagree about what is hidden.
        text = text.replace("<app>", _APP_MARK).replace("<home>", _HOME_MARK)
        text = _PATH_TOKEN.sub(_redact_token, text)
        return text.replace(_APP_MARK, "<app>").replace(_HOME_MARK, "<home>")
    if isinstance(value, Mapping):
        return {key: _redact_host_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_host_paths(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# helpers shared by the handlers
# ---------------------------------------------------------------------------


def _ee_available() -> bool | None:
    """The tri-state ``Frame.earth_engine_available`` wants.

    ``None`` means nobody has looked yet, which is not "unavailable" -- reading
    ``ee_status()`` rather than calling ``ee_ready()`` is what keeps a chat turn
    from initialising Earth Engine (SPEC 4.4).
    """
    state = bootstrap.ee_status()
    if not state.get("attempted"):
        return None
    return bool(state.get("ready"))


def _frame(bag: SlotBag, settings: Settings, *, text: str = "", **kwargs: Any) -> policy_mod.Frame:
    return policy_mod.Frame.from_bag(
        bag,
        text=text,
        earth_engine_available=_ee_available(),
        policy=policy_mod.PolicyConfig.from_settings(settings),
        **kwargs,
    )


def _confirmed_point(session: Session) -> tuple[float | None, float | None]:
    """The confirmed field's coordinates, for the coverage cliffs. Never a guess."""
    location = session.bag.get("location")
    if location is None or not location.confirmed:
        return None, None
    coords = location.coordinates()
    return coords if coords is not None else (None, None)


def _answer_turn(
    session: Session,
    action: policy_mod.Action,
    settings: Settings,
    *,
    parse: Mapping[str, Any] | None = None,
    slot_outcomes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Render whatever the policy decided. Never runs Earth Engine.

    This is the whole body of a chat turn and of a questionnaire write. It has
    no access to :func:`cropup.web.dispatch.run` and cannot mint the
    authorisation that function requires, so SPEC 4.4 holds structurally rather
    than by inspection of this function.
    """
    retrieval: dict[str, Any] | None = None
    degraded_reason: str | None = None

    if action.kind == policy_mod.ACTION_ANSWER_FROM_RAG:
        answer, retrieval, degraded_reason = replies.knowledge_answer(
            action, session.bag, settings=settings
        )
    else:
        answer = replies.dialog_answer(action)

    envelope = replies.reply_envelope(
        action=action,
        answer=answer,
        bag=session.bag,
        ran_earth_engine=False,
        parse=parse,
        slot_outcomes=slot_outcomes,
        retrieval=retrieval,
        degraded_reason=degraded_reason,
    )
    # The invariant this endpoint exists to hold. If it ever fails, the turn
    # must not be served: a farmer being told a chat message spent quota is the
    # failure SPEC 4.4 is written against.
    if envelope["ran_earth_engine"]:  # pragma: no cover - unreachable by construction
        raise AssertionError("a natural-language turn reported an Earth Engine run (SPEC 4.4)")
    session.record_turn(
        {
            "at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "action": action.kind,
            "intent": action.intent,
            "ran_earth_engine": False,
        }
    )
    return envelope


def _run_degradation_reason(outcome: Any) -> str | None:
    """Why an Earth Engine run's answer is degraded, or ``None`` when it is not.

    The envelope's top-level ``degraded`` used to be set only on the RAG path,
    because ``degraded_reason`` was only ever passed there: a ``crop_selection``
    run in which nothing resolved came back ``degraded: false`` while its own
    ``run.answer.degradation.degraded`` said ``true``. The capability strip
    reads the top level, so the honest verdict the ledger had already reached
    never arrived on screen.

    This names the *kinds* of degradation and points at the lists. It does not
    count or format anything: the quantities are named in the ledger's own
    payload, which is where they carry their provenance (rule 2 above).
    """
    degradation = getattr(outcome.answer, "degradation", None) or {}
    if not degradation.get("degraded"):
        return None
    parts: list[str] = []
    if degradation.get("gaps"):
        parts.append(
            "quantities this run could not measure, named in "
            "run.answer.degradation.gaps"
        )
    if degradation.get("fallbacks"):
        parts.append(
            "quantities answered from further down their source chain, named in "
            "run.answer.degradation.fallbacks"
        )
    if degradation.get("stale"):
        parts.append(
            "readings older than this run's freshness window, named in "
            "run.answer.degradation.stale"
        )
    if not parts:  # pragma: no cover - degraded is true only when one of the three is
        parts.append("named in run.answer.degradation")
    return (
        "this answer is degraded: " + "; ".join(parts) + ". Nothing was "
        "substituted for a missing value (SPEC 4.1)."
    )


def _write_slots(bag: SlotBag, body: SlotsIn, settings: Settings) -> dict[str, str]:
    """Apply one questionnaire/map payload. Returns slot -> outcome."""
    origin = resolve_origin(body.origin)
    outcomes: dict[str, str] = {}

    if body.field_radius_m is not None:
        bag.set_field_radius(body.field_radius_m)

    for slot in sequence_of(body.clear):
        if bag.clear(slot) is not None:
            outcomes[slot] = "cleared"
    for slot in sequence_of(body.unlock):
        bag.unlock(slot)
        outcomes.setdefault(slot, "unlocked")

    if body.crop_key:
        entry = nlu_slots.find_crop(body.crop_key, settings)
        if entry is None:
            raise ValueError(
                f"crop {body.crop_key!r} is not in the committed vocabulary; "
                "pick one from GET /api/vocab/crops"
            )
        stored = bag.set(
            "crop",
            entry.name,
            origin=origin,
            label=entry.name,
            source="crops.json",
            detail=entry.as_dict(),
        )
        outcomes["crop"] = "set" if stored is not None else "refused_locked"

    if body.place_key:
        place = nlu_slots.find_place(body.place_key, settings)
        if place is None:
            raise ValueError(
                f"place {body.place_key!r} is not in the committed gazetteer; "
                "pick one from GET /api/vocab/places"
            )
        stored = bag.set(
            "location",
            place.key,
            origin=origin,
            label=place.name,
            # The gazetteer is the source even when the farmer picked the row:
            # the coordinates are the artifact's, and the provenance of a
            # coordinate is not the click that selected it.
            source="gazetteer.json",
            detail={**place.as_dict(), "lat": place.lat, "lon": place.lon},
            # A picked row is unambiguous by the act of picking, whatever the
            # gazetteer flags about the name.
            needs_confirmation=False,
        )
        outcomes["location"] = "set" if stored is not None else "refused_locked"

    if body.point is not None:
        label = (body.point.label or "").strip() or (
            f"pinned field {body.point.lat:.5f}, {body.point.lon:.5f}"
        )
        stored = bag.set(
            "location",
            label,
            origin=origin,
            label=label,
            source="map_pin",
            detail={"lat": float(body.point.lat), "lon": float(body.point.lon)},
        )
        outcomes["location"] = "set" if stored is not None else "refused_locked"

    if body.intent:
        if body.intent not in intents_mod.BY_NAME:
            raise ValueError(
                f"unknown intent {body.intent!r} (known: {', '.join(intents_mod.INTENT_NAMES)})"
            )
        bag.set_intent(body.intent, origin=origin)
        outcomes[INTENT_SLOT] = "set"

    if body.slots:
        outcomes.update(initial_writes(bag, body.slots, origin=origin))
    return outcomes


def _placeholder_page(settings: Settings) -> HTMLResponse:
    """What ``GET /`` serves before the frontend exists. Never a 500, never a 404.

    The static half of SPEC section 9 is written by a different author against
    the contract below; until their ``index.html`` lands, the app is running and
    should say so in the browser rather than looking broken.
    """
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CropUp {__version__}</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ font: 16px/1.5 system-ui, sans-serif; margin: 0 auto; padding: 2rem 1rem;
        max-width: 46rem; }}
 code {{ font-family: ui-monospace, monospace; }}
 li {{ margin: .2rem 0; }}
 .note {{ border-left: 3px solid currentColor; padding-left: .8rem; opacity: .85; }}
</style></head><body>
<h1>CropUp {__version__} is running</h1>
<p class="note">The API is serving. The single-page app is not installed yet:
there is no <code>index.html</code> in the configured
<code>CROPUP_STATIC_DIR</code>. This page is a placeholder, not an error.
(The directory is named in the server's own logs, not here: this page is
public and the path is the operator's.)</p>
<h2>Start here</h2>
<ul>
 <li><a href="/api/health">/api/health</a> — startup assertions</li>
 <li><a href="/api/capabilities">/api/capabilities</a> — what is degraded right now</li>
 <li><a href="/docs">/docs</a> — the full request/response contract</li>
 <li><a href="/api/vocab/crops?q=mahindi">/api/vocab/crops?q=mahindi</a> — crop autocomplete</li>
 <li><a href="/api/vocab/places?q=arusha">/api/vocab/places?q=arusha</a> — gazetteer autocomplete</li>
</ul>
<p>Earth Engine never runs on a field the farmer has not confirmed, and a chat
message never starts a measurement (SPEC 4.4).</p>
</body></html>"""
    return HTMLResponse(body, status_code=200)


# ---------------------------------------------------------------------------
# the app
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. One :class:`SessionStore` and one :class:`ProgressHub` per app."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        bootstrap.configure_logging(settings=settings)
        log = bootstrap.get_logger("web")
        # Attempted once, off the request path, on a background thread: a dead
        # or slow Earth Engine endpoint must not delay the first page load. The
        # attempt is bounded by CROPUP_EE_REQUEST_TIMEOUT_S inside bootstrap,
        # and it never raises -- a failure is recorded as the verdict that
        # /api/health and /api/capabilities report.
        thread = threading.Thread(
            target=bootstrap.initialize, name="cropup-ee-initialize", daemon=True
        )
        thread.start()
        log.info(
            bootstrap.literal("cropup web app starting"),
            extra={"version": __version__, "ee_enabled": settings.ee_enabled},
        )
        # The flag :class:`LifespanRequired` gates on. Set last, so nothing is
        # served before the bootstrap attempt has been *started* and the store
        # exists; cleared first on the way down, so a request arriving during
        # shutdown is refused rather than half-served.
        application.state.started = True
        yield
        application.state.started = False
        application.state.sessions.clear()

    application = FastAPI(
        title="CropUp",
        version=__version__,
        summary=(
            "Crop understanding grounded in Earth observation. Every number "
            "carries its instrument; anything not measured is named as missing."
        ),
        lifespan=lifespan,
    )
    application.state.settings = settings
    #: Flipped by the lifespan above. Until then nothing is served: see
    #: :class:`LifespanRequired`.
    application.state.started = False
    application.state.progress = progress.ProgressHub()
    # An evicted session takes its SSE backlog with it, so the two bounded
    # structures cannot keep each other alive.
    application.state.sessions = SessionStore(
        settings, on_evict=application.state.progress.forget
    )

    # Added first, so it ends up *inside* BodySizeLimit: a body too large to
    # read is refused for being too large whether or not startup has run.
    application.add_middleware(LifespanRequired, state=application.state)
    # Outermost, so an oversized body is refused before a route, a handler or a
    # validator has seen a byte of it.
    application.add_middleware(BodySizeLimit, max_bytes=MAX_BODY_BYTES)

    for kind, _status in ERROR_STATUS:
        application.add_exception_handler(
            kind, lambda _request, exc: _error_response(exc)  # noqa: ARG005
        )
    # Registered after the loop, and separately: ``RequestValidationError`` is a
    # subclass of ``ValueError``, so without this it would either take the 400
    # above or -- as it did -- FastAPI's built-in handler, which answers in a
    # shape this app does not use and quotes the rejected input back.
    application.add_exception_handler(
        RequestValidationError, lambda _request, exc: _validation_response(exc)  # noqa: ARG005
    )

    _register_routes(application, settings)

    if settings.static_dir.is_dir():
        application.mount(
            "/static", StaticFiles(directory=str(settings.static_dir)), name="static"
        )
    return application


def _register_routes(application: FastAPI, settings: Settings) -> None:  # noqa: C901 - 14 endpoints
    store: SessionStore = application.state.sessions
    hub: progress.ProgressHub = application.state.progress

    # -- 1. the single-page app -------------------------------------------

    @application.get("/", include_in_schema=False)
    def index() -> Response:
        """SPEC 8: the single-page app. A placeholder while static/ is empty."""
        page = settings.static_dir / "index.html"
        if page.is_file():
            return FileResponse(str(page), media_type="text/html")
        return _placeholder_page(settings)

    # -- 2. liveness -------------------------------------------------------

    @application.get("/api/health")
    def health() -> Response:
        """Liveness plus the startup assertions. 503 when a required one failed.

        Unauthenticated, so it reports *presence*, never location: an operator's
        home directory is not a liveness signal, and ``/api/health`` is reachable
        by anyone who can reach the port. Absolute paths are stripped from the
        whole report (:func:`_redact_host_paths`), including the assertion
        details that ``bootstrap`` writes.
        """
        report = _redact_host_paths(bootstrap.health_report(settings))
        report["app"] = {
            "name": "cropup",
            "version": __version__,
            "sessions": store.status(),
            "progress": hub.status(),
            # Presence, not the path: whether the single-page app is installed is
            # what a probe needs to know, and where it lives on this machine is
            # the operator's business.
            "static_dir_present": settings.static_dir.is_dir(),
            "index_html": (settings.static_dir / "index.html").is_file(),
        }
        return JSONResponse(report, status_code=200 if report["status"] == "ok" else 503)

    # -- 3. the honesty endpoint ------------------------------------------

    @application.get("/api/capabilities")
    def capabilities(
        sid: str | None = Query(default=None, description="answer the coverage cliffs at this session's confirmed field"),
        lat: float | None = Query(default=None),
        lon: float | None = Query(default=None),
    ) -> Response:
        """The live degradation matrix. Answers 200 during an outage, by design.

        Unauthenticated like ``/api/health``, and redacted the same way: which
        artifact is missing is the answer; where it would have lived on this
        machine is not.
        """
        if lat is None and lon is None and sid:
            session = store.peek(sid)
            if session is not None:
                lat, lon = _confirmed_point(session)
        report = _redact_host_paths(capability_report(settings, lat=lat, lon=lon))
        report["version"] = __version__
        report["sessions"] = store.status()
        return JSONResponse(report, status_code=200)

    # -- 4. open a session -------------------------------------------------

    @application.post("/api/session", status_code=201)
    def open_session(body: SessionOpenIn | None = None) -> Response:
        """Open one SlotBag. ``restore`` is sanitised: it cannot mint a confirmation."""
        body = body or SessionOpenIn()
        session = store.open(
            restore=body.restore.as_payload() if body.restore is not None else None,
            field_radius_m=body.field_radius_m,
        )
        outcomes: dict[str, str] = {}
        if body.slots:
            outcomes = initial_writes(session.bag, body.slots, origin=ORIGIN_USER)
        if body.intent:
            if body.intent not in intents_mod.BY_NAME:
                raise ValueError(f"unknown intent {body.intent!r}")
            session.bag.set_intent(body.intent, origin=ORIGIN_USER)
        action = policy_mod.next_action(_frame(session.bag, settings))
        payload = session.as_dict()
        payload.update(
            {
                "action": action.as_dict(),
                "slot_outcomes": outcomes,
                "restored": bool(body.restore),
                "restore_note": (
                    "a restored snapshot never carries confirmation: SPEC 4.4's "
                    "gate is opened by POST /api/session/{sid}/confirm in this "
                    "process and nowhere else"
                )
                if body.restore
                else None,
                "field": geofield.field_report(session.bag, settings=settings),
                "ttl_s": store.ttl_s,
            }
        )
        return JSONResponse(payload, status_code=201)

    # -- 5. rehydrate ------------------------------------------------------

    @application.get("/api/session/{sid}")
    def read_session(sid: str) -> Response:
        """Everything this session knows, for a reload. 404 once the TTL expires."""
        session = store.get(sid)
        payload = session.as_dict()
        payload["action"] = policy_mod.next_action(_frame(session.bag, settings)).as_dict()
        payload["field"] = geofield.field_report(session.bag, settings=settings)
        payload["ttl_s"] = store.ttl_s
        return JSONResponse(payload, status_code=200)

    # -- 6. a natural-language turn ---------------------------------------

    @application.post("/api/session/{sid}/message")
    def message(sid: str, body: MessageIn) -> Response:
        """One chat turn. **Never** runs Earth Engine (SPEC 4.4).

        Classifies, folds the extraction into the shared bag as *suggestions*
        that never overwrite what the farmer set, and answers from the cited
        corpus when the intent routes there. An Earth-Engine-routed intent stops
        at the confirmation gate and asks; even a fully confirmed field only
        gets "press Run", because starting a measurement is an explicit act.
        """
        session = store.get(sid)
        with session.lock:
            parse = nlu_classify.parse(body.text, settings)
            outcomes = session.bag.apply_extraction(parse.extraction)
            if parse.classification.intent and parse.classification.intent != intents_mod.CLARIFY_INTENT:
                session.bag.set_intent(parse.classification.intent)
            frame = policy_mod.Frame.from_parse(
                parse,
                session.bag,
                earth_engine_available=_ee_available(),
                policy=policy_mod.PolicyConfig.from_settings(settings),
            )
            action = policy_mod.next_action(frame)
            envelope = _answer_turn(
                session, action, settings, parse=parse.as_dict(), slot_outcomes=outcomes
            )
            envelope["field"] = geofield.field_report(
                session.bag,
                analysis=dispatch.ORCHESTRATORS[action.intent].analysis
                if action.intent in dispatch.ORCHESTRATORS
                else None,
                settings=settings,
            )
        hub.publish(sid, progress.EVENT_SLOTS, {"source": "message", "outcomes": outcomes})
        return JSONResponse(envelope, status_code=200)

    # -- 7. questionnaire and map writes ----------------------------------

    @application.post("/api/session/{sid}/slots")
    def write_slots(sid: str, body: SlotsIn) -> Response:
        """Questionnaire, picker and map writes. A user value **locks** the slot.

        The same bag the chat writes into, so switching panels mid-conversation
        loses nothing (SPEC 9). Writing a slot drops any confirmation it carried,
        because the confirmed thing was the old value.
        """
        session = store.get(sid)
        with session.lock:
            outcomes = _write_slots(session.bag, body, settings)
            action = policy_mod.next_action(_frame(session.bag, settings))
            answer = replies.dialog_answer(action)
            envelope = replies.reply_envelope(
                action=action,
                answer=answer,
                bag=session.bag,
                ran_earth_engine=False,
                slot_outcomes=outcomes,
            )
            envelope["field"] = geofield.field_report(
                session.bag,
                analysis=dispatch.ORCHESTRATORS[action.intent].analysis
                if action.intent in dispatch.ORCHESTRATORS
                else None,
                settings=settings,
            )
        hub.publish(sid, progress.EVENT_SLOTS, {"source": "slots", "outcomes": outcomes})
        return JSONResponse(envelope, status_code=200)

    # -- 8. the gate -------------------------------------------------------

    @application.post("/api/session/{sid}/confirm")
    def confirm(sid: str, body: ConfirmIn | None = None) -> Response:
        """SPEC 4.4's gate. The only act in this process that authorises a run.

        With no ``slots`` it confirms the confirmable slots the policy is
        waiting on -- location and crop -- and nothing else. Confirming an empty
        slot is a 400, not a silent no-op.

        **An explicit ``unconfirm`` shuts the gate and does not reopen it.**
        This used to be unreachable: the withdrawal happened, and then the
        ``slots``-is-empty branch asked the policy which slots were unconfirmed
        and confirmed exactly those -- the ones just withdrawn. The reply said
        ``confirmed: ["location"]`` and ``unconfirmed: ["location"]`` at once,
        the bag stayed confirmed, and a farmer who had realised the pin was on
        the wrong field had no way to take the authorisation back. So a request
        that names ``unconfirm`` confirms only what it *also* names in
        ``slots``, and naming the same slot in both is a 400 rather than a
        coin toss.
        """
        body = body or ConfirmIn()
        session = store.get(sid)
        with session.lock:
            bag = session.bag
            asked_to_withdraw = sequence_of(body.unconfirm)
            wanted = sequence_of(body.slots)
            contradictory = tuple(s for s in wanted if s in asked_to_withdraw)
            if contradictory:
                raise ValueError(
                    f"{', '.join(contradictory)} named in both slots and unconfirm: "
                    "one request cannot both grant and withdraw the same "
                    "authorisation, and this endpoint will not pick for you"
                )
            withdrawn = bag.unconfirm(*asked_to_withdraw) if asked_to_withdraw else ()
            if not wanted and not asked_to_withdraw:
                # Only a request that withdraws nothing may be read as "confirm
                # whatever is outstanding". Filling it in after a withdrawal is
                # what re-granted the endorsement that had just been taken back.
                before = policy_mod.next_action(_frame(bag, settings))
                wanted = tuple(before.unconfirmed_slots) or tuple(
                    s for s in policy_mod.CONFIRMABLE_SLOTS if bag.has(s)
                )
            missing = tuple(s for s in wanted if not bag.has(s))
            if missing:
                raise ValueError(
                    f"cannot confirm {', '.join(missing)}: nothing is filled in "
                    f"{'them' if len(missing) > 1 else 'it'} yet"
                )
            confirmed = bag.confirm(*wanted) if wanted else ()
            action = policy_mod.next_action(_frame(bag, settings))
            envelope = replies.reply_envelope(
                action=action,
                answer=replies.dialog_answer(action),
                bag=bag,
                ran_earth_engine=False,
            )
            envelope["confirmed"] = list(confirmed)
            envelope["unconfirmed"] = list(withdrawn)
            envelope["field"] = geofield.field_report(
                bag,
                analysis=dispatch.ORCHESTRATORS[action.intent].analysis
                if action.intent in dispatch.ORCHESTRATORS
                else None,
                settings=settings,
            )
        hub.publish(
            sid,
            progress.EVENT_SLOTS,
            {"source": "confirm", "confirmed": list(confirmed), "unconfirmed": list(withdrawn)},
        )
        return JSONResponse(envelope, status_code=200)

    # -- 9. the explicit run ----------------------------------------------

    @application.post("/api/session/{sid}/run")
    def run_analysis(sid: str, body: RunIn | None = None) -> Response:
        """Run the authorised analysis. The only endpoint that may reach Earth Engine.

        It does not check the bag itself: it calls ``dispatch.authorise``, which
        asks ``dialog.policy`` the same question the chat turn asked. An
        unconfirmed field is a 409 here exactly as it is there, so posting
        straight to this path bypasses nothing.
        """
        body = body or RunIn()
        session = store.get(sid)
        with session.lock:
            try:
                authorization = dispatch.authorise(
                    session,
                    intent=body.intent,
                    settings=settings,
                    earth_engine_available=_ee_available(),
                )
            except BaseException as exc:
                # The open frame advertises ``run_failed``, and a client watching
                # /events pressed Run: a refusal at the gate is a run that
                # failed, and leaving the stream silent makes a refused run look
                # like a hung one. ``dispatch.run`` publishes this event for
                # every failure after the gate; this is the one before it.
                hub.publish(
                    sid,
                    progress.EVENT_RUN_FAILED,
                    {
                        "intent": body.intent or session.bag.intent,
                        "analysis": None,
                        "stage": "authorisation",
                        "error": type(exc).__name__,
                        "detail": str(exc),
                        "status": _status_for(exc),
                        "ran_earth_engine": False,
                    },
                )
                raise
            outcome = dispatch.run(authorization, hub=hub, settings=settings)
            envelope = replies.reply_envelope(
                action=authorization.action,
                answer=outcome.answer,
                bag=session.bag,
                ran_earth_engine=True,
                run=outcome.as_dict(),
                degraded_reason=_run_degradation_reason(outcome),
                diagnostics=outcome.diagnostics,
            )
            envelope["field"] = geofield.field_report(
                session.bag, analysis=authorization.analysis, settings=settings
            )
            session.last_run = {
                "at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "analysis": authorization.analysis,
                "elapsed_s": round(outcome.elapsed_s, 2),
            }
            session.record_turn(
                {
                    "at": session.last_run["at"],
                    "action": authorization.action.kind,
                    "intent": authorization.intent,
                    "ran_earth_engine": True,
                }
            )
        return JSONResponse(envelope, status_code=200)

    # -- 10. progress ------------------------------------------------------

    @application.get("/api/session/{sid}/events")
    def events(
        sid: str,
        request: Request,
        last_event_id: int = Query(
            default=0,
            alias="last_event_id",
            description="replay backlog events numbered above this",
        ),
        max_events: int | None = Query(
            default=None,
            ge=1,
            description="close the stream after this many events; for polling clients",
        ),
        max_wait_s: float | None = Query(
            default=None,
            gt=0.0,
            le=3600.0,
            description="close the stream after this many seconds, whatever arrived",
        ),
        heartbeat_s: float = Query(default=15.0, gt=0.0, le=300.0),
    ) -> Response:
        """Server-sent progress, one stream per session (SPEC 3.4).

        Named events: ``open``, ``run_planned``, ``progress``, ``leg``,
        ``run_finished``, ``run_failed``, ``slots``. Reconnect with the
        ``Last-Event-ID`` header, or the ``last_event_id`` query parameter.

        By default the stream stays open until the client goes away, which is
        what a browser wants. ``max_events`` and ``max_wait_s`` bound it for a
        polling client; ``max_events`` alone will wait indefinitely if that many
        events never arrive, so a poller should pass both.
        """
        store.get(sid)  # 404 before a stream is opened, not inside it
        cursor = last_event_id
        header = request.headers.get("last-event-id")
        if header:
            try:
                cursor = max(cursor, int(header))
            except ValueError:
                pass

        async def body():
            import time as _time  # noqa: PLC0415 - only the bounded stream needs a clock

            sent = 0
            deadline = None if max_wait_s is None else _time.monotonic() + max_wait_s
            beat = heartbeat_s if deadline is None else min(heartbeat_s, max_wait_s)
            async for frame in hub.stream(sid, after_id=cursor, heartbeat_s=beat):
                yield frame
                if deadline is not None and _time.monotonic() >= deadline:
                    return
                if frame.startswith(":"):  # a keep-alive comment is not an event
                    continue
                sent += 1
                if max_events is not None and sent >= max_events:
                    return

        return StreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # nginx must not buffer a progress stream
            },
        )

    # -- 11. pure NLU inspection ------------------------------------------

    @application.post("/api/nlu/parse")
    def nlu_parse(body: ParseIn) -> Response:
        """Classify and extract, with no side effects at all.

        No session is read or written, no bag is touched, and nothing here can
        reach Earth Engine. ``would_ask`` shows what the policy would do with a
        *fresh* bag, which is why it is computed against a throwaway one.
        """
        parse = nlu_classify.parse(body.text, settings)
        scratch = SlotBag(session_id=None)
        scratch.apply_extraction(parse.extraction)
        preview = policy_mod.next_action(
            policy_mod.Frame.from_parse(
                parse,
                scratch,
                earth_engine_available=_ee_available(),
                policy=policy_mod.PolicyConfig.from_settings(settings),
            )
        )
        # Redacted like /api/health and /api/capabilities: classifier_status
        # carries the on-disk model path, which names the operator's home
        # directory and the exact model revision, and this endpoint is
        # unauthenticated.
        return JSONResponse(
            _redact_host_paths(
                {
                    **parse.as_dict(),
                    "would_ask": preview.as_dict(),
                    "classifier": nlu_classify.classifier_status(settings),
                    "side_effects": None,
                }
            ),
            status_code=200,
        )

    # -- 12/13. autocomplete ----------------------------------------------

    @application.get("/api/vocab/crops")
    def vocab_crops(
        q: str = Query(default="", description="a prefix, an alias or a Swahili name"),
        limit: int = Query(default=10, ge=1, le=100),
    ) -> Response:
        """The 134-crop / 379-alias vocabulary. A blank query lists it alphabetically."""
        rows = nlu_slots.suggest_crops(q, limit=limit, settings=settings)
        return JSONResponse(
            {
                "query": q,
                "count": len(rows),
                "results": rows,
                "note": (
                    "matched_alias shows why a row was returned (mahindi -> Maize); "
                    "nothing here fills a slot, POST /api/session/{sid}/slots does"
                ),
            },
            status_code=200,
        )

    @application.get("/api/vocab/places")
    def vocab_places(
        q: str = Query(default=""),
        limit: int = Query(default=10, ge=1, le=100),
    ) -> Response:
        """The k-anonymised gazetteer (SPEC 7). ``ambiguous`` rides on every row."""
        rows = nlu_slots.suggest_places(q, limit=limit, settings=settings)
        return JSONResponse(
            {
                "query": q,
                "count": len(rows),
                "results": rows,
                "note": (
                    "coordinates are rounded to ~1 km and entries with fewer than "
                    "5 pings are suppressed (SPEC 7); ambiguous=true means several "
                    "real places share the name"
                ),
            },
            status_code=200,
        )

    # -- 14. the exact field ----------------------------------------------

    @application.get("/api/geo/field")
    def geo_field(
        sid: str | None = Query(default=None, description="read the field from this session's bag"),
        lat: float | None = Query(default=None),
        lon: float | None = Query(default=None),
        radius_m: float | None = Query(default=None, gt=0.0),
        analysis: str | None = Query(
            default=None, description="plant_health, irrigation or crop_selection"
        ),
    ) -> Response:
        """The exact polygon that **will** be sent to Earth Engine.

        With ``sid`` it reads the session's bag. With ``lat``/``lon`` it previews
        an arbitrary point, which is what the map needs while the farmer is still
        dragging the pin -- previewing costs nothing and touches no session.
        """
        if lat is not None and lon is not None:
            scratch = SlotBag(session_id=None, field_radius_m=radius_m)
            scratch.set(
                "location",
                f"preview {lat:.5f}, {lon:.5f}",
                origin=ORIGIN_USER,
                source="preview",
                detail={"lat": float(lat), "lon": float(lon)},
            )
            report = geofield.field_report(scratch, analysis=analysis, settings=settings)
            report["preview"] = True
            report["session_id"] = None
            return JSONResponse(report, status_code=200)

        if not sid:
            raise ValueError(
                "give either sid=<session id> or lat=&lon=: this endpoint describes "
                "a specific field, and there is no default field"
            )
        session = store.get(sid)
        chosen = analysis
        if chosen is None and session.bag.intent in dispatch.ORCHESTRATORS:
            chosen = dispatch.ORCHESTRATORS[session.bag.intent].analysis
        report = geofield.field_report(session.bag, analysis=chosen, settings=settings)
        report["preview"] = False
        report["session_id"] = sid
        return JSONResponse(report, status_code=200)


#: The module-level app uvicorn is pointed at: ``cropup.web.server:app``.
app = create_app()
