"""The only door to Earth Engine, and the intent-to-orchestrator table.

SPEC section 4.4 says a natural-language turn never triggers Earth Engine. That
is easy to honour by convention and easy to lose by accident, so it is a type
here instead: an orchestrator is never called with a latitude and a longitude,
it is called with a :class:`RunAuthorization`, and the only function that can
produce one is :func:`authorise`, which routes the decision through
``dialog.policy`` and raises :class:`~cropup.errors.ConfirmationRequired` when
the gate is shut.

Two consequences, both deliberate:

* ``POST /api/session/{sid}/message`` cannot reach Earth Engine even if a
  future edit to the message handler wanted to, because it has nothing to hand
  :func:`run` and cannot mint one -- :class:`RunAuthorization` refuses to be
  constructed without the module-private mint.
* ``POST /api/session/{sid}/run`` cannot bypass the gate either. It does not
  check the bag itself; it calls :func:`authorise`, which calls
  ``policy.authorise_earth_engine``, which is the same pure decision the chat
  turn got. There is no second copy of the rule to fall out of step.

The deadline of SPEC section 3.4 is applied here too: the whole fan-out runs
under one :class:`~cropup.bootstrap.Deadline`, so a wedged asset abandons the
turn instead of holding it open.

This module imports ``analysis`` and therefore Earth Engine's call sites; it is
imported by ``server`` and by nothing else in ``web``.
"""

from __future__ import annotations

import datetime as dt
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .. import bootstrap
from ..analysis import crop_selection as _crop_selection
from ..analysis import irrigation as _irrigation
from ..analysis import plant_health as _plant_health
from ..config import Settings, get_settings
from ..dialog import policy as policy_mod
from ..dialog.slots import ORIGIN_USER, SlotBag, validate_field_radius_m
from ..errors import EarthEngineUnavailable
from ..nlu import intents as intents_mod
from ..render import templates as render
from . import replies
from .progress import (
    EVENT_LEG,
    EVENT_PROGRESS,
    EVENT_RUN_FAILED,
    EVENT_RUN_FINISHED,
    EVENT_RUN_PLANNED,
    ProgressHub,
)
from .session import Session

__all__ = [
    "ORCHESTRATORS",
    "EE_INTENTS",
    "RunAuthorization",
    "RunOutcome",
    "authorise",
    "run",
    "planned_legs",
    "run_budget_s",
    "resolve_end_date",
]


@dataclass(frozen=True)
class _Orchestrator:
    """One Earth Engine pipeline: what to call, and which legs it fans out to."""

    intent: str
    analysis: str
    call: Callable[..., Any]
    module: Any

    @property
    def legs(self) -> tuple[str, ...]:
        # Read off the orchestrator's own leg/asset table rather than restated
        # here, so the SSE plan cannot drift from the code that runs.
        return tuple(getattr(self.module, "_LEG_ASSETS", {}) or ())

    def assets(self, leg: str) -> tuple[str, ...]:
        return tuple((getattr(self.module, "_LEG_ASSETS", {}) or {}).get(leg, ()))


#: SPEC section 5.2: four of the twelve intents route to Earth Engine. Two of
#: them -- a status report and a named symptom -- are answered by the same
#: measurements; they differ in what the farmer is told, which is the renderer's
#: job, not this table's.
ORCHESTRATORS: dict[str, _Orchestrator] = {
    "field_health_check": _Orchestrator(
        "field_health_check", "plant_health", _plant_health.analyze_plant_health, _plant_health
    ),
    "crop_problem_diagnosis": _Orchestrator(
        "crop_problem_diagnosis", "plant_health", _plant_health.analyze_plant_health, _plant_health
    ),
    "irrigation_advice": _Orchestrator(
        "irrigation_advice", "irrigation", _irrigation.analyze_irrigation, _irrigation
    ),
    "crop_selection": _Orchestrator(
        "crop_selection", "crop_selection", _crop_selection.analyze_crop_selection, _crop_selection
    ),
}

EE_INTENTS: tuple[str, ...] = tuple(ORCHESTRATORS)

# The mint. An object identity no caller outside this module can obtain, which
# is what makes RunAuthorization unforgeable from the request handlers.
_MINT = object()

#: The check line ``dialog.policy`` appends when, and only when, it reached the
#: Earth-Engine-availability gate: intent known, slots filled, field confirmed,
#: instrument down. It is the one honest signal that a clarify verdict is about
#: Earth Engine rather than about the conversation. See :func:`authorise`.
_EE_DOWN_CHECK = "Earth Engine measured as unavailable"
_EE_DOWN_REASON = "the field is confirmed but the measurement layer is down"


def _is_earth_engine_outage(action: policy_mod.Action) -> bool:
    """Is this clarify the availability gate's, or the conversation's?

    Both markers are ``dialog/policy.py``'s own words for the same branch; either
    one identifies it. If that branch were ever reworded past both, this returns
    ``False`` and the run is refused as "no question to run" -- wrong, but still
    a refusal whose type and message agree, which is the failure mode to prefer.
    """
    return _EE_DOWN_CHECK in action.checks or action.reason == _EE_DOWN_REASON


@dataclass(frozen=True)
class RunAuthorization:
    """Proof that SPEC section 4.4's gate was opened for this exact field.

    Constructing one outside :func:`authorise` raises. It carries the resolved
    point rather than a session reference so that what was authorised and what
    is queried cannot drift between the check and the call.
    """

    session_id: str
    intent: str
    analysis: str
    lat: float
    lon: float
    radius_m: float
    crop: str | None
    place: str | None
    end_date: dt.date | None
    end_date_note: str | None
    action: policy_mod.Action
    authorised_at: dt.datetime
    mint: Any = field(repr=False, default=None)

    def __post_init__(self) -> None:
        if self.mint is not _MINT:
            raise PermissionError(
                "a RunAuthorization is minted by cropup.web.dispatch.authorise() and "
                "nowhere else: SPEC 4.4's gate is not a field a caller may set"
            )
        if not self.action.authorises_earth_engine:
            raise PermissionError(
                f"{self.action.kind!r} does not authorise Earth Engine; "
                "only policy.next_action's run_analysis does"
            )
        # The geometry is the point *and* the buffer. The point is bounded by
        # SlotValue and the buffer by SlotBag, but this object is what actually
        # reaches ``ee.Geometry``, so it re-checks the radius here rather than
        # trusting that every path to it did.
        validate_field_radius_m(self.radius_m, source="the authorised field radius")

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "intent": self.intent,
            "analysis": self.analysis,
            "lat": self.lat,
            "lon": self.lon,
            "radius_m": self.radius_m,
            "crop": self.crop,
            "place": self.place,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "end_date_note": self.end_date_note,
            "authorised_at": self.authorised_at.isoformat(),
            "checks": list(self.action.checks),
            "reason": self.action.reason,
        }


@dataclass(frozen=True)
class RunOutcome:
    """One completed Earth Engine turn: the result, the rendering, the timings."""

    authorization: RunAuthorization
    result: Any
    answer: render.Answer
    legs: tuple[dict[str, Any], ...]
    elapsed_s: float
    budget_s: float
    #: Present when the gap list was too long to read and was summarised: every
    #: gap, with the raw error its source returned. See
    #: :func:`cropup.web.replies.condense_gaps`.
    diagnostics: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "analysis": self.authorization.analysis,
            "intent": self.authorization.intent,
            "authorization": self.authorization.as_dict(),
            "answer": self.answer.as_dict(),
            "legs": list(self.legs),
            "elapsed_s": round(self.elapsed_s, 2),
            "budget_s": self.budget_s,
            "result": self.result.as_dict(),
            "diagnostics": self.diagnostics,
        }


def planned_legs(intent: str) -> tuple[dict[str, Any], ...]:
    """The legs the orchestrator for ``intent`` will fan out to, with assets.

    ``status`` is ``"planned"`` on every row: this is the fan-out that is about
    to be started, not a claim that any of it has begun.
    """
    orchestrator = ORCHESTRATORS.get(intent)
    if orchestrator is None:
        return ()
    return tuple(
        {"name": leg, "status": "planned", "assets": list(orchestrator.assets(leg))}
        for leg in orchestrator.legs
    )


def run_budget_s(intent: str, settings: Settings | None = None) -> float:
    """The wall-clock budget for one run; ``0.0`` means no deadline.

    Derived, not picked: ``CROPUP_EE_REQUEST_TIMEOUT_S`` is a per-round-trip
    budget, and an orchestrator runs its legs ``CROPUP_EE_MAX_WORKERS`` at a
    time, so a fan-out of *n* legs takes ``ceil(n / workers)`` waves of round
    trips. Quoting the per-call budget for the whole fan-out would abandon a
    healthy run; quoting an arbitrary multiple would be a number nobody
    measured.
    """
    settings = settings or get_settings()
    per_call = float(settings.ee_request_timeout_s)
    if per_call <= 0.0:
        return 0.0
    legs = max(1, len(ORCHESTRATORS[intent].legs) if intent in ORCHESTRATORS else 1)
    workers = max(1, int(settings.ee_max_workers))
    return per_call * math.ceil(legs / workers)


def resolve_end_date(bag: SlotBag, *, today: dt.date | None = None) -> tuple[dt.date | None, str | None]:
    """The end date the timeframe slot implies, and any adjustment made to it.

    ``nlu/slots.py`` refuses to invent season boundaries, so a named season
    carries ``end=None`` and this returns ``None`` -- the orchestrator then uses
    its own documented window. A window that ends in the future is clamped to
    today, because there is no imagery after today, and the clamp is *returned*
    so the answer can say it happened rather than silently shifting the period
    the farmer asked about.
    """
    today = today or dt.date.today()
    slot = bag.get("timeframe")
    if slot is None:
        return None, None
    raw = slot.detail.get("end")
    if not isinstance(raw, str) or not raw.strip():
        return None, (
            f"{slot.label!r} has no dated boundary in the vocabulary, so the "
            "analysis used its own default window"
        )
    try:
        parsed = dt.date.fromisoformat(raw)
    except ValueError:
        return None, f"{slot.label!r} carried an unreadable end date ({raw!r}); it was not used"
    if parsed > today:
        return today, (
            f"{slot.label!r} ends {parsed.isoformat()}, which is in the future; "
            f"the window was clamped to {today.isoformat()}"
        )
    return parsed, None


def authorise(
    session: Session,
    *,
    intent: str | None = None,
    settings: Settings | None = None,
    earth_engine_available: bool | None = None,
) -> RunAuthorization:
    """Open SPEC section 4.4's gate, or raise.

    Raises :class:`~cropup.errors.ConfirmationRequired` when the farmer has not
    confirmed the field, and :class:`~cropup.errors.EarthEngineUnavailable` when
    the field is confirmed but the measurement layer is down. Every other
    verdict comes from ``policy.next_action`` unchanged.
    """
    settings = settings or get_settings()
    bag = session.bag

    if intent:
        # The form's "run this analysis" button is the farmer choosing the
        # question, so it is a user write and locks like one. It is written
        # before the frame is built so the policy sees what was asked for.
        if intent not in intents_mod.BY_NAME:
            raise ValueError(f"unknown intent {intent!r}")
        bag.set_intent(intent, origin=ORIGIN_USER)

    frame = policy_mod.Frame.from_bag(
        bag,
        text="",
        earth_engine_available=earth_engine_available,
        policy=policy_mod.PolicyConfig.from_settings(settings),
    )
    action = policy_mod.next_action(frame)

    if action.kind != policy_mod.ACTION_RUN_ANALYSIS or not action.authorises_earth_engine:
        if action.kind == policy_mod.ACTION_CLARIFY and _is_earth_engine_outage(action):
            # The gate is open; the instrument is not. That is a 503, not a
            # confirmation problem, and saying so is the difference between
            # "confirm your field again" and "the satellites are unreachable".
            #
            # The discriminator is the policy's own check line, not
            # ``frame.earth_engine_available``. ``policy.next_action`` reaches
            # clarify from several places and only appends that line at the
            # availability gate, which it reaches only after the intent, the
            # slots and the confirmation have all passed. Keying on the frame
            # instead meant that while Earth Engine was down, *every* clarify --
            # including "I am not sure what you are asking" on a session with no
            # intent at all -- was reported as ``EarthEngineUnavailable`` with
            # the clarify prompt as its message: a 503 whose text was about
            # something else entirely.
            raise EarthEngineUnavailable(action.prompt or action.reason)
        if action.kind == policy_mod.ACTION_CLARIFY:
            # Nothing to run: the policy could not name an analysis. Not a
            # confirmation problem and not an outage, so it is neither of those
            # exceptions -- it is a request this session cannot serve yet, and
            # the message says exactly that.
            raise ValueError(
                "there is no question to run on this session yet: "
                f"{action.prompt or action.reason} "
                "Set the question with POST /api/session/{sid}/slots "
                '{"intent": "..."}, confirm the field, then run.'
            )
        # One implementation of the rule: let the policy raise its own exception.
        policy_mod.authorise_earth_engine(frame)
        raise AssertionError("policy.authorise_earth_engine returned for a non-run action")

    name = action.intent or bag.intent
    if name not in ORCHESTRATORS:
        raise ValueError(
            f"intent {name!r} is routed to Earth Engine but has no orchestrator "
            f"(have: {', '.join(EE_INTENTS)})"
        )

    location = bag.get("location")
    coords = location.coordinates() if location is not None else None
    if coords is None:  # policy already required this; belt and braces at the door
        raise ValueError("the confirmed location carries no coordinates")
    lat, lon = coords
    end_date, note = resolve_end_date(bag)
    radius_m = _run_radius(bag, settings)

    return RunAuthorization(
        session_id=session.session_id,
        intent=name,
        analysis=ORCHESTRATORS[name].analysis,
        lat=lat,
        lon=lon,
        radius_m=radius_m,
        crop=bag.value("crop"),
        place=location.label if location is not None else None,
        end_date=end_date,
        end_date_note=note,
        action=action,
        authorised_at=dt.datetime.now(dt.timezone.utc),
        mint=_MINT,
    )


def _run_radius(bag: SlotBag, settings: Settings) -> float:
    """The radius that will really be buffered, and the bound it has to clear.

    Two sources, one check. The session's radius is already bounded by
    ``SlotBag.set_field_radius`` and is part of the confirmed location's
    identity, so a farmer who confirmed a 15 m field cannot have it widened
    underneath them -- widening it withdraws the confirmation and this function
    is never reached. The *fallback* is the reason the check is repeated here:
    ``config.Settings`` only requires ``CROPUP_FIELD_RADIUS_M`` to be positive
    and finite, so a mistyped deployment variable could otherwise put a
    1,000 km disc into the geometry with no farmer involved at all. Both go
    through ``dialog.slots.validate_field_radius_m``, which is the one
    definition of "that is not a field".
    """
    chosen = bag.field_radius_m
    if chosen is not None:
        return validate_field_radius_m(chosen, source="the field radius set on this session")
    return validate_field_radius_m(settings.field_radius_m, source="CROPUP_FIELD_RADIUS_M")


def _call(auth: RunAuthorization, settings: Settings) -> Any:
    """Invoke the orchestrator for this authorisation. The one EE entry point."""
    orchestrator = ORCHESTRATORS[auth.intent]
    kwargs: dict[str, Any] = {"settings": settings}
    if auth.analysis == "crop_selection":
        # crop_selection ranks the catalogue at a point; it takes no radius and
        # no end date, and passing either would be inventing an argument.
        return orchestrator.call(auth.lat, auth.lon, auth.crop, **kwargs)
    kwargs["radius_m"] = auth.radius_m
    if auth.end_date is not None:
        kwargs["end_date"] = auth.end_date
    return orchestrator.call(auth.lat, auth.lon, auth.crop, **kwargs)


def run(
    auth: RunAuthorization,
    *,
    hub: ProgressHub | None = None,
    settings: Settings | None = None,
) -> RunOutcome:
    """Run the authorised analysis, stream its progress, render its answer.

    ``auth`` is the whole precondition: this function never looks at a session
    or a bag, so there is no state it could re-read and disagree with.

    Raises ``TimeoutError`` when the run outlives :func:`run_budget_s`, and
    whatever the orchestrator raises otherwise -- ``geo/`` records a masked
    pixel as a :class:`~cropup.evidence.Missing`, so an exception here is a real
    failure and is reported as one.
    """
    if not isinstance(auth, RunAuthorization):  # pragma: no cover - the type is the gate
        raise PermissionError(
            "run() takes a RunAuthorization from authorise(); a latitude and a "
            "longitude are not an authorisation (SPEC 4.4)"
        )
    settings = settings or get_settings()
    budget = run_budget_s(auth.intent, settings)
    sid = auth.session_id
    started = time.perf_counter()

    if hub is not None:
        hub.publish(
            sid,
            EVENT_RUN_PLANNED,
            {
                "intent": auth.intent,
                "analysis": auth.analysis,
                "legs": list(planned_legs(auth.intent)),
                "budget_s": budget,
                "field": {
                    "lat": auth.lat,
                    "lon": auth.lon,
                    "radius_m": auth.radius_m,
                    "label": auth.place,
                },
                "crop": auth.crop,
                "note": (
                    "legs are listed as planned; each one reports its measured "
                    "wall time in its own 'leg' event when the fan-out joins"
                ),
            },
        )

    deadline = bootstrap.Deadline.for_request(
        budget if budget > 0 else None,
        label=f"{auth.analysis} for session {sid}",
        settings=settings,
    )
    heartbeat = _Heartbeat(hub, sid, auth, budget)
    try:
        heartbeat.start()
        result = bootstrap.call_with_deadline(
            lambda: _call(auth, settings),
            deadline=deadline if budget > 0 else None,
            timeout_s=None if budget > 0 else 0.0,
            label=f"{auth.analysis} run",
            settings=settings,
        )
    except BaseException as exc:
        heartbeat.stop()
        if hub is not None:
            hub.publish(
                sid,
                EVENT_RUN_FAILED,
                {
                    "intent": auth.intent,
                    "analysis": auth.analysis,
                    "error": type(exc).__name__,
                    "detail": str(exc),
                    "elapsed_s": round(time.perf_counter() - started, 2),
                    "budget_s": budget,
                },
            )
        raise
    heartbeat.stop()

    legs = tuple(leg.as_dict() for leg in getattr(result, "legs", ()) or ())
    if hub is not None:
        for leg in legs:
            hub.publish(sid, EVENT_LEG, {"intent": auth.intent, "status": "finished", **leg})

    answer = render.analysis_answer(
        result,
        crop=auth.crop,
        place=auth.place,
        settings=settings,
    )
    # A run with every source down names ~92 gaps, one rendered line each,
    # each ending in the exception its source raised. That is a diagnostic, not
    # a reply, so it is summarised for the farmer and kept in full here.
    answer, diagnostics = replies.condense_gaps(answer)
    elapsed = time.perf_counter() - started

    if hub is not None:
        hub.publish(
            sid,
            EVENT_RUN_FINISHED,
            {
                "intent": auth.intent,
                "analysis": auth.analysis,
                "elapsed_s": round(elapsed, 2),
                "budget_s": budget,
                "legs_ok": sum(1 for leg in legs if leg.get("ok")),
                "legs_total": len(legs),
                "facts": len(answer.facts_used()),
                "gaps": len(answer.gaps_named()),
                "degraded": bool((answer.degradation or {}).get("degraded")),
            },
        )

    return RunOutcome(
        authorization=auth,
        result=result,
        answer=answer,
        legs=legs,
        elapsed_s=elapsed,
        budget_s=budget,
        diagnostics=diagnostics,
    )


class _Heartbeat:
    """Emits a clock reading every few seconds while a run is in flight.

    It reports elapsed wall time and the budget, and nothing else: it has no way
    to know which leg is where, and a progress bar that invents a percentage is
    the same fabrication as a default pH.
    """

    INTERVAL_S = 2.0

    def __init__(
        self, hub: ProgressHub | None, session_id: str, auth: RunAuthorization, budget_s: float
    ) -> None:
        self._hub = hub
        self._sid = session_id
        self._auth = auth
        self._budget = budget_s
        self._stop = None
        self._thread = None

    def start(self) -> None:
        if self._hub is None:
            return
        self._stop = threading.Event()
        started = time.perf_counter()

        def tick() -> None:
            while not self._stop.wait(self.INTERVAL_S):
                self._hub.publish(
                    self._sid,
                    EVENT_PROGRESS,
                    {
                        "intent": self._auth.intent,
                        "analysis": self._auth.analysis,
                        "elapsed_s": round(time.perf_counter() - started, 1),
                        "budget_s": self._budget,
                        "state": "in_flight",
                        "note": "elapsed wall time; no leg has reported yet",
                    },
                )

        self._thread = threading.Thread(
            target=tick, name=f"cropup-progress-{self._sid}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        self._thread = None
