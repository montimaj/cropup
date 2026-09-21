"""In-process session store: one :class:`~cropup.dialog.slots.SlotBag` per sid.

SPEC section 8 gives the browser a session id and then lets it come back to the
same frame after a reload. That is all this module is: a dict of live
:class:`Session` objects, a TTL taken from ``CROPUP_SESSION_TTL_S``, and an
eviction sweep so a long-running process does not accumulate every farmer who
ever opened the page.

**The rehydration rule, and why it has its own function.**

``SlotBag.from_dict`` writes straight into ``_values``. That is correct for the
store's own snapshots -- it is how a bag survives a round trip through JSON --
but it means a payload that says ``{"confirmed": true}`` produces a bag whose
location reads as confirmed without the farmer ever having confirmed anything.
``dialog.policy.next_action`` reads exactly that flag to decide whether SPEC
section 4.4's gate is open, so a client that could post a bag could mint its own
authorisation to spend Earth Engine quota on a field nobody endorsed.

So nothing arriving over HTTP is ever handed to ``SlotBag.from_dict`` as-is.
:func:`sanitise_bag_payload` strips the two flags a client must not be able to
assert -- ``confirmed`` and its timestamp -- and recomputes ``locked`` from the
origin the same way ``SlotBag.set`` does, so a restored bag is exactly a bag
whose slots were written through the normal path. Confirmation is then something
only ``POST /api/session/{sid}/confirm`` can produce, in this process, against a
bag this process is holding.

Imports: ``cropup.config``, ``cropup.dialog`` and the standard library. No
Earth Engine, no model, no network.
"""

from __future__ import annotations

import datetime as dt
import secrets
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

from ..config import Settings, get_settings
from ..errors import CropUpError
from ..dialog.slots import (
    BAG_SLOTS,
    OUTCOME_REFUSED_LOCKED,
    OUTCOME_SET,
    OUTCOME_UNCHANGED,
    LOCKING_ORIGINS,
    ORIGINS,
    ORIGIN_GAZETTEER,
    ORIGIN_GPS,
    ORIGIN_NLU,
    ORIGIN_USER,
    SlotBag,
)

__all__ = [
    "SessionNotFound",
    "SessionCapacityReached",
    "Session",
    "SessionStore",
    "sanitise_bag_payload",
    "DEFAULT_MAX_SESSIONS",
    "DEFAULT_EVICTION_GRACE_S",
    "DEFAULT_MIN_EVICTION_GRACE_S",
]

#: A cap on the number of live sessions, so an unattended process cannot be
#: grown without bound by opening sessions. It is a bounded cache, not a
#: database -- but see :meth:`SessionStore.open` for *which* session is dropped
#: when it is full, which is the part that matters.
DEFAULT_MAX_SESSIONS = 1000

#: How long a session that holds something must have been idle before a new,
#: unrelated session may take its place.
#:
#: ``POST /api/session`` needs no credential, so without this the eviction rule
#: was itself the attack: 1,000 anonymous opens dropped the 1,000 oldest-idle
#: sessions, and "oldest idle" includes a farmer who is reading the answer they
#: just received. Five minutes is the reading time this protects; a conversation
#: idle longer than that is recycled, and a conversation shorter than that is
#: never destroyed to make room for a stranger.
DEFAULT_EVICTION_GRACE_S = 300.0

#: The floor the grace above is allowed to fall to under sustained pressure, and
#: the whole of :meth:`SessionStore._effective_grace_s`.
#:
#: **The tradeoff, stated rather than hidden.** Protecting every in-use session
#: for a fixed five minutes traded one denial of service for another: a flood
#: that writes a single slot into each of its 1,000 sessions makes all of them
#: "in use", and every real farmer arriving afterwards is refused a *new*
#: conversation for up to the full grace. Nothing in SPEC section 7 lets this be
#: fixed by identifying who is asking -- no cookie, no client address, no
#: fingerprint -- so the only lever left is the grace itself.
#:
#: So the grace is pressure-adaptive: each refusal halves it, down to this
#: floor, and it snaps back to the full value as soon as the store is not full.
#: What degrades is the *reading-time guarantee* for the least recently seen
#: conversation, and only while the store is full: at the floor, a conversation
#: idle for 30 seconds may be recycled to admit a newcomer, where normally it
#: has five minutes. That is a bounded, named degradation -- it is reported in
#: :meth:`SessionStore.status`, which ``/api/health`` and ``/api/capabilities``
#: both publish, and in the refusal itself -- rather than a silent one.
#:
#: It also raises the cost of the flood by the same factor it lowers the
#: guarantee: to keep a store of capacity *n* full at the floor, every one of
#: those sessions must be touched every 30 seconds instead of every 300, which
#: is ten times the traffic for the same denial, and a flood that goes quiet is
#: evicting itself within the floor rather than within five minutes.
DEFAULT_MIN_EVICTION_GRACE_S = 30.0

_SID_BYTES = 16


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class SessionNotFound(KeyError):
    """No live session with that id: never opened, or evicted after its TTL."""

    def __init__(self, session_id: str, detail: str = "") -> None:
        self.session_id = session_id
        self.detail = detail or "no such session; it was never opened or its TTL has expired"
        super().__init__(self.detail)

    def __str__(self) -> str:
        # KeyError.__str__ reprs its argument, which would put the message in
        # quotes on the way out to the client. This is a sentence, not a key.
        return self.detail


class SessionCapacityReached(CropUpError):
    """Every live session is in use, so this process will not open another.

    Refusing a *new* conversation is the honest failure here. The alternative --
    the old behaviour -- was to evict the least recently seen session, which
    meant an anonymous client could end somebody else's conversation simply by
    opening enough of its own. A newcomer waiting is recoverable; a farmer
    losing a confirmed field mid-turn is not.
    """

    def __init__(
        self,
        live: int,
        capacity: int,
        retry_after_s: float,
        *,
        grace_s: float | None = None,
        degraded: bool = False,
    ) -> None:
        self.live = int(live)
        self.capacity = int(capacity)
        self.retry_after_s = float(retry_after_s)
        #: The grace actually in force when this refusal happened, which is not
        #: the configured one once the store has been under pressure.
        self.grace_s = float(grace_s) if grace_s is not None else None
        #: True when that grace has been shortened by pressure. The client is
        #: told, because a silently shortened guarantee is the thing this
        #: mitigation is not allowed to be.
        self.degraded = bool(degraded)
        message = (
            f"all {self.capacity} sessions are in active use, so no new session was "
            f"opened; none has been idle long enough to reuse. Retry in about "
            f"{self.retry_after_s:.0f}s, or keep using the session you already have"
        )
        if self.degraded and self.grace_s is not None:
            message += (
                f". This process is under session pressure, so the idle time a "
                f"conversation is protected for has been shortened to "
                f"{self.grace_s:.0f}s"
            )
        super().__init__(message)


def _iso_or_none(value: Any) -> str | None:
    """``value`` when it is an ISO timestamp the bag can parse, else ``None``."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    return value


def sanitise_bag_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """A client-supplied bag snapshot, stripped of everything it may not assert.

    Returns a payload safe to hand to :meth:`SlotBag.from_dict`:

    * ``confirmed`` is forced to ``False`` and ``confirmed_at`` to ``None`` on
      every slot. SPEC section 4.4's gate is opened by an act in this process,
      not by a field in a request body.
    * ``locked`` is recomputed from ``origin`` exactly as ``SlotBag.set`` does,
      so a restored slot cannot claim a lock its origin does not earn.
    * an unknown ``origin`` or an unknown slot name is dropped rather than
      guessed at, and the session-level ``confirmed``/``locked`` summaries are
      dropped because they are derived views, not inputs.
    * ``events`` are dropped: the journal is this process's record of what it
      did, and a client cannot add to it.

    Everything else -- the value, the label, the NLU confidence, the
    alternatives, the ``needs_confirmation`` flag, the coordinates in ``detail``
    -- travels, because those are the suggestion the farmer is being shown and
    losing them would make a reload look like a fresh conversation.
    """
    out: dict[str, Any] = {}
    if payload.get("session_id") is not None:
        out["session_id"] = str(payload["session_id"])
    for key in ("created_at", "updated_at"):
        stamp = _iso_or_none(payload.get(key))
        if stamp is not None:
            out[key] = stamp
    radius = payload.get("field_radius_m")
    if isinstance(radius, (int, float)) and not isinstance(radius, bool) and radius > 0:
        out["field_radius_m"] = float(radius)

    slots: dict[str, Any] = {}
    # ``payload`` is typed by ``server.RestoreIn`` before it gets here, but this
    # function is exported and is the documented sanitiser, so it does not
    # assume its caller validated anything: a ``slots`` that is a list, a string
    # or a number is not a mapping of slots and is dropped, not iterated.
    raw_slots = payload.get("slots")
    if not isinstance(raw_slots, Mapping):
        raw_slots = {}
    for name, value in raw_slots.items():
        if not isinstance(value, Mapping):
            continue
        slot = str(value.get("slot") or name)
        if slot not in BAG_SLOTS:
            continue
        text = value.get("value")
        if not isinstance(text, str) or not text.strip():
            continue
        origin = str(value.get("origin") or ORIGIN_NLU)
        if origin not in ORIGINS:
            origin = ORIGIN_NLU
        cleaned = dict(value)
        cleaned.update(
            {
                "slot": slot,
                "value": text,
                "origin": origin,
                # The two flags a client may not assert, and the lock that is a
                # function of the origin rather than a claim of its own.
                "confirmed": False,
                "confirmed_at": None,
                "locked": origin in LOCKING_ORIGINS,
            }
        )
        # A malformed timestamp would make SlotValue.from_dict raise; the write
        # time of a restored suggestion is not worth failing a reload over, so
        # an unusable one is dropped and the bag stamps "now" instead.
        if _iso_or_none(cleaned.get("set_at")) is None:
            cleaned.pop("set_at", None)
        # Same rule for the three fields ``SlotValue.from_dict`` reads
        # structurally: a wrong shape is dropped, never coerced and never
        # allowed through to raise. ``{"alternatives": 5}`` is not "no
        # alternatives" to a client, but it is not an alternative either, and a
        # reload is not worth a 500.
        if not isinstance(cleaned.get("detail"), Mapping):
            cleaned.pop("detail", None)
        alternatives = cleaned.get("alternatives")
        if not isinstance(alternatives, (list, tuple)) or not all(
            isinstance(item, str) for item in alternatives
        ):
            cleaned.pop("alternatives", None)
        confidence = cleaned.get("confidence")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            cleaned.pop("confidence", None)
        slots[slot] = cleaned
    out["slots"] = slots
    return out


@dataclass
class Session:
    """One browser tab's state: the shared frame plus what this process did with it.

    ``bag`` is the single :class:`SlotBag` SPEC section 1.2 calls the one shared
    state; the chat handler, the questionnaire handler and the map pin all write
    into this one object. ``lock`` serialises the turns of a single session so
    two concurrent posts cannot interleave inside the bag's write path.
    """

    session_id: str
    bag: SlotBag
    created_at: dt.datetime = field(default_factory=_now)
    last_seen_at: dt.datetime = field(default_factory=_now)
    #: Rendered replies this session has produced, newest last, bounded.
    turns: list[dict[str, Any]] = field(default_factory=list)
    #: The last Earth Engine run's envelope, so a reload can show the answer again.
    last_run: dict[str, Any] | None = None
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    MAX_TURNS = 40

    def touch(self, at: dt.datetime | None = None) -> None:
        self.last_seen_at = at or _now()

    def record_turn(self, turn: Mapping[str, Any]) -> None:
        self.turns.append(dict(turn))
        del self.turns[: -self.MAX_TURNS]

    @property
    def in_use(self) -> bool:
        """Has anything happened in this session since it was opened?

        A session with a filled slot, a rendered turn or a completed run is
        somebody's conversation. One with none of those is a session id and
        nothing else -- a closed tab, a probe, or one of a flood -- and
        :meth:`SessionStore._evictable` recycles it without ceremony.
        """
        return bool(
            self.turns
            or self.last_run is not None
            or self.bag.filled()
            or self.bag.intent
            or self.bag.field_radius_m is not None
        )

    def expired(self, ttl_s: float, now: dt.datetime | None = None) -> bool:
        if ttl_s <= 0:
            return False
        return ((now or _now()) - self.last_seen_at).total_seconds() > ttl_s

    def as_dict(self, *, include_turns: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "bag": self.bag.to_dict(),
            "last_run": self.last_run,
        }
        if include_turns:
            payload["turns"] = list(self.turns)
        return payload


class SessionStore:
    """Live sessions, keyed by sid, with a TTL and a bounded population.

    In-process on purpose: a SlotBag is a conversation, not a record, and
    SPEC section 7 forbids persisting what the farmer said. Restarting the
    process ends every conversation, which is why the browser holds its own copy
    and can restore it through :func:`sanitise_bag_payload`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        eviction_grace_s: float = DEFAULT_EVICTION_GRACE_S,
        min_eviction_grace_s: float = DEFAULT_MIN_EVICTION_GRACE_S,
        on_evict: Callable[[str], None] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._max = max(1, int(max_sessions))
        self._grace_s = max(0.0, float(eviction_grace_s))
        self._min_grace_s = min(self._grace_s, max(0.0, float(min_eviction_grace_s)))
        #: How many times in a row a newcomer has been refused because every
        #: conversation was inside its grace. Each one halves the grace for the
        #: next attempt; it decays back one halving per quiet grace period, and
        #: a successful open with room to spare clears it outright.
        self._pressure = 0
        #: When the last refusal happened, which is what the decay is measured
        #: from. A clock reading, not a client.
        self._pressure_at: dt.datetime | None = None
        #: Refusals since this process started, for ``/api/health``. A counter,
        #: not a log: it holds no farmer text and nothing identifying (SPEC 7).
        self._refusals = 0
        #: Called with each evicted sid. The web app passes
        #: ``ProgressHub.forget`` so an evicted session's SSE backlog goes with
        #: it; without that, the one bounded structure would keep the other
        #: growing.
        self._on_evict = on_evict
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()

    def _evicted(self, session_id: str) -> None:
        if self._on_evict is None:
            return
        try:
            self._on_evict(session_id)
        except Exception:  # noqa: BLE001 - housekeeping must not fail a request
            pass

    # -- lifecycle ----------------------------------------------------------

    @property
    def ttl_s(self) -> int:
        return int(self._settings.session_ttl_s)

    def _decayed_pressure(self, now: dt.datetime | None = None) -> int:
        """Recorded pressure, less one halving for each quiet grace period.

        Without the decay the shortened guarantee would be permanent: the store
        stays full long after a flood stops (a session lives for the TTL), so
        "reset when there is room to spare" alone would never fire. Pressure is
        a statement about *now*, and it expires like one.
        """
        if self._pressure <= 0:
            return 0
        if self._pressure_at is None or self._grace_s <= 0:
            return self._pressure
        quiet_for = ((now or _now()) - self._pressure_at).total_seconds()
        return max(0, self._pressure - int(quiet_for // self._grace_s))

    def _effective_grace_s(self, now: dt.datetime | None = None) -> float:
        """The grace in force right now: halved per refusal, floored, never zero-ed.

        Caller holds ``self._lock`` (or is only reporting).
        """
        pressure = self._decayed_pressure(now)
        if pressure <= 0:
            return self._grace_s
        return max(self._min_grace_s, self._grace_s / float(2**pressure))

    def _evictable(self, now: dt.datetime) -> Session | None:
        """The session a newcomer may take the place of, or ``None``.

        Two tiers, and the order is the whole mitigation:

        1. a session nobody has used since it was opened -- empty bag, no turn,
           no run. That is what an anonymous flood produces, and what a browser
           that opened a tab and closed it leaves behind. It is recycled
           immediately, oldest first, so a flood evicts *itself*.
        2. otherwise the least recently seen session that holds something, and
           only once it has been idle for :meth:`_effective_grace_s` -- the
           configured grace, shortened towards
           :data:`DEFAULT_MIN_EVICTION_GRACE_S` while newcomers are being
           refused. A flood that writes one slot per session defeats tier 1 (a
           written slot is "in use"), so tier 2 is what has to give, and it
           gives by a stated amount rather than by locking everybody out for the
           full five minutes.

        Anything else is refused, because the caller of this method is somebody
        who has no conversation yet and the session it would destroy belongs to
        somebody who does. Caller holds ``self._lock``.
        """
        idle: list[Session] = []
        for session in self._sessions.values():
            if not session.in_use:
                idle.append(session)
        if idle:
            return min(idle, key=lambda s: s.last_seen_at)
        oldest = min(self._sessions.values(), key=lambda s: s.last_seen_at)
        if (now - oldest.last_seen_at).total_seconds() >= self._effective_grace_s(now):
            return oldest
        return None

    def open(
        self,
        *,
        restore: Mapping[str, Any] | None = None,
        field_radius_m: float | None = None,
    ) -> Session:
        """Open a session. ``restore`` is a client snapshot and is sanitised.

        The new sid is always minted here: a client does not choose its own
        session id, and a restored snapshot's ``session_id`` is discarded.

        Raises :class:`SessionCapacityReached` when the store is full of
        conversations that are actually in use; see :meth:`_evictable`.
        """
        self.sweep()
        sid = secrets.token_urlsafe(_SID_BYTES)
        if restore:
            payload = sanitise_bag_payload(restore)
            payload["session_id"] = sid
            bag = SlotBag.from_dict(payload)
        else:
            bag = SlotBag(session_id=sid)
        bag.session_id = sid
        if field_radius_m is not None:
            bag.set_field_radius(field_radius_m)
        session = Session(session_id=sid, bag=bag)
        dropped: list[str] = []
        now = _now()
        with self._lock:
            while len(self._sessions) >= self._max:
                victim = self._evictable(now)
                if victim is None:
                    live = len(self._sessions)
                    oldest = min(self._sessions.values(), key=lambda s: s.last_seen_at)
                    grace = self._effective_grace_s(now)
                    wait = grace - (now - oldest.last_seen_at).total_seconds()
                    # Record the pressure *before* raising, so the next newcomer
                    # is measured against a shorter grace than this one was.
                    # Bounded by the floor, so this stops mattering quickly.
                    self._refusals += 1
                    self._pressure = self._decayed_pressure(now)
                    if grace > self._min_grace_s:
                        self._pressure += 1
                    self._pressure_at = now
                    raise SessionCapacityReached(
                        live,
                        self._max,
                        max(1.0, wait),
                        grace_s=grace,
                        degraded=grace < self._grace_s,
                    )
                self._sessions.pop(victim.session_id, None)
                dropped.append(victim.session_id)
            self._sessions[sid] = session
            if len(self._sessions) < self._max:
                # Room to spare: whatever pressure there was is over, and the
                # full reading-time guarantee comes back immediately.
                self._pressure = 0
                self._pressure_at = None
        for gone in dropped:
            self._evicted(gone)
        return session

    def get(self, session_id: str) -> Session:
        """The live session, touched. Raises :class:`SessionNotFound`."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFound(session_id)
            if session.expired(self.ttl_s):
                self._sessions.pop(session_id, None)
                self._evicted(session_id)
                raise SessionNotFound(
                    session_id, f"session expired after {self.ttl_s}s of inactivity"
                )
            session.touch()
            return session

    def peek(self, session_id: str) -> Session | None:
        """The session without touching it; ``None`` when absent or expired."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.expired(self.ttl_s):
                return None
            return session

    def close(self, session_id: str) -> bool:
        with self._lock:
            gone = self._sessions.pop(session_id, None) is not None
        if gone:
            self._evicted(session_id)
        return gone

    def sweep(self, now: dt.datetime | None = None) -> tuple[str, ...]:
        """Drop expired sessions. Returns the ids evicted."""
        ttl = self.ttl_s
        when = now or _now()
        with self._lock:
            dead = [sid for sid, s in self._sessions.items() if s.expired(ttl, when)]
            for sid in dead:
                self._sessions.pop(sid, None)
        for sid in dead:
            self._evicted(sid)
        return tuple(dead)

    def clear(self) -> None:
        with self._lock:
            gone = list(self._sessions)
            self._sessions.clear()
        for sid in gone:
            self._evicted(sid)

    # -- reading ------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def __contains__(self, session_id: object) -> bool:
        return isinstance(session_id, str) and self.peek(session_id) is not None

    def __iter__(self) -> Iterator[Session]:
        with self._lock:
            return iter(list(self._sessions.values()))

    def status(self) -> dict[str, Any]:
        """Counts for /api/health; holds no farmer text (SPEC section 7).

        ``eviction_grace_degraded`` is the honest half of the mitigation
        described at :data:`DEFAULT_MIN_EVICTION_GRACE_S`: when it is true, an
        in-use conversation is being protected for less idle time than
        ``eviction_grace_s`` promises, and the capability strip can say so.
        """
        with self._lock:
            effective = self._effective_grace_s()
            pressure = self._decayed_pressure()
            return {
                "live": len(self._sessions),
                "in_use": sum(1 for s in self._sessions.values() if s.in_use),
                "capacity": self._max,
                "ttl_s": self.ttl_s,
                "eviction_grace_s": self._grace_s,
                "effective_eviction_grace_s": effective,
                "min_eviction_grace_s": self._min_grace_s,
                "eviction_grace_degraded": effective < self._grace_s,
                "eviction_pressure": pressure,
                "capacity_refusals": self._refusals,
            }


def initial_writes(
    bag: SlotBag,
    values: Mapping[str, Any],
    *,
    origin: str = ORIGIN_USER,
) -> dict[str, str]:
    """Write plain ``slot -> value`` pairs into a bag, honouring the lock rule.

    Used by ``POST /api/session`` and ``POST /api/session/{sid}/slots``: the
    questionnaire and the map send values, never SlotValue structures, so there
    is nothing here a client could use to assert a flag.
    """
    outcomes: dict[str, str] = {}
    for slot, raw in values.items():
        if slot not in BAG_SLOTS or raw is None:
            continue
        before = bag.get(slot)
        text = str(raw).strip()
        if not text:
            if before is not None:
                bag.clear(slot)
                outcomes[slot] = "cleared"
            continue
        stored = bag.set(slot, text, origin=origin)
        if stored is None:
            outcomes[slot] = OUTCOME_REFUSED_LOCKED
        elif before is not None and before.value == stored.value:
            outcomes[slot] = OUTCOME_UNCHANGED
        else:
            outcomes[slot] = OUTCOME_SET
    return outcomes


#: Origins a client is allowed to name on a slot write. ``nlu_suggestion`` is
#: deliberately not on the list: only this process's own classifier may claim to
#: have heard something, and ``gazetteer`` is written by the place picker.
CLIENT_ORIGINS: tuple[str, ...] = (ORIGIN_USER, ORIGIN_GPS, ORIGIN_GAZETTEER)


def resolve_origin(requested: str | None) -> str:
    """The origin a slot write may claim, defaulting to the farmer's own."""
    if requested is None:
        return ORIGIN_USER
    value = str(requested).strip()
    if value not in CLIENT_ORIGINS:
        raise ValueError(
            f"origin {requested!r} may not be set by a client "
            f"(allowed: {', '.join(CLIENT_ORIGINS)})"
        )
    return value


def sequence_of(value: Any) -> tuple[str, ...]:
    """A tolerant reader for a JSON list-of-strings field."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(v) for v in value if str(v).strip())
    return ()
