"""Server-sent events, one stream per session (SPEC sections 3.4 and 8).

The vendored ``plant_health`` took 28.7 s. A farmer staring at a blank panel for
half a minute has no way to tell a slow satellite from a broken app, so
``GET /api/session/{sid}/events`` streams what the run is doing while it does
it.

**What this stream promises, precisely.** The orchestrators fan their legs out
with ``concurrent.futures`` and report each :class:`~cropup.analysis.rules.Leg`
with its own measured wall time once the fan-out has joined. This module does
not pretend to know more than that:

* ``run_planned`` names the legs the chosen orchestrator will run, and says so
  -- ``status: "planned"``. It is read off the orchestrator's own
  ``_LEG_ASSETS`` table rather than copied, so it cannot drift from the code.
* ``progress`` is a heartbeat carrying the elapsed wall time of the run. It is
  a clock reading, not a claim about any leg's state.
* ``leg`` is one event per leg **with the leg's real measurements** --
  ``ok``, ``elapsed_s``, ``facts``, ``gaps``, the assets it read -- emitted as
  soon as the result carries them.
* ``run_finished`` or ``run_failed`` closes the run, and ``run_failed`` names
  the reason rather than a generic failure.

Nothing here invents a per-leg start time it did not observe, and no event
carries a rendered number: the numbers reach the farmer through
``render/templates.py`` and nowhere else (SPEC section 4.2).

Publishing happens on whatever thread the run is on; delivery happens on the
event loop. Each subscriber therefore carries the loop it belongs to, and a
publish hops onto that loop with ``call_soon_threadsafe``. A bounded backlog is
kept per session so a browser that connects after the run started -- or
reconnects with ``Last-Event-ID`` -- still sees what it missed.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping

__all__ = [
    "EVENT_OPEN",
    "EVENT_RUN_PLANNED",
    "EVENT_PROGRESS",
    "EVENT_LEG",
    "EVENT_RUN_FINISHED",
    "EVENT_RUN_FAILED",
    "EVENT_SLOTS",
    "EVENT_HEARTBEAT",
    "EVENT_NAMES",
    "ProgressEvent",
    "ProgressHub",
    "sse_format",
]

EVENT_OPEN = "open"  # the stream is live; carries the session id and the backlog cursor
EVENT_RUN_PLANNED = "run_planned"  # which legs this run will fan out to
EVENT_PROGRESS = "progress"  # a clock reading while the fan-out is in flight
EVENT_LEG = "leg"  # one leg, with its measured outcome
EVENT_RUN_FINISHED = "run_finished"
EVENT_RUN_FAILED = "run_failed"
EVENT_SLOTS = "slots"  # the bag changed (a second tab, the map, the form)
EVENT_HEARTBEAT = "heartbeat"  # keeps a proxy from closing an idle stream

EVENT_NAMES: tuple[str, ...] = (
    EVENT_OPEN,
    EVENT_RUN_PLANNED,
    EVENT_PROGRESS,
    EVENT_LEG,
    EVENT_RUN_FINISHED,
    EVENT_RUN_FAILED,
    EVENT_SLOTS,
    EVENT_HEARTBEAT,
)

#: How many events a session keeps for a late or reconnecting subscriber.
BACKLOG = 200
#: How many events one subscriber may fall behind by before it is dropped. A
#: browser that stopped reading is a closed tab, not a reason to grow a queue.
SUBSCRIBER_QUEUE = 256


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class ProgressEvent:
    """One SSE frame: a monotonically numbered, named, JSON-carrying event."""

    id: int
    name: str
    data: Mapping[str, Any]
    at: str = field(default_factory=_now_iso)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "event": self.name, "at": self.at, **dict(self.data)}


def sse_format(event: ProgressEvent) -> str:
    """The wire form. ``id:`` is what a browser replays with ``Last-Event-ID``."""
    body = json.dumps(event.as_dict(), default=str)
    return f"id: {event.id}\nevent: {event.name}\ndata: {body}\n\n"


@dataclass
class _Subscriber:
    queue: asyncio.Queue
    loop: asyncio.AbstractEventLoop


class ProgressHub:
    """Fan-out of progress events, keyed by session id.

    Thread-safe: :meth:`publish` may be called from the worker thread a run is
    executing on, while :meth:`stream` is being consumed on the event loop.
    """

    def __init__(self, *, backlog: int = BACKLOG) -> None:
        self._backlog = max(1, int(backlog))
        self._history: dict[str, deque[ProgressEvent]] = {}
        self._subscribers: dict[str, list[_Subscriber]] = {}
        self._next_id: dict[str, int] = {}
        self._lock = threading.RLock()

    # -- publishing ---------------------------------------------------------

    def publish(self, session_id: str, name: str, data: Mapping[str, Any]) -> ProgressEvent:
        """Emit one named event to every live subscriber and to the backlog."""
        if name not in EVENT_NAMES:
            raise ValueError(f"unknown progress event {name!r} (known: {', '.join(EVENT_NAMES)})")
        with self._lock:
            number = self._next_id.get(session_id, 0) + 1
            self._next_id[session_id] = number
            event = ProgressEvent(id=number, name=name, data=dict(data))
            history = self._history.setdefault(session_id, deque(maxlen=self._backlog))
            history.append(event)
            targets = list(self._subscribers.get(session_id, ()))
        for subscriber in targets:
            self._deliver(subscriber, event)
        return event

    def _deliver(self, subscriber: _Subscriber, event: ProgressEvent) -> None:
        def push() -> None:
            try:
                subscriber.queue.put_nowait(event)
            except asyncio.QueueFull:
                # A reader that is this far behind is gone. Dropping the event
                # is better than growing without bound; the backlog still has it
                # for a reconnect with Last-Event-ID.
                pass

        try:
            subscriber.loop.call_soon_threadsafe(push)
        except RuntimeError:
            # The loop this subscriber belonged to has closed: its stream is
            # over and its own finally block will unsubscribe.
            pass

    # -- subscribing --------------------------------------------------------

    def _subscribe(self, session_id: str) -> _Subscriber:
        subscriber = _Subscriber(
            queue=asyncio.Queue(maxsize=SUBSCRIBER_QUEUE),
            loop=asyncio.get_running_loop(),
        )
        with self._lock:
            self._subscribers.setdefault(session_id, []).append(subscriber)
        return subscriber

    def _unsubscribe(self, session_id: str, subscriber: _Subscriber) -> None:
        with self._lock:
            live = self._subscribers.get(session_id)
            if not live:
                return
            try:
                live.remove(subscriber)
            except ValueError:
                return
            if not live:
                self._subscribers.pop(session_id, None)

    def replay(self, session_id: str, after_id: int = 0) -> tuple[ProgressEvent, ...]:
        """Backlog events numbered above ``after_id``, oldest first."""
        with self._lock:
            history = self._history.get(session_id)
            if not history:
                return ()
            return tuple(e for e in history if e.id > after_id)

    async def stream(
        self,
        session_id: str,
        *,
        after_id: int = 0,
        heartbeat_s: float = 15.0,
    ) -> AsyncIterator[str]:
        """The SSE body for one subscriber, as already-formatted frames."""
        subscriber = self._subscribe(session_id)
        try:
            for event in self.replay(session_id, after_id):
                yield sse_format(event)
            with self._lock:
                cursor = self._next_id.get(session_id, 0)
            yield sse_format(
                ProgressEvent(
                    id=cursor,
                    name=EVENT_OPEN,
                    data={
                        "session_id": session_id,
                        "replayed_after": after_id,
                        "heartbeat_s": heartbeat_s,
                        "events": list(EVENT_NAMES),
                    },
                )
            )
            while True:
                try:
                    event = await asyncio.wait_for(subscriber.queue.get(), timeout=heartbeat_s)
                except asyncio.TimeoutError:
                    # A comment frame: it keeps the connection open without
                    # claiming anything happened.
                    yield ": keep-alive\n\n"
                    continue
                yield sse_format(event)
        finally:
            self._unsubscribe(session_id, subscriber)

    # -- housekeeping -------------------------------------------------------

    def forget(self, session_id: str) -> None:
        """Drop a session's backlog; called when its session is evicted."""
        with self._lock:
            self._history.pop(session_id, None)
            self._next_id.pop(session_id, None)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "streams": sum(len(v) for v in self._subscribers.values()),
                "sessions_with_backlog": len(self._history),
                "backlog": self._backlog,
            }
