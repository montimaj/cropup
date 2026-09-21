"""SlotBag: the one state behind both the chat panel and the questionnaire.

SPEC section 1.2 is the reason this module exists. Of 78 real farmer questions,
none carries GPS and 60 give no location at all, so slot filling is not a
fallback path -- it is the mechanism that makes anything answerable. The
questionnaire and the chat are therefore two *views* of one frame, never two
features: both write into the same :class:`SlotBag`, and
``dialog.policy.next_action`` reads it without being told which panel wrote it.

Two rules carry the honesty requirements of SPEC sections 4 and 9 into the
dialog layer.

1. **A value the farmer set is locked, and NLU may never overwrite it.** The
   classifier is a suggestion engine; the farmer is the authority on their own
   field. A refused suggestion is *recorded* in the journal rather than
   dropped, so the UI can say "you set Arusha, so I ignored the Dodoma I heard".
2. **An unscored value has no score.** A slot the farmer typed carries
   ``confidence=None``, not ``1.0``. A confidence of 1.0 would be a measurement
   nobody made (SPEC section 4).
3. **A confirmation belongs to a meaning, not to a label.** SPEC section 4.4
   exists to stop an Earth Engine run answering about the wrong field, so
   :meth:`SlotValue.identity` -- not the displayed string -- is what a
   confirmation is bound to. Any write that moves the coordinates, resizes the
   field, changes the resolved entry or changes the crop clears ``confirmed``;
   an identical re-write keeps it. Two map pins can share the label "arusha"
   and sit on different continents, which is precisely the case the label
   cannot catch -- and one pin buffered to 15 m and to 5 km are two different
   fields sharing every other attribute, which is the case a coordinate check
   alone cannot catch. ``field_radius_m`` is therefore part of the location's
   identity rather than a free-standing attribute, so there is no second gate
   to remember.

This module imports ``cropup.nlu`` and nothing heavier; it runs no model, opens
no socket and never touches Earth Engine. It does read the two committed
vocabularies -- a cached dict lookup, not a model -- so that a value matching
nothing in them is stored *named as unresolved* rather than as an ordinary
locked slot (SPEC section 5.1).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Iterator, Mapping, Sequence

from ..nlu import slots as nlu_slots
from ..nlu.slots import SLOT_NAMES, SlotCandidate, SlotExtraction

__all__ = [
    "ORIGIN_NLU",
    "ORIGIN_USER",
    "ORIGIN_GAZETTEER",
    "ORIGIN_GPS",
    "ORIGINS",
    "LOCKING_ORIGINS",
    "INTENT_SLOT",
    "BAG_SLOTS",
    "OUTCOME_SET",
    "OUTCOME_UNCHANGED",
    "OUTCOME_REFUSED_LOCKED",
    "OUTCOME_CLEARED",
    "OUTCOME_RECONFIRM",
    "MIN_FIELD_RADIUS_M",
    "MAX_FIELD_RADIUS_M",
    "validate_field_radius_m",
    "SlotValue",
    "SlotBag",
]

# -- origins ---------------------------------------------------------------

ORIGIN_NLU = "nlu_suggestion"  # heard in a sentence; always correctable
ORIGIN_USER = "user_set"  # typed, picked or tapped by the farmer
ORIGIN_GAZETTEER = "gazetteer"  # a name resolved to a committed gazetteer entry
ORIGIN_GPS = "device_gps"  # the farmer's device reported the position

ORIGINS: tuple[str, ...] = (ORIGIN_NLU, ORIGIN_USER, ORIGIN_GAZETTEER, ORIGIN_GPS)

#: Origins that speak for the farmer. They lock the slot; the others suggest.
LOCKING_ORIGINS = frozenset({ORIGIN_USER, ORIGIN_GPS})

INTENT_SLOT = "intent"
#: Everything the bag can hold: the three NLU slots plus the chosen question.
BAG_SLOTS: tuple[str, ...] = SLOT_NAMES + (INTENT_SLOT,)

# -- write outcomes (journalled, and returned by the bulk writers) ----------

OUTCOME_SET = "set"
OUTCOME_UNCHANGED = "unchanged"
OUTCOME_REFUSED_LOCKED = "refused_locked"
OUTCOME_CLEARED = "cleared"

_MAX_EVENTS = 50  # a session journal for the UI, not an audit log

#: Journalled when a write changed what a confirmed slot *means*. The farmer
#: endorsed the old meaning, so the endorsement does not travel to the new one.
OUTCOME_RECONFIRM = "reconfirm_required"

#: Coordinate ranges. A point outside them is not a place on Earth, so it is
#: refused at construction rather than clamped: clamping would invent a field
#: the farmer never pinned (SPEC section 4).
_LAT_RANGE = (-90.0, 90.0)
_LON_RANGE = (-180.0, 180.0)

#: Bounds on the disc a confirmed point is buffered into before it reaches
#: Earth Engine. This is the *only* geometry input a client can set that is not
#: a coordinate, so it is bounded at the same boundary the coordinates are, and
#: it is refused rather than clamped: a clamp would send Earth Engine a field
#: the farmer never asked for, which is the same fabrication a defaulted pH
#: would be (SPEC section 4).
#:
#: **The floor (1 m).** ``web/geofield.circle_ring`` draws the disc as an
#: angular offset from the point, so below a metre every vertex rounds onto the
#: same coordinate and the "polygon" is a repeated point. Nothing under the
#: finest asset in any chain (Sentinel-2, 10 m -- SPEC section 3.1) can select a
#: different pixel from the bare point anyway, so below a metre the number has
#: stopped meaning anything.
#:
#: **The ceiling (5 km).** That disc is ~78.5 km2, i.e. ~7,850 ha: three orders
#: of magnitude larger than any field in the corpus SPEC section 1.2 is built
#: on, so it cannot refuse a real farmer's field. It is also where a buffer
#: stops being a field at all: the committed gazetteer is rounded to ~1 km for
#: privacy (SPEC section 7), so a wider disc spans several distinct places and
#: the ``reduceRegion`` mean would be a district average shown to the farmer as
#: "your field" -- a plausible wrong number, which is the failure mode this
#: whole codebase is written against.
MIN_FIELD_RADIUS_M = 1.0
MAX_FIELD_RADIUS_M = 5_000.0


def validate_field_radius_m(radius_m: Any, *, source: str = "the field radius") -> float:
    """Return ``radius_m`` as a bounded float, or raise ``ValueError``.

    One implementation, used by :class:`SlotBag` at every write and by
    ``web/dispatch`` at the door to Earth Engine, so the bound cannot be
    enforced in one place and forgotten in the other.
    """
    if isinstance(radius_m, bool) or not isinstance(radius_m, (int, float)):
        raise ValueError(f"{source} must be a number of metres, got {radius_m!r}")
    radius = float(radius_m)
    if not math.isfinite(radius):
        raise ValueError(f"{source} must be a finite number of metres, got {radius_m!r}")
    if radius < MIN_FIELD_RADIUS_M or radius > MAX_FIELD_RADIUS_M:
        raise ValueError(
            f"{source} must be between {MIN_FIELD_RADIUS_M:g} and "
            f"{MAX_FIELD_RADIUS_M:g} metres, got {radius:g}; "
            "outside that range the buffer is not a field, and CropUp will not "
            "send Earth Engine a geometry nobody could have confirmed"
        )
    return radius


#: Named when a value cannot be resolved to any committed artifact. SPEC
#: section 5.1: misrouting silently is worse than asking.
_UNRESOLVED_NOTES: dict[str, str] = {
    "crop": (
        "not in the committed crop vocabulary (crops.json), so I cannot tell "
        "which crop this is"
    ),
    "location": (
        "not in the committed gazetteer (gazetteer.json) and carries no "
        "coordinates, so I cannot put it on the map"
    ),
}


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _norm(text: Any) -> str:
    """Casefolded, whitespace-collapsed form used only for identity comparison.

    Deliberately weaker than the NLU squasher: erring towards "these differ"
    costs one re-confirmation, while erring the other way is the bug SPEC
    section 4.4 exists to prevent.
    """
    return " ".join(str(text).split()).casefold()


def _check_coordinates(slot: str, detail: dict[str, Any]) -> None:
    """Refuse an impossible point at the one boundary every write crosses.

    ``detail`` is normalised in place to floats, so nothing downstream has to
    re-validate what it reads back out.
    """
    lat, lon = detail.get("lat"), detail.get("lon")
    if lat is None and lon is None:
        return
    if lat is None or lon is None:
        missing = "lon" if lon is None else "lat"
        raise ValueError(f"slot {slot!r}: a point needs both lat and lon, {missing} is missing")
    for name, raw, (low, high) in (("lat", lat, _LAT_RANGE), ("lon", lon, _LON_RANGE)):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"slot {slot!r}: {name} must be a number, got {raw!r}")
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"slot {slot!r}: {name} must be finite, got {raw!r}")
        if not low <= number <= high:
            raise ValueError(
                f"slot {slot!r}: {name}={number} is outside {low}..{high} -- "
                "that is not a point on Earth"
            )
        detail[name] = number


def _parse_dt(value: Any) -> dt.datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, str):
        try:
            return dt.datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"not an ISO timestamp: {value!r}") from exc
    raise ValueError(f"not a timestamp: {value!r}")


@dataclass(frozen=True)
class SlotValue:
    """One slot's current reading, and where it came from.

    ``confidence`` is ``None`` whenever nothing scored the value -- a farmer
    typing "Maize" is not a 100%-confident match, it is an unscored fact about
    what they planted.
    """

    slot: str
    value: str
    label: str = ""
    origin: str = ORIGIN_USER
    confidence: float | None = None
    locked: bool = False
    confirmed: bool = False
    needs_confirmation: bool = False
    source: str | None = None
    note: str | None = None
    alternatives: tuple[str, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=dict)
    set_at: dt.datetime = field(default_factory=_now)
    confirmed_at: dt.datetime | None = None
    #: Tri-state, and never defaulted: ``True`` the value was found in a
    #: committed artifact (or carries its own coordinates), ``False`` it was
    #: looked up and is not there, ``None`` nothing looked. ``None`` is not
    #: "fine" -- it is "unchecked" (SPEC section 4).
    resolved: bool | None = None

    def __post_init__(self) -> None:
        set_ = partial(object.__setattr__, self)  # frozen dataclass: normalise in place
        if self.slot not in BAG_SLOTS:
            raise ValueError(f"unknown slot {self.slot!r} (known: {', '.join(BAG_SLOTS)})")
        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError(f"slot {self.slot!r} needs a non-empty value, got {self.value!r}")
        if self.origin not in ORIGINS:
            raise ValueError(f"unknown origin {self.origin!r} (known: {', '.join(ORIGINS)})")
        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool):
                raise ValueError(f"confidence must be a number on 0-1 or None, got {self.confidence!r}")
            if not 0.0 <= float(self.confidence) <= 1.0:
                raise ValueError(f"confidence must be on 0-1, got {self.confidence!r}")
            set_("confidence", round(float(self.confidence), 4))
        if self.confirmed != (self.confirmed_at is not None):
            # A confirmation is an act with a time. Without one it is a flag
            # somebody asserted, which is exactly how SPEC 4.4 gets bypassed.
            raise ValueError(
                f"slot {self.slot!r}: confirmed={self.confirmed} does not match "
                f"confirmed_at={self.confirmed_at!r}; a confirmation needs the moment it happened"
            )
        set_("value", self.value.strip())
        set_("label", (self.label or self.value).strip())
        set_("alternatives", tuple(self.alternatives or ()))
        detail = dict(self.detail or {})
        _check_coordinates(self.slot, detail)
        set_("detail", detail)

    # -- reading ------------------------------------------------------------

    @property
    def settled(self) -> bool:
        """Safe to act on: the farmer has confirmed this exact value."""
        return self.confirmed

    @property
    def from_farmer(self) -> bool:
        return self.origin in LOCKING_ORIGINS

    def coordinates(self) -> tuple[float, float] | None:
        """(lat, lon) when the detail carries them, else None -- never a guess."""
        lat, lon = self.detail.get("lat"), self.detail.get("lon")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            return float(lat), float(lon)
        return None

    def identity(self) -> tuple[Any, ...]:
        """What this value *means*: what a run on it would actually be about.

        A confirmation is bound to this tuple, never to the label. Two map pins
        may share the label "arusha" while sitting on different continents, so
        the label cannot be what the gate remembers (SPEC section 4.4). The
        tuple is (slot, what it resolves to, where it is, how big it is): change
        any of them and the farmer is looking at a different field or a
        different crop, and has to say so again.

        The last element is why widening a confirmed field is not a loophole.
        The geometry Earth Engine is sent is ``Point(lon, lat).buffer(radius)``,
        so the radius is as much a part of "which field" as the point is: a
        farmer who endorsed 15 m at Arusha did not endorse 5 km at Arusha. It
        rides in ``detail['radius_m']``, stamped there by
        :meth:`SlotBag._write` from the bag's own chosen radius, so it travels
        through the one door and cannot be enforced separately or forgotten.
        """
        resolved_id = None
        for key in ("key", "place_key", "crop_key", "id"):
            found = self.detail.get(key)
            if isinstance(found, str) and found.strip():
                resolved_id = _norm(found)
                break
        return (self.slot, resolved_id or _norm(self.value), self._point(), self._extent())

    def _point(self) -> tuple[float, float] | None:
        """Coordinates rounded to ~1cm, so a JSON round trip is not a new field."""
        coords = self.coordinates()
        if coords is None:
            return None
        return (round(coords[0], 7), round(coords[1], 7))

    def _extent(self) -> float | None:
        """The buffer radius in metres, rounded to ~1mm; ``None`` when unset.

        ``None`` is not "the default": it means the farmer has chosen no field
        size, and the run will document that it used the configured one. Moving
        from ``None`` to an explicit number changes the identity on purpose --
        the farmer stated a size, which is a statement about their field that
        the previous confirmation did not contain.
        """
        radius = self.detail.get("radius_m")
        if isinstance(radius, bool) or not isinstance(radius, (int, float)):
            return None
        return round(float(radius), 3)

    def as_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "value": self.value,
            "label": self.label,
            "origin": self.origin,
            "confidence": self.confidence,
            "locked": self.locked,
            "confirmed": self.confirmed,
            "needs_confirmation": self.needs_confirmation,
            "source": self.source,
            "note": self.note,
            "alternatives": list(self.alternatives),
            "detail": dict(self.detail),
            "set_at": self.set_at.isoformat(),
            "confirmed_at": self.confirmed_at.isoformat() if self.confirmed_at else None,
            "resolved": self.resolved,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SlotValue":
        return cls(
            slot=str(payload["slot"]),
            value=str(payload["value"]),
            label=str(payload.get("label") or ""),
            origin=str(payload.get("origin") or ORIGIN_USER),
            confidence=payload.get("confidence"),
            locked=bool(payload.get("locked", False)),
            confirmed=bool(payload.get("confirmed", False)),
            needs_confirmation=bool(payload.get("needs_confirmation", False)),
            source=payload.get("source"),
            note=payload.get("note"),
            alternatives=tuple(payload.get("alternatives") or ()),
            detail=dict(payload.get("detail") or {}),
            set_at=_parse_dt(payload.get("set_at")) or _now(),
            confirmed_at=_parse_dt(payload.get("confirmed_at")),
            resolved=payload.get("resolved"),
        )

    @classmethod
    def from_candidate(
        cls,
        candidate: SlotCandidate,
        *,
        origin: str = ORIGIN_NLU,
        at: dt.datetime | None = None,
    ) -> "SlotValue":
        """Wrap one NLU reading. The candidate's own flags travel with it, so an
        ambiguous "Same" arrives flagged rather than quietly resolved."""
        return cls(
            slot=candidate.slot,
            value=candidate.value,
            label=candidate.label,
            origin=origin,
            confidence=candidate.confidence,
            locked=origin in LOCKING_ORIGINS,
            needs_confirmation=candidate.needs_confirmation,
            source=candidate.source,
            note=candidate.note,
            alternatives=candidate.alternatives,
            detail=dict(candidate.detail),
            set_at=at or _now(),
        )

    def __repr__(self) -> str:
        flags = "".join(
            (
                "L" if self.locked else "-",
                "C" if self.confirmed else "-",
                "?" if self.needs_confirmation else "-",
            )
        )
        conf = "unscored" if self.confidence is None else f"{self.confidence:.2f}"
        return f"SlotValue({self.slot}={self.value!r} {self.origin} {conf} [{flags}])"


def _resolution(value: SlotValue) -> bool | None:
    """Can this value be resolved to something the app actually knows?

    ``None`` means the lookup could not be made, which is *not* a pass: an
    unchecked value is named as unchecked rather than assumed good (SPEC
    section 4). The vocabularies are the committed artifacts, so this is a
    dict lookup over a cached file, not a model.
    """
    if value.slot == "location":
        if value.coordinates() is not None:
            return True  # a point is its own resolution
        try:
            return nlu_slots.find_place(value.value) is not None
        except Exception:  # pragma: no cover - a missing artifact is not a verdict
            return None
    if value.slot == "crop":
        try:
            return nlu_slots.find_crop(value.value) is not None
        except Exception:  # pragma: no cover - a missing artifact is not a verdict
            return None
    return None  # timeframe and intent are validated by their own writers


def _with_resolution(value: SlotValue) -> SlotValue:
    """Stamp the resolution verdict on a value, and name it when it fails.

    SPEC section 5.1: misrouting silently is worse than asking. A farmer's typo
    must not become a locked slot that looks as good as a gazetteer hit.
    """
    verdict = _resolution(value)
    if verdict == value.resolved and not (verdict is False and value.note is None):
        return value
    note = value.note
    if verdict is False and not note:
        note = _UNRESOLVED_NOTES.get(value.slot, "could not be resolved")
    return replace(value, resolved=verdict, note=note)


def _with_field_radius(value: SlotValue, radius_m: float | None) -> SlotValue:
    """Mirror the bag's chosen field radius into a location value's detail.

    The bag holds one radius; this copy exists so that
    :meth:`SlotValue.identity` -- which is all the confirmation gate looks at --
    sees it. Because every location write is re-stamped here, a client cannot
    smuggle a ``radius_m`` of its own into ``detail`` and it cannot drift from
    the bag's value.
    """
    if value.slot != "location":
        return value
    detail = dict(value.detail)
    if radius_m is None:
        if "radius_m" not in detail:
            return value
        detail.pop("radius_m")
    else:
        if detail.get("radius_m") == radius_m and not isinstance(detail.get("radius_m"), bool):
            return value
        detail["radius_m"] = float(radius_m)
    return replace(value, detail=detail)


def _check_confirmable(value: SlotValue) -> None:
    """Refuse to let the farmer endorse something the app cannot act on.

    SPEC section 4.4's gate authorises a run *on a particular field*. A value
    that resolved to nothing in the committed artifacts, or a location with no
    coordinates, is not a field: confirming it would mark the session ready and
    let ``/api/geo/field`` report a confirmed field that cannot be drawn. The
    fix is to refuse the endorsement, which is what ``dialog.policy`` is already
    telling the farmer to do -- it asks for the slot again before it ever asks
    for a confirmation.
    """
    if value.resolved is False:
        raise ValueError(
            f"cannot confirm {value.slot} {value.label!r}: "
            f"{value.note or 'it resolves to nothing the app knows'}. "
            "A value that resolves to nothing cannot be the field a run is "
            "authorised for; pick one from the committed vocabulary instead"
        )
    if value.slot == "location" and value.coordinates() is None:
        raise ValueError(
            f"cannot confirm location {value.label!r}: it carries no coordinates, "
            "and a place that cannot be put on the map is not a field Earth "
            "Engine can be pointed at; pick the matching place from the "
            "gazetteer or drop a pin"
        )


class SlotBag:
    """Everything one session knows about the farmer's question.

    Mutable on purpose: it is session state, written by the chat handler, the
    questionnaire handler and the map pin alike. Every write goes through
    :meth:`set`, so the locking rule has exactly one implementation.
    """

    def __init__(
        self,
        session_id: str | None = None,
        *,
        created_at: dt.datetime | None = None,
        field_radius_m: float | None = None,
    ) -> None:
        self.session_id = session_id
        self.created_at = created_at or _now()
        self.updated_at = self.created_at
        #: None means the farmer has not chosen one; analysis documents its own
        #: default rather than this module inventing a field size. Read through
        #: the :attr:`field_radius_m` property and written only by
        #: :meth:`set_field_radius`, which is what keeps it in step with the
        #: copy the confirmation gate reads out of the location's identity.
        self._field_radius_m = (
            None if field_radius_m is None else validate_field_radius_m(field_radius_m)
        )
        self._values: dict[str, SlotValue] = {}
        self._events: list[dict[str, Any]] = []

    # -- writing ------------------------------------------------------------

    def set(
        self,
        slot: str,
        value: str,
        *,
        origin: str = ORIGIN_USER,
        label: str = "",
        confidence: float | None = None,
        source: str | None = None,
        note: str | None = None,
        alternatives: Sequence[str] = (),
        detail: Mapping[str, Any] | None = None,
        needs_confirmation: bool = False,
        lock: bool | None = None,
        confirm: bool = False,
        at: dt.datetime | None = None,
    ) -> SlotValue | None:
        """Write a slot. Returns the stored value, or ``None`` if it was refused.

        A non-locking origin (NLU, gazetteer) cannot overwrite a locked slot:
        the attempt is journalled and ``None`` comes back. ``lock`` defaults to
        whether the origin speaks for the farmer.
        """
        stored = self._write(
            SlotValue(
                slot=slot,
                value=value,
                label=label,
                origin=origin,
                confidence=confidence,
                locked=(origin in LOCKING_ORIGINS) if lock is None else bool(lock),
                needs_confirmation=needs_confirmation,
                source=source,
                note=note,
                alternatives=tuple(alternatives),
                detail=dict(detail or {}),
                set_at=at or _now(),
            )
        )
        if confirm and stored is not None:
            # Confirming is its own act, applied to what was actually stored.
            # Routing it through :meth:`confirm` is what stops a caller from
            # handing the bag a value that arrives pre-endorsed (SPEC 4.4).
            self.confirm(slot, at=at)
            return self._values[slot]
        return stored

    def suggest(self, slot: str, value: str, **kwargs: Any) -> SlotValue | None:
        """Write a slot as an NLU suggestion: correctable, never locking."""
        kwargs.setdefault("origin", ORIGIN_NLU)
        return self.set(slot, value, **kwargs)

    def apply_candidate(
        self, candidate: SlotCandidate, *, origin: str = ORIGIN_NLU, at: dt.datetime | None = None
    ) -> SlotValue | None:
        return self._write(SlotValue.from_candidate(candidate, origin=origin, at=at))

    def apply_extraction(
        self,
        extraction: SlotExtraction,
        *,
        origin: str = ORIGIN_NLU,
        at: dt.datetime | None = None,
    ) -> dict[str, str]:
        """Fold one parsed sentence into the bag; returns slot -> outcome.

        The *top* candidate is used, not the *best* one: a contested reading is
        recorded with its ``needs_confirmation`` flag and its alternatives so the
        dialog can ask, instead of being silently dropped.
        """
        outcomes: dict[str, str] = {}
        for slot in SLOT_NAMES:
            candidate = extraction.top(slot)
            if candidate is None:
                continue
            before = self._values.get(slot)
            stored = self.apply_candidate(candidate, origin=origin, at=at)
            if stored is None:
                outcomes[slot] = OUTCOME_REFUSED_LOCKED
            elif before is not None and before.identity() == stored.identity():
                outcomes[slot] = OUTCOME_UNCHANGED
            else:
                outcomes[slot] = OUTCOME_SET
        return outcomes

    def set_intent(self, name: str, *, origin: str = ORIGIN_NLU, **kwargs: Any) -> SlotValue | None:
        """The chosen question type is part of the shared frame: the form picks
        it from a list, the classifier suggests it, and the same lock applies."""
        return self.set(INTENT_SLOT, name, origin=origin, **kwargs)

    def _write(self, candidate_value: SlotValue, *, farmer_act: bool = False) -> SlotValue | None:
        """The one door into ``_values``, and the one place SPEC 4.4 is enforced.

        A confirmation survives a write only when the write does not change
        what the slot *means* -- :meth:`SlotValue.identity`. Writing the same
        label over different coordinates is a different field, so the farmer
        has to confirm it again; that is the whole point of the gate. Resizing
        the field is the same kind of change and is caught by the same
        comparison, because the radius is stamped into the location's detail
        here and read back by ``identity``.

        ``farmer_act`` is for a write the farmer made *about* a slot they
        already own -- today only :meth:`set_field_radius`, which resizes the
        field without touching where it is. It skips the locked-slot refusal
        (the farmer may always edit their own locked slot) and nothing else: in
        particular it does not skip the confirmation comparison below, so it
        cannot be used to change the field quietly.
        """
        candidate_value = _with_resolution(candidate_value)
        candidate_value = _with_field_radius(candidate_value, self._field_radius_m)
        slot = candidate_value.slot
        current = self._values.get(slot)
        if (
            current is not None
            and current.locked
            and not candidate_value.from_farmer
            and not farmer_act
        ):
            self._journal(
                slot,
                OUTCOME_REFUSED_LOCKED,
                candidate_value,
                detail=f"kept {current.value!r} set by {current.origin}",
            )
            return None
        same_meaning = current is not None and current.identity() == candidate_value.identity()
        if same_meaning:
            # Same reading again: keep the confirmation the farmer already gave,
            # and keep the stronger origin rather than demoting it to a guess.
            keep_origin = current.origin if current.from_farmer else candidate_value.origin
            merged = replace(
                candidate_value,
                origin=keep_origin,
                locked=current.locked or candidate_value.locked,
                confirmed=current.confirmed or candidate_value.confirmed,
                confirmed_at=current.confirmed_at or candidate_value.confirmed_at,
                confidence=candidate_value.confidence if candidate_value.confidence is not None else current.confidence,
                set_at=current.set_at,
            )
            self._values[slot] = merged
            self.updated_at = candidate_value.set_at
            self._journal(slot, OUTCOME_UNCHANGED, merged)
            return merged
        stored = candidate_value
        if stored.confirmed:
            # Structural, not a convention: a value that arrives claiming a
            # confirmation it was not given here loses it, whatever built it.
            stored = replace(stored, confirmed=False, confirmed_at=None)
        self._values[slot] = stored
        self.updated_at = stored.set_at
        self._journal(slot, OUTCOME_SET, stored)
        if current is not None and current.confirmed:
            self._journal(
                slot,
                OUTCOME_RECONFIRM,
                stored,
                detail=(
                    f"{current.value!r} was confirmed, but this write means something "
                    f"else ({current.identity()[1:]!r} -> {stored.identity()[1:]!r}); "
                    "confirmation withdrawn"
                ),
            )
        return stored

    def confirm(self, *slots: str, at: dt.datetime | None = None) -> tuple[str, ...]:
        """The farmer endorses these readings. Confirming also locks: an
        endorsed value is the farmer's, whoever first proposed it.

        This is the act SPEC section 4.4 requires before Earth Engine runs, so
        it refuses what cannot be endorsed: an empty slot, a value that resolves
        to nothing in the committed artifacts, or a location with no
        coordinates. See :func:`_check_confirmable`.
        """
        when = at or _now()
        wanted = tuple(slots or tuple(self.filled()))
        # Check every slot before endorsing any of them. A confirmation is one
        # act over one field; half of it applied and then a refusal would leave
        # the session claiming the farmer endorsed a crop for a field that was
        # rejected in the same breath.
        for slot in wanted:
            current = self._values.get(slot)
            if current is None:
                raise ValueError(f"cannot confirm empty slot {slot!r}")
            _check_confirmable(current)
        confirmed: list[str] = []
        for slot in wanted:
            current = self._values[slot]
            self._values[slot] = replace(
                current,
                confirmed=True,
                confirmed_at=when,
                locked=True,
                needs_confirmation=False,
            )
            confirmed.append(slot)
            self._journal(slot, "confirmed", self._values[slot])
        self.updated_at = when
        return tuple(confirmed)

    def unconfirm(self, *slots: str) -> tuple[str, ...]:
        """Withdraw confirmation, e.g. when the farmer edits the field."""
        touched = []
        for slot in slots or SLOT_NAMES:
            current = self._values.get(slot)
            if current is None or not current.confirmed:
                continue
            self._values[slot] = replace(current, confirmed=False, confirmed_at=None)
            touched.append(slot)
            self._journal(slot, "unconfirmed", self._values[slot])
        if touched:
            self.updated_at = _now()
        return tuple(touched)

    def clear(self, slot: str) -> SlotValue | None:
        """Drop a slot. Only the farmer clears a locked slot, so this is the
        user-driven path; NLU never calls it."""
        removed = self._values.pop(slot, None)
        if removed is not None:
            self.updated_at = _now()
            self._journal(slot, OUTCOME_CLEARED, removed)
        return removed

    def unlock(self, slot: str) -> SlotValue | None:
        """Hand a slot back to the suggestion engine."""
        current = self._values.get(slot)
        if current is None or not current.locked:
            return current
        self._values[slot] = replace(current, locked=False, confirmed=False, confirmed_at=None)
        self.updated_at = _now()
        self._journal(slot, "unlocked", self._values[slot])
        return self._values[slot]

    @property
    def field_radius_m(self) -> float | None:
        """The field size the farmer chose, or ``None`` if they chose none.

        Read-only on purpose. It is not a free-standing number any more: it is
        half of the geometry the confirmation gate is bound to, so it is
        changed through :meth:`set_field_radius`, which routes the change
        through :meth:`_write` like every other change to the field.
        """
        return self._field_radius_m

    def set_field_radius(self, radius_m: float | None) -> float | None:
        """Choose (or clear) the size of the field, and re-open the gate if it moved.

        The radius is the second half of ``Point(lon, lat).buffer(radius)`` --
        the one client-settable input to the geometry that is not a coordinate.
        Widening it after a confirmation would mean Earth Engine measuring a
        field the farmer never endorsed, exactly as re-pointing the coordinates
        would, so this does not carry its own gate: it re-writes the location
        through :meth:`_write`, the radius is part of
        :meth:`SlotValue.identity`, and the existing withdrawal machinery does
        the rest -- the confirmation is dropped and ``reconfirm_required`` is
        journalled with the same words a moved pin gets.

        The value itself is bounded by :func:`validate_field_radius_m` and
        refused, never clamped, when it is out of range.
        """
        radius = None if radius_m is None else validate_field_radius_m(radius_m)
        self._field_radius_m = radius
        current = self._values.get("location")
        if current is not None:
            # Same value, new size: _write stamps the radius in, compares the
            # identity and withdraws the confirmation if the field changed.
            # ``farmer_act`` because resizing your own field is your own act,
            # even when the coordinates came from the gazetteer or from NLU.
            self._write(replace(current, set_at=_now()), farmer_act=True)
        self.updated_at = _now()
        return radius

    def _journal(
        self, slot: str, outcome: str, value: SlotValue, detail: str | None = None
    ) -> None:
        self._events.append(
            {
                "at": _now().isoformat(),
                "slot": slot,
                "outcome": outcome,
                "origin": value.origin,
                "value": value.value,
                "detail": detail,
            }
        )
        del self._events[:-_MAX_EVENTS]

    # -- reading ------------------------------------------------------------

    def get(self, slot: str) -> SlotValue | None:
        if slot not in BAG_SLOTS:
            raise ValueError(f"unknown slot {slot!r} (known: {', '.join(BAG_SLOTS)})")
        return self._values.get(slot)

    def value(self, slot: str) -> str | None:
        current = self.get(slot)
        return current.value if current else None

    def label(self, slot: str) -> str | None:
        current = self.get(slot)
        return current.label if current else None

    @property
    def intent(self) -> str | None:
        return self.value(INTENT_SLOT)

    def has(self, slot: str) -> bool:
        return self.get(slot) is not None

    def is_locked(self, slot: str) -> bool:
        current = self.get(slot)
        return bool(current and current.locked)

    def is_confirmed(self, slot: str) -> bool:
        current = self.get(slot)
        return bool(current and current.confirmed)

    def filled(self) -> dict[str, SlotValue]:
        """The three dialog slots that hold something, in SLOT_NAMES order."""
        return {slot: self._values[slot] for slot in SLOT_NAMES if slot in self._values}

    def missing(self, required: Sequence[str]) -> tuple[str, ...]:
        return tuple(slot for slot in required if not self.has(slot))

    def unconfirmed(self, required: Sequence[str]) -> tuple[str, ...]:
        """Filled but not yet endorsed: what the confirmation step must cover."""
        return tuple(slot for slot in required if self.has(slot) and not self.is_confirmed(slot))

    def unresolved(self, slots: Sequence[str] | None = None) -> tuple[str, ...]:
        """Filled slots whose value was looked up and is not in any vocabulary.

        ``resolved is None`` (nobody looked) is deliberately not reported here:
        this answers "what did we check and fail to find", not "what are we
        unsure about".
        """
        names = tuple(slots) if slots is not None else SLOT_NAMES
        out = []
        for slot in names:
            current = self._values.get(slot)
            if current is not None and current.resolved is False:
                out.append(slot)
        return tuple(out)

    def contested(self) -> tuple[str, ...]:
        """Slots holding a reading the extractor itself flagged as uncertain."""
        return tuple(
            slot
            for slot, value in self.filled().items()
            if value.needs_confirmation and not value.confirmed
        )

    def suggestions(self) -> tuple[str, ...]:
        """Slots the farmer has not touched; the UI shows these as correctable."""
        return tuple(slot for slot, value in self.filled().items() if not value.from_farmer)

    def coordinates(self) -> tuple[float, float] | None:
        current = self.get("location")
        return current.coordinates() if current else None

    def field_ref(self) -> dict[str, Any] | None:
        """What ``GET /api/geo/field`` describes: the point that would be sent
        to Earth Engine, or ``None`` when no coordinates are known."""
        current = self.get("location")
        if current is None:
            return None
        coords = current.coordinates()
        if coords is None:
            return None
        lat, lon = coords
        return {
            "label": current.label,
            "lat": lat,
            "lon": lon,
            "radius_m": self.field_radius_m,
            "origin": current.origin,
            "confirmed": current.confirmed,
            "source": current.source,
        }

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._events)

    def describe(self) -> str:
        """One line for a log or a policy trace."""
        if not self._values:
            return "SlotBag(empty)"
        parts = []
        for slot in BAG_SLOTS:
            current = self._values.get(slot)
            if current is None:
                continue
            marks = "".join(("*" if current.locked else "", "!" if current.confirmed else ""))
            parts.append(f"{slot}={current.value}{marks}({current.origin})")
        return "SlotBag(" + ", ".join(parts) + ")"

    # -- session persistence -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise for ``GET /api/session/{sid}`` and for rehydration."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "field_radius_m": self.field_radius_m,
            "intent": self.intent,
            "slots": {slot: value.as_dict() for slot, value in self._values.items()},
            "filled": {slot: value.value for slot, value in self.filled().items()},
            "locked": [slot for slot in BAG_SLOTS if self.is_locked(slot)],
            "confirmed": [slot for slot in BAG_SLOTS if self.is_confirmed(slot)],
            "contested": list(self.contested()),
            "unresolved": list(self.unresolved()),
            "events": list(self._events),
        }

    as_dict = to_dict  # the name the rest of the codebase uses for JSON views

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SlotBag":
        bag = cls(
            session_id=payload.get("session_id"),
            created_at=_parse_dt(payload.get("created_at")),
            field_radius_m=payload.get("field_radius_m"),
        )
        for slot, value in (payload.get("slots") or {}).items():
            stored = SlotValue.from_dict({**value, "slot": value.get("slot", slot)})
            # A snapshot is a client's copy: its ``detail['radius_m']`` is not
            # evidence of anything, so the bag's own bounded radius is stamped
            # over it rather than trusted.
            stored = _with_field_radius(stored, bag._field_radius_m)
            bag._values[stored.slot] = stored
        bag.updated_at = _parse_dt(payload.get("updated_at")) or bag.created_at
        bag._events = [dict(e) for e in (payload.get("events") or [])][-_MAX_EVENTS:]
        return bag

    # -- dunders -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.filled())

    def __iter__(self) -> Iterator[SlotValue]:
        return iter(self.filled().values())

    def __contains__(self, item: object) -> bool:
        return isinstance(item, str) and item in self._values

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SlotBag):
            return NotImplemented
        return self.session_id == other.session_id and self._values == other._values

    def __repr__(self) -> str:
        return self.describe()
