"""What to do next, decided once for both panels.

:func:`next_action` is **pure and mode-blind**. It is handed a :class:`Frame` --
an intent plus a :class:`~cropup.dialog.slots.SlotBag` -- and returns one of five
actions. There is no ``mode`` argument and no way to ask the frame whether the
farmer is typing in the chat box or filling the questionnaire, which is exactly
what makes the two panels one system rather than two features that drift apart
(SPEC section 9).

Purity means: no clock, no environment, no I/O, no Earth Engine, no model. The
only data the decision touches is the frame it was given and the static intent
table in :mod:`cropup.nlu.intents`. Settings are read when a ``PolicyConfig`` is
built, not while deciding, so the same frame always yields the same action and a
test can construct one by hand.

The gate of SPEC section 4.4 lives here: ``run_analysis`` is the only action
that sets ``authorises_earth_engine``, and it is only reached when every
required slot is filled, the location carries coordinates, and the farmer has
confirmed the field. A natural-language turn on its own never gets there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..config import Settings, get_settings
from ..errors import ConfirmationRequired
from ..nlu import intents as intents_mod
from ..nlu.classify import Parse
from .slots import SlotBag

__all__ = [
    "ACTION_ASK_FOR_SLOT",
    "ACTION_CONFIRM_FIELD",
    "ACTION_RUN_ANALYSIS",
    "ACTION_ANSWER_FROM_RAG",
    "ACTION_CLARIFY",
    "ACTIONS",
    "PolicyConfig",
    "Frame",
    "Action",
    "next_action",
    "authorise_earth_engine",
    "SLOT_PROMPTS",
    "CONFIRMABLE_SLOTS",
    "confirmable_slots",
]

ACTION_ASK_FOR_SLOT = "ask_for_slot"
ACTION_CONFIRM_FIELD = "confirm_field"
ACTION_RUN_ANALYSIS = "run_analysis"
ACTION_ANSWER_FROM_RAG = "answer_from_rag"
ACTION_CLARIFY = "clarify"

ACTIONS: tuple[str, ...] = (
    ACTION_ASK_FOR_SLOT,
    ACTION_CONFIRM_FIELD,
    ACTION_RUN_ANALYSIS,
    ACTION_ANSWER_FROM_RAG,
    ACTION_CLARIFY,
)

#: One question per slot. Deterministic text: the dialog asks, it does not
#: compose (SPEC section 4.3).
SLOT_PROMPTS: dict[str, str] = {
    "location": (
        "Which field should I look at? Give me the village, ward or district, "
        "pick one from the list, or drop a pin on the map."
    ),
    "crop": "Which crop is growing in that field?",
    "timeframe": "Which period should I look at?",
}

#: The two slots SPEC section 4.4 names. They are confirmed whenever they are
#: filled, even where the intent calls one of them optional: crop_selection can
#: run without a crop, but if a crop was heard it steers the answer and the
#: farmer has to have endorsed it. A timeframe never gates a run.
CONFIRMABLE_SLOTS: tuple[str, ...] = ("location", "crop")


def confirmable_slots(bag: SlotBag, *, exclude: Sequence[str] = ()) -> tuple[str, ...]:
    """The slots a bare ``POST /confirm`` endorses: filled, and gate-relevant.

    One definition, so the endpoint does not carry a second copy of what
    :func:`next_action` means by "confirm the field". ``exclude`` is how the
    caller says "the farmer just withdrew these": a slot named there is left out
    whatever the bag currently says about it, which is what makes withdrawing a
    confirmation and re-granting it in the same request impossible *by
    construction* rather than by getting the order of two calls right. Asking
    the policy which slots are unconfirmed *after* applying a withdrawal is
    precisely how the withdrawn slot gets confirmed again.

    Pure, like everything else here: it reads the bag and the static table.
    """
    dropped = {str(name) for name in exclude}
    return tuple(slot for slot in CONFIRMABLE_SLOTS if bag.has(slot) and slot not in dropped)


_NO_COORDINATES_PROMPT = (
    "I have {label} written down but no coordinates for it, and I will not "
    "query satellites over a place I cannot put on the map. Pick the matching "
    "place from the list, or drop a pin on your field."
)

#: A value the bag looked up and could not find. Asking is the cheap move; SPEC
#: section 5.1 -- misrouting silently is worse than asking.
_UNRESOLVED_PROMPTS: dict[str, str] = {
    "crop": (
        "I do not have {label} in the crop list I carry, so I cannot tell what "
        "it is or which rules apply to it. Pick the crop from the list, or "
        "spell it another way."
    ),
    "location": (
        "I do not have {label} in the gazetteer I carry, so I cannot put it on "
        "the map. Pick the matching place from the list, or drop a pin on your field."
    ),
}
_UNRESOLVED_FALLBACK = (
    "I could not resolve {label}, so I do not know what it refers to. "
    "Pick one from the list instead."
)


@dataclass(frozen=True)
class PolicyConfig:
    """The few settings the decision depends on, frozen into the frame.

    Read from the environment when the frame is built, never inside
    :func:`next_action`, so the decision stays a function of its argument.
    Confidence floors are deliberately absent: the classifier already applied
    them and reports the verdict as ``route``/``below_floor``, and a second copy
    of a threshold is a second thing to get out of step.
    """

    require_confirmation: bool = True  # CROPUP_REQUIRE_CONFIRMATION, SPEC 4.4
    require_coordinates: bool = True  # a field with no coordinates is not a field

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "PolicyConfig":
        settings = settings or get_settings()
        return cls(require_confirmation=settings.require_confirmation_before_ee)

    def as_dict(self) -> dict[str, Any]:
        return {
            "require_confirmation": self.require_confirmation,
            "require_coordinates": self.require_coordinates,
        }


@dataclass(frozen=True)
class Frame:
    """One turn's whole input to the policy: an intent and the shared bag.

    ``confidence`` is tri-state for the same reason it is in
    :class:`~cropup.nlu.classify.Classification`: ``None`` means no model scored
    this turn (a tier-1 rule hit), which is not a low score and must not be
    treated as one.
    """

    bag: SlotBag
    intent: str | None = None
    route: str | None = None
    text: str = ""
    confidence: float | None = None
    below_floor: bool = False
    clarify_options: tuple[str, ...] = ()
    #: None = nobody checked; False = measured as unavailable (SPEC section 10).
    earth_engine_available: bool | None = None
    policy: PolicyConfig = field(default_factory=PolicyConfig.from_settings)

    def __post_init__(self) -> None:
        if self.intent and not self.route:
            object.__setattr__(self, "route", _route_of(self.intent))
        object.__setattr__(self, "clarify_options", tuple(self.clarify_options or ()))

    @classmethod
    def from_parse(cls, parse: Parse, bag: SlotBag, **kwargs: Any) -> "Frame":
        """Build a frame from a chat turn. The bag is *not* written here; the
        caller folds the extraction in with ``bag.apply_extraction`` so the
        locking rule has one implementation."""
        classification = parse.classification
        kwargs.setdefault("text", classification.text)
        kwargs.setdefault("confidence", classification.confidence)
        kwargs.setdefault("below_floor", classification.below_floor)
        kwargs.setdefault("clarify_options", classification.clarify_options)
        return cls(
            bag=bag,
            intent=classification.intent,
            route=classification.route,
            **kwargs,
        )

    @classmethod
    def from_bag(cls, bag: SlotBag, **kwargs: Any) -> "Frame":
        """Build a frame from questionnaire state: the intent is whatever the
        bag holds, which is what the form wrote into it."""
        kwargs.setdefault("intent", bag.intent)
        return cls(bag=bag, **kwargs)

    def required_slots(self) -> tuple[str, ...]:
        intent = _intent_or_none(self.intent)
        return intent.required_slots if intent else ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "route": self.route,
            "text": self.text,
            "confidence": self.confidence,
            "below_floor": self.below_floor,
            "clarify_options": list(self.clarify_options),
            "earth_engine_available": self.earth_engine_available,
            "policy": self.policy.as_dict(),
            "bag": self.bag.to_dict(),
        }


@dataclass(frozen=True)
class Action:
    """What the caller should do next, and why.

    ``checks`` is the ordered list of gates the decision walked, so a trace can
    be printed without re-running the logic or guessing at it.
    """

    kind: str
    intent: str | None = None
    slot: str | None = None
    prompt: str = ""
    reason: str = ""
    missing_slots: tuple[str, ...] = ()
    unconfirmed_slots: tuple[str, ...] = ()
    options: tuple[str, ...] = ()  # what the farmer can pick in reply
    query: str = ""  # the text to retrieve on, for answer_from_rag
    authorises_earth_engine: bool = False
    checks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ACTIONS:
            raise ValueError(f"unknown action {self.kind!r} (known: {', '.join(ACTIONS)})")
        if self.authorises_earth_engine and self.kind != ACTION_RUN_ANALYSIS:
            # The gate has exactly one door; nothing else may claim to open it.
            raise ValueError(f"{self.kind!r} may not authorise Earth Engine")

    @property
    def is_question(self) -> bool:
        return self.kind in (ACTION_ASK_FOR_SLOT, ACTION_CONFIRM_FIELD, ACTION_CLARIFY)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "intent": self.intent,
            "slot": self.slot,
            "prompt": self.prompt,
            "reason": self.reason,
            "missing_slots": list(self.missing_slots),
            "unconfirmed_slots": list(self.unconfirmed_slots),
            "options": list(self.options),
            "query": self.query,
            "authorises_earth_engine": self.authorises_earth_engine,
            "checks": list(self.checks),
        }

    def __repr__(self) -> str:
        target = f" {self.slot}" if self.slot else ""
        return f"Action({self.kind}{target}, intent={self.intent})"


def _intent_or_none(name: str | None) -> intents_mod.Intent | None:
    if not name:
        return None
    return intents_mod.BY_NAME.get(name)


def _route_of(name: str | None) -> str | None:
    intent = _intent_or_none(name)
    return intent.route if intent else None


def _labels(names: Sequence[str]) -> str:
    out = []
    for name in names:
        intent = _intent_or_none(name)
        out.append(intent.label if intent else name)
    return "; ".join(out)


def next_action(frame: Frame) -> Action:
    """Decide the next move. Pure, mode-blind, and the same for both panels.

    Order of the gates, which is also the order of ``Action.checks``:

    1. an intent we can act on at all;
    2. the route the intent declares (RAG never touches Earth Engine);
    3. every required slot filled;
    4. every filled value resolving to something the app actually knows;
    5. the location actually placeable on the map;
    6. the farmer's confirmation (SPEC section 4.4);
    7. only then, ``run_analysis``.
    """
    checks: list[str] = []
    bag = frame.bag
    intent = _intent_or_none(frame.intent)

    if intent is None:
        checks.append(f"intent={frame.intent!r}: not in the 12-intent table")
        return _clarify(frame, checks, "the message did not land on a known intent")
    checks.append(f"intent={intent.name} route={intent.route}")

    if intent.name == intents_mod.CLARIFY_INTENT or intent.route == intents_mod.ROUTE_CLARIFY:
        return _clarify(frame, checks, "the classifier routed this turn to clarify")

    if frame.below_floor:
        checks.append("classifier reported below_floor: asking beats guessing")
        return _clarify(frame, checks, "the intent score did not clear its floor")

    if intent.route == intents_mod.ROUTE_RAG:
        checks.append("route=rag: knowledge work, no Earth Engine")
        query = frame.text.strip()
        if not query:
            checks.append("no question text to retrieve on")
            return _clarify(frame, checks, "a knowledge answer needs the farmer's question text")
        return Action(
            kind=ACTION_ANSWER_FROM_RAG,
            intent=intent.name,
            query=query,
            reason=f"{intent.label} is answered from the cited corpus, not from satellites",
            checks=tuple(checks),
        )

    # -- the Earth Engine route ------------------------------------------------
    missing = bag.missing(intent.required_slots)
    checks.append(
        "required slots "
        + ", ".join(intent.required_slots)
        + (f": missing {', '.join(missing)}" if missing else ": all filled")
    )
    if missing:
        slot = missing[0]
        return Action(
            kind=ACTION_ASK_FOR_SLOT,
            intent=intent.name,
            slot=slot,
            prompt=SLOT_PROMPTS.get(slot, f"Tell me the {slot}."),
            reason=f"{intent.label} cannot run without {slot}",
            missing_slots=missing,
            checks=tuple(checks),
        )

    unresolved = bag.unresolved(intent.slots)
    if unresolved:
        # The value is filled, so the missing-slot gate above let it through,
        # but nothing in the committed vocabularies matches it. Running on it
        # would be answering about a field or a crop that does not exist.
        slot = unresolved[0]
        value = bag.get(slot)
        label = value.label if value is not None else slot
        checks.append(f"{slot}={label!r} resolves to nothing in the committed vocabularies")
        return Action(
            kind=ACTION_ASK_FOR_SLOT,
            intent=intent.name,
            slot=slot,
            prompt=_UNRESOLVED_PROMPTS.get(slot, _UNRESOLVED_FALLBACK).format(label=label),
            reason=f"{label} is not in the vocabulary, so {slot} is not actually known",
            missing_slots=unresolved,
            options=value.alternatives if value is not None else (),
            checks=tuple(checks),
        )
    checks.append("every filled slot resolves to something committed")

    if frame.policy.require_coordinates and "location" in intent.slots:
        location = bag.get("location")
        if location is not None and location.coordinates() is None:
            checks.append(f"location={location.value!r} carries no coordinates")
            return Action(
                kind=ACTION_ASK_FOR_SLOT,
                intent=intent.name,
                slot="location",
                prompt=_NO_COORDINATES_PROMPT.format(label=location.label),
                reason="the named place was never resolved to a point",
                missing_slots=("location",),
                options=location.alternatives,
                checks=tuple(checks),
            )
        checks.append("location resolves to coordinates")

    to_confirm = tuple(s for s in CONFIRMABLE_SLOTS if s in intent.slots and bag.has(s))
    unconfirmed = bag.unconfirmed(to_confirm)
    if frame.policy.require_confirmation and unconfirmed:
        checks.append(f"unconfirmed: {', '.join(unconfirmed)} (SPEC 4.4 gate is shut)")
        contested = [s for s in unconfirmed if (bag.get(s) and bag.get(s).needs_confirmation)]
        options: tuple[str, ...] = ()
        if len(contested) == 1:
            options = bag.get(contested[0]).alternatives
        return Action(
            kind=ACTION_CONFIRM_FIELD,
            intent=intent.name,
            slot=unconfirmed[0],
            prompt=_confirm_prompt(bag, to_confirm, contested),
            reason="Earth Engine never runs on a field the farmer has not confirmed",
            unconfirmed_slots=unconfirmed,
            options=options,
            checks=tuple(checks),
        )
    if frame.policy.require_confirmation:
        checks.append("confirmed: " + (", ".join(to_confirm) or "nothing to confirm"))
    else:
        checks.append("confirmation disabled by CROPUP_REQUIRE_CONFIRMATION")

    if frame.earth_engine_available is False:
        checks.append("Earth Engine measured as unavailable")
        return Action(
            kind=ACTION_CLARIFY,
            intent=intent.name,
            prompt=(
                "Earth Engine is unavailable right now, so I cannot measure this field. "
                "I can answer from the cited knowledge base instead, or you can try again later."
            ),
            reason="the field is confirmed but the measurement layer is down",
            options=("answer_from_rag", "retry_later"),
            checks=tuple(checks),
        )

    return Action(
        kind=ACTION_RUN_ANALYSIS,
        intent=intent.name,
        prompt="",
        reason=f"{intent.label} on a confirmed field: authorised to query Earth Engine",
        authorises_earth_engine=True,
        checks=tuple(checks),
    )


def _confirm_prompt(
    bag: SlotBag,
    to_confirm: Sequence[str],
    contested: Sequence[str],
) -> str:
    parts = []
    for slot in to_confirm:
        value = bag.get(slot)
        if value is not None:
            parts.append(f"{slot} {value.label}")
    described = ", ".join(parts)
    text = f"Before I query the satellites, confirm the field: {described}."
    if contested:
        contested_value = bag.get(contested[0])
        if contested_value is not None and contested_value.alternatives:
            text += (
                f" {contested_value.label} is an ambiguous name -- "
                f"did you mean {', '.join(contested_value.alternatives)}?"
            )
        elif contested_value is not None:
            text += f" {contested_value.label} is an ambiguous name; confirm it before I use it."
    return text


def _clarify(frame: Frame, checks: list[str], reason: str) -> Action:
    options = frame.clarify_options
    prompt = "I am not sure what you are asking."
    if options:
        prompt += f" Is it one of these: {_labels(options)}?"
    else:
        prompt += (
            " Tell me a bit more, or use the questionnaire to describe your field and your question."
        )
    return Action(
        kind=ACTION_CLARIFY,
        intent=frame.intent,
        prompt=prompt,
        reason=reason,
        options=options,
        checks=tuple(checks),
    )


def authorise_earth_engine(frame: Frame) -> Action:
    """Return the ``run_analysis`` action, or raise ``ConfirmationRequired``.

    This is what ``POST /api/session/{sid}/run`` calls: it turns the policy's
    verdict into the exception the web layer already knows how to report, so the
    gate cannot be bypassed by calling the run endpoint directly.
    """
    action = next_action(frame)
    if action.kind == ACTION_RUN_ANALYSIS and action.authorises_earth_engine:
        return action
    missing = tuple(action.missing_slots) + tuple(action.unconfirmed_slots)
    raise ConfirmationRequired(missing or (action.kind,), action.reason)
