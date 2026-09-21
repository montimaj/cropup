"""Turning a policy verdict into the one thing the farmer is shown.

SPEC section 4.2 makes ``render/templates.py`` the only farmer-facing text
channel, and that has to include the dialog's own sentences, not just the
numbers. So every reply this server produces -- a question about a missing slot,
a confirmation request, a knowledge answer, an honest decline -- leaves here as
a :class:`~cropup.render.templates.Answer`, segmented, with whatever provenance
the segment carries. The frontend then has exactly one shape to draw and one
place to hang SPEC section 9's hover.

The wording itself is not composed here either. ``Action.prompt`` is written in
``dialog/policy.py`` and travels verbatim; the decline sentences are literal
constants in this module. Nothing in CropUp generates text (SPEC section 4.3),
and this module is where that could most easily have been broken.

Imports: ``render``, ``rag``, ``dialog``, ``evidence``. No Earth Engine.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from ..dialog import policy as policy_mod
from ..dialog.slots import SlotBag
from ..errors import CropUpError
from ..evidence import Ledger
from ..nlu import intents as intents_mod
from ..rag import retrieve as rag_retrieve
from ..render import templates as render

__all__ = [
    "ANSWER_KIND_DIALOG",
    "ANSWER_KIND_RAG",
    "ANSWER_KIND_UNAVAILABLE",
    "MAX_GAP_LINES",
    "dialog_answer",
    "unavailable_answer",
    "knowledge_answer",
    "condense_gaps",
    "reply_envelope",
]

#: How many "I could not measure this" lines a farmer is shown. The same cap
#: SPEC section 4.3 puts on risks, for the same reason: with Earth Engine
#: unreachable a plant-health run names 92 gaps, and the renderer prints one line
#: per gap ending in the raw exception the source raised. Ninety-two lines of
#: ``RuntimeError: ...`` is not a reply. The rest is not discarded -- it moves to
#: the diagnostic block :func:`condense_gaps` returns.
MAX_GAP_LINES = 5

ANSWER_KIND_DIALOG = "dialog"
ANSWER_KIND_RAG = "rag"
ANSWER_KIND_UNAVAILABLE = "unavailable"

#: Titles per action kind. One line each, chosen from a closed table.
#:
#: ``clarify`` is deliberately neutral. ``dialog/policy.py`` reaches it for two
#: quite different situations -- the message did not land on an intent, and the
#: field is confirmed but Earth Engine is down -- and the prompt says which. A
#: title of "I am not sure what you are asking" over "Earth Engine is
#: unavailable right now" would be the heading contradicting the sentence under
#: it, so the heading claims only what is true of both.
_TITLES: dict[str, str] = {
    policy_mod.ACTION_ASK_FOR_SLOT: "I need one more thing",
    policy_mod.ACTION_CONFIRM_FIELD: "Confirm the field before I query the satellites",
    policy_mod.ACTION_CLARIFY: "I need to check something before I answer",
    policy_mod.ACTION_RUN_ANALYSIS: "Ready to measure this field",
}

#: SPEC 4.4 in the farmer's words. A natural-language turn stops here even when
#: every slot is already confirmed: the run is a separate, explicit act.
_READY_TO_RUN = (
    "Your field is confirmed, so I can measure it. Earth Engine is never started "
    "by a chat message -- press Run to start the measurement."
)


def dialog_answer(
    action: policy_mod.Action,
    *,
    title: str | None = None,
    extra: tuple[str, ...] = (),
) -> render.Answer:
    """One dialog turn, rendered. The prompt is ``dialog/policy.py``'s, verbatim."""
    builder = render.AnswerBuilder(
        ANSWER_KIND_DIALOG,
        title or _TITLES.get(action.kind, "CropUp"),
        Ledger(turn=f"dialog:{action.kind}"),
    )
    builder.section(action.kind, "")
    if action.kind == policy_mod.ACTION_RUN_ANALYSIS and not action.prompt:
        builder.text(_READY_TO_RUN)
    elif action.prompt:
        builder.text(action.prompt)
    else:
        builder.text(action.reason)
    for line in extra:
        if line:
            builder.text(line)
    return builder.build()


def unavailable_answer(
    *,
    what: str,
    detail: str,
    title: str = "I cannot answer that right now",
    offer: str | None = None,
) -> render.Answer:
    """An honest decline: what is unavailable, why, and what is still possible.

    SPEC section 10's null-adapter run ends here rather than in a traceback --
    "the app must still answer or honestly decline, never crash".
    """
    builder = render.AnswerBuilder(
        ANSWER_KIND_UNAVAILABLE, title, Ledger(turn=f"unavailable:{what}")
    )
    builder.section("unavailable", "")
    builder.text(f"{what} is unavailable right now, so I have nothing measured to show you.")
    builder.text(detail)
    if offer:
        builder.text(offer)
    return builder.build()


def knowledge_answer(
    action: policy_mod.Action,
    bag: SlotBag,
    *,
    settings: Any | None = None,
) -> tuple[render.Answer, dict[str, Any] | None, str | None]:
    """Retrieve for a RAG-routed turn and render the result verbatim.

    Returns ``(answer, retrieval, unavailable_reason)``. On a healthy path
    ``retrieval`` is ``RetrievalResult.as_dict()`` and the reason is ``None``;
    when the index or the encoder is missing the answer is an honest decline,
    ``retrieval`` is ``None`` and the reason names the failure. Retrieval is 72%
    of real traffic (SPEC section 1.2), so its failure is reported as a degraded
    answer, not as a 500.
    """
    query = action.query or ""
    try:
        result = rag_retrieve.retrieve(
            query,
            intent=action.intent,
            crop=bag.value("crop"),
            settings=settings,
        )
    except CropUpError as exc:
        reason = f"{type(exc).__name__}: {exc}"
        return (
            unavailable_answer(
                what="The knowledge base",
                detail=(
                    f"Retrieval could not run ({reason}). I will not answer an "
                    "agronomy question from memory, because an invented "
                    "fertiliser dose is a safety failure, not a feature."
                ),
                offer=(
                    "Use the questionnaire: with a location and a crop I can "
                    "measure the field directly instead."
                ),
            ),
            None,
            reason,
        )
    return render.rag_answer(result, question=query), result.as_dict(), None


def _plain_gap_text(provenance: Mapping[str, Any], fallback: str) -> str:
    """One gap in the farmer's terms: what is missing, and why, in plain words.

    Deliberately *not* the renderer's full line. That one appends ``detail``,
    which is ``f"{type(exc).__name__}: {exc}"`` from ``geo/`` -- the operator's
    diagnostic. No farmer's sentence ends in ``RuntimeError: no network: EE
    unreachable``, and with every source down there were 92 sentences that did.
    The raw string stays on the segment's provenance, in the answer's
    degradation block and in the diagnostics :func:`condense_gaps` returns,
    where the UI can reveal it; it just does not belong in the prose.
    """
    quantity = str(provenance.get("quantity") or "").replace("_", " ").strip()
    reason = str(provenance.get("reason_text") or "").strip()
    if not quantity:
        return fallback
    return f"{quantity}: not available \u2014 {reason}." if reason else f"{quantity}: not available."


def _gap_row(provenance: Mapping[str, Any], rendered: str) -> dict[str, Any]:
    return {
        "quantity": provenance.get("quantity"),
        "reason": provenance.get("reason"),
        "reason_text": provenance.get("reason_text"),
        "chain_tried": list(provenance.get("chain_tried") or ()),
        # The raw source error: kept here, and nowhere in the prose.
        "detail": provenance.get("detail"),
        "rendered": rendered,
    }


def condense_gaps(
    answer: render.Answer, *, cap: int = MAX_GAP_LINES
) -> tuple[render.Answer, dict[str, Any] | None]:
    """Make an answer's absences readable, and hand the raw detail back separately.

    Two changes, both only to what is *said*:

    1. every "not available" sentence, in every section, loses the chain and the
       raw exception its source raised and keeps the quantity and the plain
       reason. The dropped text is not lost: it is still on that segment's
       ``provenance``, which is what SPEC section 9's hover reads.
    2. the "what I could not measure" section is capped at ``cap`` lines and the
       rest are counted, by reason, on one closing line -- the same cap SPEC
       section 4.3 puts on risks, for the same reason.

    Returns ``(answer, diagnostics)``, where ``diagnostics`` is ``None`` when
    there was nothing to condense, and otherwise carries every absence in the
    answer with the raw error each source returned.

    Nothing is invented and nothing is hidden: the counts are counts of the
    ledger's own gaps, the names are the ledger's names, and
    ``Answer.degradation`` -- which the capability strip reads -- is passed
    through untouched and still lists all of them.
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    changed = False
    sections: list[render.Section] = []

    for section in answer.sections:
        lines: list[render.RenderedText] = []
        for line in section.lines:
            segments: list[render.Segment] = []
            for segment in line.segments:
                if segment.kind != render.SEGMENT_MISSING or not segment.provenance:
                    segments.append(segment)
                    continue
                provenance = dict(segment.provenance)
                quantity = str(provenance.get("quantity") or segment.slot or "")
                if quantity not in seen:
                    seen.add(quantity)
                    rows.append(_gap_row(provenance, segment.text))
                plain = _plain_gap_text(provenance, segment.text)
                if plain != segment.text:
                    changed = True
                    segment = replace(segment, text=plain)
                segments.append(segment)
            lines.append(
                render.RenderedText(
                    template=line.template, source=line.source, segments=tuple(segments)
                )
            )

        if section.key == "gaps" and len(lines) > max(0, cap):
            changed = True
            by_reason: dict[str, int] = {}
            for line in lines:
                for segment in line.segments:
                    if segment.kind == render.SEGMENT_MISSING and segment.provenance:
                        key = str(segment.provenance.get("reason_text") or "no reason recorded")
                        by_reason[key] = by_reason.get(key, 0) + 1
            causes = "; ".join(f"{count} because {reason}" for reason, count in sorted(by_reason.items()))
            remaining = len(lines) - cap
            lines = lines[:cap]
            lines.append(
                render.RenderedText(
                    template="<literal>",
                    source="gap summary",
                    segments=(
                        render.Segment(
                            render.SEGMENT_TEXT,
                            f"{remaining} further readings were also unavailable"
                            + (f" ({causes})" if causes else "")
                            + ". The full list, with the technical reason each "
                            "source gave, is in this answer's diagnostics.",
                        ),
                    ),
                )
            )

        sections.append(
            render.Section(
                key=section.key,
                heading=section.heading,
                lines=tuple(lines),
                not_measured=section.not_measured,
            )
        )

    if not changed:
        return answer, None

    condensed = render.Answer(
        kind=answer.kind,
        title=answer.title,
        sections=tuple(sections),
        citations=answer.citations,
        not_measured=answer.not_measured,
        degradation=answer.degradation,
    )
    by_reason_all: dict[str, int] = {}
    for row in rows:
        key = str(row["reason_text"] or "no reason recorded")
        by_reason_all[key] = by_reason_all.get(key, 0) + 1
    diagnostics = {
        "gaps_total": len(rows),
        "gaps_by_reason": by_reason_all,
        "note": (
            "the reply names each absence in plain words and lists at most "
            f"{cap} of them (SPEC 4.3 caps a list at 5); every absence, with the "
            "raw error its source returned, is here"
        ),
        "gaps": rows,
    }
    return condensed, diagnostics


def reply_envelope(
    *,
    action: policy_mod.Action,
    answer: render.Answer,
    bag: SlotBag,
    ran_earth_engine: bool,
    parse: Mapping[str, Any] | None = None,
    slot_outcomes: Mapping[str, str] | None = None,
    retrieval: Mapping[str, Any] | None = None,
    run: Mapping[str, Any] | None = None,
    degraded_reason: str | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The uniform shape every turn endpoint returns.

    ``ran_earth_engine`` is stated on every turn rather than inferred, because
    "did that cost quota and touch a satellite" is the one thing SPEC section
    4.4 makes the server responsible for and the frontend should never have to
    deduce it.
    """
    intent = action.intent
    declared = intents_mod.BY_NAME.get(intent) if intent else None
    return {
        "action": action.as_dict(),
        "answer": answer.as_dict(),
        "intent": (
            {
                "name": declared.name,
                "label": declared.label,
                "route": declared.route,
                "required_slots": list(declared.required_slots),
                "optional_slots": list(declared.optional_slots),
            }
            if declared is not None
            else None
        ),
        "ran_earth_engine": bool(ran_earth_engine),
        "awaiting": _awaiting(action),
        "bag": bag.to_dict(),
        "parse": dict(parse) if parse is not None else None,
        "slot_outcomes": dict(slot_outcomes) if slot_outcomes is not None else None,
        "retrieval": dict(retrieval) if retrieval is not None else None,
        "run": dict(run) if run is not None else None,
        # Derived, not trusted from the caller. `degraded_reason` only ever
        # reached here from the RAG path, so a run whose own ledger had already
        # concluded it was degraded still came back degraded=false and the
        # SPEC section 9 capability strip never saw the verdict. The run's own
        # degradation verdict wins; degraded_reason stays the readable half.
        "degraded": _is_degraded(run, degraded_reason),
        "degraded_reason": degraded_reason,
        # Operator detail the UI reveals on demand: the full gap list with the
        # raw error each source returned. Never rendered into the farmer's prose.
        "diagnostics": dict(diagnostics) if diagnostics is not None else None,
    }


def _is_degraded(run: Mapping[str, Any] | None, degraded_reason: str | None) -> bool:
    """Whether this turn is degraded, from the evidence rather than the caller.

    A stated ``degraded_reason`` always counts. Beyond that, a run carries its
    own verdict at ``run['answer']['degradation']['degraded']``, computed by
    :meth:`Ledger.degradation` from the facts and named gaps the turn actually
    collected. Trusting only the caller's string meant a run that measured
    nothing still reported ``degraded: false``.
    """
    if degraded_reason:
        return True
    if not run:
        return False
    answer = run.get("answer")
    if not isinstance(answer, Mapping):
        return False
    degradation = answer.get("degradation")
    if not isinstance(degradation, Mapping):
        return False
    return bool(degradation.get("degraded"))


def _awaiting(action: policy_mod.Action) -> dict[str, Any]:
    """What the UI must collect next, named rather than implied by the prompt."""
    return {
        "kind": action.kind,
        "slot": action.slot,
        "missing_slots": list(action.missing_slots),
        "unconfirmed_slots": list(action.unconfirmed_slots),
        "options": list(action.options),
        # The run endpoint is the only thing that may spend quota, and the UI
        # should only offer the button when the policy already said yes.
        "may_run": bool(action.authorises_earth_engine),
    }
