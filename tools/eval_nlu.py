"""Evaluate the NLU cascade against a labelled question set (SPEC section 10).

    python tools/eval_nlu.py                      # committed synthetic set
    python tools/eval_nlu.py --set both           # + the real corpus, if present
    python tools/eval_nlu.py --show-errors        # list the synthetic misses
    python tools/eval_nlu.py --min-accuracy 0.65  # tighten the CI gate

Reports per-intent precision / recall / F1, a confusion matrix, overall
accuracy, a breakdown by the cascade tier that decided (SPEC section 5.1: the
tiers *are* the design, so an aggregate that hides them hides the thing being
measured), and how often the confidence floor sent a question to clarify.

**Why the committed set is synthetic.** SPEC section 10 asks for 98 questions,
78 of them real. The real ones exist only in the gitignored
``Data/Kijani Whatsapp messaged answered by ChatGPT - overview.xlsx``, and
SPEC section 7 is unconditional: *no raw ping, ProfileId or farmer message is
ever committed or logged*. Committing the real questions would break section 7
to satisfy section 10, so what is committed is 98 questions written fresh in
the shapes and the distribution section 1.2 describes -- 72% knowledge work,
Earth Engine as the minority path, most questions naming no location and many
naming no crop, some Swahili -- and never copied from the corpus.

``--set real`` reads the real questions straight from ``Data/`` when the
developer has it locally. Nothing from that file is ever printed, written to
the JSON report, or logged: only counts. The real set carries no intent column,
so it is scored only when the developer supplies their own labels sidecar
(``--real-labels``, gitignored); without one it still reports the routing
distribution, the tier mix and the clarify rate on real traffic, which is what
section 1.2's percentages can actually be checked against.

**Offline.** Nothing here reaches the network: no Earth Engine (SPEC section
4.4 -- this is NLU, there is nothing to run), and the encoder is read from the
local Hugging Face cache. Run with ``CROPUP_EE_ENABLED=0 HF_HUB_OFFLINE=1`` to
prove it. **Torch is never imported** (SPEC section 1.1); the run fails loudly
if anything pulls it in.

Exit codes: ``0`` pass, ``1`` accuracy below the gate, ``2`` the set could not
be loaded, ``3`` the encoder was unavailable so only the rule tier ran, ``4``
torch was imported.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from cropup import config  # noqa: E402
from cropup.nlu import classify, intents as intents_mod  # noqa: E402

SYNTHETIC_SET = Path(REPO) / "tools" / "data" / "nlu_eval_synthetic.jsonl"
REAL_XLSX = Path(REPO) / "Data" / "Kijani Whatsapp messaged answered by ChatGPT - overview.xlsx"
REAL_QUESTION_HEADER = "Question ChatGPT"  # the cleaned-up question column
REAL_FALLBACK_HEADER = "Original question"

# The CI gate. It is a regression tripwire, not a quality bar: the set measured
# 59.2% on the cascade as shipped, so the gate sits a few points under that to
# catch a change that makes routing worse. Raising it is a task for the router,
# not for the question set -- rewriting questions until the number looks better
# would measure nothing.
DEFAULT_MIN_ACCURACY = 0.55

SOURCE_SYNTHETIC = "synthetic"
SOURCE_REAL = "real"

# SPEC 1.2, as fractions of the 78 real questions.
SPEC_RAG_SHARE = 56 / 78
SPEC_EE_SHARE = 22 / 78


# --------------------------------------------------------------------------
# the sets
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """One evaluation question. ``intent`` is ``None`` when it is unlabelled."""

    id: str
    text: str
    intent: str | None
    source: str
    lang: str = "en"
    names_crop: bool | None = None
    names_location: bool | None = None

    @property
    def publishable(self) -> bool:
        """True only for text that may be printed. Real farmer text never may."""
        return self.source == SOURCE_SYNTHETIC


def load_synthetic(path: Path | None = None) -> list[Item]:
    # Resolved at call time, not at def time, so a test can repoint the
    # constant and actually exercise the missing-file branch.
    path = path or SYNTHETIC_SET
    if not path.exists():
        raise FileNotFoundError(f"the committed synthetic set is missing: {path}")
    items: list[Item] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
        items.append(
            Item(
                id=row["id"],
                text=row["text"],
                intent=row["intent"],
                source=SOURCE_SYNTHETIC,
                lang=row.get("lang", "en"),
                names_crop=row.get("names_crop"),
                names_location=row.get("names_location"),
            )
        )
    _validate_synthetic(items, path)
    return items


def _validate_synthetic(items: Sequence[Item], path: Path) -> None:
    """A bad edit to the set must fail here, not quietly change the score."""
    if not items:
        raise ValueError(f"{path} is empty")
    known = set(intents_mod.INTENT_NAMES)
    seen_text: dict[str, str] = {}
    seeds = {seed: name for name, seed in intents_mod.seed_pairs()}
    for item in items:
        if item.intent not in known:
            raise ValueError(f"{item.id}: unknown intent {item.intent!r}")
        if item.text in seen_text:
            raise ValueError(f"{item.id}: duplicate of {seen_text[item.text]}")
        seen_text[item.text] = item.id
        # Scoring the router on its own training data would measure nothing.
        if item.text in seeds:
            raise ValueError(f"{item.id}: is a seed utterance of {seeds[item.text]}")
    missing = known - {item.intent for item in items}
    if missing:
        raise ValueError(f"{path}: no question for intent(s) {sorted(missing)}")


def load_real(
    xlsx: Path | None = None, labels_path: Path | None = None
) -> tuple[list[Item], str | None]:
    """Read the real corpus from the gitignored workbook. Never prints it.

    Returns (items, label source description). ``intent`` is ``None`` on every
    item unless a labels sidecar supplies one.
    """
    xlsx = xlsx or REAL_XLSX  # resolved at call time; see load_synthetic
    if not xlsx.exists():
        raise FileNotFoundError(str(xlsx))
    try:
        import openpyxl  # noqa: PLC0415  -- a dev-only dependency
    except ImportError as exc:  # pragma: no cover - depends on the local env
        raise RuntimeError(f"reading {xlsx.name} needs openpyxl ({exc})") from exc

    workbook = openpyxl.load_workbook(xlsx, read_only=True, data_only=False)
    sheet = workbook.worksheets[0]
    rows = list(sheet.iter_rows(values_only=True))
    header_row = None
    for index, row in enumerate(rows):
        cells = [str(c).strip() if c is not None else "" for c in row]
        if REAL_FALLBACK_HEADER in cells:
            header_row = index
            headers = cells
            break
    if header_row is None:
        raise ValueError(f"{xlsx.name}: no {REAL_FALLBACK_HEADER!r} header row")
    column = (
        headers.index(REAL_QUESTION_HEADER)
        if REAL_QUESTION_HEADER in headers
        else headers.index(REAL_FALLBACK_HEADER)
    )
    fallback = headers.index(REAL_FALLBACK_HEADER)

    labels = _load_labels(labels_path) if labels_path else {}
    items: list[Item] = []
    for offset, row in enumerate(rows[header_row + 1 :], start=header_row + 2):
        value = row[column] if column < len(row) else None
        if value is None and fallback < len(row):
            value = row[fallback]
        if not isinstance(value, str) or not value.strip():
            continue
        item_id = f"real-r{offset}"
        items.append(
            Item(
                id=item_id,
                text=value.strip(),
                intent=labels.get(item_id),
                source=SOURCE_REAL,
            )
        )
    workbook.close()
    if not items:
        raise ValueError(f"{xlsx.name}: found no questions under the header row")
    described = None
    if labels_path:
        matched = sum(1 for i in items if i.intent)
        described = f"{labels_path} ({matched}/{len(items)} matched)"
    return items, described


def _load_labels(path: Path) -> dict[str, str]:
    """id,intent CSV or JSON object. Kept out of the repo: it is per-developer."""
    if not path.exists():
        raise FileNotFoundError(f"labels file not found: {path}")
    known = set(intents_mod.INTENT_NAMES)
    labels: dict[str, str] = {}
    if path.suffix.lower() == ".json":
        labels = {str(k): str(v) for k, v in json.loads(path.read_text("utf-8")).items()}
    else:
        import csv  # noqa: PLC0415

        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row.get("id") or row.get("row") or "").strip()
                value = (row.get("intent") or row.get("label") or "").strip()
                if key and value:
                    labels[key if key.startswith("real-") else f"real-r{key}"] = value
    unknown = set(labels.values()) - known
    if unknown:
        raise ValueError(f"{path}: unknown intent(s) {sorted(unknown)}")
    return labels


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


@dataclass
class Result:
    item: Item
    predicted: str
    route: str
    tier: str
    score: float
    score_kind: str
    below_floor: bool
    floor_reason: str | None
    runner_up: str | None
    elapsed_ms: float

    @property
    def correct(self) -> bool | None:
        if self.item.intent is None:
            return None
        return self.predicted == self.item.intent


@dataclass
class PerIntent:
    support: int = 0
    predicted: int = 0
    tp: int = 0

    @property
    def precision(self) -> float | None:
        return self.tp / self.predicted if self.predicted else None

    @property
    def recall(self) -> float | None:
        return self.tp / self.support if self.support else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None
        return 2 * p * r / (p + r)


@dataclass
class Report:
    label: str
    results: list[Result] = field(default_factory=list)
    encoder_available: bool | None = None

    @property
    def labelled(self) -> list[Result]:
        return [r for r in self.results if r.item.intent is not None]

    @property
    def accuracy(self) -> float | None:
        scored = self.labelled
        if not scored:
            return None
        return sum(1 for r in scored if r.correct) / len(scored)

    def per_intent(self) -> dict[str, PerIntent]:
        table = {name: PerIntent() for name in intents_mod.INTENT_NAMES}
        for result in self.labelled:
            table[result.item.intent].support += 1
            table[result.predicted].predicted += 1
            if result.correct:
                table[result.predicted].tp += 1
        return table

    def confusion(self) -> dict[str, Counter]:
        matrix: dict[str, Counter] = {n: Counter() for n in intents_mod.INTENT_NAMES}
        for result in self.labelled:
            matrix[result.item.intent][result.predicted] += 1
        return matrix

    def by_tier(self) -> dict[str, dict[str, int]]:
        tiers: dict[str, dict[str, int]] = {}
        for result in self.results:
            bucket = tiers.setdefault(result.tier, {"n": 0, "scored": 0, "correct": 0})
            bucket["n"] += 1
            if result.correct is not None:
                bucket["scored"] += 1
                bucket["correct"] += int(result.correct)
        return tiers


FLOOR_CONFIDENCE = "below the cosine confidence floor"
FLOOR_MARGIN = "two intents inside the margin"
FLOOR_NO_MODEL = "no encoder, and the rules did not clear their floors"


def _floor_reason(c: classify.Classification, settings: config.Settings) -> str | None:
    """Which guard fired. They are different defects and must not be summed blind.

    The confidence floor firing means the question resembled no intent at all.
    The margin floor firing means two intents could not be separated -- which
    for a band of Earth Engine intents is the SPEC 4.4 guard doing its job, and
    for anything else is a clarify loop on an answerable question.
    """
    if not c.below_floor:
        return None
    if c.score_kind != classify.SCORE_COSINE:
        return FLOOR_NO_MODEL
    if c.score < settings.intent_confidence_floor:
        return FLOOR_CONFIDENCE
    return FLOOR_MARGIN


def run(items: Sequence[Item], label: str, settings: config.Settings) -> Report:
    report = Report(label=label)
    for item in items:
        c = classify.classify(item.text, settings)
        if c.model_available is not None:
            report.encoder_available = c.model_available
        report.results.append(
            Result(
                item=item,
                predicted=c.intent,
                route=c.route,
                tier=c.tier,
                score=c.score,
                score_kind=c.score_kind,
                below_floor=c.below_floor,
                floor_reason=_floor_reason(c, settings),
                runner_up=c.runner_up,
                elapsed_ms=c.elapsed_ms,
            )
        )
    return report


# --------------------------------------------------------------------------
# printing -- real farmer text never reaches here
# --------------------------------------------------------------------------

SHORT = {
    "field_health_check": "FHC",
    "crop_problem_diagnosis": "DIA",
    "irrigation_advice": "IRR",
    "crop_selection": "SEL",
    "fertilizer_advice": "FRT",
    "soil_fertility_management": "SOI",
    "seed_variety_selection": "SED",
    "crop_management_practice": "MGT",
    "market_and_inputs_supply": "MKT",
    "livestock_and_adjacent": "LIV",
    "agronomy_concept_explainer": "CON",
    "out_of_scope_or_unclear": "UNC",
}


def _pct(value: float | None) -> str:
    return "  --  " if value is None else f"{value * 100:5.1f}%"


def print_report(report: Report, *, show_errors: bool) -> None:
    line = "=" * 78
    print(f"\n{line}\n{report.label}\n{line}")
    scored = report.labelled
    print(f"questions: {len(report.results)}   labelled: {len(scored)}")
    if report.encoder_available is False:
        print("ENCODER UNAVAILABLE -- only the rule tier ran; these numbers are not the router.")

    if scored:
        table = report.per_intent()
        print("\nper intent (labelled only)")
        print(f"  {'intent':<28}{'route':<14}{'sup':>4}{'pred':>6}"
              f"{'precision':>11}{'recall':>9}{'f1':>8}")
        macro = []
        for name in intents_mod.INTENT_NAMES:
            stats = table[name]
            if stats.support == 0 and stats.predicted == 0:
                continue
            print(
                f"  {name:<28}{intents_mod.route_of(name):<14}{stats.support:>4}"
                f"{stats.predicted:>6}{_pct(stats.precision):>11}"
                f"{_pct(stats.recall):>9}{_pct(stats.f1):>8}"
            )
            if stats.support:
                macro.append((stats.precision or 0.0, stats.recall or 0.0, stats.f1 or 0.0))
        if macro:
            n = len(macro)
            print(
                f"  {'MACRO AVG (' + str(n) + ' intents with support)':<42}"
                f"{_pct(sum(m[0] for m in macro) / n):>11}"
                f"{_pct(sum(m[1] for m in macro) / n):>9}"
                f"{_pct(sum(m[2] for m in macro) / n):>8}"
            )
        print(f"\n  overall accuracy: {_pct(report.accuracy).strip()} "
              f"({sum(1 for r in scored if r.correct)}/{len(scored)})")

        route_hits = sum(
            1 for r in scored if r.route == intents_mod.route_of(r.item.intent)
        )
        print(f"  route accuracy:   {_pct(route_hits / len(scored)).strip()} "
              f"({route_hits}/{len(scored)}) -- earth_engine / rag / clarify")
        wrong_ee = [
            r for r in scored
            if r.route == intents_mod.ROUTE_EARTH_ENGINE
            and intents_mod.route_of(r.item.intent) != intents_mod.ROUTE_EARTH_ENGINE
        ]
        print(f"  sent to Earth Engine that should not have been: {len(wrong_ee)}")

        print("\nconfusion matrix (rows = gold, columns = predicted)")
        header = "".join(f"{SHORT[n]:>5}" for n in intents_mod.INTENT_NAMES)
        print(f"  {'':<28}{header}")
        matrix = report.confusion()
        for name in intents_mod.INTENT_NAMES:
            row = matrix[name]
            if not sum(row.values()):
                continue
            cells = "".join(
                f"{(row.get(p) or ''):>5}" if name != p else f"{('[' + str(row.get(p, 0)) + ']'):>5}"
                for p in intents_mod.INTENT_NAMES
            )
            print(f"  {name:<28}{cells}")

    print("\nby cascade tier (SPEC 5.1)")
    print(f"  {'tier':<12}{'n':>5}{'share':>8}{'scored':>8}{'accuracy':>10}")
    total = len(report.results) or 1
    for tier in (classify.TIER_RULES, classify.TIER_CENTROID, classify.TIER_NLI,
                 classify.TIER_FLOOR):
        bucket = report.by_tier().get(tier)
        if not bucket:
            continue
        acc = bucket["correct"] / bucket["scored"] if bucket["scored"] else None
        print(f"  {tier:<12}{bucket['n']:>5}{bucket['n'] / total * 100:>7.1f}%"
              f"{bucket['scored']:>8}{_pct(acc):>10}")

    floored = [r for r in report.results if r.below_floor]
    print("\nconfidence floor -> clarify (SPEC 5.1: asking beats misrouting)")
    print(f"  routed to clarify by the floor: {len(floored)}/{len(report.results)} "
          f"({len(floored) / total * 100:.1f}%)")
    for reason, n in Counter(r.floor_reason for r in floored).most_common():
        print(f"    {reason:<44}{n:>4}")
    floored_scored = [r for r in floored if r.item.intent is not None]
    if floored_scored:
        justified = [r for r in floored_scored if r.item.intent == intents_mod.CLARIFY_INTENT]
        unjustified = [r for r in floored_scored if r.item.intent != intents_mod.CLARIFY_INTENT]
        rag_loss = [
            r for r in unjustified
            if intents_mod.route_of(r.item.intent) == intents_mod.ROUTE_RAG
        ]
        print(f"    genuinely unclear:            {len(justified)}")
        print(f"    answerable but asked anyway:  {len(unjustified)}"
              f"  (of which {len(rag_loss)} were RAG knowledge questions)")
        missed = [
            r for r in report.results
            if r.item.intent == intents_mod.CLARIFY_INTENT and not r.below_floor
            and r.predicted != intents_mod.CLARIFY_INTENT
        ]
        print(f"    unclear questions answered anyway: {len(missed)}")

    langs = {r.item.lang for r in scored}
    if len(langs) > 1:
        # The seed utterances are English and the centroids are built from them,
        # so a Swahili question is measured against English vectors. Whether
        # that works is not a detail of the score, it is the score for part of
        # the traffic (SPEC 5.3 keeps Swahili in the crop vocabulary).
        print("\nby language of the question")
        print(f"  {'lang':<6}{'n':>4}{'accuracy':>10}{'to clarify':>12}")
        for lang in sorted(langs):
            rows = [r for r in scored if r.item.lang == lang]
            acc = sum(1 for r in rows if r.correct) / len(rows)
            floored_n = sum(1 for r in rows if r.below_floor)
            print(f"  {lang:<6}{len(rows):>4}{_pct(acc):>10}"
                  f"{f'{floored_n}/{len(rows)}':>12}")

    latencies = sorted(r.elapsed_ms for r in report.results)
    if latencies:
        print(f"\nlatency: median {latencies[len(latencies) // 2]:.1f} ms, "
              f"max {latencies[-1]:.1f} ms")

    if show_errors:
        misses = [r for r in scored if r.correct is False]
        printable = [r for r in misses if r.item.publishable]
        print(f"\nmisclassified: {len(misses)}"
              + ("" if len(printable) == len(misses)
                 else f" ({len(misses) - len(printable)} withheld: real farmer text)"))
        for r in printable:
            print(f"  {r.item.id}  gold {SHORT[r.item.intent]} -> got {SHORT[r.predicted]} "
                  f"[{r.tier} {r.score:.3f}]")
            print(f"    {r.item.text}")


def print_composition(items: Sequence[Item], label: str) -> None:
    """What the set is made of, against SPEC 1.2. Counts only."""
    total = len(items) or 1
    routes = Counter(intents_mod.route_of(i.intent) for i in items if i.intent)
    print(f"\n{label} composition ({len(items)} questions)")
    if routes:
        for route in intents_mod.ROUTES:
            share = routes.get(route, 0) / total
            print(f"  {route:<14}{routes.get(route, 0):>4}{share * 100:>7.1f}%")
        print(f"  SPEC 1.2 on real traffic: rag {SPEC_RAG_SHARE * 100:.0f}%, "
              f"earth_engine {SPEC_EE_SHARE * 100:.0f}%")
    tagged = [i for i in items if i.names_crop is not None]
    if tagged:
        no_crop = sum(1 for i in tagged if not i.names_crop)
        no_loc = sum(1 for i in tagged if not i.names_location)
        swahili = sum(1 for i in items if i.lang == "sw")
        print(f"  names no crop: {no_crop}/{len(tagged)} "
              f"(real corpus: 24/78)")
        print(f"  names no location: {no_loc}/{len(tagged)} "
              f"(real corpus: ~60/78)")
        print(f"  Swahili: {swahili}/{len(items)}")


def print_unlabelled(report: Report) -> None:
    """Real traffic with no labels: distribution only, no text, no score."""
    total = len(report.results) or 1
    print(f"\n{'=' * 78}\n{report.label}\n{'=' * 78}")
    print(f"questions: {len(report.results)}  (unlabelled -- no precision/recall to report)")
    print("no question from this set is printed, logged or written to the report file.")
    routes = Counter(r.route for r in report.results)
    print("\nwhere the router sent them")
    for route in intents_mod.ROUTES:
        n = routes.get(route, 0)
        print(f"  {route:<14}{n:>4}{n / total * 100:>7.1f}%")
    print(f"  SPEC 1.2 says real traffic is rag {SPEC_RAG_SHARE * 100:.0f}% / "
          f"earth_engine {SPEC_EE_SHARE * 100:.0f}%")
    print("\npredicted intents")
    for name, n in Counter(r.predicted for r in report.results).most_common():
        print(f"  {name:<30}{n:>4}{n / total * 100:>7.1f}%")
    print("\nby cascade tier")
    for tier, bucket in sorted(report.by_tier().items()):
        print(f"  {tier:<12}{bucket['n']:>4}{bucket['n'] / total * 100:>7.1f}%")
    floored = [r for r in report.results if r.below_floor]
    print(f"\nconfidence floor -> clarify: {len(floored)}/{len(report.results)} "
          f"({len(floored) / total * 100:.1f}%)")
    for reason, n in Counter(r.floor_reason for r in floored).most_common():
        print(f"    {reason:<44}{n:>4}")


def as_json(report: Report) -> dict[str, Any]:
    """Machine-readable aggregates. Real farmer text is never included."""
    table = report.per_intent()
    payload: dict[str, Any] = {
        "set": report.label,
        "questions": len(report.results),
        "labelled": len(report.labelled),
        "accuracy": report.accuracy,
        "encoder_available": report.encoder_available,
        "per_intent": {
            name: {
                "support": s.support,
                "predicted": s.predicted,
                "tp": s.tp,
                "precision": s.precision,
                "recall": s.recall,
                "f1": s.f1,
            }
            for name, s in table.items()
            if s.support or s.predicted
        },
        "confusion": {g: dict(c) for g, c in report.confusion().items() if sum(c.values())},
        "by_tier": report.by_tier(),
        "floor_to_clarify": sum(1 for r in report.results if r.below_floor),
        "floor_reasons": dict(
            Counter(r.floor_reason for r in report.results if r.below_floor)
        ),
    }
    if all(r.item.publishable for r in report.results):
        payload["items"] = [
            {"id": r.item.id, "text": r.item.text, "gold": r.item.intent,
             "predicted": r.predicted, "tier": r.tier, "score": round(r.score, 4)}
            for r in report.results
        ]
    else:
        payload["items_withheld"] = "real farmer messages are never written to a file"
    return payload


# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", dest="which", choices=("synthetic", "real", "both"),
                        default="synthetic", help="which question set to evaluate")
    parser.add_argument("--min-accuracy", type=float, default=DEFAULT_MIN_ACCURACY,
                        help=f"CI gate on the synthetic set (default: {DEFAULT_MIN_ACCURACY})")
    parser.add_argument("--real-labels", type=Path, default=None,
                        help="your own id,intent labels for the real set (keep it out of git)")
    parser.add_argument("--show-errors", action="store_true",
                        help="list the misclassified SYNTHETIC questions (real text is withheld)")
    parser.add_argument("--json", type=Path, default=None, help="write the aggregates here")
    args = parser.parse_args(argv)

    settings = config.get_settings()
    reports: list[Report] = []
    exit_code = 0

    if args.which in ("synthetic", "both"):
        try:
            items = load_synthetic()
        except (OSError, ValueError) as exc:
            print(f"cannot load the synthetic set: {exc}", file=sys.stderr)
            return 2
        print_composition(items, "SYNTHETIC SET (committed)")
        report = run(items, "SYNTHETIC SET (98 questions, committed, no farmer text)", settings)
        print_report(report, show_errors=args.show_errors)
        reports.append(report)
        accuracy = report.accuracy or 0.0
        print(f"\ngate: accuracy {accuracy * 100:.1f}% vs "
              f"minimum {args.min_accuracy * 100:.1f}% -> "
              f"{'PASS' if accuracy >= args.min_accuracy else 'FAIL'}")
        if accuracy < args.min_accuracy:
            exit_code = 1
        if report.encoder_available is False:
            exit_code = 3

    if args.which in ("real", "both"):
        try:
            real, label_source = load_real(labels_path=args.real_labels)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"\nREAL SET: not evaluated -- {exc}")
            print("The 78 real questions live only in the gitignored Data/ workbook "
                  "(SPEC 7), so this is expected on a clean checkout.")
            if args.which == "real":
                return 2
            real = []
            label_source = None
        if real:
            report = run(real, f"REAL SET ({len(real)} questions, never printed)", settings)
            if any(r.item.intent for r in report.results):
                print(f"\nreal labels from {label_source}")
                print_report(report, show_errors=args.show_errors)
            else:
                if args.real_labels is None:
                    print("\nREAL SET: no labels supplied (--real-labels), so no "
                          "precision/recall. Distribution only.")
                print_unlabelled(report)
            reports.append(report)

    if args.json:
        args.json.write_text(
            json.dumps([as_json(r) for r in reports], indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json}")

    # SPEC 1.1: the local torch is broken against this NumPy and aborts the
    # process. A dependency that starts importing it must fail here, loudly.
    if "torch" in sys.modules:
        print("FAIL: torch was imported (SPEC 1.1 forbids it)", file=sys.stderr)
        return 4
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
