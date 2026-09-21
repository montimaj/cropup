"""The three-tier intent cascade (SPEC section 5.1).

1. **Rules** -- weighted keyword and regex evidence over the 12 intents. Always
   runs, costs nothing, and short-circuits only on a hit that is both strong
   (``CROPUP_RULE_SCORE_FLOOR``) and clear (``CROPUP_RULE_MARGIN_FLOOR``).
2. **MiniLM centroids** -- the primary router. Leave-one-out over the seed
   utterances measured centroid scoring at 67.3% against kNN's 54.5%, so each
   intent is one averaged vector, not a neighbour vote.
3. **Zero-shot NLI** -- ``MoritzLaurer/deberta-v3-base-zeroshot-v2.0``, 739 MB,
   entailment at logit index 0. A **tie-breaker only**, consulted when tier 2
   cannot separate the top two intents, and **disabled by default**
   (``CROPUP_NLI_ENABLED``).

Below the confidence floor the answer is ``out_of_scope_or_unclear``: asking is
better than misrouting, and a misrouted question is how a farmer ends up being
told to spray for a disease they do not have.

The *margin* floor is read differently, because it guards something else. It is
there so that two intents the centroids cannot separate never become an Earth
Engine run on the wrong reading (SPEC section 4.4). When every intent in that
tie routes to RAG there is no run to guard: all of them read the same corpus, so
the choice of label changes nothing the farmer sees that SPEC section 6's
similarity floor does not already decide by scoring the cards themselves. Those
ties are settled here rather than handed back as a question. 72% of real traffic
is knowledge work (SPEC section 1.2), and a clarify loop between two RAG labels
is the commonest way that traffic fails to get an answer it could have had.

Every result names the tier that decided, the score in that tier's own units,
and the runner-up, so the UI can show *why* a question was routed.

Rule weights carry their own suppressors. Two of them matter:

* the definition frame ("what does X mean", "explain X", and the bare "what is
  X") pushes to the concept explainer and pulls weight off every topical intent,
  so "what does top dressing mean" is not fertiliser advice and "what is NDVI"
  is not a field health check. It pulls harder off the four Earth Engine intents
  than off the RAG ones, because that is the route which would answer a
  definition by demanding a location;
* an animal subject pulls weight off the crop intents, so "my chickens are
  dying" is not a crop diagnosis.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..config import Settings, get_settings
from ..errors import ModelUnavailable
from . import embed, intents as intents_mod, slots as slots_mod
from .intents import CLARIFY_INTENT, INTENT_NAMES, Intent

__all__ = [
    "Rule",
    "IntentScore",
    "Classification",
    "Parse",
    "classify",
    "parse",
    "rule_scores",
    "centroids",
    "classifier_status",
    "reset",
]

TIER_RULES = "rules"
TIER_CENTROID = "centroid"
TIER_NLI = "nli"
TIER_FLOOR = "floor"

SCORE_RULE_WEIGHT = "rule_weight"
SCORE_COSINE = "cosine"
SCORE_ENTAILMENT = "entailment_probability"


@dataclass(frozen=True)
class Rule:
    """One piece of keyword evidence for (or against) one intent."""

    pattern: re.Pattern[str]
    intent: str
    weight: float
    label: str


@dataclass(frozen=True)
class IntentScore:
    intent: str
    score: float
    evidence: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "score": round(self.score, 4),
            "evidence": list(self.evidence),
        }


# --------------------------------------------------------------------------
# tier 1: weighted rules
# --------------------------------------------------------------------------

# (pattern, intent, weight, label). Patterns are matched case-insensitively
# against whitespace-collapsed text; a pattern counts once however often it hits.
_RULE_SPECS: tuple[tuple[str, str, float, str], ...] = (
    # -- field_health_check -------------------------------------------------
    (
        r"\b(ndvi|ndmi|ndre|psri|satellite|imagery|remote sensing)\b",
        "field_health_check", 3.0, "remote-sensing term",
    ),
    (
        r"\bhow (?:is|are) (?:my|the|our)\b",
        "field_health_check", 1.5, "asks how something of theirs is",
    ),
    (
        r"\b(field|farm|shamba|plot|acres?|garden)\b",
        "field_health_check", 1.5, "whole-field subject",
    ),
    (
        r"\b(health|condition|status) (?:check|report|of)\b",
        "field_health_check", 2.5, "status report wanted",
    ),
    (
        r"\b(?:check|scan|look at) (?:my|the|on)\b",
        "field_health_check", 2.0, "asks for an inspection",
    ),
    (
        r"\b(stress|stressed|vigou?r|greenness|biomass|uniform)\b",
        "field_health_check", 2.0, "canopy vigour term",
    ),
    (
        r"\bhow green\b",
        "field_health_check", 2.5, "greenness comparison",
    ),
    (
        r"\bhow (?:stressed|healthy|dry|green|well) (?:is|are)\b",
        "field_health_check", 1.5, "asks for a condition in the abstract",
    ),
    # -- crop_problem_diagnosis --------------------------------------------
    (
        r"\b(?:turning|going|becoming) (?:yellow|brown|black|white|purple|pale|red)\b",
        "crop_problem_diagnosis", 3.0, "colour change",
    ),
    (
        r"\b(leaves|leaf|stems?|plants?|crops?|seedlings?|tips?|pods?|fruits?"
        r"|roots?)\b.{0,15}\b(?:turning|going|becoming) (?:yellow|brown|black|white|purple|pale"
        r"|red)\b",
        "crop_problem_diagnosis", 2.0, "plant part changing colour",
    ),
    (
        r"\b(?:yellow|brown|black|white|purple|pale|red)(?:ing|ish)? (?:spots?|patches|leaves"
        r"|leaf|streaks?|edges?|tips?)\b",
        "crop_problem_diagnosis", 2.5, "discoloured plant part",
    ),
    (
        r"\b(spots?|lesions?|blight|rot|rotting|rust|mildew|mould|mold|wilt|wilting|curl"
        r"|curling|stunted|drooping|holes?|webbing|galls?|chewed)\b",
        "crop_problem_diagnosis", 2.5, "symptom noun",
    ),
    (
        r"\b(?:dry|drying|dried|withering|burnt) (?:at |on |from )?(?:the )?(?:tips?|edges?"
        r"|margins?|leaves|leaf|stems?)\b",
        "crop_problem_diagnosis", 2.0, "drying plant part",
    ),
    (
        r"\b(?:tips?|edges?|margins?|veins?|undersides?) of (?:the )?(?:leaves|leaf)\b",
        "crop_problem_diagnosis", 2.0, "symptom located on the leaf",
    ),
    (
        r"\bmy \w+ (?:are|is) (?:drying|wilting|yellowing|rotting|dying|sick|stunted)\b",
        "crop_problem_diagnosis", 2.5, "their plants are failing",
    ),
    (
        r"\b(pests?|insects?|worms?|caterpillars?|aphids?|armyworm|borers?|weevils?|mites?"
        r"|thrips|whitefl(?:y|ies))\b",
        "crop_problem_diagnosis", 2.5, "pest named",
    ),
    (
        r"\b(disease|diseased|infected|infection|infest\w*|fungus|fungal|virus)\b",
        "crop_problem_diagnosis", 2.5, "disease named",
    ),
    (
        r"\bwhat(?:'s| is)? (?:wrong|happening|attacking|eating|causing)\b",
        "crop_problem_diagnosis", 3.0, "asks what is wrong",
    ),
    (
        r"\b(cobs?|pods?|tubers?|bulbs?|berries|grains?|husks?|stems?|leaves|leaf|flowers?)\b",
        "crop_problem_diagnosis", 0.5, "a plant organ is named",
    ),
    # -- irrigation_advice --------------------------------------------------
    (
        r"\birrigat\w*\b",
        "irrigation_advice", 3.0, "irrigation named",
    ),
    (
        r"\bwater(?:s|ed|ing)?\b",
        "irrigation_advice", 2.0, "watering named",
    ),
    (
        r"\bshould i (?:water|irrigate)\b",
        "irrigation_advice", 2.0, "asks whether to water",
    ),
    (
        r"\b(?:soil )?moisture\b",
        "irrigation_advice", 2.0, "soil moisture",
    ),
    (
        r"\b(drip|sprinkler|furrow|watering can|water pump)\b",
        "irrigation_advice", 2.0, "irrigation equipment",
    ),
    (
        r"\bhow (?:much|often) (?:water|to water)\b",
        "irrigation_advice", 2.0, "asks water amount",
    ),
    # -- crop_selection -----------------------------------------------------
    (
        r"\bwhich crops?\b",
        "crop_selection", 2.0, "asks which crop",
    ),
    (
        r"\bwhat (?:should|can|could) i (?:plant|grow)\b",
        "crop_selection", 3.0, "asks what to plant",
    ),
    (
        r"\b(thrives?|thriving|grows? (?:well|best)|does? well|suitab\w*|suited)\b",
        "crop_selection", 2.0, "suitability wording",
    ),
    (
        r"\bbest crops?\b",
        "crop_selection", 2.5, "asks for the best crop",
    ),
    (
        r"\bis (?:my|the|this) (?:land|soil|farm|area|field) suitable\b",
        "crop_selection", 3.0, "asks if land suits a crop",
    ),
    (
        r"\bworth planting\b",
        "crop_selection", 2.0, "asks what is worth planting",
    ),
    # -- fertilizer_advice --------------------------------------------------
    (
        r"\bfertili[sz]ers?\b",
        "fertilizer_advice", 3.0, "fertiliser named",
    ),
    (
        r"\b(urea|dap|npk|mop|ammonium|sulphate of ammonia|nitrate|phosphate|foliar feed"
        r"|basal dressing)\b",
        "fertilizer_advice", 2.5, "fertiliser product",
    ),
    (
        r"\btop ?dress\w*\b",
        "fertilizer_advice", 2.5, "top dressing",
    ),
    (
        r"\b(nitrogen|phosphorus|potassium|potash)\b",
        "fertilizer_advice", 1.5, "nutrient named",
    ),
    (
        r"\b(?:kg|kgs|bags?|grams?|handfuls?) per (?:acre|hectare|ha|plant|hole)\b",
        "fertilizer_advice", 1.5, "application rate",
    ),
    (
        r"\b(dose|dosage|second dressing|split application)\b",
        "fertilizer_advice", 1.5, "dose wording",
    ),
    # -- soil_fertility_management ------------------------------------------
    (
        r"\bsoil (fertility|health|test|testing|ph|structure|organic)\b",
        "soil_fertility_management", 3.0, "soil property",
    ),
    (
        r"\b(acidic|acidity|alkaline|lime|liming)\b",
        "soil_fertility_management", 2.5, "soil acidity",
    ),
    (
        r"\b(manure|compost|organic matter|cover crops?|green manure)\b",
        "soil_fertility_management", 2.0, "organic amendment",
    ),
    (
        r"\b(mulch|mulching)\b",
        "soil_fertility_management", 2.0, "mulching",
    ),
    (
        r"\berosion\b",
        "soil_fertility_management", 3.0, "erosion",
    ),
    (
        r"\bsoil\b",
        "soil_fertility_management", 1.0, "soil mentioned",
    ),
    (
        r"\bfertility\b",
        "soil_fertility_management", 2.0, "fertility named",
    ),
    (
        r"\b(lost|losing|poor|tired|exhausted|depleted|infertile)\b.{0,20}\b(soil|land"
        r"|fertility)\b",
        "soil_fertility_management", 2.5, "the soil has run down",
    ),
    (
        r"\bimprove\b.{0,25}\b(soil|fertility|land)\b",
        "soil_fertility_management", 2.5, "asks to improve the soil",
    ),
    # -- seed_variety_selection ---------------------------------------------
    (
        r"\b(variety|varieties|cultivars?|hybrids?|opv|open.pollinated|certified seed)\b",
        "seed_variety_selection", 3.0, "variety named",
    ),
    (
        r"\bseeds?\b",
        "seed_variety_selection", 1.5, "seed mentioned",
    ),
    (
        r"\b(matures?|maturity|maturing|drought tolerant|disease resistant|resists?"
        r"|resistant to)\b",
        "seed_variety_selection", 2.0, "variety trait",
    ),
    # -- crop_management_practice -------------------------------------------
    (
        r"\b(spacing|plant(?:ing)? (?:holes?|depth|distance)|how deep|how far apart)\b",
        "crop_management_practice", 3.0, "field geometry",
    ),
    (
        r"\b(weed|weeding|prun(?:e|ing)|thin(?:ning)?|transplant\w*|harden(?:ing)? off"
        r"|stak(?:e|ing)|earthing up|intercrop\w*)\b",
        "crop_management_practice", 2.5, "field operation",
    ),
    (
        r"\b(?:when (?:is|to|should)|what time)\b.{0,25}\b(plant|sow|harvest|transplant"
        r"|weed)\w*\b",
        "crop_management_practice", 2.5, "timing of an operation",
    ),
    (
        r"\b(after harvest|store|storage|storing|curing|shelling|threshing|post.harvest)\b",
        "crop_management_practice", 2.0, "post-harvest handling",
    ),
    # -- market_and_inputs_supply -------------------------------------------
    (
        r"\b(price|prices|pricing|cost|costs|market|buyers?|agro.?dealer|cooperative|loan"
        r"|credit|subsid\w*)\b",
        "market_and_inputs_supply", 2.5, "market term",
    ),
    (
        r"\bwhere (?:can|do) i (?:buy|sell|get|find|order)\b",
        "market_and_inputs_supply", 3.0, "asks where to buy or sell",
    ),
    (
        r"\b(selling for|sell my|buy my)\b",
        "market_and_inputs_supply", 2.5, "trading",
    ),
    (
        r"\bhow much (?:is|are|does)\b",
        "market_and_inputs_supply", 1.5, "asks a price",
    ),
    (
        r"\b(shops?|stores?|stockists?|suppliers?|inputs)\b",
        "market_and_inputs_supply", 2.0, "input supply",
    ),
    (
        r"\bsells?\b",
        "market_and_inputs_supply", 1.5, "someone is selling",
    ),
    # -- livestock_and_adjacent ---------------------------------------------
    (
        r"\b(cows?|cattle|calf|calves|goats?|sheep|chickens?|chicks?|poultry|layers|broilers?"
        r"|pigs?|donkeys?|rabbits?|fish|tilapia|fish pond|bees?|beekeeping|hives?|dairy|milk)\b",
        "livestock_and_adjacent", 3.0, "animal named",
    ),
    (
        r"\b(vaccines?|vaccinat\w*|ticks?|deworm\w*|newcastle)\b",
        "livestock_and_adjacent", 2.5, "animal health",
    ),
    (
        r"\b(fodder|napier|silage|feeds?|feeding)\b",
        "livestock_and_adjacent", 1.5, "animal feed",
    ),
    # -- agronomy_concept_explainer -----------------------------------------
    (
        r"\b(concept|terminology|meaning of|definition)\b",
        "agronomy_concept_explainer", 2.0, "asks for a meaning",
    ),
    # -- out_of_scope_or_unclear --------------------------------------------
    (
        r"^\s*(hi|hello|hey|habari|mambo|salama|good (morning|afternoon|evening))\b",
        "out_of_scope_or_unclear", 3.0, "greeting",
    ),
    (
        r"\b(thank you|thanks|asante)\b",
        "out_of_scope_or_unclear", 2.5, "thanks",
    ),
    (
        r"\b(who (?:built|made|are) you|what can you do|how do you work)\b",
        "out_of_scope_or_unclear", 3.0, "asks about the app",
    ),
    (
        r"\b(phone|sim card|battery|password|bus|taxi|football|politics)\b",
        "out_of_scope_or_unclear", 2.5, "off-topic noun",
    ),
)

# "what does X mean" is a definition question whatever X is, so the frame both
# scores the explainer and takes the same evidence away from every topical
# intent it would otherwise have triggered.
#
# The bare form -- "what is NDVI", "what is drip irrigation" -- carries no
# "mean" and no "explain", and it is the commonest way a definition is actually
# asked. It is recognised by shape: "what is/are" followed by a short term and
# then the end of the message. A question that keeps going ("what is the price
# of maize in dodoma market", "what is a good crop to rotate into after maize
# here") is not a definition and does not match, and a handful of heads that
# open a field question rather than a term -- wrong, happening, my, best, price
# -- are excluded outright.
_BARE_TERM_FRAME = (
    r"\bwhat(?:'s|s| is| are)\s+(?:an?\s+|the\s+)?"
    r"(?!wrong\b|happening\b|this\b|that\b|it\b|my\b|our\b|your\b|best\b|good\b"
    r"|matter\b|price\b|cost\b|worth\b|left\b|available\b)"
    # A term is a noun phrase. A preposition means the question has a subject
    # somewhere -- "the weather in paris", "the price of maize" -- and is not a
    # request for a definition.
    r"\w[\w-]*(?:\s+(?!in\b|of\b|for\b|at\b|on\b|to\b|near\b|from\b|with\b|about\b)"
    r"[\w-]+){0,2}\s*[?.!]*\s*$"
)
_DEFINITION_FRAME = (
    r"(\bwhat (?:do|does|is|are)\b.{0,40}\bmeans?\b"
    r"|\bwhat (?:do|does|is|are)\b.{0,40}\b(?:measures?|stands? for|indicates?)\b"
    r"|\bwhat is meant by\b|\bwhat do people mean by\b"
    r"|\bexplain\b|\bdefine\b|\bdefinition of\b|\bmeaning of\b"
    r"|\bwhat is the difference between\b"
    rf"|{_BARE_TERM_FRAME})"
)
_ANIMAL_SUBJECT = (
    r"\b(cows?|cattle|goats?|sheep|chickens?|chicks?|poultry|layers|broilers?"
    r"|pigs?|donkeys?|rabbits?|fish|tilapia|bees?|hives?)\b"
)
_SUPPRESSED_BY_DEFINITION = tuple(
    name for name in INTENT_NAMES if name not in {"agronomy_concept_explainer", CLARIFY_INTENT}
)
# A definition question has no field, so the four Earth Engine intents are the
# expensive way to be wrong about it: that route demands a location and a
# confirmation (SPEC section 4.4) for a question that needs neither, and the
# farmer is asked to place a pin in order to be told what NDVI stands for.
# Landing on a neighbouring RAG intent instead still returns a citation, so the
# suppression is deliberately asymmetric: hard against Earth Engine, light
# against the other knowledge intents.
_DEFINITION_SUPPRESSION_EE = -3.0
_DEFINITION_SUPPRESSION_RAG = -1.5


def _build_rules() -> tuple[Rule, ...]:
    built = [
        Rule(re.compile(pattern, re.IGNORECASE), intent, weight, label)
        for pattern, intent, weight, label in _RULE_SPECS
    ]
    built.append(
        Rule(re.compile(_DEFINITION_FRAME, re.IGNORECASE), "agronomy_concept_explainer", 4.0,
             "definition frame")
    )
    for name in _SUPPRESSED_BY_DEFINITION:
        earth_engine = intents_mod.route_of(name) == intents_mod.ROUTE_EARTH_ENGINE
        built.append(
            Rule(
                re.compile(_DEFINITION_FRAME, re.IGNORECASE),
                name,
                _DEFINITION_SUPPRESSION_EE if earth_engine else _DEFINITION_SUPPRESSION_RAG,
                "asks for a definition, not a measurement of a field"
                if earth_engine
                else "asks for a definition, not advice on a field",
            )
        )
    for name in ("crop_problem_diagnosis", "field_health_check"):
        built.append(
            Rule(re.compile(_ANIMAL_SUBJECT, re.IGNORECASE), name, -2.0,
                 "the subject is an animal, not a crop")
        )
    unknown = {rule.intent for rule in built} - set(INTENT_NAMES)
    if unknown:  # a typo here would silently disable a rule
        raise ValueError(f"rules reference unknown intents: {sorted(unknown)}")
    return tuple(built)


RULES: tuple[Rule, ...] = _build_rules()

_WHITESPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", text or "").strip()


def rule_scores(text: str) -> tuple[IntentScore, ...]:
    """Weighted keyword evidence per intent, best first.

    Scores are clamped at zero: a suppressor removes evidence, it does not
    create evidence for everything else.
    """
    normalised = _normalise(text)
    totals = {name: 0.0 for name in INTENT_NAMES}
    evidence: dict[str, list[str]] = {name: [] for name in INTENT_NAMES}
    for rule in RULES:
        if rule.pattern.search(normalised):
            totals[rule.intent] += rule.weight
            evidence[rule.intent].append(f"{rule.label} ({rule.weight:+g})")
    ranked = [
        IntentScore(name, max(0.0, totals[name]), tuple(evidence[name]))
        for name in INTENT_NAMES
    ]
    ranked.sort(key=lambda s: (-s.score, s.intent))
    return tuple(ranked)


# --------------------------------------------------------------------------
# tier 2: MiniLM centroids
# --------------------------------------------------------------------------

_LOCK = threading.Lock()
_CENTROIDS: tuple[tuple[str, ...], np.ndarray] | None = None


def centroids(settings: Settings | None = None) -> tuple[tuple[str, ...], np.ndarray] | None:
    """(intent names, centroid matrix), or ``None`` when the encoder is down.

    One mean vector per intent over its seed utterances -- the measured winner
    over kNN (67.3% vs 54.5% leave-one-out).
    """
    global _CENTROIDS
    if _CENTROIDS is not None:
        return _CENTROIDS
    encoder = embed.get_encoder(settings)
    if encoder is None:
        return None
    with _LOCK:
        if _CENTROIDS is not None:
            return _CENTROIDS
        names: list[str] = []
        rows: list[np.ndarray] = []
        for intent in intents_mod.INTENTS:
            vectors = encoder.encode(list(intent.seeds))
            centroid = vectors.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            rows.append(centroid / norm if norm else centroid)
            names.append(intent.name)
        _CENTROIDS = (tuple(names), np.vstack(rows).astype(np.float32))
        return _CENTROIDS


def _centroid_scores(text: str, settings: Settings) -> tuple[IntentScore, ...] | None:
    built = centroids(settings)
    if built is None:
        return None
    names, matrix = built
    vector = embed.encode_one(text, settings)
    if vector is None:
        return None
    similarities = embed.cosine_similarity(vector, matrix)
    ranked = [
        IntentScore(name, float(score), (f"cosine to {len(intents_mod.BY_NAME[name].seeds)} seeds",))
        for name, score in zip(names, similarities)
    ]
    ranked.sort(key=lambda s: (-s.score, s.intent))
    return tuple(ranked)


# --------------------------------------------------------------------------
# tier 3: optional zero-shot NLI tie-breaker
# --------------------------------------------------------------------------

_NLI: tuple[Any, Any] | None = None  # (session, tokenizer)
_NLI_ERROR: str | None = None
_NLI_ATTEMPTED = False


def _load_nli(settings: Settings) -> tuple[Any, Any] | None:
    """Load the DeBERTa zero-shot model, once, and never fatally."""
    global _NLI, _NLI_ERROR, _NLI_ATTEMPTED
    if _NLI is not None:
        return _NLI
    with _LOCK:
        if _NLI is not None:
            return _NLI
        if _NLI_ATTEMPTED:
            return None
        _NLI_ATTEMPTED = True
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            cache_dir = str(settings.model_cache_dir) if settings.model_cache_dir else None
            onnx_path = hf_hub_download(
                repo_id=settings.nli_model_id,
                filename="onnx/model.onnx",
                cache_dir=cache_dir,
                local_files_only=not settings.allow_model_download,
            )
            tokenizer_path = hf_hub_download(
                repo_id=settings.nli_model_id,
                filename="tokenizer.json",
                cache_dir=cache_dir,
                local_files_only=not settings.allow_model_download,
            )
            tokenizer = Tokenizer.from_file(tokenizer_path)
            pad_id = tokenizer.token_to_id("[PAD]")
            if pad_id is None:
                raise ModelUnavailable(settings.nli_model_id, "tokenizer has no [PAD]")
            tokenizer.enable_truncation(max_length=256)
            tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            _NLI = (session, tokenizer)
            _NLI_ERROR = None
        except Exception as exc:
            _NLI = None
            _NLI_ERROR = f"{type(exc).__name__}: {exc}"
        return _NLI


def _nli_scores(
    text: str, candidates: Sequence[Intent], settings: Settings
) -> tuple[IntentScore, ...] | None:
    """Entailment probability for "this message is <intent hypothesis>".

    Entailment is logit index ``CROPUP_NLI_ENTAILMENT_INDEX`` (measured: 0).
    """
    loaded = _load_nli(settings)
    if loaded is None:
        return None
    session, tokenizer = loaded
    try:
        encodings = tokenizer.encode_batch([(text, c.hypothesis) for c in candidates])
        feed = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        names = {i.name for i in session.get_inputs()}
        logits = session.run(None, {k: v for k, v in feed.items() if k in names})[0]
    except Exception:
        return None
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    entailment = probabilities[:, settings.nli_entailment_index]
    total = float(entailment.sum())
    ranked = [
        IntentScore(
            candidate.name,
            float(value / total) if total else 0.0,
            (f"entailment {value:.3f} for: {candidate.hypothesis}",),
        )
        for candidate, value in zip(candidates, entailment)
    ]
    ranked.sort(key=lambda s: (-s.score, s.intent))
    return tuple(ranked)


# --------------------------------------------------------------------------
# the cascade
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Classification:
    """A routed intent plus the evidence that routed it."""

    text: str
    intent: str
    route: str
    tier: str
    score: float
    score_kind: str
    margin: float
    runner_up: str | None
    runner_up_score: float | None
    reason: str
    scores: tuple[IntentScore, ...]
    rule_scores: tuple[IntentScore, ...]
    below_floor: bool = False
    degraded: bool = False
    # True/False once the encoder has been asked; None when a tier-1 hit meant
    # it was never consulted, which is not the same as "the model is fine".
    model_available: bool | None = None
    clarify_options: tuple[str, ...] = ()
    elapsed_ms: float = 0.0

    @property
    def confidence(self) -> float | None:
        """A probability-like number, or ``None``.

        Only the model tiers produce one. A rule weight is not a confidence and
        is not dressed up as one.
        """
        if self.score_kind in (SCORE_COSINE, SCORE_ENTAILMENT):
            return round(self.score, 4)
        return None

    @property
    def runs_earth_engine(self) -> bool:
        return self.route == intents_mod.ROUTE_EARTH_ENGINE

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "intent": self.intent,
            "route": self.route,
            "tier": self.tier,
            "score": round(self.score, 4),
            "score_kind": self.score_kind,
            "confidence": self.confidence,
            "margin": round(self.margin, 4),
            "runner_up": self.runner_up,
            "runner_up_score": round(self.runner_up_score, 4)
            if self.runner_up_score is not None
            else None,
            "reason": self.reason,
            "below_floor": self.below_floor,
            "degraded": self.degraded,
            "model_available": self.model_available,
            "clarify_options": list(self.clarify_options),
            "elapsed_ms": round(self.elapsed_ms, 2),
            "scores": [s.as_dict() for s in self.scores[:5]],
            "rule_scores": [s.as_dict() for s in self.rule_scores if s.score > 0],
        }


def _decide(
    text: str,
    ranked: Sequence[IntentScore],
    *,
    tier: str,
    score_kind: str,
    reason: str,
    rules: Sequence[IntentScore],
    started: float,
    below_floor: bool = False,
    degraded: bool = False,
    model_available: bool | None = None,
    forced_intent: str | None = None,
) -> Classification:
    top = ranked[0]
    # An intent that matched nothing is not a runner-up; reporting one would
    # suggest the router weighed an alternative it never saw evidence for.
    second = ranked[1] if len(ranked) > 1 and ranked[1].score > 0 else None
    name = forced_intent or top.intent
    return Classification(
        text=text,
        intent=name,
        route=intents_mod.route_of(name),
        tier=tier,
        score=top.score,
        score_kind=score_kind,
        margin=top.score - second.score if second else top.score,
        runner_up=second.intent if second else None,
        runner_up_score=second.score if second else None,
        reason=reason,
        scores=tuple(ranked),
        rule_scores=tuple(rules),
        below_floor=below_floor,
        degraded=degraded,
        model_available=model_available,
        clarify_options=tuple(s.intent for s in ranked[:2]) if below_floor else (),
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


def _all_rag(scored: Sequence[IntentScore]) -> bool:
    """True when every intent in ``scored`` answers from the corpus.

    ``False`` for an empty sequence and for any band that holds an Earth Engine
    intent or ``out_of_scope_or_unclear``: both of those are ties worth asking
    about, because one ends in a satellite query and the other in no answer.
    """
    return bool(scored) and all(
        intents_mod.route_of(s.intent) == intents_mod.ROUTE_RAG for s in scored
    )


def classify(text: str, settings: Settings | None = None) -> Classification:
    """Route one farmer message through the cascade."""
    settings = settings or get_settings()
    started = time.perf_counter()
    rules = rule_scores(text)
    if not _normalise(text):
        return _decide(
            text,
            rules,
            tier=TIER_FLOOR,
            score_kind=SCORE_RULE_WEIGHT,
            reason="the message is empty",
            rules=rules,
            started=started,
            below_floor=True,
            forced_intent=CLARIFY_INTENT,
        )
    rule_top, rule_second = rules[0], rules[1]
    rule_margin = rule_top.score - rule_second.score

    if rule_top.score >= settings.rule_score_floor and rule_margin >= settings.rule_margin_floor:
        return _decide(
            text,
            rules,
            tier=TIER_RULES,
            score_kind=SCORE_RULE_WEIGHT,
            reason=(
                f"keyword rules scored {rule_top.score:g} for {rule_top.intent}, "
                + (
                    f"{rule_margin:g} clear of {rule_second.intent}"
                    if rule_second.score > 0
                    else "and nothing else matched"
                )
                + f"; floors are {settings.rule_score_floor:g}/{settings.rule_margin_floor:g}"
            ),
            rules=rules,
            started=started,
        )

    ranked = _centroid_scores(text, settings)
    if ranked is None:
        # No encoder. The rules did not clear their floors, so the honest answer
        # is to ask rather than to accept weaker evidence than normal.
        return _decide(
            text,
            rules,
            tier=TIER_FLOOR,
            score_kind=SCORE_RULE_WEIGHT,
            reason=(
                "the embedding model is unavailable and the keyword rules did not "
                f"clear their floors (top {rule_top.intent} at {rule_top.score:g})"
            ),
            rules=rules,
            started=started,
            below_floor=True,
            degraded=True,
            model_available=False,
            forced_intent=CLARIFY_INTENT,
        )

    top, second = ranked[0], ranked[1]
    margin = top.score - second.score
    # The intents the centroids could not separate from the top.
    band = [s for s in ranked if top.score - s.score <= settings.intent_margin_floor]
    joint_rag = _all_rag(band)

    # The confidence floor answers "does this look like a farming question at
    # all", and it is unchanged for every band, all-RAG included: a message that
    # resembles nothing is a message to ask about. Standing it down was measured
    # to let "i need a loan for school fees" (cosine 0.333) through to retrieval
    # while recovering nothing retrieval could actually answer.
    if top.score < settings.intent_confidence_floor:
        return _decide(
            text,
            ranked,
            tier=TIER_FLOOR,
            score_kind=SCORE_COSINE,
            reason=(
                f"closest intent {top.intent} at cosine {top.score:.3f} is below the "
                f"{settings.intent_confidence_floor:g} floor"
            ),
            rules=rules,
            started=started,
            below_floor=True,
            model_available=True,
            forced_intent=CLARIFY_INTENT,
        )

    if margin >= settings.intent_margin_floor:
        return _decide(
            text,
            ranked,
            tier=TIER_CENTROID,
            score_kind=SCORE_COSINE,
            reason=(
                f"MiniLM centroid: {top.intent} at cosine {top.score:.3f}, "
                f"{margin:.3f} clear of {second.intent}"
            ),
            rules=rules,
            started=started,
            model_available=True,
        )

    contenders = [intents_mod.BY_NAME[s.intent] for s in band][:4]
    if settings.nli_enabled and len(contenders) > 1:
        nli = _nli_scores(text, contenders, settings)
        if nli is not None and nli[0].score >= settings.nli_confidence_floor:
            nli_margin = nli[0].score - nli[1].score if len(nli) > 1 else nli[0].score
            return _decide(
                text,
                nli,
                tier=TIER_NLI,
                score_kind=SCORE_ENTAILMENT,
                reason=(
                    f"centroids tied ({top.intent} {top.score:.3f} vs {second.intent} "
                    f"{second.score:.3f}); NLI entailment picked {nli[0].intent} at "
                    f"{nli[0].score:.3f} ({nli_margin:.3f} clear)"
                ),
                rules=rules,
                started=started,
                model_available=True,
            )

    if joint_rag:
        # The margin floor is a different guard: it stops a misparse becoming an
        # Earth Engine run (SPEC section 4.4). When every intent in the band is
        # a RAG intent there is no run to stop -- they all read the same corpus,
        # and SPEC section 6's similarity floor, which scores the cards rather
        # than the label, still decides whether there is an answer at all. So
        # the tie is settled here instead of being handed to the farmer as a
        # choice between two names for the same shelf. A band holding an Earth
        # Engine intent or the clarify intent falls through to clarify below,
        # exactly as before.
        return _decide(
            text,
            ranked,
            tier=TIER_CENTROID,
            score_kind=SCORE_COSINE,
            reason=(
                f"{top.intent} ({top.score:.3f}) and {second.intent} ({second.score:.3f}) are "
                f"within the {settings.intent_margin_floor:g} margin, but "
                + ", ".join(s.intent for s in band)
                + " all answer from the same cited corpus and none of them runs Earth "
                "Engine, so the tie is not worth a question"
            ),
            rules=rules,
            started=started,
            model_available=True,
        )

    return _decide(
        text,
        ranked,
        tier=TIER_FLOOR,
        score_kind=SCORE_COSINE,
        reason=(
            f"{top.intent} ({top.score:.3f}) and {second.intent} ({second.score:.3f}) are "
            f"within the {settings.intent_margin_floor:g} margin"
            + ("" if settings.nli_enabled else "; the NLI tie-breaker is disabled")
        ),
        rules=rules,
        started=started,
        below_floor=True,
        model_available=True,
        forced_intent=CLARIFY_INTENT,
    )


@dataclass(frozen=True)
class Parse:
    """Intent plus slots for one message: what /api/nlu/parse returns."""

    classification: Classification
    extraction: slots_mod.SlotExtraction

    @property
    def intent(self) -> str:
        return self.classification.intent

    @property
    def route(self) -> str:
        return self.classification.route

    def missing_required_slots(self) -> tuple[str, ...]:
        """Required slots this message did not settle -- what to ask next."""
        intent = intents_mod.get_intent(self.classification.intent)
        return tuple(
            slot for slot in intent.required_slots if self.extraction.best(slot) is None
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.classification.as_dict(),
            "slots": self.extraction.as_dict(),
            "missing_required_slots": list(self.missing_required_slots()),
        }


def parse(text: str, settings: Settings | None = None, **kwargs: Any) -> Parse:
    """Classify and extract slots in one call. Runs no Earth Engine (SPEC 4.4)."""
    settings = settings or get_settings()
    return Parse(
        classification=classify(text, settings),
        extraction=slots_mod.extract_slots(text, settings=settings, **kwargs),
    )


def classifier_status(settings: Settings | None = None) -> dict[str, Any]:
    """Router health for /api/capabilities. Loads nothing."""
    settings = settings or get_settings()
    return {
        "component": "nlu:classify",
        "intents": len(intents_mod.INTENTS),
        "rules": len(RULES),
        "seeds": sum(len(i.seeds) for i in intents_mod.INTENTS),
        "centroids_built": _CENTROIDS is not None,
        "embed": embed.embed_status(settings),
        "nli": {
            "enabled": settings.nli_enabled,
            "model_id": settings.nli_model_id,
            "attempted": _NLI_ATTEMPTED,
            "loaded": _NLI is not None,
            "error": _NLI_ERROR,
            "entailment_index": settings.nli_entailment_index,
        },
        "floors": {
            "rule_score": settings.rule_score_floor,
            "rule_margin": settings.rule_margin_floor,
            "intent_confidence": settings.intent_confidence_floor,
            "intent_margin": settings.intent_margin_floor,
            "nli_confidence": settings.nli_confidence_floor,
        },
    }


def reset() -> None:
    """Drop centroids and the NLI session (tests, tools/)."""
    global _CENTROIDS, _NLI, _NLI_ERROR, _NLI_ATTEMPTED
    with _LOCK:
        _CENTROIDS = None
        _NLI = None
        _NLI_ERROR = None
        _NLI_ATTEMPTED = False
