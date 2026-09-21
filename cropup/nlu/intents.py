"""The 12 intents (SPEC section 5.2) with their seed utterances.

The taxonomy comes from 78 real farmer questions, so the shape of it is not a
guess: only 22/78 can be served by Earth Engine at all, and 56/78 is knowledge
work. Four intents route to EE, seven to RAG, one to clarify.

Seed utterances are training data, and the measured confusion is between
``crop_problem_diagnosis`` / ``fertilizer_advice`` / ``field_health_check``.
They pull apart on *what the farmer is telling you*, not on the crop:

* ``crop_problem_diagnosis`` names a **symptom on the plant** -- a colour, a
  spot, a wilt, a hole, a rot. Something is visibly wrong.
* ``field_health_check`` asks for a **status report** on a whole field with no
  symptom named. Nothing is known to be wrong yet.
* ``fertilizer_advice`` asks for a **product, a rate or a timing**. The farmer
  has already decided to apply something.

Seeds are therefore written so that no diagnosis seed mentions a fertiliser
product, no fertiliser seed describes a symptom, and every field-health seed
asks about the field as a whole. None of them is a verbatim evaluation
sentence.

This module holds no model and reads no file; it is a data table plus its
invariants, checked at import so a bad edit fails loudly rather than quietly
degrading the router.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from ..errors import ConfigError
from .slots import SLOT_NAMES

__all__ = [
    "Intent",
    "INTENTS",
    "INTENT_NAMES",
    "BY_NAME",
    "ROUTE_EARTH_ENGINE",
    "ROUTE_RAG",
    "ROUTE_CLARIFY",
    "ROUTES",
    "CLARIFY_INTENT",
    "get_intent",
    "route_of",
    "intents_for_route",
    "seed_pairs",
    "as_dict",
]

ROUTE_EARTH_ENGINE = "earth_engine"
ROUTE_RAG = "rag"
ROUTE_CLARIFY = "clarify"
ROUTES = (ROUTE_EARTH_ENGINE, ROUTE_RAG, ROUTE_CLARIFY)

CLARIFY_INTENT = "out_of_scope_or_unclear"


@dataclass(frozen=True)
class Intent:
    """One routing destination, its slot frame and its seed utterances."""

    name: str
    label: str  # for the UI
    description: str  # for the clarify prompt and the intent list endpoint
    hypothesis: str  # the tier-3 NLI hypothesis; a full sentence on purpose
    route: str
    required_slots: tuple[str, ...]  # must be filled AND confirmed before acting
    optional_slots: tuple[str, ...]  # used when present, never asked for twice
    seeds: tuple[str, ...]

    @property
    def runs_earth_engine(self) -> bool:
        return self.route == ROUTE_EARTH_ENGINE

    @property
    def slots(self) -> tuple[str, ...]:
        return self.required_slots + self.optional_slots

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "route": self.route,
            "required_slots": list(self.required_slots),
            "optional_slots": list(self.optional_slots),
            "seed_count": len(self.seeds),
        }


INTENTS: tuple[Intent, ...] = (
    # ---------------------------------------------------------------- Earth Engine
    Intent(
        name="field_health_check",
        label="Field health check",
        description="a status report on a whole field, with no symptom named",
        hypothesis="The farmer is asking for an overall condition report on their field.",
        route=ROUTE_EARTH_ENGINE,
        required_slots=("location", "crop"),
        optional_slots=("timeframe",),
        seeds=(
            "how is my maize field doing this week",
            "check the condition of my farm from satellite",
            "is my crop under stress right now",
            "give me an ndvi report for my shamba",
            "how green is my field compared to last month",
            "satellite check on my two acres of beans",
            "is the vigour uniform across my rice plot",
            "scan my field and tell me if anything looks off",
            "what does the imagery say about my crop condition",
            "run a health check on the farm i registered",
        ),
    ),
    Intent(
        name="crop_problem_diagnosis",
        label="Crop problem diagnosis",
        description="a visible symptom on the plant that needs identifying",
        hypothesis="The farmer is describing a visible symptom on their plants and wants to know the cause.",
        route=ROUTE_EARTH_ENGINE,
        required_slots=("crop", "location"),
        optional_slots=("timeframe",),
        seeds=(
            "there are brown spots on my tomato leaves",
            "my beans are wilting even though the soil is wet",
            "something is eating holes in my cabbage leaves",
            "white powder is covering my pumpkin leaves",
            "my cassava stems are rotting at the base",
            "what disease is attacking my banana plants",
            "my coffee berries drop before they ripen",
            "the lower leaves of my sorghum have purple streaks",
            "my cabbage leaves have dry brown edges",
            "my sunflower seedlings are stunted and twisted",
        ),
    ),
    Intent(
        name="irrigation_advice",
        label="Irrigation advice",
        description="whether, when or how much to water",
        hypothesis="The farmer is asking whether or when to water their crop.",
        route=ROUTE_EARTH_ENGINE,
        required_slots=("location", "crop"),
        optional_slots=("timeframe",),
        seeds=(
            "do my tomatoes need watering this evening",
            "how much water does my maize need this week",
            "is the soil moisture enough or should i irrigate",
            "when is the next time i should irrigate my onions",
            "do i still need to irrigate after yesterday's rain",
            "how many times a week should i water my vegetables",
            "is my drip schedule enough in this heat",
            "should i skip watering because rain is coming",
            "my furrow irrigation takes too long, how often is enough",
        ),
    ),
    Intent(
        name="crop_selection",
        label="Crop selection",
        description="which crop suits a place, its climate and its soil",
        hypothesis="The farmer is asking which crop is suitable for their land.",
        route=ROUTE_EARTH_ENGINE,
        required_slots=("location",),
        optional_slots=("crop", "timeframe"),
        seeds=(
            "what should i plant on my farm in dodoma",
            "is my land suitable for growing avocado",
            "which crop grows best in this area",
            "can sunflower do well in singida",
            "what is a good crop to rotate into after maize here",
            "which vegetables suit the climate of morogoro",
            "would coffee be suitable at my altitude",
            "which crops do well in the njombe highlands",
            "i have five acres near the lake, what is worth planting",
        ),
    ),
    # ------------------------------------------------------------------------ RAG
    Intent(
        name="fertilizer_advice",
        label="Fertiliser advice",
        description="which fertiliser to use, how much of it, and when to apply it",
        hypothesis="The farmer is asking which fertiliser to apply, at what rate, or at what time.",
        route=ROUTE_RAG,
        required_slots=(),
        optional_slots=("crop", "location", "timeframe"),
        seeds=(
            "how much urea should i apply per acre",
            "when should i do top dressing on my maize",
            "is dap or npk better at planting",
            "what fertiliser rate do tomatoes need",
            "can i use can instead of urea for top dressing",
            "how do i apply foliar feed to my beans",
            "what npk ratio is recommended for onions",
            "which fertiliser should i buy for my rice field",
            "do i broadcast or place the fertiliser in the planting hole",
        ),
    ),
    Intent(
        name="soil_fertility_management",
        label="Soil fertility management",
        description="soil health itself: pH, organic matter, manure, erosion, testing",
        hypothesis="The farmer is asking how to improve or manage the soil itself.",
        route=ROUTE_RAG,
        required_slots=(),
        optional_slots=("crop", "location"),
        seeds=(
            "how do i improve the fertility of my soil",
            "my soil is too acidic, what should i do",
            "how much lime do i need to raise the soil ph",
            "is farmyard manure better than compost",
            "how do i stop erosion on my sloping field",
            "which cover crop adds nitrogen to the soil",
            "should i do a soil test before the season",
            "my soil is hard and does not hold water",
            "how do i build up organic matter in a tired field",
        ),
    ),
    Intent(
        name="seed_variety_selection",
        label="Seed and variety selection",
        description="which variety or seed type to plant, and its maturity or resistance",
        hypothesis="The farmer is asking which seed variety to plant.",
        route=ROUTE_RAG,
        required_slots=(),
        optional_slots=("crop", "location"),
        seeds=(
            "which maize variety matures fastest",
            "is hybrid seed better than seed i saved",
            "which tomato variety resists blight",
            "recommend a drought tolerant maize seed for my area",
            "how long does sc627 take to mature",
            "can i replant seed from last season's harvest",
            "which rice variety suits lowland paddies",
            "what is the difference between open pollinated and hybrid bean seed",
        ),
    ),
    Intent(
        name="crop_management_practice",
        label="Crop management practice",
        description="how to carry out a field operation: spacing, planting, weeding, pruning, harvest, storage",
        hypothesis="The farmer is asking how to carry out a field operation.",
        route=ROUTE_RAG,
        required_slots=(),
        optional_slots=("crop", "location", "timeframe"),
        seeds=(
            "what spacing should i use for maize",
            "when is the right time to plant beans",
            "how deep should i sow sunflower seed",
            "how do i prune my coffee bushes",
            "how many times should i weed my cassava",
            "what is the best way to store maize after harvest",
            "how do i harden off tomato seedlings before transplanting",
            "should i thin my onion seedlings and how",
            "how do i intercrop beans with maize properly",
        ),
    ),
    Intent(
        name="market_and_inputs_supply",
        label="Market and input supply",
        description="prices, buyers, where to buy inputs, credit",
        hypothesis="The farmer is asking about prices, buyers, or where to obtain inputs.",
        route=ROUTE_RAG,
        required_slots=(),
        optional_slots=("crop", "location", "timeframe"),
        seeds=(
            "what is the price of maize in dodoma market",
            "where can i buy certified seed near me",
            "which agrodealer sells knapsack sprayers",
            "how much are tomatoes selling for this week",
            "where do i sell my sunflower harvest",
            "is there a cooperative that buys coffee here",
            "how do i get a loan to buy inputs",
            "what does a bag of dap cost now",
        ),
    ),
    Intent(
        name="livestock_and_adjacent",
        label="Livestock and adjacent",
        description="animals, poultry, fodder, fish and bees rather than crops",
        hypothesis="The farmer is asking about livestock, poultry, fish or bees.",
        route=ROUTE_RAG,
        required_slots=(),
        optional_slots=("location",),
        seeds=(
            "my chickens are dying, what should i do",
            "how much feed does a dairy cow need per day",
            "how do i treat ticks on my cattle",
            "is napier grass good fodder for goats",
            "how many layers can i keep in one shed",
            "which vaccine do my chicks need",
            "how do i start beekeeping on my farm",
            "my goat is not eating and looks weak",
            "what do i feed tilapia in a fish pond",
        ),
    ),
    Intent(
        name="agronomy_concept_explainer",
        label="Agronomy concept explainer",
        description="what a term or practice means, in general, not for one field",
        hypothesis="The farmer is asking for the meaning or definition of an agricultural term.",
        route=ROUTE_RAG,
        required_slots=(),
        optional_slots=(),
        seeds=(
            "explain crop rotation to me",
            "what is integrated pest management",
            "what does ndvi actually measure",
            "what is the difference between hybrid and open pollinated seed",
            "what does conservation agriculture mean",
            "define soil organic carbon",
            "what does top dressing mean",
            "explain what evapotranspiration is",
            "what do people mean by agroforestry",
        ),
    ),
    # -------------------------------------------------------------------- clarify
    Intent(
        name=CLARIFY_INTENT,
        label="Unclear or out of scope",
        description="not a farming question, or too little to act on",
        hypothesis="The message is a greeting, small talk, or not about farming at all.",
        route=ROUTE_CLARIFY,
        required_slots=(),
        optional_slots=(),
        seeds=(
            "hello",
            "thank you very much",
            "asante sana",
            "i need help",
            "what can you do",
            "who built this app",
            "how do i reset my phone",
            "what time does the bus to arusha leave",
            "please call me back later",
        ),
    ),
)

INTENT_NAMES: tuple[str, ...] = tuple(intent.name for intent in INTENTS)
BY_NAME: dict[str, Intent] = {intent.name: intent for intent in INTENTS}


def get_intent(name: str) -> Intent:
    try:
        return BY_NAME[name]
    except KeyError as exc:
        raise ConfigError(f"unknown intent {name!r} (known: {', '.join(INTENT_NAMES)})") from exc


def route_of(name: str) -> str:
    return get_intent(name).route


def intents_for_route(route: str) -> tuple[Intent, ...]:
    if route not in ROUTES:
        raise ConfigError(f"unknown route {route!r} (known: {', '.join(ROUTES)})")
    return tuple(intent for intent in INTENTS if intent.route == route)


def seed_pairs() -> Iterator[tuple[str, str]]:
    """(intent name, utterance) for every seed, in declaration order."""
    for intent in INTENTS:
        for seed in intent.seeds:
            yield intent.name, seed


def as_dict() -> dict[str, Any]:
    """The intent table, for /api/nlu and the docs page."""
    return {
        "count": len(INTENTS),
        "routes": {route: [i.name for i in intents_for_route(route)] for route in ROUTES},
        "clarify_intent": CLARIFY_INTENT,
        "intents": [intent.as_dict() for intent in INTENTS],
    }


def _validate() -> None:
    """Invariants that a careless edit would otherwise break silently."""
    if len(INTENTS) != 12:
        raise ConfigError(f"SPEC section 5.2 defines 12 intents, found {len(INTENTS)}")
    if len(BY_NAME) != len(INTENTS):
        raise ConfigError("duplicate intent name")
    if CLARIFY_INTENT not in BY_NAME:
        raise ConfigError(f"the clarify intent {CLARIFY_INTENT!r} is not in the table")
    seen: dict[str, str] = {}
    for intent in INTENTS:
        if intent.route not in ROUTES:
            raise ConfigError(f"{intent.name}: unknown route {intent.route!r}")
        if not 6 <= len(intent.seeds) <= 12:
            raise ConfigError(
                f"{intent.name}: SPEC asks for 6-12 seed utterances, found {len(intent.seeds)}"
            )
        for slot in intent.slots:
            if slot not in SLOT_NAMES:
                raise ConfigError(
                    f"{intent.name}: slot {slot!r} is not extractable "
                    f"(slots.py fills {', '.join(SLOT_NAMES)})"
                )
        if set(intent.required_slots) & set(intent.optional_slots):
            raise ConfigError(f"{intent.name}: a slot is both required and optional")
        for seed in intent.seeds:
            if seed != seed.strip().lower():
                raise ConfigError(f"{intent.name}: seed {seed!r} must be lowercase and stripped")
            if seed in seen:
                raise ConfigError(f"seed {seed!r} is shared by {seen[seed]} and {intent.name}")
            seen[seed] = intent.name


_validate()
