"""The fabrication firewall (SPEC section 4).

Three types and one rule.

* :class:`Fact` is a measured value plus the instrument that produced it.
* :class:`Missing` is an absence plus the reason for it and the chain that was
  tried.
* :class:`Ledger` is everything one turn measured and everything it could not.

The rule: a bare ``float`` is not evidence. ``render/templates.py`` accepts only
``Fact`` objects, so "silently print pH 7.0 because iSDA was masked" is not a
discouraged practice here, it is unrepresentable -- there is no code path that
turns a default into a ``Fact``.

Nothing in this module imports ``ee``, ``numpy`` or ``onnxruntime``; the Earth
Engine helpers take the plain dictionary that ``reduceRegion().getInfo()``
returns.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from functools import partial
from enum import Enum
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .errors import FabricationError, InvalidFactError, MissingProvenanceError

__all__ = [
    "MissingReason",
    "Evidence",
    "Fact",
    "Missing",
    "Ledger",
    "SourceDescriptor",
    "demote_if_stale",
    "fact_from_reduce_region",
    "facts_from_reduce_region",
]

# Units that read badly when appended to a number.
_DIMENSIONLESS = frozenset({"", "1", "index", "ratio", "fraction", "unitless", "dimensionless"})

_NUMERIC = (int, float)


class MissingReason(str, Enum):
    """Why a quantity has no value. There is no sixth reason: if a value is
    absent for a reason not on this list, the reason has not been diagnosed."""

    OUT_OF_COVERAGE = "out_of_coverage"
    MASKED = "masked"
    STALE = "stale"
    SOURCE_FAILED = "source_failed"
    NOT_REQUESTED = "not_requested"

    def describe(self) -> str:
        return _REASON_PHRASE[self]

    def __str__(self) -> str:  # so f"{reason}" is the wire value, not the enum repr
        return self.value


_REASON_PHRASE = {
    MissingReason.OUT_OF_COVERAGE: "this location is outside the data's coverage",
    MissingReason.MASKED: "the pixel is masked (cloud, water or no valid observation)",
    MissingReason.STALE: "the most recent observation is too old to use",
    MissingReason.SOURCE_FAILED: "the source could not be read",
    MissingReason.NOT_REQUESTED: "it was not requested for this question",
}


def _coerce_date(value: Any, quantity: str, what: str) -> dt.date | None:
    """Accept date / datetime / ISO string / None; refuse anything else."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value[:10])
        except ValueError as exc:
            raise InvalidFactError(quantity, f"{what} is not an ISO date: {value!r}", value) from exc
    raise InvalidFactError(quantity, f"{what} must be a date, got {type(value).__name__}", value)


class Evidence:
    """Common surface of :class:`Fact` and :class:`Missing`.

    Both answer ``.quantity``, ``.render()`` and ``.provenance()``; only a Fact
    answers ``True`` to ``.measured``.
    """

    __slots__ = ()

    measured: bool = False

    def render(self) -> str:
        raise NotImplementedError

    def provenance(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class Fact(Evidence):
    """One measured value and the instrument that produced it.

    ``observed_on`` is ``None`` only for genuinely static layers (climate
    normals, soil property grids); it is a required argument everywhere so that
    "I forgot the date" cannot look like "this layer has no date".
    """

    quantity: str
    value: float | int | str | bool
    unit: str
    source_asset: str
    observed_on: dt.date | None
    resolution_m: float | None
    chain_position: int = 0
    band: str | None = None
    scaling_applied: str | None = None
    stale_after_days: int | None = None
    derived_from: tuple[str, ...] = ()
    note: str | None = None
    retrieved_at: dt.date = field(default_factory=dt.date.today, compare=False)
    precision: int | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        set_ = partial(object.__setattr__, self)  # frozen dataclass: normalise in place

        if not isinstance(self.quantity, str) or not self.quantity.strip():
            raise InvalidFactError(str(self.quantity), "quantity must be a non-empty name")
        if not isinstance(self.source_asset, str) or not self.source_asset.strip():
            raise InvalidFactError(self.quantity, "source_asset must name the asset that produced the value")
        if not isinstance(self.unit, str):
            raise InvalidFactError(self.quantity, f"unit must be a string, got {type(self.unit).__name__}")

        value = self.value
        if value is None:
            raise InvalidFactError(self.quantity, "None is a masked pixel, not a measurement; use Missing")
        if isinstance(value, bool):
            pass
        elif isinstance(value, _NUMERIC):
            if not math.isfinite(value):
                raise InvalidFactError(self.quantity, f"{value!r} is not a finite measurement; use Missing", value)
        elif isinstance(value, str):
            if not value.strip():
                raise InvalidFactError(self.quantity, "empty string is not a measurement; use Missing", value)
        else:
            raise InvalidFactError(
                self.quantity, f"value must be a number, string or bool, got {type(value).__name__}", value
            )

        if self.resolution_m is not None:
            if not isinstance(self.resolution_m, _NUMERIC) or isinstance(self.resolution_m, bool):
                raise InvalidFactError(self.quantity, "resolution_m must be a number of metres or None")
            if not math.isfinite(self.resolution_m) or self.resolution_m <= 0:
                raise InvalidFactError(self.quantity, f"resolution_m must be positive, got {self.resolution_m!r}")
            set_("resolution_m", float(self.resolution_m))

        if not isinstance(self.chain_position, int) or isinstance(self.chain_position, bool) or self.chain_position < 0:
            raise InvalidFactError(self.quantity, f"chain_position must be a non-negative int, got {self.chain_position!r}")

        if self.stale_after_days is not None:
            if not isinstance(self.stale_after_days, int) or isinstance(self.stale_after_days, bool) or self.stale_after_days <= 0:
                raise InvalidFactError(self.quantity, f"stale_after_days must be a positive int, got {self.stale_after_days!r}")

        set_("observed_on", _coerce_date(self.observed_on, self.quantity, "observed_on"))
        set_("retrieved_at", _coerce_date(self.retrieved_at, self.quantity, "retrieved_at") or dt.date.today())
        set_("derived_from", tuple(self.derived_from or ()))

    # -- derived properties -------------------------------------------------

    @property
    def measured(self) -> bool:  # type: ignore[override]
        return True

    @property
    def age_days(self) -> int | None:
        """Days between the observation and the moment it was fetched.

        ``None`` means the layer is static or undated -- not "fresh"."""
        if self.observed_on is None:
            return None
        return (self.retrieved_at - self.observed_on).days

    @property
    def is_stale(self) -> bool | None:
        """``None`` when staleness is unknown, which is not the same as False."""
        age = self.age_days
        if age is None or self.stale_after_days is None:
            return None
        return age > self.stale_after_days

    @property
    def chain_label(self) -> str:
        return "primary" if self.chain_position == 0 else f"fallback #{self.chain_position}"

    @property
    def is_numeric(self) -> bool:
        return isinstance(self.value, _NUMERIC) and not isinstance(self.value, bool)

    # -- display ------------------------------------------------------------

    def render(self, precision: int | None = None) -> str:
        """The value as it appears on screen, with its unit. No provenance:
        that goes in the hover, via :meth:`provenance`."""
        if isinstance(self.value, bool):
            text = "yes" if self.value else "no"
        elif self.is_numeric:
            dp = precision if precision is not None else self.precision
            if dp is None:
                dp = _auto_precision(float(self.value))
            text = f"{float(self.value):.{dp}f}"
        else:
            text = str(self.value)
        unit = self.unit.strip()
        if unit.lower() in _DIMENSIONLESS:
            return text
        return f"{text} {unit}"

    def staleness_note(self) -> str | None:
        """A short phrase for the UI when the observation is old, else None."""
        age = self.age_days
        if age is None:
            return None
        if self.is_stale:
            return f"observed {age} days ago, past the {self.stale_after_days}-day freshness limit"
        if self.stale_after_days is None and age > 365:
            return f"observed {age} days ago; no freshness limit is defined for this source"
        return None

    def provenance(self) -> dict[str, Any]:
        """JSON-safe dict for the per-number hover in the UI."""
        return {
            "quantity": self.quantity,
            "value": self.value,
            "unit": self.unit,
            "rendered": self.render(),
            "source_asset": self.source_asset,
            "band": self.band,
            "observed_on": self.observed_on.isoformat() if self.observed_on else None,
            "retrieved_at": self.retrieved_at.isoformat(),
            "resolution_m": self.resolution_m,
            "chain_position": self.chain_position,
            "chain_label": self.chain_label,
            "scaling_applied": self.scaling_applied,
            "age_days": self.age_days,
            "stale_after_days": self.stale_after_days,
            "is_stale": self.is_stale,
            "staleness_note": self.staleness_note(),
            "derived_from": list(self.derived_from),
            "note": self.note,
            "measured": True,
        }

    # -- construction -------------------------------------------------------

    @classmethod
    def derive(
        cls,
        quantity: str,
        value: float | int | str | bool,
        unit: str,
        inputs: Sequence["Fact"],
        *,
        scaling_applied: str | None = None,
        precision: int | None = None,
        note: str | None = None,
    ) -> "Fact":
        """Build a Fact computed from other Facts (e.g. Koppen from climate
        normals, a stress index from LST and ET).

        The derived Fact inherits the *worst* provenance of its inputs: the
        coarsest resolution, the oldest observation, the deepest chain position.
        That is the honest summary -- a number is no better than its weakest
        ingredient.
        """
        if not inputs:
            raise FabricationError(f"cannot derive {quantity!r} from no inputs", value)
        bad = [f for f in inputs if not isinstance(f, Fact)]
        if bad:
            raise FabricationError(
                f"cannot derive {quantity!r}: inputs must all be Facts, got {type(bad[0]).__name__}", value
            )
        dated = [f.observed_on for f in inputs if f.observed_on is not None]
        resolutions = [f.resolution_m for f in inputs if f.resolution_m is not None]
        assets = []
        for f in inputs:
            if f.source_asset not in assets:
                assets.append(f.source_asset)
        stale_limits = [f.stale_after_days for f in inputs if f.stale_after_days is not None]
        return cls(
            quantity=quantity,
            value=value,
            unit=unit,
            source_asset=" + ".join(assets),
            observed_on=min(dated) if dated else None,
            resolution_m=max(resolutions) if resolutions else None,
            chain_position=max(f.chain_position for f in inputs),
            scaling_applied=scaling_applied,
            stale_after_days=min(stale_limits) if stale_limits else None,
            derived_from=tuple(f.quantity for f in inputs),
            note=note,
            retrieved_at=max(f.retrieved_at for f in inputs),
            precision=precision,
        )

    def __repr__(self) -> str:
        when = self.observed_on.isoformat() if self.observed_on else "static"
        res = f"{self.resolution_m:g}m" if self.resolution_m else "res?"
        return (
            f"Fact({self.quantity}={self.render()!r} @ {self.source_asset}"
            f"{'/' + self.band if self.band else ''} {when} {res} chain#{self.chain_position})"
        )


def _auto_precision(value: float) -> int:
    """Decimal places that neither hide a difference nor invent precision."""
    magnitude = abs(value)
    if magnitude >= 100:
        return 0
    if magnitude >= 10:
        return 1
    if magnitude >= 1:
        return 2
    return 3


@dataclass(frozen=True)
class Missing(Evidence):
    """A named absence: which quantity, why, and what was tried to get it."""

    quantity: str
    reason: MissingReason
    chain_tried: tuple[str, ...] = ()
    detail: str | None = None
    last_observed_on: dt.date | None = None
    age_days: int | None = None
    requested_at: dt.date = field(default_factory=dt.date.today, compare=False)

    def __post_init__(self) -> None:
        set_ = partial(object.__setattr__, self)
        if not isinstance(self.quantity, str) or not self.quantity.strip():
            raise InvalidFactError(str(self.quantity), "quantity must be a non-empty name")
        try:
            set_("reason", MissingReason(self.reason))
        except ValueError as exc:
            allowed = ", ".join(r.value for r in MissingReason)
            raise InvalidFactError(self.quantity, f"reason must be one of: {allowed}", self.reason) from exc
        if isinstance(self.chain_tried, str):
            set_("chain_tried", (self.chain_tried,))
        else:
            set_("chain_tried", tuple(self.chain_tried or ()))
        set_("last_observed_on", _coerce_date(self.last_observed_on, self.quantity, "last_observed_on"))
        set_("requested_at", _coerce_date(self.requested_at, self.quantity, "requested_at") or dt.date.today())

    @property
    def measured(self) -> bool:  # type: ignore[override]
        return False

    @property
    def label(self) -> str:
        return self.quantity.replace("_", " ")

    def render(self) -> str:
        """A sentence the farmer can read. Never a number, never a hedge."""
        text = f"{self.label}: not available — {self.reason.describe()}"
        if self.reason is MissingReason.STALE and self.last_observed_on:
            age = f", {self.age_days} days ago" if self.age_days is not None else ""
            text += f" (last observation {self.last_observed_on.isoformat()}{age})"
        if self.chain_tried:
            text += f". Tried: {' → '.join(self.chain_tried)}"
        if self.detail:
            text += f". {self.detail}"
        return text

    def provenance(self) -> dict[str, Any]:
        return {
            "quantity": self.quantity,
            "measured": False,
            "reason": self.reason.value,
            "reason_text": self.reason.describe(),
            "chain_tried": list(self.chain_tried),
            "detail": self.detail,
            "last_observed_on": self.last_observed_on.isoformat() if self.last_observed_on else None,
            "age_days": self.age_days,
            "requested_at": self.requested_at.isoformat(),
            "rendered": self.render(),
        }

    @classmethod
    def from_stale_fact(cls, fact: Fact, detail: str | None = None) -> "Missing":
        """Demote a Fact whose observation is too old to act on."""
        return cls(
            quantity=fact.quantity,
            reason=MissingReason.STALE,
            chain_tried=(fact.source_asset,),
            detail=detail or f"last value {fact.render()} from {fact.source_asset}",
            last_observed_on=fact.observed_on,
            age_days=fact.age_days,
            requested_at=fact.retrieved_at,
        )

    def __repr__(self) -> str:
        chain = "->".join(self.chain_tried) if self.chain_tried else "-"
        return f"Missing({self.quantity}, {self.reason.value}, tried={chain})"


def demote_if_stale(item: Fact | Missing, detail: str | None = None) -> Fact | Missing:
    """Replace a Fact past its declared freshness limit with ``Missing(STALE)``.

    The one place a value is allowed to stop being evidence. SPEC 3.3 records
    ESI returning a reading over a year old at Arusha; without this, that
    renders as a confident number with a stale date nobody reads.

    Tri-state on purpose: only ``is_stale is True`` demotes. A source that
    declared no ``stale_after_days`` has *unknown* freshness, and guessing a
    limit here would be the same class of invention as defaulting a value --
    such a Fact keeps its :meth:`Fact.staleness_note` instead.

    Anything that is already a ``Missing`` is returned unchanged, so this can be
    dropped in front of any Fact-or-Missing without a type test.
    """
    if isinstance(item, Fact) and item.is_stale:
        return Missing.from_stale_fact(item, detail)
    return item


@dataclass(frozen=True)
class SourceDescriptor:
    """Everything ``geo/registry.py`` knows about one band of one asset.

    The scaling traps of SPEC section 3.2 live in ``transform`` and are
    described in ``scaling``; both travel with the Fact so the UI can show what
    arithmetic was applied to the raw pixel.
    """

    quantity: str
    asset_id: str
    unit: str
    band: str | None = None
    result_key: str | None = None  # reduceRegion key, when it is not the band name
    resolution_m: float | None = None
    chain: tuple[str, ...] = ()  # the full fallback chain, in order
    chain_position: int = 0
    scaling: str | None = None
    transform: Callable[[float], float] | None = field(default=None, compare=False, repr=False)
    valid_range: tuple[float, float] | None = None
    invalid_values: tuple[float, ...] = ()  # e.g. PEST-CHEMGRIDS -1.5 sentinels
    stale_after_days: int | None = None
    is_static: bool = False  # climate normals, soil grids: no observation date
    coverage: str | None = None

    def __post_init__(self) -> None:
        set_ = partial(object.__setattr__, self)
        set_("chain", tuple(self.chain or ()))
        set_("invalid_values", tuple(self.invalid_values or ()))

    @property
    def key(self) -> str:
        """The key this source's value appears under in a reduceRegion dict."""
        return self.result_key or self.band or self.quantity

    @property
    def tried(self) -> tuple[str, ...]:
        """Assets to report as attempted when this source yields nothing."""
        return self.chain or (self.asset_id,)


def fact_from_reduce_region(
    result: Mapping[str, Any] | None,
    source: SourceDescriptor,
    *,
    observed_on: dt.date | str | None,
    quantity: str | None = None,
    key: str | None = None,
    detail: str | None = None,
    demote_stale: bool = True,
) -> Fact | Missing:
    """Turn one entry of an ``ee.Reducer`` result into a Fact or a Missing.

    ``observed_on`` is required (pass ``None`` only for a static layer), because
    a missing date must be a decision, not an omission. Masked pixels arrive as
    an absent key or a ``None`` value; both become ``Missing(MASKED)``, never 0.

    A reading older than the freshness limit its source declared is demoted to
    ``Missing(STALE)`` here, carrying the value and the date it was taken, so
    "ESI at Arusha is over a year old" cannot reach the screen as a bare number
    (SPEC 3.3). Only a *declared* limit demotes: ``stale_after_days`` unset
    means staleness is unknown, and unknown is never treated as fresh -- the
    Fact keeps its :meth:`Fact.staleness_note`, which says so. Pass
    ``demote_stale=False`` where the caller shows the old value together with
    its age on purpose.
    """
    quantity = quantity or source.quantity
    key = key or source.key
    tried = source.tried

    if observed_on is None and not source.is_static:
        raise InvalidFactError(
            quantity,
            f"observed_on is required for non-static source {source.asset_id}; "
            "set is_static=True on the descriptor if the layer genuinely has no date",
        )

    if result is None:
        return Missing(quantity, MissingReason.SOURCE_FAILED, tried, detail or "no result returned from Earth Engine")
    if not isinstance(result, Mapping):
        return Missing(
            quantity,
            MissingReason.SOURCE_FAILED,
            tried,
            detail or f"expected a reduceRegion dictionary, got {type(result).__name__}",
        )
    if key not in result:
        keys = ", ".join(sorted(map(str, result))) or "none"
        return Missing(quantity, MissingReason.MASKED, tried, detail or f"{key!r} absent from result (keys: {keys})")

    raw = result[key]
    if raw is None:
        return Missing(quantity, MissingReason.MASKED, tried, detail or f"{key!r} is null: no unmasked pixel in the region")
    if isinstance(raw, bool) or isinstance(raw, str):
        value: float | int | str | bool = raw
    elif isinstance(raw, _NUMERIC):
        if not math.isfinite(raw):
            return Missing(quantity, MissingReason.MASKED, tried, detail or f"{key!r} is {raw!r}")
        value = float(raw)
    else:
        return Missing(
            quantity, MissingReason.SOURCE_FAILED, tried, detail or f"{key!r} is a {type(raw).__name__}, not a value"
        )

    if isinstance(value, float):
        if source.invalid_values and any(math.isclose(value, s, rel_tol=1e-9, abs_tol=1e-9) for s in source.invalid_values):
            return Missing(
                quantity, MissingReason.MASKED, tried, detail or f"raw {value!r} is a no-data sentinel for {source.asset_id}"
            )
        if source.transform is not None:
            try:
                value = float(source.transform(value))
            except Exception as exc:  # a bad transform must not fabricate a number
                return Missing(quantity, MissingReason.SOURCE_FAILED, tried, f"scaling failed: {exc}")
            if not math.isfinite(value):
                return Missing(quantity, MissingReason.SOURCE_FAILED, tried, "scaling produced a non-finite value")
        if source.valid_range is not None:
            low, high = source.valid_range
            if not (low <= value <= high):
                return Missing(
                    quantity,
                    MissingReason.MASKED,
                    tried,
                    detail or f"scaled value {value:g} is outside the plausible range [{low:g}, {high:g}]",
                )

    fact = Fact(
        quantity=quantity,
        value=value,
        unit=source.unit,
        source_asset=source.asset_id,
        observed_on=observed_on,
        resolution_m=source.resolution_m,
        chain_position=source.chain_position,
        band=source.band,
        scaling_applied=source.scaling,
        stale_after_days=source.stale_after_days,
        note=detail,
    )
    return demote_if_stale(fact) if demote_stale else fact


def facts_from_reduce_region(
    result: Mapping[str, Any] | None,
    sources: Iterable[SourceDescriptor],
    *,
    observed_on: dt.date | str | None,
    demote_stale: bool = True,
) -> dict[str, Fact | Missing]:
    """Unpack one batched ``reduceRegion`` dictionary into evidence.

    SPEC section 3.4 wants one round trip per geometry, so several sources share
    one result dict; each still gets its own Fact-or-Missing.
    """
    out: dict[str, Fact | Missing] = {}
    for source in sources:
        out[source.quantity] = fact_from_reduce_region(
            result, source, observed_on=observed_on, demote_stale=demote_stale
        )
    return out


class Ledger:
    """What one turn measured, and what it could not.

    The renderer reads from it, and ``/api/capabilities`` reports
    :meth:`degradation` so the capability strip can say what is degraded right
    now instead of quietly showing a thinner answer.
    """

    def __init__(self, turn: str | None = None) -> None:
        self.turn = turn
        self._entries: list[Evidence] = []

    # -- writing ------------------------------------------------------------

    def add(self, item: Evidence) -> Evidence:
        """Record a Fact or a Missing. Anything else is a firewall breach."""
        if not isinstance(item, (Fact, Missing)):
            raise FabricationError(
                f"a Ledger holds Fact or Missing, not {type(item).__name__}; "
                "wrap the value with its source before recording it",
                item,
            )
        self._entries.append(item)
        return item

    def record(self, fact: Fact) -> Fact:
        if not isinstance(fact, Fact):
            raise FabricationError(f"record() takes a Fact, not {type(fact).__name__}", fact)
        self._entries.append(fact)
        return fact

    def gap(
        self,
        quantity: str,
        reason: MissingReason | str,
        chain_tried: Sequence[str] = (),
        detail: str | None = None,
    ) -> Missing:
        """Shorthand for ``add(Missing(...))``.

        ``reason`` is handed to :class:`Missing` unvalidated on purpose: the
        dataclass turns an unknown reason into an :class:`InvalidFactError`
        naming the quantity and the five allowed reasons. Coercing it here first
        raised a bare ``ValueError`` from inside the enum, which is neither a
        :class:`~cropup.errors.CropUpError` nor traceable to a quantity.
        """
        missing = Missing(quantity, reason, tuple(chain_tried), detail)
        self._entries.append(missing)
        return missing

    def extend(self, items: Iterable[Evidence]) -> "Ledger":
        for item in items:
            self.add(item)
        return self

    def merge(self, other: "Ledger") -> "Ledger":
        """Fold another ledger (e.g. from a sub-analysis) into this one."""
        if not isinstance(other, Ledger):
            raise FabricationError(f"merge() takes a Ledger, not {type(other).__name__}", other)
        self._entries.extend(other._entries)
        return self

    # -- reading ------------------------------------------------------------

    @property
    def entries(self) -> tuple[Evidence, ...]:
        return tuple(self._entries)

    @property
    def facts(self) -> tuple[Fact, ...]:
        return tuple(e for e in self._entries if isinstance(e, Fact))

    @property
    def gaps(self) -> tuple[Missing, ...]:
        return tuple(e for e in self._entries if isinstance(e, Missing))

    def quantities(self) -> tuple[str, ...]:
        seen: list[str] = []
        for entry in self._entries:
            if entry.quantity not in seen:
                seen.append(entry.quantity)
        return tuple(seen)

    def get(self, quantity: str) -> Evidence | None:
        """The best entry for a quantity: the latest Fact, else the latest Missing.

        Deliberately not last-write-wins. The orchestrators fan out over several
        legs, and ``analysis.rules.absorb`` writes ``Missing(source_failed)``
        for *every* quantity a failed leg owed -- including quantities another
        leg has already measured. Under last-write-wins a leg that never
        answered could erase a measurement that did, and the farmer would be
        told "not available" about a number sitting in this ledger.

        A measurement is never undone by a later absence. An absence that
        genuinely supersedes one -- a reading too old to act on -- has to
        replace the Fact before it is recorded, which is what
        :func:`fact_from_reduce_region` does with
        :meth:`Missing.from_stale_fact`.
        """
        latest_missing: Missing | None = None
        for entry in reversed(self._entries):
            if entry.quantity != quantity:
                continue
            if isinstance(entry, Fact):
                return entry
            if latest_missing is None and isinstance(entry, Missing):
                latest_missing = entry
        return latest_missing

    def fact(self, quantity: str) -> Fact | None:
        """The most recent *measured* value, or None. Never a default."""
        for entry in reversed(self._entries):
            if entry.quantity == quantity and isinstance(entry, Fact):
                return entry
        return None

    def require(self, quantity: str, template: str = "") -> Fact:
        """The Fact for a quantity, or raise. Used by provenance-gated rendering.

        Resolves through :meth:`get`, so a measured value wins over a later
        blanket ``Missing`` from a failed leg; rendering must not refuse a
        number the turn actually measured.
        """
        entry = self.get(quantity)
        if isinstance(entry, Fact):
            return entry
        if isinstance(entry, Missing):
            raise MissingProvenanceError(quantity, template, entry.render())
        raise MissingProvenanceError(quantity, template, "nothing was recorded for this quantity")

    def has(self, quantity: str) -> bool:
        return self.fact(quantity) is not None

    def fallbacks_used(self) -> tuple[Fact, ...]:
        """Facts that came from a fallback source rather than the primary."""
        return tuple(f for f in self.facts if f.chain_position > 0)

    def stale_facts(self) -> tuple[Fact, ...]:
        return tuple(f for f in self.facts if f.is_stale)

    def undated_facts(self) -> tuple[Fact, ...]:
        """Facts whose freshness cannot be judged -- reported, not assumed fresh."""
        return tuple(f for f in self.facts if f.is_stale is None and not f.observed_on)

    # -- honesty reporting --------------------------------------------------

    def degradation(self) -> dict[str, Any]:
        """The payload behind /api/capabilities: what is degraded right now."""
        gaps = self.gaps
        fallbacks = self.fallbacks_used()
        stale = self.stale_facts()
        by_reason: dict[str, list[str]] = {}
        for miss in gaps:
            by_reason.setdefault(miss.reason.value, []).append(miss.quantity)
        return {
            "turn": self.turn,
            "degraded": bool(gaps or fallbacks or stale),
            "fact_count": len(self.facts),
            "gap_count": len(gaps),
            "measured": [f.quantity for f in self.facts],
            "gaps": [m.provenance() for m in gaps],
            "gaps_by_reason": by_reason,
            "fallbacks": [
                {
                    "quantity": f.quantity,
                    "source_asset": f.source_asset,
                    "chain_position": f.chain_position,
                    "chain_label": f.chain_label,
                }
                for f in fallbacks
            ],
            "stale": [
                {
                    "quantity": f.quantity,
                    "source_asset": f.source_asset,
                    "observed_on": f.observed_on.isoformat() if f.observed_on else None,
                    "age_days": f.age_days,
                    "stale_after_days": f.stale_after_days,
                }
                for f in stale
            ],
            "undated": [f.quantity for f in self.undated_facts()],
            "summary": self.summary(),
        }

    def as_dict(self) -> dict[str, Any]:
        """Everything this turn measured and missed, JSON-safe, for the API."""
        return {
            "turn": self.turn,
            "facts": [f.provenance() for f in self.facts],
            "gaps": [m.provenance() for m in self.gaps],
            "degradation": self.degradation(),
        }

    def summary(self) -> str:
        """One line, safe to log and to show in the capability strip."""
        parts = [f"{len(self.facts)} measured", f"{len(self.gaps)} missing"]
        fallbacks = self.fallbacks_used()
        if fallbacks:
            parts.append(f"{len(fallbacks)} on fallback sources")
        stale = self.stale_facts()
        if stale:
            parts.append(f"{len(stale)} stale")
        head = f"{self.turn}: " if self.turn else ""
        line = head + ", ".join(parts)
        if self.gaps:
            names = ", ".join(f"{m.quantity} ({m.reason.value})" for m in self.gaps)
            line += f" | gaps: {names}"
        return line

    # -- dunders ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Evidence]:
        return iter(self._entries)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return any(e.quantity == item for e in self._entries)
        return item in self._entries

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ledger):
            return NotImplemented
        return self.turn == other.turn and self._entries == other._entries

    def __repr__(self) -> str:
        return f"Ledger({self.summary()})"
