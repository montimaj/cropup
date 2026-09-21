"""Exception hierarchy for CropUp.

Everything raised on purpose by this package derives from :class:`CropUpError`,
so a web handler can catch one class and still tell an operator what broke.

The branch that matters most is :class:`ProvenanceError`: it is what the
fabrication firewall (SPEC section 4) raises instead of letting an unmeasured
value reach a farmer.
"""

from __future__ import annotations

__all__ = [
    "CropUpError",
    "ConfigError",
    "DataFileError",
    "EarthEngineUnavailable",
    "SourceUnavailable",
    "ModelUnavailable",
    "ConfirmationRequired",
    "ProvenanceError",
    "MissingProvenanceError",
    "FabricationError",
    "InvalidFactError",
]


class CropUpError(Exception):
    """Base class for every error CropUp raises deliberately."""


class ConfigError(CropUpError):
    """Settings could not be built: a CROPUP_* variable is unusable."""


class DataFileError(ConfigError):
    """A committed artifact under cropup/data/ is missing or unreadable."""

    def __init__(self, path: object, detail: str = "") -> None:
        self.path = str(path)
        self.detail = detail
        message = f"data file unusable: {self.path}"
        if detail:
            message = f"{message} ({detail})"
        super().__init__(message)


class EarthEngineUnavailable(CropUpError):
    """Earth Engine was asked for work but is not initialised.

    The app boots degraded rather than refusing to start, so every geo/ call
    site has to be prepared for this.
    """

    def __init__(self, reason: str = "Earth Engine is not initialised") -> None:
        self.reason = reason
        super().__init__(reason)


class SourceUnavailable(CropUpError):
    """Every source in a chain was tried and none produced a value.

    Callers that can degrade should build an ``evidence.Missing`` instead of
    raising this; it exists for the cases where a caller genuinely cannot
    continue.
    """

    def __init__(self, quantity: str, chain_tried: object = (), detail: str = "") -> None:
        self.quantity = quantity
        self.chain_tried = tuple(chain_tried or ())
        self.detail = detail
        chain = " -> ".join(self.chain_tried) if self.chain_tried else "no sources configured"
        message = f"no source produced {quantity}: {chain}"
        if detail:
            message = f"{message} ({detail})"
        super().__init__(message)


class ModelUnavailable(CropUpError):
    """An ONNX model or tokenizer could not be loaded.

    NLU degrades to the rule tier when this is raised; it is never fatal.
    """

    def __init__(self, model_id: str, detail: str = "") -> None:
        self.model_id = model_id
        self.detail = detail
        message = f"model unavailable: {model_id}"
        if detail:
            message = f"{message} ({detail})"
        super().__init__(message)


class ConfirmationRequired(CropUpError):
    """An Earth Engine run was requested for a field the farmer never confirmed.

    SPEC section 4.4: a natural-language turn never triggers EE.
    """

    def __init__(self, missing_confirmations: object = (), detail: str = "") -> None:
        self.missing_confirmations = tuple(missing_confirmations or ())
        self.detail = detail
        what = ", ".join(self.missing_confirmations) or "location and crop"
        message = f"unconfirmed field: {what} must be confirmed before Earth Engine runs"
        if detail:
            message = f"{message} ({detail})"
        super().__init__(message)


class ProvenanceError(CropUpError):
    """Base class for fabrication-firewall violations."""


class MissingProvenanceError(ProvenanceError):
    """Something tried to render a value that carries no provenance.

    Raised by render/templates.py when a template slot holds no ``Fact`` --
    it never falls back to a default (SPEC section 4.2).
    """

    def __init__(self, slot: str, template: str = "", detail: str = "") -> None:
        self.slot = slot
        self.template = template
        self.detail = detail
        where = f" in template {template!r}" if template else ""
        message = f"slot {slot!r}{where} has no Fact; refusing to render an unmeasured value"
        if detail:
            message = f"{message} ({detail})"
        super().__init__(message)


class FabricationError(ProvenanceError):
    """A raw or defaulted value reached a boundary that only accepts Facts."""

    def __init__(self, detail: str, value: object = None) -> None:
        self.value = value
        super().__init__(detail)


class InvalidFactError(FabricationError):
    """A Fact was constructed from something that is not a measurement.

    None, NaN and infinity are masked pixels or arithmetic accidents, not
    observations; they must become ``evidence.Missing`` instead.
    """

    def __init__(self, quantity: str, detail: str, value: object = None) -> None:
        self.quantity = quantity
        super().__init__(f"cannot build Fact for {quantity!r}: {detail}", value=value)
