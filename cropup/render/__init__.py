"""Deterministic, provenance-gated rendering.

Imports ``cropup.evidence`` only. Templates accept ``Fact`` objects and literal
text; a slot with no ``Fact`` raises ``MissingProvenanceError`` rather than
falling back to a default (SPEC 4.2).
"""
