"""Earth Engine data layer: one module per measured quantity.

May import ``cropup.evidence``, ``cropup.config``, ``cropup.bootstrap`` and
``cropup.geo.registry``. Must not import ``cropup.analysis``, ``cropup.nlu`` or
``cropup.rag``. Every leaf returns ``Fact`` or ``Missing``, never a bare float.
"""
