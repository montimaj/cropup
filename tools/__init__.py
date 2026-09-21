"""Dev-time CLIs that build the committed artifacts under ``cropup/data/``.

Not imported at runtime: the app reads the artifacts, never the builders. The
directory is named ``tools`` rather than ``scripts`` because the vendored
reference backend ships its own ``scripts/`` (SPEC 1.1), and this file makes it
a regular package so the two can never merge into one PEP-420 namespace.

Each module is run from the repo root, e.g. ``python tools/build_corpus.py``.
"""
