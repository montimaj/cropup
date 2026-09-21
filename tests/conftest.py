"""Shared fixtures. Everything here runs offline: no Earth Engine, no network.

The three environment variables below are set *before* anything imports
``cropup``, because :func:`cropup.config.get_settings` caches the first
``Settings`` it builds:

* ``CROPUP_EE_ENABLED=0`` is SPEC section 10's ``null_adapter`` run.
  ``bootstrap.initialize`` returns before ``import ee``, so nothing in the
  process can reach Earth Engine.
* ``HF_HUB_OFFLINE=1`` and ``CROPUP_ALLOW_MODEL_DOWNLOAD=0`` make
  ``huggingface_hub`` read the local cache only. Without them the NLU tier
  reaches the hub on the first turn, which is a network call and a source of
  non-determinism.

Nothing here asserts that the MiniLM encoder is present: it may be cached on one
machine and absent on another, so no test depends on which NLU tier answered.
"""

from __future__ import annotations

import os

# Forced, not defaulted: this suite must never reach Earth Engine or the network,
# whatever the shell it is started from says. SPEC section 10's live-EE
# integration run is a separate harness behind the ``ee`` marker.
os.environ["CROPUP_EE_ENABLED"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CROPUP_ALLOW_MODEL_DOWNLOAD"] = "0"
os.environ.setdefault("CROPUP_LOG_LEVEL", "critical")

import types  # noqa: E402

import pytest  # noqa: E402

from cropup import bootstrap  # noqa: E402
from cropup.config import reload_settings  # noqa: E402
from cropup.evidence import Ledger  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def settings():
    """One Settings for the whole run, rebuilt from the environment above."""
    return reload_settings()


@pytest.fixture(scope="session")
def app(settings):
    from cropup.web.server import create_app

    return create_app(settings)


@pytest.fixture(scope="session")
def client(app):
    """The real app, driven through TestClient. Session-scoped: the lifespan
    starts a background bootstrap thread, and one per test would be waste."""
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def session_id(client):
    response = client.post("/api/session")
    assert response.status_code == 201
    return response.json()["session_id"]


@pytest.fixture
def ee_tripwire(monkeypatch):
    """Put a booby-trapped ``ee`` in ``sys.modules``.

    Touching *any* attribute of it fails the test. ``import ee`` finds this
    object rather than the real client library, so a code path that reaches for
    Earth Engine on a turn that must not is caught here rather than silently
    working on a machine with credentials.
    """
    touched: list[str] = []

    module = types.ModuleType("ee")

    def _boom(name: str):
        touched.append(name)
        raise AssertionError(
            f"Earth Engine was touched (ee.{name}) on a turn that must never reach it (SPEC 4.4)"
        )

    module.__getattr__ = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "ee", module)
    # The recorded verdict must stay "not ready", or a handler could decide the
    # instrument is up and go looking for it.
    assert bootstrap.ee_status().get("ready") is not True
    yield touched
    assert touched == []


@pytest.fixture
def ledger():
    return Ledger(turn="test")
