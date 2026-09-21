"""Every refusal this app makes, in one shape, saying nothing it should not.

SPEC section 8 lists the endpoints; nothing in it says what a *failure* looks
like, and that gap is where three real bugs lived:

* FastAPI's own 422 bypassed the app envelope entirely and echoed the rejected
  value back in ``input`` on every entry -- so an unauthenticated POST of 64 kB
  of junk came back as 64 kB of junk, an amplifier pointed at anybody whose
  address can be spoofed;
* host filesystem paths leaked. Redaction was applied to ``/api/health`` and
  ``/api/capabilities`` only, while the error handler answers *every* endpoint
  unauthenticated and was not redacted at all -- and the redaction itself was a
  denylist of five prefixes, so it failed open for any data directory outside
  ``/Users``, ``/home``, ``/root``, ``/var/folders`` or ``C:\\Users``;
* behaviour depended on whether the ASGI lifespan had run. Without it the
  bootstrap never happens, "has anybody looked at Earth Engine" stays *no*, and
  a confirmed ``POST /run`` sailed past the gate and started a run.

The path tests below assert against **this machine's real absolute paths**, so
a redaction that fails open fails here rather than passing on a laptop whose
home directory happens to match a hard-coded prefix.
"""

from __future__ import annotations

import json
import re
import types
from pathlib import Path

import pytest

from cropup.errors import DataFileError


REPO_ROOT = str(Path(__file__).resolve().parents[1])
HOME = str(Path.home())

#: The four the old denylist knew about, plus the four it did not. ``/opt``,
#: ``/srv``, ``/mnt`` and ``/data`` are where a container or a packaged deploy
#: puts ``CROPUP_DATA_DIR``, and none of them start with ``/Users/``.
LEAKY_PREFIXES = (
    "/Users/",
    "/home/",
    "/root/",
    "/var/folders/",
    "/opt/",
    "/srv/",
    "/mnt/",
    "/data/",
)


# --------------------------------------------------------------------------
# 422: the app's envelope, and none of the body back
# --------------------------------------------------------------------------


def _bad_bodies(sid: str):
    """One rejected request per endpoint that takes a body (SPEC section 8)."""
    return [
        ("/api/session", {"restore": {}, "nonsense": 1}),
        (f"/api/session/{sid}/message", {"text": "hi", "nonsense": 1}),
        (f"/api/session/{sid}/slots", {"lat": -3.38, "lon": 36.68}),
        (f"/api/session/{sid}/confirm", {"slots": ["crop"], "nonsense": 1}),
        (f"/api/session/{sid}/run", {"intent": "field_health_check", "nonsense": 1}),
        ("/api/nlu/parse", {"text": "hello", "nonsense": 1}),
    ]


def test_every_validation_failure_uses_this_apps_error_envelope(client, session_id):
    """``static/app.js`` carried a second code path purely to read FastAPI's
    ``{"detail": [...]}`` array. One failure shape, everywhere."""
    for path, body in _bad_bodies(session_id):
        response = client.post(path, json=body)
        assert response.status_code == 422, (path, response.text)
        payload = response.json()
        assert set(payload) >= {"error", "ran_earth_engine"}, path
        assert payload["ran_earth_engine"] is False
        error = payload["error"]
        assert error["type"] == "RequestValidationError"
        assert error["status"] == 422
        assert isinstance(error["detail"]["fields"], list)
        assert error["detail"]["fields"], path
        assert not isinstance(payload.get("detail"), list), (
            f"{path} still answers with FastAPI's own 422 array"
        )


def test_a_rejected_field_is_named_with_the_rule_it_broke(client, session_id):
    response = client.post(f"/api/session/{session_id}/slots", json={"lat": 999, "lon": 999})
    assert response.status_code == 422
    fields = response.json()["error"]["detail"]["fields"]
    assert sorted(entry["field"] for entry in fields) == ["body.lat", "body.lon"]
    assert all(entry["kind"] == "extra_forbidden" for entry in fields)
    assert all(entry["reason"] for entry in fields)


def test_a_client_typo_is_a_422_and_never_a_silent_no_op(client, session_id):
    """A pin travels as ``point``; top-level ``lat``/``lon`` is the obvious
    spelling and the wrong one. It used to be dropped by pydantic and answered
    200 with an empty ``slot_outcomes`` -- the farmer is then shown a field they
    did not set, which is the same class of failure as a silent default."""
    before = client.get(f"/api/session/{session_id}").json()["bag"]
    assert client.post(
        f"/api/session/{session_id}/slots", json={"lat": -3.38, "lon": 36.68}
    ).status_code == 422
    after = client.get(f"/api/session/{session_id}").json()["bag"]
    assert after["slots"] == before["slots"]


@pytest.mark.parametrize(
    "body",
    [
        {"text": "hi", "junk": "Q" * 4_000},
        {"slots": {"crop": "Q" * 4_000}},
        {"slots": {"crop": {"nested": "Q" * 4_000}}},
        {"intent": "Q" * 4_000},
    ],
)
def test_a_rejected_value_is_never_echoed_back_however_big_it_is(client, session_id, body):
    """FastAPI puts the rejected value in ``input`` on every entry. The field
    names and the rule each one broke are the whole diagnosis."""
    response = client.post(f"/api/session/{session_id}/slots", json=body)
    assert response.status_code == 422, response.text
    assert "Q" * 100 not in response.text
    # Comfortably smaller than the value it refused, i.e. not an amplifier.
    assert len(response.content) < 2_000, len(response.content)
    # FastAPI's entry is {"type", "loc", "msg", "input", "url"}; ours is
    # {"field", "kind", "reason"} and carries no room for a value.
    for entry in response.json()["error"]["detail"]["fields"]:
        assert set(entry) == {"field", "kind", "reason"}


def test_a_slot_value_is_bounded_by_the_text_it_becomes(client, session_id):
    """``MAX_VALUE_CHARS`` is documented as bounding one slot value, but the
    check only measured items that were already ``str``: a dict became a
    5,014-character crop by way of ``str(raw)``."""
    from cropup.web.server import MAX_VALUE_CHARS

    response = client.post(
        f"/api/session/{session_id}/slots",
        json={"slots": {"crop": {"nested": "z" * (MAX_VALUE_CHARS * 20)}}},
    )
    assert response.status_code == 422
    stored = client.get(f"/api/session/{session_id}").json()["bag"]["slots"].get("crop")
    if stored is not None:
        assert len(str(stored["value"])) <= MAX_VALUE_CHARS


def test_the_body_size_limit_is_answered_in_the_envelope_too(client, session_id):
    response = client.post(
        f"/api/session/{session_id}/message",
        content=json.dumps({"text": "x" * 200_000}),
        headers={"content-type": "application/json"},
    )
    assert response.status_code in (413, 422), response.status_code
    assert "error" in response.json()
    assert "x" * 100 not in response.text


# --------------------------------------------------------------------------
# no client-visible payload names a directory on this machine
# --------------------------------------------------------------------------


def _redacted_surface(client, session_id) -> list[str]:
    """Every SPEC section 8 endpoint whose payload goes through the redaction:
    the two honesty endpoints, the page, the vocabularies, the session view, and
    every failure -- the error handler answers all of those, unauthenticated."""
    seen = [
        client.get(path).text
        for path in (
            "/",
            "/api/health",
            "/api/capabilities",
            "/api/vocab/crops?q=maize",
            "/api/vocab/places?q=arusha",
            f"/api/session/{session_id}",
            "/api/session/does-not-exist",
        )
    ]
    seen += [client.post(path, json=body).text for path, body in _bad_bodies(session_id)]
    seen.append(client.post(f"/api/session/{session_id}/run").text)
    seen.append(client.get("/api/geo/field", params={"lat": 91, "lon": 0}).text)
    return seen


def _every_visible_payload(client, session_id) -> list[str]:
    """The whole client-visible surface, ``/api/nlu/parse`` included."""
    return _redacted_surface(client, session_id) + [
        client.post("/api/nlu/parse", json={"text": "my maize in arusha"}).text,
        client.post(
            f"/api/session/{session_id}/message", json={"text": "what fertilizer for maize"}
        ).text,
    ]


def _assert_no_host_paths(blob: str) -> None:
    assert REPO_ROOT not in blob, "the app told the client where it is installed"
    assert HOME not in blob, "the app told the client the operator's home directory"
    for prefix in LEAKY_PREFIXES:
        # ``<app>/cropup/data/corpus`` is deliberate and does not match: the
        # token has to *start* at a boundary to be an absolute path.
        assert not re.search(rf"(?<![\w.:~>/-]){re.escape(prefix)}", blob), (
            f"an absolute path under {prefix} reached the client"
        )


def test_the_redacted_surface_prints_no_absolute_path_of_this_machine(client, session_id):
    """Asserted against the real paths of *this* checkout, not a guessed
    prefix: if the redaction fails open, it fails here. This is the surface the
    app-wide redaction actually covers, and it is clean."""
    _assert_no_host_paths("".join(_redacted_surface(client, session_id)))


# Regression: POST /api/nlu/parse used to answer 200 with the operator's
# absolute model path (username, home layout and model revision) because
# _redact_host_paths reached _error_response, /api/health and
# /api/capabilities but not this unauthenticated 200. Now redacted at the
# same boundary as the other two.
def test_no_endpoint_at_all_prints_this_machines_absolute_paths(client, session_id):
    """The claim as it should hold: *no* client-visible payload, anywhere."""
    _assert_no_host_paths("".join(_every_visible_payload(client, session_id)))


#: A route on this app rather than a directory on this machine. Anything else
#: path-shaped in a JSON payload is a leak.
_APP_ROUTE = re.compile(r"^/$|^/(?:api|static|docs|redoc|openapi\.json|favicon\.ico)(?:[/?#.].*)?$")

#: A POSIX absolute path or a Windows drive path, starting at a boundary.
_PATH_SHAPED = re.compile(r"(?<![\w.:~>/-])(?:[A-Za-z]:[\\/]|/)[^\s,;'\"`<>()\[\]\\]*")


def test_the_only_absolute_paths_a_json_payload_carries_are_routes(client, session_id):
    """Fail-closed, stated as a test: a path nobody anticipated is a failure
    here rather than something the next deployment discovers. The denylist this
    replaced passed while ``/srv`` and ``/opt`` walked straight through it.

    Scoped to the redacted surface for the same reason as the test above; the
    ``/api/nlu/parse`` leak has its own strict xfail and is not hidden here."""
    payloads = [
        text for text in _redacted_surface(client, session_id) if text.startswith(("{", "["))
    ]
    assert len(payloads) >= 10, "the crawl stopped finding JSON"
    leaked = sorted(
        {
            token
            for text in payloads
            for token in _PATH_SHAPED.findall(text)
            if not _APP_ROUTE.match(token)
        }
    )
    assert leaked == [], f"path-shaped tokens that are not routes on this app: {leaked}"


def test_the_error_handler_itself_redacts_and_by_allowlist(client, monkeypatch):
    """``_error_response`` answers every endpoint, unauthenticated, and was the
    actual leak: ``DataFileError`` puts an absolute path in its message *and* in
    ``detail['path']``. ``/srv`` is the case the old denylist failed open on."""
    from cropup.web import server

    leaked = "/srv/cropup-data/gazetteer.json"

    def boom(*_args, **_kwargs):
        raise DataFileError(leaked, "unreadable")

    monkeypatch.setattr(server, "capability_report", boom)
    response = client.get("/api/capabilities")

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["type"] == "DataFileError"
    assert leaked not in response.text
    assert "/srv" not in response.text
    assert "<path hidden>" in error["message"]
    assert error["detail"]["path"] == "<path hidden>"
    # Still says *what* is wrong: the diagnosis survives the redaction.
    assert "data file unusable" in error["message"]
    assert error["remedy"]


def test_a_path_inside_the_app_is_named_relative_to_the_app_not_hidden(client, monkeypatch):
    """Which artifact is missing is the answer the probe is owed, so ``<app>``
    and ``<home>`` are kept deliberately -- they name a root the client may
    already know without naming this machine."""
    from cropup.web import server

    def boom(*_args, **_kwargs):
        raise DataFileError(f"{REPO_ROOT}/cropup/data/gazetteer.json", "missing")

    monkeypatch.setattr(server, "capability_report", boom)
    response = client.get("/api/capabilities")

    assert REPO_ROOT not in response.text
    assert "<app>/cropup/data/gazetteer.json" in response.json()["error"]["message"]


def test_redaction_leaves_the_things_that_only_look_like_paths_alone():
    """``mm/day`` and ``ISDASOIL/Africa/v1`` are a unit and an asset id. A
    redaction that ate them would take the provenance with it."""
    from cropup.web.server import _redact_host_paths

    kept = {
        "unit": "mm/day",
        "source_asset": "ISDASOIL/Africa/v1/ph",
        "route": "/api/capabilities",
        "docs": "/docs",
        "root": "/",
        "nested": ["COPERNICUS/S2_SR_HARMONIZED", "UCSB-CHG/CHIRPS/DAILY"],
    }
    assert _redact_host_paths(kept) == {**kept, "nested": list(kept["nested"])}
    assert _redact_host_paths("/etc/passwd") == "<path hidden>"
    assert _redact_host_paths(r"C:\Users\operator\cropup") == "<path hidden>"
    assert _redact_host_paths({"path": "/opt/cropup/data"})["path"] == "<path hidden>"


# --------------------------------------------------------------------------
# an app whose lifespan never ran refuses loudly instead of guessing
# --------------------------------------------------------------------------


def test_an_app_without_its_lifespan_refuses_every_request(settings):
    """Without the lifespan the bootstrap never happens, so nobody has looked at
    Earth Engine -- and a confirmed ``POST /run`` on such an app used to return
    200 and actually start a run. It is now a named 503 on every path, which is
    the loud failure the silent one deserved."""
    from fastapi.testclient import TestClient

    from cropup.web.server import create_app

    bare = TestClient(create_app(settings))  # deliberately NOT a context manager
    for method, path in (
        ("get", "/api/health"),
        ("get", "/api/capabilities"),
        ("post", "/api/session"),
        ("get", "/api/vocab/crops?q=maize"),
    ):
        response = getattr(bare, method)(path)
        assert response.status_code == 503, (path, response.status_code)
        assert response.json()["error"]["type"] == "LifespanNotRun"
        assert response.json()["ran_earth_engine"] is False


def test_the_conftest_client_does_run_the_lifespan(client):
    """The other half of the claim above: the fixture every test here uses is a
    context manager, so these tests are measuring a started app."""
    from cropup import bootstrap

    assert bootstrap.ee_status()["attempted"] is True
    assert client.get("/api/health").json()["app"]["name"] == "cropup"


# --------------------------------------------------------------------------
# the envelope's own verdict matches the ledger's
# --------------------------------------------------------------------------


def test_a_degraded_run_reports_itself_degraded_at_the_top_level():
    """``degraded``/``degraded_reason`` used to be set only on the RAG path, so
    a run whose own ``run.answer.degradation.degraded`` was true came back
    ``degraded: false`` and the capability strip never saw the verdict the
    ledger had already reached."""
    from cropup.web.server import _run_degradation_reason

    clean = types.SimpleNamespace(
        answer=types.SimpleNamespace(degradation={"degraded": False, "gaps": []})
    )
    assert _run_degradation_reason(clean) is None

    degraded = types.SimpleNamespace(
        answer=types.SimpleNamespace(
            degradation={
                "degraded": True,
                "gaps": ["soil_ph"],
                "fallbacks": ["ndvi"],
                "stale": ["esi"],
            }
        )
    )
    reason = _run_degradation_reason(degraded)
    assert reason
    # It points at the lists rather than restating values without provenance.
    assert "run.answer.degradation.gaps" in reason
    assert "run.answer.degradation.fallbacks" in reason
    assert "run.answer.degradation.stale" in reason
    assert "Nothing was substituted" in reason
    # And it does not quote a number it has no Fact for (SPEC section 4).
    assert "soil_ph" not in reason
