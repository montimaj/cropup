"""SPEC section 8: all fourteen endpoints respond, and no client input 500s.

Server rule 3 is "errors are named, not 500'd" -- ``errors.CropUpError`` has a
branch per failure mode and each maps to a status code. A 500 from this app
means a bug, so the second half of this file is a small fuzz over every endpoint
with input a hostile or confused client might send.
"""

from __future__ import annotations

import json

import pytest

#: SPEC section 8, in the order it lists them. ``{sid}`` is substituted.
ENDPOINTS = (
    ("GET", "/", None),
    ("GET", "/api/health", None),
    ("GET", "/api/capabilities", None),
    ("POST", "/api/session", {}),
    ("GET", "/api/session/{sid}", None),
    ("POST", "/api/session/{sid}/message", {"text": "my maize in Arusha is yellow"}),
    ("POST", "/api/session/{sid}/slots", {"crop_key": "maize"}),
    ("POST", "/api/session/{sid}/confirm", {"slots": ["crop"]}),
    ("POST", "/api/session/{sid}/run", {}),
    ("GET", "/api/session/{sid}/events?max_events=1&max_wait_s=2", None),
    ("POST", "/api/nlu/parse", {"text": "mahindi yangu"}),
    ("GET", "/api/vocab/crops?q=mah", None),
    ("GET", "/api/vocab/places?q=aru", None),
    ("GET", "/api/geo/field?lat=-3.38&lon=36.68", None),
)


def call(client, method: str, path: str, body, sid: str):
    path = path.format(sid=sid)
    if method == "GET":
        return client.get(path)
    return client.post(path, json=body)


def test_spec_section_8_lists_fourteen_endpoints():
    assert len(ENDPOINTS) == 14


@pytest.mark.parametrize("method,path,body", ENDPOINTS, ids=[p for _, p, _ in ENDPOINTS])
def test_every_spec_endpoint_responds(client, session_id, method, path, body):
    response = call(client, method, path, body, session_id)
    assert response.status_code < 500, response.text
    # Every one of them answers something a client can act on.
    assert response.status_code in (200, 201, 400, 409, 503), response.status_code


def test_the_routes_the_app_registers_cover_the_spec_list(app):
    registered = {
        (method, route.path)
        for route in app.routes
        for method in getattr(route, "methods", ()) or ()
        if method in ("GET", "POST")
    }
    for method, path, _ in ENDPOINTS:
        template = path.split("?")[0]
        assert (method, template) in registered, (method, template)


def test_the_index_serves_html_not_an_error(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_the_progress_stream_is_server_sent_events(client, session_id):
    response = client.get(
        f"/api/session/{session_id}/events", params={"max_events": 1, "max_wait_s": 2.0}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: open" in response.text


def test_a_session_survives_a_reload(client):
    sid = client.post("/api/session").json()["session_id"]
    client.post(f"/api/session/{sid}/slots", json={"crop_key": "maize"})
    rehydrated = client.get(f"/api/session/{sid}").json()
    assert rehydrated["bag"]["filled"]["crop"] == "Maize"
    assert rehydrated["ttl_s"] > 0


# --------------------------------------------------------------------------
# no client input produces a 500
# --------------------------------------------------------------------------

HOSTILE = (
    ("GET", "/api/session/does-not-exist", None),
    ("GET", "/api/session/../../etc/passwd", None),
    ("POST", "/api/session/does-not-exist/message", {"text": "hi"}),
    ("POST", "/api/session/{sid}/message", {"text": ""}),
    ("POST", "/api/session/{sid}/message", {"text": " \t\n "}),
    ("POST", "/api/session/{sid}/message", {"text": chr(0) + "é中 <script>x</script>"}),
    ("POST", "/api/session/{sid}/message", {"text": "{" * 500}),
    ("POST", "/api/session/{sid}/message", {}),
    ("POST", "/api/session/{sid}/message", {"text": 7}),
    ("POST", "/api/session/{sid}/slots", {"crop_key": "not-a-crop"}),
    ("POST", "/api/session/{sid}/slots", {"place_key": "not-a-place"}),
    ("POST", "/api/session/{sid}/slots", {"intent": "not-an-intent"}),
    ("POST", "/api/session/{sid}/slots", {"point": {"lat": 999.0, "lon": 0.0}}),
    ("POST", "/api/session/{sid}/slots", {"point": {"lat": "north", "lon": 0.0}}),
    ("POST", "/api/session/{sid}/slots", {"field_radius_m": -1.0}),
    ("POST", "/api/session/{sid}/slots", {"clear": ["nonsense"]}),
    ("POST", "/api/session/{sid}/slots", {"unlock": ["intent"] * 100}),
    ("POST", "/api/session/{sid}/slots", {"slots": {"crop": {"alternatives": 5}}}),
    ("POST", "/api/session/{sid}/slots", {"slots": {"crop": "Maize", "bogus": "x"}}),
    ("POST", "/api/session/{sid}/slots", {"origin": "not-an-origin"}),
    ("POST", "/api/session/{sid}/confirm", {"slots": ["timeframe"]}),
    ("POST", "/api/session/{sid}/confirm", {"slots": ["not-a-slot"]}),
    ("POST", "/api/session/{sid}/confirm", {"unconfirm": ["crop"]}),
    ("POST", "/api/session/{sid}/run", {"intent": "not-an-intent"}),
    ("POST", "/api/session/{sid}/run", {"intent": "fertilizer_advice"}),
    ("GET", "/api/session/{sid}/events?last_event_id=abc&max_events=1&max_wait_s=1", None),
    ("GET", "/api/session/{sid}/events?max_events=0", None),
    ("POST", "/api/nlu/parse", {"text": ""}),
    ("POST", "/api/nlu/parse", {"text": "x" * 5000}),
    ("POST", "/api/nlu/parse", {"text": None}),
    ("GET", "/api/vocab/crops?q=%20&limit=0", None),
    ("GET", "/api/vocab/crops?limit=not-a-number", None),
    ("GET", "/api/vocab/places?q=" + "z" * 400, None),
    ("GET", "/api/geo/field", None),
    ("GET", "/api/geo/field?sid=nope", None),
    ("GET", "/api/geo/field?lat=91&lon=0", None),
    ("GET", "/api/geo/field?lat=-3.38&lon=36.68&radius_m=-5", None),
    ("GET", "/api/geo/field?lat=-3.38&lon=36.68&analysis=made_up", None),
    ("GET", "/api/capabilities?sid=nope", None),
    ("GET", "/api/capabilities?lat=abc&lon=abc", None),
    ("POST", "/api/session", {"field_radius_m": 0.0}),
    ("POST", "/api/session", {"intent": "not-an-intent"}),
    ("POST", "/api/session", {"restore": {"slots": {"crop": {"value": ""}}}}),
    ("POST", "/api/session", {"restore": {"slots": {"crop": {"value": "Maize", "confidence": 5}}}}),
    ("POST", "/api/session", {"restore": {"created_at": "not-a-date"}}),
)


@pytest.mark.parametrize(
    "method,path,body", HOSTILE, ids=[f"{m}{p}{json.dumps(b)[:40]}" for m, p, b in HOSTILE]
)
def test_no_client_input_produces_a_500(client, session_id, method, path, body):
    response = call(client, method, path, body, session_id)
    assert response.status_code != 500, response.text
    assert response.status_code < 500 or response.status_code == 503


def test_an_oversized_body_is_refused_before_it_is_parsed(client, session_id):
    response = client.post(
        f"/api/session/{session_id}/message", json={"text": "a" * 200_000}
    )
    assert response.status_code == 413
    assert response.json()["error"]["type"] == "RequestTooLarge"
    assert response.json()["ran_earth_engine"] is False


def test_malformed_json_is_a_422_not_a_500(client, session_id):
    response = client.post(
        f"/api/session/{session_id}/message",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_every_named_error_carries_a_type_a_status_and_a_message(client):
    response = client.get("/api/session/does-not-exist")
    error = response.json()["error"]
    assert error["type"] == "SessionNotFound"
    assert error["status"] == 404
    assert error["message"]
    assert error["remedy"]
    assert response.json()["ran_earth_engine"] is False
