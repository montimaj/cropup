"""SPEC section 10's ``null_adapter`` run: every Earth Engine source unavailable.

``CROPUP_EE_ENABLED=0`` is set for the whole test session (see conftest), so
``bootstrap.initialize`` records "not ready" without ever importing ``ee``. The
app must still **answer or honestly decline**, and never crash.

"Honestly decline" is the load-bearing half: a decline that is really a quiet
default -- pH 7.0 interpolated into prose because iSDA was masked -- is the
failure SPEC section 4 exists to prevent, so the declines are checked for
content, not just for a status code.
"""

from __future__ import annotations

import pytest

from cropup import bootstrap


def test_earth_engine_is_genuinely_unavailable_for_this_whole_run(client):
    # ``client`` starts the app, whose lifespan attempts the bootstrap once.
    status = bootstrap.ee_status()
    assert status["attempted"] is True
    assert status["ready"] is False
    assert "null-adapter" in (status["error"] or "")


def test_the_app_boots_and_reports_itself_alive(client):
    response = client.get("/api/health")
    assert response.status_code in (200, 503)
    payload = response.json()
    assert payload["app"]["name"] == "cropup"
    assert payload["app"]["version"]
    # A degraded boot is a supported state, not a crash.
    assert isinstance(payload["status"], str)


def test_the_honesty_endpoint_answers_200_during_the_outage(client):
    """/api/capabilities is the one endpoint that must work when nothing else
    does, because it is what says so."""
    response = client.get("/api/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert payload["degraded"] is True
    assert payload["summary"]
    # The substance, rather than one vendor name: the outage is reported as a
    # capability that is not ok, and what the farmer loses by it is NAMED.
    # (This used to match the literal phrase "earth engine", which broke the
    # moment that farmer-facing string was written in plain language.)
    caps = payload["can"]
    assert caps["run_field_analysis"]["ok"] is False
    assert caps["run_field_analysis"]["detail"]
    assert payload["lost"], "an outage must name what it costs the farmer"

    # And the naming must stay farmer-readable. These strings render in the
    # capability strip on first paint with no gesture, so a spec citation or an
    # internal identifier reaching them is a regression.
    farmer_text = " ".join(payload["lost"]).lower()
    for leak in ("spec ", "_", "earth engine", "intent", "ledger"):
        assert leak not in farmer_text, f"internal vocabulary on first paint: {leak!r}"


def test_the_capability_report_never_leaks_the_operators_paths(client):
    blob = client.get("/api/capabilities").text + client.get("/api/health").text
    for prefix in ("/Users/", "/home/", "/root/", "/var/folders/"):
        assert prefix not in blob


def test_a_knowledge_turn_is_still_answered_from_the_committed_corpus(client, ee_tripwire):
    sid = client.post("/api/session").json()["session_id"]
    response = client.post(
        f"/api/session/{sid}/message",
        json={"text": "what fertilizer should I use for maize"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ran_earth_engine"] is False
    assert payload["answer"]["text"].strip(), "the app went silent instead of answering"


def test_an_earth_engine_question_is_declined_with_a_reason_not_a_guess(client):
    sid = client.post("/api/session").json()["session_id"]
    client.post(
        f"/api/session/{sid}/slots",
        json={"intent": "irrigation_advice", "crop_key": "maize", "place_key": "arusha"},
    )
    client.post(f"/api/session/{sid}/confirm")
    response = client.post(f"/api/session/{sid}/run")

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["type"] == "EarthEngineUnavailable"
    # The property, not the sentence: the decline says the measurement could not
    # happen, and points at what still works, instead of producing a number.
    message = error["message"].lower()
    assert any(w in message for w in ("cannot", "could not", "not available")), message
    assert any(w in message for w in ("satellite", "measure")), message
    assert any(w in message for w in ("guide", "knowledge", "ask")), (
        "a decline must name what the farmer can still do"
    )
    assert response.json()["ran_earth_engine"] is False
    # It says where to look and what still works, rather than offering a number.
    assert "/api/capabilities" in (error["remedy"] or "")
    assert "7.0" not in response.text


@pytest.mark.parametrize(
    "intent", ["field_health_check", "crop_problem_diagnosis", "irrigation_advice", "crop_selection"]
)
def test_every_earth_engine_intent_declines_the_same_way(client, intent):
    sid = client.post("/api/session").json()["session_id"]
    client.post(
        f"/api/session/{sid}/slots",
        json={"intent": intent, "crop_key": "maize", "place_key": "arusha"},
    )
    client.post(f"/api/session/{sid}/confirm")
    response = client.post(f"/api/session/{sid}/run")
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "EarthEngineUnavailable"


def test_the_field_that_would_have_been_sent_is_still_describable(client):
    """SPEC 8's ``/api/geo/field``: the farmer can see the exact polygon even
    when nothing can be measured on it."""
    response = client.get("/api/geo/field", params={"lat": -3.38, "lon": 36.68})
    assert response.status_code == 200
    payload = response.json()
    assert payload["preview"] is True
    assert payload["session_id"] is None


def test_the_vocabularies_are_committed_artifacts_and_need_no_network(client):
    crops = client.get("/api/vocab/crops", params={"q": "mahindi"}).json()
    places = client.get("/api/vocab/places", params={"q": "arusha"}).json()
    assert crops["results"][0]["name"] == "Maize"
    assert places["results"][0]["name"] == "Arusha"


def test_a_whole_conversation_completes_without_a_single_500(client, ee_tripwire):
    """The end-to-end null-adapter walk: open, chat, fill, confirm, run, reload."""
    statuses = []
    sid = client.post("/api/session").json()["session_id"]
    statuses.append(client.post(f"/api/session/{sid}/message", json={"text": "hello"}).status_code)
    statuses.append(
        client.post(
            f"/api/session/{sid}/message", json={"text": "my maize in Arusha is yellow"}
        ).status_code
    )
    statuses.append(
        client.post(
            f"/api/session/{sid}/slots",
            json={"intent": "field_health_check", "crop_key": "maize", "place_key": "arusha"},
        ).status_code
    )
    statuses.append(client.post(f"/api/session/{sid}/run").status_code)  # 409, the gate
    statuses.append(client.post(f"/api/session/{sid}/confirm").status_code)
    statuses.append(client.post(f"/api/session/{sid}/run").status_code)  # 503, the outage
    statuses.append(client.get(f"/api/session/{sid}").status_code)
    statuses.append(client.get("/api/capabilities", params={"sid": sid}).status_code)

    assert statuses == [200, 200, 200, 409, 200, 503, 200, 200]
