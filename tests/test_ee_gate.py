"""SPEC section 4.4: never run Earth Engine on an unconfirmed field.

Six claims, driven through the real app:

* a natural-language turn never triggers Earth Engine;
* ``POST /run`` without ``POST /confirm`` is refused, and refused with the same
  verdict the chat turn got -- posting straight to ``/run`` bypasses nothing;
* **re-pointing a CONFIRMED field to different coordinates clears the
  confirmation.** A confirmation belongs to a meaning, not to a label: two map
  pins can share the label "arusha" and sit on different continents, and the
  farmer endorsed the old one;
* **resizing a CONFIRMED field clears it too, and the size itself is bounded.**
  The geometry Earth Engine is sent is ``Point(lon, lat).buffer(radius)``, so
  the radius is as much "which field" as the point is -- and it is the only
  part of that geometry a client can set which is not a coordinate;
* **a value that resolves to nothing cannot be confirmed**, and a confirmation
  is all-or-nothing;
* **an explicit ``unconfirm`` shuts the gate**, and one request can never both
  grant and withdraw the same authorisation.
"""

from __future__ import annotations

import pytest

from cropup.dialog.slots import ORIGIN_NLU, ORIGIN_USER, SlotBag
from cropup.errors import ConfirmationRequired


ARUSHA = {"lat": -3.38, "lon": 36.68}
DAR = {"lat": -6.80, "lon": 39.28}


def ready_session(client) -> str:
    """A session with an EE-routed intent and a resolved field, unconfirmed."""
    sid = client.post("/api/session").json()["session_id"]
    response = client.post(
        f"/api/session/{sid}/slots",
        json={"intent": "field_health_check", "crop_key": "maize", "place_key": "arusha"},
    )
    assert response.status_code == 200
    assert response.json()["action"]["kind"] == "confirm_field"
    return sid


def bag_of(client, sid) -> dict:
    return client.get(f"/api/session/{sid}").json()["bag"]


# --------------------------------------------------------------------------
# a natural-language turn never triggers Earth Engine
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "my maize in Arusha is yellow",
        "run the field health check on my field in Arusha now",
        "what is wrong with my mahindi",
        "",
    ],
)
def test_a_chat_turn_never_runs_earth_engine(client, ee_tripwire, text):
    sid = ready_session(client)
    response = client.post(f"/api/session/{sid}/message", json={"text": text})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ran_earth_engine"] is False
    assert payload["action"]["authorises_earth_engine"] is False
    assert payload["action"]["kind"] != "run_analysis"


def test_a_chat_turn_on_a_fully_confirmed_field_still_only_offers_run(client, ee_tripwire):
    sid = ready_session(client)
    assert client.post(f"/api/session/{sid}/confirm").status_code == 200
    payload = client.post(
        f"/api/session/{sid}/message", json={"text": "check my field now please"}
    ).json()
    assert payload["ran_earth_engine"] is False


def test_pure_nlu_inspection_has_no_side_effects_and_no_earth_engine(client, ee_tripwire):
    response = client.post("/api/nlu/parse", json={"text": "my maize in Arusha is yellow"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["side_effects"] is None
    assert payload["would_ask"]["authorises_earth_engine"] is False


def test_the_message_handler_cannot_mint_an_authorisation():
    """SPEC 4.4 holds structurally: the gate is a type, not a convention."""
    from cropup.web import dispatch

    with pytest.raises(PermissionError):
        dispatch.RunAuthorization(
            session_id="s",
            intent="field_health_check",
            analysis="plant_health",
            lat=-3.38,
            lon=36.68,
            radius_m=15.0,
            crop="Maize",
            place="Arusha",
            end_date=None,
            end_date_note=None,
            action=None,  # type: ignore[arg-type]
            authorised_at=None,  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------
# /run without /confirm is refused
# --------------------------------------------------------------------------


def test_run_without_confirm_is_a_409_naming_the_unconfirmed_slots(client, ee_tripwire):
    sid = ready_session(client)
    response = client.post(f"/api/session/{sid}/run")
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["type"] == "ConfirmationRequired"
    assert sorted(error["detail"]["missing_confirmations"]) == ["crop", "location"]
    assert "confirm" in error["remedy"]
    assert response.json()["ran_earth_engine"] is False


def test_the_run_endpoint_asks_the_same_pure_function_the_conversation_did(client):
    """Posting straight to /run bypasses nothing: the verdict comes from
    ``dialog.policy`` either way, so the two name the same unconfirmed slots."""
    sid = ready_session(client)
    conversation = client.get(f"/api/session/{sid}").json()["action"]
    assert conversation["kind"] == "confirm_field"
    assert conversation["authorises_earth_engine"] is False
    error = client.post(f"/api/session/{sid}/run").json()["error"]
    assert sorted(error["detail"]["missing_confirmations"]) == sorted(
        conversation["unconfirmed_slots"]
    )
    assert sorted(conversation["unconfirmed_slots"]) == ["crop", "location"]


def test_confirming_opens_the_gate_and_the_refusal_changes_kind(client):
    sid = ready_session(client)
    assert client.post(f"/api/session/{sid}/run").status_code == 409
    confirmed = client.post(f"/api/session/{sid}/confirm").json()
    assert sorted(confirmed["confirmed"]) == ["crop", "location"]
    response = client.post(f"/api/session/{sid}/run")
    # The gate is open; with CROPUP_EE_ENABLED=0 the instrument is not. That is
    # a different refusal, and saying so is the whole point.
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "EarthEngineUnavailable"


def test_a_restored_snapshot_cannot_mint_a_confirmation(client):
    """The browser keeps its own copy of the bag. It cannot bring a confirmation
    back with it: the gate is opened in this process and nowhere else."""
    response = client.post(
        "/api/session",
        json={
            "restore": {
                "slots": {
                    "location": {
                        "slot": "location",
                        "value": "arusha",
                        "confirmed": True,
                        "detail": ARUSHA,
                    },
                    "crop": {"slot": "crop", "value": "Maize", "confirmed": True},
                }
            }
        },
    )
    assert response.status_code == 201
    payload = response.json()
    assert payload["restored"] is True
    assert payload["bag"]["confirmed"] == []
    sid = payload["session_id"]
    client.post(f"/api/session/{sid}/slots", json={"intent": "field_health_check"})
    assert client.post(f"/api/session/{sid}/run").status_code == 409


def test_unconfirming_shuts_the_gate_again(client):
    """Was a strict xfail: POST /confirm {'unconfirm': [...]} withdrew the
    endorsement and then re-granted it in the same request, because the empty
    ``slots`` branch asked the policy which slots were unconfirmed and confirmed
    exactly those. A request that names ``unconfirm`` now confirms only what it
    also names in ``slots``, so the gate can be shut again."""
    sid = ready_session(client)
    client.post(f"/api/session/{sid}/confirm")
    withdrawn = client.post(
        f"/api/session/{sid}/confirm", json={"unconfirm": ["location"]}
    ).json()
    assert withdrawn["unconfirmed"] == ["location"]
    assert "location" not in withdrawn["confirmed"]
    assert "location" not in bag_of(client, sid)["confirmed"]
    response = client.post(f"/api/session/{sid}/run")
    assert response.status_code == 409
    assert response.json()["error"]["detail"]["missing_confirmations"] == ["location"]


def test_unconfirm_works_on_the_bag_itself(client):
    """The SlotBag primitive is correct; it is the endpoint above that undoes it."""
    bag = SlotBag("s")
    bag.set("location", "arusha", origin=ORIGIN_USER, detail=dict(ARUSHA))
    bag.confirm("location")
    assert bag.unconfirm("location") == ("location",)
    assert bag.is_confirmed("location") is False


# --------------------------------------------------------------------------
# the regression: re-pointing a CONFIRMED field clears the confirmation
# --------------------------------------------------------------------------


def test_repointing_a_confirmed_field_to_new_coordinates_clears_confirmation(client):
    sid = ready_session(client)
    assert sorted(client.post(f"/api/session/{sid}/confirm").json()["confirmed"]) == [
        "crop",
        "location",
    ]
    before = bag_of(client, sid)
    assert before["slots"]["location"]["confirmed"] is True
    assert before["slots"]["location"]["detail"]["lat"] == pytest.approx(ARUSHA["lat"])

    # The same label, a different place -- 500 km away.
    moved = client.post(
        f"/api/session/{sid}/slots", json={"point": {**DAR, "label": "arusha"}}
    )
    assert moved.status_code == 200
    assert moved.json()["slot_outcomes"]["location"] == "set"
    assert moved.json()["action"]["kind"] == "confirm_field"

    after = bag_of(client, sid)
    assert after["slots"]["location"]["detail"]["lat"] == pytest.approx(DAR["lat"])
    assert after["slots"]["location"]["confirmed"] is False
    assert after["confirmed"] == ["crop"]  # the crop did not move

    refused = client.post(f"/api/session/{sid}/run")
    assert refused.status_code == 409
    assert refused.json()["error"]["detail"]["missing_confirmations"] == ["location"]


def test_the_withdrawal_is_journalled_so_the_ui_can_say_why(client):
    sid = ready_session(client)
    client.post(f"/api/session/{sid}/confirm")
    client.post(f"/api/session/{sid}/slots", json={"point": {**DAR, "label": "arusha"}})
    events = bag_of(client, sid)["events"]
    reconfirm = [event for event in events if event["outcome"] == "reconfirm_required"]
    assert len(reconfirm) == 1
    assert "confirmation withdrawn" in reconfirm[0]["detail"]


def test_re_writing_the_identical_point_keeps_the_confirmation(client):
    """Only a change of *meaning* withdraws it; an idle re-write does not, or
    every round trip through the browser would cost a re-confirmation."""
    sid = client.post("/api/session").json()["session_id"]
    client.post(f"/api/session/{sid}/slots", json={"intent": "field_health_check"})
    client.post(
        f"/api/session/{sid}/slots", json={"point": {**ARUSHA, "label": "my field"}}
    )
    client.post(f"/api/session/{sid}/slots", json={"crop_key": "maize"})
    client.post(f"/api/session/{sid}/confirm")
    assert bag_of(client, sid)["slots"]["location"]["confirmed"] is True

    again = client.post(
        f"/api/session/{sid}/slots", json={"point": {**ARUSHA, "label": "my field"}}
    )
    assert again.json()["slot_outcomes"]["location"] == "set"
    assert bag_of(client, sid)["slots"]["location"]["confirmed"] is True


# --------------------------------------------------------------------------
# the same rule, at the SlotBag level
# --------------------------------------------------------------------------


def test_a_confirmation_is_bound_to_the_identity_not_the_label():
    bag = SlotBag("s")
    bag.set("location", "arusha", origin=ORIGIN_USER, detail=dict(ARUSHA))
    bag.confirm("location")
    assert bag.is_confirmed("location") is True

    bag.set("location", "arusha", origin=ORIGIN_USER, detail=dict(DAR))
    assert bag.is_confirmed("location") is False, "same label, different continent"


def test_a_value_arriving_pre_confirmed_loses_the_flag():
    from cropup.dialog.slots import SlotValue

    bag = SlotBag("s")
    stored = bag._write(  # the one door into the bag
        SlotValue(
            slot="crop",
            value="Maize",
            origin=ORIGIN_USER,
            confirmed=True,
            confirmed_at=__import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ),
        )
    )
    assert stored.confirmed is False
    assert stored.confirmed_at is None


def test_a_confirmed_flag_without_a_timestamp_is_refused():
    from cropup.dialog.slots import SlotValue

    with pytest.raises(ValueError) as caught:
        SlotValue(slot="crop", value="Maize", confirmed=True)
    assert "needs the moment it happened" in str(caught.value)


def test_nlu_never_overwrites_a_slot_the_farmer_set():
    bag = SlotBag("s")
    bag.set("location", "arusha", origin=ORIGIN_USER, detail=dict(ARUSHA))
    refused = bag.set("location", "dodoma", origin=ORIGIN_NLU)
    assert refused is None
    assert bag.value("location") == "arusha"
    assert any(event["outcome"] == "refused_locked" for event in bag.events)


def test_confirming_an_empty_slot_is_an_error_not_a_silent_no_op():
    bag = SlotBag("s")
    with pytest.raises(ValueError):
        bag.confirm("location")


def test_the_gate_raises_the_named_error_rather_than_a_bare_exception():
    error = ConfirmationRequired(("location", "crop"))
    assert error.missing_confirmations == ("location", "crop")
    assert "must be confirmed before Earth Engine runs" in str(error)


# --------------------------------------------------------------------------
# the same rule for the OTHER half of the geometry: the field radius
#
# ``Point(lon, lat).buffer(radius)`` is what Earth Engine is sent, so the
# radius is as much "which field" as the point is. A farmer who endorsed a
# 15 m field at Arusha did not endorse a 2.5 km disc at Arusha -- that disc is
# ~19 km2, spanning several distinct gazetteer entries (SPEC 7 rounds them to
# ~1 km), and its ``reduceRegion`` mean would be a district average shown as
# "your field": a plausible wrong number, which is the failure SPEC 4 exists
# to prevent. It is also the only geometry input a client can set that is not
# a coordinate, so it is bounded at the same boundary the coordinates are.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "radius_m",
    [100_000.0, 5_000.1, 0.5, 0.0, -15.0],
)
def test_a_field_radius_outside_the_bounds_is_refused_not_clamped(client, radius_m):
    """Refused, never clamped: a clamp would send Earth Engine a field the
    farmer never asked for, which is the same fabrication a defaulted pH is."""
    sid = ready_session(client)
    response = client.post(f"/api/session/{sid}/slots", json={"field_radius_m": radius_m})
    assert response.status_code in (400, 422), response.text
    # Nothing was written: the bag still has no radius of its own.
    assert bag_of(client, sid)["field_radius_m"] is None


@pytest.mark.parametrize("literal", ["1e400", "-1e400", "1e999"])
def test_an_infinite_radius_is_refused_at_the_wire(client, literal):
    """``inf`` is the one that slips past a naive bound: ``inf > 0`` is True,
    so every "must be positive" check passes it. Python's json module will not
    emit it, so it goes on the wire as an overflowing literal -- which is how a
    real client would send it."""
    sid = ready_session(client)
    response = client.post(
        f"/api/session/{sid}/slots",
        content=f'{{"field_radius_m": {literal}}}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code in (400, 422), response.text
    assert bag_of(client, sid)["field_radius_m"] is None


@pytest.mark.parametrize("radius_m", [1.0, 15.0, 5_000.0])
def test_the_bounds_are_inclusive_and_a_real_field_fits_inside_them(client, radius_m):
    sid = ready_session(client)
    response = client.post(f"/api/session/{sid}/slots", json={"field_radius_m": radius_m})
    assert response.status_code == 200, response.text
    assert bag_of(client, sid)["field_radius_m"] == pytest.approx(radius_m)


def test_widening_a_confirmed_field_clears_the_confirmation_and_shuts_the_gate(client):
    """The radius regression, driven exactly like the re-point one above.

    Confirm a 15 m field, widen it to 2.5 km, and the gate must be shut again.
    This used to widen the geometry while the confirmation stayed open, so the
    run that followed sent Earth Engine a field nobody had approved.
    """
    sid = ready_session(client)
    assert client.post(f"/api/session/{sid}/slots", json={"field_radius_m": 15}).status_code == 200
    assert sorted(client.post(f"/api/session/{sid}/confirm").json()["confirmed"]) == [
        "crop",
        "location",
    ]
    before = bag_of(client, sid)
    assert before["slots"]["location"]["confirmed"] is True
    assert before["slots"]["location"]["detail"]["radius_m"] == pytest.approx(15.0)

    widened = client.post(f"/api/session/{sid}/slots", json={"field_radius_m": 2500})
    assert widened.status_code == 200

    after = bag_of(client, sid)
    assert after["field_radius_m"] == pytest.approx(2500.0)
    assert after["slots"]["location"]["detail"]["radius_m"] == pytest.approx(2500.0)
    assert after["slots"]["location"]["confirmed"] is False
    assert after["confirmed"] == ["crop"]  # the crop did not change size

    refused = client.post(f"/api/session/{sid}/run")
    assert refused.status_code == 409
    assert refused.json()["error"]["detail"]["missing_confirmations"] == ["location"]


def test_the_resize_is_journalled_with_the_two_sizes(client):
    """The UI has to be able to say *why* it is asking again."""
    sid = ready_session(client)
    client.post(f"/api/session/{sid}/slots", json={"field_radius_m": 15})
    client.post(f"/api/session/{sid}/confirm")
    client.post(f"/api/session/{sid}/slots", json={"field_radius_m": 2500})

    withdrawals = [
        event
        for event in bag_of(client, sid)["events"]
        if event["outcome"] == "reconfirm_required"
    ]
    assert len(withdrawals) == 1
    detail = withdrawals[0]["detail"]
    assert "confirmation withdrawn" in detail
    assert "15.0" in detail and "2500.0" in detail


def test_re_stating_the_same_radius_keeps_the_confirmation(client):
    """Only a change of meaning withdraws it, here as for the coordinates."""
    sid = ready_session(client)
    client.post(f"/api/session/{sid}/slots", json={"field_radius_m": 15})
    client.post(f"/api/session/{sid}/confirm")
    client.post(f"/api/session/{sid}/slots", json={"field_radius_m": 15.0})
    assert bag_of(client, sid)["slots"]["location"]["confirmed"] is True
    assert client.post(f"/api/session/{sid}/run").status_code == 503  # the gate is open


def test_choosing_a_size_for_the_first_time_also_withdraws_the_confirmation(client):
    """Deliberate. Moving from "no size chosen" to an explicit number is a
    statement about the field that the previous confirmation did not contain,
    and the module errs towards "these differ": one extra re-confirmation is
    cheap, the other error is the bug SPEC 4.4 exists to prevent."""
    sid = ready_session(client)
    client.post(f"/api/session/{sid}/confirm")
    assert bag_of(client, sid)["slots"]["location"]["detail"].get("radius_m") is None

    client.post(f"/api/session/{sid}/slots", json={"field_radius_m": 2500})
    assert bag_of(client, sid)["slots"]["location"]["confirmed"] is False
    assert client.post(f"/api/session/{sid}/run").status_code == 409


def test_the_polygon_preview_will_not_draw_an_infinite_field(client):
    """``Query(gt=0.0)`` lets ``inf`` through -- ``inf > 0`` is True -- so the
    preview used to build a scratch bag with an infinite radius. SPEC 8 calls
    this endpoint "the exact polygon that WILL be sent to EE"; an unbounded one
    is not a polygon."""
    for radius_m in ("1e400", "100000", "0"):
        response = client.get(
            "/api/geo/field",
            params={"lat": -3.38, "lon": 36.68, "radius_m": radius_m},
        )
        assert response.status_code in (400, 422), (radius_m, response.text)
    good = client.get(
        "/api/geo/field", params={"lat": -3.38, "lon": 36.68, "radius_m": 5000}
    )
    assert good.status_code == 200


def test_a_restored_snapshot_cannot_smuggle_an_out_of_range_radius_in(client):
    """The browser's snapshot goes through the same bound the endpoint does."""
    refused = client.post("/api/session", json={"restore": {"field_radius_m": 100_000}})
    assert refused.status_code == 400
    assert "between 1 and 5000" in refused.json()["error"]["message"]

    accepted = client.post("/api/session", json={"restore": {"field_radius_m": 25}})
    assert accepted.status_code == 201
    assert accepted.json()["bag"]["field_radius_m"] == pytest.approx(25.0)


# -- the bound itself, at the module that owns it --------------------------


@pytest.mark.parametrize(
    "radius_m",
    [
        float("inf"),
        float("-inf"),
        float("nan"),
        0.0,
        -1.0,
        0.999,
        5_000.001,
        1_000_000.0,
        "15",
        None,
        True,
    ],
)
def test_the_bag_refuses_every_radius_that_is_not_a_field(radius_m):
    from cropup.dialog.slots import validate_field_radius_m

    with pytest.raises(ValueError):
        validate_field_radius_m(radius_m)


def test_the_radius_is_read_only_so_the_gate_cannot_be_side_stepped():
    """Assigning it would resize the field without going through ``_write``,
    which is where the confirmation comparison lives."""
    bag = SlotBag("s")
    with pytest.raises(AttributeError):
        bag.field_radius_m = 2500.0  # type: ignore[misc]
    assert bag.set_field_radius(2500.0) == 2500.0
    assert bag.field_radius_m == 2500.0
    assert bag.set_field_radius(None) is None


def test_the_radius_is_part_of_the_locations_identity():
    bag = SlotBag("s")
    bag.set("location", "arusha", origin=ORIGIN_USER, detail=dict(ARUSHA))
    bag.set_field_radius(15.0)
    bag.confirm("location")
    assert bag.is_confirmed("location") is True
    assert bag.get("location").identity()[-1] == pytest.approx(15.0)

    bag.set_field_radius(2500.0)
    assert bag.is_confirmed("location") is False, "same pin, a field 166x wider"


def test_a_client_cannot_smuggle_its_own_radius_into_the_detail():
    """``_write`` re-stamps the bag's radius over whatever arrives, so the
    detail the gate reads cannot drift from the number the bag holds."""
    bag = SlotBag("s")
    bag.set_field_radius(15.0)
    bag.set(
        "location",
        "arusha",
        origin=ORIGIN_USER,
        detail={**ARUSHA, "radius_m": 4_999.0},
    )
    assert bag.get("location").detail["radius_m"] == pytest.approx(15.0)
    assert bag.get("location").identity()[-1] == pytest.approx(15.0)


# --------------------------------------------------------------------------
# a confirmation is only ever given to something that CAN be endorsed
# --------------------------------------------------------------------------


def test_an_unresolvable_location_cannot_be_confirmed(client):
    """``999,999`` is in no gazetteer and carries no coordinates, so the policy
    is asking about it -- which is exactly when the old fallback ("confirm the
    confirmable slots the bag has") reached for it. Confirming it reported a
    confirmed field that could never be put on a map."""
    sid = client.post("/api/session").json()["session_id"]
    client.post(f"/api/session/{sid}/slots", json={"intent": "field_health_check"})
    written = client.post(f"/api/session/{sid}/slots", json={"slots": {"location": "999,999"}})
    assert written.status_code == 200
    assert written.json()["bag"]["slots"]["location"]["resolved"] is False

    for body in (None, {"slots": ["location"]}):
        refused = client.post(f"/api/session/{sid}/confirm", json=body)
        assert refused.status_code == 400, refused.text
        assert "cannot confirm location" in refused.json()["error"]["message"]
    assert bag_of(client, sid)["confirmed"] == []
    assert client.post(f"/api/session/{sid}/run").status_code == 409


def test_a_confirmation_is_all_or_nothing(client):
    """One act over one field. Half of it applied and then a refusal would
    leave the session claiming the farmer endorsed a crop for a field that was
    rejected in the same breath."""
    sid = client.post("/api/session").json()["session_id"]
    client.post(
        f"/api/session/{sid}/slots",
        json={"intent": "field_health_check", "crop_key": "maize"},
    )
    client.post(f"/api/session/{sid}/slots", json={"slots": {"location": "999,999"}})
    refused = client.post(f"/api/session/{sid}/confirm", json={"slots": ["crop", "location"]})
    assert refused.status_code == 400
    assert bag_of(client, sid)["confirmed"] == [], "the crop was endorsed anyway"


def test_the_bag_refuses_to_endorse_a_location_with_no_coordinates():
    bag = SlotBag("s")
    bag.set("location", "somewhere nobody mapped", origin=ORIGIN_USER)
    with pytest.raises(ValueError) as caught:
        bag.confirm("location")
    assert "cannot confirm location" in str(caught.value)
    assert bag.is_confirmed("location") is False


# --------------------------------------------------------------------------
# withdrawing an authorisation
# --------------------------------------------------------------------------


def test_one_request_cannot_both_grant_and_withdraw_the_same_authorisation(client):
    sid = ready_session(client)
    client.post(f"/api/session/{sid}/confirm")
    response = client.post(
        f"/api/session/{sid}/confirm",
        json={"slots": ["location"], "unconfirm": ["location"]},
    )
    assert response.status_code == 400
    assert "named in both slots and unconfirm" in response.json()["error"]["message"]
    # And it picked neither: the bag is exactly as it was.
    assert sorted(bag_of(client, sid)["confirmed"]) == ["crop", "location"]


def test_an_unconfirm_only_request_confirms_nothing_at_all(client):
    """The shape of the old bug: the reply said ``confirmed: ['location']`` and
    ``unconfirmed: ['location']`` in one breath."""
    sid = ready_session(client)
    client.post(f"/api/session/{sid}/confirm")
    payload = client.post(f"/api/session/{sid}/confirm", json={"unconfirm": ["location"]}).json()
    assert payload["confirmed"] == []
    assert payload["unconfirmed"] == ["location"]
    assert payload["action"]["authorises_earth_engine"] is False


def test_withdrawing_every_slot_leaves_nothing_authorised(client):
    sid = ready_session(client)
    client.post(f"/api/session/{sid}/confirm")
    client.post(f"/api/session/{sid}/confirm", json={"unconfirm": ["location", "crop"]})
    assert bag_of(client, sid)["confirmed"] == []
    refused = client.post(f"/api/session/{sid}/run")
    assert refused.status_code == 409
    assert sorted(refused.json()["error"]["detail"]["missing_confirmations"]) == [
        "crop",
        "location",
    ]


def test_the_policy_owns_the_one_definition_of_what_a_bare_confirm_covers():
    """``confirmable_slots`` exists so the endpoint does not carry a second
    copy of the rule; ``exclude`` is what makes withdraw-and-re-grant-in-one
    -request impossible by construction rather than by call ordering."""
    from cropup.dialog import policy as policy_mod

    bag = SlotBag("s")
    bag.set("location", "arusha", origin=ORIGIN_USER, detail=dict(ARUSHA))
    bag.set("crop", "Maize", origin=ORIGIN_USER)
    assert sorted(policy_mod.confirmable_slots(bag)) == ["crop", "location"]
    assert policy_mod.confirmable_slots(bag, exclude=("location",)) == ("crop",)
    assert policy_mod.confirmable_slots(bag, exclude=("location", "crop")) == ()
