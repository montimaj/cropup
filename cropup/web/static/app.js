/* CropUp — SPEC section 9 frontend. Vanilla JS, no build step, no framework.
 *
 * Four things this file is responsible for, in priority order:
 *
 *  1. THE FABRICATION FIREWALL, ON SCREEN. Farmer-facing text is never built
 *     here. Every sentence arrives as an Answer of sections -> lines ->
 *     segments, and this file only decides how a segment is styled. A segment
 *     of kind "fact" must carry provenance; if it does not, it is rendered as a
 *     named absence, never as a plain number. A segment of kind "missing" is
 *     always drawn -- skipping it would be the silent default the whole project
 *     exists to prevent.
 *  2. ONE SLOT BAG, TWO VIEWS. "Ask" and "Form" are tabs over the same bag,
 *     and the slot frame sits under both so switching cannot look like losing
 *     state. A slot the classifier filled is a suggestion; a slot the farmer
 *     set is locked; a slot flagged needs_confirmation never looks settled.
 *  3. THE GATE (SPEC 4.4). Earth Engine runs only from a deliberate
 *     confirmation followed by a deliberate Run. Nothing here posts /run
 *     implicitly, and Run is offered only when the server says may_run.
 *  4. HONEST DEGRADATION. The capability strip is always visible, every error
 *     envelope is shown with its remedy, and the map falls back to a local
 *     plan view rather than a grey box when the tiles or the CDN are gone.
 */
(function () {
  "use strict";

  /* ================================================================
   * small helpers
   * ============================================================= */

  var $ = function (id) { return document.getElementById(id); };

  function el(tag, attrs, kids) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        var v = attrs[k];
        if (v === null || v === undefined || v === false) return;
        if (k === "class") node.className = v;
        else if (k === "text") node.textContent = v;
        // There is deliberately no "html" key. Every farmer-facing string on
        // this page is either the farmer's own words or a verbatim corpus
        // snippet, so the only two ways text reaches the DOM are textContent
        // and createTextNode -- neither of which can parse markup. A helper
        // that CAN assign innerHTML is one refactor away from being used.
        else if (k === "style") node.style.cssText = v;  // CSSOM, not a style= attribute the CSP blocks
        else if (k.slice(0, 2) === "on") node.addEventListener(k.slice(2), v);
        else if (v === true) node.setAttribute(k, "");
        else node.setAttribute(k, String(v));
      });
    }
    (kids || []).forEach(function (kid) {
      if (kid === null || kid === undefined) return;
      node.appendChild(typeof kid === "string" ? document.createTextNode(kid) : kid);
    });
    return node;
  }

  function clear(node) { while (node && node.firstChild) node.removeChild(node.firstChild); }

  function titleise(s) {
    return String(s || "").replace(/_/g, " ").replace(/^\w/, function (c) { return c.toUpperCase(); });
  }

  function num(v, digits) {
    return typeof v === "number" && isFinite(v) ? v.toFixed(digits === undefined ? 2 : digits) : String(v);
  }

  function metres(v) {
    if (typeof v !== "number") return "at the exact pixel";
    return v >= 1000 ? (v / 1000) + " km" : v + " m";
  }

  function shortTime(iso) {
    if (!iso) return "";
    var d = new Date(iso);
    return isNaN(d.getTime()) ? String(iso) : d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  var LEG_COLOURS = {
    vegetation: "#2f8f5b", thermal: "#c2622a", soil: "#8a6a3a",
    water: "#2b6cb0", context: "#7a5aa8", suitability: "#0f8e8e", climate: "#a8577f"
  };
  var PALETTE = ["#2f8f5b", "#c2622a", "#2b6cb0", "#7a5aa8", "#8a6a3a", "#0f8e8e", "#a8577f"];
  function legColour(leg, i) { return LEG_COLOURS[leg] || PALETTE[i % PALETTE.length]; }

  /* SPEC 5.2's fixed taxonomy. The server validates it (400 on anything else),
   * and there is no endpoint that lists it, so it is mirrored here; a wrong
   * name would surface immediately as a named 400, not as a silent no-op. */
  var INTENTS = [
    { name: "field_health_check", label: "Field health check", route: "earth_engine" },
    { name: "crop_problem_diagnosis", label: "Crop problem diagnosis", route: "earth_engine" },
    { name: "irrigation_advice", label: "Irrigation advice", route: "earth_engine" },
    { name: "crop_selection", label: "Crop selection", route: "earth_engine" },
    { name: "fertilizer_advice", label: "Fertiliser advice", route: "rag" },
    { name: "soil_fertility_management", label: "Soil fertility management", route: "rag" },
    { name: "seed_variety_selection", label: "Seed and variety selection", route: "rag" },
    { name: "crop_management_practice", label: "Crop management practice", route: "rag" },
    { name: "market_and_inputs_supply", label: "Market and input supply", route: "rag" },
    { name: "livestock_and_adjacent", label: "Livestock and adjacent", route: "rag" },
    { name: "agronomy_concept_explainer", label: "Agronomy concept explainer", route: "rag" },
    { name: "out_of_scope_or_unclear", label: "Unclear or out of scope", route: "clarify" }
  ];
  var ROUTE_TEXT = {
    earth_engine: "measured from satellites — needs a confirmed field",
    rag: "answered from the cited knowledge base — never touches Earth Engine",
    clarify: "CropUp will ask you what you mean"
  };

  /* ================================================================
   * state
   * ============================================================= */

  var S = {
    sid: null,
    bag: null,
    field: null,
    action: null,
    awaiting: null,
    intentMeta: null,
    caps: null,
    events: null,          // EventSource
    run: { planned: [], byName: {}, active: false },
    mapMode: "auto",       // auto | plan
    map: null,
    busy: false
  };

  var STORE_KEY = "cropup.session.v1";

  function saveSnapshot() {
    try {
      if (!S.bag) return;
      localStorage.setItem(STORE_KEY, JSON.stringify(S.bag));
    } catch (e) { /* private mode: the session simply will not survive a reload */ }
  }
  function loadSnapshot() {
    try {
      var raw = localStorage.getItem(STORE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch (e) { return null; }
  }
  function dropSnapshot() {
    try { localStorage.removeItem(STORE_KEY); } catch (e) { /* nothing to do */ }
  }

  /* ================================================================
   * HTTP
   * ============================================================= */

  function api(path, method, body) {
    var opts = { method: method || "GET", headers: {} };
    if (body !== undefined) {
      opts.headers["content-type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    return fetch(path, opts).then(function (res) {
      return res.text().then(function (raw) {
        var parsed = null;
        try { parsed = raw ? JSON.parse(raw) : null; } catch (e) { parsed = null; }
        if (res.ok) return parsed;
        var err = new Error(errorMessage(res, parsed));
        err.status = res.status;
        err.envelope = parsed;
        throw err;
      });
    });
  }

  function errorMessage(res, parsed) {
    if (parsed && parsed.error && parsed.error.message) return parsed.error.message;
    if (parsed && Array.isArray(parsed.detail)) {
      return parsed.detail.map(function (d) { return (d.loc || []).join(".") + ": " + d.msg; }).join("; ");
    }
    return res.status + " " + res.statusText;
  }

  function showError(err, where) {
    var env = err && err.envelope && err.envelope.error;
    toast({
      title: (env && env.type ? env.type : "Request failed") + " · " + (err.status || "network"),
      body: err && err.message ? err.message : String(err),
      remedy: (env && env.remedy) || (env && env.note) || (where ? "while " + where : null),
      kind: "error"
    });
  }

  function toast(spec) {
    var host = $("toasts");
    var node = el("div", { class: "toast" + (spec.kind === "info" ? " toast--info" : "") }, [
      el("button", {
        class: "toast__x", type: "button", "aria-label": "Dismiss",
        onclick: function () { node.remove(); }
      }, ["×"]),
      el("div", { class: "toast__t", text: spec.title }),
      el("div", { text: spec.body || "" }),
      spec.remedy ? el("div", { class: "toast__r", text: spec.remedy }) : null
    ]);
    host.appendChild(node);
    setTimeout(function () { node.remove(); }, spec.kind === "info" ? 7000 : 14000);
  }

  /* ================================================================
   * segments: the firewall on screen
   * ============================================================= */

  function provRow(dl, label, value) {
    if (value === null || value === undefined || value === "") return;
    dl.appendChild(el("dt", { text: label }));
    dl.appendChild(el("dd", { text: String(value) }));
  }

  function factCard(p) {
    var dl = el("dl");
    provRow(dl, "value", p.rendered !== undefined && p.rendered !== null ? p.rendered : p.value);
    provRow(dl, "unit", p.unit);
    provRow(dl, "instrument", p.source_asset);
    provRow(dl, "band", p.band);
    provRow(dl, "observed", p.observed_on || "no observation date on this layer");
    provRow(dl, "retrieved", p.retrieved_at);
    provRow(dl, "resolution", typeof p.resolution_m === "number" ? p.resolution_m + " m/pixel" : "not stated");
    provRow(dl, "chain", p.chain_label || ("position " + p.chain_position));
    provRow(dl, "scaling", p.scaling_applied);
    provRow(dl, "age", typeof p.age_days === "number" ? p.age_days + " days" : null);
    provRow(dl, "stale after", typeof p.stale_after_days === "number" ? p.stale_after_days + " days" : null);
    provRow(dl, "derived from", (p.derived_from && p.derived_from.length) ? p.derived_from.join(", ") : null);

    var card = el("span", { class: "prov", role: "tooltip" }, [
      el("span", { class: "prov__h" }, [
        el("span", { text: p.quantity || "measurement" }),
        el("span", { class: "muted", text: "measured" })
      ]),
      dl
    ]);
    if (p.is_stale === true || p.staleness_note) {
      card.appendChild(el("span", {
        class: "prov__flag",
        text: p.staleness_note || "This reading is past its staleness limit."
      }));
    }
    if (p.note) card.appendChild(el("span", { class: "prov__note", text: p.note }));
    return card;
  }

  function missingCard(p) {
    var dl = el("dl");
    provRow(dl, "reason", p.reason);
    provRow(dl, "detail", p.detail);
    provRow(dl, "sources tried", (p.chain_tried && p.chain_tried.length) ? p.chain_tried.join(" → ") : null);
    provRow(dl, "last seen", p.last_observed_on);
    provRow(dl, "age", typeof p.age_days === "number" ? p.age_days + " days" : null);
    provRow(dl, "asked at", p.requested_at);

    return el("span", { class: "prov", role: "tooltip" }, [
      el("span", { class: "prov__h" }, [
        el("span", { text: p.quantity || "value" }),
        el("span", { class: "muted", text: "not measured" })
      ]),
      el("span", { class: "prov__flag", text: p.reason_text || "This value was not measured." }),
      dl
    ]);
  }

  function diagnosticCard(p) {
    var dl = el("dl");
    Object.keys(p).forEach(function (k) {
      if (k === "quantity" || k === "note") return;
      provRow(dl, k.replace(/_/g, " "), p[k]);
    });
    return el("span", { class: "prov", role: "tooltip" }, [
      el("span", { class: "prov__h" }, [
        el("span", { text: p.quantity || "diagnostic" }),
        el("span", { class: "muted", text: "about the system, not the field" })
      ]),
      p.note ? el("span", { class: "prov__note", text: p.note }) : null,
      dl
    ]);
  }

  function withheldCard(seg) {
    return el("span", { class: "prov", role: "tooltip" }, [
      el("span", { class: "prov__h" }, [el("span", { text: seg.slot || "value" })]),
      el("span", {
        class: "prov__flag prov__flag--bad",
        text: "The server sent this as a measurement but attached no provenance, so CropUp will not show it as a number. " +
              "A value with no instrument, date and resolution behind it is not a measurement."
      })
    ]);
  }

  function hoverable(cls, label, card, aria) {
    var node = el("span", {
      class: "seg " + cls, tabindex: "0", role: "button",
      "aria-expanded": "false", "aria-label": aria
    }, [label]);
    node.appendChild(card);
    node.addEventListener("click", function (ev) {
      if (ev.target !== node && node.contains(ev.target) && ev.target.closest(".prov")) return;
      ev.stopPropagation();
      var open = node.classList.toggle("is-open");
      node.setAttribute("aria-expanded", open ? "true" : "false");
      closeOthers(node);
    });
    node.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        var open = node.classList.toggle("is-open");
        node.setAttribute("aria-expanded", open ? "true" : "false");
        closeOthers(node);
      } else if (ev.key === "Escape") {
        node.classList.remove("is-open");
        node.setAttribute("aria-expanded", "false");
      }
    });
    return node;
  }

  function closeOthers(keep) {
    Array.prototype.forEach.call(document.querySelectorAll(".seg.is-open"), function (n) {
      if (n !== keep) { n.classList.remove("is-open"); n.setAttribute("aria-expanded", "false"); }
    });
  }

  function renderSegment(seg) {
    var p = seg.provenance;

    if (seg.kind === "fact") {
      // SPEC 9 / SPEC 4: a number with no Fact behind it is not rendered as a number.
      if (!p || p.measured !== true || !p.source_asset) {
        return hoverable("seg--withheld", "[withheld: no provenance]", withheldCard(seg),
          "Value withheld because it arrived without provenance");
      }
      var stale = p.is_stale === true || !!p.staleness_note;
      var node = hoverable("seg--fact" + (stale ? " is-stale" : ""), seg.text, factCard(p),
        (p.quantity || "measurement") + " " + seg.text +
        ", measured by " + p.source_asset +
        (p.observed_on ? " on " + p.observed_on : "") +
        (typeof p.resolution_m === "number" ? " at " + p.resolution_m + " m resolution" : "") +
        (stale ? ". This reading is stale." : ""));
      return node;
    }

    if (seg.kind === "missing") {
      return hoverable("seg--missing", seg.text, missingCard(p || { quantity: seg.slot }),
        (p && p.quantity ? p.quantity : seg.slot || "this value") + " was not measured: " +
        ((p && p.reason_text) || "no reason given"));
    }

    if (seg.kind === "quote") {
      var q = el("blockquote", { class: "quote" }, [document.createTextNode(seg.text)]);
      if (seg.citation) q.appendChild(el("cite", { text: seg.citation }));
      return q;
    }

    if (seg.kind === "diagnostic") {
      return hoverable("seg--diag", seg.text, diagnosticCard(p || { quantity: "diagnostic" }),
        "system diagnostic: " + seg.text);
    }

    return document.createTextNode(seg.text || "");
  }

  function renderLine(line) {
    var segs = line.segments || [];
    // a line of quotes is block-level; a mixed line is a paragraph
    var onlyQuotes = segs.length > 0 && segs.every(function (s) { return s.kind === "quote"; });
    var host = el("div", { class: "line" });
    if (!segs.length) {
      host.appendChild(document.createTextNode(line.text || ""));
      return host;
    }
    segs.forEach(function (seg, i) {
      var node = renderSegment(seg);
      host.appendChild(node);
      if (!onlyQuotes && i < segs.length - 1 && seg.kind !== "quote") {
        // the templates already carry their own spacing inside "text" segments;
        // this only keeps adjacent inline chips from touching.
        var next = segs[i + 1];
        if (next && next.kind !== "quote" && !/\s$/.test(seg.text || "") && !/^\s/.test(next.text || "")) {
          host.appendChild(document.createTextNode(" "));
        }
      }
    });
    return host;
  }

  function tagRow(label, items, cls) {
    if (!items || !items.length) return null;
    var row = el("div", { class: "tagrow" }, [el("span", { class: "tagrow__label", text: label })]);
    items.forEach(function (t) { row.appendChild(el("span", { class: "tag " + (cls || ""), text: t })); });
    return row;
  }

  function renderAnswer(answer) {
    if (!answer) return el("div", { class: "answer", text: "The server returned no answer object." });
    var art = el("article", { class: "answer", "data-kind": answer.kind || "" });
    art.appendChild(el("span", { class: "answer__kind", text: titleise(answer.kind) }));
    if (answer.title) art.appendChild(el("h3", { class: "answer__title", text: answer.title }));

    (answer.sections || []).forEach(function (sec) {
      var s = el("section", { class: "sect" });
      if (sec.heading) s.appendChild(el("div", { class: "sect__h", text: sec.heading }));
      (sec.lines || []).forEach(function (line) { s.appendChild(renderLine(line)); });
      if (sec.not_measured && sec.not_measured.length) {
        s.appendChild(el("div", {
          class: "notmeasured",
          text: "Wanted for this section but never attempted: " +
                sec.not_measured.map(function (q) { return q.replace(/_/g, " "); }).join(", ") + "."
        }));
      }
      art.appendChild(s);
    });

    var foot = el("div", { class: "answer__foot" });
    var any = false;
    var f1 = tagRow("measured", answer.facts_used, "tag--fact"); if (f1) { foot.appendChild(f1); any = true; }
    var f2 = tagRow("named as missing", answer.gaps_named, "tag--gap"); if (f2) { foot.appendChild(f2); any = true; }
    var f3 = tagRow("never attempted", answer.not_measured, ""); if (f3) { foot.appendChild(f3); any = true; }
    if (answer.citations && answer.citations.length) {
      any = true;
      foot.appendChild(el("div", {}, [
        el("span", { class: "tagrow__label", text: "sources" }),
        el("ol", { class: "cites" }, answer.citations.map(function (c) { return el("li", { text: c }); }))
      ]));
    }
    if (answer.degradation && answer.degradation.summary) {
      any = true;
      foot.appendChild(el("div", { class: "small muted", text: answer.degradation.summary }));
    }
    if (any) art.appendChild(foot);
    return art;
  }

  /* ================================================================
   * transcript
   * ============================================================= */

  var EXAMPLES = [
    "how is my maize field in Arusha doing this week",
    "how much fertilizer should I use for maize",
    "which crop should I plant in Moshi",
    "my beans have yellow leaves, what is wrong"
  ];

  function transcriptEmpty() {
    var box = el("div", { class: "empty" }, [
      el("h3", { text: "Ask anything about your field" }),
      el("p", { text: "Most questions are answered from the cited knowledge base and never touch a satellite. Questions about a specific field do, and those need a confirmed location and crop first." })
    ]);
    var row = el("div");
    EXAMPLES.forEach(function (q) {
      row.appendChild(el("button", {
        class: "example", type: "button", text: q,
        onclick: function () { $("composer-input").value = q; $("composer-input").focus(); }
      }));
    });
    box.appendChild(row);
    return box;
  }

  function pushTurn(node) {
    var t = $("transcript");
    var empty = t.querySelector(".empty");
    if (empty) empty.remove();
    t.appendChild(node);
    t.scrollTop = t.scrollHeight;
    node.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  function pushUser(text) {
    pushTurn(el("div", { class: "turn turn--user" }, [
      el("span", { class: "turn__label", text: "you · " + shortTime(new Date().toISOString()) }),
      el("div", { class: "bubble", text: text })
    ]));
  }

  function pushAnswer(env, note) {
    var wrap = el("div", { class: "turn" }, [
      el("span", { class: "turn__label", text: "cropup · " + shortTime(new Date().toISOString()) }),
      renderAnswer(env.answer)
    ]);
    if (note) wrap.appendChild(el("div", { class: "notmeasured", text: note }));
    pushTurn(wrap);
  }

  /* ================================================================
   * the slot frame (shared by both views)
   * ============================================================= */

  var SLOT_ORDER = ["intent", "crop", "location", "timeframe"];

  function slotState(sv) {
    if (!sv) return "empty";
    if (sv.needs_confirmation && !sv.confirmed) return "contested";
    if (sv.confirmed) return "confirmed";
    if (sv.locked) return "locked";
    return "suggested";
  }

  function renderSlots() {
    var host = $("slots");
    clear(host);
    var bag = S.bag || { slots: {} };

    SLOT_ORDER.forEach(function (name) {
      var sv = (bag.slots || {})[name] || null;
      var st = slotState(sv);
      var li = el("li", { class: "slot slot--" + st });

      var top = el("div", { class: "slot__top" }, [
        el("span", { class: "slot__name", text: name }),
        sv
          ? el("span", { class: "slot__value", text: sv.label || sv.value })
          : el("span", { class: "slot__value slot__value--empty", text: "not set" })
      ]);
      var badges = el("span", { class: "slot__badges" });
      if (st === "suggested") badges.appendChild(el("span", { class: "badge badge--suggested", text: "suggested" }));
      if (st === "locked") badges.appendChild(el("span", { class: "badge badge--locked", text: "you set this" }));
      if (st === "confirmed") badges.appendChild(el("span", { class: "badge badge--confirmed", text: "confirmed" }));
      if (sv && sv.needs_confirmation) badges.appendChild(el("span", { class: "badge badge--needs", text: "needs confirming" }));
      // SPEC 5.3: seven short gazetteer names are flagged ambiguous. Say so.
      if (sv && sv.detail && sv.detail.ambiguous) badges.appendChild(el("span", { class: "badge badge--needs", text: "ambiguous name" }));
      top.appendChild(badges);
      li.appendChild(top);

      if (sv) {
        var bits = [];
        bits.push("from " + (sv.origin || "?"));
        if (sv.source) bits.push(sv.source);
        bits.push(typeof sv.confidence === "number"
          ? "confidence " + Math.round(sv.confidence * 100) + "%"
          : "confidence not scored");
        if (sv.detail && typeof sv.detail.lat === "number") {
          bits.push(num(sv.detail.lat, 4) + ", " + num(sv.detail.lon, 4));
        }
        li.appendChild(el("div", { class: "slot__meta", text: bits.join(" · ") }));
        if (sv.note) li.appendChild(el("div", { class: "small muted", text: sv.note }));

        if (sv.alternatives && sv.alternatives.length) {
          var alts = el("div", { class: "slot__alts" }, [el("span", { class: "muted small", text: "also matches:" })]);
          sv.alternatives.forEach(function (a) { alts.appendChild(el("span", { class: "tag", text: a })); });
          li.appendChild(alts);
        }

        var acts = el("div", { class: "slot__actions" });
        if (st === "suggested" || st === "contested") {
          acts.appendChild(el("button", {
            class: "btn btn--sm", type: "button", text: "Yes, that's right",
            onclick: function () { lockSlot(name, sv); }
          }));
        }
        if (sv.locked && !sv.confirmed) {
          acts.appendChild(el("button", {
            class: "btn btn--ghost btn--sm", type: "button", text: "Let CropUp guess again",
            onclick: function () { writeSlots({ unlock: [name] }, "unlocking " + name); }
          }));
        }
        if (sv.confirmed) {
          acts.appendChild(el("button", {
            class: "btn btn--ghost btn--sm", type: "button", text: "Withdraw confirmation",
            onclick: function () { postConfirm({ unconfirm: [name] }, "withdrawing " + name); }
          }));
        }
        if (name !== "intent") {
          acts.appendChild(el("button", {
            class: "btn btn--ghost btn--sm", type: "button", text: "Clear",
            onclick: function () { writeSlots({ clear: [name] }, "clearing " + name); }
          }));
        }
        li.appendChild(acts);
      } else {
        li.appendChild(el("div", {
          class: "small muted",
          text: name === "intent" ? "CropUp will route your question itself, or pick one in the Form."
              : name === "timeframe" ? "optional"
              : "needed before any satellite runs"
        }));
      }

      host.appendChild(li);
    });

    renderSlotEvents();
  }

  function renderSlotEvents() {
    var box = $("slot-events");
    var bag = S.bag;
    if (!bag || !bag.events) { box.hidden = true; return; }
    var refused = bag.events.filter(function (e) { return e.outcome === "refused_locked"; }).slice(-3);
    if (!refused.length) { box.hidden = true; return; }
    clear(box);
    refused.forEach(function (e) {
      var kept = (bag.slots && bag.slots[e.slot]) ? (bag.slots[e.slot].label || bag.slots[e.slot].value) : "your value";
      box.appendChild(el("div", {
        text: "You set " + e.slot + " = " + kept + ", so CropUp ignored the “" + e.value + "” it heard."
      }));
    });
    box.hidden = false;
  }

  function lockSlot(name, sv) {
    if (name === "intent") { writeSlots({ intent: sv.value }, "locking intent"); return; }
    if (name === "location") {
      var key = sv.detail && sv.detail.key;
      if (key) { writeSlots({ place_key: key }, "locking location"); return; }
      if (sv.detail && typeof sv.detail.lat === "number") {
        writeSlots({ point: { lat: sv.detail.lat, lon: sv.detail.lon, label: sv.label || sv.value } }, "locking location");
        return;
      }
    }
    if (name === "crop") {
      var ckey = sv.detail && sv.detail.key;
      if (ckey) { writeSlots({ crop_key: ckey }, "locking crop"); return; }
    }
    var slots = {}; slots[name] = sv.value;
    writeSlots({ slots: slots }, "locking " + name);
  }

  /* ================================================================
   * capability strip
   * ============================================================= */

  function refreshCapabilities() {
    // The coverage cliffs of SPEC 3.3 are only answerable at a point, so ask at
    // the field as soon as there is one. This endpoint never calls Earth Engine.
    var q = "";
    if (S.field && S.field.resolved && S.field.request) {
      q = "?lat=" + encodeURIComponent(S.field.request.lat) + "&lon=" + encodeURIComponent(S.field.request.lon);
    } else if (S.sid) {
      q = "?sid=" + encodeURIComponent(S.sid);
    }
    return api("/api/capabilities" + q).then(function (caps) {
      S.caps = caps;
      renderCapabilities();
      return caps;
    }).catch(function (err) {
      // The honesty endpoint itself is down: say so rather than showing green.
      S.caps = null;
      $("capstrip-summary").textContent = "cannot reach /api/capabilities (" + (err.status || "network") + ") — status unknown";
      $("capstrip-dot").className = "capstrip__dot is-broken";
      clear($("capstrip-chips"));
    });
  }

  function renderCapabilities() {
    var caps = S.caps;
    if (!caps) return;
    $("capstrip-summary").textContent = caps.summary || "";
    var counts = caps.counts || {};
    var dot = $("capstrip-dot");
    dot.className = "capstrip__dot" + (counts.unavailable ? " is-broken" : caps.degraded ? " is-degraded" : "");

    var chips = $("capstrip-chips");
    clear(chips);
    var can = caps.can || {};
    Object.keys(can).forEach(function (k) {
      var entry = can[k] || {};
      chips.appendChild(el("span", {
        class: "capchip" + (entry.ok ? "" : " is-down"),
        title: entry.detail || "",
        tabindex: "0"
      }, [
        el("span", { class: "capchip__x", text: entry.ok ? "✓" : "✕" }),
        " " + titleise(k)
      ]));
    });

    var lost = $("cap-lost");
    clear(lost);
    if (caps.lost && caps.lost.length) {
      caps.lost.forEach(function (l) { lost.appendChild(el("li", { text: l })); });
    } else {
      lost.appendChild(el("li", { class: "muted", text: "nothing — every capability is reporting ready" }));
    }

    var unknown = $("cap-unknown");
    clear(unknown);
    (caps.unknown || []).forEach(function (u) { unknown.appendChild(el("li", { text: u })); });
    if (!(caps.unknown || []).length) unknown.appendChild(el("li", { class: "muted", text: "none" }));

    var all = $("cap-all");
    clear(all);
    (caps.capabilities || []).forEach(function (c) {
      all.appendChild(el("li", {}, [
        el("span", { class: "statusdot statusdot--" + c.status, text: c.status }),
        el("span", { class: "cap-name", text: c.name }),
        el("span", { class: "cap-detail", text: c.detail || "" })
      ]));
    });

    $("cap-stamp").textContent =
      "checked " + (caps.generated_at || "") +
      (caps.point ? " at " + num(caps.point.lat, 3) + ", " + num(caps.point.lon, 3) + " — " + caps.point.detail : " (no field point given yet)") +
      " · cropup " + (caps.version || "");
  }

  /* ================================================================
   * the field panel
   * ============================================================= */

  function renderField() {
    var report = S.field;
    var rows = $("request-rows");
    clear(rows);

    if (!report || !report.resolved) {
      rows.appendChild(el("p", {
        class: "unresolved",
        text: (report && report.reason) || "No location has been given yet, so there is nothing to send."
      }));
      $("footprints-block").hidden = true;
      if (S.map) S.map.render(report || { resolved: false }, LEG_COLOURS);
      return;
    }

    var req = report.request || {};
    var dl = el("dl", { class: "reqrows" });
    function row(k, v) { dl.appendChild(el("dt", { text: k })); dl.appendChild(el("dd", { text: v })); }
    row("expression", req.expression || "");
    row("lat, lon", num(req.lat, 5) + ", " + num(req.lon, 5));
    row("radius", metres(req.radius_m));
    row("why that radius", req.radius_source || "");
    row("crs", req.crs || "");
    if (report.field && report.field.properties) {
      row("drawn as", report.field.properties.vertices + "-vertex ring" +
        (report.field.properties.approximate ? " (display only — EE buffers the point itself)" : ""));
    }
    rows.appendChild(dl);

    var fps = report.footprints || [];
    $("footprints-block").hidden = false;
    $("footprints-count").textContent = "(" + fps.length + " — some are far wider than your plot)";
    var list = $("footprints");
    clear(list);
    fps.forEach(function (f, i) {
      var colour = legColour(f.leg, i);
      list.appendChild(el("li", {}, [
        el("span", { class: "swatch", style: "background:" + colour }),
        el("span", {}, [
          el("span", { class: "f-name", text: f.leg }),
          " ",
          el("span", { class: "f-r", text: typeof f.radius_m === "number" ? metres(f.radius_m) + " radius" : "exact pixel" }),
          typeof f.pixel_m === "number" ? el("span", { class: "f-r", text: " · " + Math.round(f.pixel_m) + " m pixels" }) : null,
          el("div", { class: "f-why", text: f.reason || "" }),
          el("div", { class: "small muted", text: (f.assets || []).join(", ") })
        ])
      ]));
    });

    if (S.map) S.map.render(report, LEG_COLOURS);
  }

  /* ================================================================
   * the gate (SPEC 4.4)
   * ============================================================= */

  function renderGate() {
    var host = $("gate-body");
    var gate = $("gate");
    clear(host);
    gate.className = "gate";

    var action = S.action || {};
    var awaiting = S.awaiting || {};
    var field = S.field;
    var kind = action.kind || "";

    var route = S.intentMeta ? S.intentMeta.route : null;
    if (route === "rag") {
      gate.classList.add("is-open");
      host.appendChild(el("p", {
        class: "gate__why",
        text: (S.intentMeta.label || S.intentMeta.name) + " is answered from the cited knowledge base. " +
              "No satellite is contacted, so there is nothing to confirm and nothing to run."
      }));
      host.appendChild(el("p", {
        class: "small muted",
        text: "Pick a satellite question in the Form — field health, crop problem, irrigation or crop selection — if you want this field measured."
      }));
      return;
    }

    if (!field || !field.resolved) {
      gate.classList.add("is-shut");
      host.appendChild(el("p", {
        class: "gate__why",
        text: (field && field.reason) || "CropUp needs a location it can put on the map before it can measure anything."
      }));
      host.appendChild(el("p", { class: "small muted", text: "Pick a place in the Form, drop a pin on the map, or use your phone's GPS." }));
      return;
    }

    var unconfirmed = awaiting.unconfirmed_slots || action.unconfirmed_slots || [];
    var missing = awaiting.missing_slots || action.missing_slots || [];

    if (missing.length) {
      gate.classList.add("is-shut");
      host.appendChild(el("p", { class: "gate__why", text: "Still needed: " + missing.join(", ") + "." }));
    }

    if (unconfirmed.length) {
      gate.classList.add("is-shut");
      host.appendChild(el("p", {
        class: "gate__why",
        text: "Earth Engine never runs on a field you have not confirmed. Check these, then confirm."
      }));
      var ul = el("ul", { class: "gate__list" });
      unconfirmed.forEach(function (name) {
        var sv = (S.bag && S.bag.slots) ? S.bag.slots[name] : null;
        ul.appendChild(el("li", { text: name + ": " + (sv ? (sv.label || sv.value) : "not set") }));
      });
      if (field.request) {
        ul.appendChild(el("li", {
          text: "geometry: " + num(field.request.lat, 5) + ", " + num(field.request.lon, 5) +
                " buffered " + metres(field.request.radius_m)
        }));
      }
      host.appendChild(ul);
      host.appendChild(el("button", {
        class: "btn btn--primary btn--wide", type: "button",
        text: "Confirm this field",
        onclick: function () { postConfirm({ slots: unconfirmed }, "confirming the field"); }
      }));
    } else if (field.confirmed) {
      gate.classList.add("is-open");
      host.appendChild(el("p", { class: "gate__why", text: "Confirmed. " + (action.prompt || "") }));
      host.appendChild(el("div", { class: "gate__row" }, [
        el("button", {
          class: "btn btn--ghost btn--sm", type: "button", text: "Withdraw confirmation",
          onclick: function () { postConfirm({ unconfirm: ["location", "crop"] }, "withdrawing confirmation"); }
        }),
        el("button", {
          class: "btn btn--ghost btn--sm", type: "button", text: "Change the field",
          onclick: function () { switchTab("form"); $("form-place").focus(); }
        })
      ]));
    }

    if (kind === "clarify" && !awaiting.may_run && field.confirmed) {
      gate.className = "gate is-down";
      host.appendChild(el("p", { class: "gate__why", text: action.prompt || "CropUp cannot measure this field right now." }));
      if (awaiting.options && awaiting.options.length) {
        var opts = el("div", { class: "gate__row" }, [el("span", { class: "muted small", text: "what is still open:" })]);
        awaiting.options.forEach(function (o) { opts.appendChild(el("span", { class: "tag", text: o.replace(/_/g, " ") })); });
        host.appendChild(opts);
      }
      host.appendChild(el("p", {
        class: "small muted",
        text: "Your confirmation was still recorded, so the moment Earth Engine answers again this field is ready to run."
      }));
    }

    // the Run button: offered only when the policy already said yes
    var mayRun = !!awaiting.may_run;
    var runBtn = el("button", {
      class: "btn btn--primary btn--wide", type: "button",
      text: S.run.active ? "Running…" : "Run the analysis on this field",
      disabled: !mayRun || S.run.active || S.busy,
      onclick: doRun
    });
    host.appendChild(runBtn);
    if (!mayRun) {
      host.appendChild(el("p", {
        class: "small muted",
        text: "Run stays off until the policy opens the gate. " + (action.reason || "")
      }));
    }

    if (action.checks && action.checks.length) {
      var det = el("details", { class: "gate__checks" }, [
        el("summary", { text: "the gate's own trace" }),
        el("ul", {}, action.checks.map(function (c) { return el("li", { text: c }); }))
      ]);
      host.appendChild(det);
    }
  }

  /* ================================================================
   * run + SSE
   * ============================================================= */

  function connectEvents() {
    if (S.events) { S.events.close(); S.events = null; }
    if (!S.sid || typeof EventSource === "undefined") return;
    var src = new EventSource("/api/session/" + encodeURIComponent(S.sid) + "/events");
    S.events = src;

    src.addEventListener("run_planned", function (ev) {
      var d = JSON.parse(ev.data);
      S.run.active = true;
      S.run.planned = (d.legs || []).map(function (l) { return { name: l.name, status: "planned", assets: l.assets || [] }; });
      S.run.byName = {};
      S.run.budget = d.budget_s;
      $("runpanel").hidden = false;
      $("run-note").textContent = (d.note || "") + (d.budget_s ? " Budget " + d.budget_s + " s." : "");
      $("run-clock").textContent = d.analysis || "";
      renderLegs();
      renderGate();
    });

    src.addEventListener("progress", function (ev) {
      var d = JSON.parse(ev.data);
      $("run-clock").textContent = (d.analysis || "") + " · " + num(d.elapsed_s, 1) + " s of " + num(d.budget_s, 1) + " s";
      if (d.note) $("run-note").textContent = d.note;
    });

    src.addEventListener("leg", function (ev) {
      var d = JSON.parse(ev.data);
      S.run.byName[d.name] = d;
      renderLegs();
    });

    src.addEventListener("run_finished", function (ev) {
      var d = JSON.parse(ev.data);
      S.run.active = false;
      $("run-clock").textContent = (d.analysis || "") + " · finished in " + num(d.elapsed_s, 1) + " s";
      $("run-note").textContent = d.legs_ok + " of " + d.legs_total + " legs returned, " +
        d.facts + " measured, " + d.gaps + " named as missing" + (d.degraded ? " (degraded)" : "") + ".";
      renderLegs();
      renderGate();
    });

    src.addEventListener("run_failed", function (ev) {
      var d = JSON.parse(ev.data);
      S.run.active = false;
      $("run-clock").textContent = "failed after " + num(d.elapsed_s, 1) + " s";
      $("run-note").textContent = d.error + ": " + (d.detail || "");
      renderLegs();
      renderGate();
    });

    src.addEventListener("slots", function () { /* the envelope that caused it already refreshed the bag */ });

    src.onerror = function () {
      // EventSource reconnects on its own with Last-Event-ID; the backlog keeps
      // the last 200 events, so a reconnect loses nothing. Nothing to do.
    };
  }

  function renderLegs() {
    var list = $("legs");
    clear(list);
    var names = S.run.planned.map(function (l) { return l.name; });
    Object.keys(S.run.byName).forEach(function (n) { if (names.indexOf(n) === -1) names.push(n); });
    if (!names.length) {
      list.appendChild(el("li", { class: "is-planned" }, [el("span", { class: "l-detail", text: "waiting for the plan…" })]));
      return;
    }
    names.forEach(function (name) {
      var done = S.run.byName[name];
      var cls = "is-planned", mark = "·";
      if (done) {
        if (done.ok && !done.gaps) { cls = "is-ok"; mark = "✓"; }
        else if (done.ok || done.facts) { cls = "is-partial"; mark = "◑"; }
        else { cls = "is-failed"; mark = "✕"; }
      }
      var detail = done
        ? num(done.elapsed_s, 1) + " s · " + done.facts + " measured, " + done.gaps + " missing" +
          (done.detail ? " · " + done.detail : "")
        : "planned";
      list.appendChild(el("li", { class: cls }, [
        el("span", { class: "l-state", text: mark }),
        el("span", { class: "l-name", text: name }),
        el("span", { class: "l-detail", text: detail })
      ]));
    });
  }

  function doRun() {
    if (S.busy || S.run.active) return;
    S.busy = true;
    S.run.active = true;
    S.run.byName = {};
    $("runpanel").hidden = false;
    $("run-note").textContent = "asking the policy for the same verdict a chat turn gets…";
    renderLegs();
    renderGate();
    var body = (S.bag && S.bag.intent) ? { intent: S.bag.intent } : {};
    api("/api/session/" + encodeURIComponent(S.sid) + "/run", "POST", body)
      .then(function (env) {
        applyEnvelope(env);
        pushAnswer(env, env.ran_earth_engine ? null : "This answer did not reach Earth Engine.");
        switchTab("ask");
      })
      .catch(function (err) {
        showError(err, "running the analysis");
        var envErr = err.envelope && err.envelope.error;
        $("run-note").textContent = (envErr ? envErr.type + ": " + envErr.message : String(err.message));
        refreshCapabilities();
      })
      .then(function () {
        S.busy = false;
        S.run.active = false;
        renderGate();
      });
  }

  /* ================================================================
   * envelope plumbing
   * ============================================================= */

  function applyEnvelope(env) {
    if (!env) return;
    if (env.session_id) S.sid = env.session_id;
    if (env.bag) S.bag = env.bag;
    if (env.field) S.field = env.field;
    if (env.action) S.action = env.action;
    S.awaiting = env.awaiting || (env.action ? {
      kind: env.action.kind,
      slot: env.action.slot,
      missing_slots: env.action.missing_slots || [],
      unconfirmed_slots: env.action.unconfirmed_slots || [],
      options: env.action.options || [],
      may_run: !!env.action.authorises_earth_engine
    } : null);
    if (env.intent !== undefined) S.intentMeta = env.intent;
    if (env.degraded && env.degraded_reason) {
      toast({ title: "Degraded turn", body: env.degraded_reason, kind: "info" });
    }
    saveSnapshot();
    renderAll();
  }

  function renderAll() {
    renderSlots();
    renderField();
    renderGate();
    syncFormFromBag();
    $("session-chip").textContent = S.sid ? "session " + S.sid.slice(0, 8) : "no session";
    $("session-chip").title = S.sid ? "Session " + S.sid : "no session";
  }

  function describeOutcomes(outcomes) {
    if (!outcomes) return null;
    var notes = [];
    Object.keys(outcomes).forEach(function (slot) {
      if (outcomes[slot] === "refused_locked") {
        notes.push("kept your " + slot + " and ignored what the classifier heard");
      } else if (outcomes[slot] === "cleared") {
        notes.push("cleared " + slot);
      }
    });
    return notes.length ? "CropUp " + notes.join("; ") + "." : null;
  }

  /* ================================================================
   * actions
   * ============================================================= */

  function sendMessage(text) {
    if (!text.trim() || !S.sid || S.busy) return;
    S.busy = true;
    pushUser(text);
    $("composer-input").value = "";
    api("/api/session/" + encodeURIComponent(S.sid) + "/message", "POST", { text: text })
      .then(function (env) {
        applyEnvelope(env);
        pushAnswer(env, describeOutcomes(env.slot_outcomes));
        if (env.parse) pushParse(env.parse);
      })
      .catch(function (err) {
        if (err.status === 404) return reopenSession();
        showError(err, "sending your message");
      })
      .then(function () { S.busy = false; renderGate(); });
  }

  function pushParse(parse) {
    if (!parse || !parse.intent) return;
    var i = parse.intent;
    var bits = [
      "routed to " + i.intent + " (" + i.route + ")",
      "tier " + i.tier,
      i.score_kind + " " + num(i.score, 3),
      typeof i.confidence === "number" ? "confidence " + num(i.confidence, 3) : "confidence not scored at this tier"
    ];
    var det = el("details", { class: "gate__checks" }, [
      el("summary", { text: "how CropUp read that" }),
      el("p", { class: "small", text: bits.join(" · ") }),
      el("p", { class: "small muted", text: i.reason || "" })
    ]);
    if (i.below_floor) {
      det.appendChild(el("p", { class: "small", text: "Below the confidence floor, so CropUp asked instead of guessing." }));
    }
    var last = $("transcript").lastElementChild;
    if (last) last.appendChild(det);
  }

  function writeSlots(body, what) {
    if (!S.sid || S.busy) return Promise.resolve();
    S.busy = true;
    return api("/api/session/" + encodeURIComponent(S.sid) + "/slots", "POST", body)
      .then(function (env) {
        applyEnvelope(env);
        var note = describeOutcomes(env.slot_outcomes);
        if (note) toast({ title: "Slot frame", body: note, kind: "info" });
        return refreshCapabilities();
      })
      .catch(function (err) {
        if (err.status === 404) return reopenSession();
        showError(err, what);
      })
      .then(function () { S.busy = false; renderGate(); });
  }

  function postConfirm(body, what) {
    if (!S.sid || S.busy) return Promise.resolve();
    S.busy = true;
    return api("/api/session/" + encodeURIComponent(S.sid) + "/confirm", "POST", body)
      .then(function (env) {
        applyEnvelope(env);
        if (env.confirmed && env.confirmed.length) {
          toast({ title: "Gate", body: "Confirmed: " + env.confirmed.join(", ") + ".", kind: "info" });
        }
        // Do not take "unconfirmed" on trust: read the bag that came back. As
        // of this server build, POST /confirm with `unconfirm` withdraws the
        // confirmation and then re-confirms the same slots in the same request,
        // so the reported withdrawal does not survive. Saying "withdrawn" over
        // a bag that still says confirmed would be exactly the silent
        // fabrication this app refuses to do anywhere else.
        if (body && body.unconfirm && body.unconfirm.length) {
          var stuck = body.unconfirm.filter(function (name) {
            var sv = env.bag && env.bag.slots && env.bag.slots[name];
            return !!(sv && sv.confirmed);
          });
          if (stuck.length) {
            toast({
              title: "Confirmation NOT withdrawn",
              body: "The server reported " + stuck.join(", ") + " as withdrawn, but sent back a bag in which " +
                    (stuck.length > 1 ? "they are" : "it is") + " still confirmed. CropUp will not pretend otherwise.",
              remedy: "Change the value, or press Clear on that slot — editing a slot does drop its confirmation.",
              kind: "error"
            });
          } else {
            toast({ title: "Gate", body: "Withdrawn: " + body.unconfirm.join(", ") + ".", kind: "info" });
          }
        }
        if (env.action && env.action.kind === "clarify") pushAnswer(env, null);
        return refreshCapabilities();
      })
      .catch(function (err) {
        if (err.status === 404) return reopenSession();
        showError(err, what);
      })
      .then(function () { S.busy = false; renderGate(); });
  }

  /* ================================================================
   * the Form view
   * ============================================================= */

  var searchTimers = {};

  function wireSearch(inputId, listId, endpoint, renderRow) {
    var input = $(inputId), list = $(listId);
    function close() { list.hidden = true; input.setAttribute("aria-expanded", "false"); clear(list); }
    input.addEventListener("input", function () {
      var q = input.value.trim();
      clearTimeout(searchTimers[inputId]);
      if (!q) { close(); return; }
      searchTimers[inputId] = setTimeout(function () {
        api(endpoint + "?q=" + encodeURIComponent(q) + "&limit=8")
          .then(function (res) {
            clear(list);
            if (!res.results || !res.results.length) {
              list.appendChild(el("li", {}, [el("div", { class: "r-empty", text: "Nothing in the committed vocabulary matches “" + q + "”." })]));
            } else {
              res.results.forEach(function (r) { list.appendChild(renderRow(r, close)); });
            }
            list.hidden = false;
            input.setAttribute("aria-expanded", "true");
          })
          .catch(function (err) { showError(err, "searching " + endpoint); });
      }, 180);
    });
    input.addEventListener("keydown", function (ev) { if (ev.key === "Escape") close(); });
    document.addEventListener("click", function (ev) {
      if (!list.contains(ev.target) && ev.target !== input) close();
    });
    return close;
  }

  function buildForm() {
    var sel = $("form-intent");
    clear(sel);
    sel.appendChild(el("option", { value: "", text: "— let CropUp decide from what I type —" }));
    INTENTS.forEach(function (i) {
      sel.appendChild(el("option", { value: i.name, text: i.label + "  (" + i.route.replace("_", " ") + ")" }));
    });
    sel.addEventListener("change", function () {
      var v = sel.value;
      $("form-intent-hint").textContent = v ? ROUTE_TEXT[intentRoute(v)] : "CropUp routes each message on its own.";
      if (v) writeSlots({ intent: v }, "setting the intent");
    });

    wireSearch("form-crop", "form-crop-results", "/api/vocab/crops", function (r, close) {
      return el("li", {}, [el("button", {
        type: "button",
        onclick: function () { close(); $("form-crop").value = r.name; writeSlots({ crop_key: r.key }, "setting the crop"); }
      }, [
        el("span", { class: "r-name", text: r.name }),
        el("span", { class: "r-why", text: r.matched_alias && r.matched_alias !== r.name.toLowerCase() ? r.matched_alias + " → " + r.name : r.key }),
        r.in_suitability ? null : el("span", { class: "r-flag", text: "no suitability data" })
      ])]);
    });

    wireSearch("form-place", "form-place-results", "/api/vocab/places", function (r, close) {
      return el("li", {}, [el("button", {
        type: "button",
        onclick: function () { close(); $("form-place").value = r.label || r.name; writeSlots({ place_key: r.key }, "setting the location"); }
      }, [
        el("span", { class: "r-name", text: r.label || r.name }),
        el("span", { class: "r-why", text: num(r.lat, 2) + ", " + num(r.lon, 2) + " · " + (r.support_band || "") }),
        r.ambiguous ? el("span", { class: "r-flag", text: "several real places share this name" }) : null
      ])]);
    });

    $("form-timeframe").addEventListener("keydown", function (ev) {
      if (ev.key !== "Enter") return;
      ev.preventDefault();
      writeSlots({ slots: { timeframe: $("form-timeframe").value.trim() } }, "setting the timeframe");
    });

    var radius = $("form-radius");
    radius.addEventListener("input", function () { $("form-radius-out").textContent = radius.value + " m"; });
    $("btn-radius-apply").addEventListener("click", function () {
      writeSlots({ field_radius_m: Number(radius.value) }, "setting the field radius");
    });

    Array.prototype.forEach.call(document.querySelectorAll("[data-clear-slot]"), function (btn) {
      btn.addEventListener("click", function () {
        var slot = btn.getAttribute("data-clear-slot");
        var input = slot === "crop" ? $("form-crop") : slot === "location" ? $("form-place") : $("form-timeframe");
        if (input) input.value = "";
        writeSlots({ clear: [slot] }, "clearing " + slot);
      });
    });

    $("btn-gps").addEventListener("click", useGps);
  }

  function intentRoute(name) {
    for (var i = 0; i < INTENTS.length; i++) if (INTENTS[i].name === name) return INTENTS[i].route;
    return "clarify";
  }

  function syncFormFromBag() {
    var slots = (S.bag && S.bag.slots) || {};
    if (document.activeElement !== $("form-crop")) $("form-crop").value = slots.crop ? (slots.crop.label || slots.crop.value) : "";
    if (document.activeElement !== $("form-place")) $("form-place").value = slots.location ? (slots.location.label || slots.location.value) : "";
    if (document.activeElement !== $("form-timeframe")) $("form-timeframe").value = slots.timeframe ? slots.timeframe.value : "";
    var intent = S.bag && S.bag.intent;
    $("form-intent").value = intent || "";
    $("form-intent-hint").textContent = intent
      ? ROUTE_TEXT[intentRoute(intent)]
      : "CropUp routes each message on its own.";
    var r = (S.bag && S.bag.field_radius_m) || (S.field && S.field.request && S.field.request.radius_m);
    if (typeof r === "number" && document.activeElement !== $("form-radius")) {
      $("form-radius").value = Math.min(1000, Math.max(5, r));
      $("form-radius-out").textContent = $("form-radius").value + " m";
    }
  }

  function useGps() {
    var note = $("gps-note");
    if (!navigator.geolocation) {
      note.textContent = "This browser has no geolocation API.";
      return;
    }
    if (!window.isSecureContext) {
      note.textContent = "Geolocation needs https (or localhost). Drop a pin on the map instead.";
      return;
    }
    note.textContent = "asking the device…";
    navigator.geolocation.getCurrentPosition(function (pos) {
      note.textContent = "device said ±" + Math.round(pos.coords.accuracy) + " m";
      writeSlots({
        point: { lat: pos.coords.latitude, lon: pos.coords.longitude, label: "My location" },
        origin: "device_gps"
      }, "using device GPS");
    }, function (err) {
      note.textContent = "GPS refused or unavailable (" + err.message + "). Drop a pin instead.";
    }, { enableHighAccuracy: true, timeout: 12000, maximumAge: 30000 });
  }

  /* ================================================================
   * the map: Leaflet from CDN, local plan view as the vendored fallback
   * ============================================================= */

  function onPick(lat, lon) {
    writeSlots({ point: { lat: Number(lat.toFixed(6)), lon: Number(lon.toFixed(6)), label: "Pinned field" } }, "placing the pin");
  }

  /* Did leaflet.css actually apply? index.html carries no onerror= handler --
   * an inline attribute IS inline script, and the CSP forbids inline script --
   * and an SRI mismatch is silent to the page anyway: the browser simply drops
   * the file. So ask the CSSOM instead. leaflet.css is the only stylesheet this
   * page loads that sets `.leaflet-pane { position: absolute }`, so one probe
   * element answers the question for a blocked CDN, a failed integrity check
   * and a truncated file alike. */
  var _cssProbe = null;
  function leafletCssApplied() {
    if (_cssProbe !== null) return _cssProbe;
    _cssProbe = false;
    try {
      var probe = el("div", { class: "leaflet-pane", style: "visibility:hidden;width:0;height:0" });
      document.body.appendChild(probe);
      var cs = window.getComputedStyle(probe);
      _cssProbe = !!cs && cs.position === "absolute";
      probe.remove();
    } catch (e) { _cssProbe = false; }
    return _cssProbe;
  }

  function leafletAvailable() {
    // Both halves must have arrived: Leaflet without its stylesheet is a broken
    // map, which is worse than the honest fallback.
    if (window.__cropupLeafletJs === false || window.__cropupLeafletCss === false) return false;
    if (typeof window.L === "undefined" || !window.L || typeof window.L.map !== "function") return false;
    return leafletCssApplied();
  }

  function createLeafletMap(host) {
    var L = window.L;
    var map = L.map(host, { scrollWheelZoom: false, zoomControl: true, attributionControl: true });
    map.setView([-3.38, 36.69], 3);
    var tileErrors = 0;
    var tiles = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 19, attribution: "Imagery &copy; Esri" }
    );
    tiles.on("tileerror", function () {
      tileErrors += 1;
      if (tileErrors === 4) {
        banner("Map tiles are not loading. Press “Plan view” to see the exact geometry without them.");
      }
    });
    tiles.on("tileload", function () { if (tileErrors < 4) banner(null); });
    tiles.addTo(map);
    var group = L.layerGroup().addTo(map);
    map.on("click", function (ev) { onPick(ev.latlng.lat, ev.latlng.lng); });

    function render(report, colours) {
      group.clearLayers();
      if (!report || !report.resolved || !report.request) return;
      var req = report.request;
      var maxR = req.radius_m || 15;
      (report.footprints || []).forEach(function (f, i) {
        if (typeof f.radius_m !== "number" || f.radius_m <= 0) return;
        maxR = Math.max(maxR, f.radius_m);
        L.circle([req.lat, req.lon], {
          radius: f.radius_m, color: legColour(f.leg, i), weight: 1.5,
          dashArray: "6 5", fillOpacity: 0.04, interactive: true
        }).bindPopup(popupNode(f.leg + " — " + metres(f.radius_m), [f.reason || ""])).addTo(group);
      });
      if (report.field) {
        L.geoJSON(report.field, {
          style: { color: "#1f6b46", weight: 2, fillColor: "#1f6b46", fillOpacity: 0.3 }
        }).bindPopup(popupNode("The plot", [metres(req.radius_m) + " radius around the pin"])).addTo(group);
      }
      L.circleMarker([req.lat, req.lon], { radius: 5, color: "#b3261e", fillColor: "#b3261e", fillOpacity: 1 })
        .bindPopup(popupNode(report.label || "field", [
          req.lat + ", " + req.lon,
          "the plot is " + metres(req.radius_m) + " radius — zoom in to see it against the wider rings"
        ])).addTo(group);
      map.fitBounds(L.latLng(req.lat, req.lon).toBounds(maxR * 2.4), { maxZoom: 18 });
      setTimeout(function () { map.invalidateSize(); }, 30);
    }

    return {
      kind: "leaflet",
      render: render,
      recentre: function () { if (S.field) render(S.field, LEG_COLOURS); },
      invalidate: function () { map.invalidateSize(); },
      destroy: function () { map.remove(); host.innerHTML = ""; }
    };
  }

  /* L.Popup.setContent accepts an HTMLElement as well as a string, and the
   * string form is parsed as HTML. Every popup here carries a leg name, a
   * server-written reason and the field label, so none of them go in as a
   * string: they go in as nodes whose text was set with textContent. That
   * removes the last HTML-parsing sink on the page, rather than guarding it
   * with an escaper a later caller could forget to call. */
  function popupNode(title, lines) {
    return el("div", { class: "mappop" },
      [el("strong", { text: title })].concat(
        (lines || []).filter(function (t) { return t; })
                     .map(function (t) { return el("div", { text: t }); })
      ));
  }

  function banner(text) {
    var b = $("map-banner");
    if (!text) { b.hidden = true; return; }
    b.textContent = text;
    b.hidden = false;
  }

  function mountPlanView(host, why) {
    S.map = window.CropUpPlanView.create(host, { onPick: onPick });
    S.mapMode = "plan";
    $("btn-map-mode").textContent = "Satellite map";
    banner(why);
    $("map-hint").textContent = "Click anywhere on the plan to place the field.";
  }

  function mountMap(mode) {
    var host = $("map");
    if (S.map) { try { S.map.destroy(); } catch (e) { /* already gone */ } S.map = null; }
    host.innerHTML = "";
    if (mode === "plan" || !leafletAvailable()) {
      mountPlanView(host, leafletAvailable()
        ? null
        : "Leaflet did not load from the CDN, or did not match the integrity digest this page pins, " +
          "so this is CropUp's own plan view — the same geometry, drawn to scale, with no network at all.");
    } else {
      try {
        S.map = createLeafletMap(host);
        S.mapMode = "leaflet";
        $("btn-map-mode").textContent = "Plan view";
        banner(null);
        $("map-hint").textContent = "Click the map to place the field.";
      } catch (e) {
        // A half-loaded Leaflet must not leave the farmer with no map at all,
        // and must not take the rest of wire() down with it.
        host.innerHTML = "";
        mountPlanView(host, "Leaflet loaded but would not start (" + ((e && e.message) || String(e)) +
          "), so this is CropUp's own plan view — the same geometry, drawn to scale.");
      }
    }
    if (S.field) S.map.render(S.field, LEG_COLOURS);
  }

  /* ================================================================
   * tabs
   * ============================================================= */

  function switchTab(which) {
    var ask = which === "ask";
    $("tab-ask").classList.toggle("is-active", ask);
    $("tab-form").classList.toggle("is-active", !ask);
    $("tab-ask").setAttribute("aria-selected", ask ? "true" : "false");
    $("tab-form").setAttribute("aria-selected", ask ? "false" : "true");
    $("view-ask").hidden = !ask;
    $("view-form").hidden = ask;
    if (!ask) syncFormFromBag();
  }

  /* ================================================================
   * session lifecycle
   * ============================================================= */

  function adoptSession(payload, note) {
    S.sid = payload.session_id;
    S.bag = payload.bag;
    S.field = payload.field;
    S.action = payload.action;
    S.awaiting = payload.action ? {
      kind: payload.action.kind,
      slot: payload.action.slot,
      missing_slots: payload.action.missing_slots || [],
      unconfirmed_slots: payload.action.unconfirmed_slots || [],
      options: payload.action.options || [],
      may_run: !!payload.action.authorises_earth_engine
    } : null;
    S.intentMeta = null;
    saveSnapshot();
    renderAll();
    connectEvents();
    if (note) toast({ title: "Session restored", body: note, kind: "info" });
  }

  function reopenSession() {
    dropSnapshot();
    return api("/api/session", "POST", {}).then(function (payload) {
      adoptSession(payload, "The old session had expired, so CropUp opened a new one. Nothing was carried over.");
    }).catch(function (err) { showError(err, "opening a session"); });
  }

  function boot() {
    api("/api/health").then(function (h) {
      $("app-version").textContent = "v" + ((h.app && h.app.version) || "?") +
        (h.status === "ok" ? "" : " · " + h.status);
    }).catch(function () {
      $("app-version").textContent = "unreachable";
    });

    refreshCapabilities();
    setInterval(refreshCapabilities, 60000);

    var saved = loadSnapshot();
    var opened;
    if (saved && saved.session_id) {
      opened = api("/api/session/" + encodeURIComponent(saved.session_id))
        .then(function (p) { adoptSession(p, null); })
        .catch(function (err) {
          if (err.status !== 404) showError(err, "rehydrating the session");
          return api("/api/session", "POST", { restore: saved })
            .then(function (p) { adoptSession(p, p.restore_note); });
        });
    } else {
      opened = api("/api/session", "POST", {}).then(function (p) { adoptSession(p, null); });
    }
    opened.catch(function (err) { showError(err, "opening a session"); });

    $("transcript").appendChild(transcriptEmpty());
  }

  /* ================================================================
   * wiring
   * ============================================================= */

  function wire() {
    $("tab-ask").addEventListener("click", function () { switchTab("ask"); });
    $("tab-form").addEventListener("click", function () { switchTab("form"); });

    $("composer").addEventListener("submit", function (ev) {
      ev.preventDefault();
      sendMessage($("composer-input").value);
    });
    $("composer-input").addEventListener("keydown", function (ev) {
      if (ev.key === "Enter" && !ev.shiftKey) {
        ev.preventDefault();
        sendMessage($("composer-input").value);
      }
    });
    $("composer-input").addEventListener("input", function () {
      var t = $("composer-input");
      t.style.height = "auto";
      t.style.height = Math.min(t.scrollHeight, 190) + "px";
    });

    $("capstrip-toggle").addEventListener("click", function () {
      var open = $("capstrip-detail").hidden;
      $("capstrip-detail").hidden = !open;
      $("capstrip-toggle").setAttribute("aria-expanded", open ? "true" : "false");
    });

    $("btn-map-mode").addEventListener("click", function () {
      mountMap(S.mapMode === "plan" ? "leaflet" : "plan");
    });
    $("btn-map-centre").addEventListener("click", function () { if (S.map) S.map.recentre(); });

    $("btn-reset").addEventListener("click", function () {
      dropSnapshot();
      clear($("transcript"));
      $("transcript").appendChild(transcriptEmpty());
      $("runpanel").hidden = true;
      S.run = { planned: [], byName: {}, active: false };
      api("/api/session", "POST", {})
        .then(function (p) { adoptSession(p, null); })
        .catch(function (err) { showError(err, "opening a session"); });
    });

    document.addEventListener("click", function (ev) {
      if (!ev.target.closest || !ev.target.closest(".seg")) closeOthers(null);
    });
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape") closeOthers(null);
    });

    buildForm();
    mountMap("auto");
    switchTab("ask");
  }

  function start() {
    wire();
    boot();
  }

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
    else start();
  }

  /* Exported so the panels can be driven against recorded server responses
   * without a browser (there is no node in this environment, so the checks run
   * in JavaScriptCore against a DOM stub). The page itself calls none of these
   * through this object. */
  window.CropUpInternals = {
    state: S,
    renderSegment: renderSegment,
    renderAnswer: renderAnswer,
    renderSlots: renderSlots,
    renderField: renderField,
    renderGate: renderGate,
    renderCapabilities: renderCapabilities,
    renderLegs: renderLegs,
    applyEnvelope: applyEnvelope,
    describeOutcomes: describeOutcomes,
    slotState: slotState,
    intentRoute: intentRoute,
    INTENTS: INTENTS,
    metres: metres
  };
})();
