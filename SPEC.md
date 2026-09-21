# CropUp — build specification

Synthesized from: a 3-way architecture panel scored by 12 judges, three grounding
studies (intent taxonomy, HuggingFace model selection, vendored-backend audit),
and a 6-family empirical probe of the GEE Community Catalog. Every dataset named
here was queried live in Earth Engine before being written down.

---

## 1. What CropUp is

A **standalone** crop-understanding webapp. A farmer asks a question in plain
English — or fills in a questionnaire — and gets an answer grounded in Earth
observation, where **every number on screen carries the instrument that produced
it**, and anything not measured is named as missing rather than guessed.

CropUp is not a thin client over someone else's service. The analysis layer and
the RAG layer both live **in this repo**.

### 1.1 Hard constraints (verified, not assumed)

| Constraint | Consequence |
|---|---|
| `Data/` is **gitignored** | Nothing under `Data/` may be a runtime dependency. All required data is copied into `cropup/data/`. |
| No `node`, no `npm` | Frontend is vanilla HTML/CSS/JS, no build step, served by the Python app. |
| Python is **x86_64 under Rosetta**; torch capped at 2.2.2 and **broken** against NumPy 2.2.6 | **Never import torch.** NLU runs on `onnxruntime` + `tokenizers`. |
| EE project is `irrigation-status-474718` | Set before any `ee.Initialize`; verified working with local user credentials. |
| `Data/googlebuildathonfarmers-main` contains a `scripts/` dir | CropUp's helper dir is **`tools/`**, never `scripts/`, to avoid a PEP-420 namespace collision. |
| The vendored package is named `app` | CropUp's package is **`cropup`**, never `app`. |

### 1.2 The number that drives the design

Of 78 real farmer questions (Kijani WhatsApp corpus, human-scored):

- **0** carry GPS. ~60 give no location at all. 24 name no crop.
- **0** are "how is my plant health today" — the reference repo's flagship intent
  has *no counterpart in real traffic*.
- Only **22/78 (28%)** can be served by any Earth Engine pipeline.
  Only **9/78 (12%)** carry both an EE-routable intent and a resolvable location.
- **56/78 (72%)** is knowledge work that must go to RAG and must never touch EE.

So the geospatial pipelines are the *minority* path. Slot-filling is not a
fallback — it is the mechanism that makes anything answerable at all. The
questionnaire and the chat are two views of one slot frame, not two features.

---

## 2. Architecture

```
cropup/
  __init__.py          version only; imports nothing heavy
  bootstrap.py         MUST be imported first by every entrypoint.
                       Sets EE_PROJECT_ID, owns the single ee.Initialize().
  config.py            Settings from env; CROPUP_* names only.
  evidence.py          Fact + Ledger. The fabrication firewall (§4).
  errors.py

  data/                committed artifacts, no Data/ dependency
    disease_library.csv    384 rules, 102 crops
    gazetteer.json         1,073 places, k-anonymised (§7)
    crops.json             134 crops, 379 aliases incl. Swahili
    ee_registry.json       92 live-verified EE datasets
    corpus/                RAG knowledge cards (§6)

  geo/                 Earth Engine data layer. One module per quantity.
    registry.py          loads ee_registry.json; resolves source chains
    soil.py              iSDA -> POLARIS -> SoilGrids
    vegetation.py        Sentinel-2 indices
    thermal.py           Landsat LST
    water.py             ET, ET0, ESI, soil moisture, precipitation
    climate.py           Köppen computed in-EE from climate normals
    suitability.py       CropSuite
    context.py           cropland / irrigated-vs-rainfed

  analysis/            orchestrators, composed from geo/ leaves
    plant_health.py
    irrigation.py
    crop_selection.py
    rules.py             disease-library rules engine (ported, bugs fixed)

  nlu/
    embed.py             ONNX MiniLM encoder
    intents.py           the 12 intents + seed utterances
    classify.py          3-tier cascade (§5)
    slots.py             crop + location + timeframe extraction

  rag/
    corpus.py            build/load the knowledge cards
    index.py             embedding index (numpy, no external vector DB)
    retrieve.py          retrieve + cite

  dialog/
    slots.py             SlotBag: the one shared state
    policy.py            next_action(frame) — pure, mode-blind

  render/
    templates.py         deterministic rendering, provenance-gated

  web/
    server.py            FastAPI app
    static/              index.html, app.js, style.css

tools/                 dev-time CLIs (build_gazetteer, build_crop_vocab, eval_nlu, ...)
tests/
```

**Import rule.** `geo/*` may import `evidence`, `config`, `registry`. `analysis/*`
may import `geo/*` and `evidence`. `nlu/*` and `rag/*` import neither `geo` nor
`analysis`. `dialog` imports `nlu`; `render` imports `evidence` only. `web`
imports everything. No cycles.

### 2.1 Porting the vendored backend

The audit found **27 bugs** in `Data/googlebuildathonfarmers-main`. Port by
**composing from its leaf functions and rewriting the orchestrators**, never by
wrapping `analyze_*`. Bugs that must not survive the port:

1. `climate.py` swallows the missing-raster error and **returns a fabricated
   Köppen code** (`Cfb`/"Temperate" for Arusha, which is wrong). Replaced
   wholesale by `geo/climate.py`, computed in EE. Unknown must be `UNKNOWN`.
2. `maxent.py::_get_maxent_conn()` has **no return statement**. Replaced by
   CropSuite.
3. `soil.py`: the iSDA REST endpoint returns **401**, and the POLARIS Duke
   endpoint is **dead**. Both replaced by EE assets.
4. `crop_recommendations.py` interpolates **silent defaults (pH 7.0, sand 30%)
   into user-facing prose** when soil is missing. Forbidden by §4.
5. NDVI zone percentages are shifted by one histogram bin.
6. `min_temp_7d` is requested from NASA POWER but never computed.
7. `rules_engine` raises `AttributeError` when `crop_type` is `None`.
8. Rules are overwhelmingly NDVI-driven: 296 of ~385 conditions key on `ndvi`
   alone, so one low NDVI fires ~18 risks at once including alarming ones
   ("Maize Lethal Necrosis", "Heavy Metal Contamination"). Must be ranked,
   deduplicated and capped before display (§4.3).

---

## 3. Earth Engine data layer

All sources are EE assets. **No REST soil APIs, no local rasters, no 80MB DB.**

### 3.1 Source chains

| Quantity | Chain | Notes |
|---|---|---|
| soil ph/clay/sand/silt/soc/n/cec | `ISDASOIL/Africa/v1/*` → `polaris/*_mean` → `projects/soilgrids-isric/*_mean` | Africa → US → global |
| NDVI/NDMI/NDRE/NDWI/PSRI | `COPERNICUS/S2_SR_HARMONIZED` | |
| LST | `LANDSAT/LC08|09/C02/T1_L2` | QA-mask required |
| actual ET | `projects/usgs-ssebop/viirs_et_v6_dekadal` → `IDAHO_EPSCOR/TERRACLIMATE` | **not MODIS** |
| reference ET0 | `projects/sat-io/open-datasets/global_et0/global_et0_monthly` | global |
| evaporative stress | `projects/climate-engine/esi/4wk` | global but **often stale** |
| soil moisture | `NASA/SMAP/SPL4SMGP/008` → `NASA/FLDAS/NOAH01/C/GL/M/V001` | 007 is dead |
| precipitation | `UCSB-CHG/CHIRPS/DAILY` | 50S–50N |
| Köppen | computed from `WORLDCLIM/V1/MONTHLY` + `IDAHO_EPSCOR/TERRACLIMATE` | no Köppen asset exists |
| crop suitability | `projects/sat-io/open-datasets/CROP_SUITE/*` | **Africa only** |
| is_cropland | `DEAF/CROPLAND-EXTENT/prob` (Africa) → `GFSAD/GCEP30` → `ESA/WorldCover/v200` | |
| irrigated vs rainfed | `GFSAD/LGRIP30` | |
| typical NPK | `projects/sat-io/open-datasets/NPKGRIDS` | descriptive, **not** a prescription |

### 3.2 Scaling traps — each one silently produces a plausible wrong number

These were found empirically. Encode them in `registry.py`, never inline.

- **iSDA `silt_content`**: the EE STAC documents `exp(x/10)-1`. **That is wrong.**
  Use the raw value.
- **iSDA `nitrogen_total`**: `exp(x/100)-1` — note `/100`, unlike every other
  log-transformed iSDA band which uses `/10`.
- **iSDA `carbon_organic`**: `exp(x/10)-1`. **iSDA `ph`**: `x/10`.
- **POLARIS `om` and `ksat`**: log10. `10**raw`. Raw `om=0.293` reads as a
  plausible 0.29% but is actually 1.96%.
- **`global_ai`**: divide by 10000. Its sibling `global_et0` needs **no** scaling.
- **ERA5-Land**: evaporation/precipitation are in **metres**, and evaporation is
  **negative** for an upward flux.
- **NASA Harvest** probabilities are already 0–1; **DEAF** probabilities are 0–100.
- **PEST-CHEMGRIDS**: negative sentinels (`-1.5`) that a naive read reports as a
  real application rate; `quality_index == 0` marks invalid pixels.
- **FUBC fertilizer table**: every attribute is a **string**, with literal `'NA'`.

### 3.3 Coverage cliffs — must degrade visibly, never silently

- **CropSuite is Africa-only**, and at the exact Arusha pixel it is **masked for
  47 of 48 crops in all 6 scenarios**. Use a small neighbourhood reduction and
  say so.
- **`fret/forecast/eto` and OpenET are CONUS-only** — null in Tanzania. So
  forward-looking irrigation advice is US-only. Never present a US-only answer
  to a Tanzanian farmer.
- **MODIS `MOD16A2GF` is permanently masked at Arusha** across all 46 images of
  2025. This is why actual ET uses SSEBop VIIRS.
- **POLARIS is US-only**; correctly returns `None` (not 0) elsewhere.
- ESI at Arusha returned a reading **over a year old**. Staleness must be shown.

### 3.4 Latency

The vendored `plant_health` takes **28.7s**. Budget: batch `reduceRegion` calls
into one `ee.Dictionary` per geometry, cache per (lat,lon,date) in-process, and
stream progress over SSE so the UI is never blank.

---

## 4. The fabrication firewall

The human rubric for this domain has an explicit band for *"risk the user does
something harmful to their crops/environment/society."* The ChatGPT baseline
scored 0 or 1 on ~16% of real questions. So:

### 4.1 `evidence.Fact`

Every measured value is a `Fact`: `value`, `unit`, `source_asset`, `observed_on`,
`resolution_m`, `chain_position`. A value with no `Fact` cannot reach the
renderer. `Ledger` collects the facts used by a turn plus a named `gaps` list.

### 4.2 Renderer-level impossibility

`render/templates.py` accepts **only** `Fact` objects and literal template text.
It has no access to raw floats. Rendering a template whose slot has no `Fact`
raises — it does not fall back to a default. This makes "silently print 7.0 for
missing pH" unrepresentable rather than merely discouraged.

### 4.3 No generative LLM

Response generation is **deterministic templates over structured facts**, plus
**verbatim cited** RAG snippets. No free text generation. A 0.5B model that
invents a fertilizer dose is a safety failure, not a feature.

Risk lists are ranked by severity, deduplicated by `issue_id`, **capped at 5**,
and each carries the observation that triggered it. An 18-risk dump is a bug.

### 4.4 Never run EE on an unconfirmed field

A natural-language turn never triggers Earth Engine. The farmer must confirm the
resolved location and crop first. This prevents burning quota on a misparse and
prevents answering about the wrong field.

---

## 5. NLU

Verified: MiniLM INT8 ONNX is **23MB**, 0.41s cold, **2.5ms** warm,
9.8ms across all 78 real questions. Torch is never imported.

**Centroid scoring beat kNN decisively** — leave-one-out on 55 seed utterances
over 9 intents: centroid **67.3%**, kNN k=5 54.5%, k=3 50.9%, top-1 45.5%.
Use centroids.

### 5.1 Three-tier cascade

1. **Rules** (always, 0ms, no model): weighted keyword/regex over the 12 intents.
   A hit with score ≥3 *and* margin ≥2 short-circuits.
2. **Embeddings**: MiniLM centroid over seed utterances. Primary router.
3. **Zero-shot NLI** (`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`, 739MB, 74ms),
   **optional tie-breaker only**. Measured 5/8 on ad-hoc labels vs 2/4 for the
   xsmall variant; entailment is index **0**. Ship disabled by default.

Below a confidence floor, route to `clarify` and ask. Misrouting silently is
worse than asking.

### 5.2 Intents (12)

`field_health_check`, `crop_problem_diagnosis`, `irrigation_advice`,
`crop_selection` → EE pipelines.
`fertilizer_advice`, `soil_fertility_management`, `seed_variety_selection`,
`crop_management_practice`, `market_and_inputs_supply`, `livestock_and_adjacent`,
`agronomy_concept_explainer` → RAG.
`out_of_scope_or_unclear` → clarify.

### 5.3 Slots

Closed vocabularies beat NER here. Crop extraction is **rapidfuzz against the
134-crop / 379-alias vocabulary** including Swahili (`mahindi`→Maize,
`muhogo`→Cassava, `mpunga`→Rice). Location is gazetteer lookup; 7 short names are
flagged `ambiguous` (`Hai`, `Same`, …) and require a corroborating parent region
or an explicit confirmation before use.

---

## 6. RAG (new — does not exist in the reference repo)

Serves 72% of real traffic. Corpus is **curated and citable**, not scraped:

- 384 disease-library rules → one card each (symptoms, thresholds, action).
- Practice cards authored from the vendored `crop_recommendations` and
  `soil_recommendations` logic, converted from code branches into documents.
- Dataset provenance cards, so "where does this number come from" is answerable.

Index: MiniLM embeddings in a numpy matrix — no external vector DB for a corpus
this size. Retrieval returns snippets **verbatim with a citation**. If nothing
clears the similarity floor, say so and offer the questionnaire. Never
paraphrase a retrieved agronomic claim.

---

## 7. Privacy

`UserLocations` is personal data (ProfileId + exact GPS). The gazetteer built
from it is committed, so: entries backed by **fewer than 5 pings are suppressed**
(2,889 dropped, 1,073 kept) and coordinates are rounded to ~1km. Seeded town
centroids are public geography and exempt. Still resolves **26/26** place names
appearing in the real corpus. No raw ping, ProfileId or farmer message is ever
committed or logged.

---

## 8. HTTP API

```
GET  /                       single-page app
GET  /api/health             liveness + startup assertions
GET  /api/capabilities       THE honesty endpoint: live degradation matrix
POST /api/session            open a session (one SlotBag)
GET  /api/session/{sid}      rehydrate after reload
POST /api/session/{sid}/message    NL turn. NEVER runs EE.
POST /api/session/{sid}/slots      questionnaire/map writes; user values LOCK
POST /api/session/{sid}/confirm    the gate that authorises an EE run
POST /api/session/{sid}/run        explicit run
GET  /api/session/{sid}/events     SSE progress per data leg
POST /api/nlu/parse          pure NLU inspection, no side effects
GET  /api/vocab/crops        autocomplete
GET  /api/vocab/places       gazetteer autocomplete
GET  /api/geo/field          the exact polygon that WILL be sent to EE
```

---

## 9. Frontend

Vanilla JS, no build. Leaflet from CDN with a vendored local fallback. Two
panels over one SlotBag: **Ask** (chat) and **Form** (questionnaire), switchable
mid-conversation without losing state. A slot filled by NLU shows as a
*suggestion* the farmer can correct; a slot the farmer sets is **locked**. Every
rendered number is hoverable to reveal its instrument, date and resolution. A
persistent capability strip shows what is currently degraded.

---

## 10. Test plan

- Unit: scaling transforms (each trap in §3.2 gets a test), rules engine,
  slot extraction, `Ledger` gap accounting.
- **Fabrication tests**: assert the renderer *raises* on a missing `Fact`;
  assert no default value ever reaches output.
- NLU eval: 98-question labeled set (78 real + 20 synthetic); report per-intent
  precision/recall; the harness lives in `tools/eval_nlu.py`.
- Integration against live EE at the three test points, including the known
  coverage cliffs — a Tanzanian point **must** report forecast-ET as unavailable.
- `null_adapter` run: every EE source unavailable; the app must still answer or
  honestly decline, never crash.
