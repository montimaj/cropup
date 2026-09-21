# CropUp

A standalone crop-understanding webapp. A farmer asks a question in plain English
— or fills in a questionnaire — and gets an answer grounded in Earth observation,
where **every number on screen carries the instrument that produced it**, and
anything that was not measured is named as missing rather than guessed. The
analysis layer and the retrieval layer both live in this repo; CropUp is not a
thin client over someone else's service.

**`SPEC.md` is the authoritative design document.** This file only tells you how
to run the repo and what is actually true today; read the SPEC before you change
anything, especially section 4 (the fabrication firewall) and section 2's import
rule.

## Constraints that will bite you on day one (SPEC 1.1)

| Constraint | What it means for you |
|---|---|
| Python is **x86_64 under Rosetta** (3.12.11 verified) | Wheels are x86_64. Do not rebuild the venv with an arm64 interpreter — `onnxruntime` and the pinned NumPy will not match. Check with `python -c "import platform; print(platform.machine())"`; it must print `x86_64`. |
| **Never import `torch`** | torch 2.2.2 is the ceiling on this platform and it is broken against NumPy 2.2.6. It *is* present in the interpreter's site-packages from unrelated projects, so nothing stops you by accident — the `no_torch_imported` startup assertion is what catches it, and `/api/health` reports it. NLU runs on `onnxruntime` + `tokenizers`. Never add `transformers` or `sentence-transformers`: both pull torch. |
| No `node`, no `npm` | The frontend is vanilla HTML/CSS/JS with no build step, served by the Python app out of `cropup/web/static/`. Leaflet comes from a CDN with a vendored `map-fallback.js` for when it does not load. Do not introduce a bundler. |
| Earth Engine project is `irrigation-status-474718` | `cropup/bootstrap.py` owns the single `ee.Initialize()` and sets the project id. Never call `ee.Initialize()` anywhere else, and never call Earth Engine from a module outside `cropup/geo/`. |
| `Data/` is gitignored | Nothing under `Data/` may be a runtime dependency. Everything the app needs at runtime is committed under `cropup/data/`. |
| The helper dir is `tools/`, never `scripts/`; the package is `cropup`, never `app` | Both alternatives collide with the vendored reference backend in `Data/googlebuildathonfarmers-main`. |

## Install

```bash
python3.12 -m venv .venv                                    # x86_64 interpreter
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import platform; print(platform.machine())"   # -> x86_64
```

`requirements.txt` is pinned to the environment that is verified working. Do not
float the pins, and keep the "deliberately absent" note at the top of it intact.

CropUp is **not pip-installed** — `pyproject.toml` deliberately has no
`[build-system]`. Run everything **from the repo root** so `cropup` is importable.

## Run it

Offline (no Earth Engine, no network) — do this first:

```bash
cd /path/to/cropup
CROPUP_EE_ENABLED=0 .venv/bin/python -m cropup.web
# -> Uvicorn running on http://127.0.0.1:8000   (CROPUP_HOST / CROPUP_PORT)
```

Open `http://127.0.0.1:8000/` for the two-panel UI; `/docs` is the generated
OpenAPI page.

**What `CROPUP_EE_ENABLED=0` does, and why you want it.** It is SPEC section
10's `null_adapter` run. `bootstrap` skips `ee.Initialize()` entirely, every EE
source reports itself unavailable, and the app must still answer from the cited
knowledge base or *decline honestly* — never crash, never invent a number. It is
the fastest way to work on dialog, NLU, RAG, the renderer or the frontend, it
burns no EE quota, and it is the run that proves the fabrication firewall holds.
What that looks like today:

```
$ curl -s localhost:8000/api/health   | jq -r '.status, .degraded[]'
ok
ee:initialized

$ curl -s localhost:8000/api/capabilities | jq -r .summary
6 ready, 1 unavailable, 5 unknown | not working: earth_engine | cannot: run_field_analysis

$ curl -s -XPOST localhost:8000/api/session/$SID/run -d '{"intent":"field_health_check"}'
{"error":{"type":"EarthEngineUnavailable","status":503,
  "message":"Earth Engine is unavailable right now, so I cannot measure this field.
             I can answer from the cited knowledge base instead, or you can try again later.",
  "remedy":"GET /api/capabilities shows what is degraded; knowledge questions are still
            answered from the cited corpus"},"ran_earth_engine":false}
```

With Earth Engine (needs credentials, see below):

```bash
.venv/bin/python -m cropup.web
```

In a test or a REPL, without a port:

```python
from fastapi.testclient import TestClient
from cropup.config import get_settings
from cropup.web.server import create_app

client = TestClient(create_app(get_settings()))   # CROPUP_EE_ENABLED=0 in the env
client.get("/api/capabilities").json()
```

### Earth Engine credentials

CropUp uses local *user* credentials, not a service account. One time:

```bash
.venv/bin/python -c "import ee; ee.Authenticate()"
```

They land in `~/.config/earthengine/credentials`. To see what the app itself
thinks of your environment — committed artifacts, the torch assertion, and
whether EE actually answered a round trip:

```bash
.venv/bin/python -c "import json; from cropup import bootstrap; print(json.dumps(bootstrap.health_report(), indent=2))"
```

## Settings

All settings are `CROPUP_*` and are read once from the environment into a frozen
`Settings` (`cropup/config.py`). **A bad value raises `ConfigError`** rather than
falling back to the shipped default — a misspelled threshold must not quietly
become the one CropUp was built with. `/api/health` echoes the whole resolved
snapshot; it contains no secrets by design.

| Variable | Default | What it does |
|---|---|---|
| `CROPUP_EE_ENABLED` | `1` | `0` is the offline null-adapter run described above. |
| `CROPUP_EE_PROJECT_ID` | `irrigation-status-474718` | SPEC 1.1. Changing it is almost certainly wrong. |
| `CROPUP_EE_VERIFY_ON_INIT` | `1` | One cheap round trip at startup so `ee_ready()` is measured, not assumed. `0` makes startup faster and `ready` a guess. |
| `CROPUP_FIELD_RADIUS_M` | `15` | Radius of the disc a confirmed point is buffered into before it is sent to EE — i.e. the exact polygon `GET /api/geo/field` shows the farmer. Must be positive and finite or it is a `ConfigError`. A session may override it per field; this is the value used when the farmer has not chosen one. Not to be confused with `CROPUP_EE_NEIGHBOURHOOD_M`. |
| `CROPUP_EE_NEIGHBOURHOOD_M` | `300` | The *wider* reduction used where a coarse or masked asset needs it (SPEC 3.3, e.g. CropSuite masking single pixels). |
| `CROPUP_EE_REQUEST_TIMEOUT_S` | `60` | Per round trip; `0` disables the deadline. |
| `CROPUP_EE_MAX_WORKERS` | `4` | Parallel EE legs. |
| `CROPUP_HOST` / `CROPUP_PORT` | `127.0.0.1` / `8000` | Where `python -m cropup.web` binds. |
| `CROPUP_LOG_LEVEL` | `info` | One of critical/error/warning/info/debug; configures uvicorn and `bootstrap.configure_logging` together. |
| `CROPUP_ALLOW_MODEL_DOWNLOAD` | `1` | `0` forbids fetching the MiniLM ONNX encoder; the encoder then raises `ModelUnavailable` instead of reaching the network. |
| `CROPUP_MODEL_CACHE_DIR` | huggingface_hub's choice | Where that encoder is cached. |
| `CROPUP_NLI_ENABLED` | `0` | The optional 739 MB zero-shot tie-breaker (SPEC 5.1). Ships disabled. |
| `CROPUP_INTENT_CONFIDENCE_FLOOR` / `CROPUP_INTENT_MARGIN_FLOOR` | `0.40` / `0.05` | Below the floor, or inside the margin, a turn routes to `clarify` instead of guessing an intent. |
| `CROPUP_RAG_SIMILARITY_FLOOR` / `CROPUP_RAG_TOP_K` | `0.30` / `4` | Below the floor, RAG says it found nothing and offers the questionnaire. |
| `CROPUP_RISK_CAP` | `5` | SPEC 4.3: risks ranked, deduplicated by `issue_id`, capped. |
| `CROPUP_REQUIRE_CONFIRMATION` | `1` | SPEC 4.4's gate. `0` would let a run fire on an unconfirmed field — do not ship it off. |
| `CROPUP_SESSION_TTL_S` | `86400` | Sessions are in-process only; a restart forgets them. |
| `CROPUP_DATA_DIR` / `CROPUP_CORPUS_DIR` / `CROPUP_STATIC_DIR` | under `cropup/` | Point the app at artifacts elsewhere. |

The rest (cache TTLs, rule/slot-match floors, staleness window, embedding model
ids) are in `Settings.from_env()` in `cropup/config.py`; every field there has a
`CROPUP_*` name.

## Committed data artifacts and how to rebuild them

The app reads only these. `tools/` builds some of them; the rest have no builder.

| Artifact | Contents | Builder | Inputs |
|---|---|---|---|
| `cropup/data/gazetteer.json` | 1,073 places, k-anonymised (SPEC 7) | `tools/build_gazetteer.py` | `Data/UserLocations_*.csv` — **gitignored personal data** |
| `cropup/data/crops.json` | 134 crops, 379 aliases incl. Swahili | `tools/build_crop_vocab.py` | `Data/googlebuildathonfarmers-main` — **gitignored** |
| `cropup/data/corpus/` | 515 RAG cards (92 dataset, 384 disease, 25 practice, 14 topic) + numpy index | `tools/build_corpus.py` | committed files only |
| `cropup/data/ee_registry.json` | 92 live-verified EE datasets and their scaling rules | none — maintained by hand | each entry was queried live in EE before being written down |
| `cropup/data/disease_library.csv` | 384 rules over 102 crops | none — copied from the vendored repo | — |

So on a clean checkout only the corpus rebuilds:

```bash
.venv/bin/python tools/build_corpus.py              # cards + embedding index
.venv/bin/python tools/build_corpus.py --no-index   # cards only, no encoder needed
.venv/bin/python tools/build_corpus.py --check      # is the built index current?
```

The index step downloads the 23 MB MiniLM INT8 ONNX encoder from HuggingFace on
first run; `CROPUP_ALLOW_MODEL_DOWNLOAD=0` forbids that, and the encoder then
raises `ModelUnavailable` instead of fetching. Exit code 2 means the cards were
written but the index was skipped because no encoder was available.

The other two builders overwrite committed artifacts and need `Data/` present,
so run them only when you have the raw inputs.

## Tests

```bash
.venv/bin/python -m pytest
.venv/bin/python -m pytest -m "not ee"     # skip anything needing live EE
```

Configuration is in `pyproject.toml`: `testpaths = ["tests"]` and the repo root
on `sys.path`, so no `conftest.py` is needed and `cropup` imports without being
installed.

The SPEC section 10 suite exists: **327 tests, all passing, in ~1.5 s**, and the
whole suite runs offline with no Earth Engine and no network. It covers each
section 3.2 scaling trap individually (they are the ones that silently produce a
plausible wrong number), the rules engine's ranking/dedup/cap-at-5, slot
extraction including the Swahili vocabulary and the cross-slot arbitration,
`Ledger` gap accounting, the section 4.4 confirmation gate, and the
**fabrication tests** section 10 asks for by name — that the renderer *raises*
on a missing `Fact` and that no default value ever reaches output.

Two conventions worth keeping:

- A test that fails because the *code* is wrong is marked `xfail(strict=True)`
  with the finding written into the reason, never weakened to pass. Strict means
  the suite breaks the moment the bug is fixed, so the marker cannot be
  forgotten. There are no such markers right now — the last one was retired when
  `/api/nlu/parse` stopped leaking the model path.
- The NLU eval harness is separate, at `tools/eval_nlu.py`, because it reports
  per-intent precision/recall rather than passing or failing:

  ```bash
  CROPUP_EE_ENABLED=0 .venv/bin/python tools/eval_nlu.py
  ```

  It runs on a committed **synthetic** 98-question set. The 78 real farmer
  questions are *not* committed — section 7 forbids it — but the harness loads
  them from the gitignored `Data/` workbook when it is present, so you can
  evaluate against real traffic locally without ever publishing it.

## Current status

Hackathon build. Verified by running it offline (`CROPUP_EE_ENABLED=0`) on
2026-09-21; nothing below is inferred from the code alone.

**Working**

- Every module under `cropup/` imports cleanly and without pulling torch.
  `/api/health` returns `status: ok` with the only degraded check being
  `ee:initialized`, and asserts `no_torch_imported`, the four committed
  artifacts (92 / 134 / 1,073 / 384 records), the corpus index (515 cards,
  384-d) and `web/static`.
- **The web tier exists and serves.** `cropup/web/` is ~2,700 lines across
  `server.py`, `session.py`, `dispatch.py`, `replies.py`, `progress.py`,
  `geofield.py` and `__main__.py`, plus `static/` (`index.html`, `app.js`,
  `style.css`, `map-fallback.js`). `python -m cropup.web` starts uvicorn and
  `GET /` returns the page.
- **All 14 SPEC section 8 endpoints are routed and answer**: `/`,
  `/api/health`, `/api/capabilities`, `POST /api/session`,
  `GET /api/session/{sid}`, `.../message`, `.../slots`, `.../confirm`,
  `.../run`, `.../events`, `/api/nlu/parse`, `/api/vocab/crops`,
  `/api/vocab/places`, `/api/geo/field`, with `/static` mounted.
- Exercised end-to-end offline: a chat turn routes through the NLU cascade and
  never touches EE; a knowledge question returns a RAG answer whose lines are
  verbatim with citations; questionnaire writes lock slots; `confirm` is the
  only thing that authorises a run and refuses to confirm an empty slot (400);
  `run` without confirmation is a `409 ConfirmationRequired`; `run` with EE
  disabled is a `503 EarthEngineUnavailable` carrying a remedy and
  `ran_earth_engine: false`; the SSE stream emits `open` / `slots` /
  `heartbeat` events with replayable ids.
- `/api/capabilities` is honest about itself: with EE off it reports
  `6 ready, 1 unavailable, 5 unknown`, names what was lost (the 4 of 12 intents
  that need EE) and lists the coverage questions it cannot answer offline.

**Incomplete or unverified**

- `tests/` is empty. None of SPEC section 10 exists: no scaling-trap unit tests,
  no fabrication tests asserting the renderer *raises* on a missing `Fact`, no
  `Ledger` gap accounting tests, no NLU eval. `tools/eval_nlu.py` is not written.
- Nothing here has been exercised against **live Earth Engine** — every check
  above was made with `CROPUP_EE_ENABLED=0`. The EE legs, the scaling rules in
  `ee_registry.json` and the SPEC 3.3 coverage cliffs are unverified end-to-end
  through the web tier.
- The frontend has been confirmed only as *served and wired* (Leaflet + local
  fallback, the Ask/Form tabs over one slot bag, the capability strip, an
  `EventSource` on the SSE endpoint). It has not been walked through in a
  browser as part of this check.
- The NLU floors are doing real work: near-tied questions (e.g. "how do I
  control fall armyworm in maize?" scored 0.488 vs 0.486) route to `clarify`
  rather than guess. That is the intended behaviour, but it means the clarify
  path is common and deserves the NLU eval it does not yet have.

## Layout

```
cropup/
  bootstrap.py   the single ee.Initialize(); import before touching EE
  config.py      CROPUP_* settings
  evidence.py    Fact / Missing / Ledger — the fabrication firewall (SPEC 4)
  capabilities.py  the live degradation matrix behind /api/capabilities
  data/          committed artifacts, no Data/ dependency
  geo/           EE data layer, one module per measured quantity
  analysis/      orchestrators composed from the geo/ leaves
  nlu/  rag/     ONNX intent routing and citable retrieval; neither touches EE
  dialog/        SlotBag + the mode-blind next_action policy
  render/        deterministic templates; accepts Fact objects only
  web/           FastAPI app (server, session, dispatch, replies, progress,
                 geofield) + the vanilla-JS frontend in static/
tools/           dev-time CLIs that build cropup/data/
tests/           empty
```

Import rule (SPEC 2): `geo/*` may import `evidence`, `config`, `registry`;
`analysis/*` may import `geo/*` and `evidence`; `nlu/*` and `rag/*` import
neither `geo` nor `analysis`; `dialog` imports `nlu`; `render` imports
`evidence` only; `web` imports everything. No cycles.

Licensed under Apache-2.0; see `LICENSE`.
