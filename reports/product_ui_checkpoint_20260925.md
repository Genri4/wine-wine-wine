# Product UI checkpoint — 2026-09-25

## Agreed scope

- Delivery target: 2026-09-29, based on the user's deadline in this task.
- MVP scope is the PDF-level desktop UI plus the current post-search flows:
  Match for You, Compare, and the personal wine collection.
- The user's latest direction overrides the PDF's mobile-first suggestion:
  desktop web only, with no mobile layout adaptation.
- ML experiments are paused; do not fine-tune, run new model experiments, or
  change retrieval/reranking while this product pass is active.

## Delivered desktop MVP

- `web/` now has a wide desktop scanner layout with local image upload and
  desktop-width result, retry and no-match screens. It has no narrow-screen
  responsive breakpoints.
- Smart Retry checks decoding, minimum longest-side resolution (640 px), and
  strong blur in the browser, then sends eligible photos to the local frozen
  recognition runtime. Backend retry reasons include unreadable, resolution,
  blur, glare, distance, barcode, missing year, front label and uncertainty.
- Match for You reads `data/processed/catalog_manifest.csv`, offers category,
  grape and region preferences, and shows up to three other catalog cards.
- Its score is the percentage of selected catalog attributes matched, with
  equal weight per selected attribute. It is not a quality rating or a learned
  preference model. Preference values remain in the page only.
- Compare lets the user keep one recognized product while scanning a second,
  then displays the two catalog cards side by side. The temporary selection is
  held in page memory.
- The personal collection stores recognized products in browser local storage
  as “Хочу попробовать” or “Уже пробовал(а)” with a required 1–5 rating. Users
  can edit or remove entries; the list is local to this browser profile.
- `web/README.md`, root `README.md` and `ARCHITECTURE.md` document the integrated
  runtime and its limits. ML and raw image data were not changed in this pass.

## Current limits

- Sharpness and recognition-agreement cutoffs are heuristics, not calibrated
  probabilities. The 100-photo dataset has no verified slug/no-match labels;
  review is required before claiming field-calibrated open-set performance.
- The site palette and wordmark are provisional because no verifiable official
  design reference was available during this pass. Desktop only, as requested.
- The local browser preview is served at `http://127.0.0.1:8765/web/`; refresh
  the open tab to load the integrated version.
- Regression audit: 334 project tests passed, with 2 pretrained-model
  reproduction tests deselected after the unfiltered run stalled at a socket
  read. The UI was not clicked through in a real browser because the Windows UI
  bridge rejects this WSL workspace URI; JS/HTML selectors and live HTTP paths
  were checked instead.

## Smart Retry goal checkpoint — 2026-09-25

### Frozen runtime decision

- Use the validated **R8 LoRA epoch 5**, not canonical or R16. The exact
  canonical baseline gate passed before R8 inference; the R8 checkpoint SHA-256
  is `7ff3bba0cc416629ee159f1692a221295da322480d1c7b41f06d9eea1c41bb28`,
  and its LoRA state SHA-256 is
  `b3dfce35160cc4c745090f7ec03dd783f86a4c0a82d7566521e7aa21e6a04a04`.
- Frozen composition is R8 + current eslav reference-OCR blend (alpha 0.30,
  text margin 0.05) + selected SIFT fusion (weight 0.40). R16 is not adopted;
  see `reports/encoder_followup_r16_capturev2_last2_report.md`.
- The adaptation cache contains 2,042 embeddings aligned by slug and catalog
  image SHA-256. OCR evidence covers all usable slugs. SIFT cache/config and
  OpenCV version are checked at startup. No benchmark labels enter inference.

### Implemented in this pass

- `src/recognition/smart_retry.py` loads and validates only the frozen runtime,
  runs one-query full-catalog retrieval, then OCR and SIFT Top-5 reranking.
  Retry outcome requires corroboration from a second signal; otherwise the UI
  requests a new photo. It derives barcode/glare/distance/year guidance from
  local image and candidate evidence. Rules are explicitly not probabilities
  and have not been field-calibrated.
- `scripts/serve_smart_retry.py` serves desktop static files plus
  `/api/recognize`, `/api/predict` (flat `{"slug":"..."}`), and `/api/health`.
  The encoder and OCR default to GPU; `--ocr-device cpu` is available for
  constrained GPU memory, but this CPU run exceeded the request's 30 s smoke
  timeout and caused system swapping on this host. It accepts only localhost
  by default, allows up to 20 MB / 40 MP, and does not persist uploaded files.
- `web/index.html`/`web/app.js` no longer expose demo outcome selection or a
  fixed wine card. UI displays live catalog product fields, targeted retry,
  and no-exact-match after two uncertain clean attempts. Match for You excludes
  the actual recognized slug.
- Root/web README and architecture notes describe the current runtime and
  limits. Mobile adaptation remains out of scope. No new training, model
  experiment, or raw-data mutation was started.

### Completion checkpoint

- The frozen runtime is serving locally, and catalog-match, evaluator-contract,
  resolution-retry and unreadable-file responses were manually checked.
- The three post-search flows are implemented: Match for You, Compare, and the
  browser-local wine collection.
- No further implementation step is pending for the agreed desktop MVP. Start
  it again with GPU OCR after stopping the existing service:
  `.venv/bin/python scripts/serve_smart_retry.py --port 8765`.

Open limitation: the user-provided 100-image dataset has no verified slug
labels yet. Its human review is required before making a calibrated open-set
false-accept or no-match claim.

### Integration smoke result — 2026-09-25

- `GET /api/health`: HTTP 200, runtime `validated-r8-ocr-sift`.
- `GET /web/`: HTTP 200.
- `POST /api/recognize` with the catalog reference for `zb-vajn-spumante-bryut-beloe` returned that exact slug and a valid-SIFT corroboration (17 OCR lines, 2 matching signals). First cold query was 8.97 s due to local model/OCR initialization.
- `POST /api/predict` returned exactly `{"slug":"zb-vajn-spumante-bryut-beloe"}` in 1.10 s after warm-up.
- Manual API checks returned `{"status":"retry","reason":"resolution"}` for a 320×240 image and `{"status":"retry","reason":"unreadable"}` for a corrupt image.
- Regression audit on 2026-09-25: `.venv/bin/pytest -q` with the two pretrained
  encoder reproduction cases deselected completed with 334 passed, 2 deselected,
  and 3 sklearn deprecation warnings. A first unfiltered attempt reached 77
  passes, then was interrupted while blocked in `socket.py`; it reported no
  failing tests.
- Static UI audit: `node --check web/app.js`; Python HTML parser passed; every
  JS ID selector exists in `web/index.html` and IDs are unique. Browser-driven
  click-through remains unverified because the computer-use bridge rejected
  the WSL project URI.
- CPU OCR fallback was manually attempted and stopped: it exceeded a 30 s request window and caused swapping on this 10 GB RAM host. GPU OCR is the default; CPU is an opt-in fallback.
- Runtime now performs a one-time catalog-reference warm-up before opening the listener. Thus the API will only announce readiness after cold kernels are initialized. Syntax checks passed; no test suite was run.
- Preview server is running at `http://127.0.0.1:8765/web/` in this workspace. Refresh the open browser tab to see the integrated UI.
