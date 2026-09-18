# REASONING — Phase 2 investigation engine

This is the isolated REASONING component. It never owns a retrieval database: `app/retrieval/interface.py` is the boundary that the future RETRIEVAL HTTP client must implement. The current `MockRetrievalClient` reads `data/documents.json`, whose records use the agreed frozen schema exactly.

## Contracts

`POST /investigate` accepts `{ "question": "..." }` and immediately returns a 202 plus `{ "investigation_id": "..." }`.

EXPERIENCE can display live progress with `GET /investigate/{id}/stream` (SSE). `GET /investigate/{id}/steps` is also supplied as a reliable polling fallback; poll it every 500 ms if an SSE client is inconvenient. `GET /investigate/{id}/report` returns `{ "status": "in_progress" }` until the final structured report is ready.

Every `InvestigationEvent` includes `investigation_id`, `step_type`, `description`, `payload`, and `timestamp`. The final report includes status, cited evidence, confidence factors, open questions, and the very same investigation steps.

## Run

```powershell
cd reasoning
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the generated request/response schemas. Example:

```powershell
$id = (Invoke-RestMethod http://127.0.0.1:8000/investigate -Method POST -ContentType 'application/json' -Body '{"question":"Why did the Orders API become slow on September 16? Was the deployment related?"}').investigation_id
Invoke-RestMethod "http://127.0.0.1:8000/investigate/$id/steps"
Invoke-RestMethod "http://127.0.0.1:8000/investigate/$id/report"
```

## Phase 2 investigation behavior

The engine has a deterministic maximum of five retrieval hops. Each follow-up is tagged to one explicit goal; a duplicate normalized query or a proposal without a targetable open goal ends the loop. The final report records `loop_termination_reason` as `goals_covered`, `max_hops_reached`, or `no_targetable_goal`.

The goal-aware status rule is deliberate:

- `evidence_supported`: every critical goal is supported.
- `insufficient_evidence`: the engine searched but one or more critical goals remain unsupported or unresolved and no useful extra search can be targeted.
- `unresolved`: the hop budget ended with a critical goal still open.

The report only adds Phase-2 fields (`goals`, `loop_termination_reason`, `what_is_known`, and `what_is_not_established`); all Phase-1 fields are retained with their original types.

For the Orders fixture, the three retrieval queries are current incident → versioned deployment → service latency postmortem. `PM-211` is retrieved through the last query, not referenced by the investigator. Its symptom is similar, but the current incident has no established root cause. The comparison was still successfully performed, so the historical-comparison goal is `supported`, the loop ends `goals_covered`, and the overall status is `evidence_supported`. The report keeps the root-cause question open. The timeline keeps PM-211 date-only and never turns DEP-882's temporal precedence into causation.

## Demo replay

First run the demo question once normally; the cache records deterministic semantic-pass responses under `.llm_cache/`. Then run it offline:

```powershell
$env:DEMO_MODE = 'replay'
pytest -q
```

Phase-2 cache keys use the `demo-deterministic-phase2` model name, which versions them separately from the old Phase-1 entries. First run the demo question normally to record its full three-hop flow, then replay it. In replay mode, a missing cache item raises a clear cache-miss error inside the noncritical semantic pass and the investigation continues with a labelled degraded result. No network provider is present in this MVP. Before integrating a real provider, retain temperature `0`, the cache key `(prompt, model, temperature)`, timeout, and one retry in `app/llm/client.py`.

## Concrete trace: Orders API

1. Planner creates five visible goals and starts `orders-api latency incident`.
2. Mock RETRIEVAL returns `INC-1042`; the current-incident goal becomes supported.
3. Version/service clues produce `orders-api v2.8.1 deployment`; `DEP-882` supports only `temporal_precedence`, never causation.
4. The remaining historical goal produces `orders-api latency postmortem`; RETRIEVAL returns `PM-211`.
5. The similarity module emits “Similar symptom, but same root cause is not established.” `comparison_performed` is true, because that evidence-backed negative conclusion is still a completed comparison. The timeline sorts PM-211 (date only), DEP-882, and INC-1042.
6. The historical-comparison goal becomes supported; all critical goals are covered, so the loop ends `goals_covered` and the goal-aware status is `evidence_supported`. The root-cause question remains in `open_questions`.

## Intentional Phase-3 limits

Contradiction detection/resolution and a real RETRIEVAL HTTP adapter are still future work. The on-demand timeline and incident-comparison endpoints remain conservative deterministic helpers.
