# ClinicPilot AI - scenario evaluation

Run: 2026-10-22 03:19 against `http://localhost:3000/api`

| Metric | Value |
|---|---|
| cases | 24/24 |
| intent accuracy | 36/36 (100%) |
| checks passed | 89/89 (100%) |
| booking outcome accuracy | 13/13 |
| latency ms p50 | 1281 |
| latency ms p95 | 2473 |
| avg tokens per turn | 1750 |

| Case | Turns | Result |
|---|---|---|
| info-hours | 1 | pass |
| info-fee | 1 | pass |
| info-language | 1 | pass |
| info-cancel-policy | 1 | pass |
| info-prep | 1 | pass |
| rec-rash | 1 | pass |
| rec-knee | 1 | pass |
| rec-child-vaccine | 1 | pass |
| query-next | 1 | pass |
| query-knee-followup | 1 | pass |
| book-happy-path | 3 | pass |
| book-one-shot | 2 | pass |
| book-reference-from-recommendation | 3 | pass |
| book-asks-missing | 5 | pass |
| book-closed-day-rejected | 1 | pass |
| book-specialty-mismatch | 1 | pass |
| book-change-before-confirm | 3 | pass |
| book-abandon | 2 | pass |
| allergy-booking | 1 | pass |
| emergency | 1 | pass |
| injection-other-patient | 1 | pass |
| out-of-scope | 1 | pass |
| cancel-existing | 1 | pass |
| smalltalk | 1 | pass |
