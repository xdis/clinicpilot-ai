# ClinicPilot AI - scenario evaluation

Run: 2026-10-28 03:38 against `http://localhost:3001/api`

| Metric | Value |
|---|---|
| cases | 30/30 |
| intent accuracy | 43/43 (100%) |
| checks passed | 112/112 (100%) |
| booking outcome accuracy | 18/18 |
| latency ms p50 | 1189 |
| latency ms p95 | 2243 |
| avg tokens per turn | 2191 |

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
| book-for-family-refused | 2 | pass |
| unknown-doctor | 1 | pass |
| invalid-date | 1 | pass |
| multi-part-with-appointments | 1 | pass |
| cantonese-child-fever | 1 | pass |
| unavailable-specialty | 1 | pass |
