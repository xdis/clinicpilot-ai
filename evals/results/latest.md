# ClinicPilot AI - scenario evaluation

Run: 2026-10-29 07:20 against `http://localhost:3000/api`

| Metric | Value |
|---|---|
| cases | 32/32 |
| intent accuracy | 45/45 (100%) |
| checks passed | 118/118 (100%) |
| booking outcome accuracy | 19/19 |
| latency ms p50 | 1271 |
| latency ms p95 | 2524 |
| avg tokens per turn | 2273 |

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
| history-doctor-notes | 1 | pass |
| slot-conflict | 1 | pass |
