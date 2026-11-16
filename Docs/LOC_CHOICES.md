# LOC choices (notebook 04, LOC stage)

- id (sha256 of the choices JSON): `e4d69acf633b817fd6153df2e2bb5e33260ae80bac096a2580db04a3ca0eef94`
- written: 2026-11-15 12:21:00
- core commit: `026f3e7702331e3a2ed4c9564e938d87d31cc790`
- run: `v4_last168`

Every choice the causal stage makes is in the JSON this id refers to (`reports/04_loc_choices.json` on Drive): FIT class boundaries, pair counts, per-model control bands and decision thresholds, the heads that passed on LOC, the sign of log-priority per head, the tertile cut points and the DAS layer.
CONF-A is opened only when this file is committed as `Docs/LOC_CHOICES.md` and its id matches the JSON. Nothing below is changed after that.

| model | d: threshold / heads passing | c: threshold / heads passing | lam: threshold / heads passing |
|---|---|---|---|
| `sigmoid_s1042` | 0.5126 / 16 | 0.0500 / 2 | 0.0500 / 0 |
| `sigmoid_s2042` | 0.4917 / 18 | 0.0500 / 5 | 0.0500 / 0 |
| `sigmoid_s3042` | 0.4358 / 22 | 0.0500 / 4 | 0.0500 / 0 |
| `sigmoid_s4042` | 0.4206 / 12 | 0.0500 / 5 | 0.0500 / 0 |
| `sigmoid_s42` | 0.4917 / 17 | 0.0500 / 2 | 0.0500 / 0 |
| `softmax_s1042` | 0.4167 / 20 | 0.0500 / 9 | 0.0500 / 0 |
| `softmax_s2042` | 0.4635 / 20 | 0.0500 / 11 | 0.0500 / 0 |
| `softmax_s3042` | 0.3891 / 22 | 0.0500 / 9 | 0.0500 / 0 |
| `softmax_s4042` | 0.3466 / 19 | 0.0500 / 15 | 0.0500 / 0 |
| `softmax_s42` | 0.4194 / 17 | 0.0500 / 12 | 0.0500 / 0 |
