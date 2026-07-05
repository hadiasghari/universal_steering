# Experiment run log

| # | date | machine | tag | model | config | result | notes |
|---|------|---------|-----|-------|--------|--------|-------|
| 001 | jul4 | local | (untagged, 4d outputs) | llama-8B | judge shootout on fixed 360 items | oss 27/45, sonnet 26/45, haiku 13/45 (v1) | haiku rejected (strict outlier); oss~sonnet kappa .72/.78 |
| 002 | jul4-5 | pod-L40S | Lall_K1ev_def | llama-8B | STAGE2: re-extract 90 (orig frames) + all-layer K=1 + gpt-oss judge | v1 57/90 (63%), v4 43/90 (48%) | PASS: matches original Mac run (55/90, 43/90); extraction cos .99 vs direction3e. First attempt corrupted by unnormalized-evals bug (fixed, commit in log) |
| 003 | -- | pod | -- | llama-8B | STAGE3 (planned): same 90, NEW frames (dual-frame 50/50 + quoted concept) | -- | measures frame effect before 600-run |
