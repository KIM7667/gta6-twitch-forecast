# GTA VI launch-day Twitch forecast

Entry for [make.fun bounty `gta6-twitch-ccu-forecast-01`](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
forecast how many people watch Grand Theft Auto VI live on Twitch in each
hour 2–24 of its release day (2026-11-19 UTC), as five quantiles per hour,
from the case's hour-1 print. Pure Python 3.13 standard library, no network,
deterministic, runs in ~0.2 s (cap: 8 s).

## Method

`main.py` reads the case file (`/input/<id>.in`) and, for every comparable
past launch shipped in the case, takes the log-ratio curve
`L(h) = ln(viewers(h)) − ln(viewers(1))`, `h = 2..24`.

A launch's hour-1 size strongly predicts its day shape (small categories
ramp many-fold as the day goes on; mega launches stay near hour-1 level, dip
overnight, then peak in the evening), so the forecast is conditioned on the
target's own hour-1 print:

1. Gaussian kernel weights on log hour-1 viewers (bandwidth 1.0).
2. Per hour, a weighted local-linear fit of `L(h)` on `(log v1 − log p1)`
   recentres every training curve to the target's size — this also
   extrapolates sanely when the target is bigger than every comparable.
3. Weighted quantiles of the recentred curves at the five scored levels
   (0.05, 0.25, 0.5, 0.75, 0.95).
4. Quantile curves smoothed across neighbouring hours (±2).
5. Blended 80/20 in log space with the house persistence baseline
   (`q = p1 · exp(z_τ · 0.15 · √(h−1))`) as a hedge against the final being
   out of distribution — GTA VI is the biggest launch in the set.

## Validation

Leave-one-out over the 37 usable comparables in the kit, scored with the
contest's own pinball loss on the log scale:

| model | all 37 | big launches (v1 ≥ 50k) |
|---|---|---|
| persistence baseline | 0.985 | 0.260 |
| this model | 0.396 | 0.205 |

Held-out spot checks with the shipped `main.py` (target removed from its own
comparables): ELDEN RING 0.091 vs baseline 0.218, Cyberpunk 2077 0.143 vs
0.251, Zelda: Tears of the Kingdom 0.283 vs 0.318, Hollow Knight: Silksong
0.930 vs 4.207.

Local test:

```
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Two runs are byte-identical. Output is 23 rows (hours 2–24), five finite,
positive, non-decreasing quantiles each, under the 65,536-byte limit.
