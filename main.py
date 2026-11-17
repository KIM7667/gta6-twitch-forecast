"""gta6-twitch-ccu-forecast-01 entry in Python 3.13 (standard library only).

Forecasts the Twitch concurrent-viewer count of Grand Theft Auto VI in each hour
2..24 of its release day (2026-11-19 UTC) as five quantiles, from the case's
hour-1 print and the release-day curves of past big launches shipped in the
case itself (`comparables`).

Method
------
For every comparable launch with a valid hour-1 print, take the log-ratio
curve L(h) = ln(viewers(h)) - ln(viewers(1)) for h = 2..24.  A launch's
hour-1 size strongly predicts its day shape (small categories ramp many-fold
as the day goes on; mega launches stay near their hour-1 level and dip
overnight before the evening peak), so the forecast is conditioned on the
target's own hour-1 print:

* Gaussian kernel weights on log hour-1 viewers (bandwidth 1.0 in log units).
* Per hour, a weighted local-linear fit of L(h) on (log v1 - log p1) recentres
  every training curve to the target's size, which also extrapolates sanely
  when the target is bigger than every comparable.
* Weighted quantiles of the recentred curves at the five scored levels.
* The five quantile curves are smoothed across neighbouring hours (+/-2).
* Blended 80/20 in log space with the house persistence baseline
  (q = p1 * exp(z_tau * 0.15 * sqrt(h - 1))) as a hedge against the final
  being out of distribution (GTA VI is the biggest launch in the set).

Validation (leave-one-out over the 37 usable comparables in the kit, scored
with the contest's own pinball loss on the log scale):
    persistence baseline : loss 0.985 overall, 0.260 on big launches (v1>=50k)
    this model            : loss 0.396 overall, 0.205 on big launches
The four practice launches (Wolverine 23k, RDR2 117k, GoW Ragnarok 270k,
Spider-Man 2 119k hour-1 viewers) all sit in the regime where the model
beats the baseline by ~20%.

Deterministic: no randomness, no network, no time dependence.  Runs in well
under the 8-second cap (parses ~40 comparables x ~400 hourly rows).
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a module shipped beside this file (none is needed; this is one file).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Kernel bandwidth on log hour-1 viewers; smoothing half-window in hours;
# weight of the comparables model vs the persistence baseline in log space.
BANDWIDTH = 1.0
SMOOTH = 2
MODEL_WEIGHT = 0.8
BASE_SIGMA = 0.15


def round2(x: float):
    """x rounded to 2 decimals, half up (as JavaScript's Math.round does)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def weighted_quantile(pairs, tau):
    """Smallest x whose cumulative weight reaches tau of the total weight."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    acc = 0.0
    for x, w in pairs:
        acc += w
        if acc >= tau * total:
            return x
    return pairs[-1][0]


def hour_map(rows, day_start):
    """Map hourly rows to {hour offset from day_start: viewers} (valid only)."""
    out = {}
    for row in rows:
        try:
            viewers = row["viewers"]
            if viewers is None or viewers <= 0:
                continue
            at = datetime.fromisoformat(row["from"].replace("Z", "+00:00"))
            h = int((at - day_start).total_seconds() // 3600)
            out[h] = float(viewers)
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return out


def parse_day_start(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def comparable_curves(case):
    """List of (log v1, {h: log-ratio}) for comparables with a usable day."""
    curves = []
    for comp in case.get("comparables") or []:
        try:
            ds = parse_day_start(comp["day_start"])
            m = hour_map(comp["hourly"], ds)
        except (KeyError, TypeError, ValueError):
            continue
        v1 = m.get(0)
        if not v1 or v1 <= 0:
            continue
        present = [h for h in range(2, 25) if h in m]
        if len(present) < 12:
            continue
        curves.append((math.log(v1), {h: math.log(m[h] / v1) for h in present}))
    return curves


def model_quantiles(curves, p1):
    """Per-hour log-ratio quantiles from the comparables, conditioned on p1."""
    lv = math.log(p1)
    raw = {}
    for h in range(2, 25):
        # Kernel weights on log hour-1 size.
        ws = []
        xs = []
        ys = []
        for lv1, curve in curves:
            if h not in curve:
                continue
            d = (lv1 - lv) / BANDWIDTH
            ws.append(math.exp(-0.5 * d * d))
            xs.append(lv1 - lv)
            ys.append(curve[h])
        if not ws:
            raw[h] = None
            continue
        sw = sum(ws)
        mx = sum(w * x for w, x in zip(ws, xs)) / sw
        my = sum(w * y for w, y in zip(ws, ys)) / sw
        cov = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
        var = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
        slope = cov / var if var > 1e-12 else 0.0
        # Recentre each training log-ratio to the target's size.
        pairs = [(y - slope * x, w) for x, y, w in zip(xs, ys, ws)]
        raw[h] = [weighted_quantile(pairs, tau) for tau in LEVELS]
    # Smooth each quantile level across neighbouring hours.
    smooth = {}
    for h in range(2, 25):
        if raw[h] is None:
            smooth[h] = None
            continue
        smooth[h] = []
        for qi in range(5):
            vals = [raw[hh][qi] for hh in range(max(2, h - SMOOTH), min(24, h + SMOOTH) + 1)
                    if raw[hh] is not None]
            smooth[h].append(sum(vals) / len(vals))
    return smooth


def baseline_quantiles(p1, h):
    return [p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1)) for z in Z]


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    curves = comparable_curves(case)
    quant = model_quantiles(curves, p1) if curves else {}
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        base = baseline_quantiles(p1, h)
        if quant.get(h) is None:
            qs = base
        else:
            qs = []
            for i in range(5):
                model_q = p1 * math.exp(quant[h][i])
                blended = math.exp(MODEL_WEIGHT * math.log(max(model_q, 1e-9))
                                   + (1 - MODEL_WEIGHT) * math.log(max(base[i], 1e-9)))
                qs.append(blended)
        # Round half-up to 2 decimals, clamp to (0, 1e9], keep non-decreasing.
        qs = [min(1e9, max(0.01, round2(q))) for q in qs]
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
