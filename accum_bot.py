"""
ACCUMULATOR Self-Calibrating + XGBoost + AI ADVISOR Bot
═══════════════════════════════════════════════════════════════════════════════
Phase 1  COLLECT (30 minutes rolling — recalibrates every 30 min while trading)
  • Subscribes to 1HZ10V, saves per-symbol CSVs (rolling 2h max window)
  • BarrierSimulator replays ticks to find best growth_rate (1/2/3/5%)
  • survival_target_ticks, knock_out_rate, survival_p50 derived per symbol

Phase 2  TRADE (starts automatically after Phase 1)
  • 5-condition confluence gate (C1-C5) — recalibrated for accumulator risk
      C1  sigma_ewma  < sigma_gate    (p20 — tighter than EXPIRYRANGE p35)
      C2  range_20    < range_gate    (p40)
      C3  |ema_gap|   < ema_gate
      C4  |Z|         < z_gate
      C5  spike_10    < spike_gate    (p85)
      C6  regime NOT TRENDING or CHAOS (hard block)
  • 3-Layer Ensemble gate (C7): 2-of-3 model vote
      L1 XGBoost, L2 LogReg, L3 IsoForest — trained on survival labels
  • ActiveContractMonitor: per-tick exit engine while contract is open
      E1 ticks >= target_ticks         (profit target reached)
      E2 sigma rising above exit gate  (volatility spike mid-contract)
      E3 spike detected                (immediate danger)
      E4 regime shift to TRENDING/CHAOS
      E5 payout >= ratchet target      (lock in profit)
      E6 max_hold_ticks ceiling
  • TakeProfitEngine: sliding payout ratchet calibrated from survival_p50
  • GrowthRateSelector: picks 1/2/3/5% from Phase 1 barrier simulation
  • KnockOutTracker: logs every knock-out with entry context → Supabase
  • AIAdvisor: L1-L9 reasoning layers, hot-swaps all parameters live

Railway + Supabase deployment:
  • pip install xgboost websockets supabase scikit-learn
  • SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET=botdata
  • DERIV_API_TOKEN
  • No Volume needed — all state persists in Supabase Storage

Run:
    python accum_bot.py                # full run
    python accum_bot.py --collect-only # Phase 1 only
    python accum_bot.py --trade-only   # Phase 2 only (needs calibration.json)
"""

import asyncio
import csv
import json
import logging
import math
import os
import pickle
import sys
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Optional, Tuple
import threading

try:
    import websockets
    from websockets.exceptions import (
        ConnectionClosed, ConnectionClosedError, ConnectionClosedOK,
    )
except ImportError:
    sys.exit("pip install websockets")

try:
    from supabase import create_client as _supa_create
    _SUPA_AVAILABLE = True
except ImportError:
    _SUPA_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

API_TOKEN   = os.getenv("DERIV_API_TOKEN", "")
APP_ID      = os.getenv("DERIV_APP_ID",    "1089")
WS_URL      = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"

COLLECT_MINS      = float(os.getenv("COLLECT_MINS", "30"))
COLLECT_SECS      = COLLECT_MINS * 60
ROLLING_MAX_HOURS = float(os.getenv("ROLLING_MAX_HOURS", "2"))
ROLLING_MAX_SECS  = ROLLING_MAX_HOURS * 3600

SUPABASE_URL    = os.getenv("SUPABASE_URL",    "")
SUPABASE_KEY    = os.getenv("SUPABASE_KEY",    "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "botdata")

_PERSIST_DIR = os.getenv("PERSIST_DIR", os.path.join("/tmp", "accum_botdata"))
os.makedirs(_PERSIST_DIR, exist_ok=True)
CAL_FILE      = os.path.join(_PERSIST_DIR, "calibration.json")
DATA_DIR      = os.path.join(_PERSIST_DIR, "symbol_data")
ADVISOR_LOG   = os.path.join(_PERSIST_DIR, "advisor_log.txt")
KNOCKOUT_LOG  = os.path.join(_PERSIST_DIR, "knockout_log.csv")
SURVIVAL_FILE = os.path.join(_PERSIST_DIR, "survival_stats.json")
GROWTH_HIST   = os.path.join(_PERSIST_DIR, "growth_rate_history.json")
PORT          = int(os.getenv("PORT", "8080"))

SURVEY_SYMBOLS = ["1HZ10V"]

BASE_STAKE       = float(os.getenv("BASE_STAKE",    "1.0"))
MARTINGALE_MULT  = float(os.getenv("MARTI_MULT",    "2.0"))
MARTINGALE_STEPS = int(os.getenv("MARTI_STEPS",     "1"))
LOSS_COOLDOWN    = float(os.getenv("LOSS_COOLDOWN", "60"))
TARGET_PROFIT    = float(os.getenv("TARGET_PROFIT", "20.0"))
STOP_LOSS        = float(os.getenv("STOP_LOSS",     "10.0"))
LOCK_TIMEOUT     = 300

GROWTH_RATE     = int(os.getenv("GROWTH_RATE",     "2"))
TARGET_TICKS    = int(os.getenv("TARGET_TICKS",    "15"))
MAX_HOLD_TICKS  = int(os.getenv("MAX_HOLD_TICKS",  "40"))
PAYOUT_TARGET_1 = float(os.getenv("PAYOUT_TARGET_1","1.30"))
PAYOUT_TARGET_2 = float(os.getenv("PAYOUT_TARGET_2","1.80"))
PAYOUT_TARGET_3 = float(os.getenv("PAYOUT_TARGET_3","2.50"))

XGB_THRESHOLD     = float(os.getenv("XGB_THRESHOLD",    "0.70"))
LR_THRESHOLD      = float(os.getenv("LR_THRESHOLD",     "0.72"))
ISO_CONTAMINATION = float(os.getenv("ISO_CONTAMINATION","0.15"))
EXIT_SIGMA_MULT   = float(os.getenv("EXIT_SIGMA_MULT",  "1.8"))
EXIT_SPIKE_MULT   = float(os.getenv("EXIT_SPIKE_MULT",  "1.5"))

CANDLE_GRAN_1 = 60
CANDLE_GRAN_5 = 300
CANDLE_COUNT  = 20

SAFE_BOUNDS = {
    "sigma_gate":       (0.04,  0.20,  0.015),
    "range_gate":       (0.10,  1.00,  0.080),
    "ema_gate":         (0.04,  0.50,  0.040),
    "z_gate":           (0.30,  2.00,  0.200),
    "spike_gate":       (0.04,  0.50,  0.040),
    "growth_rate":      (1,     5,     1),
    "target_ticks":     (6,     40,    5),
    "exit_sigma_mult":  (1.2,   3.0,   0.2),
    "exit_spike_mult":  (1.2,   3.0,   0.2),
    "base_stake":       (0.35,  2.00,  0.35),
    "martingale_steps": (0,     2,     1),
    "loss_cooldown":    (15,    180,   15),
    "payout_target_1":  (1.10,  1.60,  0.10),
    "payout_target_2":  (1.40,  2.50,  0.20),
    "payout_target_3":  (2.00,  4.00,  0.50),
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("accum_bot")

def info(m):  log.info(m)
def warn(m):  log.warning(m)
def err(m):   log.error(m)
def tlog(m):  log.info(f"[TRADE] {m}")

# ─────────────────────────────────────────────────────────────────────────────
# SUPABASE STORAGE
# ─────────────────────────────────────────────────────────────────────────────

class SupabaseStore:
    def __init__(self):
        self._client = None
        self._bucket = SUPABASE_BUCKET
        self._enabled = False
        if not _SUPA_AVAILABLE:
            warn("[SUPA] supabase not installed — local-only mode")
            return
        if not SUPABASE_URL or not SUPABASE_KEY:
            warn("[SUPA] SUPABASE_URL/KEY not set — local-only mode")
            return
        try:
            self._client  = _supa_create(SUPABASE_URL, SUPABASE_KEY)
            self._enabled = True
            info(f"[SUPA] Connected  bucket={self._bucket}")
        except Exception as exc:
            warn(f"[SUPA] Init failed: {exc} — local-only mode")

    @property
    def enabled(self): return self._enabled

    def upload(self, local_path: str, remote_key: str = None):
        if not self._enabled or not os.path.exists(local_path): return
        if remote_key is None:
            remote_key = os.path.relpath(local_path, _PERSIST_DIR)
        try:
            with open(local_path, "rb") as f: data = f.read()
            try:
                self._client.storage.from_(self._bucket).update(
                    remote_key, data,
                    file_options={"content-type": "application/octet-stream",
                                  "upsert": "true"})
            except Exception:
                self._client.storage.from_(self._bucket).upload(
                    remote_key, data,
                    file_options={"content-type": "application/octet-stream",
                                  "upsert": "true"})
            info(f"[SUPA] ↑ {remote_key} ({len(data):,}b)")
        except Exception as exc:
            warn(f"[SUPA] upload {remote_key}: {exc}")

    def download(self, remote_key: str, local_path: str) -> bool:
        if not self._enabled: return False
        try:
            data = self._client.storage.from_(self._bucket).download(remote_key)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f: f.write(data)
            info(f"[SUPA] ↓ {remote_key} ({len(data):,}b)")
            return True
        except Exception as exc:
            if "not found" in str(exc).lower() or "404" in str(exc):
                return False
            warn(f"[SUPA] download {remote_key}: {exc}")
            return False

    def restore(self):
        if not self._enabled: return
        KNOWN = [
            "calibration.json", "advisor_log.txt", "survival_stats.json",
            "growth_rate_history.json", "knockout_log.csv",
            "xgb_model.json", "lr_model.pkl", "iso_model.pkl", "gb_model.pkl",
        ]
        try:
            items = self._client.storage.from_(self._bucket).list("symbol_data")
            for item in (items or []):
                if item.get("name", "").endswith(".csv"):
                    KNOWN.append(f"symbol_data/{item['name']}")
        except Exception: pass
        restored = 0
        for key in KNOWN:
            if self.download(key, os.path.join(_PERSIST_DIR, key)):
                restored += 1
        info(f"[SUPA] Restore complete — {restored}/{len(KNOWN)} files pulled")

    def push_all(self):
        if not self._enabled: return
        for root, _, files in os.walk(_PERSIST_DIR):
            for fname in files:
                local = os.path.join(root, fname)
                self.upload(local, os.path.relpath(local, _PERSIST_DIR))

_store = SupabaseStore()

# ─────────────────────────────────────────────────────────────────────────────
# SYMBOL STATS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class SymbolStats:
    EWMA_ALPHA = 0.05

    def __init__(self, symbol: str):
        self.symbol      = symbol
        self.tick_n      = 0
        self.prices: deque = deque(maxlen=500)
        self.sigma_ewma  = None
        self._regime_counts = {"CALM":0,"RANGING":0,"TRENDING":0,"CHAOS":0}
        self._regime_start  = time.time()
        self._regime_cur    = "CALM"
        self._ema7 = self._ema14 = None
        self._k7  = 2/(7+1); self._k14 = 2/(14+1)

        os.makedirs(DATA_DIR, exist_ok=True)
        fname = os.path.join(DATA_DIR, f"{symbol}.csv")
        file_exists = os.path.exists(fname) and os.path.getsize(fname) > 0
        self._csv_f = open(fname, "a", newline="")
        self._csv_w = csv.DictWriter(self._csv_f, fieldnames=self._fields())
        if not file_exists:
            self._csv_w.writeheader()
        self._rows_since_flush = 0

    @staticmethod
    def _fields():
        return ["ts","epoch","symbol","tick_n","price","tick_delta",
                "tick_abs_delta","sigma_ewma","range_20","range_50",
                "ema7","ema14","ema_gap","zscore_50","spike_10","atr_14",
                "entropy_20","regime"]

    def update(self, price: float, epoch: float) -> dict:
        self.tick_n += 1
        prev       = self.prices[-1] if self.prices else price
        delta      = price - prev
        abs_delta  = abs(delta)
        self.prices.append(price)

        if self.sigma_ewma is None: self.sigma_ewma = abs_delta
        else: self.sigma_ewma = (self.EWMA_ALPHA * abs_delta +
                                  (1 - self.EWMA_ALPHA) * self.sigma_ewma)

        if self._ema7 is None: self._ema7 = self._ema14 = price
        else:
            self._ema7  = price * self._k7  + self._ema7  * (1 - self._k7)
            self._ema14 = price * self._k14 + self._ema14 * (1 - self._k14)
        ema_gap = abs(self._ema7 - self._ema14)

        prices   = list(self.prices)
        range_20 = (max(prices[-20:]) - min(prices[-20:])) if len(prices) >= 20 else 0
        range_50 = (max(prices[-50:]) - min(prices[-50:])) if len(prices) >= 50 else 0

        zscore_50 = 0.0
        if len(prices) >= 200:
            baseline = prices[-200:]
            mu  = sum(baseline) / 200
            var = sum((p - mu) ** 2 for p in baseline) / 200
            std = math.sqrt(var) if var > 0 else 1e-9
            zscore_50 = (sum(prices[-50:]) / 50 - mu) / (std / math.sqrt(50))

        moves    = [abs(prices[i] - prices[i-1]) for i in range(-10, 0)
                    if i - 1 >= -len(prices)]
        spike_10 = max(moves) if moves else 0
        atr_mv   = [abs(prices[i] - prices[i-1]) for i in range(-14, 0)
                    if i - 1 >= -len(prices)]
        atr_14   = sum(atr_mv) / len(atr_mv) if atr_mv else 0
        entropy_20 = self._entropy(prices[-21:]) if len(prices) >= 21 else 1.0
        regime   = self._detect_regime(ema_gap, self.sigma_ewma, zscore_50)

        if regime != self._regime_cur:
            self._regime_counts[self._regime_cur] += time.time() - self._regime_start
            self._regime_cur   = regime
            self._regime_start = time.time()

        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "epoch": epoch, "symbol": self.symbol, "tick_n": self.tick_n,
            "price": round(price, 5), "tick_delta": round(delta, 5),
            "tick_abs_delta": round(abs_delta, 5),
            "sigma_ewma": round(self.sigma_ewma, 5),
            "range_20": round(range_20, 4), "range_50": round(range_50, 4),
            "ema7": round(self._ema7, 5), "ema14": round(self._ema14, 5),
            "ema_gap": round(ema_gap, 5), "zscore_50": round(zscore_50, 4),
            "spike_10": round(spike_10, 5), "atr_14": round(atr_14, 5),
            "entropy_20": round(entropy_20, 4), "regime": regime,
        }
        self._csv_w.writerow(row)
        self._rows_since_flush += 1
        if self._rows_since_flush >= 100:
            self._csv_f.flush()
            self._rows_since_flush = 0
            _store.upload(os.path.join(DATA_DIR, f"{self.symbol}.csv"),
                          f"symbol_data/{self.symbol}.csv")
        return row

    @staticmethod
    def _entropy(prices):
        moves = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        if not moves: return 1.0
        mx = max(moves) or 1; buckets = [0] * 5
        for m in moves: buckets[min(4, int(m / mx * 4))] += 1
        n = len(moves); H = 0.0
        for b in buckets:
            if b > 0: p = b / n; H -= p * math.log2(p)
        return H / math.log2(5)

    @staticmethod
    def _detect_regime(ema_gap, sigma, zscore):
        if abs(zscore) > 2.5 or sigma > 0.3: return "CHAOS"
        if abs(zscore) > 1.5 and ema_gap > 0.3: return "TRENDING"
        if abs(zscore) < 1.0 and ema_gap < 0.15: return "CALM"
        return "RANGING"

    def summarise(self):
        self._regime_counts[self._regime_cur] += time.time() - self._regime_start
        total = sum(self._regime_counts.values()) or 1
        return {
            "symbol": self.symbol, "ticks": self.tick_n,
            "regime_pct": {k: round(v/total, 4) for k, v in self._regime_counts.items()},
            "data_file":  os.path.join(DATA_DIR, f"{self.symbol}.csv"),
        }

    def close(self):
        self._csv_f.flush(); self._csv_f.close()
        _store.upload(os.path.join(DATA_DIR, f"{self.symbol}.csv"),
                      f"symbol_data/{self.symbol}.csv")


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING CSV TRIM
# ─────────────────────────────────────────────────────────────────────────────

def rolling_csv_trim(symbol: str):
    fpath = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(fpath): return
    cutoff = time.time() - ROLLING_MAX_SECS
    kept = []; removed = 0; fields = None
    with open(fpath, newline="") as f:
        reader = csv.DictReader(f); fields = reader.fieldnames
        for row in reader:
            try:
                ts = row["ts"].replace("Z", "+00:00")
                if datetime.fromisoformat(ts).timestamp() >= cutoff:
                    kept.append(row)
                else:
                    removed += 1
            except Exception: kept.append(row)
    if removed == 0: return
    tmp = fpath + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(kept)
    os.replace(tmp, fpath)
    _store.upload(fpath, f"symbol_data/{symbol}.csv")
    info(f"[ROLLING] {symbol}: trimmed {removed}  kept={len(kept)}")


# ─────────────────────────────────────────────────────────────────────────────
# BARRIER SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class BarrierSimulator:
    """
    Simulates accumulator survival across growth rates using Phase 1 ticks.
    Deriv accumulator: barrier grows each tick as
        barrier_t = initial_barrier × (1 + growth_rate/100)^t
    Knock-out when |price - entry| >= barrier_t.
    """
    GROWTH_RATES = [1, 2, 3, 5]

    @staticmethod
    def _initial_barrier(spot: float, growth_rate: int) -> float:
        rate_map = {1: 0.0120, 2: 0.0065, 3: 0.0045, 5: 0.0028}
        return spot * rate_map.get(growth_rate, 0.006)

    @classmethod
    def simulate(cls, prices: list, growth_rate: int,
                 max_ticks: int = 60) -> dict:
        if len(prices) < max_ticks + 5:
            return {"growth_rate": growth_rate, "survival_rate": 0.0,
                    "median_ticks": 0, "ko_rate": 1.0,
                    "p_survive_10": 0.0, "p_survive_20": 0.0,
                    "p_survive_30": 0.0, "expected_value": 0.0, "n_simulated": 0}

        ticks_survived = []; n_entries = 0
        for entry_idx in range(0, len(prices) - max_ticks, 3):
            entry = prices[entry_idx]
            b0    = cls._initial_barrier(entry, growth_rate)
            surv  = 0
            for t in range(1, max_ticks + 1):
                if entry_idx + t >= len(prices): break
                bt = b0 * ((1 + growth_rate / 100) ** t)
                if abs(prices[entry_idx + t] - entry) >= bt: break
                surv = t
            ticks_survived.append(surv); n_entries += 1

        if not ticks_survived:
            return {"growth_rate": growth_rate, "survival_rate": 0.0,
                    "median_ticks": 0, "ko_rate": 1.0,
                    "p_survive_10": 0.0, "p_survive_20": 0.0,
                    "p_survive_30": 0.0, "expected_value": 0.0,
                    "n_simulated": 0}

        n          = len(ticks_survived)
        median_t   = sorted(ticks_survived)[n // 2]
        ko_rate    = sum(1 for t in ticks_survived if t < max_ticks) / n
        surv_rate  = 1.0 - ko_rate
        p10  = sum(1 for t in ticks_survived if t >= 10) / n
        p20  = sum(1 for t in ticks_survived if t >= 20) / n
        p30  = sum(1 for t in ticks_survived if t >= 30) / n
        payout_mult = (1 + growth_rate / 100) ** median_t
        ev   = surv_rate * payout_mult - ko_rate

        return {
            "growth_rate":    growth_rate,
            "survival_rate":  round(surv_rate, 4),
            "median_ticks":   median_t,
            "ko_rate":        round(ko_rate, 4),
            "p_survive_10":   round(p10, 4),
            "p_survive_20":   round(p20, 4),
            "p_survive_30":   round(p30, 4),
            "expected_value": round(ev, 4),
            "n_simulated":    n_entries,
        }

    @classmethod
    def best_rate(cls, prices: list, max_ticks: int = 30) -> dict:
        """
        max_ticks: realistic hold horizon for simulation.
        Kept at 30 (not MAX_HOLD_TICKS=40) so knock-outs are visible
        even in calm markets — prevents all-survival single-class labels.
        """
        results = {}
        for gr in cls.GROWTH_RATES:
            results[gr] = cls.simulate(prices, gr, max_ticks=max_ticks)
            info(f"[SIM] growth={gr}%  ko_rate={results[gr]['ko_rate']:.1%}  "
                 f"median_ticks={results[gr]['median_ticks']}  "
                 f"ev={results[gr]['expected_value']:+.4f}")
        valid = {gr: r for gr, r in results.items()
                 if r["survival_rate"] >= 0.50}
        if not valid: valid = results
        best_gr = max(valid, key=lambda g: valid[g]["expected_value"])
        info(f"[SIM] Best: {best_gr}%  "
             f"survival={results[best_gr]['survival_rate']:.1%}  "
             f"EV={results[best_gr]['expected_value']:+.4f}")
        return {"best": best_gr, "all": results}

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_calibration(summaries: List[dict]) -> dict:
    info("Computing accumulator calibration...")
    symbol_scores = {}

    for s in summaries:
        sym   = s["symbol"]
        fpath = s["data_file"]
        if not os.path.exists(fpath): continue

        sigmas, ranges, ema_gaps, zscores, spikes, prices_all = [], [], [], [], [], []
        with open(fpath, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    sigmas.append(float(row["sigma_ewma"]))
                    ranges.append(float(row["range_20"]))
                    ema_gaps.append(float(row["ema_gap"]))
                    zscores.append(abs(float(row["zscore_50"])))
                    spikes.append(float(row["spike_10"]))
                    prices_all.append(float(row["price"]))
                except (ValueError, KeyError): continue

        if len(sigmas) < 100:
            warn(f"{sym}: only {len(sigmas)} rows — skipping"); continue

        sigmas.sort(); ranges.sort(); ema_gaps.sort()
        zscores.sort(); spikes.sort()

        def pct(lst, p):
            idx = max(0, int(len(lst) * p / 100) - 1)
            return lst[idx]

        sigma_p20  = pct(sigmas,   20)
        range_p40  = pct(ranges,   40)
        ema_p50    = pct(ema_gaps, 50)
        z_p50      = pct(zscores,  50)
        spike_p85  = pct(spikes,   85)

        sim_result = BarrierSimulator.best_rate(prices_all, max_ticks=30)
        best_gr    = sim_result["best"]
        sim_stats  = sim_result["all"][best_gr]

        target_ticks = max(6, min(sim_stats["median_ticks"], MAX_HOLD_TICKS))
        calm_pct     = (s["regime_pct"].get("CALM", 0) +
                        s["regime_pct"].get("RANGING", 0))
        score        = calm_pct * sim_stats["survival_rate"]

        p10 = sim_stats["p_survive_10"]
        p20 = sim_stats["p_survive_20"]
        p30 = sim_stats["p_survive_30"]
        # Ratchet targets based on realistic compounded payout at each tier.
        # For 5% growth: 1.05^10=1.63, 1.05^20=2.65, 1.05^30=4.32
        # We set targets at 85% of theoretical max to give room to exit.
        r1_theoretical = (1 + best_gr/100) ** 10
        r2_theoretical = (1 + best_gr/100) ** 20
        r3_theoretical = (1 + best_gr/100) ** 30
        ratchet1 = round(max(1.20, r1_theoretical * 0.85 * p10 + 1.0 * (1-p10)), 2)
        ratchet2 = round(max(1.60, r2_theoretical * 0.85 * p20 + 1.0 * (1-p20)), 2)
        ratchet3 = round(max(2.20, r3_theoretical * 0.85 * p30 + 1.0 * (1-p30)), 2)
        # Hard floor: ratchet1 must be > what growth compounds to by tick 5
        # so E5 doesn't fire in the first few ticks
        min_r1 = round((1 + best_gr/100) ** max(5, target_ticks // 4) * 1.05, 2)
        ratchet1 = max(ratchet1, min_r1)

        entry = {
            "symbol":          sym,
            "ticks":           s["ticks"],
            "score":           round(score, 4),
            "calm_pct":        round(calm_pct, 4),
            "growth_rate":     best_gr,
            "survival_rate":   sim_stats["survival_rate"],
            "ko_rate":         sim_stats["ko_rate"],
            "median_ticks":    sim_stats["median_ticks"],
            "target_ticks":    target_ticks,
            "p_survive_10":    p10,
            "p_survive_20":    p20,
            "p_survive_30":    p30,
            "expected_value":  sim_stats["expected_value"],
            "sigma_gate":      round(sigma_p20, 5),
            "range_gate":      round(range_p40, 4),
            "ema_gate":        round(ema_p50 * 0.80, 4),
            "z_gate":          round(max(0.5, min(z_p50, 1.5)), 4),
            "spike_gate":      round(spike_p85, 5),
            "exit_sigma_gate": round(sigma_p20 * EXIT_SIGMA_MULT, 5),
            "exit_spike_gate": round(spike_p85  * EXIT_SPIKE_MULT, 5),
            "exit_sigma_mult": EXIT_SIGMA_MULT,
            "exit_spike_mult": EXIT_SPIKE_MULT,
            "payout_target_1": ratchet1,
            "payout_target_2": ratchet2,
            "payout_target_3": ratchet3,
            "regime_pct":      s["regime_pct"],
        }
        symbol_scores[sym] = entry

        info(f"  {sym}: score={score:.4f}  growth={best_gr}%  "
             f"survival={sim_stats['survival_rate']:.1%}  "
             f"target={target_ticks}t  "
             f"sigma_gate={sigma_p20:.5f}  spike_gate={spike_p85:.5f}")
        info(f"    ratchet ×{ratchet1} → ×{ratchet2} → ×{ratchet3}")

    if not symbol_scores:
        raise RuntimeError("No symbols with sufficient data")

    ranked = sorted(symbol_scores.values(), key=lambda x: x["score"], reverse=True)
    top2   = ranked[:min(2, len(ranked))]
    cal    = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "collect_mins":  COLLECT_MINS,
        "all_symbols":   ranked,
        "trade_symbols": top2,
    }
    with open(CAL_FILE, "w") as f: json.dump(cal, f, indent=2)
    _store.upload(CAL_FILE, "calibration.json")
    info(f"Calibration saved → {CAL_FILE}")
    return cal


# ─────────────────────────────────────────────────────────────────────────────
# KNOCKOUT TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class KnockOutTracker:
    FIELDS = ["ts","symbol","growth_rate","ticks_survived","entry_sigma",
              "entry_spike","entry_regime","exit_reason","stake","pnl"]

    def __init__(self):
        exists = os.path.exists(KNOCKOUT_LOG) and os.path.getsize(KNOCKOUT_LOG) > 0
        self._f = open(KNOCKOUT_LOG, "a", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        if not exists: self._w.writeheader()
        self._recent: list = []

    def record(self, symbol: str, growth_rate: int, ticks_survived: int,
               entry_sigma: float, entry_spike: float, entry_regime: str,
               exit_reason: str, stake: float, pnl: float):
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol, "growth_rate": growth_rate,
            "ticks_survived": ticks_survived,
            "entry_sigma": round(entry_sigma, 5),
            "entry_spike": round(entry_spike, 5),
            "entry_regime": entry_regime, "exit_reason": exit_reason,
            "stake": stake, "pnl": round(pnl, 4),
        }
        self._w.writerow(row); self._f.flush()
        self._recent.append(row)
        if len(self._recent) > 50: self._recent.pop(0)
        _store.upload(KNOCKOUT_LOG, "knockout_log.csv")

    @property
    def recent(self): return list(self._recent)

    def ko_rate_recent(self, n: int = 10) -> float:
        recent = self._recent[-n:]
        if not recent: return 0.0
        return sum(1 for r in recent if r["exit_reason"] == "KNOCKOUT") / len(recent)

_ko_tracker = KnockOutTracker()


# ─────────────────────────────────────────────────────────────────────────────
# CANDLE FEED + INDICATOR ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class CandleFeed:
    async def fetch(self, symbol: str) -> dict:
        result = {"candles_1m": [], "candles_5m": []}
        try:
            ws = await websockets.connect(
                WS_URL, ping_interval=20, ping_timeout=15, close_timeout=5)
            rid = 0

            async def send(data):
                nonlocal rid; rid += 1; data["req_id"] = rid
                await ws.send(json.dumps(data))

            async def recv_type(mtype, timeout=10):
                deadline = asyncio.get_event_loop().time() + timeout
                while True:
                    rem = deadline - asyncio.get_event_loop().time()
                    if rem <= 0: return None
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=rem)
                        msg = json.loads(raw)
                        if mtype in msg or "error" in msg: return msg
                    except Exception: return None

            await send({"authorize": API_TOKEN})
            auth = await recv_type("authorize", timeout=10)
            if not auth or "error" in auth:
                warn("[CANDLE] Auth failed"); return result

            end_epoch = int(time.time())
            for gran, key in [(CANDLE_GRAN_1, "candles_1m"),
                              (CANDLE_GRAN_5, "candles_5m")]:
                await send({
                    "ticks_history": symbol, "style": "candles",
                    "granularity": gran,
                    "start": end_epoch - gran * CANDLE_COUNT * 2,
                    "end":   end_epoch, "count": CANDLE_COUNT,
                })
                resp = await recv_type("candles", timeout=12)
                if resp and "candles" in resp:
                    result[key] = [
                        {"epoch": c["epoch"], "open": float(c["open"]),
                         "high": float(c["high"]), "low": float(c["low"]),
                         "close": float(c["close"])}
                        for c in resp["candles"][-CANDLE_COUNT:]
                    ]
            await ws.close()
        except Exception as exc: err(f"[CANDLE] {exc}")
        return result


class IndicatorEngine:
    @staticmethod
    def rsi(closes, period=14):
        if len(closes) < period + 1: return None
        gains = []; losses = []
        for i in range(1, len(closes)):
            d = closes[i] - closes[i-1]
            gains.append(max(d, 0)); losses.append(max(-d, 0))
        ag = sum(gains[-period:]) / period
        al = sum(losses[-period:]) / period
        return 100.0 if al == 0 else round(100 - 100 / (1 + ag / al), 2)

    @staticmethod
    def bollinger(closes, period=20, std_dev=2.0):
        if len(closes) < period: return None
        w = closes[-period:]; mid = sum(w) / period
        var = sum((x - mid) ** 2 for x in w) / period
        std = math.sqrt(var); upper = mid + std_dev * std; lower = mid - std_dev * std
        price = closes[-1]
        return {
            "upper": round(upper, 5), "mid": round(mid, 5), "lower": round(lower, 5),
            "width": round((upper - lower) / (mid + 1e-9), 6),
            "pos":   round((price - lower) / (upper - lower + 1e-9), 4),
        }

    @staticmethod
    def ema_cross(closes):
        if len(closes) < 14:
            return {"ema7": None, "ema14": None, "cross": "neutral", "gap": 0.0}
        k7 = 2/8; k14 = 2/15; e7 = closes[0]; e14 = closes[0]
        for c in closes[1:]:
            e7 = c * k7 + e7 * (1 - k7); e14 = c * k14 + e14 * (1 - k14)
        gap = e7 - e14
        return {
            "ema7": round(e7, 5), "ema14": round(e14, 5),
            "cross": "bullish" if gap > 0 else "bearish" if gap < 0 else "neutral",
            "gap": round(gap, 6),
        }

    @staticmethod
    def atr(candles, period=14):
        if len(candles) < period + 1: return None
        trs = []
        for i in range(1, len(candles)):
            h = candles[i]["high"]; l = candles[i]["low"]; pc = candles[i-1]["close"]
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        return round(sum(trs[-period:]) / period, 6)

    @classmethod
    def compute(cls, candle_data: dict) -> dict:
        out = {}
        c1m = candle_data.get("candles_1m", [])
        c5m = candle_data.get("candles_5m", [])
        cl1 = [c["close"] for c in c1m]; cl5 = [c["close"] for c in c5m]
        out["rsi_14_1m"] = cls.rsi(cl1); out["rsi_14_5m"] = cls.rsi(cl5)
        out["bb_1m"] = cls.bollinger(cl1); out["bb_5m"] = cls.bollinger(cl5)
        out["ema_1m"] = cls.ema_cross(cl1); out["ema_5m"] = cls.ema_cross(cl5)
        out["atr_1m"] = cls.atr(c1m); out["atr_5m"] = cls.atr(c5m)
        rsi = out["rsi_14_1m"]; bb = out["bb_1m"]; ema = out["ema_1m"]
        regime = "UNKNOWN"
        if rsi is not None and bb is not None:
            if bb["width"] < 0.0005: regime = "COMPRESSED"
            elif rsi > 70 and bb["pos"] > 0.85: regime = "OVERBOUGHT"
            elif rsi < 30 and bb["pos"] < 0.15: regime = "OVERSOLD"
            elif bb["width"] > 0.003 and ema["cross"] != "neutral": regime = "TRENDING"
            elif bb["width"] < 0.0015 and 40 <= (rsi or 50) <= 60: regime = "CALM"
            else: regime = "RANGING"
        out["market_regime"] = regime
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 3-LAYER ENSEMBLE GATE
# ─────────────────────────────────────────────────────────────────────────────

_FEATURES = [
    'sigma_ewma','range_20','range_50','ema_gap','zscore_50',
    'spike_10','atr_14','entropy_20','regime_enc',
    'sigma_trend','range_ratio','ema_cross','zscore_abs',
    'entropy_delta','sigma_vs_gate','spike_vs_sigma','atr_trend','hour_of_day',
]
_REGIME_ENC = {'CALM':0,'RANGING':1,'TRENDING':2,'CHAOS':3}


class EnsembleGate:
    def __init__(self, persist_dir: str, sigma_gate: float):
        self.persist_dir = persist_dir
        self.sigma_gate  = sigma_gate
        self._xgb = None; self._lr = None; self._iso = None
        self._load()

    def _load(self):
        xp = os.path.join(self.persist_dir, "xgb_model.json")
        if os.path.exists(xp):
            try:
                from xgboost import XGBClassifier
                m = XGBClassifier(); m.load_model(xp); self._xgb = m
                info("[ENS] L1 XGBoost loaded")
            except Exception as e: warn(f"[ENS] L1 XGB: {e}")
        if self._xgb is None:
            gp = os.path.join(self.persist_dir, "gb_model.pkl")
            if os.path.exists(gp):
                try:
                    with open(gp, 'rb') as f: self._xgb = pickle.load(f)
                    info("[ENS] L1 GBM fallback loaded")
                except Exception as e: warn(f"[ENS] L1 GBM: {e}")
        lp = os.path.join(self.persist_dir, "lr_model.pkl")
        if os.path.exists(lp):
            try:
                with open(lp, 'rb') as f: self._lr = pickle.load(f)
                info("[ENS] L2 LogReg loaded")
            except Exception as e: warn(f"[ENS] L2: {e}")
        ip = os.path.join(self.persist_dir, "iso_model.pkl")
        if os.path.exists(ip):
            try:
                with open(ip, 'rb') as f: self._iso = pickle.load(f)
                info(f"[ENS] L3 IsoForest loaded  thr={getattr(self._iso,'_ens_threshold','?')}")
            except Exception as e: warn(f"[ENS] L3: {e}")
        n = sum(x is not None for x in [self._xgb, self._lr, self._iso])
        info(f"[ENS] {n}/3 layers loaded")

    @property
    def active(self):
        return any(x is not None for x in [self._xgb, self._lr, self._iso])

    def predict(self, feats: dict, regime: str) -> dict:
        import numpy as np
        row = [[feats.get(f, 0.0) for f in _FEATURES]]
        if regime in ("TRENDING", "CHAOS"):
            return {"votes":0,"trade":False,"reason":f"regime_{regime}",
                    "xgb_prob":0.0,"lr_prob":0.0,"iso_score":0.0,
                    "v_xgb":False,"v_lr":False,"v_iso":False}
        v_xgb=True; xgb_prob=1.0
        v_lr=True;  lr_prob=1.0
        v_iso=True; iso_score=0.0
        if self._xgb:
            try:
                xgb_prob = float(self._xgb.predict_proba(
                    np.array(row, dtype=float))[:, 1][0])
                v_xgb = xgb_prob >= XGB_THRESHOLD
            except Exception as e: warn(f"[ENS] L1: {e}")
        if self._lr:
            try:
                lr_prob = float(self._lr.predict_proba(
                    np.array(row, dtype=float))[:, 1][0])
                v_lr = lr_prob >= LR_THRESHOLD
            except Exception as e: warn(f"[ENS] L2: {e}")
        if self._iso:
            try:
                iso_score = float(self._iso.score_samples(
                    np.array(row, dtype=float))[0])
                v_iso = iso_score >= getattr(self._iso, '_ens_threshold', -0.5)
            except Exception as e: warn(f"[ENS] L3: {e}")
        votes = sum([v_xgb, v_lr, v_iso])
        return {"votes":votes,"trade":votes>=2,
                "xgb_prob":round(xgb_prob,4),"lr_prob":round(lr_prob,4),
                "iso_score":round(iso_score,4),
                "v_xgb":v_xgb,"v_lr":v_lr,"v_iso":v_iso}

_ensemble: Optional[EnsembleGate] = None

def load_ensemble(cal: dict) -> EnsembleGate:
    global _ensemble
    _ensemble = EnsembleGate(_PERSIST_DIR, cal.get("sigma_gate", 0.10))
    return _ensemble

def _build_feature_matrix(rows_raw, sigma_gate, labels_arr=None):
    import numpy as np
    sv  = [r["sigma_ewma"] for r in rows_raw]
    av  = [r["atr_14"]     for r in rows_raw]
    ev  = [r["entropy_20"] for r in rows_raw]
    hrs = [int(r["ts"][11:13]) for r in rows_raw]
    X_rows=[]; y_rows=[]
    for i, r in enumerate(rows_raw):
        if labels_arr is not None:
            lbl = labels_arr[i]
            if lbl != lbl: continue
        X_rows.append([
            r["sigma_ewma"], r["range_20"], r["range_50"], r["ema_gap"],
            r["zscore_50"], r["spike_10"], r["atr_14"], r["entropy_20"],
            _REGIME_ENC.get(r["regime"], 0),
            sv[i] - sv[max(0, i-10)],
            r["range_20"] / (r["range_50"] + 1e-9),
            r["ema7"] - r["ema14"],
            abs(r["zscore_50"]),
            ev[i] - ev[max(0, i-5)],
            r["sigma_ewma"] / (sigma_gate + 1e-9),
            r["spike_10"] / (r["sigma_ewma"] + 1e-9),
            av[i] - av[max(0, i-10)],
            hrs[i],
        ])
        if labels_arr is not None: y_rows.append(int(labels_arr[i]))
    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int) if labels_arr is not None else None
    return X, y


def retrain_ensemble(cal: dict):
    global _ensemble
    import numpy as np

    sym         = cal.get("symbol", "1HZ10V")
    csv_path    = os.path.join(DATA_DIR, f"{sym}.csv")
    growth_rate = cal.get("growth_rate", GROWTH_RATE)
    target_t    = cal.get("target_ticks", TARGET_TICKS)
    sigma_gate  = cal.get("sigma_gate", 0.10)

    if not os.path.exists(csv_path):
        warn("[ENS] retrain: CSV not found"); return

    rows_raw = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows_raw.append({
                    "sigma_ewma": float(row["sigma_ewma"]),
                    "range_20":   float(row["range_20"]),
                    "range_50":   float(row["range_50"]),
                    "ema_gap":    float(row["ema_gap"]),
                    "ema7":       float(row["ema7"]),
                    "ema14":      float(row["ema14"]),
                    "zscore_50":  float(row["zscore_50"]),
                    "spike_10":   float(row["spike_10"]),
                    "atr_14":     float(row["atr_14"]),
                    "entropy_20": float(row["entropy_20"]),
                    "regime":     row["regime"].strip(),
                    "price":      float(row["price"]),
                    "ts":         row["ts"],
                })
            except Exception: continue

    if len(rows_raw) < 200:
        warn(f"[ENS] retrain: {len(rows_raw)} rows — need 200+"); return

    prices  = np.array([r["price"] for r in rows_raw])
    n       = len(prices)
    labels  = []
    for i in range(n):
        if i + target_t >= n: labels.append(float('nan')); continue
        entry = prices[i]
        b0    = BarrierSimulator._initial_barrier(float(entry), growth_rate)
        surv  = True
        for t in range(1, target_t + 1):
            bt = b0 * ((1 + growth_rate / 100) ** t)
            if abs(prices[i + t] - entry) >= bt: surv = False; break
        labels.append(1.0 if surv else 0.0)

    X, y = _build_feature_matrix(rows_raw, sigma_gate, labels)
    info(f"[ENS] Training {len(X)} samples  "
         f"survival_rate={y.mean()*100:.1f}%  growth={growth_rate}%  target={target_t}t")

    # Guard: need both classes to train classifiers
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        warn(f"[ENS] Single-class labels (pos={n_pos} neg={n_neg}) — "
             f"skipping L1/L2. Only IsoForest (L3) will train. "
             f"Try longer COLLECT_MINS or lower growth rate.")
        # Still train IsoForest on the available (all-win) data
        try:
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(n_estimators=200,
                                  contamination=ISO_CONTAMINATION,
                                  random_state=42, n_jobs=-1)
            iso.fit(X)
            win_scores = iso.score_samples(X)
            iso._ens_threshold = float(
                np.percentile(win_scores, ISO_CONTAMINATION * 100))
            out = os.path.join(_PERSIST_DIR, "iso_model.pkl")
            with open(out, 'wb') as f: pickle.dump(iso, f)
            _store.upload(out, "iso_model.pkl")
            info(f"[ENS] L3 IsoForest trained on {len(X)} rows  "
                 f"thr={iso._ens_threshold:.4f}")
        except Exception as e: warn(f"[ENS] L3 single-class: {e}")
        _ensemble = EnsembleGate(_PERSIST_DIR, sigma_gate)
        info("[ENS] Partial retrain complete (L3 only)")
        return

    # L1: XGBoost
    xgb_ok = False
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                             reg_alpha=0.1, reg_lambda=1.0,
                             eval_metric='logloss', verbosity=0)
        xgb.fit(X, y, verbose=False)
        out = os.path.join(_PERSIST_DIR, "xgb_model.json")
        xgb.save_model(out); _store.upload(out, "xgb_model.json")
        pr  = xgb.predict_proba(X)[:, 1]; mask = pr >= XGB_THRESHOLD
        if mask.sum() > 0:
            info(f"[ENS] L1 XGBoost: {mask.sum()} signals  precision={y[mask].mean()*100:.1f}%")
        xgb_ok = True
    except ImportError: warn("[ENS] xgboost not installed")
    except Exception as e: warn(f"[ENS] L1: {e}")

    if not xgb_ok:
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            gbm = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                              learning_rate=0.05, subsample=0.8,
                                              min_samples_leaf=10, random_state=42)
            gbm.fit(X, y)
            out = os.path.join(_PERSIST_DIR, "gb_model.pkl")
            with open(out, 'wb') as f: pickle.dump(gbm, f)
            _store.upload(out, "gb_model.pkl")
            info("[ENS] L1 GBM fallback trained")
        except Exception as e: warn(f"[ENS] L1 GBM: {e}")

    # L2: Logistic Regression
    try:
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        lr_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ('lr', LogisticRegression(C=0.1, max_iter=1000, solver='lbfgs', random_state=42)),
        ])
        lr_pipe.fit(X, y)
        out = os.path.join(_PERSIST_DIR, "lr_model.pkl")
        with open(out, 'wb') as f: pickle.dump(lr_pipe, f)
        _store.upload(out, "lr_model.pkl")
        lp = lr_pipe.predict_proba(X)[:, 1]; lm = lp >= LR_THRESHOLD
        if lm.sum() > 0:
            info(f"[ENS] L2 LogReg: {lm.sum()} signals  precision={y[lm].mean()*100:.1f}%")
    except Exception as e: warn(f"[ENS] L2: {e}")

    # L3: Isolation Forest on survived rows only
    try:
        from sklearn.ensemble import IsolationForest
        X_wins = X[y == 1]
        info(f"[ENS] L3 IsoForest: {len(X_wins)} survival rows")
        iso = IsolationForest(n_estimators=200, contamination=ISO_CONTAMINATION,
                              random_state=42, n_jobs=-1)
        iso.fit(X_wins)
        win_scores = iso.score_samples(X_wins)
        iso._ens_threshold = float(np.percentile(win_scores, ISO_CONTAMINATION * 100))
        out = os.path.join(_PERSIST_DIR, "iso_model.pkl")
        with open(out, 'wb') as f: pickle.dump(iso, f)
        _store.upload(out, "iso_model.pkl")
        all_scores = iso.score_samples(X)
        blocked = (all_scores < iso._ens_threshold).sum()
        info(f"[ENS] L3 IsoForest: thr={iso._ens_threshold:.4f}  "
             f"blocks {blocked}/{len(X)} ({blocked/len(X)*100:.1f}%)")
    except Exception as e: warn(f"[ENS] L3: {e}")

    _ensemble = EnsembleGate(_PERSIST_DIR, sigma_gate)
    info("[ENS] Ensemble retrained and hot-swapped")


# ─────────────────────────────────────────────────────────────────────────────
# TAKE PROFIT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class TakeProfitEngine:
    def __init__(self, cal: dict, stake: float):
        self.stake    = stake
        self.target_t = cal.get("target_ticks", TARGET_TICKS)
        self.r1 = cal.get("payout_target_1", PAYOUT_TARGET_1)
        self.r2 = cal.get("payout_target_2", PAYOUT_TARGET_2)
        self.r3 = cal.get("payout_target_3", PAYOUT_TARGET_3)
        self.t1 = max(1,  self.target_t // 3)
        self.t2 = max(2, (self.target_t * 2) // 3)

    def current_target(self, ticks: int) -> float:
        if ticks < self.t1: return self.r1
        if ticks < self.t2: return self.r2
        return self.r3

    def should_exit(self, ticks: int, current_payout: float) -> bool:
        return current_payout >= self.stake * self.current_target(ticks)


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVE CONTRACT MONITOR
# ─────────────────────────────────────────────────────────────────────────────

class ActiveContractMonitor:
    """
    Per-tick exit engine. Called on every tick while contract is open.
    E1 target_ticks reached
    E2 sigma >= exit_sigma_gate (vol spike)
    E3 spike >= exit_spike_gate (raw spike)
    E4 regime shifted to TRENDING or CHAOS
    E5 payout ratchet hit
    E6 max_hold_ticks absolute ceiling
    """
    def __init__(self, cal: dict, stake: float, growth_rate: int):
        self.cal          = cal
        self.stake        = stake
        self.growth_rate  = growth_rate
        self.tp           = TakeProfitEngine(cal, stake)
        self.target_t     = cal.get("target_ticks",    TARGET_TICKS)
        self.exit_sigma   = cal.get("exit_sigma_gate", 0.20)
        self.exit_spike   = cal.get("exit_spike_gate", 0.40)
        self.ticks        = 0
        self.entry_sigma  = 0.0
        self.entry_spike  = 0.0
        self.entry_regime = "CALM"

    def set_entry_context(self, sigma: float, spike: float, regime: str):
        self.entry_sigma  = sigma
        self.entry_spike  = spike
        self.entry_regime = regime

    def evaluate(self, sigma: float, spike: float, regime: str,
                 current_payout: float) -> Tuple[bool, str]:
        self.ticks += 1
        if self.ticks >= self.target_t:         return True, "E1"
        if self.ticks >= MAX_HOLD_TICKS:         return True, "E6"
        if sigma >= self.exit_sigma:             return True, "E2"
        if spike >= self.exit_spike:             return True, "E3"
        if regime in ("TRENDING", "CHAOS"):      return True, "E4"
        if self.tp.should_exit(self.ticks, current_payout): return True, "E5"
        return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# GROWTH RATE SELECTOR
# ─────────────────────────────────────────────────────────────────────────────

class GrowthRateSelector:
    _VALID = [1, 2, 3, 5]

    def __init__(self):
        self._current = GROWTH_RATE
        self._history: list = []
        if os.path.exists(GROWTH_HIST):
            try:
                with open(GROWTH_HIST) as f: self._history = json.load(f)
                # Restore last known rate
                if self._history:
                    self._current = self._history[-1].get("to", GROWTH_RATE)
            except Exception: pass

    def update(self, cal: dict, advisor_override: int = None):
        cal_rate = cal.get("growth_rate", GROWTH_RATE)
        raw      = advisor_override if advisor_override else cal_rate
        new_rate = min(self._VALID, key=lambda x: abs(x - raw))
        old_rate = self._current
        self._current = new_rate
        self._history.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "from": old_rate, "to": new_rate,
            "cal_rate": cal_rate, "advisor_override": advisor_override,
            "ko_rate": cal.get("ko_rate"), "survival_rate": cal.get("survival_rate"),
        })
        with open(GROWTH_HIST, "w") as f:
            json.dump(self._history[-50:], f, indent=2)
        _store.upload(GROWTH_HIST, "growth_rate_history.json")
        if old_rate != new_rate:
            info(f"[GROWTH] {old_rate}% → {new_rate}%")

    @property
    def current(self) -> int: return self._current

_growth_selector = GrowthRateSelector()


# ─────────────────────────────────────────────────────────────────────────────
# AI ADVISOR  (L1-L9)
# ─────────────────────────────────────────────────────────────────────────────

class AIAdvisor:
    BREAKEVEN_WR = 0.55
    MIN_TRADES   = 3

    def __init__(self):
        self._cycle  = 0
        self._last_wr: Optional[float] = None
        self._consec_hold = 0

    def advise(self, context: dict) -> dict:
        self._cycle  = context.get("cycle", self._cycle + 1)
        traders      = context.get("traders", [])
        cal          = context.get("calibration", {})
        indicators   = context.get("indicators", {})

        total_trades = sum(t.risk.wins + t.risk.losses for t in traders)
        total_wins   = sum(t.risk.wins for t in traders)
        session_pnl  = sum(t.risk.session_pnl for t in traders)
        max_streak   = max((t.risk.loss_streak for t in traders), default=0)
        wr           = total_wins / total_trades if total_trades > 0 else None
        ko_rate      = _ko_tracker.ko_rate_recent(10)
        recent_kos   = _ko_tracker.recent

        reasoning=[]; adj={}; layer="HOLD"

        # L1: EMERGENCY
        if max_streak >= max(1, MARTINGALE_STEPS):
            reasoning.append(
                f"L1-EMERGENCY: loss_streak={max_streak}. "
                f"Tightening sigma/spike gates and extending cooldown.")
            adj["sigma_gate"]    = cal.get("sigma_gate", 0.10) * 0.80
            adj["spike_gate"]    = cal.get("spike_gate", 0.20) * 0.80
            adj["loss_cooldown"] = min(LOSS_COOLDOWN * 2, 180)
            layer = "L1_EMERGENCY"

        if session_pnl < -(STOP_LOSS * 0.5) and total_trades >= self.MIN_TRADES:
            reasoning.append(
                f"L1-EMERGENCY: P&L=${session_pnl:.2f} — reset stake, tighten gates.")
            adj["sigma_gate"] = cal.get("sigma_gate", 0.10) * 0.75
            adj["spike_gate"] = cal.get("spike_gate", 0.20) * 0.75
            adj["base_stake"] = BASE_STAKE
            layer = "L1_EMERGENCY"

        # L2: PERFORMANCE
        if layer == "HOLD" and total_trades >= self.MIN_TRADES and wr is not None:
            wr_delta = (wr - self._last_wr) if self._last_wr else 0
            if wr < self.BREAKEVEN_WR - 0.08:
                reasoning.append(
                    f"L2-PERFORMANCE: WR={wr:.1%} far below breakeven. Tightening gates.")
                adj["sigma_gate"] = cal.get("sigma_gate", 0.10) * 0.85
                adj["spike_gate"] = cal.get("spike_gate", 0.20) * 0.85
                adj["z_gate"]     = cal.get("z_gate", 1.0) * 0.88
                layer = "L2_PERFORMANCE"
            elif wr < self.BREAKEVEN_WR:
                reasoning.append(f"L2-PERFORMANCE: WR={wr:.1%} below breakeven. Nudging z_gate.")
                adj["z_gate"] = cal.get("z_gate", 1.0) * 0.92
                layer = "L2_PERFORMANCE"
            elif wr > self.BREAKEVEN_WR + 0.10 and wr_delta > 0:
                reasoning.append(
                    f"L2-PERFORMANCE: WR={wr:.1%} healthy+improving. Relaxing sigma_gate.")
                adj["sigma_gate"] = min(cal.get("sigma_gate", 0.10) * 1.05,
                                        SAFE_BOUNDS["sigma_gate"][1])
                layer = "L2_PERFORMANCE"

        # L3: MARKET REGIME
        if layer == "HOLD" and indicators:
            regime = indicators.get("market_regime", "UNKNOWN")
            if regime == "CALM":
                reasoning.append(f"L3-MARKET: CALM — ideal for accumulators. Relaxing spike_gate.")
                adj["spike_gate"] = min(cal.get("spike_gate", 0.20) * 1.05,
                                        SAFE_BOUNDS["spike_gate"][1])
                layer = "L3_MARKET"
            elif regime in ("TRENDING", "OVERBOUGHT", "OVERSOLD"):
                reasoning.append(
                    f"L3-MARKET: {regime} — dangerous for accumulators. Tightening gates.")
                adj["sigma_gate"]     = cal.get("sigma_gate", 0.10) * 0.88
                adj["exit_sigma_mult"]= max(cal.get("exit_sigma_mult", EXIT_SIGMA_MULT) * 0.90,
                                            SAFE_BOUNDS["exit_sigma_mult"][0])
                layer = "L3_MARKET"
            elif regime == "COMPRESSED":
                reasoning.append(f"L3-MARKET: COMPRESSED — breakout risk. Tightening entry.")
                adj["sigma_gate"] = cal.get("sigma_gate", 0.10) * 0.85
                adj["range_gate"] = cal.get("range_gate", 0.30) * 0.85
                layer = "L3_MARKET"

        # L4: ENSEMBLE HEALTH
        if layer == "HOLD":
            ens_agree = context.get("ens_agree_rate")
            if ens_agree is not None and ens_agree < 0.60:
                reasoning.append(
                    f"L4-ENSEMBLE: agreement={ens_agree:.0%} — models diverging. "
                    f"Raising thresholds.")
                global XGB_THRESHOLD, LR_THRESHOLD
                XGB_THRESHOLD = min(XGB_THRESHOLD + 0.05, 0.90)
                LR_THRESHOLD  = min(LR_THRESHOLD  + 0.05, 0.90)
                layer = "L4_ENSEMBLE"

        # L5: KNOCK-OUT ANALYSIS
        if layer == "HOLD":
            if ko_rate > 0.50 and total_trades >= self.MIN_TRADES:
                reasoning.append(
                    f"L5-KNOCKOUT: KO rate={ko_rate:.0%} > 50%. "
                    f"Stepping down growth rate and tightening exit gates.")
                cur_gr = _growth_selector.current
                adj["growth_rate"] = max(1, cur_gr - 1)
                adj["exit_sigma_mult"] = max(
                    cal.get("exit_sigma_mult", EXIT_SIGMA_MULT) * 0.85,
                    SAFE_BOUNDS["exit_sigma_mult"][0])
                adj["target_ticks"] = max(
                    cal.get("target_ticks", TARGET_TICKS) - 3,
                    SAFE_BOUNDS["target_ticks"][0])
                layer = "L5_KNOCKOUT"
            elif ko_rate < 0.20 and total_trades >= self.MIN_TRADES:
                cur_gr = _growth_selector.current
                if cur_gr < 3 and wr and wr >= self.BREAKEVEN_WR:
                    reasoning.append(
                        f"L5-KNOCKOUT: KO rate={ko_rate:.0%} very low. "
                        f"Market calm — stepping up growth rate to {cur_gr+1}%.")
                    adj["growth_rate"] = min(5, cur_gr + 1)
                    layer = "L5_KNOCKOUT"

        # L6: EXIT CALIBRATION
        if layer == "HOLD" and recent_kos:
            avg_t     = sum(r["ticks_survived"] for r in recent_kos) / len(recent_kos)
            cal_med   = cal.get("median_ticks", TARGET_TICKS)
            if avg_t < cal_med * 0.60:
                reasoning.append(
                    f"L6-EXIT: avg_ticks={avg_t:.1f} < 60% of calibrated median {cal_med}. "
                    f"Tightening exit gates and reducing target.")
                adj["exit_sigma_mult"] = max(
                    cal.get("exit_sigma_mult", EXIT_SIGMA_MULT) * 0.88,
                    SAFE_BOUNDS["exit_sigma_mult"][0])
                adj["target_ticks"] = max(int(avg_t * 0.8), SAFE_BOUNDS["target_ticks"][0])
                layer = "L6_EXIT"
            elif avg_t > cal_med * 1.20 and wr and wr > self.BREAKEVEN_WR:
                reasoning.append(
                    f"L6-EXIT: avg_ticks={avg_t:.1f} > calibrated median. "
                    f"Extending target to capture more profit.")
                adj["target_ticks"] = min(int(avg_t * 0.9), SAFE_BOUNDS["target_ticks"][1])
                layer = "L6_EXIT"

        # L7: SYMBOL ROTATION
        if layer == "HOLD":
            for t in traders:
                t_ko = _ko_tracker.ko_rate_recent(5)
                if t_ko > 0.70 and t.risk.losses >= 3:
                    reasoning.append(
                        f"L7-ROTATION: [{t.symbol}] KO={t_ko:.0%} last 5 — pausing 1 cycle.")
                    t._advisor_paused = True; layer = "L7_ROTATION"

        # L8: PAYOUT RATCHET TUNING
        if layer == "HOLD" and wr and wr > self.BREAKEVEN_WR + 0.05:
            self._consec_hold += 1
            if self._consec_hold >= 3:
                reasoning.append(
                    f"L8-RATCHET: {self._consec_hold} HOLD cycles, WR={wr:.1%}. "
                    f"Nudging payout targets up.")
                adj["payout_target_2"] = min(
                    cal.get("payout_target_2", PAYOUT_TARGET_2) + 0.10,
                    SAFE_BOUNDS["payout_target_2"][1])
                adj["payout_target_3"] = min(
                    cal.get("payout_target_3", PAYOUT_TARGET_3) + 0.20,
                    SAFE_BOUNDS["payout_target_3"][1])
                layer = "L8_RATCHET"; self._consec_hold = 0
        else:
            self._consec_hold = 0

        # L9: HOLD
        if layer == "HOLD":
            if total_trades < self.MIN_TRADES:
                reasoning.append(f"L9-HOLD: {total_trades} trades — insufficient data.")
            else:
                reasoning.append(
                    f"L9-HOLD: WR={wr:.1%}  P&L=${session_pnl:.2f}  "
                    f"KO={ko_rate:.0%}. No changes.")

        # Apply SAFE_BOUNDS
        applied={}; rejected={}
        for key, proposed in adj.items():
            if key not in SAFE_BOUNDS: continue
            lo, hi, max_step = SAFE_BOUNDS[key]
            current = cal.get(key, proposed)
            if isinstance(proposed, float):
                delta   = proposed - current
                clamped = current + max(-max_step, min(max_step, delta))
                clamped = round(max(lo, min(hi, clamped)), 5)
            else:
                delta   = int(proposed) - int(current)
                clamped = int(current) + max(-int(max_step), min(int(max_step), int(delta)))
                clamped = max(int(lo), min(int(hi), int(clamped)))
            if clamped == current:
                rejected[key] = f"no change after bounds clip (proposed={proposed})"
            else:
                applied[key] = {"from": current, "to": clamped}

        if "growth_rate" in applied:
            _growth_selector.update(cal, applied["growth_rate"]["to"])

        self._last_wr = wr
        return {
            "cycle": self._cycle, "layer": layer,
            "reasoning": reasoning, "applied": applied, "rejected": rejected,
            "context_summary": {
                "trades": total_trades,
                "win_rate": round(wr, 4) if wr else None,
                "session_pnl": round(session_pnl, 4),
                "max_streak": max_streak,
                "ko_rate_recent": round(ko_rate, 4),
                "growth_rate": _growth_selector.current,
                "market_regime": indicators.get("market_regime", "?"),
            },
        }

    def write_log(self, result: dict):
        sep = "═" * 70
        lines = [f"\n{sep}",
                 f"CYCLE {result['cycle']}  |  "
                 f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
                 f"LAYER: {result['layer']}", sep, "CONTEXT:"]
        for k, v in result["context_summary"].items():
            lines.append(f"  {k:<22} {v}")
        lines.append("\nREASONING:")
        for r in result["reasoning"]: lines.append(f"  • {r}")
        if result["applied"]:
            lines.append("\nADJUSTMENTS APPLIED:")
            for k, v in result["applied"].items():
                lines.append(f"  ✓ {k:<22} {v['from']} → {v['to']}")
        else: lines.append("\nADJUSTMENTS APPLIED: none")
        if result["rejected"]:
            lines.append("\nADJUSTMENTS REJECTED:")
            for k, v in result["rejected"].items():
                lines.append(f"  ✗ {k:<22} {v}")
        lines.append(sep)
        block = "\n".join(lines)
        try:
            with open(ADVISOR_LOG, "a") as f: f.write(block + "\n")
            _store.upload(ADVISOR_LOG, "advisor_log.txt")
        except Exception as exc: warn(f"[ADVISOR] log: {exc}")
        info(block)

_advisor = AIAdvisor()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — COLLECTOR
# ─────────────────────────────────────────────────────────────────────────────

class Collector:
    def __init__(self):
        self._stats: Dict[str, SymbolStats] = {s: SymbolStats(s) for s in SURVEY_SYMBOLS}
        self._ws = None; self._rid = 0
        self._inbox  = asyncio.Queue()
        self._send_q = asyncio.Queue()
        self._start_time  = time.time()
        self._tick_counts = {s: 0 for s in SURVEY_SYMBOLS}

    async def run(self) -> dict:
        info(f"Phase 1: collecting {COLLECT_MINS:.0f}min rolling data...")
        await self._connect_and_auth()
        for sym in SURVEY_SYMBOLS:
            await self._send({"ticks": sym, "subscribe": 1})
        deadline = self._start_time + COLLECT_SECS
        while time.time() < deadline:
            remaining = deadline - time.time()
            try:
                msg = await asyncio.wait_for(
                    self._inbox.get(), timeout=min(30, remaining))
            except asyncio.TimeoutError:
                if time.time() >= deadline: break
                continue
            if "__disconnect__" in msg:
                warn("Collector: disconnected — reconnecting")
                await asyncio.sleep(5)
                await self._connect_and_auth()
                for sym in SURVEY_SYMBOLS:
                    await self._send({"ticks": sym, "subscribe": 1})
                continue
            if msg.get("msg_type") == "tick":
                tick = msg.get("tick", {}); sym = tick.get("symbol", "")
                if sym in self._stats:
                    self._stats[sym].update(float(tick["quote"]),
                                             float(tick.get("epoch", time.time())))
                    self._tick_counts[sym] += 1
            elapsed = time.time() - self._start_time
            if int(elapsed) % 60 == 0 and elapsed > 1:
                rem = max(0, COLLECT_SECS - elapsed)
                counts = "  ".join(f"{s}:{self._tick_counts[s]}" for s in SURVEY_SYMBOLS)
                info(f"[COLLECT] {elapsed/60:.0f}min  remaining={rem/60:.0f}min  [{counts}]")

        elapsed = time.time() - self._start_time
        info(f"Phase 1 complete ({elapsed/60:.1f}min). Computing calibration...")
        summaries = []
        for sym, st in self._stats.items():
            summaries.append(st.summarise()); st.close()
            rolling_csv_trim(sym)
        if self._ws:
            try: await self._ws.close()
            except Exception: pass
        # Cancel collector IO tasks cleanly — prevents "Task destroyed pending" warning
        for t in getattr(self, "_io_tasks", []):
            if not t.done(): t.cancel()
        if getattr(self, "_io_tasks", []):
            await asyncio.gather(*self._io_tasks, return_exceptions=True)
        return compute_calibration(summaries)

    async def _connect_and_auth(self):
        info("Collector: connecting...")
        self._ws = await websockets.connect(WS_URL, ping_interval=20, ping_timeout=15)
        # Track IO tasks so we can cancel them cleanly on Collector exit
        self._io_tasks = [
            asyncio.create_task(self._recv_pump(), name="col_recv"),
            asyncio.create_task(self._send_pump(), name="col_send"),
        ]
        await self._send({"authorize": API_TOKEN})
        resp = await self._recv_one("authorize", timeout=15)
        if not resp or "error" in resp:
            raise ConnectionError(
                f"Auth failed: {(resp or {}).get('error',{}).get('message','?')}")
        info(f"Collector: auth OK  balance=${resp['authorize'].get('balance',0):.2f}")

    async def _send_pump(self):
        while True:
            data, fut = await self._send_q.get()
            try:
                await self._ws.send(json.dumps(data))
                if fut and not fut.done(): fut.set_result(True)
            except Exception as exc:
                if fut and not fut.done(): fut.set_exception(exc)
            finally: self._send_q.task_done()

    async def _recv_pump(self):
        try:
            async for raw in self._ws:
                try: await self._inbox.put(json.loads(raw))
                except Exception: pass
        except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
            await self._inbox.put({"__disconnect__": True})
        except Exception as exc:
            err(f"Collector recv: {exc}")
            await self._inbox.put({"__disconnect__": True})

    async def _send(self, data: dict):
        self._rid += 1; data["req_id"] = self._rid
        fut = asyncio.get_event_loop().create_future()
        await self._send_q.put((data, fut)); return fut

    async def _recv_one(self, msg_type: str, timeout=10) -> Optional[dict]:
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0: return None
            try:
                msg = await asyncio.wait_for(self._inbox.get(), timeout=remaining)
            except asyncio.TimeoutError: return None
            if "__disconnect__" in msg: await self._inbox.put(msg); return None
            if msg_type in msg or "error" in msg: return msg
            await self._inbox.put(msg)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class SignalEngine:
    def __init__(self, cal: dict):
        self.cal = cal; self.tick_n = 0
        self.prices: deque = deque(maxlen=500)
        self._sigma_ewma = None; self._ema7 = self._ema14 = None
        self._k7 = 2/8; self._k14 = 2/15; self.EWMA_ALPHA = 0.05
        self._warmup = 80
        self._sigma_buf   = deque(maxlen=15)
        self._entropy_buf = deque(maxlen=10)
        self._atr_buf     = deque(maxlen=15)

    def ingest(self, price: float) -> dict:
        self.tick_n += 1
        prev = self.prices[-1] if self.prices else price
        delta = abs(price - prev); self.prices.append(price)
        if self._sigma_ewma is None: self._sigma_ewma = delta
        else: self._sigma_ewma = (self.EWMA_ALPHA*delta +
                                   (1-self.EWMA_ALPHA)*self._sigma_ewma)
        if self._ema7 is None: self._ema7 = self._ema14 = price
        else:
            self._ema7  = price*self._k7  + self._ema7 *(1-self._k7)
            self._ema14 = price*self._k14 + self._ema14*(1-self._k14)
        if self.tick_n < self._warmup:
            return {"trade":False,"reason":"warmup","tick":self.tick_n}

        prices = list(self.prices); sigma = self._sigma_ewma
        range20= (max(prices[-20:])-min(prices[-20:])) if len(prices)>=20 else 999
        range50= (max(prices[-50:])-min(prices[-50:])) if len(prices)>=50 else range20
        ema_gap= abs(self._ema7-self._ema14)

        z_raw=0.0
        if len(prices)>=200:
            bl=prices[-200:]; mu=sum(bl)/200
            var=sum((p-mu)**2 for p in bl)/200
            std=math.sqrt(var) if var>0 else 1e-9
            z_raw=(sum(prices[-50:])/50-mu)/(std/math.sqrt(50))
        z=abs(z_raw)

        moves=[abs(prices[i]-prices[i-1]) for i in range(-10,0) if i-1>=-len(prices)]
        spike=max(moves) if moves else 0
        atr_mv=[abs(prices[i]-prices[i-1]) for i in range(-14,0) if i-1>=-len(prices)]
        atr14=sum(atr_mv)/len(atr_mv) if atr_mv else 0

        if len(prices)>=21:
            ep=prices[-21:]; em=[abs(ep[i]-ep[i-1]) for i in range(1,len(ep))]
            mx=max(em) or 1; bk=[0]*5
            for m in em: bk[min(4,int(m/mx*4))]+=1
            ne=len(em); H=0.0
            for b in bk:
                if b>0: p=b/ne; H-=p*math.log2(p)
            entropy20=H/math.log2(5)
        else: entropy20=1.0

        if abs(z_raw)>2.5 or sigma>0.3: regime="CHAOS"
        elif abs(z_raw)>1.5 and ema_gap>0.3: regime="TRENDING"
        elif abs(z_raw)<1.0 and ema_gap<0.15: regime="CALM"
        else: regime="RANGING"

        c1=sigma   < self.cal["sigma_gate"]
        c2=range20 < self.cal["range_gate"]
        c3=ema_gap < self.cal["ema_gate"]
        c4=z       < self.cal["z_gate"]
        c5=spike   < self.cal["spike_gate"]
        c6=regime not in ("TRENDING","CHAOS")
        score=sum([c1,c2,c3,c4,c5])

        self._sigma_buf.append(sigma)
        self._entropy_buf.append(entropy20)
        self._atr_buf.append(atr14)
        sg=self.cal.get("sigma_gate",0.10)

        ml_feats={
            "sigma_ewma":sigma,"range_20":range20,"range_50":range50,
            "ema_gap":ema_gap,"zscore_50":z_raw,"spike_10":spike,
            "atr_14":atr14,"entropy_20":entropy20,
            "regime_enc":_REGIME_ENC.get(regime,0),
            "sigma_trend":(sigma-list(self._sigma_buf)[0]
                           if len(self._sigma_buf)>=10 else 0.0),
            "range_ratio":range20/(range50+1e-9),
            "ema_cross":self._ema7-self._ema14,
            "zscore_abs":z,
            "entropy_delta":(entropy20-list(self._entropy_buf)[0]
                             if len(self._entropy_buf)>=5 else 0.0),
            "sigma_vs_gate":sigma/(sg+1e-9),
            "spike_vs_sigma":spike/(sigma+1e-9),
            "atr_trend":(atr14-list(self._atr_buf)[0]
                         if len(self._atr_buf)>=10 else 0.0),
            "hour_of_day":datetime.now(timezone.utc).hour,
        }

        if _ensemble and _ensemble.active:
            ens=_ensemble.predict(ml_feats,regime)
        else:
            ens={"votes":3,"trade":True,"xgb_prob":1.0,"lr_prob":1.0,
                 "iso_score":0.0,"v_xgb":True,"v_lr":True,"v_iso":True}

        c7=ens["trade"]
        trade=score>=4 and c6 and c7

        return {
            "trade":trade,"score":score,"tick":self.tick_n,
            "sigma":round(sigma,5),"range20":round(range20,4),
            "ema_gap":round(ema_gap,5),"z":round(z,4),
            "spike":round(spike,5),"regime":regime,
            "votes":ens["votes"],"xgb_prob":ens["xgb_prob"],
            "lr_prob":ens["lr_prob"],"iso_score":ens["iso_score"],
            "c1":c1,"c2":c2,"c3":c3,"c4":c4,"c5":c5,"c6":c6,"c7":c7,
            "_sigma":sigma,"_spike":spike,"_regime":regime,
        }


# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class RiskManager:
    def __init__(self):
        self.stake        = BASE_STAKE
        self.loss_streak  = 0
        self.session_pnl  = 0.0
        self.wins         = 0
        self.losses       = 0
        self._cooldown_until = 0.0

    def get_stake(self): return round(self.stake, 2)

    def can_trade(self) -> Tuple[bool, str]:
        if time.monotonic() < self._cooldown_until:
            return False, f"cooldown({self._cooldown_until-time.monotonic():.0f}s)"
        if self.session_pnl <= -STOP_LOSS:   return False, "stop_loss"
        if self.session_pnl >=  TARGET_PROFIT: return False, "target_hit"
        return True, "ok"

    def record_win(self, profit: float):
        self.wins += 1; self.session_pnl += profit
        self.loss_streak = 0; self.stake = BASE_STAKE
        tlog(f"WIN +${profit:.4f}  stake→${self.stake:.2f}  P&L=${self.session_pnl:.4f}")

    def record_loss(self, amount: float):
        self.losses += 1; self.session_pnl -= amount; self.loss_streak += 1
        self._cooldown_until = time.monotonic() + LOSS_COOLDOWN
        if self.loss_streak > MARTINGALE_STEPS:
            self.stake = BASE_STAKE; self.loss_streak = 0
            warn(f"MARTINGALE exhausted — RESET  P&L=${self.session_pnl:.4f}")
        else:
            self.stake = round(BASE_STAKE * (MARTINGALE_MULT ** self.loss_streak), 2)
            tlog(f"LOSS streak={self.loss_streak}/{MARTINGALE_STEPS}  "
                 f"next=${self.stake:.2f}  P&L=${self.session_pnl:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# DERIV CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class DerivClient:
    def __init__(self):
        self._ws = None; self._send_q = asyncio.Queue()
        self._inbox = asyncio.Queue(); self._rid = 0
        self._send_task = None; self._recv_task = None
        self.balance = 0.0; self.symbol = ""

    async def connect(self) -> bool:
        try:
            info(f"Connecting → {WS_URL}")
            self._ws = await websockets.connect(
                WS_URL, ping_interval=20, ping_timeout=20, close_timeout=10)
            self._start_io()
            await self._send_msg({"authorize": API_TOKEN})
            resp = await self._recv_type("authorize", timeout=15)
            if not resp or "error" in resp:
                err(f"Auth: {(resp or {}).get('error',{}).get('message','?')}")
                return False
            auth = resp["authorize"]
            self.balance = float(auth.get("balance", 0))
            info(f"Auth OK  {auth.get('loginid')}  balance=${self.balance:.2f}")
            return True
        except Exception as exc: err(f"connect: {exc}"); return False

    def _start_io(self):
        for t in (self._send_task, self._recv_task):
            if t and not t.done(): t.cancel()
        self._send_task = asyncio.create_task(self._send_pump())
        self._recv_task = asyncio.create_task(self._recv_pump())

    async def _send_pump(self):
        while True:
            data, fut = await self._send_q.get()
            try:
                await self._ws.send(json.dumps(data))
                if fut and not fut.done(): fut.set_result(True)
            except Exception as exc:
                if fut and not fut.done(): fut.set_exception(exc)
            finally: self._send_q.task_done()

    async def _recv_pump(self):
        try:
            async for raw in self._ws:
                try: await self._inbox.put(json.loads(raw))
                except Exception: pass
        except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
            await self._inbox.put({"__disconnect__": True})
        except Exception as exc:
            err(f"recv: {exc}")
            await self._inbox.put({"__disconnect__": True})

    async def close(self):
        for t in (self._send_task, self._recv_task):
            if t and not t.done(): t.cancel()
        if self._ws:
            try: await self._ws.close()
            except Exception: pass

    async def _send_msg(self, data: dict):
        self._rid += 1; data["req_id"] = self._rid
        fut = asyncio.get_event_loop().create_future()
        await self._send_q.put((data, fut)); await fut

    async def _recv_type(self, msg_type: str, timeout=10) -> Optional[dict]:
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            rem = deadline - asyncio.get_event_loop().time()
            if rem <= 0: return None
            try:
                msg = await asyncio.wait_for(self._inbox.get(), timeout=rem)
            except asyncio.TimeoutError: return None
            if "__disconnect__" in msg: await self._inbox.put(msg); return None
            if msg_type in msg or "error" in msg: return msg
            await self._inbox.put(msg)

    async def receive(self, timeout=60) -> dict:
        try: return await asyncio.wait_for(self._inbox.get(), timeout=timeout)
        except asyncio.TimeoutError: return {}

    async def subscribe_ticks(self, symbol: str) -> bool:
        await self._send_msg({"ticks": symbol, "subscribe": 1})
        resp = await self._recv_type("tick", timeout=10)
        if not resp or "error" in resp:
            err(f"Tick sub: {(resp or {}).get('error',{}).get('message','?')}")
            return False
        info(f"Subscribed {symbol}"); return True

    async def fetch_balance(self) -> Optional[float]:
        try:
            await self._send_msg({"balance": 1})
            resp = await self._recv_type("balance", timeout=10)
            if resp and "balance" in resp:
                return float(resp["balance"]["balance"])
        except Exception as exc: warn(f"fetch_balance: {exc}")
        return None

    async def buy_accumulator(self, symbol: str, growth_rate: int,
                               stake: float) -> Tuple[Optional[int], Optional[float]]:
        # Drain any stale tick messages from inbox before sending proposal
        # so _recv_type("proposal") doesn't time out grabbing tick responses
        drained = 0
        while not self._inbox.empty():
            try:
                msg = self._inbox.get_nowait()
                # Put non-tick messages back (could be settlement updates)
                if "tick" not in msg:
                    await self._inbox.put(msg)
                else:
                    drained += 1
            except Exception: break
        if drained:
            info(f"[BUY] Drained {drained} stale tick(s) from inbox")

        await self._send_msg({
            "proposal": 1, "amount": stake, "basis": "stake",
            "contract_type": "ACCU", "currency": "USD",
            "symbol": symbol,
            "growth_rate": growth_rate / 100,
        })
        proposal = await self._recv_type("proposal", timeout=12)
        if not proposal or "error" in proposal:
            err(f"Proposal: {(proposal or {}).get('error',{}).get('message','?')}")
            return None, None
        prop = proposal.get("proposal", {}); pid = prop.get("id")
        ask  = float(prop.get("ask_price", stake))
        info(f"ACCU proposal  growth={growth_rate}%  ask=${ask:.2f}")
        if not pid: return None, None

        buy_ts = time.time()
        await self._send_msg({"buy": pid, "price": ask})
        cid = None
        for attempt in range(8):
            resp = await self._recv_type("buy", timeout=8)
            if resp is None: warn(f"Buy no resp attempt {attempt+1}"); continue
            if "error" in resp:
                err(f"Buy error: {resp['error'].get('message','')}"); return None, None
            cid = resp.get("buy", {}).get("contract_id")
            if cid: break

        if not cid:
            warn("Orphan recovery via profit_table")
            for _ in range(4):
                await asyncio.sleep(3)
                await self._send_msg({"profit_table":1,"description":1,
                                      "sort":"DESC","limit":5})
                r = await self._recv_type("profit_table", timeout=10)
                if r and "profit_table" in r:
                    for tx in r["profit_table"].get("transactions", []):
                        if (abs(float(tx.get("buy_price",0)) - stake) < 0.01
                                and float(tx.get("purchase_time",0)) >= buy_ts-10):
                            cid = tx.get("contract_id")
                            info(f"Orphan recovered → {cid}"); break
                if cid: break
            if not cid: err("Orphan recovery failed"); return None, None

        try:
            await self._send_msg({"proposal_open_contract":1,
                                   "contract_id":cid,"subscribe":1})
        except Exception: pass

        tlog(f"ACCU placed  cid={cid}  growth={growth_rate}%  ${ask:.2f}")
        return cid, ask

    async def sell_contract(self, cid: int, price: float = 0) -> Optional[dict]:
        try:
            await self._send_msg({"sell": cid, "price": price})
            resp = await self._recv_type("sell", timeout=10)
            if resp and "sell" in resp:
                sold = resp["sell"]
                info(f"SOLD cid={cid}  sold_for=${float(sold.get('sold_for',0)):.4f}")
                return sold
            if resp and "error" in resp:
                warn(f"sell error: {resp['error'].get('message','')}")
        except Exception as exc: warn(f"sell_contract: {exc}")
        return None

    async def poll_contract(self, cid: int) -> Optional[dict]:
        try:
            await self._send_msg({"proposal_open_contract":1,"contract_id":cid})
            resp = await self._recv_type("proposal_open_contract", timeout=10)
            if resp and "proposal_open_contract" in resp:
                return resp["proposal_open_contract"]
        except Exception as exc: warn(f"poll_contract: {exc}")
        return None

    @staticmethod
    def is_settled(data: dict) -> bool:
        if data.get("is_settled") or data.get("is_sold"): return True
        return data.get("status", "").lower() in ("sold","won","lost")


# ─────────────────────────────────────────────────────────────────────────────
# SYMBOL TRADER
# ─────────────────────────────────────────────────────────────────────────────

class SymbolTrader:
    def __init__(self, cal: dict):
        self.cal    = cal
        self.symbol = cal["symbol"]
        self.engine = SignalEngine(cal)
        self.risk   = RiskManager()
        self.client = DerivClient(); self.client.symbol = self.symbol
        self.waiting      = False
        self._evaluating  = False
        self._settling    = False
        self.current_trade: Optional[dict] = None
        self.lock_since:    Optional[float] = None
        self._stop         = False
        self._loss_cd_until= 0.0
        self._poller_task: Optional[asyncio.Task] = None
        self.live_ticks    = 0
        self.signals       = 0
        self._advisor_paused = False
        self._monitor: Optional[ActiveContractMonitor] = None
        self._exit_counts: Dict[str, int] = {}
        self.hot_swap_calibration = lambda c: None   # overwritten in main()

    def _unlock(self, reason="manual"):
        if self.waiting: info(f"[{self.symbol}] Unlock reason={reason}")
        self.waiting=False; self.current_trade=None
        self.lock_since=None; self._evaluating=False
        self._settling=False; self._monitor=None
        if self._poller_task and not self._poller_task.done():
            self._poller_task.cancel(); self._poller_task=None

    def _check_lock_timeout(self):
        if self.waiting and self.lock_since:
            if time.monotonic() - self.lock_since >= LOCK_TIMEOUT:
                warn(f"[{self.symbol}] Lock timeout — unlocking")
                self._unlock("timeout")

    async def on_tick(self, price: float):
        self.live_ticks += 1; self._check_lock_timeout()
        sig = self.engine.ingest(price)

        # Per-tick exit evaluation while contract is open
        if self.waiting and self._monitor and self.current_trade:
            sigma  = sig.get("_sigma", 0.0)
            spike  = sig.get("_spike", 0.0)
            regime = sig.get("_regime", "CALM")
            gr     = self.current_trade.get("growth_rate", GROWTH_RATE)
            stake  = self.current_trade.get("stake",       BASE_STAKE)
            ticks_in = self._monitor.ticks
            est_payout = stake * ((1 + gr/100) ** ticks_in)
            should_exit, reason = self._monitor.evaluate(sigma, spike, regime, est_payout)
            if should_exit:
                asyncio.get_event_loop().create_task(
                    self._execute_exit(reason, est_payout))
                return

        if self.live_ticks % 30 == 0:
            cd = max(0, self._loss_cd_until - time.monotonic())
            ok, why = self.risk.can_trade()
            ticks_in = self._monitor.ticks if self._monitor else 0
            status = ("LOCKED("+str(ticks_in)+"t)" if self.waiting
                      else f"COOLDOWN({cd:.0f}s)" if cd > 0
                      else "READY" if ok else f"BLOCKED:{why}")
            paused = "[PAUSED] " if self._advisor_paused else ""
            info(f"[{self.symbol}] {paused}t={sig['tick']} "
                 f"score={sig.get('score','?')}/5  "
                 f"sigma={sig.get('sigma','?')}  "
                 f"regime={sig.get('regime','?')}  {status}")

        if self.waiting or self._evaluating: return
        if self._advisor_paused: return
        if time.monotonic() < self._loss_cd_until: return
        if not sig.get("trade"): return
        ok, reason = self.risk.can_trade()
        if not ok: return
        self._evaluating = True
        try: await self._evaluate(sig)
        finally: self._evaluating = False

    async def _evaluate(self, sig: dict):
        if self.waiting: return
        self.signals += 1
        growth_rate = _growth_selector.current
        info(f"[{self.symbol}] SIGNAL #{self.signals}  "
             f"score={sig['score']}/5  sigma={sig['sigma']}  "
             f"spike={sig['spike']}  regime={sig['regime']}  "
             f"votes={sig.get('votes','?')}/3  xgb={sig.get('xgb_prob','?')}  "
             f"growth={growth_rate}%  target={self.cal.get('target_ticks',TARGET_TICKS)}t")

        stake = self.risk.get_stake()
        bal = await self.client.fetch_balance()
        if bal: self.client.balance = bal

        cid, buy_price = await self.client.buy_accumulator(
            self.symbol, growth_rate, stake)

        if cid:
            monitor = ActiveContractMonitor(self.cal, buy_price, growth_rate)
            monitor.set_entry_context(sig["_sigma"], sig["_spike"], sig["_regime"])
            self._monitor      = monitor
            self.current_trade = {
                "id": cid, "stake": buy_price, "growth_rate": growth_rate,
                "entry_sigma": sig["_sigma"], "entry_spike": sig["_spike"],
                "entry_regime": sig["_regime"],
            }
            self.waiting    = True
            self.lock_since = time.monotonic()
            self._poller_task = asyncio.create_task(
                self._safety_poller(cid, growth_rate),
                name=f"poller_{cid}")
        else:
            warn(f"[{self.symbol}] Trade placement failed")

    async def _execute_exit(self, reason: str, est_payout: float):
        if not self.waiting or not self.current_trade: return
        cid      = self.current_trade["id"]
        ticks_in = self._monitor.ticks if self._monitor else 0
        stake    = self.current_trade["stake"]
        info(f"[{self.symbol}] EXIT {reason} after {ticks_in}t  est=${est_payout:.4f}")
        self._exit_counts[reason] = self._exit_counts.get(reason, 0) + 1

        sold = await self.client.sell_contract(cid)
        if sold:
            sold_for = float(sold.get("sold_for", 0))
            profit   = sold_for - stake
            await self.client.fetch_balance()
            if profit > 0:
                self.risk.record_win(profit)
                tlog(f"[{self.symbol}] SOLD({reason}) +${profit:.4f} ticks={ticks_in}")
            else:
                self.risk.record_loss(stake)
                _ko_tracker.record(
                    self.symbol, self.current_trade["growth_rate"], ticks_in,
                    self.current_trade["entry_sigma"], self.current_trade["entry_spike"],
                    self.current_trade["entry_regime"], reason, stake, profit)
            self._unlock(reason)
            return

        # sell_contract returned None — contract may have already expired/settled.
        # Poll for the final result instead of silently unlocking with no record.
        warn(f"[{self.symbol}] Sell returned None for cid={cid} "
             f"(likely already expired) — polling for settlement")
        for attempt in range(1, 6):
            await asyncio.sleep(2)
            try:
                data = await self.client.poll_contract(cid)
                if data and self.client.is_settled(data):
                    profit = float(data.get("profit", 0))
                    await self.client.fetch_balance()
                    if profit > 0:
                        self.risk.record_win(profit)
                        tlog(f"[{self.symbol}] SETTLED({reason}) "
                             f"+${profit:.4f} ticks={ticks_in}")
                    else:
                        self.risk.record_loss(stake)
                        _ko_tracker.record(
                            self.symbol, self.current_trade["growth_rate"],
                            ticks_in, self.current_trade["entry_sigma"],
                            self.current_trade["entry_spike"],
                            self.current_trade["entry_regime"],
                            "KNOCKOUT", stake, profit)
                    self._unlock(reason)
                    return
            except Exception as exc:
                warn(f"[{self.symbol}] Poll attempt {attempt}: {exc}")

        # Still no result — force unlock to avoid permanent lock
        warn(f"[{self.symbol}] Could not confirm settlement for cid={cid} "
             f"— force unlocking to prevent lock-up")
        self._unlock(f"{reason}_unconfirmed")

    async def _safety_poller(self, cid: int, growth_rate: int):
        await asyncio.sleep(max(10, TARGET_TICKS + 5))
        if not self.waiting or not self.current_trade or \
                self.current_trade.get("id") != cid: return
        warn(f"[{self.symbol}] Safety poller firing cid={cid}")
        for attempt in range(1, 7):
            try:
                data = await self.client.poll_contract(cid)
                if data and self.client.is_settled(data):
                    profit   = float(data.get("profit", 0))
                    stake    = self.current_trade["stake"]
                    ticks_in = self._monitor.ticks if self._monitor else 0
                    if profit > 0: self.risk.record_win(profit)
                    else:
                        self.risk.record_loss(stake)
                        _ko_tracker.record(
                            self.symbol, growth_rate, ticks_in,
                            self.current_trade.get("entry_sigma", 0),
                            self.current_trade.get("entry_spike", 0),
                            self.current_trade.get("entry_regime", "?"),
                            "KNOCKOUT", stake, profit)
                    self._unlock("poller_settled"); return
            except Exception as exc: warn(f"Poller attempt {attempt}: {exc}")
            await asyncio.sleep(5)
        if self.waiting and self.current_trade and \
                self.current_trade.get("id") == cid:
            warn(f"[{self.symbol}] Poller exhausted — force unlock")
            self._unlock("poller_exhausted")

    async def handle_settlement(self, data: dict) -> bool:
        if self._settling: return True
        self._settling = True
        try:
            cid = data.get("contract_id")
            if not self.current_trade or \
                    str(cid) != str(self.current_trade.get("id", "")): return True
            if not self.client.is_settled(data): return True
            profit   = float(data.get("profit", 0))
            stake    = self.current_trade["stake"]
            ticks_in = self._monitor.ticks if self._monitor else 0
            await self.client.fetch_balance()
            if profit > 0: self.risk.record_win(profit)
            else:
                self.risk.record_loss(stake)
                _ko_tracker.record(
                    self.symbol, self.current_trade.get("growth_rate", GROWTH_RATE),
                    ticks_in, self.current_trade.get("entry_sigma", 0),
                    self.current_trade.get("entry_spike", 0),
                    self.current_trade.get("entry_regime", "?"),
                    "KNOCKOUT", stake, profit)
            tlog(f"[{self.symbol}] SETTLED profit={profit:+.4f} ticks={ticks_in}")
            self._unlock("settlement"); return True
        finally: self._settling = False

    async def run(self):
        retry_delay = 5
        while not self._stop:
            try:
                if not await self.client.connect():
                    raise ConnectionError("connect failed")
                if not await self.client.subscribe_ticks(self.symbol):
                    raise ConnectionError("tick sub failed")
                info(f"[{self.symbol}] Live  growth={_growth_selector.current}%  "
                     f"target={self.cal.get('target_ticks',TARGET_TICKS)}t  "
                     f"sigma_gate={self.cal['sigma_gate']}  "
                     f"spike_gate={self.cal['spike_gate']}")
                while not self._stop:
                    msg = await self.client.receive(timeout=60)
                    if "__disconnect__" in msg:
                        warn(f"[{self.symbol}] WS disconnected"); break
                    if not msg:
                        try: await self.client._ws.ping()
                        except Exception: break
                        continue
                    if "tick" in msg:
                        await self.on_tick(float(msg["tick"]["quote"]))
                    for key in ("proposal_open_contract", "buy"):
                        if key in msg: await self.handle_settlement(msg[key])
                    if "transaction" in msg:
                        tx = msg["transaction"]
                        if "contract_id" in tx:
                            await self.handle_settlement({
                                "contract_id": tx.get("contract_id"),
                                "profit": tx.get("profit", 0),
                                "status": tx.get("action","sold"),
                                "is_settled": True,
                            })
            except Exception as exc:
                err(f"[{self.symbol}] Session: {exc}"); traceback.print_exc()
            if not self._stop:
                warn(f"[{self.symbol}] Reconnecting in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
                await self.client.close()
                self.client = DerivClient(); self.client.symbol = self.symbol

        r = self.risk; tot = r.wins + r.losses
        wr = r.wins / tot * 100 if tot else 0
        info(f"[{self.symbol}] DONE  trades={tot}  W={r.wins}  L={r.losses}  "
             f"WR={wr:.1f}%  P&L=${r.session_pnl:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH SERVER
# ─────────────────────────────────────────────────────────────────────────────

_health_state: dict = {"phase": "collect", "traders": [], "collect_start": 0.0}


def start_health_server(traders: list, phase: str, collect_start: float = 0.0):
    _health_state["phase"]         = phase
    _health_state["traders"]       = traders
    _health_state["collect_start"] = collect_start
    if getattr(start_health_server, "_started", False):
        return
    start_health_server._started = True

    class _H(BaseHTTPRequestHandler):
        def do_GET(self):
            phase         = _health_state["phase"]
            traders       = _health_state["traders"]
            collect_start = _health_state["collect_start"]

            # ── JSON status ──────────────────────────────────────────────────
            if self.path == "/status":
                data = {
                    "phase": phase, "contract": "ACCUMULATOR",
                    "supabase": {
                        "enabled": _store.enabled,
                        "bucket":  SUPABASE_BUCKET if _store.enabled else None,
                    },
                    "growth_rate": _growth_selector.current,
                    "ko_rate_recent": round(_ko_tracker.ko_rate_recent(10), 4),
                    "ensemble_active": bool(_ensemble and _ensemble.active),
                    "traders": [],
                }
                for t in traders:
                    r = t.risk; tot = r.wins + r.losses
                    data["traders"].append({
                        "symbol":       t.symbol,
                        "ticks":        t.live_ticks,
                        "signals":      t.signals,
                        "trades":       tot,
                        "wins":         r.wins,
                        "losses":       r.losses,
                        "win_rate":     round(r.wins / tot, 4) if tot else 0,
                        "pnl":          round(r.session_pnl, 4),
                        "stake":        r.stake,
                        "locked":       t.waiting,
                        "ticks_in_trade": (t._monitor.ticks if t._monitor else 0),
                        "exit_counts":  t._exit_counts,
                        "advisor_paused": t._advisor_paused,
                        "cal": {
                            "growth_rate":    t.cal.get("growth_rate"),
                            "target_ticks":   t.cal.get("target_ticks"),
                            "survival_rate":  t.cal.get("survival_rate"),
                            "ko_rate":        t.cal.get("ko_rate"),
                            "sigma_gate":     t.cal.get("sigma_gate"),
                            "spike_gate":     t.cal.get("spike_gate"),
                            "exit_sigma_gate":t.cal.get("exit_sigma_gate"),
                            "payout_target_1":t.cal.get("payout_target_1"),
                            "payout_target_2":t.cal.get("payout_target_2"),
                            "payout_target_3":t.cal.get("payout_target_3"),
                        },
                    })
                body = json.dumps(data, indent=2).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers(); self.wfile.write(body)
                return

            # ── HTML dashboard ───────────────────────────────────────────────
            if phase == "collect":
                elapsed   = time.time() - collect_start
                remaining = max(0, COLLECT_SECS - elapsed)
                html_body = f"""
<h2 style="color:#58a6ff">Phase 1 — Collecting Data</h2>
<table><tr><th>Metric</th><th>Value</th></tr>
<tr><td>Elapsed</td><td>{elapsed/60:.1f} min</td></tr>
<tr><td>Remaining</td><td>{remaining/60:.1f} min</td></tr>
<tr><td>Window</td><td>{COLLECT_MINS:.0f} min rolling</td></tr>
<tr><td>Symbols</td><td>{', '.join(SURVEY_SYMBOLS)}</td></tr>
<tr><td>Supabase</td><td>{'✓ '+SUPABASE_BUCKET if _store.enabled else '✗ local-only'}</td></tr>
</table>
<p style="color:#8b949e">Running BarrierSimulator on completion to find optimal growth rate.</p>"""
            else:
                ko_recent = _ko_tracker.ko_rate_recent(10)
                rows = ""
                for t in traders:
                    r   = t.risk; tot = r.wins + r.losses
                    wr  = r.wins / tot * 100 if tot else 0
                    tin = t._monitor.ticks if t._monitor else 0
                    ec  = " ".join(f"{k}:{v}" for k, v in t._exit_counts.items()) or "—"
                    c   = t.cal
                    wr_col   = "#3fb950" if wr >= 55 else "#f85149"
                    pnl_col  = "#3fb950" if r.session_pnl >= 0 else "#f85149"
                    lock_str = f"🔒 {tin}t" if t.waiting else "🟢 ready"
                    rows += f"""<tr>
<td><strong>{t.symbol}</strong></td>
<td>{tot}</td>
<td style="color:#3fb950">{r.wins}</td>
<td style="color:#f85149">{r.losses}</td>
<td style="color:{wr_col}">{wr:.1f}%</td>
<td style="color:{pnl_col}">${r.session_pnl:+.4f}</td>
<td>${r.stake:.2f}</td>
<td>{c.get('growth_rate','?')}%</td>
<td>{c.get('target_ticks','?')}t</td>
<td>{c.get('survival_rate',0):.1%}</td>
<td>{c.get('ko_rate',0):.1%}</td>
<td>{lock_str}</td>
<td>{'⏸' if t._advisor_paused else '▶'}</td>
<td style="font-size:0.75rem">{ec}</td>
</tr>"""

                ko_col = "#f85149" if ko_recent > 0.40 else "#3fb950"
                html_body = f"""
<h2 style="color:#58a6ff">Phase 2 — Accumulator Trading</h2>
<div style="margin-bottom:1rem;font-size:0.9rem">
  Growth Rate: <strong>{_growth_selector.current}%</strong> &nbsp;|&nbsp;
  KO Rate (last 10): <strong style="color:{ko_col}">{ko_recent:.0%}</strong> &nbsp;|&nbsp;
  Ensemble: <strong>{'active' if _ensemble and _ensemble.active else 'fallback'}</strong> &nbsp;|&nbsp;
  Supabase: <strong>{'✓ '+SUPABASE_BUCKET if _store.enabled else '✗ local-only'}</strong>
</div>
<table>
<tr>
  <th>Symbol</th><th>Trades</th><th>W</th><th>L</th><th>WR</th>
  <th>P&amp;L</th><th>Stake</th><th>Growth</th><th>Target</th>
  <th>Survival</th><th>KO rate</th><th>Status</th><th>Advisor</th><th>Exits</th>
</tr>
{rows}
</table>
<h3 style="color:#58a6ff;margin-top:1.5rem">Calibration Reference</h3>
<table>
<tr><th>Symbol</th><th>sigma_gate</th><th>spike_gate</th>
    <th>exit_sigma_gate</th><th>Ratchet ×1</th><th>Ratchet ×2</th><th>Ratchet ×3</th></tr>
{"".join(
    f"<tr><td>{t.symbol}</td>"
    f"<td>{t.cal.get('sigma_gate','?')}</td>"
    f"<td>{t.cal.get('spike_gate','?')}</td>"
    f"<td>{t.cal.get('exit_sigma_gate','?')}</td>"
    f"<td>×{t.cal.get('payout_target_1','?')}</td>"
    f"<td>×{t.cal.get('payout_target_2','?')}</td>"
    f"<td>×{t.cal.get('payout_target_3','?')}</td></tr>"
    for t in traders
)}
</table>
<p style="color:#8b949e;font-size:0.8rem;margin-top:1rem">
  Breakeven ~55% &nbsp;|&nbsp; Max martingale steps: {MARTINGALE_STEPS} &nbsp;|&nbsp;
  Loss cooldown: {LOSS_COOLDOWN:.0f}s &nbsp;|&nbsp;
  Auto-refreshes 10s
</p>"""

            html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="10">
<title>ACCUMULATOR Bot</title>
<style>
  body {{font-family:monospace;background:#0d1117;color:#e6edf3;padding:2rem;}}
  h2,h3 {{color:#58a6ff;}}
  table {{border-collapse:collapse;width:100%;margin-bottom:1rem;}}
  th,td {{padding:.35rem .7rem;border:1px solid #21262d;font-size:0.82rem;}}
  th {{background:#161b22;color:#8b949e;font-weight:normal;}}
</style></head>
<body>
{html_body}
<p><a href="/status" style="color:#58a6ff">/status JSON</a></p>
</body></html>"""
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers(); self.wfile.write(body)

        def log_message(self, *a): pass

    srv = HTTPServer(("", PORT), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    info(f"Health server :{PORT}  / = dashboard  /status = JSON")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

_shutdown_event = asyncio.Event()


async def main():
    collect_only = "--collect-only" in sys.argv
    trade_only   = "--trade-only"   in sys.argv

    # Allow overriding collect duration at CLI
    for arg in sys.argv:
        if arg.startswith("--collect-mins="):
            global COLLECT_SECS
            COLLECT_SECS = float(arg.split("=")[1]) * 60

    if not API_TOKEN:
        sys.exit("ERROR: DERIV_API_TOKEN not set")

    # ── Restore persisted files from Supabase ─────────────────────────────────
    info("[SUPA] Restoring persisted files from Supabase...")
    _store.restore()

    # ── Skip Phase 1 if calibration already exists ────────────────────────────
    if os.path.exists(CAL_FILE) and not trade_only and not collect_only:
        info(f"Found existing calibration — skipping Phase 1.")
        trade_only = True

    # ── Phase 1: COLLECT ──────────────────────────────────────────────────────
    if not trade_only:
        collect_start = time.time()
        start_health_server([], phase="collect", collect_start=collect_start)
        calibration = await Collector().run()
    else:
        if not os.path.exists(CAL_FILE):
            sys.exit("calibration.json not found — run without --trade-only first.")
        with open(CAL_FILE) as f: calibration = json.load(f)
        info(f"Loaded calibration from {CAL_FILE}")
        info(f"Generated: {calibration.get('generated_at','?')}")
        for s in calibration.get("trade_symbols", []):
            info(f"  {s['symbol']}: growth={s.get('growth_rate')}%  "
                 f"target={s.get('target_ticks')}t  "
                 f"survival={s.get('survival_rate',0):.1%}  "
                 f"ko={s.get('ko_rate',0):.1%}")

    if collect_only:
        info("--collect-only: done.")
        info(json.dumps(calibration.get("trade_symbols", []), indent=2))
        return

    trade_symbols = calibration.get("trade_symbols", [])
    if not trade_symbols:
        sys.exit("No tradeable symbols found in calibration.")

    # ── Train 3-layer ensemble ────────────────────────────────────────────────
    info("Training 3-layer ensemble on Phase 1 data...")
    try:
        retrain_ensemble(trade_symbols[0])
    except Exception as exc:
        warn(f"Ensemble training failed: {exc} — 5-condition fallback mode")

    # ── Load ensemble + growth rate ───────────────────────────────────────────
    ens = load_ensemble(trade_symbols[0])
    n_loaded = sum(x is not None for x in [ens._xgb, ens._lr, ens._iso])
    info(f"[ENS] {n_loaded}/3 layers active")

    _growth_selector.update(trade_symbols[0])
    info(f"[GROWTH] Starting growth rate: {_growth_selector.current}%")

    # ── Build traders ─────────────────────────────────────────────────────────
    traders = [SymbolTrader(cal) for cal in trade_symbols]

    # Wire hot_swap
    for t in traders:
        def _make_swap(trader):
            def hot_swap(new_cal: dict):
                trader.cal    = new_cal
                trader.engine = SignalEngine(new_cal)
                info(f"[{trader.symbol}] ♻ Cal hot-swapped  "
                     f"growth={new_cal.get('growth_rate')}%  "
                     f"target={new_cal.get('target_ticks')}t  "
                     f"sigma_gate={new_cal.get('sigma_gate')}  "
                     f"spike_gate={new_cal.get('spike_gate')}")
            return hot_swap
        t.hot_swap_calibration = _make_swap(t)

    _health_state["traders"] = traders
    _health_state["phase"]   = "trade"
    if not getattr(start_health_server, "_started", False):
        start_health_server(traders, phase="trade")

    info("=" * 70)
    info("ACCUMULATOR BOT — Phase 2 starting")
    for t in traders:
        c = t.cal
        info(f"  {t.symbol}")
        info(f"    growth={c.get('growth_rate')}%  "
             f"target={c.get('target_ticks')}t  "
             f"survival={c.get('survival_rate',0):.1%}  "
             f"ko_rate={c.get('ko_rate',0):.1%}")
        info(f"    sigma_gate={c.get('sigma_gate')}  "
             f"spike_gate={c.get('spike_gate')}  "
             f"exit_sigma_gate={c.get('exit_sigma_gate')}")
        info(f"    ratchet: ×{c.get('payout_target_1')} → "
             f"×{c.get('payout_target_2')} → ×{c.get('payout_target_3')}")
    info(f"  Stake: ${BASE_STAKE:.2f}  Martingale: ×{MARTINGALE_MULT} max {MARTINGALE_STEPS} step(s)")
    info(f"  Stop loss: ${STOP_LOSS}  Target profit: ${TARGET_PROFIT}")
    info(f"  Recalibration: every {COLLECT_MINS:.0f} min (rolling)")
    info(f"  Supabase: {'✓ connected' if _store.enabled else '✗ local-only'}")
    info("=" * 70)

    # ── Rolling recalibration loop (runs while trading continues) ─────────────
    async def recal_loop():
        cycle = 1
        while not _shutdown_event.is_set():
            await asyncio.sleep(COLLECT_SECS)
            if _shutdown_event.is_set(): break
            cycle += 1
            info(f"♻ Recalibration cycle {cycle} "
                 f"({COLLECT_MINS:.0f}min rolling window — trading continues)...")

            # Fetch candle indicators for advisor
            indicators = {}
            try:
                sym = traders[0].symbol if traders else SURVEY_SYMBOLS[0]
                candle_data = await CandleFeed().fetch(sym)
                indicators  = IndicatorEngine.compute(candle_data)
                info(f"[CANDLE] market_regime={indicators.get('market_regime','?')}  "
                     f"rsi={indicators.get('rsi_14_1m','?')}  "
                     f"bb_width={indicators.get('bb_1m',{}).get('width','?') if indicators.get('bb_1m') else '?'}")
            except Exception as exc:
                warn(f"Candle fetch: {exc}")

            # Run advisor before recalibration
            try:
                result = _advisor.advise({
                    "traders":     traders,
                    "calibration": calibration.get("trade_symbols", [{}])[0],
                    "indicators":  indicators,
                    "cycle":       cycle,
                })
                _advisor.write_log(result)

                # Apply advisor adjustments directly to live trader calibrations
                for key, val in result.get("applied", {}).items():
                    if key == "growth_rate": continue  # handled by _growth_selector
                    for t in traders:
                        if key in t.cal:
                            old = t.cal[key]
                            t.cal[key] = val["to"]
                            # Propagate exit gates to active monitor
                            if t._monitor and key == "exit_sigma_gate":
                                t._monitor.exit_sigma = val["to"]
                            if t._monitor and key == "exit_spike_gate":
                                t._monitor.exit_spike = val["to"]
                            if t._monitor and key == "target_ticks":
                                t._monitor.target_t = val["to"]
                            info(f"  [ADVISOR→{t.symbol}] {key}: {old} → {val['to']}")

                # Reset advisor pause flags each cycle
                for t in traders: t._advisor_paused = False

            except Exception as exc:
                warn(f"Advisor error: {exc}")

            # Re-run Phase 1 collection on rolling window data
            try:
                info(f"♻ Running fresh {COLLECT_MINS:.0f}min collection...")
                new_cal = await Collector().run()

                for t in traders:
                    sym_cal = next(
                        (s for s in new_cal.get("trade_symbols", [])
                         if s["symbol"] == t.symbol), None)
                    if sym_cal:
                        t.hot_swap_calibration(sym_cal)
                        _growth_selector.update(sym_cal)
                    else:
                        warn(f"[{t.symbol}] not in new calibration — keeping existing")

                # Retrain ensemble in background (non-blocking)
                if new_cal.get("trade_symbols"):
                    threading.Thread(
                        target=retrain_ensemble,
                        args=(new_cal["trade_symbols"][0],),
                        daemon=True, name="ens_retrain",
                    ).start()
                    info(f"♻ Ensemble retrain launched in background")

                # Save updated survival stats
                try:
                    stats = {
                        "cycle": cycle,
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "symbols": {
                            s["symbol"]: {
                                "growth_rate":   s.get("growth_rate"),
                                "survival_rate": s.get("survival_rate"),
                                "ko_rate":       s.get("ko_rate"),
                                "median_ticks":  s.get("median_ticks"),
                                "target_ticks":  s.get("target_ticks"),
                            }
                            for s in new_cal.get("trade_symbols", [])
                        },
                    }
                    with open(SURVIVAL_FILE, "w") as f:
                        json.dump(stats, f, indent=2)
                    _store.upload(SURVIVAL_FILE, "survival_stats.json")
                except Exception as exc:
                    warn(f"Survival stats save: {exc}")

                info(f"♻ Cycle {cycle} complete")

            except Exception as exc:
                err(f"♻ Recal cycle {cycle} failed: {exc} — keeping existing calibration")

    # ── Launch everything ─────────────────────────────────────────────────────
    trader_tasks  = [asyncio.create_task(t.run(), name=f"trader_{t.symbol}")
                     for t in traders]
    recal_task    = asyncio.create_task(recal_loop(), name="recal_loop")
    shutdown_task = asyncio.create_task(_shutdown_event.wait(), name="shutdown_watch")

    done, pending = await asyncio.wait(
        trader_tasks + [recal_task, shutdown_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    if shutdown_task in done:
        info("Shutdown signal received — stopping traders...")
        for task in trader_tasks + [recal_task]:
            task.cancel()
        await asyncio.gather(*trader_tasks, recal_task, return_exceptions=True)
        info("[SUPA] Final sync to Supabase...")
        _store.push_all()
        info("All done. Goodbye.")
    else:
        # A trader or recal task exited unexpectedly
        for task in done:
            if task.exception():
                err(f"Task {task.get_name()} exited with: {task.exception()}")
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


def _handle_signal(signum, frame):
    info(f"Signal {signum} received — initiating graceful shutdown...")
    try:
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(_shutdown_event.set)
    except Exception: pass


if __name__ == "__main__":
    import signal as _signal
    _signal.signal(_signal.SIGTERM, _handle_signal)
    _signal.signal(_signal.SIGINT,  _handle_signal)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        info("Stopped by user.")
