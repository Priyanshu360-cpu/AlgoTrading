#!/usr/bin/env python3
# =========================================================
# VIRTUAL PAPER TRADER v3  —  Live Tkinter Dashboard
# + ChromaDB Vector Store for Candle State Learning
# =========================================================
#
# NEW IN v3:
#   • ChromaDB persistence — every 15-min bar is stored as a
#     vector embedding with full context:
#       – Candle type  (POSITIVE / NEGATIVE / MIXED)
#       – Score breakdown (news, trend, technical, volume,
#                          index-breadth, fundamental,
#                          momentum, pattern)
#       – Model state (proba, pred_ret, all features)
#       – Market conditions (VWAP, RSI, MACD, ATR …)
#       – Stock news sentiment at that moment
#       – Trade outcome (if a trade fired)
#   • Positive / Negative / Mixed candle classifier
#   • Attribution scoring — each candle's total score is split
#     across contributing factors (sums to |total|)
#   • EOD summary saved per session
#   • query_candle_db() helper for offline analysis / retraining
#
# INSTALL:
#   pip install chromadb sentence-transformers
#   (all other deps same as v2)
# =========================================================

import tkinter as tk
from tkinter import ttk, messagebox
import threading, queue, json, glob, time, joblib, warnings
import hashlib, datetime, uuid
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── ChromaDB (optional graceful degradation if not installed) ──
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_OK = True
except ImportError:
    CHROMA_OK = False
    print("[WARN] chromadb not installed — vector storage disabled. "
          "Install with:  pip install chromadb")

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
MODEL_BUNDLE      = "model_v3_classifier.pkl"
INDICES_FOLDER    = "Indexes"
COMPANIES_FOLDER  = "Companies"

STARTING_CAPITAL  = 2_000_000.0
MAX_POSITIONS     = 100
POSITION_CAP      = 0.25
MIN_CONFIDENCE    = 0.10
MIN_PRED_RETURN   = 0.0
STOP_LOSS_PCT     = 0.005
TRAILING_TRIGGER  = 0.003
TRAILING_STOP_PCT = 0.002
REWARD_MULT       = 1.5
SIMULATE_DELAY    = 0.015
MIN_INVEST        = 1000

# ── ChromaDB settings ─────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_trader_db"
CHROMA_COLLECTION  = "candle_states"

# ── Candle classification thresholds ─────────────────────
POSITIVE_THRESH  =  0.003   # body/close > +0.3%  →  POSITIVE
NEGATIVE_THRESH  = -0.003   # body/close < -0.3%  →  NEGATIVE
# between the two                                  →  MIXED

# ── Score-factor weights (must sum ≈ 1) ──────────────────
SCORE_WEIGHTS = {
    "news_sentiment":    0.20,
    "past_trend":        0.15,
    "technical":         0.20,   # RSI+MACD+Stoch+CCI
    "volume":            0.10,
    "index_breadth":     0.10,
    "fundamental":       0.10,   # sales / margin
    "momentum":          0.08,
    "pattern":           0.07,
}

# ─────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────
BG      = "#0a0e17"
BG2     = "#111827"
BG3     = "#1c2333"
BG4     = "#243044"
FG      = "#cdd6f4"
FG2     = "#7f8ba8"
GREEN   = "#50fa7b"
DKGREEN = "#1a6b35"
RED     = "#ff5555"
DKRED   = "#6b1a1a"
BLUE    = "#8be9fd"
YELLOW  = "#f1fa8c"
ORANGE  = "#ffb86c"
PURPLE  = "#bd93f9"
GREY    = "#6272a4"
WHITE   = "#ffffff"
ACCENT  = "#44475a"


# ═════════════════════════════════════════════════════════
# CHROMA VECTOR STORE
# ═════════════════════════════════════════════════════════

class CandleVectorStore:
    """
    Persists every 15-min bar as a ChromaDB document.

    Embedding strategy
    ------------------
    ChromaDB default embedding function (all-MiniLM-L6-v2) is used on a
    rich natural-language summary of the bar, so semantic similarity
    queries work ("show me bars like this bullish setup with high volume").
    Raw numeric features are stored as metadata for exact filtering.

    Candle types
    ------------
    POSITIVE  — close/open return  > +POSITIVE_THRESH
    NEGATIVE  — close/open return  < NEGATIVE_THRESH
    MIXED     — in between (indecision / doji / spinning top region)

    Score breakdown
    ---------------
    Each bar gets an overall 'model_score' (0-1) built from
    the weighted sub-scores below. The sub-scores are stored
    individually so a training pipeline can use them as labels.
    """

    def __init__(self):
        self.enabled = CHROMA_OK
        if not self.enabled:
            return
        try:
            self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            self.col = self.client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"[ChromaDB] Connected. Collection '{CHROMA_COLLECTION}' "
                  f"has {self.col.count()} documents.")
        except Exception as e:
            print(f"[ChromaDB] Init error: {e}")
            self.enabled = False

    # ── public API ────────────────────────────────────────

    def store_bar(self, bar_record: dict):
        """Store one 15-min bar with all context."""
        if not self.enabled:
            return
        try:
            doc_id   = self._make_id(bar_record)
            document = self._make_document(bar_record)
            metadata = self._make_metadata(bar_record)
            self.col.upsert(
                ids=[doc_id],
                documents=[document],
                metadatas=[metadata],
            )
        except Exception as e:
            print(f"[ChromaDB] store_bar error: {e}")

    def store_session_summary(self, summary: dict):
        """Store an end-of-day session summary."""
        if not self.enabled:
            return
        try:
            doc_id   = f"session_{summary['trade_date']}_{uuid.uuid4().hex[:6]}"
            document = self._make_session_document(summary)
            metadata = {k: str(v) for k, v in summary.items()
                        if isinstance(v, (str, int, float, bool))}
            metadata["record_type"] = "SESSION_SUMMARY"
            self.col.upsert(
                ids=[doc_id],
                documents=[document],
                metadatas=[metadata],
            )
        except Exception as e:
            print(f"[ChromaDB] store_session_summary error: {e}")

    def query_similar_bars(self, bar_record: dict, n=10, candle_type=None):
        """Return the n most similar historical bars."""
        if not self.enabled:
            return []
        try:
            where = {"record_type": "CANDLE_BAR"}
            if candle_type:
                where["candle_type"] = candle_type
            document = self._make_document(bar_record)
            results = self.col.query(
                query_texts=[document],
                n_results=min(n, max(self.col.count(), 1)),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            return results
        except Exception as e:
            print(f"[ChromaDB] query error: {e}")
            return []

    def export_training_dataframe(self) -> "pd.DataFrame":
        """
        Pull all CANDLE_BAR records and return a DataFrame
        suitable for ML training.
        """
        if not self.enabled:
            return pd.DataFrame()
        try:
            total = self.col.count()
            if total == 0:
                return pd.DataFrame()
            results = self.col.get(
                where={"record_type": "CANDLE_BAR"},
                include=["metadatas"],
                limit=total,
            )
            return pd.DataFrame(results["metadatas"])
        except Exception as e:
            print(f"[ChromaDB] export error: {e}")
            return pd.DataFrame()

    # ── internal helpers ──────────────────────────────────

    @staticmethod
    def _make_id(r: dict) -> str:
        key = f"{r['trade_date']}_{r['stock']}_{r['timestamp']}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    @staticmethod
    def _make_document(r: dict) -> str:
        """
        Rich natural-language description used as the embedding text.
        Keeps human readability + semantic searchability.
        """
        ct   = r.get("candle_type", "MIXED")
        sent = r.get("news_sentiment_label", "NEUTRAL")
        sig  = r.get("market_signal", "NEUTRAL")
        trend= r.get("trend_label", "SIDEWAYS")
        rsi_s= "overbought" if r.get("rsi",50)>70 else ("oversold" if r.get("rsi",50)<30 else "neutral")
        lines = [
            f"Stock {r.get('stock','')} on {r.get('trade_date','')} at {r.get('time_slot_label','')}.",
            f"Candle type: {ct}. Body return {r.get('candle_body_pct',0):.3f}%.",
            f"RSI {r.get('rsi',50):.1f} ({rsi_s}), MACD hist {r.get('macd_hist',0):.4f}.",
            f"Volume relative to 5-bar avg: {r.get('rel_vol',1):.2f}x. Volume spike: {r.get('vol_spike',0)}.",
            f"VWAP deviation: {r.get('r_vwap',0):.4f}. ATR%: {r.get('atr_pct',0):.4f}.",
            f"Bollinger band position: {r.get('bb_pct_b',0.5):.2f}. BB squeeze: {r.get('bb_squeeze',0)}.",
            f"News sentiment: {sent} (score {r.get('agg_sent',0):.3f}). Market signal: {sig}.",
            f"Stock-specific news count: {r.get('stk_cnt',0)}. National: {r.get('nat_cnt',0)}. International: {r.get('int_cnt',0)}.",
            f"Index breadth (Nifty/IT/Next50/Gift): {r.get('idx_bull',0)}/4 bullish.",
            f"Nifty return at this bar: {r.get('nifty_r',0):.4f}. Divergence from Nifty: {r.get('diverge_nifty',0):.4f}.",
            f"Trend: {trend}. MA5 slope {r.get('ma5_slope',0):.4f}. Momentum {r.get('mom',0):.4f}.",
            f"Patterns: doji={r.get('doji',0)} hammer={r.get('hammer',0)} shoot_star={r.get('shoot_star',0)} engulf={r.get('engulf',0)}.",
            f"Quarterly: Sales ₹{r.get('sales',0):.0f}Cr, OPM {r.get('opm',0):.1f}%, Net margin {r.get('profit_margin',0):.1f}%.",
            f"Model proba: {r.get('model_proba',0):.4f}. Predicted return: {r.get('pred_ret',0):.4f}.",
            f"Overall model score: {r.get('model_score',0):.4f}.",
            f"Score breakdown — news:{r.get('score_news',0):.3f} trend:{r.get('score_trend',0):.3f} "
            f"tech:{r.get('score_technical',0):.3f} vol:{r.get('score_volume',0):.3f} "
            f"index:{r.get('score_index',0):.3f} fund:{r.get('score_fundamental',0):.3f} "
            f"mom:{r.get('score_momentum',0):.3f} pattern:{r.get('score_pattern',0):.3f}.",
            f"Trade fired: {r.get('trade_fired',False)}. Trade outcome: {r.get('trade_outcome','NONE')}. "
            f"Trade PnL%: {r.get('trade_pnl_pct',0):.3f}%.",
            f"Path direction: {r.get('path_direction','NONE')}. Points awarded: {r.get('points_awarded',0):.4f}.",
        ]
        return " ".join(lines)

    @staticmethod
    def _make_session_document(s: dict) -> str:
        return (
            f"Trading session {s.get('trade_date','')}. "
            f"Final equity ₹{s.get('final_equity',0):,.0f}. "
            f"PnL ₹{s.get('pnl',0):,.2f} ({s.get('pnl_pct',0):.2f}%). "
            f"Trades: {s.get('total_trades',0)} — wins {s.get('wins',0)}, losses {s.get('losses',0)}. "
            f"Win rate {s.get('win_rate',0):.1f}%. "
            f"Bars stored: {s.get('bars_stored',0)}. "
            f"Positive candles: {s.get('positive_candles',0)}, "
            f"Negative: {s.get('negative_candles',0)}, Mixed: {s.get('mixed_candles',0)}."
        )

    @staticmethod
    def _make_metadata(r: dict) -> dict:
        """Flatten record to chromadb-compatible metadata (str/int/float/bool only)."""
        safe = {}
        for k, v in r.items():
            if isinstance(v, (str, int, float, bool)):
                safe[k] = v
            elif isinstance(v, np.integer):
                safe[k] = int(v)
            elif isinstance(v, np.floating):
                safe[k] = float(v)
            elif isinstance(v, (list, dict)):
                safe[k] = json.dumps(v)
            else:
                safe[k] = str(v)
        safe["record_type"] = "CANDLE_BAR"
        return safe


# ═════════════════════════════════════════════════════════
# CANDLE CLASSIFIER & SCORER
# ═════════════════════════════════════════════════════════

def classify_candle(open_px: float, close_px: float) -> str:
    ret = (close_px - open_px) / (open_px + 1e-9)
    if ret > POSITIVE_THRESH:  return "POSITIVE"
    if ret < NEGATIVE_THRESH:  return "NEGATIVE"
    return "MIXED"


def classify_trend(ma5_slope: float, ma10_slope: float, rma20: float) -> str:
    if ma5_slope > 0.001 and rma20 > 0.005:  return "UPTREND"
    if ma5_slope < -0.001 and rma20 < -0.005: return "DOWNTREND"
    return "SIDEWAYS"


def compute_score_breakdown(feats: dict, proba: float, pred_ret: float) -> dict:
    """
    Decompose the 'bullishness' into 8 sub-scores, each in [0,1].
    Final model_score is weighted average.
    Points awarded: signed model_score where positive candle → +score,
    negative → −score, mixed → ±0.1*score.
    """

    # 1. News sentiment  (agg_sent in [-1,1], bull/bear flags, counts)
    s_news = max(0., min(1.,
        0.5 + feats.get("agg_sent", 0) * 0.5
        + feats.get("bull_news", 0) * 0.05
        - feats.get("bear_news", 0) * 0.05
    ))

    # 2. Past trend  (MA slopes, r5, r10)
    s_trend = max(0., min(1.,
        0.5
        + feats.get("ma5_slope", 0) * 5
        + feats.get("ma10_slope", 0) * 3
        + feats.get("r5", 0) * 2
        + feats.get("r10", 0)
    ))

    # 3. Technical (RSI, MACD hist, Stoch, CCI normalised)
    rsi_sc   = (feats.get("rsi", 50) - 30) / 40      # 0 at RSI=30, 1 at RSI=70
    macd_sc  = max(0., min(1., 0.5 + feats.get("macd_hist", 0) * 100))
    stoch_sc = feats.get("stoch_k", 50) / 100
    cci_sc   = max(0., min(1., 0.5 + feats.get("cci", 0) / 200))
    s_tech   = np.mean([rsi_sc, macd_sc, stoch_sc, cci_sc])

    # 4. Volume  (rel_vol, vol_spike)
    s_vol = max(0., min(1.,
        0.3 + feats.get("rel_vol", 1) * 0.2
        + feats.get("vol_spike", 0) * 0.3
    ))

    # 5. Index breadth  (idx_bull 0-4, nifty/next50/IT/gift returns)
    s_idx = max(0., min(1.,
        feats.get("idx_bull", 2) / 4
        + feats.get("idx_avg", 0) * 5
    ))

    # 6. Fundamental  (OPM, profit margin normalised)
    opm_sc = max(0., min(1., feats.get("opm", 10) / 30))
    pm_sc  = max(0., min(1., feats.get("profit_margin", 5) / 20))
    s_fund = (opm_sc + pm_sc) / 2

    # 7. Momentum  (mom, mom_acc, autocorr)
    s_mom = max(0., min(1.,
        0.5 + feats.get("mom", 0) * 10
        + feats.get("mom_acc", 0) * 5
        + feats.get("ret_autocorr", 0) * 2
    ))

    # 8. Pattern  (doji, hammer, shoot_star, engulf, hh/ll)
    bull_pat = feats.get("engulf", 0) + feats.get("hammer", 0) + feats.get("higher_high", 0)
    bear_pat = feats.get("shoot_star", 0) + feats.get("lower_low", 0)
    s_pat = max(0., min(1., 0.5 + (bull_pat - bear_pat) * 0.1))

    scores = {
        "score_news":        round(s_news, 4),
        "score_trend":       round(s_trend, 4),
        "score_technical":   round(s_tech, 4),
        "score_volume":      round(s_vol, 4),
        "score_index":       round(s_idx, 4),
        "score_fundamental": round(s_fund, 4),
        "score_momentum":    round(s_mom, 4),
        "score_pattern":     round(s_pat, 4),
    }

    w = SCORE_WEIGHTS
    model_score = (
        w["news_sentiment"]  * s_news  +
        w["past_trend"]      * s_trend +
        w["technical"]       * s_tech  +
        w["volume"]          * s_vol   +
        w["index_breadth"]   * s_idx   +
        w["fundamental"]     * s_fund  +
        w["momentum"]        * s_mom   +
        w["pattern"]         * s_pat
    )
    scores["model_score"] = round(model_score, 4)

    # Weighted contribution of each factor to final score
    scores["contrib_news"]        = round(w["news_sentiment"]  * s_news,  4)
    scores["contrib_trend"]       = round(w["past_trend"]      * s_trend, 4)
    scores["contrib_technical"]   = round(w["technical"]       * s_tech,  4)
    scores["contrib_volume"]      = round(w["volume"]          * s_vol,   4)
    scores["contrib_index"]       = round(w["index_breadth"]   * s_idx,   4)
    scores["contrib_fundamental"] = round(w["fundamental"]     * s_fund,  4)
    scores["contrib_momentum"]    = round(w["momentum"]        * s_mom,   4)
    scores["contrib_pattern"]     = round(w["pattern"]         * s_pat,   4)

    return scores


def build_bar_record(
    trade_date:   str,
    stock:        str,
    candle_15:    dict,
    feats:        dict,
    proba:        float,
    pred_ret:     float,
    trade_fired:  bool,
    trade_outcome: str  = "NONE",   # "WIN" / "LOSS" / "EOD" / "NONE"
    trade_pnl_pct: float = 0.0,
    trade_exit_reason: str = "",
) -> dict:
    """Assemble one complete bar record ready for vector storage."""

    ts  = candle_15["timestamp"]
    op  = candle_15["open"]
    cl  = candle_15["close"]
    hi  = candle_15["high"]
    lo  = candle_15["low"]
    vol = candle_15["volume"]

    candle_type   = classify_candle(op, cl)
    body_pct      = (cl - op) / (op + 1e-9) * 100
    trend_label   = classify_trend(
        feats.get("ma5_slope", 0),
        feats.get("ma10_slope", 0),
        feats.get("r_ma20", 0),
    )

    agg_sent = feats.get("agg_sent", 0)
    news_sent_label = (
        "BULLISH"  if agg_sent >  0.2 else
        "BEARISH"  if agg_sent < -0.2 else
        "NEUTRAL"
    )

    score_data = compute_score_breakdown(feats, proba, pred_ret)

    # Path direction and points awarded
    if candle_type == "POSITIVE":
        path_direction = "POSITIVE"
        points_awarded = +score_data["model_score"]
    elif candle_type == "NEGATIVE":
        path_direction = "NEGATIVE"
        points_awarded = -score_data["model_score"]
    else:
        path_direction = "MIXED"
        points_awarded = score_data["model_score"] * 0.1  # near-zero reward

    # Time slot label
    try:
        t = pd.Timestamp(ts)
        time_slot_label = t.strftime("%H:%M")
    except Exception:
        time_slot_label = str(ts)[:5]

    record = {
        # ── identity ────────────────────────────────────
        "trade_date":    trade_date,
        "stock":         stock,
        "timestamp":     str(ts),
        "time_slot_label": time_slot_label,
        # ── OHLCV ───────────────────────────────────────
        "open_px":       float(op),
        "high_px":       float(hi),
        "low_px":        float(lo),
        "close_px":      float(cl),
        "volume":        float(vol),
        "candle_body_pct": float(body_pct),
        # ── classification ──────────────────────────────
        "candle_type":   candle_type,
        "trend_label":   trend_label,
        "path_direction": path_direction,
        "points_awarded": round(points_awarded, 6),
        # ── model state ─────────────────────────────────
        "model_proba":   round(proba, 6),
        "pred_ret":      round(pred_ret, 6),
        # ── scores ──────────────────────────────────────
        **score_data,
        # ── news context ────────────────────────────────
        "agg_sent":              feats.get("agg_sent", 0),
        "news_sentiment_label":  news_sent_label,
        "market_signal":         feats.get("market_signal_raw", "NEUTRAL"),
        "bull_news":             int(feats.get("bull_news", 0)),
        "bear_news":             int(feats.get("bear_news", 0)),
        "nat_cnt":               int(feats.get("nat_cnt", 0)),
        "int_cnt":               int(feats.get("int_cnt", 0)),
        "stk_cnt":               int(feats.get("stk_cnt", 0)),
        # ── technical features ──────────────────────────
        "rsi":           round(feats.get("rsi", 50), 4),
        "rsi_ob":        int(feats.get("rsi_ob", 0)),
        "rsi_os":        int(feats.get("rsi_os", 0)),
        "macd":          round(feats.get("macd", 0), 6),
        "macd_sig":      round(feats.get("macd_sig", 0), 6),
        "macd_hist":     round(feats.get("macd_hist", 0), 6),
        "stoch_k":       round(feats.get("stoch_k", 50), 4),
        "stoch_ob":      int(feats.get("stoch_ob", 0)),
        "stoch_os":      int(feats.get("stoch_os", 0)),
        "cci":           round(feats.get("cci", 0), 4),
        "atr_pct":       round(feats.get("atr_pct", 0), 6),
        "r_vwap":        round(feats.get("r_vwap", 0), 6),
        "bb_pct_b":      round(feats.get("bb_pct_b", 0.5), 4),
        "bb_width":      round(feats.get("bb_width", 0), 6),
        "bb_squeeze":    int(feats.get("bb_squeeze", 0)),
        # ── returns / trend ─────────────────────────────
        "r1":            round(feats.get("r1", 0), 6),
        "r3":            round(feats.get("r3", 0), 6),
        "r5":            round(feats.get("r5", 0), 6),
        "r10":           round(feats.get("r10", 0), 6),
        "ma5_slope":     round(feats.get("ma5_slope", 0), 6),
        "ma10_slope":    round(feats.get("ma10_slope", 0), 6),
        "r_ma5":         round(feats.get("r_ma5", 0), 6),
        "r_ma10":        round(feats.get("r_ma10", 0), 6),
        "r_ma20":        round(feats.get("r_ma20", 0), 6),
        "mom":           round(feats.get("mom", 0), 6),
        "mom_acc":       round(feats.get("mom_acc", 0), 6),
        "ret_autocorr":  round(feats.get("ret_autocorr", 0), 6),
        # ── volume ──────────────────────────────────────
        "rel_vol":       round(feats.get("rel_vol", 1), 4),
        "vol_spike":     int(feats.get("vol_spike", 0)),
        # ── patterns ────────────────────────────────────
        "doji":          int(feats.get("doji", 0)),
        "hammer":        int(feats.get("hammer", 0)),
        "shoot_star":    int(feats.get("shoot_star", 0)),
        "bull_candle":   int(feats.get("bull_candle", 0)),
        "engulf":        int(feats.get("engulf", 0)),
        "higher_high":   int(feats.get("higher_high", 0)),
        "lower_low":     int(feats.get("lower_low", 0)),
        # ── index context ───────────────────────────────
        "nifty_r":       round(feats.get("nifty_r", 0), 6),
        "niftyit_r":     round(feats.get("niftyit_r", 0), 6),
        "next50_r":      round(feats.get("next50_r", 0), 6),
        "gift_r":        round(feats.get("gift_r", 0), 6),
        "idx_avg":       round(feats.get("idx_avg", 0), 6),
        "idx_bull":      int(feats.get("idx_bull", 0)),
        "diverge_nifty": round(feats.get("diverge_nifty", 0), 6),
        # ── fundamentals ────────────────────────────────
        "sales":         float(feats.get("sales", 0)),
        "profit":        float(feats.get("profit", 0)),
        "opm":           float(feats.get("opm", 0)),
        "profit_margin": float(feats.get("profit_margin", 0)),
        # ── trade outcome ────────────────────────────────
        "trade_fired":        bool(trade_fired),
        "trade_outcome":      trade_outcome,
        "trade_pnl_pct":      round(trade_pnl_pct, 4),
        "trade_exit_reason":  trade_exit_reason,
        # ── candle sub-context ─────────────────────────
        "open_gap":      round(feats.get("open_gap", 0), 6),
        "candle_range":  round(feats.get("candle_range", 0), 6),
        "upper_shadow":  round(feats.get("upper_shadow", 0), 6),
        "lower_shadow":  round(feats.get("lower_shadow", 0), 6),
        "skew20":        round(feats.get("skew20", 0), 4),
        "kurt20":        round(feats.get("kurt20", 0), 4),
        "skew5":         round(feats.get("skew5", 0), 4),
    }
    return record


# ═════════════════════════════════════════════════════════
# QUERY HELPER  (standalone utility)
# ═════════════════════════════════════════════════════════

def query_candle_db(
    candle_type=None,        # "POSITIVE" / "NEGATIVE" / "MIXED" / None
    min_points=None,         # float — minimum |points_awarded|
    trade_outcome=None,      # "WIN" / "LOSS" / "EOD" / "NONE"
    stock=None,              # filter by stock name
    export_csv=None,         # path string → save full DB as CSV
    n_results=20,
):
    """
    Offline query helper.

    Examples
    --------
    # All positive candles for RELIANCE
    query_candle_db(candle_type="POSITIVE", stock="RELIANCE")

    # Export full DB to CSV for training
    query_candle_db(export_csv="training_data.csv")
    """
    if not CHROMA_OK:
        print("chromadb not installed.")
        return None

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    col = client.get_or_create_collection(CHROMA_COLLECTION)

    if export_csv:
        total = col.count()
        res = col.get(
            where={"record_type": "CANDLE_BAR"},
            include=["metadatas"],
            limit=total,
        )
        df = pd.DataFrame(res["metadatas"])
        df.to_csv(export_csv, index=False)
        print(f"Exported {len(df)} rows to {export_csv}")
        return df

    where = {"record_type": "CANDLE_BAR"}
    if candle_type:   where["candle_type"]   = candle_type
    if trade_outcome: where["trade_outcome"]  = trade_outcome
    if stock:         where["stock"]          = stock

    res = col.get(where=where, include=["metadatas", "documents"], limit=n_results)
    df  = pd.DataFrame(res["metadatas"])

    if min_points is not None and "points_awarded" in df.columns:
        df = df[df["points_awarded"].astype(float).abs() >= min_points]

    print(df[["trade_date","stock","timestamp","candle_type",
              "model_score","points_awarded","score_news",
              "score_trend","score_technical","trade_outcome"]
             ].to_string(index=False))
    return df


# ═════════════════════════════════════════════════════════
# TRADING LOGIC  (unchanged from v2, just with extra feats)
# ═════════════════════════════════════════════════════════

def to_15min(candles_1m):
    if not candles_1m:
        return []
    df = pd.DataFrame(candles_1m)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    r = df.resample("15min").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),    close=("close","last"),
        volume=("volume","sum")
    ).dropna()
    return [{"timestamp": str(ts),
             "open": float(v.open),  "high": float(v.high),
             "low":  float(v.low),   "close": float(v.close),
             "volume": float(v.volume)}
            for ts, v in r.iterrows()]


def clean_num(v):
    try:    return float(str(v).replace(",","").replace("%",""))
    except: return 0.0


def _rsi(cl, p=14):
    d = np.diff(cl[-p-1:])
    g = d[d>0].mean()  if (d>0).any()  else 1e-9
    l = (-d[d<0]).mean() if (d<0).any() else 1e-9
    return 100 - 100/(1+g/l)

def _ema(a, s):
    k=2/(s+1); v=a[0]; r=[v]
    for x in a[1:]: v=x*k+v*(1-k); r.append(v)
    return r

def _macd(cl):
    if len(cl)<27: return 0.,0.,0.
    e12=_ema(cl,12); e26=_ema(cl,26)
    ml=[a-b for a,b in zip(e12,e26)]
    sig=_ema(ml[-9:],9)[-1] if len(ml)>=9 else 0.
    return ml[-1],sig,ml[-1]-sig

def _atr(ca, p=14):
    tr=[max(c["high"]-c["low"],
            abs(c["high"]-ca[i-1]["close"]),
            abs(c["low"]-ca[i-1]["close"]))
        for i,c in enumerate(ca[-p:],1) if i>0]
    return np.mean(tr) if tr else 1e-9

def _stoch(ca, p=14):
    sl=ca[-p:]; H=max(c["high"] for c in sl); L=min(c["low"] for c in sl)
    C=sl[-1]["close"]; return (C-L)/(H-L)*100 if H!=L else 50.

def _cci(ca, p=20):
    tp=[(c["high"]+c["low"]+c["close"])/3 for c in ca[-p:]]
    m=np.mean(tp); mad=np.mean(np.abs(np.array(tp)-m))
    return (tp[-1]-m)/(0.015*mad) if mad else 0.

def _vwap(ca):
    tp=np.array([(c["high"]+c["low"]+c["close"])/3 for c in ca])
    vol=np.array([c["volume"] for c in ca])
    return (tp*vol).sum()/vol.sum() if vol.sum() else ca[-1]["close"]

def _boll(cl, p=20):
    ma=np.mean(cl[-p:]); sd=np.std(cl[-p:])
    up=ma+2*sd; lo=ma-2*sd; rng=up-lo
    return ma,up,lo,rng/ma if ma else 0,(cl[-1]-lo)/rng if rng else .5

def _skew(a):
    if len(a)<3: return 0.
    m,s=np.mean(a),np.std(a)
    return np.mean(((np.array(a)-m)/(s+1e-9))**3)

def _kurt(a):
    if len(a)<4: return 0.
    m,s=np.mean(a),np.std(a)
    return np.mean(((np.array(a)-m)/(s+1e-9))**4)-3

def _pat(cur, prev):
    bc=abs(cur["close"]-cur["open"]); bp=abs(prev["close"]-prev["open"])
    rc=cur["high"]-cur["low"]
    ls=min(cur["close"],cur["open"])-cur["low"]
    us=cur["high"]-max(cur["close"],cur["open"])
    bull=cur["close"]>cur["open"]; pbull=prev["close"]>prev["open"]
    return (int(rc>0 and bc/rc<0.1),
            int(rc>0 and ls>2*bc and us<bc),
            int(rc>0 and us>2*bc and ls<bc),
            int(bull),
            int(bull and not pbull and bc>bp))


def build_features(stock_name, c15, cl15, idx, quarterly, news,
                   index_data, encoder, feat_cols):
    if idx < 26:
        raise ValueError("Need idx >= 26")

    cur=c15[idx]; ts=cur["timestamp"]
    cp=cur["close"]; op=cur["open"]; hp=cur["high"]; lp=cur["low"]; vol=cur["volume"]

    sales =clean_num(quarterly.get("Sales",0))
    profit=clean_num(quarterly.get("Net Profit",0))
    opm   =clean_num(quarterly.get("OPM %",0))
    pm    =profit/sales*100 if sales else 0.
    agg_sent=news.get("aggregate_sentiment",0)
    sig=news.get("market_signal","NEUTRAL")
    bull_n=int(sig=="BULLISH"); bear_n=int(sig=="BEARISH")
    nat_c=len(news.get("national_news",[]));  int_c=len(news.get("international_news",[]))
    stk_c=len(news.get("stock_news",[]))

    w20=c15[idx-20:idx+1]; w14=c15[idx-14:idx+1]; w5=c15[idx-5:idx+1]
    cl20=cl15[idx-20:idx+1]; cl14=cl15[idx-14:idx+1]; cl5=cl15[idx-5:idx+1]

    cb=(cp-op)/(cp+1e-9); cr=(hp-lp)/(cp+1e-9)
    us_=(hp-max(cp,op))/(cp+1e-9); ls_=(min(cp,op)-lp)/(cp+1e-9)
    og=(op-c15[idx-1]["close"])/(c15[idx-1]["close"]+1e-9)

    ma5=np.mean(cl15[idx-5:idx]);  ma10=np.mean(cl15[idx-10:idx])
    ma20=np.mean(cl15[idx-20:idx]); std20=np.std(cl15[idx-20:idx])
    rma5=(cp-ma5)/(ma5+1e-9); rma10=(cp-ma10)/(ma10+1e-9); rma20=(cp-ma20)/(ma20+1e-9)
    ma5s=(cl15[idx-1]-cl15[idx-5])/(cl15[idx-5]+1e-9)
    ma10s=(cl15[idx-1]-cl15[idx-10])/(cl15[idx-10]+1e-9)

    r1=(cp-cl15[idx-1])/(cl15[idx-1]+1e-9); r3=(cp-cl15[idx-3])/(cl15[idx-3]+1e-9)
    r5=(cp-cl15[idx-5])/(cl15[idx-5]+1e-9); r10=(cp-cl15[idx-10])/(cl15[idx-10]+1e-9)
    rl1=(cl15[idx-1]-cl15[idx-2])/(cl15[idx-2]+1e-9)
    rl2=(cl15[idx-2]-cl15[idx-3])/(cl15[idx-3]+1e-9); rac=r1*rl1

    rv=_rsi(cl14); rob=int(rv>70); ros=int(rv<30)
    mv,ms,mh=_macd(cl15[idx-26:idx+1])
    av=_atr(w20); apct=av/(cp+1e-9); ca=cr/(apct+1e-9)
    sk=_stoch(w14); sob=int(sk>80); sos=int(sk<20)
    cv=_cci(w20)
    vv=_vwap(c15[:idx+1]); rvwap=(cp-vv)/(vv+1e-9)
    bma,bup,blo,bw,bpb=_boll(cl15[idx-20:idx+1])
    bsq=int(bw<np.mean([(max(cl15[j-20:j])+2*np.std(cl15[j-20:j])
                          -max(cl15[j-20:j])+2*np.std(cl15[j-20:j]))
                         for j in range(idx-5,idx+1) if j>=20] or [0.01])*0.8)
    vm5=np.mean([c15[idx-k]["volume"] for k in range(1,6)])
    rv2=vol/(vm5+1e-9); vsp=int(rv2>2.0)
    sk20=_skew(cl20); ku20=_kurt(cl20); sk5=_skew(cl5)
    mom=r5; macc=r1-rl1
    dj,hm,sh,bc2,en=_pat(cur,c15[idx-1])
    hh=int(hp>max(c["high"] for c in w5[:-1]))
    ll2=int(lp<min(c["low"]  for c in w5[:-1]))
    rma_=rv*mh; va=rv2*apct; pbrsi=bpb*rv/100

    def ir(n):
        lk=index_data.get(n,{})
        c=lk.get(ts)
        if c is None:
            ts_plain=str(pd.Timestamp(ts))[:19]
            for k,v_ in lk.items():
                if str(k)[:19]==ts_plain: c=v_; break
        return (c["close"]-c["open"])/(c["open"]+1e-9) if c else 0.

    ni=ir("NIFTY"); nit=ir("NIFTYIT"); n50=ir("NIFTY_NEXT_50"); gf=ir("GIFTNIFTY")
    ia=np.mean([ni,nit,n50,gf]); ib=sum(1 for x in [ni,nit,n50,gf] if x>0); dn=r1-ni

    try: t=pd.Timestamp(ts); slot=t.hour*4+t.minute//15
    except: slot=0

    enc=int(encoder.transform([stock_name])[0]) if stock_name in encoder.classes_ else 0

    feat_dict = dict(
        stock_enc=enc, time_slot=slot,
        candle_body=cb, candle_range=cr, upper_shadow=us_, lower_shadow=ls_, open_gap=og,
        r1=r1, r3=r3, r5=r5, r10=r10, r1_lag1=rl1, r1_lag2=rl2, ret_autocorr=rac,
        r_ma5=rma5, r_ma10=rma10, r_ma20=rma20, ma5_slope=ma5s, ma10_slope=ma10s, std20=std20,
        rsi=rv, rsi_ob=rob, rsi_os=ros, macd=mv, macd_sig=ms, macd_hist=mh,
        atr_pct=apct, crng_atr=ca, stoch_k=sk, stoch_ob=sob, stoch_os=sos, cci=cv,
        r_vwap=rvwap, bb_width=bw, bb_pct_b=bpb, bb_squeeze=bsq,
        rel_vol=rv2, vol_spike=vsp, skew20=sk20, kurt20=ku20, skew5=sk5,
        mom=mom, mom_acc=macc, doji=dj, hammer=hm, shoot_star=sh,
        bull_candle=bc2, engulf=en, higher_high=hh, lower_low=ll2,
        rsi_macd=rma_, vol_atr=va, price_bb_rsi=pbrsi,
        nifty_r=ni, niftyit_r=nit, next50_r=n50, gift_r=gf,
        idx_avg=ia, idx_bull=ib, diverge_nifty=dn,
        sales=sales, profit=profit, opm=opm, profit_margin=pm,
        agg_sent=agg_sent, bull_news=bull_n, bear_news=bear_n,
        nat_cnt=nat_c, int_cnt=int_c, stk_cnt=stk_c,
        # extra for scoring (not in original feat_cols, harmless extras)
        market_signal_raw=sig,
    )
    return feat_dict


# ═════════════════════════════════════════════════════════
# DATE PICKER DIALOG  (unchanged)
# ═════════════════════════════════════════════════════════

class DatePickerDialog:
    def __init__(self, root):
        self.result   = None
        self.root     = root
        self._dq      = queue.Queue()

        self.win = tk.Toplevel(root)
        self.win.title("Select Trading Date")
        self.win.configure(bg=BG)
        self.win.geometry("440x540")
        self.win.resizable(False, False)
        self.win.grab_set()

        tk.Label(self.win, text="⚡ PAPER TRADER  v3",
                 font=("Courier", 14, "bold"), bg=BG, fg=BLUE).pack(pady=(18, 2))
        tk.Label(self.win, text="Select a date to simulate",
                 font=("Courier", 10), bg=BG, fg=FG2).pack(pady=(0, 12))

        sf = tk.Frame(self.win, bg=BG3, padx=8, pady=6)
        sf.pack(fill="x", padx=20)
        tk.Label(sf, text="🔍", font=("Courier", 10), bg=BG3, fg=GREY).pack(side="left")
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self._filter)
        tk.Entry(sf, textvariable=self.search_var,
                 font=("Courier", 11), bg=BG4, fg=FG,
                 insertbackground=FG, relief="flat", bd=0
                 ).pack(side="left", fill="x", expand=True, padx=(6, 0))

        lf = tk.Frame(self.win, bg=BG3)
        lf.pack(fill="both", expand=True, padx=20, pady=8)
        sb2 = tk.Scrollbar(lf, orient="vertical", bg=BG3,
                           troughcolor=BG4, activebackground=BLUE)
        sb2.pack(side="right", fill="y")
        self.lb = tk.Listbox(lf, font=("Courier", 11),
                             bg=BG3, fg=FG,
                             selectbackground=BG4, selectforeground=BLUE,
                             activestyle="none", relief="flat", bd=0,
                             yscrollcommand=sb2.set)
        self.lb.pack(side="left", fill="both", expand=True)
        sb2.config(command=self.lb.yview)
        self.lb.bind("<Double-Button-1>", lambda e: self._confirm())
        self.lb.bind("<Return>",          lambda e: self._confirm())

        self.lbl_status = tk.Label(self.win, text="Scanning data files…",
                                   font=("Courier", 8), bg=BG, fg=GREY)
        self.lbl_status.pack(pady=(0, 6))

        bf = tk.Frame(self.win, bg=BG)
        bf.pack(pady=(0, 16))
        tk.Button(bf, text=" SIMULATE ", font=("Courier", 11, "bold"),
                  bg=BLUE, fg=BG, relief="flat", padx=12, pady=6,
                  cursor="hand2", command=self._confirm).pack(side="left", padx=8)
        tk.Button(bf, text=" CANCEL ", font=("Courier", 10),
                  bg=BG4, fg=FG2, relief="flat", padx=12, pady=6,
                  cursor="hand2", command=self._cancel).pack(side="left", padx=8)

        self.all_dates = []
        threading.Thread(target=self._scan_dates, daemon=True).start()
        self.win.after(50, self._poll_dq)
        self.win.protocol("WM_DELETE_WINDOW", self._cancel)
        self.win.wait_window()

    def _scan_dates(self):
        dates_set = set()
        files = glob.glob(f"{COMPANIES_FOLDER}/*.json")
        if not files:
            self._dq.put(("error", "⚠ No data files found in Companies/ folder."))
            return
        for file in files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    ds = json.load(f)
                for day in ds.get("data", []):
                    candles = day.get("candles", [])
                    if candles:
                        date_str = str(pd.Timestamp(candles[0]["timestamp"]).date())
                        dates_set.add(date_str)
            except Exception:
                continue
        self._dq.put(("dates", sorted(dates_set, reverse=True)))

    def _poll_dq(self):
        try:
            while True:
                msg = self._dq.get_nowait()
                if msg[0] == "dates":
                    self.all_dates = msg[1]; self._populate_list()
                elif msg[0] == "error":
                    self.lbl_status.config(text=msg[1], fg=RED)
        except queue.Empty:
            pass
        try:
            if self.win.winfo_exists():
                self.win.after(50, self._poll_dq)
        except tk.TclError:
            pass

    def _populate_list(self):
        self.lb.delete(0, tk.END)
        for d in self.all_dates:
            self.lb.insert(tk.END, f"  {d}")
        if self.all_dates:
            self.lb.selection_set(0)
            self.lbl_status.config(
                text=f"{len(self.all_dates)} trading days  •  double-click or press SIMULATE",
                fg=FG2)
        else:
            self.lbl_status.config(text="⚠ No dates found.", fg=RED)

    def _filter(self, *_):
        q = self.search_var.get().strip()
        self.lb.delete(0, tk.END)
        for d in self.all_dates:
            if q in d: self.lb.insert(tk.END, f"  {d}")
        if self.lb.size(): self.lb.selection_set(0)

    def _confirm(self):
        sel = self.lb.curselection()
        if not sel:
            messagebox.showwarning("No date selected",
                                   "Please select a trading date first.",
                                   parent=self.win)
            return
        self.result = self.lb.get(sel[0]).strip()
        self.win.destroy()

    def _cancel(self):
        self.result = None
        self.win.destroy()


# ═════════════════════════════════════════════════════════
# DASHBOARD
# ═════════════════════════════════════════════════════════

class Dashboard:
    def __init__(self, root: tk.Tk, trade_date: str):
        self.root       = root
        self.trade_date = trade_date

        self.root.title(f"Virtual Paper Trader v3  —  {trade_date}")
        self.root.configure(bg=BG)
        self.root.geometry("1520x860")
        self.root.minsize(1100, 700)

        self.q            = queue.Queue()
        self.equity_pts   = []
        self.total_ticks  = 375
        self.win_count    = 0
        self.loss_count   = 0
        self.positions    = {}
        self.scanner_rows = {}
        self.flash_ids    = {}

        # ── ChromaDB counters for status bar ─────────────
        self.bars_stored      = 0
        self.positive_candles = 0
        self.negative_candles = 0
        self.mixed_candles    = 0

        self._build_ui()
        threading.Thread(target=self._run_sim, daemon=True).start()
        self._poll()

    # ══════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ══════════════════════════════════════════════════════

    def _build_ui(self):
        self._style_ttk()

        hdr = tk.Frame(self.root, bg=BG2, pady=7)
        hdr.pack(fill="x")
        tk.Label(hdr, text="⚡ PAPER TRADER  v3", font=("Courier",15,"bold"),
                 bg=BG2, fg=BLUE).pack(side="left", padx=14)
        tk.Label(hdr, text=f"[ {self.trade_date} ]",
                 font=("Courier",12,"bold"), bg=BG2, fg=ORANGE).pack(side="left", padx=6)
        self.lbl_time = tk.Label(hdr, text="──:──", font=("Courier",14,"bold"),
                                  bg=BG2, fg=YELLOW)
        self.lbl_time.pack(side="left", padx=10)
        self.lbl_status = tk.Label(hdr, text="LOADING…", font=("Courier",10),
                                    bg=BG2, fg=GREY)
        self.lbl_status.pack(side="left", padx=6)
        tk.Label(hdr, text="Session", font=("Courier",8), bg=BG2, fg=FG2
                 ).pack(side="right", padx=(0,4))
        self.prog_var = tk.DoubleVar(value=0)
        self.prog = ttk.Progressbar(hdr, variable=self.prog_var, maximum=100,
                                     length=180, style="slim.Horizontal.TProgressbar")
        self.prog.pack(side="right", padx=(0,14))

        ribbon = tk.Frame(self.root, bg=BG3, pady=5)
        ribbon.pack(fill="x")

        def stat_box(label, sv, color=FG):
            f = tk.Frame(ribbon, bg=BG3, padx=14); f.pack(side="left")
            tk.Label(f, text=label, font=("Courier",7), bg=BG3, fg=FG2).pack(anchor="w")
            lbl = tk.Label(f, textvariable=sv, font=("Courier",12,"bold"), bg=BG3, fg=color)
            lbl.pack(anchor="w")
            tk.Frame(ribbon, bg=ACCENT, width=1, height=30).pack(side="left")
            return lbl

        self.sv_equity   = tk.StringVar(value=f"₹{STARTING_CAPITAL:,.0f}")
        self.sv_cash     = tk.StringVar(value=f"₹{STARTING_CAPITAL:,.0f}")
        self.sv_open_val = tk.StringVar(value="₹0")
        self.sv_pnl      = tk.StringVar(value="₹0.00  (0.00%)")
        self.sv_trades   = tk.StringVar(value="0")
        self.sv_wins     = tk.StringVar(value="0  (0%)")
        self.sv_losses   = tk.StringVar(value="0  (0%)")
        self.sv_slots    = tk.StringVar(value=f"0 / {MAX_POSITIONS}")
        self.sv_db       = tk.StringVar(value="DB: 0 bars")

        self.lbl_equity  = stat_box("EQUITY",       self.sv_equity,   WHITE)
        stat_box("CASH",                             self.sv_cash,     FG)
        stat_box("IN POSITIONS",                     self.sv_open_val, FG)
        self.lbl_pnl_box = stat_box("NET P&L",       self.sv_pnl,      FG)
        stat_box("TRADES",                           self.sv_trades,   FG)
        self.lbl_wins    = stat_box("WINS",          self.sv_wins,     GREEN)
        self.lbl_losses  = stat_box("LOSSES",        self.sv_losses,   RED)
        stat_box("OPEN SLOTS",                       self.sv_slots,    BLUE)
        stat_box("VECTOR DB",                        self.sv_db,       PURPLE)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=3, pady=3)

        left = tk.Frame(body, bg=BG2, bd=0)
        left.pack(side="left", fill="both", expand=True, padx=(0,3))
        self._section_label(left, "📈  EQUITY CURVE")
        self.eq_canvas = tk.Canvas(left, bg=BG, highlightthickness=0)
        self.eq_canvas.pack(fill="both", expand=True, padx=4, pady=(0,4))
        self.eq_canvas.bind("<Configure>", lambda e: self._draw_equity())

        # ── candle type mini-bar ─────────────────────────
        cbar = tk.Frame(left, bg=BG2)
        cbar.pack(fill="x", padx=4, pady=(0,4))
        tk.Label(cbar, text="CANDLE TYPES:  ", font=("Courier",8), bg=BG2, fg=FG2).pack(side="left")
        self.sv_pos  = tk.StringVar(value="▲ 0")
        self.sv_neg  = tk.StringVar(value="▼ 0")
        self.sv_mix  = tk.StringVar(value="◆ 0")
        tk.Label(cbar, textvariable=self.sv_pos,  font=("Courier",9,"bold"), bg=BG2, fg=GREEN ).pack(side="left", padx=6)
        tk.Label(cbar, textvariable=self.sv_neg,  font=("Courier",9,"bold"), bg=BG2, fg=RED   ).pack(side="left", padx=6)
        tk.Label(cbar, textvariable=self.sv_mix,  font=("Courier",9,"bold"), bg=BG2, fg=YELLOW).pack(side="left", padx=6)

        mid = tk.Frame(body, bg=BG, width=430)
        mid.pack(side="left", fill="both", padx=(0,3))
        mid.pack_propagate(False)

        pos_outer = tk.Frame(mid, bg=BG2)
        pos_outer.pack(fill="x", pady=(0,3))
        self._section_label(pos_outer, "📂  OPEN POSITIONS")
        self.pos_frame = tk.Frame(pos_outer, bg=BG2)
        self.pos_frame.pack(fill="x", padx=4, pady=(0,5))

        log_outer = tk.Frame(mid, bg=BG2)
        log_outer.pack(fill="both", expand=True)
        self._section_label(log_outer, "📋  TRADE LOG")
        self._build_trade_log(log_outer)

        right = tk.Frame(body, bg=BG2, width=290)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self._section_label(right, "📡  SIGNAL SCANNER")
        tk.Label(right, text=f"threshold ≥ {MIN_CONFIDENCE:.2f}   reward ×{REWARD_MULT}",
                 font=("Courier",7), bg=BG2, fg=GREY).pack(anchor="w", padx=8, pady=(0,4))
        self.scanner_outer = tk.Frame(right, bg=BG2)
        self.scanner_outer.pack(fill="both", expand=True, padx=4)

        # ── score breakdown panel ────────────────────────
        self._section_label(right, "🧠  SCORE BREAKDOWN  (last signal)")
        self.score_frame = tk.Frame(right, bg=BG2)
        self.score_frame.pack(fill="x", padx=6, pady=(0,6))
        self._score_labels = {}
        factors = [
            ("news",    "News Sent."),
            ("trend",   "Past Trend"),
            ("technical","Technical"),
            ("volume",  "Volume"),
            ("index",   "Idx Breadth"),
            ("fundamental","Fundamental"),
            ("momentum","Momentum"),
            ("pattern", "Pattern"),
        ]
        for key, label in factors:
            row = tk.Frame(self.score_frame, bg=BG2)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{label:<12}", font=("Courier",7), bg=BG2, fg=FG2,
                     width=12, anchor="w").pack(side="left")
            bar_bg = tk.Frame(row, bg=ACCENT, height=8, width=100)
            bar_bg.pack(side="left", padx=(2,2))
            bar_fg = tk.Frame(bar_bg, bg=GREY, height=8, width=0)
            bar_fg.place(x=0,y=0,height=8)
            lbl_v = tk.Label(row, text="0.000", font=("Courier",7), bg=BG2, fg=GREY, width=6)
            lbl_v.pack(side="left")
            self._score_labels[key] = {"bar_fg": bar_fg, "bar_bg": bar_bg, "lbl": lbl_v}

    def _section_label(self, parent, text):
        tk.Frame(parent, bg=ACCENT, height=1).pack(fill="x")
        tk.Label(parent, text=text, font=("Courier",9,"bold"),
                 bg=parent.cget("bg"), fg=BLUE, pady=4).pack(anchor="w", padx=8)

    def _style_ttk(self):
        s = ttk.Style(); s.theme_use("clam")
        s.configure("slim.Horizontal.TProgressbar",
                     troughcolor=BG3, background=BLUE, thickness=5)
        s.configure("Treeview", background=BG3, foreground=FG, fieldbackground=BG3,
                     font=("Courier",8), rowheight=19)
        s.configure("Treeview.Heading", background=BG4, foreground=BLUE,
                     font=("Courier",8,"bold"), relief="flat")
        s.map("Treeview", background=[("selected","#2a4a6b")],
              foreground=[("selected",WHITE)])

    def _build_trade_log(self, parent):
        cols = ("Time","Stock","Entry","Exit","P&L","Reason")
        self.tree = ttk.Treeview(parent, columns=cols, show="headings",
                                  height=16, selectmode="none")
        widths = {"Time":42,"Stock":84,"Entry":66,"Exit":66,"P&L":76,"Reason":76}
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=widths[c], anchor="center", stretch=False)
        self.tree.tag_configure("win",  background=DKGREEN, foreground=GREEN)
        self.tree.tag_configure("loss", background=DKRED,   foreground=RED)
        self.tree.tag_configure("eod",  background=BG4,     foreground=YELLOW)
        sb = ttk.Scrollbar(parent, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(4,0), pady=(0,4))
        sb.pack(side="right", fill="y", pady=(0,4), padx=(0,4))

    # ══════════════════════════════════════════════════════
    # EQUITY CURVE
    # ══════════════════════════════════════════════════════

    def _draw_equity(self):
        c = self.eq_canvas; c.delete("all")
        W = c.winfo_width(); H = c.winfo_height()
        if W < 20 or H < 20: return
        PL, PR, PT, PB = 52, 12, 18, 30
        if len(self.equity_pts) < 2:
            c.create_text(W//2, H//2, text="Awaiting data…", font=("Courier",11), fill=GREY)
            return
        ticks    = [t for t, _ in self.equity_pts]
        equities = [e for _, e in self.equity_pts]
        mn = min(equities + [STARTING_CAPITAL]) - 50
        mx = max(equities + [STARTING_CAPITAL]) + 50
        rng = mx - mn or 1
        t_mn = ticks[0]; t_mx = max(ticks[-1], self.total_ticks-1); t_rng = t_mx - t_mn or 1
        def sx(t): return PL + (t-t_mn)/t_rng*(W-PL-PR)
        def sy(v): return PT + (1-(v-mn)/rng)*(H-PT-PB)
        for i in range(6):
            val = mn + i*rng/5; y = sy(val)
            c.create_line(PL, y, W-PR, y, fill=BG4, dash=(3,5))
            c.create_text(PL-4, y, text=f"₹{val/1000:.1f}k", font=("Courier",7), fill=GREY, anchor="e")
        for h in range(9, 16):
            frac=(h*60-9*60)/(6*60+15); x=PL+frac*(W-PL-PR)
            if PL<=x<=W-PR:
                c.create_line(x,H-PB,x,H-PB+4,fill=GREY)
                c.create_text(x,H-PB+10,text=f"{h:02d}:00",font=("Courier",7),fill=GREY)
        y0=sy(STARTING_CAPITAL)
        c.create_line(PL,y0,W-PR,y0,fill=GREY,dash=(6,4),width=1)
        c.create_text(W-PR+2,y0,text="Start",font=("Courier",7),fill=GREY,anchor="w")
        pts=[(sx(t),sy(e)) for t,e in self.equity_pts]
        y0c=sy(STARTING_CAPITAL)
        poly_above,poly_below=[],[]
        cur_above,cur_below=[],[]
        for px,py in pts:
            if py<=y0c:
                if cur_below:
                    poly_below.append(cur_below+[(cur_below[-1][0],y0c),(cur_below[0][0],y0c)]); cur_below=[]
                cur_above.append((px,py))
            else:
                if cur_above:
                    poly_above.append(cur_above+[(cur_above[-1][0],y0c),(cur_above[0][0],y0c)]); cur_above=[]
                cur_below.append((px,py))
        if cur_above: poly_above.append(cur_above+[(cur_above[-1][0],y0c),(cur_above[0][0],y0c)])
        if cur_below: poly_below.append(cur_below+[(cur_below[-1][0],y0c),(cur_below[0][0],y0c)])
        for segs in poly_above:
            flat=[coord for p in segs for coord in p]
            if len(flat)>=6: c.create_polygon(flat,fill="#1a3a2a",outline="")
        for segs in poly_below:
            flat=[coord for p in segs for coord in p]
            if len(flat)>=6: c.create_polygon(flat,fill="#3a1a1a",outline="")
        for i in range(len(pts)-1):
            col=GREEN if equities[i]>=STARTING_CAPITAL else RED
            c.create_line(pts[i][0],pts[i][1],pts[i+1][0],pts[i+1][1],fill=col,width=2)
        lx,ly=pts[-1]; cur_eq=equities[-1]
        col=GREEN if cur_eq>=STARTING_CAPITAL else RED
        c.create_oval(lx-5,ly-5,lx+5,ly+5,fill=col,outline=BG,width=2)
        pnl=cur_eq-STARTING_CAPITAL; pct=pnl/STARTING_CAPITAL*100
        c.create_text(lx,ly-14,text=f"₹{cur_eq:,.0f}  ({pct:+.2f}%)",
                      font=("Courier",9,"bold"),fill=col,anchor="center")

    # ══════════════════════════════════════════════════════
    # POSITIONS PANEL
    # ══════════════════════════════════════════════════════

    def _refresh_positions(self):
        for w in self.pos_frame.winfo_children(): w.destroy()
        if not self.positions:
            tk.Label(self.pos_frame, text="   — no open positions —",
                     font=("Courier",9), bg=BG2, fg=GREY, pady=6).pack(anchor="w")
            return
        for name, pos in self.positions.items():
            cur_px  = pos.get("cur_px", pos["entry"])
            pnl_abs = (cur_px-pos["entry"])*pos["shares"]
            pnl_pct = pnl_abs/pos["invested"]*100 if pos["invested"] else 0
            col     = GREEN if pnl_abs >= 0 else RED
            bg_card = "#182030" if pnl_abs >= 0 else "#2a1520"
            card = tk.Frame(self.pos_frame, bg=bg_card, pady=5, padx=8)
            card.pack(fill="x", pady=(0,3))
            r1 = tk.Frame(card, bg=bg_card); r1.pack(fill="x")
            tk.Label(r1, text=name, font=("Courier",10,"bold"), bg=bg_card, fg=YELLOW).pack(side="left")
            tk.Label(r1, text=f"₹{cur_px:.2f}", font=("Courier",10,"bold"), bg=bg_card, fg=col).pack(side="right")
            r2 = tk.Frame(card, bg=bg_card); r2.pack(fill="x")
            tk.Label(r2, text=f"Entry ₹{pos['entry']:.2f}  ×{pos['shares']:.1f} sh",
                     font=("Courier",8), bg=bg_card, fg=FG2).pack(side="left")
            tk.Label(r2, text=f"{pnl_pct:+.2f}%", font=("Courier",8,"bold"), bg=bg_card, fg=col).pack(side="right")
            r3 = tk.Frame(card, bg=bg_card); r3.pack(fill="x")
            tk.Label(r3, text=f"SL ₹{pos['stop']:.2f}", font=("Courier",8), bg=bg_card, fg=RED).pack(side="left")
            tk.Label(r3, text=f"TP ₹{pos['target']:.2f}", font=("Courier",8), bg=bg_card, fg=GREEN).pack(side="left", padx=10)
            tk.Label(r3, text=f"₹{pnl_abs:+.1f}", font=("Courier",8,"bold"), bg=bg_card, fg=col).pack(side="right")
            if pos.get("trail"):
                tk.Label(card, text="⟳ trailing stop active", font=("Courier",7), bg=bg_card, fg=ORANGE).pack(anchor="w")
            pbar = tk.Canvas(card, height=5, bg=ACCENT, highlightthickness=0)
            pbar.pack(fill="x", pady=(3,0)); pbar.update_idletasks()
            bw = pbar.winfo_width() or 380
            sl=pos["stop"]; tp=pos["target"]; rng_=tp-sl
            if rng_>0:
                frac=max(0.,min(1.,(cur_px-sl)/rng_))
                pbar.create_rectangle(0,0,int(bw*frac),5,fill=col,outline="")

    # ══════════════════════════════════════════════════════
    # SIGNAL SCANNER
    # ══════════════════════════════════════════════════════

    def _ensure_scanner_row(self, name):
        if name in self.scanner_rows: return
        f = tk.Frame(self.scanner_outer, bg=BG3, pady=4, padx=6)
        f.pack(fill="x", pady=1)
        tk.Label(f, text=f"{name[:13]:<13}", font=("Courier",8,"bold"),
                 bg=BG3, fg=FG, width=13, anchor="w").pack(side="left")
        bar_bg = tk.Frame(f, bg=ACCENT, height=12, width=88)
        bar_bg.pack(side="left", padx=(4,0)); bar_bg.pack_propagate(False)
        bar_fill = tk.Frame(bar_bg, bg=GREY, height=12, width=0)
        bar_fill.place(x=0, y=0, height=12)
        bar_lbl = tk.Label(bar_bg, text="", font=("Courier",7), bg=GREY, fg=BG)
        bar_lbl.place(relx=0.5, rely=0.5, anchor="center")
        lbl_p = tk.Label(f, text=" p=-.---", font=("Courier",8), bg=BG3, fg=GREY, width=8)
        lbl_p.pack(side="left", padx=(4,0))
        lbl_r = tk.Label(f, text=" -.---% ", font=("Courier",8), bg=BG3, fg=GREY, width=9)
        lbl_r.pack(side="left")
        self.scanner_rows[name] = {"frame":f,"bar_bg":bar_bg,"bar_fill":bar_fill,
                                    "bar_lbl":bar_lbl,"lbl_p":lbl_p,"lbl_r":lbl_r}

    def _update_scanner(self, name, proba, pred_ret, fired):
        self._ensure_scanner_row(name)
        row = self.scanner_rows[name]
        bg_col=BG3; bar_col=GREY
        if fired:   bg_col=DKGREEN; bar_col=GREEN
        elif proba>=MIN_CONFIDENCE*0.7: bar_col=BLUE
        bw=row["bar_bg"].winfo_width() or 88
        fw=max(1,int(bw*min(proba,1.0)))
        row["bar_fill"].place(x=0,y=0,width=fw,height=12)
        row["bar_fill"].config(bg=bar_col)
        row["bar_lbl"].config(text=f"{proba:.2f}",bg=bar_col)
        row["bar_bg"].config(bg=ACCENT)
        pc=GREEN if pred_ret>0 else (RED if pred_ret<0 else GREY)
        row["lbl_p"].config(text=f"p={proba:.3f}",fg=bar_col,bg=bg_col)
        row["lbl_r"].config(text=f"{pred_ret*100:+.3f}%",fg=pc,bg=bg_col)
        for w in (row["frame"],row["lbl_p"],row["lbl_r"]): w.config(bg=bg_col)
        if fired:
            if name in self.flash_ids: self.root.after_cancel(self.flash_ids[name])
            self.flash_ids[name]=self.root.after(1200,lambda n=name: self._reset_scanner_row(n))

    def _reset_scanner_row(self, name):
        if name not in self.scanner_rows: return
        row=self.scanner_rows[name]
        for w in (row["frame"],row["lbl_p"],row["lbl_r"]):
            try: w.config(bg=BG3)
            except: pass

    def _update_score_breakdown(self, scores: dict):
        factor_map = {
            "news":        ("score_news",        "contrib_news"),
            "trend":       ("score_trend",        "contrib_trend"),
            "technical":   ("score_technical",    "contrib_technical"),
            "volume":      ("score_volume",       "contrib_volume"),
            "index":       ("score_index",        "contrib_index"),
            "fundamental": ("score_fundamental",  "contrib_fundamental"),
            "momentum":    ("score_momentum",     "contrib_momentum"),
            "pattern":     ("score_pattern",      "contrib_pattern"),
        }
        for key, (score_k, contrib_k) in factor_map.items():
            if key not in self._score_labels: continue
            val  = scores.get(score_k, 0)
            cont = scores.get(contrib_k, 0)
            lbl_dict = self._score_labels[key]
            col = GREEN if val > 0.6 else (RED if val < 0.4 else YELLOW)
            bw = lbl_dict["bar_bg"].winfo_width() or 100
            fw = max(1, int(bw * val))
            lbl_dict["bar_fg"].place(x=0, y=0, width=fw, height=8)
            lbl_dict["bar_fg"].config(bg=col)
            lbl_dict["lbl"].config(text=f"{cont:.3f}", fg=col)

    # ══════════════════════════════════════════════════════
    # EVENT POLLING
    # ══════════════════════════════════════════════════════

    def _poll(self):
        processed = 0
        try:
            while processed < 30:
                ev = self.q.get_nowait(); self._handle(ev); processed += 1
        except queue.Empty: pass
        self.root.after(40, self._poll)

    def _handle(self, ev):
        kind = ev[0]
        if kind == "status":
            self.lbl_status.config(text=ev[1])
        elif kind == "init":
            _, total, names = ev
            self.total_ticks = total
            for n in names: self._ensure_scanner_row(n)
        elif kind == "tick":
            _, tick_i, ts, cash, equity, pnl, open_val, pos_snap = ev
            self.equity_pts.append((tick_i, equity))
            self.positions = pos_snap
            self.lbl_time.config(text=ts)
            pct=pnl/STARTING_CAPITAL*100; col=GREEN if pnl>=0 else RED; sign="+" if pnl>=0 else ""
            self.sv_equity.set(f"₹{equity:,.0f}")
            self.sv_cash.set(f"₹{cash:,.0f}")
            self.sv_open_val.set(f"₹{open_val:,.0f}")
            self.sv_pnl.set(f"{sign}₹{pnl:,.0f}  ({pct:+.2f}%)")
            self.lbl_pnl_box.config(fg=col); self.lbl_equity.config(fg=col)
            tot=self.win_count+self.loss_count
            self.sv_trades.set(str(tot))
            wp=self.win_count/tot*100 if tot else 0; lp=self.loss_count/tot*100 if tot else 0
            self.sv_wins.set(f"{self.win_count}  ({wp:.0f}%)")
            self.sv_losses.set(f"{self.loss_count}  ({lp:.0f}%)")
            self.sv_slots.set(f"{len(pos_snap)} / {MAX_POSITIONS}")
            self.prog_var.set(tick_i/max(self.total_ticks-1,1)*100)
            self._refresh_positions(); self._draw_equity()
        elif kind == "exit":
            _, ts, name, entry, exit_px, pnl_val, pnl_pct, reason = ev
            if pnl_val>0 and reason!="EOD CLOSE": self.win_count+=1; tag="win"
            elif reason=="EOD CLOSE":
                tag="eod"
                if pnl_val>0: self.win_count+=1
                else:         self.loss_count+=1
            else: self.loss_count+=1; tag="loss"
            sign="+" if pnl_val>=0 else ""
            self.tree.insert("",0,values=(ts,name,f"₹{entry:.2f}",f"₹{exit_px:.2f}",
                                           f"{sign}₹{pnl_val:.1f}",reason),tags=(tag,))
        elif kind == "signal":
            _, name, proba, pred_ret, fired = ev
            self._update_scanner(name, proba, pred_ret, fired)
        elif kind == "scores":
            _, scores = ev
            self._update_score_breakdown(scores)
        elif kind == "db_update":
            _, stored, pos, neg, mix = ev
            self.bars_stored      = stored
            self.positive_candles = pos
            self.negative_candles = neg
            self.mixed_candles    = mix
            self.sv_db.set(f"DB: {stored} bars")
            self.sv_pos.set(f"▲ {pos}")
            self.sv_neg.set(f"▼ {neg}")
            self.sv_mix.set(f"◆ {mix}")
        elif kind == "done":
            _, final_eq, _ = ev
            pnl=final_eq-STARTING_CAPITAL; col=GREEN if pnl>=0 else RED; sign="+" if pnl>=0 else ""
            self.lbl_status.config(
                text=f"✓ COMPLETE — P&L: {sign}₹{pnl:,.2f}  ({pnl/STARTING_CAPITAL*100:+.2f}%)"
                     f"  |  {self.bars_stored} bars → ChromaDB",
                fg=col)
            self.prog_var.set(100)

    # ══════════════════════════════════════════════════════
    # SIMULATION THREAD
    # ══════════════════════════════════════════════════════

    def _run_sim(self):
        self.q.put(("status", "Loading model bundle…"))
        try:
            bundle   = joblib.load(MODEL_BUNDLE)
            xgb_clf  = bundle["xgb_clf"];  lgbm_clf = bundle["lgbm_clf"]
            rf_clf   = bundle["rf_clf"];   meta_clf  = bundle["meta_clf"]
            xgb_reg  = bundle["xgb_reg"];  lgbm_reg = bundle["lgbm_reg"]
            rf_reg   = bundle["rf_reg"];   meta_reg  = bundle["meta_reg"]
            encoder  = bundle["encoder"];  feat_cols = bundle["feat_cols"]
        except Exception as e:
            self.q.put(("status", f"ERROR loading model: {e}")); return

        # ── init ChromaDB ─────────────────────────────────
        self.q.put(("status", "Initialising ChromaDB…"))
        vstore = CandleVectorStore()

        def model_predict(X_df):
            bc=np.column_stack([xgb_clf.predict_proba(X_df)[:,1],
                                 lgbm_clf.predict_proba(X_df)[:,1],
                                 rf_clf.predict_proba(X_df)[:,1]])
            proba=meta_clf.predict_proba(bc)[:,1]
            br=np.column_stack([xgb_reg.predict(X_df),
                                 lgbm_reg.predict(X_df),
                                 rf_reg.predict(X_df)])
            return float(proba[0]), float(meta_reg.predict(br)[0])

        def kelly(p, win_r):
            if win_r<=0: win_r=STOP_LOSS_PCT*0.5
            b=win_r/STOP_LOSS_PCT; f=(p*b-(1-p))/b
            return max(0., min(f, POSITION_CAP))

        # ── indices ──────────────────────────────────────
        self.q.put(("status","Loading index data…"))
        index_data = {}
        for file in glob.glob(f"{INDICES_FOLDER}/*.json"):
            iname=file.split("\\")[-1].replace("_historical_data.json","")
            with open(file,"r",encoding="utf-8") as f: ds=json.load(f)
            lk={}
            for day in ds["data"]:
                for c_ in to_15min(day["candles"]):
                    lk[str(pd.Timestamp(c_["timestamp"]))[:19]]=c_
                    lk[c_["timestamp"]]=c_
            index_data[iname]=lk

        # ── stocks filtered to chosen date ────────────────
        self.q.put(("status", f"Loading stocks for {self.trade_date}…"))
        target_date = pd.Timestamp(self.trade_date).date()
        stocks = {}

        for file in glob.glob(f"{COMPANIES_FOLDER}/*.json"):
            name=file.split("\\")[-1].replace("_with_quarterly_results.json","")
            with open(file,"r",encoding="utf-8") as f: ds=json.load(f)
            all_days=ds["data"]
            live_day=None
            for day in all_days:
                candles=day.get("candles",[])
                if candles and pd.Timestamp(candles[0]["timestamp"]).date()==target_date:
                    live_day=day; break
            if live_day is None: continue
            live_1min=live_day.get("candles",[])
            if not live_1min: continue
            quarterly=live_day.get("latest_quarterly_result",{})
            news=live_day.get("news_analysis",{})
            ctx_1min=[]
            for day in all_days:
                candles=day.get("candles",[])
                if not candles: continue
                if pd.Timestamp(candles[0]["timestamp"]).date()<target_date:
                    ctx_1min.extend(candles)
                    if len(to_15min(ctx_1min))>=30: break
            stocks[name]={
                "live_1min":  live_1min,
                "context_15": to_15min(ctx_1min),
                "quarterly":  quarterly,
                "news":       news,
            }

        if not stocks:
            self.q.put(("status",
                f"ERROR: No stocks found for {self.trade_date}.")); return

        all_1min_ts=sorted(set(
            c["timestamp"]
            for s in stocks.values()
            for c in s["live_1min"]
            if pd.Timestamp(c["timestamp"]).date()==target_date
        ))
        if not all_1min_ts:
            self.q.put(("status",f"ERROR: Timeline empty for {self.trade_date}.")); return

        self.q.put(("init",len(all_1min_ts),list(stocks.keys())))
        self.q.put(("status",
            f"SIMULATING {self.trade_date}  "
            f"({len(stocks)} stocks, {len(all_1min_ts)} ticks)…"))

        capital        = STARTING_CAPITAL
        positions      = {}
        seen_1min      = {n:[] for n in stocks}
        last_bar_fired = {n:None for n in stocks}

        # ── per-stock trade outcome tracking ─────────────
        # maps stock → {"fired_at": bar_record, "entry_ts": ts}
        pending_bar_records = {}

        # ── counters for ChromaDB ─────────────────────────
        bars_stored = 0
        pos_cnt = neg_cnt = mix_cnt = 0

        for tick_i, ts in enumerate(all_1min_ts):
            ts_dt   = pd.Timestamp(ts)
            is_last = (ts == all_1min_ts[-1])

            for name, s in stocks.items():
                for c_ in s["live_1min"]:
                    if c_["timestamp"]==ts: seen_1min[name].append(c_); break

            # ── exit handling + update pending bar records ─
            for name, pos in list(positions.items()):
                cur1=next((c_ for c_ in reversed(seen_1min[name]) if c_["timestamp"]==ts),None)
                if cur1 is None and seen_1min[name]: cur1=seen_1min[name][-1]
                if cur1 is None: continue
                hi=cur1["high"]; lo=cur1["low"]; px=cur1["close"]
                pos["cur_px"]=px
                if hi>pos["peak"]:
                    pos["peak"]=hi
                    if pos["peak"]>=pos["entry"]*(1+TRAILING_TRIGGER):
                        nts=pos["peak"]*(1-TRAILING_STOP_PCT)
                        if nts>pos["stop"]: pos["stop"]=nts; pos["trail"]=True
                reason=None; xp=px
                if lo<=pos["stop"]:      xp=pos["stop"];   reason="STOP LOSS"
                elif hi>=pos["target"]:  xp=pos["target"]; reason="TARGET HIT"
                elif is_last:                              reason="EOD CLOSE"
                if reason:
                    proceeds=pos["shares"]*xp; pnl_abs=proceeds-pos["invested"]
                    capital+=proceeds
                    pnl_pct_trade=pnl_abs/pos["invested"]*100
                    outcome="WIN" if pnl_abs>0 else ("EOD" if reason=="EOD CLOSE" else "LOSS")

                    # ── update the stored bar record with outcome ──
                    if name in pending_bar_records:
                        pr=pending_bar_records.pop(name)
                        pr["trade_outcome"]    = outcome
                        pr["trade_pnl_pct"]    = round(pnl_pct_trade, 4)
                        pr["trade_exit_reason"]= reason
                        vstore.store_bar(pr)
                        bars_stored+=1
                        ct=pr["candle_type"]
                        if ct=="POSITIVE": pos_cnt+=1
                        elif ct=="NEGATIVE": neg_cnt+=1
                        else: mix_cnt+=1
                        self.q.put(("db_update",bars_stored,pos_cnt,neg_cnt,mix_cnt))

                    self.q.put(("exit",ts_dt.strftime("%H:%M"),name,
                                pos["entry"],xp,pnl_abs,pnl_pct_trade,reason))
                    del positions[name]

            is_bar_close=(ts_dt.minute%15==14)
            free_slots=MAX_POSITIONS-len(positions)

            if (is_bar_close or is_last) and free_slots>0:
                candidates=[]
                for name, s in stocks.items():
                    if name in positions or not seen_1min[name]: continue
                    live_15=to_15min(seen_1min[name])
                    if not live_15: continue
                    full_15=s["context_15"]+live_15
                    cl15=[c_["close"] for c_ in full_15]
                    idx=len(full_15)-1
                    if idx<26: continue
                    bar_ts=full_15[idx]["timestamp"]
                    if last_bar_fired[name]==bar_ts: continue
                    try:
                        feats=build_features(name,full_15,cl15,idx,
                                             s["quarterly"],s["news"],
                                             index_data,encoder,feat_cols)
                    except: continue
                    X=pd.DataFrame([feats])[feat_cols]
                    try: proba,pred_ret=model_predict(X)
                    except: continue
                    last_bar_fired[name]=bar_ts

                    # ── score breakdown ───────────────────
                    score_data=compute_score_breakdown(feats,proba,pred_ret)

                    fired=(proba>=MIN_CONFIDENCE and pred_ret>=MIN_PRED_RETURN)
                    self.q.put(("signal",name,proba,pred_ret,fired))
                    if fired:
                        self.q.put(("scores",score_data))

                    # ── build bar record (outcome TBD on exit) ─
                    cur_candle=full_15[idx]
                    bar_record=build_bar_record(
                        trade_date   = self.trade_date,
                        stock        = name,
                        candle_15    = cur_candle,
                        feats        = feats,
                        proba        = proba,
                        pred_ret     = pred_ret,
                        trade_fired  = fired,
                        trade_outcome= "PENDING",
                        trade_pnl_pct= 0.0,
                    )

                    if fired:
                        score=proba*pred_ret if pred_ret>0 else proba
                        candidates.append(dict(name=name,proba=proba,pred_ret=pred_ret,
                                               price=full_15[idx]["close"],score=score,
                                               bar_record=bar_record))
                    else:
                        # store non-fired bars immediately
                        vstore.store_bar(bar_record)
                        bars_stored+=1
                        ct=bar_record["candle_type"]
                        if ct=="POSITIVE": pos_cnt+=1
                        elif ct=="NEGATIVE": neg_cnt+=1
                        else: mix_cnt+=1
                        self.q.put(("db_update",bars_stored,pos_cnt,neg_cnt,mix_cnt))

                candidates.sort(key=lambda x:x["score"],reverse=True)
                for cand in candidates[:free_slots]:
                    name=cand["name"]; entry=cand["price"]
                    proba=cand["proba"]; pred_ret=cand["pred_ret"]
                    frac=kelly(proba,pred_ret)
                    invest=capital*frac
                    if invest<MIN_INVEST: invest=min(capital*0.05,capital)
                    if invest<MIN_INVEST or invest>capital: continue
                    shares=invest/entry; stop=entry*(1-STOP_LOSS_PCT)
                    eff_r=pred_ret if pred_ret>0 else STOP_LOSS_PCT
                    target=entry*(1+eff_r*REWARD_MULT)
                    capital-=invest
                    positions[name]=dict(shares=shares,entry=entry,stop=stop,target=target,
                                         peak=entry,trail=False,invested=invest,
                                         proba=proba,pred_ret=pred_ret,
                                         entry_ts=ts,cur_px=entry)
                    # park bar record; will be updated with outcome on exit
                    pending_bar_records[name]=cand["bar_record"]

            open_val=sum(
                p["shares"]*(seen_1min[n][-1]["close"] if seen_1min[n] else p["entry"])
                for n,p in positions.items())
            equity=capital+open_val; pnl=equity-STARTING_CAPITAL
            self.q.put(("tick",tick_i,ts_dt.strftime("%H:%M"),
                        capital,equity,pnl,open_val,{n:dict(p) for n,p in positions.items()}))
            time.sleep(SIMULATE_DELAY)

        # ── EOD: flush any still-open positions as PENDING→EOD ──
        for name, pr in pending_bar_records.items():
            pr["trade_outcome"]    = "EOD"
            pr["trade_exit_reason"]= "EOD CLOSE"
            if name in positions:
                pos=positions[name]
                last_px=seen_1min[name][-1]["close"] if seen_1min[name] else pos["entry"]
                pr["trade_pnl_pct"]=round((last_px-pos["entry"])/pos["entry"]*100,4)
            vstore.store_bar(pr)
            bars_stored+=1
            ct=pr["candle_type"]
            if ct=="POSITIVE": pos_cnt+=1
            elif ct=="NEGATIVE": neg_cnt+=1
            else: mix_cnt+=1

        self.q.put(("db_update",bars_stored,pos_cnt,neg_cnt,mix_cnt))

        # ── session summary ───────────────────────────────
        final_eq = capital + sum(
            p["shares"]*(seen_1min[n][-1]["close"] if seen_1min[n] else p["entry"])
            for n,p in positions.items())
        total_trades = self.win_count + self.loss_count
        session_summary = {
            "trade_date":      self.trade_date,
            "final_equity":    round(final_eq, 2),
            "pnl":             round(final_eq-STARTING_CAPITAL, 2),
            "pnl_pct":         round((final_eq-STARTING_CAPITAL)/STARTING_CAPITAL*100, 4),
            "total_trades":    total_trades,
            "wins":            self.win_count,
            "losses":          self.loss_count,
            "win_rate":        round(self.win_count/total_trades*100 if total_trades else 0, 2),
            "bars_stored":     bars_stored,
            "positive_candles":pos_cnt,
            "negative_candles":neg_cnt,
            "mixed_candles":   mix_cnt,
            "stocks_scanned":  len(stocks),
        }
        vstore.store_session_summary(session_summary)
        self.q.put(("done", final_eq, []))


# ═════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    try: root.iconbitmap("")
    except Exception: pass

    picker = DatePickerDialog(root)
    chosen_date = picker.result

    if not chosen_date:
        root.destroy()
    else:
        root.deiconify()
        Dashboard(root, chosen_date)
        root.mainloop()


# ═════════════════════════════════════════════════════════
# OFFLINE UTILITIES  (run directly, no GUI)
# ═════════════════════════════════════════════════════════
#
# Export full database to CSV for model retraining:
#   python paper_trader_chroma.py --export training_data.csv
#
# Query positive candles with high score:
#   python paper_trader_chroma.py --query POSITIVE
#
# ─────────────────────────────────────────────────────────
import sys
if __name__ == "__main__" and len(sys.argv) > 1:
    if sys.argv[1] == "--export" and len(sys.argv) > 2:
        query_candle_db(export_csv=sys.argv[2])
    elif sys.argv[1] == "--query":
        ct = sys.argv[2] if len(sys.argv) > 2 else None
        query_candle_db(candle_type=ct, n_results=30)