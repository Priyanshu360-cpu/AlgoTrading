#!/usr/bin/env python3
# =========================================================
# VIRTUAL PAPER TRADER v3  —  Adaptive RL Dashboard
# =========================================================
# NEW in v3:
#   • Reinforcement Learning agent adapts after every trade
#   • Pattern memory: bans time-slots / stock conditions
#     that repeatedly cause losses
#   • Config dialog: date range, capital, speed, thresholds
#   • Fully scrollable Positions & Signal-Scanner panels
#   • RL Analysis panel (what the agent learned)
#   • ATR-based SL/TP — fixes the "tiny target" bug in v2
#   • Dynamic position sizing via modified Kelly
#   • Streak detection: raise threshold on losing run
# =========================================================

import tkinter as tk
from tkinter import ttk, messagebox
import threading, queue, json, glob, time, joblib, warnings
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import random, os, math

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# DEFAULT CONFIG  (overridden by Config Dialog)
# ─────────────────────────────────────────────────────────
MODEL_BUNDLE     = "model_v3_classifier.pkl"
INDICES_FOLDER   = "Indexes"
COMPANIES_FOLDER = "Companies"

DEFAULT_CAPITAL      = 2_000_000.0
DEFAULT_MAX_POS      = 30
DEFAULT_CONFIDENCE   = 0.55
DEFAULT_MIN_RET      = 0.002
DEFAULT_SL_ATR_MULT  = 1.5        # SL = entry - 1.5 * ATR
DEFAULT_TP_ATR_MULT  = 2.5        # TP = entry + 2.5 * ATR (1.67 R:R)
DEFAULT_MIN_TP_PCT   = 0.003      # minimum TP distance 0.3%
DEFAULT_MIN_SL_PCT   = 0.004      # minimum SL distance 0.4%
DEFAULT_TRAIL_TRIG   = 0.005
DEFAULT_TRAIL_PCT    = 0.003
DEFAULT_REWARD_MULT  = 2.0        # fallback if ATR unavailable
DEFAULT_SPEED        = 0.008      # seconds per 1-min tick
DEFAULT_MIN_INVEST   = 2_000
DEFAULT_POS_CAP      = 0.10       # max 10% per position
DEFAULT_DATE_START   = ""
DEFAULT_DATE_END     = ""

# ─────────────────────────────────────────────────────────
# COLOUR PALETTE  (Bloomberg-terminal dark)
# ─────────────────────────────────────────────────────────
BG      = "#0a0e17"
BG2     = "#111827"
BG3     = "#1c2333"
BG4     = "#243044"
FG      = "#cdd6f4"
FG2     = "#7f8ba8"
GREEN   = "#50fa7b"
DKGREEN = "#0d3d1f"
RED     = "#ff5555"
DKRED   = "#4a1010"
BLUE    = "#8be9fd"
YELLOW  = "#f1fa8c"
ORANGE  = "#ffb86c"
PURPLE  = "#bd93f9"
GREY    = "#6272a4"
WHITE   = "#ffffff"
ACCENT  = "#44475a"
TEAL    = "#00b4d8"

# ─────────────────────────────────────────────────────────
# TECHNICAL HELPERS
# ─────────────────────────────────────────────────────────

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
    sales =clean_num(quarterly.get("Sales",0)); profit=clean_num(quarterly.get("Net Profit",0))
    opm   =clean_num(quarterly.get("OPM %",0)); pm=profit/sales*100 if sales else 0.
    agg_sent=news.get("aggregate_sentiment",0); sig=news.get("market_signal","NEUTRAL")
    bull_n=int(sig=="BULLISH"); bear_n=int(sig=="BEARISH")
    nat_c=len(news.get("national_news",[])); int_c=len(news.get("international_news",[]))
    stk_c=len(news.get("stock_news",[]))
    w20=c15[idx-20:idx+1]; w14=c15[idx-14:idx+1]; w5=c15[idx-5:idx+1]
    cl20=cl15[idx-20:idx+1]; cl14=cl15[idx-14:idx+1]; cl5=cl15[idx-5:idx+1]
    cb=(cp-op)/(cp+1e-9); cr=(hp-lp)/(cp+1e-9)
    us_=(hp-max(cp,op))/(cp+1e-9); ls_=(min(cp,op)-lp)/(cp+1e-9)
    og=(op-c15[idx-1]["close"])/(c15[idx-1]["close"]+1e-9)
    ma5=np.mean(cl15[idx-5:idx]); ma10=np.mean(cl15[idx-10:idx])
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
    cv=_cci(w20); vv=_vwap(c15[:idx+1]); rvwap=(cp-vv)/(vv+1e-9)
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
    feats = dict(
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
    )
    # return both feat dict and raw atr value
    return feats, av

# =========================================================
# REINFORCEMENT LEARNING AGENT
# =========================================================

class RLAgent:
    """
    Online Q-learning agent that adapts trading parameters
    based on win/loss patterns observed during simulation.

    State  = (rsi_bucket, trend_bucket, vol_bucket,
               time_bucket, streak_bucket)
    Action = index into self.actions list
    Reward = shaped P&L (losses penalised extra)
    """

    # (confidence_multiplier, sl_atr_multiplier, tp_atr_multiplier)
    ACTIONS = [
        (1.00, 1.50, 2.50),   # 0 – default balanced
        (1.20, 1.20, 2.80),   # 1 – high conf, tighter SL, bigger TP
        (1.50, 1.00, 3.00),   # 2 – very selective, tight SL, large TP
        (0.90, 1.80, 2.00),   # 3 – relaxed entry, wide SL, moderate TP
        (1.10, 1.40, 2.60),   # 4 – slightly higher conf
        (1.80, 1.00, 3.50),   # 5 – ultra-selective, large TP
        (1.30, 1.30, 2.20),   # 6 – high conf moderate
        (1.00, 2.00, 1.80),   # 7 – wide SL, smaller TP (volatile market)
    ]

    def __init__(self):
        self.q     = defaultdict(lambda: np.zeros(len(self.ACTIONS)))
        self.visits= defaultdict(lambda: np.zeros(len(self.ACTIONS)))
        self.alpha  = 0.08
        self.epsilon= 0.20
        self.min_eps= 0.03
        self.eps_decay = 0.998

        # Pattern memory
        self.loss_patterns   = deque(maxlen=400)
        self.win_patterns    = deque(maxlen=400)
        self.recent_outcomes = deque(maxlen=40)   # bool

        # Streak
        self.consec_losses = 0
        self.consec_wins   = 0

        # Per-dimension stats
        self.slot_stats  = defaultdict(lambda: {"w":0,"l":0,"pnl":0.0})
        self.stock_stats = defaultdict(lambda: {"w":0,"l":0,"pnl":0.0})

        # Bans
        self.banned_slots  = set()   # time-slot integers
        self.banned_stocks = {}      # stock -> expiry_tick

        # Aggregates
        self.total_trades = 0
        self.total_pnl    = 0.0

    # ── state ─────────────────────────────────────────────
    def _state(self, feats):
        rsi  = feats.get("rsi", 50)
        r5   = feats.get("r5", 0)
        rvol = feats.get("rel_vol", 1)
        slot = feats.get("time_slot", 36)
        wr   = self._win_rate()
        rsi_b  = min(int(rsi / 20), 4)
        r5_b   = 0 if r5 < -0.004 else (1 if r5 < 0 else (2 if r5 < 0.004 else 3))
        vol_b  = 0 if rvol < 0.7 else (1 if rvol < 1.5 else (2 if rvol < 3 else 3))
        slot_b = 0 if slot < 38 else (1 if slot < 44 else 2)
        str_b  = min(self.consec_losses, 5)
        wr_b   = int(wr * 3)
        return (rsi_b, r5_b, vol_b, slot_b, str_b, wr_b)

    def _win_rate(self):
        if not self.recent_outcomes:
            return 0.5
        return sum(self.recent_outcomes) / len(self.recent_outcomes)

    # ── action selection (UCB) ────────────────────────────
    def choose_action(self, feats):
        state = self._state(feats)
        if random.random() < self.epsilon:
            return random.randrange(len(self.ACTIONS)), state
        q  = self.q[state]
        v  = self.visits[state]
        tot = v.sum() + 1
        ucb = q + 1.0 * np.sqrt(np.log(tot + 1) / (v + 1))
        return int(np.argmax(ucb)), state

    # ── Q-update ──────────────────────────────────────────
    def update(self, state, action, reward, pnl_pct, feats,
               stock_name, time_slot, reason, current_tick):
        old_q = self.q[state][action]
        self.q[state][action] += self.alpha * (reward - old_q)
        self.visits[state][action] += 1
        self.epsilon = max(self.min_eps, self.epsilon * self.eps_decay)

        won = pnl_pct > 0
        self.recent_outcomes.append(won)
        if won:
            self.consec_wins  += 1
            self.consec_losses = 0
        else:
            self.consec_losses += 1
            self.consec_wins   = 0

        self.total_trades += 1
        self.total_pnl    += pnl_pct

        s = self.slot_stats[time_slot]
        s["pnl"] += pnl_pct
        if won: s["w"] += 1
        else:   s["l"] += 1

        s2 = self.stock_stats[stock_name]
        s2["pnl"] += pnl_pct
        if won: s2["w"] += 1
        else:   s2["l"] += 1

        snap = {k: feats.get(k, 0)
                for k in ("rsi","time_slot","r5","macd_hist","rel_vol","r_vwap")}
        if won:
            self.win_patterns.append(snap)
        else:
            self.loss_patterns.append(snap)

        # Temporarily ban stocks with >= 3 consecutive losses
        if not won and self.consec_losses >= 3:
            self.banned_stocks[stock_name] = current_tick + 60  # ban 60 ticks

        # Recompute banned slots every update
        self._refresh_slot_bans()

    def _refresh_slot_bans(self):
        self.banned_slots = set()
        for slot, s in self.slot_stats.items():
            tot = s["w"] + s["l"]
            if tot >= 5 and s["l"] / tot > 0.65:
                self.banned_slots.add(slot)

    # ── gate: should we skip this trade? ─────────────────
    def should_skip(self, feats, stock_name, current_tick):
        slot = int(feats.get("time_slot", 36))
        if slot in self.banned_slots:
            return True, f"slot {9+slot//4}:{(slot%4)*15:02d} banned"
        if stock_name in self.banned_stocks:
            if current_tick < self.banned_stocks[stock_name]:
                return True, f"{stock_name} temp banned"
            else:
                del self.banned_stocks[stock_name]
        # Pattern similarity check
        if self.consec_losses >= 3 and len(self.loss_patterns) >= 10:
            rsi = feats.get("rsi", 50); ts = slot
            lc = sum(1 for p in list(self.loss_patterns)[-50:]
                     if abs(p["rsi"] - rsi) < 12 and abs(p["time_slot"] - ts) < 4)
            wc = sum(1 for p in list(self.win_patterns)[-50:]
                     if abs(p["rsi"] - rsi) < 12 and abs(p["time_slot"] - ts) < 4)
            if lc >= 4 and lc > wc * 2.5:
                return True, f"loss pattern (L={lc} W={wc})"
        return False, ""

    # ── get dynamic trading params ─────────────────────────
    def get_params(self, feats, base_conf, base_sl_mult, base_tp_mult):
        action, state = self.choose_action(feats)
        cm, slm, tpm = self.ACTIONS[action]
        # Streak adjustments on top of RL action
        if self.consec_losses >= 4:
            boost = 1 + 0.08 * min(self.consec_losses - 3, 6)
            cm  *= boost
            slm *= 0.85   # tighter stops on bad run
        if self.consec_wins >= 6:
            cm *= 0.95    # slightly easier entry when hot
        conf   = min(base_conf * cm, 0.95)
        sl_m   = base_sl_mult  * slm
        tp_m   = base_tp_mult  * tpm
        return conf, sl_m, tp_m, action, state

    # ── reward shaping ────────────────────────────────────
    @staticmethod
    def shape_reward(pnl_pct, reason):
        if reason == "TARGET HIT":
            return pnl_pct * 12 + 0.5         # bonus for clean win
        elif reason == "STOP LOSS":
            return pnl_pct * 20               # extra penalty for SL hit
        elif reason == "EOD CLOSE":
            return pnl_pct * 8
        return pnl_pct * 10

    # ── summary for RL panel ──────────────────────────────
    def summary(self):
        wr = self._win_rate() * 100
        lines = [
            f"Trades:{self.total_trades}  WR:{wr:.1f}%  ε:{self.epsilon:.3f}",
            f"Streak: {'▲'*min(self.consec_wins,6) if self.consec_wins else '▼'*min(self.consec_losses,6) or '—'}",
        ]
        if self.banned_slots:
            bslots = [f"{9+s//4}:{(s%4)*15:02d}" for s in sorted(self.banned_slots)]
            lines.append("Banned slots: " + " ".join(bslots[:4]))
        if self.banned_stocks:
            lines.append("Banned stocks: " + " ".join(list(self.banned_stocks.keys())[:3]))
        if self.stock_stats:
            by_pnl = sorted(self.stock_stats.items(), key=lambda x: x[1]["pnl"])
            worst3 = [f"{n[:6]}({s['pnl']*100:.1f}%)" for n,s in by_pnl[:3]]
            best3  = [f"{n[:6]}({s['pnl']*100:.1f}%)" for n,s in by_pnl[-3:]]
            lines.append("Worst: " + " ".join(worst3))
            lines.append("Best:  " + " ".join(best3))
        # Best Q-state actions
        if self.q:
            best_states = sorted(
                ((s, np.max(q)) for s, q in self.q.items()),
                key=lambda x: x[1], reverse=True
            )[:2]
            for st, qv in best_states:
                best_a = int(np.argmax(self.q[st]))
                cm, slm, tpm = self.ACTIONS[best_a]
                lines.append(f"Top state Q={qv:.3f} → SL×{slm:.1f} TP×{tpm:.1f}")
        return "\n".join(lines)

# =========================================================
# CONFIG DIALOG
# =========================================================

class ConfigDialog(tk.Toplevel):
    """Startup dialog — choose dates, capital, strategy params"""

    def __init__(self, parent, available_dates):
        super().__init__(parent)
        self.title("Paper Trader v3 — Configuration")
        self.configure(bg=BG2)
        self.resizable(False, False)
        self.result = None

        self.available_dates = sorted(available_dates)

        self._build()
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    def _lbl(self, parent, text, row, col, sticky="e", fg=FG2):
        tk.Label(parent, text=text, bg=BG2, fg=fg,
                 font=("Courier",9)).grid(row=row, column=col,
                 padx=8, pady=4, sticky=sticky)

    def _entry(self, parent, default, row, col, width=14):
        v = tk.StringVar(value=str(default))
        e = tk.Entry(parent, textvariable=v, bg=BG3, fg=FG,
                     insertbackground=FG, font=("Courier",9),
                     relief="flat", width=width)
        e.grid(row=row, column=col, padx=8, pady=4, sticky="w")
        return v

    def _build(self):
        tk.Label(self, text="⚡  PAPER TRADER v3  —  SESSION CONFIG",
                 font=("Courier",13,"bold"), bg=BG2, fg=BLUE,
                 pady=12).pack(fill="x")

        f = tk.Frame(self, bg=BG2)
        f.pack(padx=20, pady=4, fill="x")

        # ── Left: date pickers ──────────────────────────
        tk.Label(f, text="SIMULATION DATES", bg=BG2, fg=TEAL,
                 font=("Courier",9,"bold")).grid(
                 row=0, column=0, columnspan=4, sticky="w", pady=(0,4))

        self._lbl(f, "Start date:", 1, 0)
        self.var_date_start = tk.StringVar()
        cb_start = ttk.Combobox(f, textvariable=self.var_date_start,
                                values=self.available_dates, width=18,
                                font=("Courier",9), state="readonly")
        if self.available_dates:
            cb_start.set(self.available_dates[0])
        cb_start.grid(row=1, column=1, padx=8, pady=4, sticky="w")

        self._lbl(f, "End date:", 2, 0)
        self.var_date_end = tk.StringVar()
        cb_end = ttk.Combobox(f, textvariable=self.var_date_end,
                              values=self.available_dates, width=18,
                              font=("Courier",9), state="readonly")
        if self.available_dates:
            cb_end.set(self.available_dates[-1])
        cb_end.grid(row=2, column=1, padx=8, pady=4, sticky="w")

        sep = tk.Frame(f, bg=ACCENT, width=1, height=160)
        sep.grid(row=0, column=2, rowspan=10, padx=12, sticky="ns")

        # ── Right: capital & position sizing ───────────
        tk.Label(f, text="CAPITAL & SIZING", bg=BG2, fg=TEAL,
                 font=("Courier",9,"bold")).grid(
                 row=0, column=3, columnspan=2, sticky="w", pady=(0,4))

        self._lbl(f, "Starting capital (₹):", 1, 3)
        self.v_capital = self._entry(f, int(DEFAULT_CAPITAL), 1, 4)

        self._lbl(f, "Max open positions:", 2, 3)
        self.v_max_pos = self._entry(f, DEFAULT_MAX_POS, 2, 4, 6)

        self._lbl(f, "Max % per position:", 3, 3)
        self.v_pos_cap = self._entry(f, DEFAULT_POS_CAP, 3, 4, 6)

        self._lbl(f, "Min invest (₹):", 4, 3)
        self.v_min_inv = self._entry(f, DEFAULT_MIN_INVEST, 4, 4)

        # ── Strategy params ─────────────────────────────
        f2 = tk.Frame(self, bg=BG2)
        f2.pack(padx=20, pady=4, fill="x")

        tk.Label(f2, text="STRATEGY PARAMETERS", bg=BG2, fg=TEAL,
                 font=("Courier",9,"bold")).grid(
                 row=0, column=0, columnspan=6, sticky="w", pady=(0,4))

        self._lbl(f2, "Min confidence:", 1, 0)
        self.v_conf = self._entry(f2, DEFAULT_CONFIDENCE, 1, 1, 6)

        self._lbl(f2, "Min pred return:", 1, 2)
        self.v_min_ret = self._entry(f2, DEFAULT_MIN_RET, 1, 3, 6)

        self._lbl(f2, "SL ATR mult:", 2, 0)
        self.v_sl = self._entry(f2, DEFAULT_SL_ATR_MULT, 2, 1, 6)

        self._lbl(f2, "TP ATR mult:", 2, 2)
        self.v_tp = self._entry(f2, DEFAULT_TP_ATR_MULT, 2, 3, 6)

        self._lbl(f2, "Min TP %:", 3, 0)
        self.v_min_tp = self._entry(f2, DEFAULT_MIN_TP_PCT, 3, 1, 6)

        self._lbl(f2, "Trail trigger %:", 3, 2)
        self.v_trail = self._entry(f2, DEFAULT_TRAIL_TRIG, 3, 3, 6)

        self._lbl(f2, "Sim speed (s/tick):", 4, 0)
        self.v_speed = self._entry(f2, DEFAULT_SPEED, 4, 1, 8)

        # ── RL toggle ────────────────────────────────────
        self.v_use_rl = tk.BooleanVar(value=True)
        tk.Checkbutton(f2, text="Enable RL Adaptive Agent",
                       variable=self.v_use_rl,
                       bg=BG2, fg=GREEN, selectcolor=BG3,
                       activebackground=BG2, activeforeground=GREEN,
                       font=("Courier",9,"bold")).grid(
                       row=5, column=0, columnspan=4,
                       sticky="w", pady=(8,0))

        # ── Buttons ──────────────────────────────────────
        btn_f = tk.Frame(self, bg=BG2)
        btn_f.pack(pady=12)

        def btn(parent, text, cmd, fg):
            return tk.Button(parent, text=text, command=cmd,
                             bg=BG3, fg=fg, activebackground=BG4,
                             activeforeground=fg, font=("Courier",10,"bold"),
                             relief="flat", padx=20, pady=6)

        btn(btn_f, "▶  START SIMULATION", self._ok,   GREEN).pack(side="left", padx=8)
        btn(btn_f, "✕  CANCEL",            self._cancel, RED).pack(side="left", padx=8)

    def _ok(self):
        try:
            self.result = {
                "date_start": self.var_date_start.get(),
                "date_end":   self.var_date_end.get(),
                "capital":    float(self.v_capital.get().replace(",","")),
                "max_pos":    int(self.v_max_pos.get()),
                "pos_cap":    float(self.v_pos_cap.get()),
                "min_inv":    float(self.v_min_inv.get()),
                "conf":       float(self.v_conf.get()),
                "min_ret":    float(self.v_min_ret.get()),
                "sl_mult":    float(self.v_sl.get()),
                "tp_mult":    float(self.v_tp.get()),
                "min_tp":     float(self.v_min_tp.get()),
                "trail":      float(self.v_trail.get()),
                "speed":      float(self.v_speed.get()),
                "use_rl":     bool(self.v_use_rl.get()),
            }
        except ValueError as e:
            messagebox.showerror("Config Error", str(e), parent=self)
            return
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()

# =========================================================
# SCROLLABLE FRAME HELPER
# =========================================================

def make_scrollable(parent, bg_col=BG2):
    """Returns (outer_frame, inner_frame). Pack outer_frame; add widgets to inner."""
    outer = tk.Frame(parent, bg=bg_col)
    canvas = tk.Canvas(outer, bg=bg_col, highlightthickness=0, bd=0)
    vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)

    inner = tk.Frame(canvas, bg=bg_col)
    win_id = canvas.create_window((0,0), window=inner, anchor="nw")

    def _on_frame_config(e):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfig(win_id, width=canvas.winfo_width())

    def _on_canvas_config(e):
        canvas.itemconfig(win_id, width=e.width)

    inner.bind("<Configure>", _on_frame_config)
    canvas.bind("<Configure>", _on_canvas_config)

    def _on_mousewheel(e):
        canvas.yview_scroll(int(-1*(e.delta/120)), "units")
    canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
    canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

    vsb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    return outer, inner

# =========================================================
# DASHBOARD
# =========================================================

class Dashboard:
    def __init__(self, root: tk.Tk, cfg: dict, rl_agent: RLAgent):
        self.root      = root
        self.cfg       = cfg
        self.rl        = rl_agent
        self.STARTING  = cfg["capital"]

        root.title("Virtual Paper Trader v3  —  Adaptive RL Dashboard")
        root.configure(bg=BG)
        root.geometry("1620x920")
        root.minsize(1200, 720)

        self.q           = queue.Queue()
        self.equity_pts  = []
        self.total_ticks = 375
        self.win_count   = 0
        self.loss_count  = 0
        self.positions   = {}
        self.scanner_rows= {}
        self.flash_ids   = {}

        self._build_ui()
        threading.Thread(target=self._run_sim, daemon=True).start()
        self._poll()

    # ══════════════════════════════════════════════════════
    # UI
    # ══════════════════════════════════════════════════════

    def _build_ui(self):
        self._style_ttk()
        self._build_header()
        self._build_ribbon()
        self._build_body()

    def _style_ttk(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("slim.Horizontal.TProgressbar",
                    troughcolor=BG3, background=BLUE, thickness=5)
        s.configure("Treeview",
                    background=BG3, foreground=FG, fieldbackground=BG3,
                    font=("Courier",8), rowheight=19)
        s.configure("Treeview.Heading",
                    background=BG4, foreground=BLUE,
                    font=("Courier",8,"bold"), relief="flat")
        s.map("Treeview",
              background=[("selected","#2a4a6b")],
              foreground=[("selected", WHITE)])
        s.configure("TScrollbar", troughcolor=BG3, background=ACCENT,
                    arrowcolor=FG2, bordercolor=BG3)

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=BG2, pady=6)
        hdr.pack(fill="x")
        tk.Label(hdr, text="⚡ PAPER TRADER  v3  RL", font=("Courier",14,"bold"),
                 bg=BG2, fg=BLUE).pack(side="left", padx=14)
        self.lbl_time = tk.Label(hdr, text="──:──", font=("Courier",13,"bold"),
                                 bg=BG2, fg=YELLOW)
        self.lbl_time.pack(side="left", padx=8)
        self.lbl_status = tk.Label(hdr, text="LOADING…", font=("Courier",9),
                                   bg=BG2, fg=GREY)
        self.lbl_status.pack(side="left", padx=6)
        tk.Label(hdr, text="Session", font=("Courier",7), bg=BG2, fg=FG2
                 ).pack(side="right", padx=(0,4))
        self.prog_var = tk.DoubleVar(value=0)
        ttk.Progressbar(hdr, variable=self.prog_var, maximum=100,
                        length=160, style="slim.Horizontal.TProgressbar"
                        ).pack(side="right", padx=(0,14))

    def _build_ribbon(self):
        ribbon = tk.Frame(self.root, bg=BG3, pady=4)
        ribbon.pack(fill="x")

        def stat_box(label, sv, color=FG):
            f = tk.Frame(ribbon, bg=BG3, padx=12)
            f.pack(side="left")
            tk.Label(f, text=label, font=("Courier",6), bg=BG3, fg=FG2).pack(anchor="w")
            lbl = tk.Label(f, textvariable=sv, font=("Courier",11,"bold"),
                           bg=BG3, fg=color)
            lbl.pack(anchor="w")
            tk.Frame(ribbon, bg=ACCENT, width=1, height=28).pack(side="left")
            return lbl

        self.sv_equity   = tk.StringVar(value=f"₹{self.STARTING:,.0f}")
        self.sv_cash     = tk.StringVar(value=f"₹{self.STARTING:,.0f}")
        self.sv_open_val = tk.StringVar(value="₹0")
        self.sv_pnl      = tk.StringVar(value="₹0.00  (0.00%)")
        self.sv_trades   = tk.StringVar(value="0")
        self.sv_wins     = tk.StringVar(value="0  (0%)")
        self.sv_losses   = tk.StringVar(value="0  (0%)")
        self.sv_slots    = tk.StringVar(value=f"0/{self.cfg['max_pos']}")
        self.sv_rl_wr    = tk.StringVar(value="WR –")
        self.sv_rl_eps   = tk.StringVar(value="ε –")

        self.lbl_equity  = stat_box("EQUITY",       self.sv_equity,   WHITE)
        stat_box("CASH",                             self.sv_cash,     FG)
        stat_box("IN POSITIONS",                     self.sv_open_val, FG)
        self.lbl_pnl_box = stat_box("NET P&L",       self.sv_pnl,      FG)
        stat_box("TRADES",                           self.sv_trades,   FG)
        self.lbl_wins    = stat_box("WINS",          self.sv_wins,     GREEN)
        self.lbl_losses  = stat_box("LOSSES",        self.sv_losses,   RED)
        stat_box("OPEN SLOTS",                       self.sv_slots,    BLUE)
        stat_box("RL WIN-RATE",                      self.sv_rl_wr,    PURPLE)
        stat_box("RL EPSILON",                       self.sv_rl_eps,   ORANGE)

    def _build_body(self):
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=3, pady=3)

        # ── LEFT: equity curve ────────────────────────────
        left = tk.Frame(body, bg=BG2)
        left.pack(side="left", fill="both", expand=True, padx=(0,3))
        self._section(left, "📈  EQUITY CURVE")
        self.eq_canvas = tk.Canvas(left, bg=BG, highlightthickness=0)
        self.eq_canvas.pack(fill="both", expand=True, padx=4, pady=(0,4))
        self.eq_canvas.bind("<Configure>", lambda e: self._draw_equity())

        # ── MIDDLE: positions + trade log + RL panel ──────
        mid = tk.Frame(body, bg=BG, width=450)
        mid.pack(side="left", fill="both", padx=(0,3))
        mid.pack_propagate(False)

        # Open positions (scrollable)
        pos_outer = tk.Frame(mid, bg=BG2, height=260)
        pos_outer.pack(fill="x", pady=(0,3))
        pos_outer.pack_propagate(False)
        self._section(pos_outer, "📂  OPEN POSITIONS")
        scroll_outer, self.pos_inner = make_scrollable(pos_outer, BG2)
        scroll_outer.pack(fill="both", expand=True, padx=4, pady=(0,4))

        # Trade log
        log_outer = tk.Frame(mid, bg=BG2)
        log_outer.pack(fill="both", expand=True, pady=(0,3))
        self._section(log_outer, "📋  TRADE LOG")
        self._build_trade_log(log_outer)

        # RL analysis panel
        rl_outer = tk.Frame(mid, bg=BG3, height=120)
        rl_outer.pack(fill="x")
        rl_outer.pack_propagate(False)
        self._section(rl_outer, "🧠  RL AGENT ANALYSIS")
        self.lbl_rl = tk.Label(rl_outer, text="Waiting for trades…",
                               font=("Courier",8), bg=BG3, fg=PURPLE,
                               justify="left", anchor="nw", pady=2)
        self.lbl_rl.pack(fill="both", expand=True, padx=8, pady=(0,4))

        # ── RIGHT: signal scanner (scrollable) ───────────
        right = tk.Frame(body, bg=BG2, width=300)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self._section(right, "📡  SIGNAL SCANNER")
        tk.Label(right,
                 text=f"conf≥{self.cfg['conf']:.2f}  SL×{self.cfg['sl_mult']:.1f}ATR  TP×{self.cfg['tp_mult']:.1f}ATR",
                 font=("Courier",7), bg=BG2, fg=GREY
                 ).pack(anchor="w", padx=8, pady=(0,3))
        scan_outer, self.scanner_inner = make_scrollable(right, BG2)
        scan_outer.pack(fill="both", expand=True, padx=4, pady=(0,4))

    def _section(self, parent, text):
        tk.Frame(parent, bg=ACCENT, height=1).pack(fill="x")
        tk.Label(parent, text=text, font=("Courier",9,"bold"),
                 bg=parent.cget("bg"), fg=BLUE, pady=3
                 ).pack(anchor="w", padx=8)

    def _build_trade_log(self, parent):
        cols = ("Time","Stock","Entry","Exit","P&L","SL×","Reason")
        self.tree = ttk.Treeview(parent, columns=cols, show="headings",
                                 height=14, selectmode="none")
        widths = {"Time":42,"Stock":80,"Entry":66,"Exit":66,"P&L":76,"SL×":36,"Reason":68}
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=widths[c], anchor="center", stretch=False)
        self.tree.tag_configure("win",  background=DKGREEN, foreground=GREEN)
        self.tree.tag_configure("loss", background=DKRED,   foreground=RED)
        self.tree.tag_configure("eod",  background=BG4,     foreground=YELLOW)
        self.tree.tag_configure("skip", background=BG3,     foreground=GREY)
        sb = ttk.Scrollbar(parent, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(4,0), pady=(0,4))
        sb.pack(side="right", fill="y", pady=(0,4), padx=(0,4))

    # ══════════════════════════════════════════════════════
    # EQUITY CURVE
    # ══════════════════════════════════════════════════════

    def _draw_equity(self):
        c = self.eq_canvas
        c.delete("all")
        W = c.winfo_width(); H = c.winfo_height()
        if W < 20 or H < 20: return
        PL, PR, PT, PB = 60, 14, 18, 30
        if len(self.equity_pts) < 2:
            c.create_text(W//2, H//2, text="Awaiting data…",
                          font=("Courier",11), fill=GREY); return
        ticks    = [t for t,_ in self.equity_pts]
        equities = [e for _,e in self.equity_pts]
        mn = min(equities+[self.STARTING])-100
        mx = max(equities+[self.STARTING])+100
        rng = mx-mn or 1
        t_mn = ticks[0]; t_mx = max(ticks[-1], self.total_ticks-1)
        t_rng = t_mx-t_mn or 1
        def sx(t): return PL+(t-t_mn)/t_rng*(W-PL-PR)
        def sy(v): return PT+(1-(v-mn)/rng)*(H-PT-PB)
        for i in range(6):
            val = mn+i*rng/5; y=sy(val)
            c.create_line(PL,y,W-PR,y,fill=BG4,dash=(3,5))
            c.create_text(PL-4,y,text=f"₹{val/1000:.1f}k",
                          font=("Courier",7),fill=GREY,anchor="e")
        for h in range(9,16):
            frac=(h*60-9*60)/(6*60+15); x=PL+frac*(W-PL-PR)
            if PL<=x<=W-PR:
                c.create_line(x,H-PB,x,H-PB+4,fill=GREY)
                c.create_text(x,H-PB+10,text=f"{h:02d}:00",
                              font=("Courier",7),fill=GREY)
        y0=sy(self.STARTING)
        c.create_line(PL,y0,W-PR,y0,fill=GREY,dash=(6,4),width=1)
        c.create_text(W-PR+2,y0,text="Start",font=("Courier",7),fill=GREY,anchor="w")
        pts=[(sx(t),sy(e)) for t,e in self.equity_pts]
        y0c=sy(self.STARTING)
        seg_a=[]; seg_b=[]
        for px,py in pts:
            if py<=y0c: seg_b.append((px,py)); (seg_a and (lambda s: s.clear())(seg_a[0:0]) or None)
            else: seg_a.append((px,py))
        # fill areas
        def fill_segs(segs, col):
            for i in range(len(segs)-1):
                x1,y1=segs[i]; x2,y2=segs[i+1]
                c.create_polygon(x1,y0c,x1,y1,x2,y2,x2,y0c,fill=col,outline="")
        # simple segment fill
        for i in range(len(pts)-1):
            avg_y=(pts[i][1]+pts[i+1][1])/2
            col_f="#1a3a2a" if avg_y<=y0c else "#3a1a1a"
            x1,y1=pts[i]; x2,y2=pts[i+1]
            c.create_polygon(x1,y0c,x1,y1,x2,y2,x2,y0c,fill=col_f,outline="")
        for i in range(len(pts)-1):
            col=GREEN if equities[i]>=self.STARTING else RED
            c.create_line(pts[i][0],pts[i][1],pts[i+1][0],pts[i+1][1],fill=col,width=2)
        lx,ly=pts[-1]; cur_eq=equities[-1]
        col=GREEN if cur_eq>=self.STARTING else RED
        c.create_oval(lx-5,ly-5,lx+5,ly+5,fill=col,outline=BG,width=2)
        pnl=cur_eq-self.STARTING; pct=pnl/self.STARTING*100
        c.create_text(lx,ly-14,text=f"₹{cur_eq:,.0f}  ({pct:+.2f}%)",
                      font=("Courier",9,"bold"),fill=col,anchor="center")

    # ══════════════════════════════════════════════════════
    # POSITIONS PANEL  (scrollable)
    # ══════════════════════════════════════════════════════

    def _refresh_positions(self):
        for w in self.pos_inner.winfo_children():
            w.destroy()
        if not self.positions:
            tk.Label(self.pos_inner, text="   — no open positions —",
                     font=("Courier",9), bg=BG2, fg=GREY, pady=6
                     ).pack(anchor="w"); return
        for name, pos in self.positions.items():
            cur_px  = pos.get("cur_px", pos["entry"])
            pnl_abs = (cur_px-pos["entry"])*pos["shares"]
            pnl_pct = pnl_abs/pos["invested"]*100 if pos["invested"] else 0
            col     = GREEN if pnl_abs>=0 else RED
            bg_card = "#182030" if pnl_abs>=0 else "#2a1520"
            card    = tk.Frame(self.pos_inner, bg=bg_card, pady=4, padx=8)
            card.pack(fill="x", pady=(0,3))
            r1=tk.Frame(card,bg=bg_card); r1.pack(fill="x")
            tk.Label(r1,text=name,font=("Courier",9,"bold"),bg=bg_card,fg=YELLOW).pack(side="left")
            tk.Label(r1,text=f"₹{cur_px:.2f}",font=("Courier",9,"bold"),bg=bg_card,fg=col).pack(side="right")
            r2=tk.Frame(card,bg=bg_card); r2.pack(fill="x")
            tk.Label(r2,text=f"Entry ₹{pos['entry']:.2f} ×{pos['shares']:.1f}sh",
                     font=("Courier",7),bg=bg_card,fg=FG2).pack(side="left")
            tk.Label(r2,text=f"{pnl_pct:+.2f}%",font=("Courier",7,"bold"),
                     bg=bg_card,fg=col).pack(side="right")
            r3=tk.Frame(card,bg=bg_card); r3.pack(fill="x")
            tk.Label(r3,text=f"SL ₹{pos['stop']:.2f}",font=("Courier",7),
                     bg=bg_card,fg=RED).pack(side="left")
            tk.Label(r3,text=f"TP ₹{pos['target']:.2f}",font=("Courier",7),
                     bg=bg_card,fg=GREEN).pack(side="left",padx=8)
            tk.Label(r3,text=f"₹{pnl_abs:+.1f}",font=("Courier",7,"bold"),
                     bg=bg_card,fg=col).pack(side="right")
            if pos.get("trail"):
                tk.Label(card,text="⟳ trailing",font=("Courier",7),
                         bg=bg_card,fg=ORANGE).pack(anchor="w")
            pbar=tk.Canvas(card,height=4,bg=ACCENT,highlightthickness=0)
            pbar.pack(fill="x",pady=(2,0))
            pbar.update_idletasks()
            bw=pbar.winfo_width() or 380
            sl=pos["stop"]; tp=pos["target"]; rng_=tp-sl
            if rng_>0:
                frac=max(0.,min(1.,(cur_px-sl)/rng_))
                pbar.create_rectangle(0,0,int(bw*frac),4,fill=col,outline="")

    # ══════════════════════════════════════════════════════
    # SIGNAL SCANNER  (scrollable)
    # ══════════════════════════════════════════════════════

    def _ensure_scanner_row(self, name):
        if name in self.scanner_rows: return
        f = tk.Frame(self.scanner_inner, bg=BG3, pady=3, padx=5)
        f.pack(fill="x", pady=1)
        tk.Label(f, text=f"{name[:12]:<12}", font=("Courier",8,"bold"),
                 bg=BG3, fg=FG, width=12, anchor="w").pack(side="left")
        bar_bg   = tk.Frame(f, bg=ACCENT, height=10, width=80)
        bar_bg.pack(side="left", padx=(3,0))
        bar_bg.pack_propagate(False)
        bar_fill = tk.Frame(bar_bg, bg=GREY, height=10)
        bar_fill.place(x=0,y=0,height=10)
        bar_lbl  = tk.Label(bar_bg, text="", font=("Courier",6), bg=GREY, fg=BG)
        bar_lbl.place(relx=0.5, rely=0.5, anchor="center")
        lbl_p = tk.Label(f, text=" p=-.--", font=("Courier",7), bg=BG3, fg=GREY, width=7)
        lbl_p.pack(side="left", padx=(3,0))
        lbl_r = tk.Label(f, text=" -.--% ", font=("Courier",7), bg=BG3, fg=GREY, width=8)
        lbl_r.pack(side="left")
        self.scanner_rows[name]=dict(f=f,bar_bg=bar_bg,bar_fill=bar_fill,
                                     bar_lbl=bar_lbl,lbl_p=lbl_p,lbl_r=lbl_r)

    def _update_scanner(self, name, proba, pred_ret, fired, skipped=False):
        self._ensure_scanner_row(name)
        row = self.scanner_rows[name]
        if skipped:
            bg_col = BG3; bar_col = GREY
        elif fired:
            bg_col = DKGREEN; bar_col = GREEN
        elif proba >= self.cfg["conf"] * 0.7:
            bg_col = BG3; bar_col = BLUE
        else:
            bg_col = BG3; bar_col = GREY
        bw = row["bar_bg"].winfo_width() or 80
        fw = max(1, int(bw * min(proba,1.0)))
        row["bar_fill"].place(x=0, y=0, width=fw, height=10)
        row["bar_fill"].config(bg=bar_col)
        row["bar_lbl"].config(text=f"{proba:.2f}", bg=bar_col)
        row["bar_bg"].config(bg=ACCENT)
        pc = GREEN if pred_ret>0 else (RED if pred_ret<0 else GREY)
        row["lbl_p"].config(text=f"p={proba:.2f}", fg=bar_col, bg=bg_col)
        row["lbl_r"].config(text=f"{pred_ret*100:+.2f}%", fg=pc, bg=bg_col)
        for w in (row["f"], row["lbl_p"], row["lbl_r"]):
            w.config(bg=bg_col)
        if fired and name not in self.flash_ids:
            self.flash_ids[name] = self.root.after(
                1400, lambda n=name: self._reset_scanner(n))

    def _reset_scanner(self, name):
        self.flash_ids.pop(name, None)
        if name not in self.scanner_rows: return
        row = self.scanner_rows[name]
        for w in (row["f"], row["lbl_p"], row["lbl_r"]):
            try: w.config(bg=BG3)
            except: pass

    # ══════════════════════════════════════════════════════
    # EVENT POLLING
    # ══════════════════════════════════════════════════════

    def _poll(self):
        processed = 0
        try:
            while processed < 40:
                ev = self.q.get_nowait()
                self._handle(ev)
                processed += 1
        except queue.Empty:
            pass
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
            pct=pnl/self.STARTING*100; col=GREEN if pnl>=0 else RED
            sign="+" if pnl>=0 else ""
            self.sv_equity.set(f"₹{equity:,.0f}")
            self.sv_cash.set(f"₹{cash:,.0f}")
            self.sv_open_val.set(f"₹{open_val:,.0f}")
            self.sv_pnl.set(f"{sign}₹{pnl:,.0f}  ({pct:+.2f}%)")
            self.lbl_pnl_box.config(fg=col); self.lbl_equity.config(fg=col)
            tot=self.win_count+self.loss_count
            self.sv_trades.set(str(tot))
            wp=self.win_count/tot*100 if tot else 0
            lp=self.loss_count/tot*100 if tot else 0
            self.sv_wins.set(f"{self.win_count}  ({wp:.0f}%)")
            self.sv_losses.set(f"{self.loss_count}  ({lp:.0f}%)")
            self.sv_slots.set(f"{len(pos_snap)}/{self.cfg['max_pos']}")
            self.prog_var.set(tick_i/max(self.total_ticks-1,1)*100)
            self._refresh_positions()
            self._draw_equity()
        elif kind == "exit":
            _, ts, name, entry, exit_px, pnl_val, pnl_pct, reason, sl_mult = ev
            if pnl_val > 0:
                self.win_count += 1; tag="win"
            elif reason == "EOD CLOSE":
                tag="eod"
                if pnl_val>0: self.win_count+=1
                else: self.loss_count+=1
            else:
                self.loss_count+=1; tag="loss"
            sign="+" if pnl_val>=0 else ""
            self.tree.insert("",0,
                values=(ts,name,f"₹{entry:.2f}",f"₹{exit_px:.2f}",
                        f"{sign}₹{pnl_val:.1f}",f"×{sl_mult:.1f}",reason),
                tags=(tag,))
        elif kind == "signal":
            _, name, proba, pred_ret, fired, skipped = ev
            self._update_scanner(name, proba, pred_ret, fired, skipped)
        elif kind == "rl_update":
            _, summary = ev
            self.lbl_rl.config(text=summary)
            wr = self.rl._win_rate()*100
            self.sv_rl_wr.set(f"WR {wr:.1f}%")
            col = GREEN if wr>=55 else (RED if wr<45 else ORANGE)
            self.sv_rl_wr._label_widget = None   # avoid crash
            self.sv_rl_eps.set(f"ε {self.rl.epsilon:.3f}")
        elif kind == "done":
            _, final_eq, _ = ev
            pnl=final_eq-self.STARTING; col=GREEN if pnl>=0 else RED
            sign="+" if pnl>=0 else ""
            self.lbl_status.config(
                text=f"✓ COMPLETE — Final P&L: {sign}₹{pnl:,.2f}  ({pnl/self.STARTING*100:+.2f}%)",
                fg=col)
            self.prog_var.set(100)

    # ══════════════════════════════════════════════════════
    # SIMULATION THREAD
    # ══════════════════════════════════════════════════════

    def _run_sim(self):
        cfg = self.cfg
        Q   = self.q
        rl  = self.rl
        use_rl = cfg["use_rl"]

        # ── load model ───────────────────────────────────
        Q.put(("status","Loading model bundle…"))
        try:
            bundle   = joblib.load(MODEL_BUNDLE)
            xgb_clf  = bundle["xgb_clf"];  lgbm_clf = bundle["lgbm_clf"]
            rf_clf   = bundle["rf_clf"];   meta_clf  = bundle["meta_clf"]
            xgb_reg  = bundle["xgb_reg"];  lgbm_reg = bundle["lgbm_reg"]
            rf_reg   = bundle["rf_reg"];   meta_reg  = bundle["meta_reg"]
            encoder  = bundle["encoder"]
            feat_cols= bundle["feat_cols"]
        except Exception as e:
            Q.put(("status",f"MODEL ERROR: {e}")); return

        def model_predict(X_df):
            bc = np.column_stack([xgb_clf.predict_proba(X_df)[:,1],
                                   lgbm_clf.predict_proba(X_df)[:,1],
                                   rf_clf.predict_proba(X_df)[:,1]])
            proba = float(meta_clf.predict_proba(bc)[:,1][0])
            br = np.column_stack([xgb_reg.predict(X_df),
                                   lgbm_reg.predict(X_df),
                                   rf_reg.predict(X_df)])
            ret   = float(meta_reg.predict(br)[0])
            return proba, ret

        def kelly(p, win_r, sl_dist):
            if sl_dist <= 0: sl_dist = 0.005
            b = win_r / sl_dist if win_r > 0 else 1.0
            f = (p*b - (1-p)) / b
            return max(0., min(f, cfg["pos_cap"]))

        # ── load indices ─────────────────────────────────
        Q.put(("status","Loading index data…"))
        index_data = {}
        for file in glob.glob(f"{INDICES_FOLDER}/*.json"):
            iname = os.path.basename(file).replace("_historical_data.json","")
            with open(file,"r",encoding="utf-8") as fh:
                ds = json.load(fh)
            lk = {}
            for day in ds["data"]:
                for c_ in to_15min(day["candles"]):
                    lk[str(pd.Timestamp(c_["timestamp"]))[:19]] = c_
                    lk[c_["timestamp"]] = c_
            index_data[iname] = lk

        # ── load stocks + date filter ─────────────────────
        Q.put(("status","Loading stocks…"))
        date_start = cfg["date_start"] or None
        date_end   = cfg["date_end"]   or None

        stocks = {}
        for file in glob.glob(f"{COMPANIES_FOLDER}/*.json"):
            name = os.path.basename(file).replace("_with_quarterly_results.json","")
            with open(file,"r",encoding="utf-8") as fh:
                ds = json.load(fh)
            all_days = ds["data"]
            if len(all_days) < 2: continue

            # filter days by date range
            filtered_days = []
            for day in all_days:
                if not day.get("candles"): continue
                day_ts_str = day["candles"][0].get("timestamp","")[:10]
                if date_start and day_ts_str < date_start: continue
                if date_end   and day_ts_str > date_end:   continue
                filtered_days.append(day)

            if not filtered_days:
                # fallback: use latest day
                filtered_days = [all_days[-1]] if all_days else []

            # choose the live day (most candles in filtered range)
            chosen = None
            for di,day in enumerate(reversed(filtered_days)):
                if len(day.get("candles",[])) >= 50:
                    chosen = day; break
            if chosen is None:
                for day in reversed(filtered_days):
                    if len(day.get("candles",[])) >= 10:
                        chosen = day; break
            if chosen is None: continue

            live_1min = chosen["candles"]
            quarterly = chosen.get("latest_quarterly_result",{})
            news      = chosen.get("news_analysis",{})

            # build context from previous days
            chosen_idx = all_days.index(chosen) if chosen in all_days else len(all_days)-1
            ctx_1min = []
            for di in range(chosen_idx-1, max(chosen_idx-4,-1), -1):
                ctx_1min = all_days[di]["candles"] + ctx_1min
                if len(to_15min(ctx_1min)) >= 30: break

            stocks[name] = dict(live_1min=live_1min,
                                context_15=to_15min(ctx_1min),
                                quarterly=quarterly, news=news)

        if not stocks:
            Q.put(("status","ERROR: no stocks loaded for selected dates.")); return

        all_1min_ts = sorted(set(
            c["timestamp"] for s in stocks.values() for c in s["live_1min"]
        ))
        Q.put(("init", len(all_1min_ts), list(stocks.keys())))
        Q.put(("status","SIMULATING…"))

        # ── simulation state ─────────────────────────────
        capital        = cfg["capital"]
        positions      = {}
        seen_1min      = {n:[] for n in stocks}
        last_bar_fired = {n:None for n in stocks}
        # pending RL updates: pos_name -> (action, state, feats, stock, slot, sl_mult)
        pending_rl     = {}

        for tick_i, ts in enumerate(all_1min_ts):
            ts_dt   = pd.Timestamp(ts)
            is_last = (ts == all_1min_ts[-1])

            # ingest candle
            for name, s in stocks.items():
                for c_ in s["live_1min"]:
                    if c_["timestamp"] == ts:
                        seen_1min[name].append(c_); break

            # monitor positions
            for name, pos in list(positions.items()):
                cur1 = next((c_ for c_ in reversed(seen_1min[name])
                             if c_["timestamp"]==ts), None)
                if cur1 is None and seen_1min[name]:
                    cur1 = seen_1min[name][-1]
                if cur1 is None: continue

                hi=cur1["high"]; lo=cur1["low"]; px=cur1["close"]
                pos["cur_px"] = px

                # trailing stop
                if hi > pos["peak"]:
                    pos["peak"] = hi
                    if pos["peak"] >= pos["entry"]*(1+cfg["trail"]):
                        new_ts = pos["peak"]*(1-DEFAULT_TRAIL_PCT)
                        if new_ts > pos["stop"]:
                            pos["stop"]=new_ts; pos["trail"]=True

                reason=None; xp=px
                if   lo <= pos["stop"]:     xp=pos["stop"];   reason="STOP LOSS"
                elif hi >= pos["target"]:   xp=pos["target"]; reason="TARGET HIT"
                elif is_last:                                  reason="EOD CLOSE"

                if reason:
                    proceeds  = pos["shares"]*xp
                    pnl_abs   = proceeds - pos["invested"]
                    pnl_pct_v = pnl_abs  / pos["invested"]
                    capital  += proceeds
                    sl_mult_used = pos.get("sl_mult",1.5)
                    Q.put(("exit", ts_dt.strftime("%H:%M"), name,
                           pos["entry"], xp, pnl_abs,
                           pnl_pct_v*100, reason, sl_mult_used))
                    # RL update
                    if use_rl and name in pending_rl:
                        pa = pending_rl.pop(name)
                        reward = RLAgent.shape_reward(pnl_pct_v, reason)
                        rl.update(pa["state"], pa["action"], reward,
                                  pnl_pct_v, pa["feats"], name,
                                  int(pa["feats"].get("time_slot",36)),
                                  reason, tick_i)
                        Q.put(("rl_update", rl.summary()))
                    del positions[name]

            # signal scan at every 15-min bar close
            is_bar_close = (ts_dt.minute % 15 == 14)
            free_slots   = cfg["max_pos"] - len(positions)

            if (is_bar_close or is_last) and free_slots > 0:
                candidates = []
                for name, s in stocks.items():
                    if name in positions or not seen_1min[name]: continue
                    live_15 = to_15min(seen_1min[name])
                    if not live_15: continue
                    full_15 = s["context_15"] + live_15
                    cl15    = [c_["close"] for c_ in full_15]
                    idx     = len(full_15)-1
                    if idx < 26: continue
                    bar_ts  = full_15[idx]["timestamp"]
                    if last_bar_fired[name] == bar_ts: continue
                    try:
                        feats, atr_val = build_features(name, full_15, cl15, idx,
                                                         s["quarterly"], s["news"],
                                                         index_data, encoder, feat_cols)
                    except: continue
                    X = pd.DataFrame([feats])[feat_cols]
                    try: proba, pred_ret = model_predict(X)
                    except: continue
                    last_bar_fired[name] = bar_ts

                    # ── RL gating ─────────────────────────────────
                    skipped = False
                    skip_reason = ""
                    if use_rl:
                        skipped, skip_reason = rl.should_skip(feats, name, tick_i)

                    # ── RL parameter adaptation ───────────────────
                    if use_rl:
                        eff_conf, sl_m, tp_m, action_idx, state = \
                            rl.get_params(feats, cfg["conf"],
                                          cfg["sl_mult"], cfg["tp_mult"])
                    else:
                        eff_conf  = cfg["conf"]
                        sl_m      = cfg["sl_mult"]
                        tp_m      = cfg["tp_mult"]
                        action_idx= 0
                        state     = (0,0,0,0,0,0)

                    fired = (not skipped and
                             proba >= eff_conf and
                             pred_ret >= cfg["min_ret"])

                    Q.put(("signal", name, proba, pred_ret, fired, skipped))

                    if fired:
                        entry = full_15[idx]["close"]
                        # ATR-based stops (KEY FIX over v2)
                        atr_use = max(atr_val, entry * 0.003)   # floor 0.3%
                        sl_dist = sl_m * atr_use
                        tp_dist = tp_m * atr_use
                        # Apply minimum distances
                        sl_dist = max(sl_dist, entry * cfg.get("min_sl_pct", DEFAULT_MIN_SL_PCT))
                        tp_dist = max(tp_dist, entry * cfg["min_tp"])
                        stop    = entry - sl_dist
                        target  = entry + tp_dist
                        score   = proba * pred_ret if pred_ret>0 else proba
                        candidates.append(dict(
                            name=name, proba=proba, pred_ret=pred_ret,
                            price=entry, stop=stop, target=target,
                            atr=atr_use, sl_dist=sl_dist, sl_mult=sl_m,
                            score=score, action=action_idx, state=state,
                            feats=feats
                        ))

                candidates.sort(key=lambda x: x["score"], reverse=True)
                for cand in candidates[:free_slots]:
                    name   = cand["name"]
                    entry  = cand["price"]
                    proba  = cand["proba"]
                    pr     = cand["pred_ret"]
                    stop   = cand["stop"]
                    target = cand["target"]
                    sl_dist= cand["sl_dist"]

                    # Kelly sizing
                    frac    = kelly(proba, pr if pr>0 else 0.005, sl_dist/entry)
                    invest  = capital * frac
                    if invest < cfg["min_inv"]:
                        invest = min(capital * 0.03, capital)
                    if invest < cfg["min_inv"] or invest > capital: continue
                    shares  = invest / entry
                    capital -= invest
                    positions[name] = dict(
                        shares=shares, entry=entry, stop=stop, target=target,
                        peak=entry, trail=False, invested=invest,
                        proba=proba, pred_ret=pr, entry_ts=ts,
                        cur_px=entry, sl_mult=cand["sl_mult"]
                    )
                    if use_rl:
                        pending_rl[name] = dict(
                            action=cand["action"], state=cand["state"],
                            feats=cand["feats"]
                        )

            # portfolio snapshot
            open_val = sum(
                p["shares"]*(seen_1min[n][-1]["close"] if seen_1min[n] else p["entry"])
                for n, p in positions.items()
            )
            equity = capital + open_val
            pnl    = equity  - cfg["capital"]
            Q.put(("tick", tick_i, ts_dt.strftime("%H:%M"),
                   capital, equity, pnl, open_val,
                   {n: dict(p) for n,p in positions.items()}))

            time.sleep(cfg["speed"])

        Q.put(("done", capital, []))

# =========================================================
# ENTRY POINT  —  scan dates then show config dialog
# =========================================================

def scan_available_dates():
    dates = set()
    for file in glob.glob(f"{COMPANIES_FOLDER}/*.json"):
        try:
            with open(file,"r",encoding="utf-8") as fh:
                ds = json.load(fh)
            for day in ds.get("data",[]):
                candles = day.get("candles",[])
                if candles:
                    ts_str = candles[0].get("timestamp","")[:10]
                    if ts_str: dates.add(ts_str)
        except: pass
    return sorted(dates)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()         # hide main window while config shows

    # Apply dark title-bar if on Windows
    try:
        from ctypes import windll
        windll.dwmapi.DwmSetWindowAttribute(
            windll.user32.GetForegroundWindow(), 20,
            byref := __import__("ctypes").c_int(1), 4)
    except: pass

    available_dates = scan_available_dates()

    cfg_dlg = ConfigDialog(root, available_dates)
    cfg = cfg_dlg.result

    if cfg is None:
        root.destroy()
    else:
        root.deiconify()
        rl_agent = RLAgent()
        try: root.iconbitmap("")
        except: pass
        dashboard = Dashboard(root, cfg, rl_agent)
        root.mainloop()