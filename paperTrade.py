#!/usr/bin/env python3
# =========================================================
# VIRTUAL PAPER TRADER v5  —  GUARANTEED BUY / SELL
# =========================================================
# KEY FIXES:
#   • Trades ARE guaranteed to happen every session
#   • DEMO MODE: if model fails to load, uses price-action
#     signals (RSI + momentum) so trades always occur
#   • Threshold fallback: if no stock passes MIN_CONFIDENCE,
#     best available candidate is still traded (relaxed mode)
#   • MIN_CONFIDENCE lowered to 0.50 (was 0.55)
#   • FORCE_TRADE_FALLBACK = True — if nothing fires at any
#     bar, pick top-score stock anyway (threshold relaxed)
#   • Margin properly blocks oversized orders
#   • Every BUY is immediately shown; every SELL is shown
#     with realised P&L in the Order Book panel
# =========================================================

import tkinter as tk
from tkinter import ttk
import threading, queue, json, glob, time, warnings
import numpy as np
import pandas as pd

try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# CONFIG  — tweak these freely
# ─────────────────────────────────────────────────────────
MODEL_BUNDLE       = "model_v3_classifier.pkl"
INDICES_FOLDER     = "Indexes"
COMPANIES_FOLDER   = "Companies"

STARTING_CAPITAL   = 100_000.0
MARGIN_RATIO       = 0.25          # 25% margin → 4× leverage
MAX_POSITIONS      = 3
POSITION_CAP       = 0.40          # max 40% of available margin per trade
MIN_CONFIDENCE     = 0.50          # relaxed from 0.55
MIN_PRED_RETURN    = 0.003         # relaxed from 0.005
STOP_LOSS_PCT      = 0.08          # 8% hard stop
TRAILING_TRIGGER   = 0.04          # trailing kicks in at +4%
TRAILING_STOP_PCT  = 0.025         # trail 2.5% below peak
REWARD_MULT        = 2.5           # target = entry × (1 + pred_ret × 2.5)
SIMULATE_DELAY     = 0.012         # seconds per 1-min tick
MIN_INVEST         = 300           # minimum ₹ per trade

# If True: when no stock passes threshold at a bar, we FORCE
# a trade on the best candidate (ignoring confidence gate).
# This guarantees trades happen every session.
FORCE_TRADE_FALLBACK  = True
FORCE_TRADE_BARS      = {9*4+0, 9*4+4, 10*4+0, 11*4+0, 12*4+0, 13*4+0}
                        # slots (hour*4 + quarter) at which we force if idle

BEARISH_CONFIRM_BARS  = 4
NEUTRAL_EXIT_THRESHOLD= 0.0

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


# ─────────────────────────────────────────────────────────
# ORDER BOOK  (margin engine)
# ─────────────────────────────────────────────────────────
class OrderBook:
    def __init__(self, starting_capital: float, margin_ratio: float):
        self.orders           = []
        self.order_id         = 0
        self.starting_capital = starting_capital
        self.margin_ratio     = margin_ratio
        self.available_margin = starting_capital
        self.used_margin      = 0.0
        self.realised_pnl     = 0.0

    def _next_id(self):
        self.order_id += 1
        return self.order_id

    def can_buy(self, invest_amount: float) -> bool:
        return (self.available_margin >= max(invest_amount * self.margin_ratio, MIN_INVEST)
                and invest_amount >= MIN_INVEST)

    def place_buy(self, time_str, stock, qty, price, invest, stop, target, proba, mode="MODEL") -> dict:
        """Deduct full invest from available_margin (paper trade — no broker leverage needed)."""
        self.available_margin -= invest
        self.used_margin      += invest
        order = dict(
            id=self._next_id(), time=time_str, stock=stock,
            side="BUY", qty=qty, price=price, value=invest,
            margin_used=invest * self.margin_ratio,
            stop=stop, target=target, proba=proba,
            pnl=None, status="OPEN", mode=mode,
        )
        self.orders.append(order)
        return order

    def place_sell(self, time_str, stock, qty, price, buy_order, reason) -> dict:
        """Settle the paired BUY, restore capital + P&L to margin pool."""
        proceeds = qty * price
        invest   = buy_order["value"]
        pnl      = proceeds - invest
        self.available_margin += invest + pnl
        self.used_margin      -= invest
        self.realised_pnl     += pnl
        buy_order["status"] = "CLOSED"
        buy_order["pnl"]    = pnl
        order = dict(
            id=self._next_id(), time=time_str, stock=stock,
            side="SELL", qty=qty, price=price, value=proceeds,
            margin_used=0.0, stop=None, target=None,
            proba=buy_order["proba"], pnl=pnl, status=reason, mode=buy_order["mode"],
        )
        self.orders.append(order)
        return order

    def equity(self, open_positions: dict) -> float:
        unrealised = sum(
            (p["cur_px"] - p["entry"]) * p["shares"]
            for p in open_positions.values()
        )
        return self.available_margin + self.used_margin + unrealised

    def net_pnl(self, open_positions: dict) -> float:
        return self.equity(open_positions) - self.starting_capital


# ─────────────────────────────────────────────────────────
# TECHNICAL INDICATORS  (self-contained, no sklearn needed)
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
             "open": float(v.open), "high": float(v.high),
             "low":  float(v.low),  "close": float(v.close),
             "volume": float(v.volume)}
            for ts, v in r.iterrows()]

def clean_num(v):
    try:    return float(str(v).replace(",","").replace("%",""))
    except: return 0.0

def _rsi(cl, p=14):
    if len(cl) < p + 1:
        return 50.0
    d = np.diff(cl[-(p+1):])
    g = d[d > 0].mean() if (d > 0).any() else 1e-9
    l = (-d[d < 0]).mean() if (d < 0).any() else 1e-9
    return 100 - 100 / (1 + g / l)

def _ema(a, s):
    k = 2 / (s + 1); v = a[0]; r = [v]
    for x in a[1:]: v = x * k + v * (1 - k); r.append(v)
    return r

def _macd(cl):
    if len(cl) < 27: return 0., 0., 0.
    e12 = _ema(cl, 12); e26 = _ema(cl, 26)
    ml  = [a - b for a, b in zip(e12, e26)]
    sig = _ema(ml[-9:], 9)[-1] if len(ml) >= 9 else 0.
    return ml[-1], sig, ml[-1] - sig

def _atr(ca, p=14):
    if len(ca) < 2: return 1e-9
    tr = [max(c["high"] - c["low"],
              abs(c["high"] - ca[i-1]["close"]),
              abs(c["low"]  - ca[i-1]["close"]))
          for i, c in enumerate(ca[-p:], 1) if i > 0]
    return np.mean(tr) if tr else 1e-9

def _vwap(ca):
    tp  = np.array([(c["high"] + c["low"] + c["close"]) / 3 for c in ca])
    vol = np.array([c["volume"] for c in ca])
    return (tp * vol).sum() / vol.sum() if vol.sum() else ca[-1]["close"]

def _boll(cl, p=20):
    if len(cl) < p: p = len(cl)
    ma = np.mean(cl[-p:]); sd = np.std(cl[-p:])
    up = ma + 2*sd; lo = ma - 2*sd; rng = up - lo
    return ma, up, lo, rng / ma if ma else 0, (cl[-1] - lo) / rng if rng else .5

# ─────────────────────────────────────────────────────────
# DEMO SIGNAL  (used when model is absent / fails)
# Returns (score 0-1, pred_return, description)
# ─────────────────────────────────────────────────────────
def demo_signal(c15: list, idx: int) -> tuple:
    """
    Pure price-action signal using RSI + MACD + VWAP + Bollinger.
    Returns (confidence, predicted_return) both in [0, 1] / [-1, 1].
    """
    if idx < 20:
        return 0.0, 0.0

    cl = [c["close"] for c in c15[:idx+1]]
    rsi = _rsi(cl, 14)
    _, _, macd_h = _macd(cl)
    vwap = _vwap(c15[:idx+1])
    _, bup, blo, bw, bpb = _boll(cl, 20)
    px   = cl[-1]

    # Momentum: 5-bar return
    mom5 = (px - cl[-6]) / (cl[-6] + 1e-9) if len(cl) >= 6 else 0.

    score = 0.0
    # RSI oversold → bounce expected
    if rsi < 35:  score += 0.25
    elif rsi < 45: score += 0.12
    # MACD histogram crossing up
    if macd_h > 0: score += 0.20
    # Price above VWAP (trend confirmation)
    if px > vwap:  score += 0.18
    # Bollinger: price near lower band → mean reversion buy
    if bpb < 0.25: score += 0.20
    elif bpb < 0.45: score += 0.10
    # Positive short momentum
    if mom5 > 0.002: score += 0.17

    # Predicted return ~ momentum + mean-reversion pull
    pred_ret = max(0.002, abs(mom5) * 0.4 + (1 - bpb) * 0.008)

    return min(score, 0.97), pred_ret


# ─────────────────────────────────────────────────────────
# FULL FEATURE BUILD  (used when model IS available)
# ─────────────────────────────────────────────────────────
def build_features(stock_name, c15, cl15, idx, quarterly, news,
                   index_data, encoder, feat_cols):
    if idx < 26:
        raise ValueError("idx < 26")
    cur  = c15[idx]; ts = cur["timestamp"]
    cp   = cur["close"]; op = cur["open"]
    hp   = cur["high"];  lp = cur["low"]; vol = cur["volume"]

    sales  = clean_num(quarterly.get("Sales", 0))
    profit = clean_num(quarterly.get("Net Profit", 0))
    opm    = clean_num(quarterly.get("OPM %", 0))
    pm     = profit / sales * 100 if sales else 0.
    agg_sent = news.get("aggregate_sentiment", 0)
    sig      = news.get("market_signal", "NEUTRAL")
    bull_n   = int(sig == "BULLISH"); bear_n = int(sig == "BEARISH")
    nat_c    = len(news.get("national_news", []))
    int_c    = len(news.get("international_news", []))
    stk_c    = len(news.get("stock_news", []))

    w20  = c15[idx-20:idx+1]; w14 = c15[idx-14:idx+1]; w5 = c15[idx-5:idx+1]
    cl20 = cl15[idx-20:idx+1]; cl14 = cl15[idx-14:idx+1]; cl5 = cl15[idx-5:idx+1]

    cb  = (cp-op)/(cp+1e-9); cr  = (hp-lp)/(cp+1e-9)
    us_ = (hp-max(cp,op))/(cp+1e-9); ls_ = (min(cp,op)-lp)/(cp+1e-9)
    og  = (op-c15[idx-1]["close"])/(c15[idx-1]["close"]+1e-9)

    ma5  = np.mean(cl15[idx-5:idx]);  ma10  = np.mean(cl15[idx-10:idx])
    ma20 = np.mean(cl15[idx-20:idx]); std20 = np.std(cl15[idx-20:idx])
    rma5  = (cp-ma5)/(ma5+1e-9);   rma10 = (cp-ma10)/(ma10+1e-9)
    rma20 = (cp-ma20)/(ma20+1e-9)
    ma5s  = (cl15[idx-1]-cl15[idx-5])/(cl15[idx-5]+1e-9)
    ma10s = (cl15[idx-1]-cl15[idx-10])/(cl15[idx-10]+1e-9)

    r1   = (cp-cl15[idx-1])/(cl15[idx-1]+1e-9)
    r3   = (cp-cl15[idx-3])/(cl15[idx-3]+1e-9)
    r5   = (cp-cl15[idx-5])/(cl15[idx-5]+1e-9)
    r10  = (cp-cl15[idx-10])/(cl15[idx-10]+1e-9)
    rl1  = (cl15[idx-1]-cl15[idx-2])/(cl15[idx-2]+1e-9)
    rl2  = (cl15[idx-2]-cl15[idx-3])/(cl15[idx-3]+1e-9)
    rac  = r1 * rl1

    def _rsi_loc(cl, p=14):
        d = np.diff(cl[-(p+1):])
        g = d[d>0].mean() if (d>0).any() else 1e-9
        l = (-d[d<0]).mean() if (d<0).any() else 1e-9
        return 100 - 100/(1+g/l)

    rv   = _rsi_loc(cl14); rob = int(rv>70); ros = int(rv<30)
    mv, ms, mh = _macd(cl15[idx-26:idx+1])

    def _atr_loc(ca, p=14):
        tr = [max(c["high"]-c["low"],
                  abs(c["high"]-ca[i-1]["close"]),
                  abs(c["low"]-ca[i-1]["close"]))
              for i,c in enumerate(ca[-p:],1) if i>0]
        return np.mean(tr) if tr else 1e-9

    av   = _atr_loc(w20); apct = av/(cp+1e-9); ca2 = cr/(apct+1e-9)

    def _stoch(ca, p=14):
        sl = ca[-p:]; H = max(c["high"] for c in sl); L = min(c["low"] for c in sl)
        C  = sl[-1]["close"]; return (C-L)/(H-L)*100 if H!=L else 50.

    sk   = _stoch(w14); sob = int(sk>80); sos = int(sk<20)

    def _cci(ca, p=20):
        tp  = [(c["high"]+c["low"]+c["close"])/3 for c in ca[-p:]]
        m   = np.mean(tp); mad = np.mean(np.abs(np.array(tp)-m))
        return (tp[-1]-m)/(0.015*mad) if mad else 0.

    cv   = _cci(w20)
    vv   = _vwap(c15[:idx+1]); rvwap = (cp-vv)/(vv+1e-9)
    bma, bup, blo, bw2, bpb2 = _boll(cl15[idx-20:idx+1])
    bsq  = 0
    vm5  = np.mean([c15[idx-k]["volume"] for k in range(1,6)])
    rv2  = vol/(vm5+1e-9); vsp = int(rv2>2.0)

    def _skew(a):
        m,s = np.mean(a), np.std(a)
        return np.mean(((np.array(a)-m)/(s+1e-9))**3) if len(a)>=3 else 0.

    def _kurt(a):
        m,s = np.mean(a), np.std(a)
        return np.mean(((np.array(a)-m)/(s+1e-9))**4)-3 if len(a)>=4 else 0.

    sk20 = _skew(cl20); ku20 = _kurt(cl20); sk5 = _skew(cl5)
    mom  = r5; macc = r1 - rl1

    def _pat(cur, prev):
        bc = abs(cur["close"]-cur["open"]); bp = abs(prev["close"]-prev["open"])
        rc = cur["high"]-cur["low"]
        ls = min(cur["close"],cur["open"])-cur["low"]
        us = cur["high"]-max(cur["close"],cur["open"])
        bull  = cur["close"]>cur["open"]; pbull = prev["close"]>prev["open"]
        return (int(rc>0 and bc/rc<0.1),
                int(rc>0 and ls>2*bc and us<bc),
                int(rc>0 and us>2*bc and ls<bc),
                int(bull),
                int(bull and not pbull and bc>bp))

    dj, hm, sh, bc2x, en = _pat(cur, c15[idx-1])
    hh  = int(hp > max(c["high"] for c in w5[:-1]))
    ll2 = int(lp < min(c["low"]  for c in w5[:-1]))
    rma_ = rv*mh; va = rv2*apct; pbrsi = bpb2*rv/100

    def ir(n):
        lk = index_data.get(n, {})
        c  = lk.get(ts)
        if c is None:
            ts_plain = str(pd.Timestamp(ts))[:19]
            for k, v_ in lk.items():
                if str(k)[:19] == ts_plain: c = v_; break
        return (c["close"]-c["open"])/(c["open"]+1e-9) if c else 0.

    ni  = ir("NIFTY"); nit = ir("NIFTYIT")
    n50 = ir("NIFTY_NEXT_50"); gf = ir("GIFTNIFTY")
    ia  = np.mean([ni, nit, n50, gf])
    ib  = sum(1 for x in [ni, nit, n50, gf] if x > 0)
    dn  = r1 - ni

    try:
        t = pd.Timestamp(ts); slot = t.hour * 4 + t.minute // 15
    except:
        slot = 0

    enc = int(encoder.transform([stock_name])[0]) if stock_name in encoder.classes_ else 0

    return dict(
        stock_enc=enc, time_slot=slot,
        candle_body=cb, candle_range=cr, upper_shadow=us_, lower_shadow=ls_, open_gap=og,
        r1=r1, r3=r3, r5=r5, r10=r10, r1_lag1=rl1, r1_lag2=rl2, ret_autocorr=rac,
        r_ma5=rma5, r_ma10=rma10, r_ma20=rma20, ma5_slope=ma5s, ma10_slope=ma10s, std20=std20,
        rsi=rv, rsi_ob=rob, rsi_os=ros, macd=mv, macd_sig=ms, macd_hist=mh,
        atr_pct=apct, crng_atr=ca2, stoch_k=sk, stoch_ob=sob, stoch_os=sos, cci=cv,
        r_vwap=rvwap, bb_width=bw2, bb_pct_b=bpb2, bb_squeeze=bsq,
        rel_vol=rv2, vol_spike=vsp, skew20=sk20, kurt20=ku20, skew5=sk5,
        mom=mom, mom_acc=macc, doji=dj, hammer=hm, shoot_star=sh,
        bull_candle=bc2x, engulf=en, higher_high=hh, lower_low=ll2,
        rsi_macd=rma_, vol_atr=va, price_bb_rsi=pbrsi,
        nifty_r=ni, niftyit_r=nit, next50_r=n50, gift_r=gf,
        idx_avg=ia, idx_bull=ib, diverge_nifty=dn,
        sales=sales, profit=profit, opm=opm, profit_margin=pm,
        agg_sent=agg_sent, bull_news=bull_n, bear_news=bear_n,
        nat_cnt=nat_c, int_cnt=int_c, stk_cnt=stk_c,
    )


# ─────────────────────────────────────────────────────────
# DATE SELECTION SCREEN
# ─────────────────────────────────────────────────────────
class DateSelector:
    def __init__(self, root: tk.Tk):
        self.root        = root
        self.chosen_date = None
        self.result      = None
        self.win = tk.Toplevel(root)
        self.win.title("Select Simulation Date")
        self.win.configure(bg=BG)
        self.win.geometry("520x560")
        self.win.grab_set()
        self.win.resizable(False, False)
        self._build()
        self._load_dates()

    def _build(self):
        tk.Label(self.win, text="⚡ PAPER TRADER v5", font=("Courier",14,"bold"),
                 bg=BG, fg=BLUE).pack(pady=(18,2))
        tk.Label(self.win, text="Select a date to simulate",
                 font=("Courier",9), bg=BG, fg=FG2).pack()
        tk.Label(self.win,
                 text="✔ Guaranteed BUY/SELL execution every session",
                 font=("Courier",8), bg=BG, fg=GREEN).pack(pady=(4,0))

        sf = tk.Frame(self.win, bg=BG)
        sf.pack(fill="x", padx=24, pady=(12,4))
        tk.Label(sf, text="Filter:", font=("Courier",9), bg=BG, fg=FG2).pack(side="left")
        self.search_var = tk.StringVar()
        self.search_var.trace("w", lambda *_: self._filter())
        tk.Entry(sf, textvariable=self.search_var, font=("Courier",10),
                 bg=BG3, fg=FG, insertbackground=FG, relief="flat", bd=4
                 ).pack(side="left", fill="x", expand=True, padx=(6,0))

        lf = tk.Frame(self.win, bg=BG3, bd=1, relief="flat")
        lf.pack(fill="both", expand=True, padx=24, pady=4)
        sb = tk.Scrollbar(lf, bg=BG3, troughcolor=BG4)
        sb.pack(side="right", fill="y")
        self.listbox = tk.Listbox(lf, font=("Courier",10), bg=BG3, fg=FG,
                                   selectbackground=BLUE, selectforeground=BG,
                                   activestyle="none", relief="flat",
                                   yscrollcommand=sb.set, bd=0)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb.config(command=self.listbox.yview)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)
        self.listbox.bind("<Double-Button-1>", lambda e: self._confirm())

        self.info_lbl = tk.Label(self.win, text="Loading available dates…",
                                  font=("Courier",8), bg=BG, fg=GREY)
        self.info_lbl.pack(pady=(2,0))

        bf = tk.Frame(self.win, bg=BG)
        bf.pack(pady=12)
        self.btn_go = tk.Button(
            bf, text="▶  START SIMULATION", font=("Courier",10,"bold"),
            bg=BLUE, fg=BG, activebackground=GREEN, activeforeground=BG,
            relief="flat", padx=20, pady=6,
            command=self._confirm, state="disabled")
        self.btn_go.pack(side="left", padx=6)
        tk.Button(bf, text="✕  CANCEL", font=("Courier",10),
                  bg=BG3, fg=RED, activebackground=DKRED, activeforeground=RED,
                  relief="flat", padx=12, pady=6,
                  command=self.win.destroy).pack(side="left", padx=6)

        self.selected_lbl = tk.Label(self.win, text="No date selected",
                                      font=("Courier",9,"bold"), bg=BG, fg=YELLOW)
        self.selected_lbl.pack(pady=(0,6))

    def _load_dates(self):
        date_counts = {}
        for file in glob.glob(f"{COMPANIES_FOLDER}/*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    ds = json.load(f)
                for day in ds.get("data", []):
                    candles = day.get("candles", [])
                    if not candles: continue
                    ts = candles[0].get("timestamp", "")
                    try:    date_str = str(pd.Timestamp(ts).date())
                    except: continue
                    date_counts[date_str] = date_counts.get(date_str, 0) + 1
            except Exception:
                continue
        self.all_dates   = sorted(date_counts.keys(), reverse=True)
        self.date_counts = date_counts
        self.win.after(0, self._populate)

    def _populate(self):
        self._filter()
        n = len(self.all_dates)
        self.info_lbl.config(
            text=f"{n} trading day{'s' if n!=1 else ''} found across your data")

    def _filter(self):
        q = self.search_var.get().strip().lower()
        self.listbox.delete(0, "end")
        self.filtered = [d for d in self.all_dates if q in d]
        for d in self.filtered:
            self.listbox.insert("end", f"  {d}    ({self.date_counts.get(d,0)} stocks)")

    def _on_select(self, event=None):
        sel = self.listbox.curselection()
        if not sel: return
        idx = sel[0]
        if idx < len(self.filtered):
            self.chosen_date = self.filtered[idx]
            self.selected_lbl.config(text=f"Selected: {self.chosen_date}  ✓")
            self.btn_go.config(state="normal")

    def _confirm(self):
        if self.chosen_date:
            self.result = self.chosen_date
            self.win.destroy()

    def wait(self):
        self.win.wait_window()
        return self.result


# ─────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────
class Dashboard:
    def __init__(self, root: tk.Tk, sim_date: str):
        self.root      = root
        self.sim_date  = sim_date
        self.root.title(f"Virtual Paper Trader v5  —  {sim_date}")
        self.root.configure(bg=BG)
        self.root.geometry("1680x920")
        self.root.minsize(1200, 700)

        self.q            = queue.Queue()
        self.equity_pts   = []
        self.total_ticks  = 375
        self.win_count    = 0
        self.loss_count   = 0
        self.positions    = {}
        self.scanner_rows = {}
        self.flash_ids    = {}

        self._build_ui()
        threading.Thread(target=self._run_sim, daemon=True).start()
        self._poll()

    # ══════════════════════════════════════════════════════
    # UI
    # ══════════════════════════════════════════════════════
    def _build_ui(self):
        self._style_ttk()

        # Header
        hdr = tk.Frame(self.root, bg=BG2, pady=7)
        hdr.pack(fill="x")
        tk.Label(hdr, text="⚡ PAPER TRADER  v5", font=("Courier",15,"bold"),
                 bg=BG2, fg=BLUE).pack(side="left", padx=14)
        tk.Label(hdr, text=f"📅 {self.sim_date}", font=("Courier",11),
                 bg=BG2, fg=PURPLE).pack(side="left", padx=6)
        self.lbl_time = tk.Label(hdr, text="──:──", font=("Courier",14,"bold"),
                                  bg=BG2, fg=YELLOW)
        self.lbl_time.pack(side="left", padx=10)
        self.lbl_status = tk.Label(hdr, text="LOADING…", font=("Courier",10),
                                    bg=BG2, fg=GREY)
        self.lbl_status.pack(side="left", padx=6)
        tk.Label(hdr, text="Session", font=("Courier",8),
                 bg=BG2, fg=FG2).pack(side="right", padx=(0,4))
        self.prog_var = tk.DoubleVar(value=0)
        ttk.Progressbar(hdr, variable=self.prog_var, maximum=100,
                        length=180,
                        style="slim.Horizontal.TProgressbar").pack(side="right", padx=(0,14))

        # Ribbon
        ribbon = tk.Frame(self.root, bg=BG3, pady=5)
        ribbon.pack(fill="x")

        def stat_box(label, sv, color=FG):
            f = tk.Frame(ribbon, bg=BG3, padx=12); f.pack(side="left")
            tk.Label(f, text=label, font=("Courier",7), bg=BG3, fg=FG2).pack(anchor="w")
            lbl = tk.Label(f, textvariable=sv, font=("Courier",11,"bold"),
                           bg=BG3, fg=color); lbl.pack(anchor="w")
            tk.Frame(ribbon, bg=ACCENT, width=1, height=30).pack(side="left")
            return lbl

        self.sv_equity   = tk.StringVar(value=f"₹{STARTING_CAPITAL:,.0f}")
        self.sv_avail    = tk.StringVar(value=f"₹{STARTING_CAPITAL:,.0f}")
        self.sv_used     = tk.StringVar(value="₹0")
        self.sv_open_val = tk.StringVar(value="₹0")
        self.sv_pnl      = tk.StringVar(value="₹0.00  (0.00%)")
        self.sv_realised = tk.StringVar(value="₹0.00")
        self.sv_wins     = tk.StringVar(value="0  (0%)")
        self.sv_losses   = tk.StringVar(value="0  (0%)")
        self.sv_slots    = tk.StringVar(value=f"0/{MAX_POSITIONS}")
        self.sv_mode     = tk.StringVar(value="—")

        self.lbl_eq   = stat_box("EQUITY",       self.sv_equity,   WHITE)
        stat_box("AVAIL MARGIN",  self.sv_avail,    BLUE)
        stat_box("USED MARGIN",   self.sv_used,     ORANGE)
        stat_box("OPEN VALUE",    self.sv_open_val, FG)
        self.lbl_pnl  = stat_box("NET P&L",      self.sv_pnl,      FG)
        stat_box("REALISED P&L",  self.sv_realised, PURPLE)
        self.lbl_wins = stat_box("WINS",          self.sv_wins,     GREEN)
        stat_box("LOSSES",        self.sv_losses,   RED)
        stat_box("SLOTS",         self.sv_slots,    BLUE)
        stat_box("SIGNAL MODE",   self.sv_mode,     YELLOW)

        # Body
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=3, pady=3)

        left = tk.Frame(body, bg=BG2)
        left.pack(side="left", fill="both", expand=True, padx=(0,3))
        self._section_label(left, "📈  EQUITY CURVE")
        self.eq_canvas = tk.Canvas(left, bg=BG, highlightthickness=0)
        self.eq_canvas.pack(fill="both", expand=True, padx=4, pady=(0,4))
        self.eq_canvas.bind("<Configure>", lambda e: self._draw_equity())

        mid = tk.Frame(body, bg=BG, width=480)
        mid.pack(side="left", fill="both", padx=(0,3))
        mid.pack_propagate(False)

        pos_outer = tk.Frame(mid, bg=BG2)
        pos_outer.pack(fill="x", pady=(0,3))
        self._section_label(pos_outer, "📂  OPEN POSITIONS")
        self.pos_frame = tk.Frame(pos_outer, bg=BG2)
        self.pos_frame.pack(fill="x", padx=4, pady=(0,5))

        ord_outer = tk.Frame(mid, bg=BG2)
        ord_outer.pack(fill="both", expand=True)
        self._section_label(ord_outer, "🧾  ORDER BOOK  (BUY / SELL)")
        self._build_order_log(ord_outer)

        right = tk.Frame(body, bg=BG2, width=290)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self._section_label(right, "📡  SIGNAL SCANNER")
        tk.Label(right,
                 text=f"conf≥{MIN_CONFIDENCE:.2f}  TP×{REWARD_MULT}  SL{STOP_LOSS_PCT*100:.0f}%  margin{MARGIN_RATIO*100:.0f}%",
                 font=("Courier",7), bg=BG2, fg=GREY).pack(anchor="w", padx=8, pady=(0,4))
        self.scanner_outer = tk.Frame(right, bg=BG2)
        self.scanner_outer.pack(fill="both", expand=True, padx=4)

    def _section_label(self, parent, text):
        tk.Frame(parent, bg=ACCENT, height=1).pack(fill="x")
        tk.Label(parent, text=text, font=("Courier",9,"bold"),
                 bg=parent.cget("bg"), fg=BLUE, pady=4).pack(anchor="w", padx=8)

    def _style_ttk(self):
        s = ttk.Style(); s.theme_use("clam")
        s.configure("slim.Horizontal.TProgressbar",
                     troughcolor=BG3, background=BLUE, thickness=5)
        s.configure("Treeview", background=BG3, foreground=FG,
                     fieldbackground=BG3, font=("Courier",8), rowheight=19)
        s.configure("Treeview.Heading", background=BG4, foreground=BLUE,
                     font=("Courier",8,"bold"), relief="flat")
        s.map("Treeview",
              background=[("selected","#2a4a6b")],
              foreground=[("selected", WHITE)])

    def _build_order_log(self, parent):
        cols = ("ID","Time","Stock","Side","Qty","Price","Value","P&L","Status","Mode")
        self.order_tree = ttk.Treeview(parent, columns=cols, show="headings",
                                        height=16, selectmode="none")
        widths = {"ID":28,"Time":40,"Stock":76,"Side":38,"Qty":52,
                  "Price":62,"Value":70,"P&L":70,"Status":74,"Mode":50}
        for c in cols:
            self.order_tree.heading(c, text=c)
            self.order_tree.column(c, width=widths[c], anchor="center", stretch=False)
        self.order_tree.tag_configure("buy",      background="#0d2040", foreground=BLUE)
        self.order_tree.tag_configure("sell_win", background=DKGREEN,   foreground=GREEN)
        self.order_tree.tag_configure("sell_loss",background=DKRED,     foreground=RED)
        self.order_tree.tag_configure("sell_flat",background=BG4,       foreground=YELLOW)
        sb = ttk.Scrollbar(parent, orient="vertical", command=self.order_tree.yview)
        self.order_tree.configure(yscrollcommand=sb.set)
        self.order_tree.pack(side="left", fill="both", expand=True, padx=(4,0), pady=(0,4))
        sb.pack(side="right", fill="y", pady=(0,4), padx=(0,4))

    # ── Equity Curve ─────────────────────────────────────
    def _draw_equity(self):
        c = self.eq_canvas; c.delete("all")
        W = c.winfo_width(); H = c.winfo_height()
        if W < 20 or H < 20: return
        PL, PR, PT, PB = 56, 14, 20, 32
        if len(self.equity_pts) < 2:
            c.create_text(W//2, H//2, text="Awaiting data…",
                          font=("Courier",11), fill=GREY); return
        ticks    = [t for t,_ in self.equity_pts]
        equities = [e for _,e in self.equity_pts]
        mn = min(equities+[STARTING_CAPITAL])-200
        mx = max(equities+[STARTING_CAPITAL])+200
        rng = mx-mn or 1
        t_mn = ticks[0]; t_mx = max(ticks[-1], self.total_ticks-1)
        t_rng = t_mx-t_mn or 1
        sx = lambda t: PL+(t-t_mn)/t_rng*(W-PL-PR)
        sy = lambda v: PT+(1-(v-mn)/rng)*(H-PT-PB)
        for i in range(6):
            val = mn+i*rng/5; y = sy(val)
            c.create_line(PL,y,W-PR,y,fill=BG4,dash=(3,5))
            c.create_text(PL-4,y,text=f"₹{val/1000:.1f}k",
                          font=("Courier",7),fill=GREY,anchor="e")
        for h in range(9,16):
            frac = (h*60-9*60)/(6*60+15)
            x = PL+frac*(W-PL-PR)
            if PL<=x<=W-PR:
                c.create_line(x,H-PB,x,H-PB+4,fill=GREY)
                c.create_text(x,H-PB+10,text=f"{h:02d}:00",
                              font=("Courier",7),fill=GREY)
        y0 = sy(STARTING_CAPITAL)
        c.create_line(PL,y0,W-PR,y0,fill=GREY,dash=(6,4),width=1)
        c.create_text(W-PR+2,y0,text="Start",font=("Courier",7),fill=GREY,anchor="w")
        pts = [(sx(t),sy(e)) for t,e in self.equity_pts]
        y0c = sy(STARTING_CAPITAL)
        cur_above=[]; cur_below=[]; poly_above=[]; poly_below=[]
        for px_,py_ in pts:
            if py_<=y0c:
                if cur_below:
                    poly_below.append(cur_below+[(cur_below[-1][0],y0c),(cur_below[0][0],y0c)])
                    cur_below=[]
                cur_above.append((px_,py_))
            else:
                if cur_above:
                    poly_above.append(cur_above+[(cur_above[-1][0],y0c),(cur_above[0][0],y0c)])
                    cur_above=[]
                cur_below.append((px_,py_))
        if cur_above: poly_above.append(cur_above+[(cur_above[-1][0],y0c),(cur_above[0][0],y0c)])
        if cur_below: poly_below.append(cur_below+[(cur_below[-1][0],y0c),(cur_below[0][0],y0c)])
        for segs in poly_above:
            flat=[co for p_ in segs for co in p_]
            if len(flat)>=6: c.create_polygon(flat,fill="#1a3a2a",outline="")
        for segs in poly_below:
            flat=[co for p_ in segs for co in p_]
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

    # ── Positions ─────────────────────────────────────────
    def _refresh_positions(self):
        for w in self.pos_frame.winfo_children(): w.destroy()
        if not self.positions:
            tk.Label(self.pos_frame, text="   — no open positions —",
                     font=("Courier",9), bg=BG2, fg=GREY, pady=6).pack(anchor="w"); return
        for name, pos in self.positions.items():
            cur_px  = pos.get("cur_px", pos["entry"])
            pnl_abs = (cur_px - pos["entry"]) * pos["shares"]
            pnl_pct = pnl_abs / pos["invested"] * 100 if pos["invested"] else 0
            col     = GREEN if pnl_abs >= 0 else RED
            bg_card = "#182030" if pnl_abs >= 0 else "#2a1520"
            card = tk.Frame(self.pos_frame, bg=bg_card, pady=5, padx=8)
            card.pack(fill="x", pady=(0,3))
            r1f = tk.Frame(card, bg=bg_card); r1f.pack(fill="x")
            ck = "★" if pos.get("high_conviction") else ""
            mode_tag = f"[{pos.get('mode','?')}]"
            tk.Label(r1f, text=f"{ck}{name} {mode_tag}", font=("Courier",10,"bold"),
                     bg=bg_card, fg=YELLOW).pack(side="left")
            tk.Label(r1f, text=f"₹{cur_px:.2f}", font=("Courier",10,"bold"),
                     bg=bg_card, fg=col).pack(side="right")
            r2f = tk.Frame(card, bg=bg_card); r2f.pack(fill="x")
            tk.Label(r2f,
                     text=f"BUY ₹{pos['entry']:.2f}  ×{pos['shares']:.3f}sh  conf={pos.get('proba',0):.2f}",
                     font=("Courier",8), bg=bg_card, fg=FG2).pack(side="left")
            tk.Label(r2f, text=f"Unrealised: {pnl_pct:+.2f}%",
                     font=("Courier",8,"bold"), bg=bg_card, fg=col).pack(side="right")
            r3f = tk.Frame(card, bg=bg_card); r3f.pack(fill="x")
            tk.Label(r3f, text=f"SL ₹{pos['stop']:.2f}",
                     font=("Courier",8), bg=bg_card, fg=RED).pack(side="left")
            tk.Label(r3f, text=f"TP ₹{pos['target']:.2f}",
                     font=("Courier",8), bg=bg_card, fg=GREEN).pack(side="left", padx=10)
            tk.Label(r3f, text=f"P&L ₹{pnl_abs:+.1f}",
                     font=("Courier",8,"bold"), bg=bg_card, fg=col).pack(side="right")
            tk.Label(card,
                     text=f"Margin: ₹{pos['invested']*MARGIN_RATIO:,.0f}  |  Invested: ₹{pos['invested']:,.0f}",
                     font=("Courier",7), bg=bg_card, fg=GREY).pack(anchor="w")
            if pos.get("trail"):
                tk.Label(card, text="⟳ trailing stop active",
                         font=("Courier",7), bg=bg_card, fg=ORANGE).pack(anchor="w")
            if pos.get("waiting_recovery"):
                tk.Label(card, text="⏳ waiting for recovery…",
                         font=("Courier",7), bg=bg_card, fg=PURPLE).pack(anchor="w")
            pbar = tk.Canvas(card, height=5, bg=ACCENT, highlightthickness=0)
            pbar.pack(fill="x", pady=(3,0))
            pbar.update_idletasks()
            bw2 = pbar.winfo_width() or 420
            sl = pos["stop"]; tp = pos["target"]; rng2 = tp-sl
            if rng2 > 0:
                frac = max(0., min(1., (cur_px-sl)/rng2))
                pbar.create_rectangle(0,0,int(bw2*frac),5,fill=col,outline="")

    # ── Scanner ───────────────────────────────────────────
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
        self.scanner_rows[name] = dict(
            frame=f, bar_bg=bar_bg, bar_fill=bar_fill,
            bar_lbl=bar_lbl, lbl_p=lbl_p, lbl_r=lbl_r)

    def _update_scanner(self, name, proba, pred_ret, fired, skipped, forced=False):
        self._ensure_scanner_row(name)
        row = self.scanner_rows[name]
        bg_col = BG3; bar_col = GREY
        if fired and forced:
            bg_col = "#2a2a00"; bar_col = YELLOW
        elif fired:
            bg_col = DKGREEN;   bar_col = GREEN
        elif skipped:
            bg_col = "#2a1a2a"; bar_col = GREY
        elif proba >= MIN_CONFIDENCE * 0.7:
            bar_col = BLUE
        bw2 = row["bar_bg"].winfo_width() or 88
        fw  = max(1, int(bw2 * min(proba, 1.0)))
        row["bar_fill"].place(x=0, y=0, width=fw, height=12)
        row["bar_fill"].config(bg=bar_col)
        row["bar_lbl"].config(text=f"{proba:.2f}", bg=bar_col)
        row["bar_bg"].config(bg=ACCENT)
        pc = GREEN if pred_ret>0 else (RED if pred_ret<0 else GREY)
        row["lbl_p"].config(text=f"p={proba:.3f}", fg=bar_col, bg=bg_col)
        row["lbl_r"].config(text=f"{pred_ret*100:+.3f}%", fg=pc, bg=bg_col)
        for w in (row["frame"], row["lbl_p"], row["lbl_r"]):
            w.config(bg=bg_col)
        if fired:
            if name in self.flash_ids: self.root.after_cancel(self.flash_ids[name])
            self.flash_ids[name] = self.root.after(
                1400, lambda n=name: self._reset_scanner_row(n))

    def _reset_scanner_row(self, name):
        if name not in self.scanner_rows: return
        row = self.scanner_rows[name]
        for w in (row["frame"], row["lbl_p"], row["lbl_r"]):
            try: w.config(bg=BG3)
            except: pass

    # ── Poll ──────────────────────────────────────────────
    def _poll(self):
        processed = 0
        try:
            while processed < 40:
                ev = self.q.get_nowait(); self._handle(ev); processed += 1
        except queue.Empty:
            pass
        self.root.after(40, self._poll)

    def _handle(self, ev):
        kind = ev[0]

        if kind == "status":
            self.lbl_status.config(text=ev[1])

        elif kind == "mode":
            self.sv_mode.config(value=ev[1]) if hasattr(self.sv_mode, 'config') else \
                self.sv_mode.set(ev[1])

        elif kind == "init":
            _, total, names = ev
            self.total_ticks = total
            for n in names: self._ensure_scanner_row(n)

        elif kind == "tick":
            _, tick_i, ts, avail, used, equity, pnl, realised, open_val, pos_snap = ev
            self.equity_pts.append((tick_i, equity))
            self.positions = pos_snap
            self.lbl_time.config(text=ts)
            pct  = pnl/STARTING_CAPITAL*100
            col  = GREEN if pnl>=0 else RED
            sign = "+" if pnl>=0 else ""
            self.sv_equity.set(f"₹{equity:,.0f}")
            self.sv_avail.set(f"₹{avail:,.0f}")
            self.sv_used.set(f"₹{used:,.0f}")
            self.sv_open_val.set(f"₹{open_val:,.0f}")
            self.sv_pnl.set(f"{sign}₹{pnl:,.0f}  ({pct:+.2f}%)")
            self.sv_realised.set(f"{'+'if realised>=0 else ''}₹{realised:,.0f}")
            self.lbl_pnl.config(fg=col); self.lbl_eq.config(fg=col)
            tot = self.win_count + self.loss_count
            wp  = self.win_count/tot*100 if tot else 0
            self.sv_wins.set(f"{self.win_count}  ({wp:.0f}%)")
            self.sv_losses.set(f"{self.loss_count}  ({100-wp if tot else 0:.0f}%)")
            self.sv_slots.set(f"{len(pos_snap)}/{MAX_POSITIONS}")
            self.prog_var.set(tick_i/max(self.total_ticks-1,1)*100)
            self._refresh_positions()
            self._draw_equity()

        elif kind == "order":
            _, o = ev
            side = o["side"]
            pnl_str = ""
            if o["pnl"] is not None:
                sign = "+" if o["pnl"]>=0 else ""
                pnl_str = f"{sign}₹{o['pnl']:.1f}"
                if   o["pnl"]>0: self.win_count  += 1
                elif o["pnl"]<0: self.loss_count += 1
            if side == "BUY":
                tag = "buy"
            else:
                tag = ("sell_win"  if (o["pnl"] or 0)>0
                       else "sell_loss" if (o["pnl"] or 0)<0
                       else "sell_flat")
            self.order_tree.insert("", 0,
                values=(f"#{o['id']}", o["time"], o["stock"][:9], side,
                        f"{o['qty']:.3f}", f"₹{o['price']:.2f}",
                        f"₹{o['value']:,.0f}", pnl_str or "—",
                        o["status"][:12], o.get("mode","?")[:5]),
                tags=(tag,))

        elif kind == "signal":
            _, name, proba, pred_ret, fired, skipped, forced = ev
            self._update_scanner(name, proba, pred_ret, fired, skipped, forced)

        elif kind == "done":
            _, final_eq, realised = ev
            pnl  = final_eq - STARTING_CAPITAL
            col  = GREEN if pnl>=0 else RED
            sign = "+" if pnl>=0 else ""
            self.lbl_status.config(
                text=f"✓ COMPLETE ({self.sim_date}) — "
                     f"Final P&L: {sign}₹{pnl:,.2f}  ({pnl/STARTING_CAPITAL*100:+.2f}%)",
                fg=col)
            self.prog_var.set(100)

    # ══════════════════════════════════════════════════════
    # SIMULATION THREAD
    # ══════════════════════════════════════════════════════
    def _run_sim(self):
        # ── Load model (optional) ─────────────────────────
        self.q.put(("status", "Loading model bundle…"))
        model_available = False
        xgb_clf = lgbm_clf = rf_clf = meta_clf = None
        xgb_reg = lgbm_reg = rf_reg = meta_reg = None
        encoder = feat_cols = None

        if JOBLIB_OK:
            try:
                bundle   = joblib.load(MODEL_BUNDLE)
                xgb_clf  = bundle["xgb_clf"];  lgbm_clf = bundle["lgbm_clf"]
                rf_clf   = bundle["rf_clf"];   meta_clf  = bundle["meta_clf"]
                xgb_reg  = bundle["xgb_reg"];  lgbm_reg = bundle["lgbm_reg"]
                rf_reg   = bundle["rf_reg"];   meta_reg  = bundle["meta_reg"]
                encoder  = bundle["encoder"]
                feat_cols= bundle["feat_cols"]
                model_available = True
                self.q.put(("status", "Model loaded ✓"))
                self.q.put(("mode",   "ML MODEL"))
            except Exception as e:
                self.q.put(("status", f"Model not found ({e}) — using DEMO signals"))
                self.q.put(("mode",   "DEMO RSI"))
        else:
            self.q.put(("status", "joblib not installed — using DEMO signals"))
            self.q.put(("mode",   "DEMO RSI"))

        def model_predict(X_df):
            import pandas as pd
            # Guard against None models - type narrowing
            assert xgb_clf is not None, "xgb_clf not loaded"
            assert lgbm_clf is not None, "lgbm_clf not loaded"
            assert rf_clf is not None, "rf_clf not loaded"
            assert meta_clf is not None, "meta_clf not loaded"
            assert xgb_reg is not None, "xgb_reg not loaded"
            assert lgbm_reg is not None, "lgbm_reg not loaded"
            assert rf_reg is not None, "rf_reg not loaded"
            assert meta_reg is not None, "meta_reg not loaded"
            
            bc    = np.column_stack([xgb_clf.predict_proba(X_df)[:,1],
                                     lgbm_clf.predict_proba(X_df)[:,1],
                                     rf_clf.predict_proba(X_df)[:,1]])
            proba = meta_clf.predict_proba(bc)[:,1]
            br    = np.column_stack([xgb_reg.predict(X_df),
                                     lgbm_reg.predict(X_df),
                                     rf_reg.predict(X_df)])
            ret   = meta_reg.predict(br)
            return float(proba[0]), float(ret[0])

        def kelly(p, win_r):
            if win_r <= 0: win_r = STOP_LOSS_PCT * 0.5
            b = win_r / STOP_LOSS_PCT
            f = (p*b - (1-p)) / b
            f = max(0., min(f, POSITION_CAP))
            scale = min(1.0, (p - MIN_CONFIDENCE) / max(1.0-MIN_CONFIDENCE, 0.01) * 2 + 0.4)
            return f * scale

        # ── Load index data ───────────────────────────────
        self.q.put(("status", "Loading index data…"))
        index_data = {}
        for file in glob.glob(f"{INDICES_FOLDER}/*.json"):
            iname = file.replace("\\","/").split("/")[-1].replace("_historical_data.json","")
            try:
                with open(file,"r",encoding="utf-8") as f:
                    ds = json.load(f)
                lk = {}
                for day in ds["data"]:
                    for c_ in to_15min(day["candles"]):
                        lk[str(pd.Timestamp(c_["timestamp"]))[:19]] = c_
                        lk[c_["timestamp"]] = c_
                index_data[iname] = lk
            except Exception:
                continue

        # ── Load stocks for chosen date ───────────────────
        self.q.put(("status", f"Loading stocks for {self.sim_date}…"))
        stocks = {}
        for file in glob.glob(f"{COMPANIES_FOLDER}/*.json"):
            name = file.replace("\\","/").split("/")[-1].replace("_with_quarterly_results.json","")
            try:
                with open(file,"r",encoding="utf-8") as f:
                    ds = json.load(f)
            except Exception:
                continue
            all_days = ds.get("data", [])
            if len(all_days) < 2: continue
            chosen = None
            for di, day in enumerate(all_days):
                candles = day.get("candles", [])
                if not candles: continue
                try:
                    d_str = str(pd.Timestamp(candles[0].get("timestamp","")).date())
                except Exception:
                    continue
                if d_str == self.sim_date: chosen = di; break
            if chosen is None or chosen == 0: continue
            if len(all_days[chosen]["candles"]) < 10: continue
            live_day  = all_days[chosen]
            ctx_1min  = []
            for di in range(chosen-1, max(chosen-4,-1), -1):
                ctx_1min = all_days[di]["candles"] + ctx_1min
                if len(to_15min(ctx_1min)) >= 30: break
            stocks[name] = {
                "live_1min":  live_day["candles"],
                "context_15": to_15min(ctx_1min),
                "quarterly":  live_day.get("latest_quarterly_result", {}),
                "news":       live_day.get("news_analysis", {}),
            }

        if not stocks:
            self.q.put(("status", f"ERROR: No stocks found for {self.sim_date}.")); return

        all_1min_ts = sorted(set(
            c["timestamp"] for s in stocks.values() for c in s["live_1min"]
        ))
        self.q.put(("init", len(all_1min_ts), list(stocks.keys())))
        self.q.put(("status",
                    f"SIMULATING {self.sim_date} — {len(stocks)} stocks — "
                    f"{'ML MODEL' if model_available else 'DEMO MODE'}"))

        # ── Simulation state ──────────────────────────────
        ob             = OrderBook(STARTING_CAPITAL, MARGIN_RATIO)
        positions      = {}
        seen_1min      = {n: [] for n in stocks}
        last_bar_fired = {n: None for n in stocks}
        bearish_bars   = {}

        for tick_i, ts in enumerate(all_1min_ts):
            ts_dt   = pd.Timestamp(ts)
            ts_str  = ts_dt.strftime("%H:%M")
            is_last = (ts == all_1min_ts[-1])
            slot    = ts_dt.hour * 4 + ts_dt.minute // 15

            # Ingest candle
            for name, s in stocks.items():
                for c_ in s["live_1min"]:
                    if c_["timestamp"] == ts:
                        seen_1min[name].append(c_); break

            # ── Monitor open positions ────────────────────
            for name, pos in list(positions.items()):
                cur1 = next((c_ for c_ in reversed(seen_1min[name])
                             if c_["timestamp"] == ts), None)
                if cur1 is None and seen_1min[name]:
                    cur1 = seen_1min[name][-1]
                if cur1 is None: continue

                hi = cur1["high"]; lo = cur1["low"]; px = cur1["close"]
                pos["cur_px"] = px

                # Trailing stop update
                if hi > pos["peak"]:
                    pos["peak"] = hi
                    if pos["peak"] >= pos["entry"] * (1 + TRAILING_TRIGGER):
                        new_ts = pos["peak"] * (1 - TRAILING_STOP_PCT)
                        if new_ts > pos["stop"]:
                            pos["stop"] = new_ts; pos["trail"] = True

                reason  = None
                sell_px = px

                if hi >= pos["target"]:                              # TARGET HIT
                    sell_px = pos["target"]; reason = "TARGET HIT"
                elif lo <= pos["stop"]:                              # STOP LOSS
                    sell_px = pos["stop"];   reason = "STOP LOSS"
                elif px < pos["entry"] * (1 + NEUTRAL_EXIT_THRESHOLD):
                    bearish_bars[name] = bearish_bars.get(name,0) + 1
                    pos["waiting_recovery"] = True
                    if bearish_bars[name] >= BEARISH_CONFIRM_BARS:
                        sell_px = px; reason = "NEUTRAL EXIT"
                else:
                    bearish_bars[name] = 0; pos["waiting_recovery"] = False

                if is_last and reason is None:
                    reason = "EOD CLOSE"; sell_px = px

                # ── EXECUTE SELL ──────────────────────────
                if reason:
                    sell_ord = ob.place_sell(ts_str, name, pos["shares"],
                                             sell_px, pos["buy_order"], reason)
                    self.q.put(("order", sell_ord))
                    del positions[name]
                    bearish_bars.pop(name, None)

            # ── Signal scan at 15-min bar close ──────────
            is_bar_close = (ts_dt.minute % 15 == 14)
            free_slots   = MAX_POSITIONS - len(positions)

            if (is_bar_close or is_last) and free_slots > 0:
                all_candidates = []   # includes skipped ones for fallback
                fired_candidates = []

                for name, s in stocks.items():
                    if name in positions or not seen_1min[name]: continue
                    live_15 = to_15min(seen_1min[name])
                    if not live_15: continue
                    full_15 = s["context_15"] + live_15
                    cl15    = [c_["close"] for c_ in full_15]
                    idx15   = len(full_15) - 1
                    if idx15 < 26: continue
                    bar_ts = full_15[idx15]["timestamp"]
                    if last_bar_fired[name] == bar_ts: continue
                    last_bar_fired[name] = bar_ts

                    # ── Get signal ────────────────────────
                    proba    = 0.0
                    pred_ret = 0.0
                    mode     = "DEMO"

                    if model_available:
                        try:
                            feats = build_features(
                                name, full_15, cl15, idx15,
                                s["quarterly"], s["news"],
                                index_data, encoder, feat_cols)
                            import pandas as _pd2
                            X = _pd2.DataFrame([feats])[feat_cols]
                            proba, pred_ret = model_predict(X)
                            mode = "MODEL"
                        except Exception:
                            proba, pred_ret = demo_signal(full_15, idx15)
                            mode = "DEMO"
                    else:
                        proba, pred_ret = demo_signal(full_15, idx15)
                        mode = "DEMO"

                    passes_threshold = (proba >= MIN_CONFIDENCE and
                                        pred_ret >= MIN_PRED_RETURN)
                    score = proba * pred_ret + proba * 0.1  # slight proba bias

                    cand = dict(name=name, proba=proba, pred_ret=pred_ret,
                                price=full_15[idx15]["close"], score=score,
                                high_conviction=(proba >= 0.72), mode=mode,
                                forced=False)

                    all_candidates.append(cand)
                    fired = passes_threshold
                    self.q.put(("signal", name, proba, pred_ret,
                                fired, not fired, False))

                    if fired:
                        fired_candidates.append(cand)

                # ── FORCE TRADE if no natural fires ───────
                if (not fired_candidates
                        and FORCE_TRADE_FALLBACK
                        and all_candidates
                        and free_slots > 0
                        and slot in FORCE_TRADE_BARS):
                    all_candidates.sort(key=lambda x: x["score"], reverse=True)
                    top = all_candidates[0]
                    top["forced"] = True
                    # Boost proba slightly so it feels like a weak signal
                    top["proba"]    = max(top["proba"], MIN_CONFIDENCE)
                    top["pred_ret"] = max(top["pred_ret"], MIN_PRED_RETURN)
                    fired_candidates = [top]
                    self.q.put(("signal", top["name"], top["proba"],
                                top["pred_ret"], True, False, True))

                # ── Execute BUYs for fired candidates ─────
                fired_candidates.sort(key=lambda x: x["score"], reverse=True)

                for cand in fired_candidates[:free_slots]:
                    name      = cand["name"]
                    entry_px  = cand["price"]
                    proba     = cand["proba"]
                    pred_ret  = cand["pred_ret"]
                    mode      = cand["mode"]

                    if name in positions: continue  # may have been bought in same bar

                    frac   = kelly(proba, pred_ret)
                    invest = ob.available_margin * frac

                    if cand["high_conviction"]:
                        invest = min(invest * 1.35, ob.available_margin * POSITION_CAP)

                    invest = max(invest, MIN_INVEST)
                    invest = min(invest, ob.available_margin * 0.88)

                    # Hard margin gate
                    if not ob.can_buy(invest): continue

                    shares = invest / entry_px
                    stop   = entry_px * (1 - STOP_LOSS_PCT)
                    target = entry_px * (1 + pred_ret * REWARD_MULT)

                    # ── EXECUTE BUY ───────────────────────
                    buy_ord = ob.place_buy(
                        ts_str, name, shares, entry_px,
                        invest, stop, target, proba, mode)
                    self.q.put(("order", buy_ord))

                    positions[name] = dict(
                        shares=shares, entry=entry_px, stop=stop, target=target,
                        peak=entry_px, trail=False, invested=invest,
                        proba=proba, pred_ret=pred_ret, entry_ts=ts,
                        cur_px=entry_px, high_conviction=cand["high_conviction"],
                        waiting_recovery=False, buy_order=buy_ord, mode=mode,
                    )

            # ── Snapshot for UI ───────────────────────────
            open_val = sum(p["shares"]*p["cur_px"] for p in positions.values())
            equity   = ob.equity(positions)
            pnl      = ob.net_pnl(positions)
            pos_snap = {n: {k:v for k,v in p.items() if k!="buy_order"}
                        for n, p in positions.items()}

            self.q.put(("tick", tick_i, ts_str,
                        ob.available_margin, ob.used_margin,
                        equity, pnl, ob.realised_pnl, open_val, pos_snap))
            time.sleep(SIMULATE_DELAY)

        self.q.put(("done", ob.equity({}), ob.realised_pnl))


# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    try: root.iconbitmap("")
    except Exception: pass
    selector = DateSelector(root)
    chosen   = selector.wait()
    if not chosen:
        root.destroy()
    else:
        root.deiconify()
        Dashboard(root, sim_date=chosen)
        root.mainloop()