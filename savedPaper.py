#!/usr/bin/env python3
# =========================================================
# VIRTUAL PAPER TRADER v2  —  Live Tkinter Dashboard
# =========================================================
# Runs the full simulation in a background thread and
# streams every event to the UI in real time:
#   • Equity curve (canvas, coloured by P&L)
#   • Open-positions cards with live unrealised P&L
#   • Signal-scanner bar for every 15-min bar close
#   • Scrollable trade log (green = win, red = loss)
#   • Top stats ribbon updated every tick
# =========================================================

import tkinter as tk
from tkinter import ttk
import threading, queue, json, glob, time, joblib, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# CONFIG  (edit these to tune the strategy)
# ─────────────────────────────────────────────────────────
MODEL_BUNDLE      = "model_v3_classifier.pkl"
INDICES_FOLDER    = "Indexes"
COMPANIES_FOLDER  = "Companies"

STARTING_CAPITAL  = 2000_000.0
MAX_POSITIONS     = 100
POSITION_CAP      = 0.25
MIN_CONFIDENCE    = 0.10      # UI entry threshold (not the stored model threshold)
MIN_PRED_RETURN   = 0.0
STOP_LOSS_PCT     = 0.005     # 0.5 %
TRAILING_TRIGGER  = 0.003
TRAILING_STOP_PCT = 0.002
REWARD_MULT       = 1.5
SIMULATE_DELAY    = 0.015     # seconds per 1-min tick  (lower = faster replay)
MIN_INVEST        = 1000

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
# TRADING LOGIC  (identical to paperTrade_fixed.py)
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

    return dict(
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


# ─────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────

class Dashboard:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Virtual Paper Trader v2  —  Live Dashboard")
        self.root.configure(bg=BG)
        self.root.geometry("1520x860")
        self.root.minsize(1100, 700)

        # ── state ──────────────────────────────────────────
        self.q            = queue.Queue()
        self.equity_pts   = []          # list of (tick_i, equity_value)
        self.total_ticks  = 375
        self.win_count    = 0
        self.loss_count   = 0
        self.positions    = {}          # name -> snapshot dict
        self.scanner_rows = {}          # name -> widget refs
        self.flash_ids    = {}          # name -> after-id for flash reset
        self.pnl_history  = []          # recent P&Ls for sparkline

        self._build_ui()
        threading.Thread(target=self._run_sim, daemon=True).start()
        self._poll()

    # ══════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ══════════════════════════════════════════════════════

    def _build_ui(self):
        self._style_ttk()

        # ── header bar ─────────────────────────────────────
        hdr = tk.Frame(self.root, bg=BG2, pady=7)
        hdr.pack(fill="x")

        tk.Label(hdr, text="⚡ PAPER TRADER  v2", font=("Courier",15,"bold"),
                 bg=BG2, fg=BLUE).pack(side="left", padx=14)

        self.lbl_time = tk.Label(hdr, text="──:──", font=("Courier",14,"bold"),
                                  bg=BG2, fg=YELLOW)
        self.lbl_time.pack(side="left", padx=10)

        self.lbl_status = tk.Label(hdr, text="LOADING…", font=("Courier",10),
                                    bg=BG2, fg=GREY)
        self.lbl_status.pack(side="left", padx=6)

        # session progress bar (right side)
        tk.Label(hdr, text="Session", font=("Courier",8), bg=BG2, fg=FG2
                 ).pack(side="right", padx=(0,4))
        self.prog_var = tk.DoubleVar(value=0)
        self.prog = ttk.Progressbar(hdr, variable=self.prog_var, maximum=100,
                                     length=180, style="slim.Horizontal.TProgressbar")
        self.prog.pack(side="right", padx=(0,14))

        # ── stats ribbon ───────────────────────────────────
        ribbon = tk.Frame(self.root, bg=BG3, pady=5)
        ribbon.pack(fill="x")

        def stat_box(label, sv, color=FG):
            f = tk.Frame(ribbon, bg=BG3, padx=14)
            f.pack(side="left")
            tk.Label(f, text=label, font=("Courier",7), bg=BG3, fg=FG2).pack(anchor="w")
            lbl = tk.Label(f, textvariable=sv, font=("Courier",12,"bold"),
                           bg=BG3, fg=color)
            lbl.pack(anchor="w")
            # separator
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

        self.lbl_equity  = stat_box("EQUITY",    self.sv_equity,   WHITE)
        stat_box("CASH",             self.sv_cash,     FG)
        stat_box("IN POSITIONS",     self.sv_open_val, FG)
        self.lbl_pnl_box = stat_box("NET P&L",   self.sv_pnl,      FG)
        stat_box("TRADES",           self.sv_trades,   FG)
        self.lbl_wins    = stat_box("WINS",       self.sv_wins,     GREEN)
        self.lbl_losses  = stat_box("LOSSES",     self.sv_losses,   RED)
        stat_box("OPEN SLOTS",       self.sv_slots,    BLUE)

        # ── body ───────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=3, pady=3)

        # LEFT column: equity curve
        left = tk.Frame(body, bg=BG2, bd=0)
        left.pack(side="left", fill="both", expand=True, padx=(0,3))

        self._section_label(left, "📈  EQUITY CURVE")
        self.eq_canvas = tk.Canvas(left, bg=BG, highlightthickness=0)
        self.eq_canvas.pack(fill="both", expand=True, padx=4, pady=(0,4))
        self.eq_canvas.bind("<Configure>", lambda e: self._draw_equity())

        # MIDDLE column: positions + trade log
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

        # RIGHT column: signal scanner
        right = tk.Frame(body, bg=BG2, width=290)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self._section_label(right, "📡  SIGNAL SCANNER")
        tk.Label(right, text=f"threshold ≥ {MIN_CONFIDENCE:.2f}   reward ×{REWARD_MULT}",
                 font=("Courier",7), bg=BG2, fg=GREY).pack(anchor="w", padx=8, pady=(0,4))
        self.scanner_outer = tk.Frame(right, bg=BG2)
        self.scanner_outer.pack(fill="both", expand=True, padx=4)

    def _section_label(self, parent, text):
        f = tk.Frame(parent, bg=ACCENT, height=1)
        f.pack(fill="x")
        tk.Label(parent, text=text, font=("Courier",9,"bold"),
                 bg=parent.cget("bg"), fg=BLUE, pady=4).pack(anchor="w", padx=8)

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
              background=[("selected", "#2a4a6b")],
              foreground=[("selected", WHITE)])

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
        sb.pack(side="right",  fill="y", pady=(0,4), padx=(0,4))

    # ══════════════════════════════════════════════════════
    # EQUITY CURVE DRAWING
    # ══════════════════════════════════════════════════════

    def _draw_equity(self):
        c = self.eq_canvas
        c.delete("all")
        W = c.winfo_width(); H = c.winfo_height()
        if W < 20 or H < 20:
            return

        PL, PR, PT, PB = 52, 12, 18, 30   # padding

        if len(self.equity_pts) < 2:
            # placeholder grid while loading
            c.create_text(W//2, H//2, text="Awaiting data…",
                          font=("Courier",11), fill=GREY)
            return

        ticks   = [t for t, _ in self.equity_pts]
        equities= [e for _, e in self.equity_pts]
        mn = min(equities + [STARTING_CAPITAL]) - 50
        mx = max(equities + [STARTING_CAPITAL]) + 50
        rng = mx - mn or 1
        t_mn = ticks[0]; t_mx = max(ticks[-1], self.total_ticks-1)
        t_rng = t_mx - t_mn or 1

        def sx(t): return PL + (t-t_mn)/t_rng*(W-PL-PR)
        def sy(v): return PT + (1-(v-mn)/rng)*(H-PT-PB)

        # ── grid ──
        n_y = 5
        for i in range(n_y+1):
            val = mn + i*rng/n_y
            y   = sy(val)
            c.create_line(PL, y, W-PR, y, fill=BG4, dash=(3,5))
            c.create_text(PL-4, y, text=f"₹{val/1000:.1f}k",
                          font=("Courier",7), fill=GREY, anchor="e")

        # time axis ticks
        for h in range(9, 16):
            frac = (h*60 - 9*60) / (6*60+15)
            x = PL + frac*(W-PL-PR)
            if PL <= x <= W-PR:
                c.create_line(x, H-PB, x, H-PB+4, fill=GREY)
                c.create_text(x, H-PB+10, text=f"{h:02d}:00",
                              font=("Courier",7), fill=GREY)

        # ── starting-capital reference ──
        y0 = sy(STARTING_CAPITAL)
        c.create_line(PL, y0, W-PR, y0, fill=GREY, dash=(6,4), width=1)
        c.create_text(W-PR+2, y0, text="Start", font=("Courier",7),
                      fill=GREY, anchor="w")

        # ── equity polyline ──
        pts = [(sx(t), sy(e)) for t, e in self.equity_pts]
        # draw filled polygon for area under/over baseline
        y0c = sy(STARTING_CAPITAL)
        poly_above = []
        poly_below = []
        cur_above  = []
        cur_below  = []
        for px, py in pts:
            if py <= y0c:    # above baseline
                if cur_below:
                    poly_below.append(cur_below + [(cur_below[-1][0], y0c), (cur_below[0][0], y0c)])
                    cur_below = []
                cur_above.append((px, py))
            else:
                if cur_above:
                    poly_above.append(cur_above + [(cur_above[-1][0], y0c), (cur_above[0][0], y0c)])
                    cur_above = []
                cur_below.append((px, py))
        if cur_above:
            poly_above.append(cur_above + [(cur_above[-1][0], y0c), (cur_above[0][0], y0c)])
        if cur_below:
            poly_below.append(cur_below + [(cur_below[-1][0], y0c), (cur_below[0][0], y0c)])

        for segs in poly_above:
            flat = [coord for p in segs for coord in p]
            if len(flat) >= 6:
                c.create_polygon(flat, fill="#1a3a2a", outline="")
        for segs in poly_below:
            flat = [coord for p in segs for coord in p]
            if len(flat) >= 6:
                c.create_polygon(flat, fill="#3a1a1a", outline="")

        # main line, coloured segment by segment
        for i in range(len(pts)-1):
            col = GREEN if equities[i] >= STARTING_CAPITAL else RED
            c.create_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1],
                          fill=col, width=2)

        # current-value dot + label
        lx, ly = pts[-1]
        cur_eq = equities[-1]
        col = GREEN if cur_eq >= STARTING_CAPITAL else RED
        c.create_oval(lx-5, ly-5, lx+5, ly+5, fill=col, outline=BG, width=2)
        pnl = cur_eq - STARTING_CAPITAL
        pct = pnl/STARTING_CAPITAL*100
        label = f"₹{cur_eq:,.0f}  ({pct:+.2f}%)"
        c.create_text(lx, ly-14, text=label, font=("Courier",9,"bold"),
                      fill=col, anchor="center")

    # ══════════════════════════════════════════════════════
    # POSITIONS PANEL
    # ══════════════════════════════════════════════════════

    def _refresh_positions(self):
        for w in self.pos_frame.winfo_children():
            w.destroy()

        if not self.positions:
            tk.Label(self.pos_frame, text="   — no open positions —",
                     font=("Courier",9), bg=BG2, fg=GREY, pady=6).pack(anchor="w")
            return

        for name, pos in self.positions.items():
            cur_px  = pos.get("cur_px", pos["entry"])
            pnl_abs = (cur_px - pos["entry"]) * pos["shares"]
            pnl_pct = pnl_abs / pos["invested"] * 100 if pos["invested"] else 0
            col     = GREEN if pnl_abs >= 0 else RED
            bg_card = "#182030" if pnl_abs >= 0 else "#2a1520"

            card = tk.Frame(self.pos_frame, bg=bg_card, pady=5, padx=8)
            card.pack(fill="x", pady=(0,3))

            # row 1: name + price
            r1 = tk.Frame(card, bg=bg_card); r1.pack(fill="x")
            tk.Label(r1, text=name, font=("Courier",10,"bold"),
                     bg=bg_card, fg=YELLOW).pack(side="left")
            tk.Label(r1, text=f"₹{cur_px:.2f}", font=("Courier",10,"bold"),
                     bg=bg_card, fg=col).pack(side="right")

            # row 2: entry / shares
            r2 = tk.Frame(card, bg=bg_card); r2.pack(fill="x")
            tk.Label(r2, text=f"Entry ₹{pos['entry']:.2f}  ×{pos['shares']:.1f} sh",
                     font=("Courier",8), bg=bg_card, fg=FG2).pack(side="left")
            tk.Label(r2, text=f"{pnl_pct:+.2f}%", font=("Courier",8,"bold"),
                     bg=bg_card, fg=col).pack(side="right")

            # row 3: SL / TP / unrealised
            r3 = tk.Frame(card, bg=bg_card); r3.pack(fill="x")
            tk.Label(r3, text=f"SL ₹{pos['stop']:.2f}",
                     font=("Courier",8), bg=bg_card, fg=RED).pack(side="left")
            tk.Label(r3, text=f"TP ₹{pos['target']:.2f}",
                     font=("Courier",8), bg=bg_card, fg=GREEN).pack(side="left", padx=10)
            tk.Label(r3, text=f"₹{pnl_abs:+.1f}",
                     font=("Courier",8,"bold"), bg=bg_card, fg=col).pack(side="right")

            if pos.get("trail"):
                tk.Label(card, text="⟳ trailing stop active",
                         font=("Courier",7), bg=bg_card, fg=ORANGE).pack(anchor="w")

            # progress bar showing where price sits between SL and TP
            pbar = tk.Canvas(card, height=5, bg=ACCENT,
                             highlightthickness=0)
            pbar.pack(fill="x", pady=(3,0))
            pbar.update_idletasks()
            bw = pbar.winfo_width() or 380
            sl = pos["stop"]; tp = pos["target"]; rng_ = tp - sl
            if rng_ > 0:
                frac = max(0., min(1., (cur_px-sl)/rng_))
                pbar.create_rectangle(0, 0, int(bw*frac), 5,
                                      fill=col, outline="")

    # ══════════════════════════════════════════════════════
    # SIGNAL SCANNER
    # ══════════════════════════════════════════════════════

    def _ensure_scanner_row(self, name):
        if name in self.scanner_rows:
            return
        f = tk.Frame(self.scanner_outer, bg=BG3, pady=4, padx=6)
        f.pack(fill="x", pady=1)

        tk.Label(f, text=f"{name[:13]:<13}", font=("Courier",8,"bold"),
                 bg=BG3, fg=FG, width=13, anchor="w").pack(side="left")

        bar_bg = tk.Frame(f, bg=ACCENT, height=12, width=88)
        bar_bg.pack(side="left", padx=(4,0))
        bar_bg.pack_propagate(False)
        bar_fill = tk.Frame(bar_bg, bg=GREY, height=12, width=0)
        bar_fill.place(x=0, y=0, height=12)
        bar_lbl  = tk.Label(bar_bg, text="", font=("Courier",7),
                             bg=GREY, fg=BG)
        bar_lbl.place(relx=0.5, rely=0.5, anchor="center")

        lbl_p = tk.Label(f, text=" p=-.---", font=("Courier",8),
                         bg=BG3, fg=GREY, width=8)
        lbl_p.pack(side="left", padx=(4,0))
        lbl_r = tk.Label(f, text=" -.---% ", font=("Courier",8),
                         bg=BG3, fg=GREY, width=9)
        lbl_r.pack(side="left")

        self.scanner_rows[name] = {
            "frame": f, "bar_bg": bar_bg, "bar_fill": bar_fill,
            "bar_lbl": bar_lbl, "lbl_p": lbl_p, "lbl_r": lbl_r
        }

    def _update_scanner(self, name, proba, pred_ret, fired):
        self._ensure_scanner_row(name)
        row = self.scanner_rows[name]
        bg_col   = BG3
        bar_col  = GREY

        if fired:
            bg_col  = DKGREEN
            bar_col = GREEN
        elif proba >= MIN_CONFIDENCE * 0.7:
            bar_col = BLUE

        # update bar width
        bw = row["bar_bg"].winfo_width() or 88
        fw = max(1, int(bw * min(proba, 1.0)))
        row["bar_fill"].place(x=0, y=0, width=fw, height=12)
        row["bar_fill"].config(bg=bar_col)
        row["bar_lbl"].config(text=f"{proba:.2f}", bg=bar_col)
        row["bar_bg"].config(bg=ACCENT)

        pc = GREEN if pred_ret > 0 else (RED if pred_ret < 0 else GREY)
        row["lbl_p"].config(text=f"p={proba:.3f}", fg=bar_col, bg=bg_col)
        row["lbl_r"].config(text=f"{pred_ret*100:+.3f}%", fg=pc, bg=bg_col)

        for widget in (row["frame"], row["lbl_p"], row["lbl_r"]):
            widget.config(bg=bg_col)

        # flash-reset after 1 s if fired
        if fired:
            if name in self.flash_ids:
                self.root.after_cancel(self.flash_ids[name])
            self.flash_ids[name] = self.root.after(
                1200, lambda n=name: self._reset_scanner_row(n))

    def _reset_scanner_row(self, name):
        if name not in self.scanner_rows:
            return
        row = self.scanner_rows[name]
        for widget in (row["frame"], row["lbl_p"], row["lbl_r"]):
            try: widget.config(bg=BG3)
            except: pass

    # ══════════════════════════════════════════════════════
    # EVENT POLLING (main thread)
    # ══════════════════════════════════════════════════════

    def _poll(self):
        processed = 0
        try:
            while processed < 30:
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
            for n in names:
                self._ensure_scanner_row(n)

        elif kind == "tick":
            _, tick_i, ts, cash, equity, pnl, open_val, pos_snap = ev
            self.equity_pts.append((tick_i, equity))
            self.positions = pos_snap

            # header
            self.lbl_time.config(text=ts)
            pct  = pnl / STARTING_CAPITAL * 100
            col  = GREEN if pnl >= 0 else RED
            sign = "+" if pnl >= 0 else ""

            self.sv_equity.set(f"₹{equity:,.0f}")
            self.sv_cash.set(f"₹{cash:,.0f}")
            self.sv_open_val.set(f"₹{open_val:,.0f}")
            self.sv_pnl.set(f"{sign}₹{pnl:,.0f}  ({pct:+.2f}%)")
            self.lbl_pnl_box.config(fg=col)
            self.lbl_equity.config(fg=col)

            tot  = self.win_count + self.loss_count
            self.sv_trades.set(str(tot))
            wp   = self.win_count/tot*100 if tot else 0
            lp   = self.loss_count/tot*100 if tot else 0
            self.sv_wins.set(f"{self.win_count}  ({wp:.0f}%)")
            self.sv_losses.set(f"{self.loss_count}  ({lp:.0f}%)")
            self.sv_slots.set(f"{len(pos_snap)} / {MAX_POSITIONS}")

            self.prog_var.set(tick_i / max(self.total_ticks-1, 1) * 100)
            self._refresh_positions()
            self._draw_equity()

        elif kind == "exit":
            _, ts, name, entry, exit_px, pnl_val, pnl_pct, reason = ev
            if pnl_val > 0:
                self.win_count  += 1; tag = "win"
            elif reason == "EOD CLOSE":
                tag = "eod"
                if pnl_val > 0: self.win_count += 1
                else:            self.loss_count += 1
            else:
                self.loss_count += 1; tag = "loss"
            sign  = "+" if pnl_val >= 0 else ""
            self.tree.insert("", 0,
                values=(ts, name,
                        f"₹{entry:.2f}", f"₹{exit_px:.2f}",
                        f"{sign}₹{pnl_val:.1f}", reason),
                tags=(tag,))

        elif kind == "signal":
            _, name, proba, pred_ret, fired = ev
            self._update_scanner(name, proba, pred_ret, fired)

        elif kind == "done":
            _, final_eq, _ = ev
            pnl = final_eq - STARTING_CAPITAL
            col = GREEN if pnl >= 0 else RED
            sign = "+" if pnl >= 0 else ""
            self.lbl_status.config(
                text=f"✓ SIMULATION COMPLETE — "
                     f"Final P&L: {sign}₹{pnl:,.2f}  "
                     f"({pnl/STARTING_CAPITAL*100:+.2f}%)",
                fg=col)
            self.prog_var.set(100)

    # ══════════════════════════════════════════════════════
    # SIMULATION THREAD
    # ══════════════════════════════════════════════════════

    def _run_sim(self):
        # ── load model ────────────────────────────────────
        self.q.put(("status", "Loading model bundle…"))
        try:
            bundle  = joblib.load(MODEL_BUNDLE)
            xgb_clf = bundle["xgb_clf"];  lgbm_clf = bundle["lgbm_clf"]
            rf_clf  = bundle["rf_clf"];   meta_clf  = bundle["meta_clf"]
            xgb_reg = bundle["xgb_reg"];  lgbm_reg = bundle["lgbm_reg"]
            rf_reg  = bundle["rf_reg"];   meta_reg  = bundle["meta_reg"]
            encoder = bundle["encoder"]
            feat_cols= bundle["feat_cols"]
        except Exception as e:
            self.q.put(("status", f"ERROR: {e}")); return

        ENTRY_THRESHOLD = MIN_CONFIDENCE

        def model_predict(X_df):
            bc = np.column_stack([xgb_clf.predict_proba(X_df)[:,1],
                                   lgbm_clf.predict_proba(X_df)[:,1],
                                   rf_clf.predict_proba(X_df)[:,1]])
            proba = meta_clf.predict_proba(bc)[:,1]
            br = np.column_stack([xgb_reg.predict(X_df),
                                   lgbm_reg.predict(X_df),
                                   rf_reg.predict(X_df)])
            ret = meta_reg.predict(br)
            return float(proba[0]), float(ret[0])

        def kelly(p, win_r):
            if win_r <= 0: win_r = STOP_LOSS_PCT * 0.5
            b = win_r / STOP_LOSS_PCT
            f = (p*b - (1-p)) / b
            return max(0., min(f, POSITION_CAP))

        # ── load indices ──────────────────────────────────
        self.q.put(("status", "Loading index data…"))
        index_data = {}
        for file in glob.glob(f"{INDICES_FOLDER}/*.json"):
            iname = file.split("\\")[-1].replace("_historical_data.json","")
            with open(file,"r",encoding="utf-8") as f:
                ds = json.load(f)
            lk = {}
            for day in ds["data"]:
                for c_ in to_15min(day["candles"]):
                    lk[str(pd.Timestamp(c_["timestamp"]))[:19]] = c_
                    lk[c_["timestamp"]] = c_
            index_data[iname] = lk

        # ── load stocks ───────────────────────────────────
        self.q.put(("status", "Loading stocks…"))
        stocks = {}
        for file in glob.glob(f"{COMPANIES_FOLDER}/*.json"):
            name = file.split("\\")[-1].replace("_with_quarterly_results.json","")
            with open(file,"r",encoding="utf-8") as f:
                ds = json.load(f)
            all_days = ds["data"]
            if len(all_days) < 2: continue
            chosen = None
            for di in range(len(all_days)-1, 0, -1):
                if len(all_days[di]["candles"]) >= 50: chosen=di; break
            if chosen is None:
                for di in range(len(all_days)-1, 0, -1):
                    if len(all_days[di]["candles"]) >= 10: chosen=di; break
            if chosen is None: continue
            live_day  = all_days[chosen]
            live_1min = live_day["candles"]
            quarterly = live_day.get("latest_quarterly_result",{})
            news      = live_day.get("news_analysis",{})
            ctx_1min  = []
            for di in range(chosen-1, max(chosen-4,-1), -1):
                ctx_1min = all_days[di]["candles"] + ctx_1min
                if len(to_15min(ctx_1min)) >= 30: break
            stocks[name] = {"live_1min": live_1min,
                            "context_15": to_15min(ctx_1min),
                            "quarterly": quarterly, "news": news}

        if not stocks:
            self.q.put(("status", "ERROR: no stocks loaded.")); return

        all_1min_ts = sorted(set(
            c["timestamp"] for s in stocks.values() for c in s["live_1min"]
        ))
        self.q.put(("init", len(all_1min_ts), list(stocks.keys())))
        self.q.put(("status", "SIMULATING…"))

        # ── simulation state ──────────────────────────────
        capital        = STARTING_CAPITAL
        positions      = {}
        seen_1min      = {n: [] for n in stocks}
        last_bar_fired = {n: None for n in stocks}

        for tick_i, ts in enumerate(all_1min_ts):
            ts_dt   = pd.Timestamp(ts)
            is_last = (ts == all_1min_ts[-1])

            # ingest this 1-min candle per stock
            for name, s in stocks.items():
                for c_ in s["live_1min"]:
                    if c_["timestamp"] == ts:
                        seen_1min[name].append(c_); break

            # monitor open positions
            for name, pos in list(positions.items()):
                cur1 = next((c_ for c_ in reversed(seen_1min[name])
                             if c_["timestamp"]==ts), None)
                if cur1 is None and seen_1min[name]:
                    cur1 = seen_1min[name][-1]
                if cur1 is None: continue

                hi=cur1["high"]; lo=cur1["low"]; px=cur1["close"]
                pos["cur_px"] = px

                if hi > pos["peak"]:
                    pos["peak"] = hi
                    if pos["peak"] >= pos["entry"]*(1+TRAILING_TRIGGER):
                        new_ts = pos["peak"]*(1-TRAILING_STOP_PCT)
                        if new_ts > pos["stop"]:
                            pos["stop"]=new_ts; pos["trail"]=True

                reason=None; xp=px
                if lo <= pos["stop"]:     xp=pos["stop"];   reason="STOP LOSS"
                elif hi >= pos["target"]: xp=pos["target"]; reason="TARGET HIT"
                elif is_last:                               reason="EOD CLOSE"

                if reason:
                    proceeds = pos["shares"]*xp
                    pnl_abs  = proceeds - pos["invested"]
                    capital += proceeds
                    self.q.put(("exit", ts_dt.strftime("%H:%M"), name,
                                pos["entry"], xp, pnl_abs,
                                pnl_abs/pos["invested"]*100, reason))
                    del positions[name]

            # signal scan at every 15-min bar close
            is_bar_close = (ts_dt.minute % 15 == 14)
            free_slots   = MAX_POSITIONS - len(positions)

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
                        feats = build_features(name, full_15, cl15, idx,
                                               s["quarterly"], s["news"],
                                               index_data, encoder, feat_cols)
                    except: continue
                    X = pd.DataFrame([feats])[feat_cols]
                    try: proba, pred_ret = model_predict(X)
                    except: continue
                    last_bar_fired[name] = bar_ts
                    fired = (proba >= ENTRY_THRESHOLD and pred_ret >= MIN_PRED_RETURN)
                    self.q.put(("signal", name, proba, pred_ret, fired))
                    if fired:
                        score = proba*pred_ret if pred_ret > 0 else proba
                        candidates.append(dict(name=name, proba=proba,
                                               pred_ret=pred_ret,
                                               price=full_15[idx]["close"],
                                               score=score))

                candidates.sort(key=lambda x: x["score"], reverse=True)
                for cand in candidates[:free_slots]:
                    name=cand["name"]; entry=cand["price"]
                    proba=cand["proba"]; pred_ret=cand["pred_ret"]
                    frac   = kelly(proba, pred_ret)
                    invest = capital * frac
                    if invest < MIN_INVEST:
                        invest = min(capital*0.05, capital)
                    if invest < MIN_INVEST or invest > capital: continue
                    shares = invest / entry
                    stop   = entry*(1-STOP_LOSS_PCT)
                    eff_r  = pred_ret if pred_ret > 0 else STOP_LOSS_PCT
                    target = entry*(1+eff_r*REWARD_MULT)
                    capital -= invest
                    positions[name] = dict(shares=shares, entry=entry,
                                           stop=stop, target=target,
                                           peak=entry, trail=False,
                                           invested=invest, proba=proba,
                                           pred_ret=pred_ret, entry_ts=ts,
                                           cur_px=entry)

            # portfolio snapshot → UI
            open_val = sum(
                p["shares"]*(seen_1min[n][-1]["close"] if seen_1min[n] else p["entry"])
                for n, p in positions.items()
            )
            equity  = capital + open_val
            pnl     = equity - STARTING_CAPITAL
            pos_snap = {n: dict(p) for n, p in positions.items()}
            self.q.put(("tick", tick_i, ts_dt.strftime("%H:%M"),
                        capital, equity, pnl, open_val, pos_snap))

            time.sleep(SIMULATE_DELAY)

        self.q.put(("done", capital, []))


# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.iconbitmap("")       # suppress missing icon warning
    except Exception:
        pass
    app = Dashboard(root)
    root.mainloop()