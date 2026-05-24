# =========================================================
# MODEL v3 — TEST / INFERENCE SCRIPT
# =========================================================
#
# TWO MODES — set TEST_MODE below:
#
#   "backtest"  →  Run on historical company JSON files,
#                  compare predictions to actual outcomes,
#                  print accuracy / MAE / trade stats.
#
#   "live"      →  Pass a single stock's latest candles
#                  and get a BUY/HOLD signal + predicted
#                  return for the next 15-min candle.
#
# =========================================================

import json
import glob
import joblib
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    average_precision_score,
)

# =========================================================
# CONFIG
# =========================================================

MODEL_BUNDLE    = "model_v3_classifier.pkl"   # contains ALL models
INDICES_FOLDER  = "Indexes"
COMPANIES_FOLDER= "Companies"

# "backtest" → evaluate on historical data
# "live"     → predict the very next candle for one stock
TEST_MODE = "backtest"

# For live mode: path to a single company JSON file
LIVE_STOCK_FILE = "Companies/paytm_with_quarterly_results.json"

# Confidence level — only show signals above this probability
CONFIDENCE_FLOOR = 0.55

# =========================================================
# LOAD MODEL BUNDLE
# =========================================================

print("Loading model bundle...")
bundle = joblib.load(MODEL_BUNDLE)

xgb_clf   = bundle["xgb_clf"]
lgbm_clf  = bundle["lgbm_clf"]
rf_clf    = bundle["rf_clf"]
meta_clf  = bundle["meta_clf"]

xgb_reg   = bundle["xgb_reg"]
lgbm_reg  = bundle["lgbm_reg"]
rf_reg    = bundle["rf_reg"]
meta_reg  = bundle["meta_reg"]

encoder   = bundle["encoder"]
threshold = bundle["threshold"]
feat_cols = bundle["feat_cols"]

print(f"  Threshold  : {threshold:.4f}")
print(f"  Features   : {len(feat_cols)}")
print(f"  Stocks known: {list(encoder.classes_)}")

# =========================================================
# INFERENCE HELPERS
# =========================================================

def predict_proba(X: pd.DataFrame) -> np.ndarray:
    """Returns bullish probability for each row in X."""
    base = np.column_stack([
        xgb_clf.predict_proba(X)[:, 1],
        lgbm_clf.predict_proba(X)[:, 1],
        rf_clf.predict_proba(X)[:, 1],
    ])
    return meta_clf.predict_proba(base)[:, 1]

def predict_return(X: pd.DataFrame) -> np.ndarray:
    """Returns predicted % return for each row in X."""
    base = np.column_stack([
        xgb_reg.predict(X),
        lgbm_reg.predict(X),
        rf_reg.predict(X),
    ])
    return meta_reg.predict(base)

def signal_label(proba: float, thresh: float) -> str:
    if proba >= thresh + 0.15:  return "STRONG BUY"
    if proba >= thresh:         return "BUY"
    return "HOLD / SKIP"

# =========================================================
# SHARED UTILITIES  (identical to training script)
# =========================================================

def convert_to_15min(candles):
    if len(candles) < 15:
        return []
    df = pd.DataFrame(candles)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    r = df.resample("15min").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),   close=("close","last"),
        volume=("volume","sum")
    ).dropna()
    return [
        {"timestamp": ts.isoformat(),
         "open": float(v.open), "high": float(v.high),
         "low":  float(v.low),  "close": float(v.close),
         "volume": float(v.volume)}
        for ts, v in r.iterrows()
    ]

def clean_number(v):
    try:   return float(str(v).replace(",","").replace("%",""))
    except: return 0.0

def rsi(closes, p=14):
    diffs = np.diff(closes[-p-1:])
    g = diffs[diffs > 0].mean() if (diffs > 0).any() else 1e-9
    l = (-diffs[diffs < 0]).mean() if (diffs < 0).any() else 1e-9
    return 100 - 100 / (1 + g / l)

def ema_series(arr, span):
    k, s = 2/(span+1), arr[0]
    out = [s]
    for v in arr[1:]:
        s = v*k + s*(1-k)
        out.append(s)
    return out

def macd(closes):
    if len(closes) < 27: return 0., 0., 0.
    e12 = ema_series(closes, 12)
    e26 = ema_series(closes, 26)
    ml  = [a-b for a,b in zip(e12,e26)]
    sig = ema_series(ml[-9:], 9)[-1] if len(ml)>=9 else 0.
    return ml[-1], sig, ml[-1]-sig

def atr(candles, p=14):
    trs = [max(c["high"]-c["low"],
               abs(c["high"]-candles[i-1]["close"]),
               abs(c["low"] -candles[i-1]["close"]))
           for i,c in enumerate(candles[-p:], 1) if i>0]
    return np.mean(trs) if trs else 1e-9

def stoch(candles, p=14):
    sl = candles[-p:]
    H,L,C = max(c["high"] for c in sl), min(c["low"] for c in sl), sl[-1]["close"]
    return (C-L)/(H-L)*100 if H!=L else 50.

def cci(candles, p=20):
    tps = [(c["high"]+c["low"]+c["close"])/3 for c in candles[-p:]]
    m   = np.mean(tps)
    mad = np.mean(np.abs(np.array(tps)-m))
    return (tps[-1]-m)/(0.015*mad) if mad else 0.

def vwap(candles):
    tp  = np.array([(c["high"]+c["low"]+c["close"])/3 for c in candles])
    vol = np.array([c["volume"] for c in candles])
    return (tp*vol).sum()/vol.sum() if vol.sum() else candles[-1]["close"]

def bollinger(closes, p=20):
    ma,sd = np.mean(closes[-p:]), np.std(closes[-p:])
    up,lo = ma+2*sd, ma-2*sd
    rng   = up-lo
    return ma, up, lo, rng/ma if ma else 0, (closes[-1]-lo)/rng if rng else .5

def rolling_skew(arr):
    if len(arr)<3: return 0.
    m,s = np.mean(arr), np.std(arr)
    return np.mean(((np.array(arr)-m)/(s+1e-9))**3)

def rolling_kurt(arr):
    if len(arr)<4: return 0.
    m,s = np.mean(arr), np.std(arr)
    return np.mean(((np.array(arr)-m)/(s+1e-9))**4) - 3

def pattern(cur, prev):
    body_c  = abs(cur["close"]-cur["open"])
    body_p  = abs(prev["close"]-prev["open"])
    rng_c   = cur["high"]-cur["low"]
    lower_s = min(cur["close"],cur["open"])-cur["low"]
    upper_s = cur["high"]-max(cur["close"],cur["open"])
    bull_c  = cur["close"]  > cur["open"]
    bull_p  = prev["close"] > prev["open"]
    doji       = int(rng_c>0 and body_c/rng_c < 0.1)
    hammer     = int(rng_c>0 and lower_s>2*body_c and upper_s<body_c)
    shoot_star = int(rng_c>0 and upper_s>2*body_c and lower_s<body_c)
    engulf     = int(bull_c and not bull_p and body_c>body_p)
    return doji, hammer, shoot_star, int(bull_c), engulf

# =========================================================
# FEATURE BUILDER
# Returns one feature-row dict for candle at index i.
# Pass include_target=True for backtest, False for live.
# =========================================================

def build_feature_row(
    stock_name, candles, closes, i,
    quarterly, news, index_data,
    include_target=True
):
    cur = candles[i]
    ts  = cur["timestamp"]
    cp  = cur["close"];  op = cur["open"]
    hp  = cur["high"];   lp = cur["low"]
    vol = cur["volume"]

    # ---- Quarterly ----
    q      = quarterly
    sales  = clean_number(q.get("Sales", 0))
    profit = clean_number(q.get("Net Profit", 0))
    opm    = clean_number(q.get("OPM %", 0))
    pm     = profit / sales * 100 if sales else 0.

    # ---- News ----
    nws      = news
    agg_sent = nws.get("aggregate_sentiment", 0)
    sig      = nws.get("market_signal", "NEUTRAL")
    bull_news= int(sig == "BULLISH")
    bear_news= int(sig == "BEARISH")
    nat_cnt  = len(nws.get("national_news", []))
    int_cnt  = len(nws.get("international_news", []))
    stk_cnt  = len(nws.get("stock_news", []))

    # ---- Windows ----
    win20 = candles[i-20:i+1]
    win14 = candles[i-14:i+1]
    win5  = candles[i-5: i+1]
    cl20  = closes[i-20:i+1]
    cl14  = closes[i-14:i+1]
    cl5   = closes[i-5: i+1]

    # ---- Candle ratios ----
    candle_body  = (cp - op) / (cp + 1e-9)
    candle_range = (hp - lp) / (cp + 1e-9)
    upper_shadow = (hp - max(cp,op)) / (cp + 1e-9)
    lower_shadow = (min(cp,op) - lp)  / (cp + 1e-9)
    open_gap     = (op - candles[i-1]["close"]) / (candles[i-1]["close"] + 1e-9)

    # ---- MAs ----
    ma5  = np.mean(closes[i-5: i])
    ma10 = np.mean(closes[i-10:i])
    ma20 = np.mean(closes[i-20:i])
    std20= np.std(closes[i-20:i])

    r_ma5  = (cp-ma5)  / (ma5  + 1e-9)
    r_ma10 = (cp-ma10) / (ma10 + 1e-9)
    r_ma20 = (cp-ma20) / (ma20 + 1e-9)
    ma5_slope  = (closes[i-1]-closes[i-5])  / (closes[i-5]  + 1e-9)
    ma10_slope = (closes[i-1]-closes[i-10]) / (closes[i-10] + 1e-9)

    # ---- Returns ----
    r1  = (cp-closes[i-1]) / (closes[i-1]+1e-9)
    r3  = (cp-closes[i-3]) / (closes[i-3]+1e-9)
    r5  = (cp-closes[i-5]) / (closes[i-5]+1e-9)
    r10 = (cp-closes[i-10])/ (closes[i-10]+1e-9)
    r1_lag1    = (closes[i-1]-closes[i-2]) / (closes[i-2]+1e-9)
    r1_lag2    = (closes[i-2]-closes[i-3]) / (closes[i-3]+1e-9)
    ret_autocorr = r1 * r1_lag1

    # ---- RSI ----
    rsi_val = rsi(cl14)
    rsi_ob  = int(rsi_val > 70)
    rsi_os  = int(rsi_val < 30)

    # ---- MACD ----
    macd_v, macd_sig, macd_hist = macd(closes[i-26:i+1])

    # ---- ATR ----
    atr_v    = atr(win20)
    atr_pct  = atr_v / (cp + 1e-9)
    crng_atr = candle_range / (atr_pct + 1e-9)

    # ---- Stochastic ----
    stoch_k  = stoch(win14)
    stoch_ob = int(stoch_k > 80)
    stoch_os = int(stoch_k < 20)

    # ---- CCI ----
    cci_v = cci(win20)

    # ---- VWAP ----
    vwap_v = vwap(candles[:i+1])
    r_vwap = (cp - vwap_v) / (vwap_v + 1e-9)

    # ---- Bollinger ----
    bb_ma, bb_up, bb_lo, bb_w, bb_pb = bollinger(closes[i-20:i+1])
    bb_squeeze = int(bb_w < np.mean([
        (max(closes[j-20:j])+2*np.std(closes[j-20:j]) -
         max(closes[j-20:j])+2*np.std(closes[j-20:j]))
        for j in range(i-5, i+1) if j>=20
    ] or [0.01]) * 0.8)

    # ---- Volume ----
    vol_ma5   = np.mean([candles[i-k]["volume"] for k in range(1,6)])
    rel_vol   = vol / (vol_ma5 + 1e-9)
    vol_spike = int(rel_vol > 2.0)

    # ---- Rolling stats ----
    skew20 = rolling_skew(cl20)
    kurt20 = rolling_kurt(cl20)
    skew5  = rolling_skew(cl5)

    # ---- Momentum ----
    mom     = r5
    mom_acc = r1 - r1_lag1

    # ---- Patterns ----
    doji_v, hammer_v, shoot_v, bull_c, engulf_v = pattern(cur, candles[i-1])

    # ---- Structure ----
    hh = int(hp > max(c["high"] for c in win5[:-1]))
    ll = int(lp < min(c["low"]  for c in win5[:-1]))

    # ---- Interactions ----
    rsi_macd     = rsi_val * macd_hist
    vol_atr      = rel_vol * atr_pct
    price_bb_rsi = bb_pb * rsi_val / 100

    # ---- Index features ----
    def idx_ret(name):
        c = index_data.get(name, {}).get(ts)
        return (c["close"]-c["open"])/(c["open"]+1e-9) if c else 0.

    ni  = idx_ret("NIFTY")
    nit = idx_ret("NIFTYIT")
    n50 = idx_ret("NIFTY_NEXT_50")
    gft = idx_ret("GIFTNIFTY")
    idx_avg  = np.mean([ni, nit, n50, gft])
    idx_bull = sum(1 for r in [ni, nit, n50, gft] if r > 0)
    diverge_nifty = r1 - ni

    # ---- Time slot ----
    try:
        t    = pd.Timestamp(ts)
        slot = t.hour * 4 + t.minute // 15
    except:
        slot = 0

    # ---- Encode stock ----
    if stock_name in encoder.classes_:
        stock_enc = int(encoder.transform([stock_name])[0])
    else:
        stock_enc = -1   # unseen stock — model will extrapolate

    row = dict(
        stock_enc=stock_enc,
        time_slot=slot,
        candle_body=candle_body, candle_range=candle_range,
        upper_shadow=upper_shadow, lower_shadow=lower_shadow,
        open_gap=open_gap,
        r1=r1, r3=r3, r5=r5, r10=r10,
        r1_lag1=r1_lag1, r1_lag2=r1_lag2,
        ret_autocorr=ret_autocorr,
        r_ma5=r_ma5, r_ma10=r_ma10, r_ma20=r_ma20,
        ma5_slope=ma5_slope, ma10_slope=ma10_slope,
        std20=std20,
        rsi=rsi_val, rsi_ob=rsi_ob, rsi_os=rsi_os,
        macd=macd_v, macd_sig=macd_sig, macd_hist=macd_hist,
        atr_pct=atr_pct, crng_atr=crng_atr,
        stoch_k=stoch_k, stoch_ob=stoch_ob, stoch_os=stoch_os,
        cci=cci_v,
        r_vwap=r_vwap,
        bb_width=bb_w, bb_pct_b=bb_pb, bb_squeeze=bb_squeeze,
        rel_vol=rel_vol, vol_spike=vol_spike,
        skew20=skew20, kurt20=kurt20, skew5=skew5,
        mom=mom, mom_acc=mom_acc,
        doji=doji_v, hammer=hammer_v, shoot_star=shoot_v,
        bull_candle=bull_c, engulf=engulf_v,
        higher_high=hh, lower_low=ll,
        rsi_macd=rsi_macd, vol_atr=vol_atr,
        price_bb_rsi=price_bb_rsi,
        nifty_r=ni, niftyit_r=nit, next50_r=n50, gift_r=gft,
        idx_avg=idx_avg, idx_bull=idx_bull,
        diverge_nifty=diverge_nifty,
        sales=sales, profit=profit, opm=opm, profit_margin=pm,
        agg_sent=agg_sent, bull_news=bull_news, bear_news=bear_news,
        nat_cnt=nat_cnt, int_cnt=int_cnt, stk_cnt=stk_cnt,
    )

    if include_target:
        fut_c   = candles[i+1]["close"]
        fr      = (fut_c - cp) / (cp + 1e-9)
        dir_tgt = int(fr > 0.003)
        row["direction_target"] = dir_tgt
        row["return_target"]    = fr
        row["close_price"]      = cp
        row["timestamp"]        = ts
        row["stock_name"]       = stock_name

    return row

# =========================================================
# LOAD INDEX DATA  (shared by both modes)
# =========================================================

print("\nLoading index data...")
index_data = {}
for file in glob.glob(f"{INDICES_FOLDER}/*.json"):
    name = file.split("\\")[-1].replace("_historical_data.json","")
    with open(file, "r", encoding="utf-8") as f:
        ds = json.load(f)
    lk = {}
    for day in ds["data"]:
        for c in convert_to_15min(day["candles"]):
            lk[c["timestamp"]] = c
    index_data[name] = lk
    print(f"  Loaded: {name}")

# =========================================================
# =========================================================
#  MODE A — BACKTEST
# =========================================================
# =========================================================

if TEST_MODE == "backtest":

    print("\n" + "="*55)
    print("  BACKTEST MODE")
    print("="*55)

    all_rows = []
    company_files = glob.glob(f"{COMPANIES_FOLDER}/*.json")
    print(f"\nFound {len(company_files)} company files\n")

    for file in company_files:
        stock_name = (
            file.split("\\")[-1]
                .replace("_with_quarterly_results.json","")
        )
        with open(file, "r", encoding="utf-8") as f:
            ds = json.load(f)

        stock_rows = 0
        for day in ds["data"]:
            candles = convert_to_15min(day["candles"])
            if len(candles) < 30:
                continue
            quarterly = day.get("latest_quarterly_result", {})
            news      = day.get("news_analysis", {})
            closes    = [c["close"] for c in candles]

            for i in range(26, len(candles) - 1):
                row = build_feature_row(
                    stock_name, candles, closes, i,
                    quarterly, news, index_data,
                    include_target=True
                )
                all_rows.append(row)
                stock_rows += 1

        print(f"  {stock_name}: {stock_rows} candles")

    # ---- Build DataFrame ----
    df = pd.DataFrame(all_rows)

    meta_cols = ["direction_target","return_target",
                 "close_price","timestamp","stock_name"]
    X    = df[feat_cols]
    y_c  = df["direction_target"]
    y_r  = df["return_target"]

    # ---- Predict ----
    print("\nRunning predictions...")
    proba  = predict_proba(X)
    preds  = (proba >= threshold).astype(int)
    r_pred = predict_return(X)

    df["proba"]       = proba
    df["pred_signal"] = preds
    df["pred_return"] = r_pred

    # =========================================================
    # OVERALL METRICS
    # =========================================================

    ap    = average_precision_score(y_c, proba)
    acc   = accuracy_score(y_c, preds)
    mae   = mean_absolute_error(y_r, r_pred)
    rmse  = np.sqrt(mean_squared_error(y_r, r_pred))
    r2    = r2_score(y_r, r_pred)
    reg_dir_acc = accuracy_score(
        (y_r > 0).astype(int),
        (r_pred > 0).astype(int)
    )

    print("\n" + "="*55)
    print("  OVERALL RESULTS")
    print("="*55)
    print(f"  Samples          : {len(df):,}")
    print(f"  Threshold        : {threshold:.4f}")
    print(f"  Direction Acc    : {acc*100:.2f}%")
    print(f"  Avg Precision    : {ap:.4f}")
    print(f"  Reg MAE          : {mae:.6f}")
    print(f"  Reg RMSE         : {rmse:.6f}")
    print(f"  Reg R²           : {r2:.6f}")
    print(f"  Reg Dir Accuracy : {reg_dir_acc*100:.2f}%")
    print("="*55)

    print("\nClassification Report:")
    print(classification_report(y_c, preds,
          target_names=["HOLD","BUY"]))

    # =========================================================
    # HIGH-CONFIDENCE ACCURACY
    # =========================================================

    print("High-Confidence Accuracy:")
    for conf in [0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = proba >= conf
        n    = mask.sum()
        if n > 0:
            hc_acc = accuracy_score(y_c[mask], preds[mask])
            hit    = y_c[mask].sum()
            print(f"  proba≥{conf:.2f} → Acc={hc_acc*100:.1f}%  "
                  f"BUY signals={n:,}  Hits={hit:,}")

    # =========================================================
    # PER-STOCK BREAKDOWN
    # =========================================================

    print("\nPer-Stock Breakdown:")
    print(f"  {'Stock':<20} {'Candles':>8} {'Acc':>7} "
          f"{'Precision':>10} {'Recall':>8} {'BUY signals':>12}")
    print("  " + "-"*70)

    for stock in df["stock_name"].unique():
        mask     = df["stock_name"] == stock
        y_true_s = y_c[mask]
        y_pred_s = preds[mask]
        proba_s  = proba[mask]

        acc_s  = accuracy_score(y_true_s, y_pred_s)
        ap_s   = average_precision_score(y_true_s, proba_s)
        n_buy  = y_pred_s.sum()
        n_hit  = ((y_pred_s == 1) & (y_true_s == 1)).sum()
        prec_s = n_hit / n_buy if n_buy > 0 else 0.
        rec_s  = n_hit / y_true_s.sum() if y_true_s.sum() > 0 else 0.

        print(f"  {stock:<20} {mask.sum():>8,} {acc_s*100:>6.1f}% "
              f"{prec_s*100:>9.1f}% {rec_s*100:>7.1f}% {n_buy:>12,}")

    # =========================================================
    # SIMULATED TRADE P&L
    # =========================================================
    # Simple simulation: enter at current close when BUY is predicted,
    # exit at next candle close. No slippage or commission assumed.

    print("\n" + "="*55)
    print("  SIMULATED TRADE P&L  (no slippage / commission)")
    print("="*55)

    buy_mask   = preds == 1
    n_trades   = buy_mask.sum()
    actual_ret = y_r[buy_mask]

    wins       = (actual_ret > 0).sum()
    losses     = (actual_ret <= 0).sum()
    avg_ret    = actual_ret.mean() * 100
    total_ret  = actual_ret.sum() * 100
    win_rate   = wins / n_trades * 100 if n_trades else 0
    avg_win    = actual_ret[actual_ret > 0].mean() * 100 if wins else 0
    avg_loss   = actual_ret[actual_ret <= 0].mean()* 100 if losses else 0
    expectancy = (win_rate/100 * avg_win) + ((1-win_rate/100) * avg_loss)

    print(f"  Total Trades     : {n_trades:,}")
    print(f"  Win Rate         : {win_rate:.1f}%")
    print(f"  Avg Return/Trade : {avg_ret:.4f}%")
    print(f"  Total Return     : {total_ret:.2f}%")
    print(f"  Avg Win          : {avg_win:.4f}%")
    print(f"  Avg Loss         : {avg_loss:.4f}%")
    print(f"  Expectancy       : {expectancy:.4f}%")
    print("="*55)

    # High-confidence trades only
    for conf in [0.65, 0.70, 0.75, 0.80]:
        hc_mask = proba >= conf
        nt = hc_mask.sum()
        if nt == 0:
            continue
        ar = y_r[hc_mask]
        wr = (ar > 0).mean() * 100
        avg_r = ar.mean() * 100
        print(f"  Conf≥{conf:.2f} → Trades={nt:,}  "
              f"WinRate={wr:.1f}%  AvgRet={avg_r:.4f}%")

    print("\nBacktest complete.")

# =========================================================
# =========================================================
#  MODE B — LIVE (single stock, latest candles)
# =========================================================
# =========================================================

elif TEST_MODE == "live":

    print("\n" + "="*55)
    print("  LIVE PREDICTION MODE")
    print("="*55)

    stock_name = (
        LIVE_STOCK_FILE.split("/")[-1]
                       .replace("_with_quarterly_results.json","")
    )
    print(f"  Stock: {stock_name}")

    with open(LIVE_STOCK_FILE, "r", encoding="utf-8") as f:
        ds = json.load(f)

    # Use the LAST day's data
    last_day  = ds["data"][-1]
    candles   = convert_to_15min(last_day["candles"])
    quarterly = last_day.get("latest_quarterly_result", {})
    news      = last_day.get("news_analysis", {})
    closes    = [c["close"] for c in candles]

    if len(candles) < 28:
        print("  Not enough candles for prediction (need ≥28).")
    else:
        # Predict on the LAST available candle
        # (next candle = what we're trying to predict)
        i   = len(candles) - 1
        row = build_feature_row(
            stock_name, candles, closes, i,
            quarterly, news, index_data,
            include_target=False          # no future candle available
        )

        X_live = pd.DataFrame([row])[feat_cols]

        prob    = predict_proba(X_live)[0]
        ret_hat = predict_return(X_live)[0]
        sig     = signal_label(prob, threshold)

        cur = candles[i]
        print(f"\n  Timestamp      : {cur['timestamp']}")
        print(f"  Current Close  : ₹{cur['close']:.2f}")
        print(f"  RSI            : {rsi(closes[i-14:i+1]):.1f}")
        print(f"  VWAP           : ₹{vwap(candles[:i+1]):.2f}")
        print()
        print(f"  ┌─────────────────────────────────┐")
        print(f"  │  Signal       : {sig:<17}│")
        print(f"  │  Confidence   : {prob*100:>5.1f}%             │")
        print(f"  │  Pred Return  : {ret_hat*100:>+6.3f}%            │")
        print(f"  │  Threshold    : {threshold:.4f}             │")
        print(f"  └─────────────────────────────────┘")

        if prob >= threshold:
            entry  = cur["close"]
            target = entry * (1 + ret_hat)
            print(f"\n  Entry Price    : ₹{entry:.2f}")
            print(f"  Target Price   : ₹{target:.2f}")

        # Show last 5 candles for context
        print("\n  Last 5 candles:")
        print(f"  {'Time':<22} {'Open':>8} {'High':>8} "
              f"{'Low':>8} {'Close':>8} {'Volume':>12}")
        print("  " + "-"*72)
        for c in candles[-5:]:
            t = c["timestamp"][11:16]
            print(f"  {c['timestamp'][:19]:<22} "
                  f"{c['open']:>8.2f} {c['high']:>8.2f} "
                  f"{c['low']:>8.2f} {c['close']:>8.2f} "
                  f"{int(c['volume']):>12,}")

    print("\nLive prediction complete.")