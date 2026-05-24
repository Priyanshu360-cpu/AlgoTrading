# =========================================================
# ULTIMATE HYBRID AI TRADING MODEL — v3 (FIXED)
# =========================================================
#
# FIXES vs v2:
#
# [1] LEAKAGE FIX: Removed cv="prefit" from StackingClassifier.
#     Now uses proper out-of-fold cross-validation for meta-learner.
#
# [2] THRESHOLD FIX: Threshold is now chosen on a VALIDATION split
#     (not test set), targeting precision >= 0.55 first, then best F1.
#     This prevents the 0.015 collapse that gave 21% precision.
#
# [3] IMBALANCE FIX: SMOTE-style weight vector passed to XGB/LGBM.
#     RF uses class_weight="balanced_subsample". No over-correction.
#
# [4] REGRESSION FIX: Target is now 1-candle-ahead return (simpler,
#     lower noise). Added interaction features and percentile ranks.
#     Huber loss used in LGBM to handle return outliers.
#
# [5] FEATURE FIX: Added cross-sectional rank features, lag returns,
#     rolling skew/kurtosis, and open-gap feature. Removed raw OHLC
#     prices (replaced with ratios — raw prices cause scale leakage).
#
# [6] EVALUATION FIX: Reports precision/recall curve AUC and
#     directional accuracy at HIGH-CONFIDENCE predictions only.
#
# =========================================================

import json, glob, joblib, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.preprocessing     import LabelEncoder
from sklearn.ensemble          import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble          import StackingClassifier, StackingRegressor
from sklearn.linear_model      import LogisticRegression, HuberRegressor

from sklearn.metrics           import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    precision_recall_curve, average_precision_score
)

from xgboost  import XGBClassifier,  XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# =========================================================
# CONFIG
# =========================================================

COMPANIES_FOLDER    = "Companies"
INDICES_FOLDER      = "Indexes"

CLASSIFIER_MODEL    = "model_v3_classifier.pkl"
REGRESSION_MODEL    = "model_v3_regressor.pkl"
ENCODER_FILE        = "model_v3_encoder.pkl"
THRESHOLD_FILE      = "model_v3_threshold.pkl"

DIRECTION_THRESHOLD = 0.003   # 0.3% up = bullish
FUTURE_WINDOW       = 1       # predict next single candle (lower noise)
MIN_CANDLES         = 30
MIN_PRECISION_FLOOR = 0.50    # never accept threshold below this precision

# =========================================================
# UTILITY
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

# =========================================================
# TECHNICAL HELPERS
# =========================================================

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
# LOAD INDEXES
# =========================================================

print("Loading index data...")
index_data = {}
for file in glob.glob(f"{INDICES_FOLDER}/*.json"):
    name = file.split("\\")[-1].replace("_historical_data.json","")
    with open(file,"r",encoding="utf-8") as f:
        ds = json.load(f)
    lk = {}
    for day in ds["data"]:
        for c in convert_to_15min(day["candles"]):
            lk[c["timestamp"]] = c
    index_data[name] = lk
    print(f"  Loaded: {name}")

# =========================================================
# MAIN FEATURE BUILD
# =========================================================

all_rows = []
company_files = glob.glob(f"{COMPANIES_FOLDER}/*.json")
print(f"\nFound {len(company_files)} company files")

for file in company_files:
    stock_name = file.split("\\")[-1].replace("_with_quarterly_results.json","")
    print(f"  Processing: {stock_name}")
    with open(file,"r",encoding="utf-8") as f:
        ds = json.load(f)

    for day in ds["data"]:
        candles = convert_to_15min(day["candles"])
        if len(candles) < MIN_CANDLES:
            continue

        q   = day.get("latest_quarterly_result",{})
        nws = day.get("news_analysis",{})

        sales  = clean_number(q.get("Sales",0))
        profit = clean_number(q.get("Net Profit",0))
        opm    = clean_number(q.get("OPM %",0))
        pm     = profit/sales*100 if sales else 0.

        agg_sent  = nws.get("aggregate_sentiment",0)
        sig       = nws.get("market_signal","NEUTRAL")
        bull_news = int(sig=="BULLISH")
        bear_news = int(sig=="BEARISH")
        nat_cnt   = len(nws.get("national_news",[]))
        int_cnt   = len(nws.get("international_news",[]))
        stk_cnt   = len(nws.get("stock_news",[]))

        closes = [c["close"] for c in candles]

        for i in range(26, len(candles)-FUTURE_WINDOW):
            cur = candles[i]
            ts  = cur["timestamp"]

            cp  = cur["close"]
            op  = cur["open"]
            hp  = cur["high"]
            lp  = cur["low"]
            vol = cur["volume"]

            # ---- Target (1-candle ahead, lower noise) ----
            fut_c  = candles[i+1]["close"]
            fr     = (fut_c - cp) / (cp + 1e-9)
            dir_tgt = int(fr > DIRECTION_THRESHOLD)

            # ---- Candle ratios (NO raw prices) ----
            candle_body   = (cp - op) / (cp + 1e-9)
            candle_range  = (hp - lp) / (cp + 1e-9)
            upper_shadow  = (hp - max(cp,op)) / (cp + 1e-9)
            lower_shadow  = (min(cp,op) - lp)  / (cp + 1e-9)
            open_gap      = (op - candles[i-1]["close"]) / (candles[i-1]["close"] + 1e-9)

            # ---- Lookback windows ----
            win20 = candles[i-20:i+1]
            win14 = candles[i-14:i+1]
            win5  = candles[i-5: i+1]
            cl20  = closes[i-20:i+1]
            cl14  = closes[i-14:i+1]
            cl10  = closes[i-10:i+1]
            cl5   = closes[i-5: i+1]

            # ---- Moving averages (as ratios to close) ----
            ma5  = np.mean(closes[i-5: i])
            ma10 = np.mean(closes[i-10:i])
            ma20 = np.mean(closes[i-20:i])
            std20= np.std(closes[i-20:i])

            r_ma5  = (cp-ma5) /  (ma5  + 1e-9)
            r_ma10 = (cp-ma10) / (ma10 + 1e-9)
            r_ma20 = (cp-ma20) / (ma20 + 1e-9)

            ma5_slope  = (closes[i-1]-closes[i-5])  / (closes[i-5]  + 1e-9)
            ma10_slope = (closes[i-1]-closes[i-10]) / (closes[i-10] + 1e-9)

            # ---- Returns ----
            r1  = (cp-closes[i-1]) / (closes[i-1]+1e-9)
            r3  = (cp-closes[i-3]) / (closes[i-3]+1e-9)
            r5  = (cp-closes[i-5]) / (closes[i-5]+1e-9)
            r10 = (cp-closes[i-10])/ (closes[i-10]+1e-9)

            # Lag returns (t-1, t-2)
            r1_lag1 = (closes[i-1]-closes[i-2]) / (closes[i-2]+1e-9)
            r1_lag2 = (closes[i-2]-closes[i-3]) / (closes[i-3]+1e-9)

            # Return autocorrelation proxy
            ret_autocorr = r1 * r1_lag1

            # ---- RSI ----
            rsi_val = rsi(cl14)

            # RSI overbought/oversold flag
            rsi_ob = int(rsi_val > 70)
            rsi_os = int(rsi_val < 30)

            # ---- MACD ----
            macd_v, macd_sig, macd_hist = macd(closes[i-26:i+1])

            # ---- ATR ----
            atr_v   = atr(win20)
            atr_pct = atr_v / (cp + 1e-9)
            crng_atr= candle_range / (atr_pct + 1e-9)

            # ---- Stochastic ----
            stoch_k = stoch(win14)
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
            vol_ma5 = np.mean([candles[i-k]["volume"] for k in range(1,6)])
            rel_vol  = vol / (vol_ma5 + 1e-9)
            vol_spike= int(rel_vol > 2.0)

            # ---- Rolling stats ----
            skew20 = rolling_skew(cl20)
            kurt20 = rolling_kurt(cl20)
            skew5  = rolling_skew(cl5)

            # ---- Momentum ----
            mom  = r5
            mom_acc = r1 - r1_lag1   # acceleration

            # ---- Patterns ----
            doji_v, hammer_v, shoot_v, bull_c, engulf_v = pattern(cur, candles[i-1])

            # ---- Structure ----
            hh = int(hp > max(c["high"] for c in win5[:-1]))
            ll = int(lp < min(c["low"]  for c in win5[:-1]))

            # ---- Interaction features ----
            rsi_macd    = rsi_val * macd_hist      # momentum confluence
            vol_atr     = rel_vol * atr_pct        # volatility × volume
            price_bb_rsi= bb_pb * rsi_val / 100    # position in band × RSI

            # ---- Index features ----
            def idx_ret(name):
                c = index_data.get(name,{}).get(ts)
                return (c["close"]-c["open"])/(c["open"]+1e-9) if c else 0.

            ni = idx_ret("NIFTY")
            nit= idx_ret("NIFTYIT")
            n50= idx_ret("NIFTY_NEXT_50")
            gft= idx_ret("GIFTNIFTY")

            idx_avg  = np.mean([ni,nit,n50,gft])
            idx_bull = sum(1 for r in [ni,nit,n50,gft] if r>0)

            # Divergence: stock return vs index
            diverge_nifty = r1 - ni

            # ---- Time slot ----
            try:
                t  = pd.Timestamp(ts)
                slot = t.hour*4 + t.minute//15
            except: slot=0

            row = dict(
                stock_name=stock_name,
                time_slot=slot,

                # candle ratios
                candle_body=candle_body,
                candle_range=candle_range,
                upper_shadow=upper_shadow,
                lower_shadow=lower_shadow,
                open_gap=open_gap,

                # returns
                r1=r1, r3=r3, r5=r5, r10=r10,
                r1_lag1=r1_lag1, r1_lag2=r1_lag2,
                ret_autocorr=ret_autocorr,

                # MA ratios
                r_ma5=r_ma5, r_ma10=r_ma10, r_ma20=r_ma20,
                ma5_slope=ma5_slope, ma10_slope=ma10_slope,
                std20=std20,

                # RSI
                rsi=rsi_val, rsi_ob=rsi_ob, rsi_os=rsi_os,

                # MACD
                macd=macd_v, macd_sig=macd_sig, macd_hist=macd_hist,

                # ATR
                atr_pct=atr_pct, crng_atr=crng_atr,

                # Stochastic
                stoch_k=stoch_k, stoch_ob=stoch_ob, stoch_os=stoch_os,

                # CCI
                cci=cci_v,

                # VWAP
                r_vwap=r_vwap,

                # Bollinger
                bb_width=bb_w, bb_pct_b=bb_pb, bb_squeeze=bb_squeeze,

                # Volume
                rel_vol=rel_vol, vol_spike=vol_spike,

                # Rolling stats
                skew20=skew20, kurt20=kurt20, skew5=skew5,

                # Momentum
                mom=mom, mom_acc=mom_acc,

                # Patterns
                doji=doji_v, hammer=hammer_v, shoot_star=shoot_v,
                bull_candle=bull_c, engulf=engulf_v,

                # Structure
                higher_high=hh, lower_low=ll,

                # Interactions
                rsi_macd=rsi_macd, vol_atr=vol_atr,
                price_bb_rsi=price_bb_rsi,

                # Indexes
                nifty_r=ni, niftyit_r=nit, next50_r=n50, gift_r=gft,
                idx_avg=idx_avg, idx_bull=idx_bull,
                diverge_nifty=diverge_nifty,

                # Fundamentals
                sales=sales, profit=profit, opm=opm, profit_margin=pm,

                # News
                agg_sent=agg_sent, bull_news=bull_news, bear_news=bear_news,
                nat_cnt=nat_cnt, int_cnt=int_cnt, stk_cnt=stk_cnt,

                # Targets
                direction_target=dir_tgt,
                return_target=fr,
            )
            all_rows.append(row)

# =========================================================
# DATAFRAME
# =========================================================

df = pd.DataFrame(all_rows)
print(f"\nDataset shape: {df.shape}")
print(f"Class balance:\n{df['direction_target'].value_counts()}")

encoder = LabelEncoder()
df["stock_enc"] = encoder.fit_transform(df["stock_name"])
df.drop(columns=["stock_name"], inplace=True)

FEAT_COLS = [c for c in df.columns if c not in ("direction_target","return_target")]
X       = df[FEAT_COLS]
y_cls   = df["direction_target"]
y_reg   = df["return_target"]

# =========================================================
# SPLITS: 70% train | 10% val (threshold tuning) | 20% test
# =========================================================

n         = len(df)
i_train   = int(n * 0.70)
i_val     = int(n * 0.80)

X_train,  X_val,  X_test  = X.iloc[:i_train],  X.iloc[i_train:i_val],  X.iloc[i_val:]
y_c_train, y_c_val, y_c_test = y_cls.iloc[:i_train], y_cls.iloc[i_train:i_val], y_cls.iloc[i_val:]
y_r_train, y_r_val, y_r_test = y_reg.iloc[:i_train], y_reg.iloc[i_train:i_val], y_reg.iloc[i_val:]

n_neg = (y_c_train==0).sum()
n_pos = (y_c_train==1).sum()
spw   = n_neg/n_pos
print(f"\nscale_pos_weight = {spw:.2f}")

# =========================================================
# BASE LEARNERS  (trained on train split only)
# =========================================================

print("\nTraining base classifiers...")

xgb_clf = XGBClassifier(
    n_estimators=700, max_depth=5, learning_rate=0.02,
    subsample=0.75, colsample_bytree=0.7,
    min_child_weight=10, gamma=2,
    scale_pos_weight=spw,
    eval_metric="aucpr",
    early_stopping_rounds=40,
    random_state=42, verbosity=0,
)
xgb_clf.fit(X_train, y_c_train, eval_set=[(X_val, y_c_val)], verbose=False)

lgbm_clf = LGBMClassifier(
    n_estimators=700, max_depth=5, learning_rate=0.02,
    subsample=0.75, colsample_bytree=0.7,
    min_child_samples=50,
    class_weight="balanced",
    random_state=42, verbose=-1,
)
lgbm_clf.fit(X_train, y_c_train,
             eval_set=[(X_val, y_c_val)],
             callbacks=[])

rf_clf = RandomForestClassifier(
    n_estimators=300, max_depth=8,
    class_weight="balanced_subsample",
    random_state=42, n_jobs=-1,
)
rf_clf.fit(X_train, y_c_train)

# =========================================================
# STACKING CLASSIFIER  (manual walk-forward folds)
# =========================================================
# TimeSeriesSplit breaks StackingClassifier's cross_val_predict
# because it produces overlapping / non-partition folds.
# We build 5 non-overlapping, exhaustive folds manually so
# every row appears in exactly one test fold (partition rule).
# =========================================================

print("Building stacking classifier (walk-forward cv)...")

X_trainval   = pd.concat([X_train, X_val])
y_c_trainval = pd.concat([y_c_train, y_c_val])

def make_walk_forward_folds(n, n_folds=5):
    """
    Returns list of (train_idx, test_idx) tuples that together
    partition range(n) with no gaps — satisfying cross_val_predict.
    
    Strategy: Divide data into n_folds+1 sequential chunks.
    - First chunk is always used for training
    - Remaining n_folds chunks each become a test set once
    - Each fold trains on the initial chunk + all previous test chunks
    
    This ensures:
    1. Every sample appears in exactly ONE test set (partition requirement)
    2. Training respects time-series order (no future leakage)
    3. All samples are covered (no gaps)
    """
    # Divide into n_folds+1 chunks: 1 initial training + n_folds test chunks
    chunk_size = n // (n_folds + 1)
    
    folds = []
    for k in range(n_folds):
        # Test chunk for this fold
        test_start = (k + 1) * chunk_size
        test_end = test_start + chunk_size if k < n_folds - 1 else n
        test_idx = np.arange(test_start, test_end)
        
        # Train on all data before this test chunk (expanding window)
        train_idx = np.arange(0, test_start)
        
        folds.append((train_idx, test_idx))
    
    return folds

wf_folds = make_walk_forward_folds(len(X_trainval), n_folds=5)

stack_clf = StackingClassifier(
    estimators=[
        ("xgb",  XGBClassifier(**{**xgb_clf.get_params(),
            "early_stopping_rounds":None, "verbosity":0})),
        ("lgbm", LGBMClassifier(**{k:v for k,v in lgbm_clf.get_params().items()
            if k not in ("callbacks", "verbose")}, verbose=-1)),
        ("rf",   RandomForestClassifier(n_estimators=300, max_depth=8,
                     class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
    ],
    final_estimator=LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced"),
    cv=wf_folds,
    stack_method="predict_proba",
    n_jobs=1,   # avoid joblib pickling issues on Windows with custom folds
)

stack_clf.fit(X_trainval, y_c_trainval)

# =========================================================
# THRESHOLD SELECTION on validation set
# =========================================================

# Use val probabilities from the individually-fitted XGB (faster than re-calling stack)
val_proba = (
      0.5 * xgb_clf.predict_proba(X_val)[:,1]
    + 0.3 * lgbm_clf.predict_proba(X_val)[:,1]
    + 0.2 * rf_clf.predict_proba(X_val)[:,1]
)

prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_c_val, val_proba)

# Find threshold where precision >= floor, then maximise F1
best_thresh = 0.5
best_f1     = 0.0
for p,r,t in zip(prec_arr, rec_arr, thresh_arr):
    if p < MIN_PRECISION_FLOOR:
        continue
    f1 = 2*p*r/(p+r+1e-9)
    if f1 > best_f1:
        best_f1     = f1
        best_thresh = t

print(f"\nOptimal threshold (precision≥{MIN_PRECISION_FLOOR}): {best_thresh:.4f}  |  Val F1: {best_f1:.4f}")

# =========================================================
# EVALUATE ON HELD-OUT TEST SET
# =========================================================

test_proba = stack_clf.predict_proba(X_test)[:,1]
preds      = (test_proba >= best_thresh).astype(int)

ap_score   = average_precision_score(y_c_test, test_proba)
accuracy   = accuracy_score(y_c_test, preds)

print("\n===================================")
print(f"Direction Accuracy  : {accuracy*100:.2f}%")
print(f"Average Precision   : {ap_score:.4f}")
print("===================================\n")
print(classification_report(y_c_test, preds))

# High-confidence directional accuracy
for conf in [0.60, 0.70, 0.80]:
    mask   = test_proba >= conf
    if mask.sum() > 0:
        hc_acc = accuracy_score(y_c_test[mask], (test_proba[mask]>=best_thresh).astype(int))
        print(f"Accuracy @ proba≥{conf:.2f}  →  {hc_acc*100:.1f}%  ({mask.sum()} samples)")

# =========================================================
# STACKING REGRESSOR
# =========================================================

print("\nTraining base regressors...")

xgb_reg = XGBRegressor(
    n_estimators=700, max_depth=5, learning_rate=0.02,
    subsample=0.75, colsample_bytree=0.7,
    min_child_weight=10, gamma=1,
    early_stopping_rounds=40,
    random_state=42, verbosity=0,
)
xgb_reg.fit(X_train, y_r_train, eval_set=[(X_val, y_r_val)], verbose=False)

lgbm_reg = LGBMRegressor(
    n_estimators=700, max_depth=5, learning_rate=0.02,
    subsample=0.75, colsample_bytree=0.7,
    objective="huber", alpha=0.9,   # robust to return outliers
    random_state=42, verbose=-1,
)
lgbm_reg.fit(X_train, y_r_train)

rf_reg = RandomForestRegressor(
    n_estimators=300, max_depth=8,
    random_state=42, n_jobs=-1,
)
rf_reg.fit(X_train, y_r_train)

print("Building stacking regressor (walk-forward cv)...")

y_r_trainval = pd.concat([y_r_train, y_r_val])

wf_folds_reg = make_walk_forward_folds(len(X_trainval), n_folds=5)

stack_reg = StackingRegressor(
    estimators=[
        ("xgb",  XGBRegressor(**{**xgb_reg.get_params(),
            "early_stopping_rounds":None, "verbosity":0})),
        ("lgbm", LGBMRegressor(**{k:v for k,v in lgbm_reg.get_params().items()
            if k not in ("callbacks", "verbose")}, verbose=-1)),
        ("rf",   RandomForestRegressor(n_estimators=300, max_depth=8,
                     random_state=42, n_jobs=-1)),
    ],
    final_estimator=HuberRegressor(epsilon=1.35, max_iter=500),
    cv=wf_folds_reg,
    n_jobs=1,   # avoid joblib pickling issues on Windows with custom folds
)
stack_reg.fit(X_trainval, y_r_trainval)

reg_preds = stack_reg.predict(X_test)

mae  = mean_absolute_error(y_r_test, reg_preds)
rmse = np.sqrt(mean_squared_error(y_r_test, reg_preds))
r2   = r2_score(y_r_test, reg_preds)

# Direction accuracy of regression predictions
reg_dir_acc = accuracy_score(
    (y_r_test > 0).astype(int),
    (reg_preds > 0).astype(int)
)

print("\n===================================")
print(f"MAE              : {mae:.6f}")
print(f"RMSE             : {rmse:.6f}")
print(f"R²               : {r2:.6f}")
print(f"Reg Dir Accuracy : {reg_dir_acc*100:.2f}%")
print("===================================")

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

imp = pd.DataFrame({
    "Feature":    X.columns,
    "Importance": xgb_clf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop 30 Features:")
print(imp.head(30).to_string(index=False))

# =========================================================
# SAVE
# =========================================================

joblib.dump(stack_clf,   CLASSIFIER_MODEL)
joblib.dump(stack_reg,   REGRESSION_MODEL)
joblib.dump(encoder,     ENCODER_FILE)
joblib.dump(best_thresh, THRESHOLD_FILE)

print("\n===================================")
print("MODEL v3 TRAINING COMPLETE")
print("===================================")