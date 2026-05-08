"""
RakeshGold Scanner
Author: Rakesh
Modules: Nifty 500 → EMA Stack → RS Rank (MarketSmith) → VCP → RSI → NR7 → Score → Sheets Push
Runs daily at 16:30 IST via GitHub Actions
"""

import os
import io
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import gspread
from google.oauth2.service_account import Credentials

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SHEET_ID            = "1RqBkeuLoLz5WnUo5SJJ2ttBVmb3Jc-9tow7F2zawISw"
LOOKBACK_DAYS       = 365
EMA_SHORT           = 20
EMA_MID             = 50
EMA_LONG            = 200
NR7_PERIOD          = 7
VOL_CONTRACT_DAYS   = 5
VOL_MA_DAYS         = 50
NEAR_52W_PCT        = 15
RS_STRONG           = 85    # RS rank ≥ 85 → 2 pts
RS_OK               = 70    # RS rank ≥ 70 → 1 pt
RSI_DAILY_WARN      = 80    # Daily RSI overbought threshold
RSI_WEEKLY_WARN     = 75    # Weekly RSI overbought threshold
MIN_SCORE           = 3     # minimum score to include in output


# ─────────────────────────────────────────────
# MODULE 1: FETCH NIFTY 500 SYMBOLS
# ─────────────────────────────────────────────
def get_nifty500_symbols():
    """Fetch current Nifty 500 constituents from NSE"""
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.nseindia.com"
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        df = pd.read_csv(io.StringIO(r.text))
        symbols = df["Symbol"].dropna().tolist()
        print(f"✅ Fetched {len(symbols)} Nifty 500 symbols")
        return symbols
    except Exception as e:
        print(f"⚠️ Could not fetch Nifty 500 list: {e}")
        return [
            "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR",
            "SBIN","BHARTIARTL","KOTAKBANK","LT","AXISBANK","ASIANPAINT",
            "MARUTI","TITAN","SUNPHARMA","BAJFINANCE","WIPRO","ULTRACEMCO",
            "NESTLEIND","POWERGRID","NTPC","TATAMOTORS","BAJAJFINSV","TECHM",
            "HCLTECH","DIVISLAB","GRASIM","CIPLA","DRREDDY","BRITANNIA",
            "EICHERMOT","HEROMOTOCO","BPCL","ONGC","COALINDIA","IOC",
            "ADANIENT","ADANIPORTS","TATACONSUM","APOLLOHOSP","DABUR",
            "PIDILITIND","SIEMENS","HAVELLS","VOLTAS","MUTHOOTFIN",
            "CHOLAFIN","PERSISTENT","COFORGE","DIXON"
        ]


# ─────────────────────────────────────────────
# MODULE 2: FETCH HISTORICAL DATA
# ─────────────────────────────────────────────
def fetch_stock_data(symbol, days=LOOKBACK_DAYS):
    """Fetch historical OHLCV from NSE via yfinance"""
    try:
        import yfinance as yf
        ticker = f"{symbol}.NS"
        end   = datetime.today()
        start = end - timedelta(days=days + 50)
        df = yf.download(ticker, start=start, end=end, progress=False,
                         auto_adjust=True, group_by='column')
        if df.empty or len(df) < 50:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower().strip() for c in df.columns]

        col_map = {}
        for col in df.columns:
            if   'open'   in col: col_map[col] = 'open'
            elif 'high'   in col: col_map[col] = 'high'
            elif 'low'    in col: col_map[col] = 'low'
            elif 'close'  in col: col_map[col] = 'close'
            elif 'volume' in col: col_map[col] = 'volume'
        df = df.rename(columns=col_map)

        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in df.columns for c in required):
            return None

        df = df[required].copy()
        df = df[pd.to_numeric(df['close'], errors='coerce').notna()]
        df = df.astype(float)
        df.dropna(inplace=True)

        if len(df) < 50:
            return None

        df = df.reset_index(drop=True)
        return df

    except Exception:
        return None


# ─────────────────────────────────────────────
# MODULE 3: EMA STACK FILTER
# ─────────────────────────────────────────────
def check_ema_stack(df):
    """Returns True if 20 EMA > 50 EMA > 200 EMA and close > all three"""
    if len(df) < EMA_LONG + 10:
        return False, {}

    df = df.copy()
    df['ema20']  = df['close'].ewm(span=EMA_SHORT, adjust=False).mean()
    df['ema50']  = df['close'].ewm(span=EMA_MID,   adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=EMA_LONG,  adjust=False).mean()

    last = df.iloc[-1]
    bull_stack = (
        last['close']  > last['ema20'] and
        last['ema20']  > last['ema50'] and
        last['ema50']  > last['ema200']
    )

    details = {
        'close':     round(last['close'], 2),
        'ema20':     round(last['ema20'], 2),
        'ema50':     round(last['ema50'], 2),
        'ema200':    round(last['ema200'], 2),
        'ema_stack': bull_stack
    }
    return bull_stack, details


# ─────────────────────────────────────────────
# MODULE 4: 52-WEEK HIGH PROXIMITY
# ─────────────────────────────────────────────
def check_52w_proximity(df):
    """Returns True if within NEAR_52W_PCT% of 52-week high"""
    if len(df) < 50:
        return False, 999

    high_52w   = df['high'].tail(252).max()
    last_close = df['close'].iloc[-1]
    pct_from_high = round(((high_52w - last_close) / high_52w) * 100, 2)
    return pct_from_high <= NEAR_52W_PCT, pct_from_high


# ─────────────────────────────────────────────
# MODULE 5: WEIGHTED RETURN  (MarketSmith RS)
# ─────────────────────────────────────────────
def calc_weighted_return(df):
    """
    MarketSmith-style weighted 12-month return.
    Splits 12 months into 4 quarters (~63 trading days each).
    Most-recent quarter (Q4) is weighted 2×; Q1-Q3 each weight 1×.
    weighted = (Q1 + Q2 + Q3 + 2 × Q4) / 5
    Returns percentage used for universe-wide percentile ranking.
    """
    if len(df) < 253:
        return None

    p     = df['close']
    p_now = float(p.iloc[-1])
    p_3m  = float(p.iloc[-64])    # ~3 months ago
    p_6m  = float(p.iloc[-127])   # ~6 months ago
    p_9m  = float(p.iloc[-190])   # ~9 months ago
    p_12m = float(p.iloc[-253])   # ~12 months ago

    if any(v == 0 for v in [p_3m, p_6m, p_9m, p_12m]):
        return None

    q4 = (p_now - p_3m)  / p_3m    # most recent 3 months (2× weight)
    q3 = (p_3m  - p_6m)  / p_6m
    q2 = (p_6m  - p_9m)  / p_9m
    q1 = (p_9m  - p_12m) / p_12m   # oldest 3 months

    weighted = (q1 + q2 + q3 + 2 * q4) / 5
    return round(weighted * 100, 4)


# ─────────────────────────────────────────────
# MODULE 6: RSI CALCULATION
# ─────────────────────────────────────────────
def _calc_rsi(close, period=14):
    """Wilder RSI on a price Series"""
    if len(close) < period + 2:
        return None
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi      = 100 - (100 / (1 + rs))
    last     = rsi.iloc[-1]
    return round(float(last), 1) if pd.notna(last) else None


def calc_rsi_from_df(df):
    """Returns (daily_rsi, weekly_rsi, daily_overbought, weekly_overbought)"""
    daily_rsi = _calc_rsi(df['close'], 14)

    # Approximate weekly closes using every 5th trading day
    weekly_close = df['close'].iloc[::5].reset_index(drop=True)
    weekly_rsi   = _calc_rsi(weekly_close, 14) if len(weekly_close) >= 20 else None

    daily_ob  = daily_rsi  is not None and daily_rsi  > RSI_DAILY_WARN
    weekly_ob = weekly_rsi is not None and weekly_rsi > RSI_WEEKLY_WARN

    return daily_rsi, weekly_rsi, daily_ob, weekly_ob


def _rsi_alert(daily_ob, weekly_ob):
    if daily_ob and weekly_ob: return "⚠️ D+W OB"
    if daily_ob:               return "⚠️ D RSI>80"
    if weekly_ob:              return "⚠️ W RSI>75"
    return ""


# ─────────────────────────────────────────────
# MODULE 7: VOLUME CONTRACTION (VCP Proxy)
# ─────────────────────────────────────────────
def check_volume_contraction(df):
    """True if recent 5-day avg volume < 60% of 50-day avg volume"""
    if len(df) < VOL_MA_DAYS + VOL_CONTRACT_DAYS:
        return False, {}

    vol_baseline = df['volume'].iloc[-(VOL_MA_DAYS + VOL_CONTRACT_DAYS):-VOL_CONTRACT_DAYS].mean()
    vol_recent   = df['volume'].tail(VOL_CONTRACT_DAYS).mean()

    if vol_baseline == 0:
        return False, {}

    ratio = round(vol_recent / vol_baseline, 2)
    contracted = ratio < 0.60
    return contracted, {'vol_ratio': ratio, 'vol_contracted': contracted}


# ─────────────────────────────────────────────
# MODULE 8: ATR CONTRACTION (Price VCP)
# ─────────────────────────────────────────────
def check_atr_contraction(df):
    """True if 5-day ATR < 70% of 20-day ATR"""
    if len(df) < 30:
        return False, {}

    df = df.copy()
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low']  - df['close'].shift(1))
        )
    )

    atr5  = df['tr'].tail(5).mean()
    atr20 = df['tr'].tail(20).mean()

    if atr20 == 0:
        return False, {}

    ratio = round(atr5 / atr20, 2)
    contracted = ratio < 0.70
    return contracted, {'atr_ratio': ratio, 'atr_contracted': contracted}


# ─────────────────────────────────────────────
# MODULE 9: NR7 TRIGGER
# ─────────────────────────────────────────────
def check_nr7(df):
    """NR7: today's range is smallest of last 7 days; optionally an Inside Bar"""
    if len(df) < NR7_PERIOD + 2:
        return False, False, {}

    df = df.copy()
    df['range'] = df['high'] - df['low']

    last_7_ranges = df['range'].tail(NR7_PERIOD).values
    today_range   = last_7_ranges[-1]

    is_nr7 = today_range == min(last_7_ranges)

    today    = df.iloc[-1]
    prev     = df.iloc[-2]
    is_inside    = (today['high'] < prev['high']) and (today['low'] > prev['low'])
    nr7_plus_ib  = is_nr7 and is_inside

    details = {
        'nr7':           is_nr7,
        'inside_bar':    is_inside,
        'nr7_ib':        nr7_plus_ib,
        'today_range':   round(today_range, 2),
        'nr7_high':      round(today['high'], 2),
        'nr7_low':       round(today['low'], 2),
        'entry_trigger': round(today['high'] * 1.001, 2),
        'stop_loss':     round(today['low']  * 0.999, 2),
    }

    return is_nr7, nr7_plus_ib, details


# ─────────────────────────────────────────────
# MODULE 10: SCORING ENGINE  (max 9 pts)
# ─────────────────────────────────────────────
def compute_score(ema_ok, near_52w, vol_ok, atr_ok, nr7, nr7_ib, rs_rank):
    """
    Score 0-9:
    EMA Stack       → 2 pts
    Near 52W High   → 1 pt
    Volume Contract → 1 pt
    ATR Contract    → 1 pt
    NR7             → 1 pt
    NR7 + IB        → bonus 1 pt
    RS ≥ 70         → 1 pt   (MarketSmith percentile rank)
    RS ≥ 85         → 2 pts  (replaces the 1 pt tier above)
    """
    score = 0
    if ema_ok:   score += 2
    if near_52w: score += 1
    if vol_ok:   score += 1
    if atr_ok:   score += 1
    if nr7:      score += 1
    if nr7_ib:   score += 1

    if rs_rank is not None:
        if rs_rank >= RS_STRONG:
            score += 2
        elif rs_rank >= RS_OK:
            score += 1

    return score


# ─────────────────────────────────────────────
# MODULE 11: GOOGLE SHEETS PUSH
# ─────────────────────────────────────────────
def push_to_sheets(results_df, scan_date):
    """Push results to Google Sheets with formatted headers"""
    try:
        creds_json = os.environ.get("GOOGLE_CREDENTIALS")
        if not creds_json:
            print("⚠️ GOOGLE_CREDENTIALS env var not set")
            return False

        creds_dict = json.loads(creds_json)
        scopes = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc    = gspread.authorize(creds)
        sh    = gc.open_by_key(SHEET_ID)

        # ── Main results sheet ──
        try:
            ws = sh.worksheet("EOD_SCAN")
        except Exception:
            ws = sh.add_worksheet(title="EOD_SCAN", rows=600, cols=30)

        ws.clear()

        headers = [
            "Scan Date", "Symbol", "Close", "EMA20", "EMA50", "EMA200",
            "EMA Stack ✅", "52W High %", "Near 52W ✅",
            "Vol Ratio", "Vol Contract ✅", "ATR Ratio", "ATR Contract ✅",
            "NR7 ✅", "Inside Bar", "NR7+IB 🔥",
            "RS Rank", "Daily RSI", "Weekly RSI", "RSI Alert",
            "Entry Trigger", "Stop Loss", "Risk %", "SCORE /9", "SIGNAL"
        ]

        rows = [headers]
        for _, r in results_df.iterrows():
            risk_pct = round(
                ((r['entry_trigger'] - r['stop_loss']) / r['entry_trigger']) * 100, 2
            ) if r['entry_trigger'] > 0 else ""

            score = r['score']
            if score >= 8:          signal = "🔥 PRIME SETUP"
            elif 6 <= score <= 7:   signal = "⭐ STRONG"
            elif 4 <= score <= 5:   signal = "👀 WATCHLIST"
            else:                   signal = ""

            rows.append([
                scan_date,
                r['symbol'],
                r['close'],
                r['ema20'],
                r['ema50'],
                r['ema200'],
                "✅" if r['ema_stack']     else "❌",
                r['pct_from_52w'],
                "✅" if r['near_52w']      else "❌",
                r['vol_ratio'],
                "✅" if r['vol_contracted'] else "❌",
                r['atr_ratio'],
                "✅" if r['atr_contracted'] else "❌",
                "✅" if r['nr7']            else "❌",
                "✅" if r['inside_bar']     else "❌",
                "🔥" if r['nr7_ib'] else ("✅" if r['nr7'] else "❌"),
                r['rs_rank'],
                r['daily_rsi']  if r['daily_rsi']  is not None else "",
                r['weekly_rsi'] if r['weekly_rsi'] is not None else "",
                r['rsi_alert'],
                r['entry_trigger'],
                r['stop_loss'],
                risk_pct,
                score,
                signal
            ])

        ws.update(rows, value_input_option="USER_ENTERED")

        ws.format("A1:Y1", {
            "backgroundColor": {"red": 0.1, "green": 0.1, "blue": 0.1},
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 0.84, "blue": 0}},
            "horizontalAlignment": "CENTER"
        })

        # ── Summary sheet ──
        try:
            ws2 = sh.worksheet("SUMMARY")
        except Exception:
            ws2 = sh.add_worksheet(title="SUMMARY", rows=100, cols=10)

        ws2.clear()
        prime   = results_df[results_df['score'] >= 8]
        strong  = results_df[(results_df['score'] >= 6) & (results_df['score'] <= 7)]
        watch   = results_df[(results_df['score'] >= 4) & (results_df['score'] <= 5)]
        nr7_all = results_df[results_df['nr7'] == True]

        summary = [
            ["RAKESHGOLD SCANNER", scan_date],
            [""],
            ["Total Stocks Scanned",      len(results_df)],
            ["🔥 PRIME SETUPS (8-9)",     len(prime)],
            ["⭐ STRONG (Score 6-7)",      len(strong)],
            ["👀 WATCHLIST (Score 4-5)",  len(watch)],
            ["NR7 Triggers Today",         len(nr7_all)],
            [""],
            ["── PRIME SETUPS ──", ""],
        ]

        for _, r in prime.iterrows():
            summary.append([
                r['symbol'],
                f"Score {r['score']}/9 | RS:{r['rs_rank']} | Entry {r['entry_trigger']} | SL {r['stop_loss']}"
            ])

        summary.append([""])
        summary.append(["── NR7 TRIGGERS ──", ""])
        for _, r in nr7_all.iterrows():
            summary.append([
                r['symbol'],
                f"NR7+IB:{r['nr7_ib']} | RS:{r['rs_rank']} | Entry {r['entry_trigger']} | SL {r['stop_loss']}"
            ])

        ws2.update(summary, value_input_option="USER_ENTERED")
        ws2.format("A1:B1", {"textFormat": {"bold": True, "fontSize": 14}})

        print(f"✅ Pushed {len(results_df)} stocks to Google Sheets")
        print(f"   🔥 Prime: {len(prime)} | ⭐ Strong: {len(strong)} | NR7: {len(nr7_all)}")
        return True

    except Exception as e:
        print(f"❌ Sheets push failed: {e}")
        return False


# ─────────────────────────────────────────────
# MAIN RUNNER  (two-pass: rank first, filter second)
# ─────────────────────────────────────────────
def run_scanner():
    scan_date = datetime.now().strftime("%d-%b-%Y")
    print(f"\n{'='*60}")
    print(f"  RAKESHGOLD SCANNER — {scan_date}")
    print(f"{'='*60}\n")

    symbols = get_nifty500_symbols()

    # ── PASS 1: Download universe + compute weighted returns for RS ranking ──
    print("[Pass 1] Downloading universe data and computing RS returns...")
    stock_data  = {}   # symbol → df
    raw_returns = {}   # symbol → weighted 12m return

    for i, symbol in enumerate(symbols, 1):
        df = fetch_stock_data(symbol)
        if df is None:
            continue
        stock_data[symbol] = df
        wr = calc_weighted_return(df)
        if wr is not None:
            raw_returns[symbol] = wr
        if i % 100 == 0:
            print(f"  ... {i}/{len(symbols)} fetched")

    print(f"  Fetched {len(stock_data)} stocks | RS-eligible: {len(raw_returns)}")

    # Percentile rank 1-99 across full Nifty 500 universe
    rs_ranks = {}
    if raw_returns:
        s = pd.Series(raw_returns)
        ranked = (s.rank(pct=True) * 99).round().clip(1, 99).astype(int)
        rs_ranks = ranked.to_dict()
        print(f"  RS universe ranked: median={int(s.median()):.0f}%, "
              f"top10={sum(v >= 90 for v in rs_ranks.values())} stocks ≥90")

    # ── PASS 2: Apply filters and compute scores ──
    print("\n[Pass 2] Applying filters and computing scores...")
    results = []

    for i, (symbol, df) in enumerate(stock_data.items(), 1):
        try:
            if len(df) < 210:
                continue

            ema_ok,  ema_data      = check_ema_stack(df)
            near_52w, pct_52w      = check_52w_proximity(df)
            vol_ok,  vol_data      = check_volume_contraction(df)
            atr_ok,  atr_data      = check_atr_contraction(df)
            nr7, nr7_ib, nr7_data  = check_nr7(df)
            rs_rank                = rs_ranks.get(symbol)
            daily_rsi, weekly_rsi, daily_ob, weekly_ob = calc_rsi_from_df(df)
            score = compute_score(ema_ok, near_52w, vol_ok, atr_ok, nr7, nr7_ib, rs_rank)

            if score < MIN_SCORE:
                continue

            results.append({
                'symbol':         symbol,
                'close':          ema_data.get('close', 0),
                'ema20':          ema_data.get('ema20', 0),
                'ema50':          ema_data.get('ema50', 0),
                'ema200':         ema_data.get('ema200', 0),
                'ema_stack':      ema_ok,
                'pct_from_52w':   pct_52w,
                'near_52w':       near_52w,
                'vol_ratio':      vol_data.get('vol_ratio', 0),
                'vol_contracted': vol_ok,
                'atr_ratio':      atr_data.get('atr_ratio', 0),
                'atr_contracted': atr_ok,
                'nr7':            nr7,
                'inside_bar':     nr7_data.get('inside_bar', False),
                'nr7_ib':         nr7_ib,
                'rs_rank':        rs_rank if rs_rank is not None else 0,
                'daily_rsi':      daily_rsi,
                'weekly_rsi':     weekly_rsi,
                'rsi_alert':      _rsi_alert(daily_ob, weekly_ob),
                'entry_trigger':  nr7_data.get('entry_trigger', 0),
                'stop_loss':      nr7_data.get('stop_loss', 0),
                'score':          score
            })

            rs_str = f"RS:{rs_rank:2d}" if rs_rank else "RS:--"
            status = "🔥" if score >= 8 else ("⭐" if score >= 6 else "·")
            print(f"  {status} [{i:3d}] {symbol:<15} Score:{score}/9 | {rs_str} | NR7:{nr7} | EMA:{ema_ok}")

        except Exception as e:
            print(f"  ⚠️  {symbol}: {e}")
            continue

    if not results:
        print("No qualifying stocks found today.")
        return

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['score', 'rs_rank'], ascending=[False, False])

    print(f"\n{'='*60}")
    print(f"  SCAN COMPLETE — {len(results_df)} qualifying stocks")
    print(f"{'='*60}\n")

    push_to_sheets(results_df, scan_date)

    os.makedirs("outputs", exist_ok=True)
    csv_path = f"outputs/scan_{datetime.now().strftime('%Y%m%d')}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"✅ CSV saved: {csv_path}")


if __name__ == "__main__":
    run_scanner()
