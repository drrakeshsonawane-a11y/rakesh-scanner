"""
Divitiae EOD Scanner
Author: Rakesh
Modules: Bhavcopy Download → EMA Stack → RS Rank → VCP → NR7 → Score → Sheets Push
Runs daily at 16:30 IST via GitHub Actions
"""

import os
import io
import json
import zipfile
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import gspread
from google.oauth2.service_account import Credentials

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SHEET_ID = "1RqBkeuLoLz5WnUo5SJJ2ttBVmb3Jc-9tow7F2zawISw"
LOOKBACK_DAYS = 365          # days of history to fetch per stock
EMA_SHORT     = 20
EMA_MID       = 50
EMA_LONG      = 200
NR7_PERIOD    = 7
VOL_CONTRACT_DAYS  = 5       # recent days for volume contraction check
VOL_MA_DAYS        = 50      # baseline volume MA
PRICE_CONTRACT_DAYS = 10     # ATR contraction window
NEAR_52W_PCT   = 15          # within X% of 52-week high
RS_TOP_PCT     = 70          # keep stocks in top X percentile of RS


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
        # fallback: top 50 liquid stocks
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
    """Fetch historical OHLCV from NSE via yfinance fallback"""
    try:
        import yfinance as yf
        ticker = f"{symbol}.NS"
        end = datetime.today()
        start = end - timedelta(days=days + 50)
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty or len(df) < 50:
            return None
        df = df.rename(columns={"Open":"open","High":"high","Low":"low",
                                  "Close":"close","Volume":"volume"})
        df = df[["open","high","low","close","volume"]].copy()
        df.dropna(inplace=True)
        return df
    except Exception as e:
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
        'close':  round(last['close'], 2),
        'ema20':  round(last['ema20'], 2),
        'ema50':  round(last['ema50'], 2),
        'ema200': round(last['ema200'], 2),
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
    
    high_52w = df['high'].tail(252).max()
    last_close = df['close'].iloc[-1]
    pct_from_high = round(((high_52w - last_close) / high_52w) * 100, 2)
    near = pct_from_high <= NEAR_52W_PCT
    return near, pct_from_high


# ─────────────────────────────────────────────
# MODULE 5: RELATIVE STRENGTH RANK
# ─────────────────────────────────────────────
def calc_rs_score(df, skip_recent=1):
    """
    RS Score = 6M return (skipping last month)
    Returns raw score for ranking across universe
    """
    if len(df) < 130:
        return None
    
    # skip most recent ~21 trading days (1 month), look back 6 months (~126 days)
    end_idx   = len(df) - skip_recent * 21
    start_idx = end_idx - 126
    
    if start_idx < 0:
        return None
    
    price_now  = df['close'].iloc[end_idx - 1]
    price_then = df['close'].iloc[start_idx]
    
    if price_then == 0:
        return None
    
    return round(((price_now - price_then) / price_then) * 100, 2)


# ─────────────────────────────────────────────
# MODULE 6: VOLUME CONTRACTION (VCP Proxy)
# ─────────────────────────────────────────────
def check_volume_contraction(df):
    """
    True if recent 5-day avg volume < 60% of 50-day avg volume
    Signals institutional quiet / accumulation phase
    """
    if len(df) < VOL_MA_DAYS + VOL_CONTRACT_DAYS:
        return False, {}
    
    vol_ma50    = df['volume'].tail(VOL_MA_DAYS).mean()
    vol_recent5 = df['volume'].tail(VOL_CONTRACT_DAYS).mean()
    
    if vol_ma50 == 0:
        return False, {}
    
    ratio = round(vol_recent5 / vol_ma50, 2)
    contracted = ratio < 0.60
    
    return contracted, {'vol_ratio': ratio, 'vol_contracted': contracted}


# ─────────────────────────────────────────────
# MODULE 7: ATR CONTRACTION (Price VCP)
# ─────────────────────────────────────────────
def check_atr_contraction(df):
    """
    True if 5-day ATR < 50% of 20-day ATR
    Confirms price is coiling / range narrowing
    """
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
# MODULE 8: NR7 TRIGGER
# ─────────────────────────────────────────────
def check_nr7(df):
    """
    NR7: Today's (High - Low) range is smallest in last 7 days
    NR7 + Inside Bar = strongest signal
    """
    if len(df) < NR7_PERIOD + 2:
        return False, False, {}
    
    df = df.copy()
    df['range'] = df['high'] - df['low']
    
    last_7_ranges = df['range'].tail(NR7_PERIOD).values
    today_range   = last_7_ranges[-1]
    prior_ranges  = last_7_ranges[:-1]
    
    is_nr7 = today_range == min(last_7_ranges)
    
    # Inside bar: today's high < yesterday's high AND today's low > yesterday's low
    today   = df.iloc[-1]
    prev    = df.iloc[-2]
    is_inside = (today['high'] < prev['high']) and (today['low'] > prev['low'])
    
    # NR7 + Inside Bar = VCP confirmation (Minervini + Crabel combined)
    nr7_plus_ib = is_nr7 and is_inside
    
    details = {
        'nr7': is_nr7,
        'inside_bar': is_inside,
        'nr7_ib': nr7_plus_ib,
        'today_range': round(today_range, 2),
        'nr7_high': round(today['high'], 2),
        'nr7_low':  round(today['low'], 2),
        'entry_trigger': round(today['high'] * 1.001, 2),  # 0.1% above NR7 high
        'stop_loss':     round(today['low']  * 0.999, 2),  # 0.1% below NR7 low
    }
    
    return is_nr7, nr7_plus_ib, details


# ─────────────────────────────────────────────
# MODULE 9: SCORING ENGINE
# ─────────────────────────────────────────────
def compute_score(ema_ok, near_52w, vol_ok, atr_ok, nr7, nr7_ib):
    """
    Score 0-7:
    EMA Stack       → 2 pts (most important trend filter)
    Near 52W High   → 1 pt
    Volume Contract → 1 pt
    ATR Contract    → 1 pt
    NR7             → 1 pt
    NR7 + IB        → bonus 1 pt
    """
    score = 0
    if ema_ok:   score += 2
    if near_52w: score += 1
    if vol_ok:   score += 1
    if atr_ok:   score += 1
    if nr7:      score += 1
    if nr7_ib:   score += 1
    return score


# ─────────────────────────────────────────────
# MODULE 10: GOOGLE SHEETS PUSH
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
        except:
            ws = sh.add_worksheet(title="EOD_SCAN", rows=600, cols=25)
        
        ws.clear()
        
        headers = [
            "Scan Date", "Symbol", "Close", "EMA20", "EMA50", "EMA200",
            "EMA Stack ✅", "52W High %", "Near 52W ✅",
            "Vol Ratio", "Vol Contract ✅", "ATR Ratio", "ATR Contract ✅",
            "NR7 ✅", "Inside Bar", "NR7+IB 🔥", "RS Score",
            "Entry Trigger", "Stop Loss", "Risk %", "SCORE /7", "SIGNAL"
        ]
        
        rows = [headers]
        for _, r in results_df.iterrows():
            risk_pct = round(((r['entry_trigger'] - r['stop_loss']) / r['entry_trigger']) * 100, 2) if r['entry_trigger'] > 0 else ""
            signal = ""
            if r['score'] >= 6:    signal = "🔥 PRIME SETUP"
            elif r['score'] == 5:  signal = "⭐ STRONG"
            elif r['score'] == 4:  signal = "👀 WATCHLIST"
            
            rows.append([
                scan_date,
                r['symbol'],
                r['close'],
                r['ema20'],
                r['ema50'],
                r['ema200'],
                "✅" if r['ema_stack'] else "❌",
                r['pct_from_52w'],
                "✅" if r['near_52w'] else "❌",
                r['vol_ratio'],
                "✅" if r['vol_contracted'] else "❌",
                r['atr_ratio'],
                "✅" if r['atr_contracted'] else "❌",
                "✅" if r['nr7'] else "❌",
                "✅" if r['inside_bar'] else "❌",
                "🔥" if r['nr7_ib'] else ("✅" if r['nr7'] else "❌"),
                r['rs_score'],
                r['entry_trigger'],
                r['stop_loss'],
                risk_pct,
                r['score'],
                signal
            ])
        
        ws.update(rows, value_input_option="USER_ENTERED")
        
        # Format header row
        ws.format("A1:V1", {
            "backgroundColor": {"red": 0.1, "green": 0.1, "blue": 0.1},
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 0.84, "blue": 0}},
            "horizontalAlignment": "CENTER"
        })
        
        # ── Summary sheet ──
        try:
            ws2 = sh.worksheet("SUMMARY")
        except:
            ws2 = sh.add_worksheet(title="SUMMARY", rows=100, cols=10)
        
        ws2.clear()
        prime   = results_df[results_df['score'] >= 6]
        strong  = results_df[results_df['score'] == 5]
        watch   = results_df[results_df['score'] == 4]
        nr7_all = results_df[results_df['nr7'] == True]
        
        summary = [
            ["DIVITIAE EOD SCANNER", scan_date],
            [""],
            ["Total Stocks Scanned",  len(results_df)],
            ["🔥 PRIME SETUPS (6-7)", len(prime)],
            ["⭐ STRONG (Score 5)",   len(strong)],
            ["👀 WATCHLIST (Score 4)",len(watch)],
            ["NR7 Triggers Today",    len(nr7_all)],
            [""],
            ["── PRIME SETUPS ──", ""],
        ]
        
        for _, r in prime.iterrows():
            summary.append([r['symbol'], f"Score {r['score']}/7 | Entry {r['entry_trigger']} | SL {r['stop_loss']}"])
        
        summary.append([""])
        summary.append(["── NR7 TRIGGERS ──", ""])
        for _, r in nr7_all.iterrows():
            summary.append([r['symbol'], f"NR7+IB:{r['nr7_ib']} | Entry {r['entry_trigger']} | SL {r['stop_loss']}"])
        
        ws2.update(summary, value_input_option="USER_ENTERED")
        ws2.format("A1:B1", {"textFormat": {"bold": True, "fontSize": 14}})
        
        print(f"✅ Pushed {len(results_df)} stocks to Google Sheets")
        print(f"   🔥 Prime: {len(prime)} | ⭐ Strong: {len(strong)} | NR7: {len(nr7_all)}")
        return True
        
    except Exception as e:
        print(f"❌ Sheets push failed: {e}")
        return False


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────
def run_scanner():
    scan_date = datetime.now().strftime("%d-%b-%Y")
    print(f"\n{'='*60}")
    print(f"  DIVITIAE EOD SCANNER — {scan_date}")
    print(f"{'='*60}\n")
    
    symbols = get_nifty500_symbols()
    results = []
    
    for i, symbol in enumerate(symbols):
        try:
            df = fetch_stock_data(symbol)
            if df is None or len(df) < 210:
                continue
            
            # Run all modules
            ema_ok,  ema_data   = check_ema_stack(df)
            near_52w, pct_52w   = check_52w_proximity(df)
            vol_ok,  vol_data   = check_volume_contraction(df)
            atr_ok,  atr_data   = check_atr_contraction(df)
            nr7, nr7_ib, nr7_data = check_nr7(df)
            rs_score            = calc_rs_score(df)
            score               = compute_score(ema_ok, near_52w, vol_ok, atr_ok, nr7, nr7_ib)
            
            # Only keep score >= 3 to reduce noise
            if score < 3:
                continue
            
            results.append({
                'symbol':       symbol,
                'close':        ema_data.get('close', 0),
                'ema20':        ema_data.get('ema20', 0),
                'ema50':        ema_data.get('ema50', 0),
                'ema200':       ema_data.get('ema200', 0),
                'ema_stack':    ema_ok,
                'pct_from_52w': pct_52w,
                'near_52w':     near_52w,
                'vol_ratio':    vol_data.get('vol_ratio', 0),
                'vol_contracted': vol_ok,
                'atr_ratio':    atr_data.get('atr_ratio', 0),
                'atr_contracted': atr_ok,
                'nr7':          nr7,
                'inside_bar':   nr7_data.get('inside_bar', False),
                'nr7_ib':       nr7_ib,
                'rs_score':     rs_score,
                'entry_trigger':nr7_data.get('entry_trigger', 0),
                'stop_loss':    nr7_data.get('stop_loss', 0),
                'score':        score
            })
            
            status = "🔥" if score >= 6 else ("⭐" if score >= 5 else "·")
            print(f"  {status} [{i+1:3d}] {symbol:<15} Score:{score}/7 | NR7:{nr7} | EMA:{ema_ok}")
            
        except Exception as e:
            print(f"  ⚠️  {symbol}: {e}")
            continue
    
    if not results:
        print("No qualifying stocks found today.")
        return
    
    results_df = pd.DataFrame(results)
    
    # Apply RS rank filter — keep top RS_TOP_PCT percentile
    if rs_score is not None:
        rs_threshold = results_df['rs_score'].dropna().quantile(1 - RS_TOP_PCT/100)
        results_df = results_df[results_df['rs_score'] >= rs_threshold]
    
    # Sort by score desc, then RS desc
    results_df = results_df.sort_values(['score','rs_score'], ascending=[False, False])
    
    print(f"\n{'='*60}")
    print(f"  SCAN COMPLETE — {len(results_df)} qualifying stocks")
    print(f"{'='*60}\n")
    
    # Push to Sheets
    push_to_sheets(results_df, scan_date)
    
    # Also save locally as CSV backup
    os.makedirs("outputs", exist_ok=True)
    csv_path = f"outputs/scan_{datetime.now().strftime('%Y%m%d')}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"✅ CSV saved: {csv_path}")


if __name__ == "__main__":
    run_scanner()
