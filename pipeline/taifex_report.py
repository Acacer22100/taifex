"""
台指週選每日盤後報告（週三到期系列：W 週選 + 月選）
=====================================================
在 Colab 每天收盤後執行一次：
  1. 下載最新交易日（與前一交易日）的台指選擇權每日行情
  2. 對週三到期系列最近 3 個到期，計算選擇權平價遠期、價平 IV、隱含分布分位數、偏態
     並對照同到期小台（週選→週小台，月選→近月小台）
  3. 從小台連續月計算價量描述統計（不是預測）
  4. 輸出 report_latest.json（與 reports/report_YYYYMMDD.json），網頁與 Claude 讀同一份

選擇權分布的演算法與網頁（engine.js）一致，數字可以互相對照。
研究用途，不構成投資建議。
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm

OPT_URL = "https://www.taifex.com.tw/cht/3/optDataDown"
TPE = timezone(timedelta(hours=8))
WED_RE = re.compile(r"^\d{6}(W\d)?$")          # 週三到期系列：週選 W 與月選；排除 F 系列

# 依期交所「中華民國115年期貨集中交易市場行事曆」；颱風等臨時休市無法預知
TW_HOLIDAYS = {
    "2026-01-01", "2026-02-12", "2026-02-13", "2026-02-16", "2026-02-17", "2026-02-18",
    "2026-02-19", "2026-02-20", "2026-02-27", "2026-04-03", "2026-04-06", "2026-05-01",
    "2026-06-19", "2026-09-25", "2026-09-28", "2026-10-09", "2026-10-26", "2026-12-25",
    "2027-01-01",
}


# ----------------------------------------------------------------------------
# 選擇權資料
# ----------------------------------------------------------------------------
def _num(s):
    try:
        v = float(str(s).strip().replace(",", ""))
        return v if math.isfinite(v) else np.nan
    except ValueError:
        return np.nan


def fetch_options_day(d: date, cache: str, retries: int = 3) -> str | None:
    """下載某日 TXO 每日行情（快取）。非交易日或尚未公布回傳 None。"""
    import requests
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"TXO_{d:%Y%m%d}.csv")
    if os.path.exists(path):
        text = open(path, encoding="utf-8").read()
        return text if ",TXO," in text else None
    params = {"down_type": "1", "commodity_id": "TXO",
              "queryStartDate": d.strftime("%Y/%m/%d"), "queryEndDate": d.strftime("%Y/%m/%d")}
    for k in range(retries):
        try:
            r = requests.get(OPT_URL, params=params, timeout=60,
                             headers={"User-Agent": "Mozilla/5.0 (research script)"})
            r.raise_for_status()
            text = r.content.decode("cp950", errors="replace")
            break
        except Exception:  # noqa: BLE001
            if k == retries - 1:
                raise
            time.sleep(3 * (k + 1))
    has_day = any(",TXO," in ln and "一般" in ln for ln in text.splitlines()[:400])
    # 只快取「已完整公布」的過去日期；今天的檔可能還沒收盤，不存
    if has_day and d < datetime.now(TPE).date():
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text if has_day else None


def latest_two_days(cache: str, start: date | None = None, max_back: int = 12, pause: float = 1.0):
    """往回找最近兩個有資料的交易日，回傳 [(日期, 文字), (日期, 文字)]（新到舊）。"""
    d = start or datetime.now(TPE).date()
    found = []
    for _ in range(max_back):
        if d.weekday() < 5 and d.isoformat() not in TW_HOLIDAYS:
            try:
                t = fetch_options_day(d, cache)
            except Exception as e:  # noqa: BLE001
                print(f"  {d} 選擇權下載失敗（{e}），往前一天找")
                t = None
            if t:
                found.append((d, t))
                if len(found) == 2:
                    break
            time.sleep(pause)
        d -= timedelta(days=1)
    if not found:
        raise RuntimeError("最近幾天都抓不到選擇權資料：可能是期交所網站連不上，或今天尚未公布（約 15:00 後）")
    return found


def parse_options(text: str) -> pd.DataFrame:
    rows = []
    for ln in text.splitlines()[1:]:
        c = ln.split(",")
        if len(c) < 18 or c[1].strip() != "TXO" or "一般" not in c[17]:
            continue
        cp = "C" if "買" in c[4] else ("P" if "賣" in c[4] else None)
        if cp is None:
            continue
        rows.append({"date": c[0].strip(), "contract": c[2].strip(), "strike": _num(c[3]), "cp": cp,
                     "close": _num(c[8]), "volume": _num(c[9]), "settle": _num(c[10]), "oi": _num(c[11]),
                     "bid": _num(c[12]), "ask": _num(c[13]),
                     "expiry": c[20].strip() if len(c) > 20 else ""})
    return pd.DataFrame(rows)


def chain_for(df: pd.DataFrame, contract: str) -> pd.DataFrame:
    """strike, C, P；結算價優先，其次買賣中價，再其次收盤價（與網頁一致）。"""
    d = df[df.contract == contract].copy()
    mid = (d.bid + d.ask) / 2
    d["px"] = np.where(d.settle > 0, d.settle,
                       np.where((d.bid > 0) & (d.ask > 0), mid, np.where(d.close > 0, d.close, np.nan)))
    w = d.pivot_table(index="strike", columns="cp", values="px", aggfunc="first").reset_index()
    for k in ("C", "P"):
        if k not in w:
            w[k] = np.nan
    return w[["strike", "C", "P"]].sort_values("strike").reset_index(drop=True)


def trading_days_between(trade_date: str, expiry: str) -> int:
    d0 = datetime.strptime(trade_date.replace("/", "-")[:10], "%Y-%m-%d").date()
    d1 = datetime.strptime(expiry, "%Y%m%d").date()
    n, t = 0, d0 + timedelta(days=1)
    while t < d1:
        if t.weekday() < 5 and t.isoformat() not in TW_HOLIDAYS:
            n += 1
        t += timedelta(days=1)
    return n


# ----------------------------------------------------------------------------
# 隱含分布（與 engine.js 相同演算法）
# ----------------------------------------------------------------------------
def b76(F, K, T, vol, cp):
    if vol <= 0 or T <= 0:
        return max(F - K, 0) if cp == "C" else max(K - F, 0)
    sd = vol * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    if cp == "C":
        return F * norm.cdf(d1) - K * norm.cdf(d2)
    return K * norm.cdf(-d2) - F * norm.cdf(-d1)


def implied_vol(price, F, K, T, cp):
    intrinsic = max(F - K, 0) if cp == "C" else max(K - F, 0)
    if not (price > intrinsic + 1e-6) or b76(F, K, T, 5.0, cp) < price:
        return np.nan
    lo, hi = 1e-4, 5.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if b76(F, K, T, mid, cp) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def implied_forward(chain: pd.DataFrame) -> float:
    w = chain[(chain.C > 0) & (chain.P > 0)].copy()
    if w.empty:
        raise ValueError("沒有同時有買權與賣權價格的履約價")
    w["f"] = w.strike + w.C - w.P
    w["d"] = (w.C - w.P).abs()
    near = sorted(w.nsmallest(7, "d")["f"])
    return float(near[len(near) // 2])


def build(chain: pd.DataFrame, T: float) -> dict:
    F = implied_forward(chain)
    sm = []
    for _, r in chain.iterrows():
        side = "P" if r.strike < F else "C"
        px = r[side]
        if not (px > 0.4) or abs(math.log(r.strike / F)) > 0.15:
            continue
        iv = implied_vol(px, F, r.strike, T, side)
        if np.isfinite(iv):
            sm.append([r.strike, side, px, iv])
    keep = []
    for i, s in enumerate(sm):
        nb = sorted(z[3] for z in sm[max(0, i - 2): i + 3])
        med = nb[len(nb) // 2]
        if abs(s[3] - med) < 0.35 * med:
            keep.append(s)
    if len(keep) < 6:
        raise ValueError("可用履約價少於 6 檔")
    K = np.array([s[0] for s in keep])
    iv = np.array([s[3] for s in keep])
    scale = 0.05
    ks = np.log(K / F) / scale
    tv = iv ** 2 * T
    coef = np.polyfit(ks, tv, 4 if len(keep) >= 12 else 2)
    min_tv = tv.min() * 0.3

    def vol_at(x):
        return np.sqrt(np.maximum(np.polyval(coef, np.log(np.asarray(x) / F) / scale), min_tv) / T)

    lo, hi, N = K[0], K[-1], 1201
    grid = np.linspace(lo, hi, N)
    dK = grid[1] - grid[0]
    calls = np.array([b76(F, k, T, v, "C") for k, v in zip(grid, vol_at(grid))])
    d1 = np.gradient(calls, dK)
    pdf = np.zeros(N)
    pdf[1:-1] = (calls[2:] - 2 * calls[1:-1] + calls[:-2]) / dK ** 2
    pdf[0], pdf[-1] = pdf[1], pdf[-2]
    pdf = np.clip(pdf, 0, None)
    p_above = float(np.clip(-d1[-1], 0, 1))
    p_below = float(np.clip(1 + d1[0], 0, 1))
    mass = float(np.sum((pdf[1:] + pdf[:-1]) / 2 * dK))
    pdf *= max(1 - p_above - p_below, 1e-6) / mass
    cdf = p_below + np.concatenate([[0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * dK)])
    atm = float(vol_at(F))
    return {"F": F, "T": T, "grid": grid, "pdf": pdf, "cdf": cdf, "atm_iv": atm, "vol_at": vol_at,
            "smile": keep}


def quantile(d, q):
    return float(np.interp(q, d["cdf"], d["grid"]))


def prob_above(d, x):
    return float(1 - np.interp(x, d["grid"], d["cdf"]))


def summarize(d: dict) -> dict:
    F = d["F"]
    sd = F * d["atm_iv"] * math.sqrt(d["T"])
    qs = {f"q{int(q * 100):02d}": round(quantile(d, q)) for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)}
    step = 250
    base = round(F / step) * step
    levels = [{"x": base + i * step, "p_above": round(prob_above(d, base + i * step), 4)} for i in range(-6, 7)]
    skew = float(d["vol_at"](F * 0.97) - d["vol_at"](F * 1.03))      # 價外賣權 IV − 價外買權 IV
    return {"F": round(F, 1), "atm_iv": round(d["atm_iv"], 4), "sd_pts": round(sd), **qs,
            "skew_3pct": round(skew, 4), "levels": levels}


# ----------------------------------------------------------------------------
# 期貨價量（描述統計，不是預測）
# ----------------------------------------------------------------------------
def futures_stats(cont: pd.DataFrame, raw: pd.DataFrame) -> dict:
    c = cont.dropna(subset=["close"]).copy()
    last = c.index[-1]
    r = c["ret"].fillna(0)

    def cum(n):
        return float(np.expm1(r.iloc[-n:].sum()))

    def pos(n):
        w = c.iloc[-n:]
        lo, hi = w["low"].min(), w["high"].max()
        return float((c["close"].iloc[-1] - lo) / (hi - lo)) if hi > lo else None

    vol20 = c["total_volume"].iloc[-21:-1].mean()
    oi = c["total_oi"]
    held = c["contract"].iloc[-1]
    # 夜盤：當日歸屬的夜盤收盤 vs 前一日一般時段收盤（同一合約）
    night = raw[(raw.session == "night") & (raw.date == last) & (raw.contract == held)]
    prev_day = c.index[-2]
    prev_close = raw[(raw.session == "day") & (raw.date == prev_day) & (raw.contract == held)]["close"]
    night_chg = (float(night["close"].iloc[0] - prev_close.iloc[0])
                 if len(night) and len(prev_close) and night["volume"].iloc[0] > 0 else None)
    tail = c.iloc[-120:]
    return {
        "date": last.strftime("%Y-%m-%d"), "contract": held,
        "close": float(c["raw_close"].iloc[-1]), "settle": float(c["raw_settle"].iloc[-1]),
        "ret_1d": cum(1), "ret_5d": cum(5), "ret_20d": cum(20),
        "rv20": float(r.iloc[-20:].std(ddof=1) * math.sqrt(252)),
        "rv60": float(r.iloc[-60:].std(ddof=1) * math.sqrt(252)),
        "pos20": pos(20), "pos60": pos(60),
        "volume": float(c["total_volume"].iloc[-1]), "volume_vs_20d": float(c["total_volume"].iloc[-1] / vol20) if vol20 else None,
        "oi": float(oi.iloc[-1]), "oi_chg_1d": float(oi.iloc[-1] - oi.iloc[-2]), "oi_chg_5d": float(oi.iloc[-1] - oi.iloc[-6]),
        "night_chg": night_chg,
        "series": {"date": [d.strftime("%Y-%m-%d") for d in tail.index],
                   "close": [round(float(x), 1) for x in tail["close"]],
                   "volume": [float(x) for x in tail["total_volume"]],
                   "oi": [float(x) for x in tail["total_oi"]]},
    }


# ----------------------------------------------------------------------------
# 報告
# ----------------------------------------------------------------------------
def wed_contracts(df: pd.DataFrame, trade_date: str, n: int = 3):
    cs = (df[df.contract.str.match(WED_RE)][["contract", "expiry"]].drop_duplicates()
          .sort_values("expiry"))
    td = trade_date.replace("/", "")
    cs = cs[cs.expiry > td]
    return list(cs.itertuples(index=False))[:n]


def make_report(opt_days, fut_raw: pd.DataFrame | None, cont: pd.DataFrame | None, n_contracts: int = 3) -> dict:
    (d0, t0) = opt_days[0]
    df = parse_options(t0)
    trade_date = df["date"].iloc[0]
    prev = parse_options(opt_days[1][1]) if len(opt_days) > 1 else None
    fut_today = {}
    if fut_raw is not None:
        fd = fut_raw[(fut_raw.session == "day") & (fut_raw.date == pd.Timestamp(d0))]
        for _, r in fd.iterrows():
            px = r.settle if r.settle > 0 else r.close
            if np.isfinite(px):
                fut_today[r.contract] = float(px)
    out = {"version": 1, "date": trade_date, "generated_at": datetime.now(TPE).isoformat(timespec="seconds"),
           "series": "週三到期（週選＋月選）", "contracts": [], "futures": None,
           "model": {"status": "未通過", "note": MODEL_NOTE}, "risk_flag": None}
    for c in wed_contracts(df, trade_date, n_contracts):
        ch = chain_for(df, c.contract)
        tdays = trading_days_between(trade_date, c.expiry) + 1
        item = {"contract": c.contract, "expiry": c.expiry, "tdays": tdays,
                "chain": [[float(k), None if not np.isfinite(cv) else float(cv), None if not np.isfinite(pv) else float(pv)]
                          for k, cv, pv in ch[["strike", "C", "P"]].itertuples(index=False)],
                "future": None, "summary": None, "prev": None}
        if c.contract in fut_today:
            item["future"] = {"product": "MTX", "contract": c.contract, "price": fut_today[c.contract]}
        try:
            dist = build(ch, max(tdays, 0.25) / 252)
            item["summary"] = summarize(dist)
            if item["future"]:
                item["summary"]["basis_vs_future"] = round(dist["F"] - item["future"]["price"], 1)
        except Exception as e:  # noqa: BLE001
            item["error"] = str(e)
        if prev is not None and (prev.contract == c.contract).any():
            try:
                pdate = prev["date"].iloc[0]
                pt = trading_days_between(pdate, c.expiry) + 1
                pdist = build(chain_for(prev, c.contract), max(pt, 0.25) / 252)
                item["prev"] = {"date": pdate, "F": round(pdist["F"], 1), "atm_iv": round(pdist["atm_iv"], 4),
                                "q10": round(quantile(pdist, .1)), "q90": round(quantile(pdist, .9))}
            except Exception:  # noqa: BLE001
                pass
        out["contracts"].append(item)
    if cont is not None and fut_raw is not None:
        out["futures"] = futures_stats(cont, fut_raw)
        first = next((x for x in out["contracts"] if x["summary"]), None)
        if first:
            out["futures"]["iv_vs_rv20"] = round(first["summary"]["atm_iv"] / out["futures"]["rv20"], 3)
        out["risk_flag"] = risk_flag(out["futures"])
    try:
        out["plan"] = trade_plan(out, cont)
    except Exception as e:  # noqa: BLE001
        out["plan"] = None; out["plan_error"] = str(e)
    return out


# ----------------------------------------------------------------------------
# 低位警示（2013–2026 共 658 週回測的結論，不是預測方向）
# ----------------------------------------------------------------------------
MODEL_NOTE = ("價量模型已回測（2016–2026 逐年樣本外 537 週），用 20 日位置、5 日漲跌調整市場分布寬度，"
              "改善不到 0.2%、不顯著，所以沒有疊加；分布圖仍是市場隱含分布。價量只保留為下方的低位警示。")
LOW_POS = 1 / 3


def risk_flag(f: dict | None) -> dict | None:
    if not f or f.get("pos20") is None:
        return None
    p = float(f["pos20"])
    if p < LOW_POS:
        return {"level": "low", "pos20": round(p, 3),
                "title": "低位警示：收盤在近 20 日區間的下方三分之一",
                "text": ("回測 658 週中，這種情況下週選市場的 80% 區間實際只命中約 71–74%（應為 80%），"
                         "實際波動的變異數約是市場隱含的 1.5 倍，跌出 5% 下緣的次數也較多。"
                         "賣窄區間（鐵兀鷹、賣勒式）的風險比權利金反映的更高。方向上沒有預測力。")}
    return {"level": "normal", "pos20": round(p, 3),
            "title": "未觸發低位警示",
            "text": "收盤不在近 20 日區間的下方三分之一；回測中這時市場區間大致準確或略寬。"}


# ----------------------------------------------------------------------------
# 上週結算對照：拿已到期合約的官方最後結算價，對照當初報告給的分布
# ----------------------------------------------------------------------------
FSP_URL = "https://www.taifex.com.tw/cht/5/futIndxFSP"


def fetch_final_settlements(work: str = ".") -> pd.DataFrame:
    """期交所「最後結算價一覽表」（預設列出近一年，含週合約），與本機快取合併。
    回傳 date, contract, settle（臺股期貨／小台／微台共用的結算價）。"""
    path = os.path.join(work, "final_settle.csv")
    hist_path = os.path.join(work, "final_settle_history.csv")     # 期交所歷史結算價表（筆記本內建）
    frames = [pd.read_csv(f, dtype={"contract": str}) for f in (hist_path, path) if os.path.exists(f)]
    old = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "contract", "settle"])
    new = pd.DataFrame(columns=["date", "contract", "settle"])
    try:
        import requests
        r = requests.get(FSP_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0 (research script)"})
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        for t in pd.read_html(io.StringIO(r.text)):
            flat = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
            if any("契約月份" in c for c in flat) and t.shape[1] >= 3:
                t = t.iloc[:, :3]
                t.columns = ["date", "contract", "settle"]
                t["contract"] = t["contract"].astype(str).str.strip()
                t["settle"] = pd.to_numeric(t["settle"], errors="coerce")
                t = t[t["contract"].str.match(r"^\d{6}(W\d)?$") & t["settle"].notna()]
                t["date"] = pd.to_datetime(t["date"].astype(str).str.strip(), format="%Y/%m/%d").dt.strftime("%Y-%m-%d")
                new = t
                break
    except Exception as e:  # noqa: BLE001
        print(f"  最後結算價頁面讀取失敗（{e}），使用已存的資料")
    allf = pd.concat([old, new], ignore_index=True).drop_duplicates("contract", keep="last")
    allf = allf.sort_values("date").reset_index(drop=True)
    allf.to_csv(path, index=False)
    return allf


def _front_reports(work: str):
    """所有歷史報告；回傳 {合約: 該合約成為「最近到期」的第一份報告}。"""
    folder = os.path.join(work, "reports")
    if not os.path.isdir(folder):
        return {}, []
    reps = []
    for fn in sorted(os.listdir(folder)):
        if fn.startswith("report_") and fn.endswith(".json"):
            try:
                reps.append(json.load(open(os.path.join(folder, fn), encoding="utf-8")))
            except Exception:  # noqa: BLE001
                pass
    front = {}
    for r in sorted(reps, key=lambda x: x["date"]):
        cs = [c for c in r.get("contracts", []) if c.get("chain")]
        if cs and cs[0]["contract"] not in front:
            front[cs[0]["contract"]] = (r["date"], cs[0])
    return front, reps


def scorecard(work: str, finals: pd.DataFrame) -> dict | None:
    """對每個已到期、且曾在報告中當過「最近到期」的合約，計算結算價落在當時分布的哪個分位。"""
    front, _ = _front_reports(work)
    fin = dict(zip(finals["contract"], zip(finals["date"], finals["settle"])))
    rows = []
    for con, (rdate, c) in front.items():
        if con not in fin:
            continue
        fdate, S = fin[con]
        ch = pd.DataFrame(c["chain"], columns=["strike", "C", "P"]).astype(float)
        try:
            d = build(ch, max(c["tdays"], 0.25) / 252)
        except Exception:  # noqa: BLE001
            continue
        F, iv, T = d["F"], d["atm_iv"], d["T"]
        pctile = float(np.interp(S, d["grid"], d["cdf"]))
        rows.append({"contract": con, "settle_date": fdate, "settle": float(S), "report_date": rdate,
                     "tdays": c["tdays"], "F": round(F, 1), "atm_iv": round(iv, 4),
                     "pctile": round(pctile, 4), "err_pts": round(S - F, 1),
                     "err_sigma": round(math.log(S / F) / (iv * math.sqrt(T)), 3),
                     "in50": bool(0.25 <= pctile <= 0.75), "in80": bool(0.10 <= pctile <= 0.90)})
    if not rows:
        return None
    log = pd.DataFrame(rows).sort_values("settle_date")
    log.to_csv(os.path.join(work, "settle_log.csv"), index=False)
    last = log.iloc[-1].to_dict()
    return {"last": {k: (v.item() if hasattr(v, "item") else v) for k, v in last.items()},
            "n": int(len(log)), "in50_rate": float(log["in50"].mean()), "in80_rate": float(log["in80"].mean()),
            "mean_pctile": float(log["pctile"].mean()),
            "mean_abs_sigma": float(log["err_sigma"].abs().mean())}


def save_report(rep: dict, work: str = "."):
    os.makedirs(os.path.join(work, "reports"), exist_ok=True)
    def _j(o):                                      # numpy 數值 → 一般數字
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)
    s = json.dumps(rep, ensure_ascii=False, separators=(",", ":"), default=_j)
    with open(os.path.join(work, "report_latest.json"), "w", encoding="utf-8") as f:
        f.write(s)
    p = os.path.join(work, "reports", f"report_{rep['date'].replace('/', '')}.json")
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)
    return p


def print_report(rep: dict):
    print(f"資料日 {rep['date']}　{rep['series']}")
    for c in rep["contracts"]:
        s = c["summary"]
        if not s:
            print(f"  {c['contract']}：{c.get('error')}")
            continue
        fut = f"，小台 {c['future']['price']:,.0f}（差 {s.get('basis_vs_future', 0):+,.0f}）" if c["future"] else ""
        chg = ""
        if c["prev"]:
            chg = f"　較前日 F {s['F'] - c['prev']['F']:+,.0f}、IV {100 * (s['atm_iv'] - c['prev']['atm_iv']):+.1f} 點"
        print(f"\n{c['contract']}（{c['expiry'][4:6]}/{c['expiry'][6:]} 到期，剩 {c['tdays']} 交易日）")
        print(f"  平價遠期 {s['F']:,.0f}{fut}")
        print(f"  價平 IV {s['atm_iv']:.1%}　±1σ ±{s['sd_pts']:,} 點　偏態（−3% 對 +3%）{100 * s['skew_3pct']:+.1f} 點{chg}")
        print(f"  10%～90%：{s['q10']:,}～{s['q90']:,}　25%～75%：{s['q25']:,}～{s['q75']:,}")
    sc = rep.get("scorecard")
    if sc:
        L = sc["last"]
        print(f"\n上週結算對照：{L['contract']} 於 {L['settle_date']} 結算 {L['settle']:,.0f}")
        print(f"  {L['report_date']} 報告（剩 {L['tdays']} 交易日）的平價遠期 {L['F']:,.0f}，誤差 {L['err_pts']:+,.0f} 點（{L['err_sigma']:+.2f}σ）")
        print(f"  落在當時分布的第 {L['pctile'] * 100:.0f} 百分位，{'在' if L['in50'] else '不在'} 50% 區間、{'在' if L['in80'] else '不在'} 80% 區間內")
        print(f"  累計 {sc['n']} 次：落在 50% 區間 {sc['in50_rate']:.0%}（應約 50%）、80% 區間 {sc['in80_rate']:.0%}（應約 80%）")
    else:
        print("\n上週結算對照：還沒有可對照的到期合約（報告累積到第一個到期日後就會出現）")
    f = rep.get("futures")
    if f:
        print(f"\n小台 {f['contract']} 收 {f['close']:,.0f}　1日 {f['ret_1d']:+.2%}　5日 {f['ret_5d']:+.2%}　20日 {f['ret_20d']:+.2%}")
        print(f"  20日實際波動 {f['rv20']:.1%}　IV/RV {f.get('iv_vs_rv20', float('nan')):.2f}　"
              f"20日區間位置 {f['pos20']:.0%}　60日 {f['pos60']:.0%}")
        print(f"  成交量為20日均量的 {f['volume_vs_20d']:.2f} 倍　未平倉 {f['oi']:,.0f}（1日 {f['oi_chg_1d']:+,.0f}、5日 {f['oi_chg_5d']:+,.0f}）")
    pl = rep.get("plan")
    if pl:
        ru = pl["rule"]
        tag = {"go": "符合規則", "skip": "規則判斷：跳過", "not_wed": "今天不是週三（規則只在週三判斷），以下僅供參考"}[pl["status"]]
        print(f"\n【週選賣權多頭價差】{tag}")
        print(f"  {'週三' if pl['is_wed'] else '當日'}漲跌 {ru['r1']:+.2%}（上限 {ru['r1_max']:+.2%}）{'✓' if ru['pass_r1'] else '✗'}　20日波動 {ru['rv20']:.1%}（下限 {ru['rv20_min']:.1%}）{'✓' if ru['pass_rv'] else '✗'}　"
              f"200日線 {ru.get('ma200') or 0:,.0f}（收 {ru.get('close') or 0:,.0f}）{'✓' if ru.get('pass_ma') else '✗'}")
        print(f"  價平 {pl['atm']:,.0f}（依{pl['ref_src']} {pl['ref']:,.0f}）")
        print(f"  {pl['contract']}：賣 {pl['short']:,.0f} Put（{pl['short_px']:g}）／買 {pl['long']:,.0f} Put（{pl['long_px']:g}）　結算價權利金 {pl['credit']:g} 點，寬 {pl['width']:,.0f}")
        print(f"  建議權利金至少 {pl['min_credit']} 點（寬度 40%）；回測兩平約 {pl['breakeven_credit']} 點；最多賺 {pl['max_profit_ntd']:,} 元／最多賠 {pl['max_loss_ntd']:,} 元")
    pt = rep.get("plan_track")
    if pt:
        print(f"  追蹤：符合規則 {pt['n_go']} 次，合計 {pt['go_ntd']:+,} 元；跳過 {pt['n_skip']} 次（假設有做 {pt['skip_ntd']:+,} 元）；未到期 {pt['pending']} 筆")
    rf = rep.get("risk_flag")
    if rf:
        print(("\n⚠ " if rf["level"] == "low" else "\n") + rf["title"] + "\n  " + rf["text"])


# ----------------------------------------------------------------------------
# 週三規則與候選價差（賣權多頭價差：賣價平+200 Put、買價平−200 Put）
# 規則來自 2013–2026 回測（候選 A）：週三小台漲幅不在歷史前 1/3、20 日波動不在歷史後 1/3，
# 再加上「小台（連續月）收盤站上 200 日均線」作為防大空頭的保險。三項都成立才判定可進場。
# 研究用途，不構成投資建議。
# ----------------------------------------------------------------------------
PLAN = {"short_off": 200, "long_off": -200, "width": 400, "min_credit_frac": 0.40, "typical_frac": 0.45,
        "breakeven_frac": 0.36, "r1_q": 2 / 3, "rv_q": 1 / 3, "start": "2013-01-01",
        "fee": 20.0, "mult": 50, "tax": 0.001}


def rule_thresholds(cont: pd.DataFrame) -> dict:
    c = cont.dropna(subset=["close"])
    r = c["ret"].fillna(0)
    rv = r.rolling(20).std(ddof=1) * math.sqrt(252)
    hist = (c.index.weekday == 2) & (c.index >= PLAN["start"]) & (c.index < c.index[-1])
    return {"r1": float(r[hist].quantile(PLAN["r1_q"])), "rv20": float(rv[hist].quantile(PLAN["rv_q"])),
            "n_wed": int(hist.sum())}


def trade_plan(rep: dict, cont: pd.DataFrame | None) -> dict | None:
    if cont is None or not rep.get("contracts"):
        return None
    thr = rule_thresholds(cont)
    c = cont.dropna(subset=["close"])
    r = c["ret"].fillna(0)
    r1 = float(r.iloc[-1]); rv20 = float(r.iloc[-20:].std(ddof=1) * math.sqrt(252))
    td = datetime.strptime(rep["date"].replace("/", "-")[:10], "%Y-%m-%d").date()
    item = next((x for x in rep["contracts"] if x.get("summary") and x["tdays"] >= 2), None)
    if item is None:
        return None
    F = float(item["summary"]["F"])
    # 價平用同到期的週小台（週選對週小台、月選對近月小台）；沒有期貨價時才用選擇權平價遠期
    fut = item.get("future") or {}
    ref = float(fut["price"]) if fut.get("price") else F
    ref_src = f"週小台 {fut.get('contract', item['contract'])}" if fut.get("price") else "選擇權平價遠期"
    puts = {float(k): float(pv) for k, _, pv in item["chain"] if pv is not None and pv > 0}
    ks_all = np.array(sorted(puts))
    if len(ks_all) < 5:
        return None
    near = lambda arr, x: float(arr[np.argmin(np.abs(arr - x))])
    atm = near(ks_all, ref)
    ks = near(ks_all, atm + PLAN["short_off"])
    lower = ks_all[ks_all < ks]
    if not len(lower):
        return None
    kl = near(lower, ks - PLAN["width"])            # 固定差 400 點，不論履約價每檔 50 或 100
    ps, pl = puts[ks], puts[kl]
    credit = ps - pl; width = ks - kl
    pass_r1 = r1 <= thr["r1"]; pass_rv = rv20 > thr["rv20"]
    close = float(c["close"].iloc[-1]); ma200 = float(c["close"].iloc[-200:].mean()) if len(c) >= 200 else np.nan
    pass_ma = bool(np.isfinite(ma200) and close > ma200)
    is_wed = td.weekday() == 2
    status = ("go" if pass_r1 and pass_rv and pass_ma else "skip") if is_wed else "not_wed"
    return {
        "date": rep["date"], "is_wed": is_wed, "status": status,
        "contract": item["contract"], "expiry": item["expiry"], "tdays": item["tdays"], "F": round(F, 1), "atm": atm,
        "ref": round(ref, 1), "ref_src": ref_src,
        "short": ks, "long": kl, "short_px": ps, "long_px": pl, "credit": round(credit, 1), "width": width,
        "min_credit": round(PLAN["min_credit_frac"] * width), "typical_credit": round(PLAN["typical_frac"] * width),
        "breakeven_credit": round(PLAN["breakeven_frac"] * width),
        "max_profit_ntd": round(credit * PLAN["mult"]), "max_loss_ntd": round((width - credit) * PLAN["mult"]),
        "breakeven_index": round(ks - credit, 1),
        "rule": {"r1": round(r1, 5), "r1_max": round(thr["r1"], 5), "pass_r1": bool(pass_r1),
                 "rv20": round(rv20, 4), "rv20_min": round(thr["rv20"], 4), "pass_rv": bool(pass_rv),
                 "close": round(close, 1), "ma200": round(ma200, 1) if np.isfinite(ma200) else None, "pass_ma": pass_ma},
    }


def _plan_cost(ps, pl, intr_s, intr_l):
    m, fee, tax = PLAN["mult"], PLAN["fee"], PLAN["tax"]
    n_itm = (intr_s > 0) + (intr_l > 0)
    return 2 * fee + tax * m * (ps + pl) + n_itm * fee + tax * m * (intr_s + intr_l)


def update_plan_log(work: str, plan: dict | None, finals: pd.DataFrame) -> dict | None:
    """週三把當週的規則判斷與候選價差記下來（不論做或跳過），到期後用官方最後結算價算出損益。"""
    path = os.path.join(work, "plan_log.csv")
    log = pd.read_csv(path, dtype={"contract": str}) if os.path.exists(path) else pd.DataFrame()
    if plan and plan["is_wed"] and (log.empty or not ((log.date == plan["date"]) & (log.contract == plan["contract"])).any()):
        row = {k: plan[k] for k in ("date", "status", "contract", "expiry", "F", "short", "long", "short_px", "long_px", "credit", "width", "min_credit")}
        row.update({"r1": plan["rule"]["r1"], "rv20": plan["rule"]["rv20"], "settle": np.nan, "pnl_pts": np.nan, "pnl_ntd": np.nan})
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    if log.empty:
        return None
    fin = dict(zip(finals["contract"].astype(str), finals["settle"]))
    for i, r in log.iterrows():
        if pd.notna(r.get("pnl_ntd")) or str(r.contract) not in fin:
            continue
        S = float(fin[str(r.contract)])
        a, b = max(r.short - S, 0.0), max(r.long - S, 0.0)
        pts = r.credit - (a - b)
        log.loc[i, ["settle", "pnl_pts", "pnl_ntd"]] = [S, pts, pts * PLAN["mult"] - _plan_cost(r.short_px, r.long_px, a, b)]
    log.to_csv(path, index=False)
    done = log.dropna(subset=["pnl_ntd"])
    go, sk = done[done.status == "go"], done[done.status == "skip"]
    return {"n_go": int(len(go)), "go_ntd": round(float(go.pnl_ntd.sum())), "go_win": float((go.pnl_ntd > 0).mean()) if len(go) else None,
            "n_skip": int(len(sk)), "skip_ntd": round(float(sk.pnl_ntd.sum())),
            "pending": int(log.pnl_ntd.isna().sum()),
            "last": [{k: (None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v)) for k, v in rr.items()}
                     for rr in log.tail(6).to_dict("records")]}
