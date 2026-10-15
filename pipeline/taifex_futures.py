"""
小台（MTX）／台指期（TX）歷史資料下載 + 連續月（換月調整）+ 週小台配對
==========================================================================

為什麼要做換月調整
------------------
期貨每個月結算一次，近月合約到期後要換到次月。次月和近月之間有價差
（正價差或逆價差，大小每次不同，由實際資料決定）。如果直接把「每天的近月價格」
接起來，換月那天就會出現一個假的跳空，模型會把它誤認為真的漲跌。

選擇權與期貨的配對
------------------
  週選（202610W2 等）→ 同代碼的週小台（MTX 202610W2）
  月選（202610）     → 同月份的小台（MTX 202610）
  週五／週一到期的 F 系列週選目前沒有對應的週小台，只能用選擇權平價遠期
build 會另外輸出 *_weekly.csv：每天每個週小台合約的結算價，供配對使用。

做法：
  1. 決定每天「持有」哪個合約（預設：結算日前一個交易日收盤後換到次月）
  2. 在換月前最後一天，同時取新舊合約的結算價，價差就是要消除的跳空
  3. 把換月前的所有歷史價格往回調整（比例或點數），最新一段保持真實價格
  4. 每日報酬一律用「同一個合約」的前後收盤計算，換月日也不例外

使用方式（需在能連到期交所的電腦上執行下載）：
  pip install requests pandas numpy matplotlib

  # 1) 下載小台（2001/04 上市；以月為單位，已下載的月份會跳過，每月間隔 1.5 秒）
  python taifex_futures.py download --product MTX --start 2001-04 --cache data/raw_mtx

  # 2) 建連續月（同時輸出週小台結算價表）
  python taifex_futures.py build --product MTX --cache data/raw_mtx --out mtx_cont.csv

  # 想要更長的歷史可改抓大台：--product TX --start 1998-07 --cache data/raw_tx

參數說明：
  --adjust  ratio（預設，比例調整，保留百分比報酬，適合建模）
            diff （點數調整，保留每個換月區段內的點數漲跌）
            none （不調整，只把近月接起來，僅供對照）
  --session day （預設，只用一般時段 08:45-13:45）
            full（夜盤 + 一般時段合成一根日 K；2017/5 以前沒有夜盤，自動只用一般時段）
  --roll    expiry（預設，結算日前 N 個交易日收盤後換月，N 由 --roll-offset 指定，預設 1）
            oi    （次月未平倉量超過近月就換，最晚仍在結算日前換）
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

URL = "https://www.taifex.com.tw/cht/3/futDataDown"
CONTRACT_RE = re.compile(r"^\d{6}(W\d)?$")   # 月合約與週合約；排除 202609/202610 這類價差單
MONTH_RE = re.compile(r"^\d{6}$")


# ----------------------------------------------------------------------------
# 1. 下載
# ----------------------------------------------------------------------------
def _month_iter(start: str, end: str):
    y, m = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    while (y, m) <= (ye, me):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def _fetch(product: str, d0: date, d1: date, retries: int = 3) -> str:
    import requests
    params = {"down_type": "1", "commodity_id": product,
              "queryStartDate": d0.strftime("%Y/%m/%d"), "queryEndDate": d1.strftime("%Y/%m/%d")}
    for k in range(retries):
        try:
            r = requests.get(URL, params=params, timeout=60,
                             headers={"User-Agent": "Mozilla/5.0 (research script)"})
            r.raise_for_status()
            return r.content.decode("cp950", errors="replace")
        except Exception as e:  # noqa: BLE001
            if k == retries - 1:
                raise
            time.sleep(3 * (k + 1))
    return ""


def download(start: str, end: str, cache: str, product: str = "MTX", pause: float = 1.5):
    """以月為單位下載，存成 cache/TX_YYYYMM.csv。已存在的完整月份跳過；當月每次重抓。"""
    os.makedirs(cache, exist_ok=True)
    today = date.today()
    for y, m in _month_iter(start, end):
        path = os.path.join(cache, f"{product}_{y}{m:02d}.csv")
        is_current = (y, m) >= (today.year, today.month)
        if os.path.exists(path) and not is_current:
            continue
        d0 = date(y, m, 1)
        d1 = (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1))
        d1 = min(d1, today)
        if d0 > today:
            break
        try:
            text = _fetch(product, d0, d1)
        except Exception as e:  # noqa: BLE001
            print(f"  {y}-{m:02d} 下載失敗：{e}，改逐日下載")
            text = _fetch_by_day(product, d0, d1, pause)
        n = max(text.count("\n") - 1, 0)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  {y}-{m:02d}  {n:>5} 行")
        time.sleep(pause)


def _fetch_by_day(product, d0, d1, pause):
    parts, header = [], None
    d = d0
    while d <= d1:
        if d.weekday() < 5:
            t = _fetch(product, d, d)
            lines = t.splitlines()
            if len(lines) > 1:
                header = header or lines[0]
                parts.extend(lines[1:])
            time.sleep(pause)
        d += timedelta(days=1)
    return "\n".join([header or ""] + parts)


# ----------------------------------------------------------------------------
# 2. 解析
# ----------------------------------------------------------------------------
def _num(s):
    s = (s or "").strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_text(text: str, product: str = "MTX") -> pd.DataFrame:
    rows = []
    for c in csv.reader(io.StringIO(text)):
        if len(c) < 18 or not re.match(r"^\d{4}/\d{2}/\d{2}$", c[0].strip()):
            continue
        if c[1].strip() != product:
            continue
        contract = c[2].strip()
        if not CONTRACT_RE.match(contract):
            continue
        sess_raw = c[17].strip()
        session = "night" if ("盤後" in sess_raw or "後" in sess_raw) else "day"
        rows.append({
            "date": c[0].strip(), "contract": contract, "session": session,
            "kind": "month" if MONTH_RE.match(contract) else "week",
            "open": _num(c[3]), "high": _num(c[4]), "low": _num(c[5]), "close": _num(c[6]),
            "volume": _num(c[9]), "settle": _num(c[10]), "oi": _num(c[11]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    # 結算日當天結算價欄為 0，視為缺值
    df.loc[df["settle"] <= 0, "settle"] = np.nan
    return df


def load_cache(cache: str, product: str = "MTX") -> pd.DataFrame:
    frames = []
    for fn in sorted(os.listdir(cache)):
        if fn.startswith(product + "_") and fn.endswith(".csv"):
            with open(os.path.join(cache, fn), encoding="utf-8", errors="replace") as f:
                frames.append(parse_text(f.read(), product))
    if not frames:
        raise SystemExit(f"{cache} 裡沒有 {product}_*.csv，請先執行 download")
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(["date", "contract", "session"], keep="last").sort_values(["date", "contract"])


# ----------------------------------------------------------------------------
# 3. 到期日與持有合約
# ----------------------------------------------------------------------------
def third_wednesday(contract: str) -> pd.Timestamp:
    """月合約：第三個週三；週合約 YYYYMMWn：第 n 個週三（未到期合約的推估值）。"""
    y, m = int(contract[:4]), int(contract[4:6])
    n = int(contract[7]) if len(contract) >= 8 and contract[6] == "W" else 3
    d = date(y, m, 1)
    d += timedelta(days=(2 - d.weekday()) % 7)      # 第一個週三
    return pd.Timestamp(d + timedelta(days=7 * (n - 1)))


def expiries(day: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """已到期合約：資料中最後出現的日期（就是實際結算日，會自動反映假日順延）。
    尚未到期合約：推估為該月第三個週三。"""
    last_date = day["date"].max()
    out = {}
    for c, g in day.groupby("contract"):
        last = g["date"].max()
        out[c] = last if last < last_date else max(third_wednesday(c), last)
    return out


def held_contracts(day: pd.DataFrame, roll: str = "expiry", offset: int = 1) -> pd.Series:
    """回傳 index=交易日、值=當天持有的合約。"""
    exp = expiries(day)
    dates = np.array(sorted(day["date"].unique()))
    pos = {d: i for i, d in enumerate(dates)}

    def last_hold_day(c):
        """持有 c 的最後一天：結算日往前 offset 個交易日。"""
        e = exp[c]
        i = np.searchsorted(dates, np.datetime64(e))      # 結算日（或其後第一個交易日）的位置
        return dates[max(i - offset, 0)]

    by_date = {d: g.set_index("contract") for d, g in day.groupby("date")}
    held, cur = {}, None
    for d in dates:
        g = by_date[d]
        alive = sorted(c for c in g.index if d <= last_hold_day(c))
        if not alive:
            alive = sorted(g.index)
        if cur is None or cur not in alive:
            cur = alive[0]
        if roll == "oi" and len(alive) > 1:
            nxt = alive[alive.index(cur) + 1] if alive.index(cur) + 1 < len(alive) else None
            if nxt is not None and g.at[nxt, "oi"] > g.at[cur, "oi"]:
                cur = nxt
        held[d] = cur
    return pd.Series(held, name="contract")


# ----------------------------------------------------------------------------
# 4. 連續月
# ----------------------------------------------------------------------------
def _bars(raw: pd.DataFrame, session: str) -> pd.DataFrame:
    """每個 (date, contract) 一根 K。session=full 時把同日歸屬的夜盤併進來。"""
    day = raw[raw["session"] == "day"].set_index(["date", "contract"])
    if session == "day":
        return day[["open", "high", "low", "close", "volume", "settle", "oi"]].copy()
    night = raw[raw["session"] == "night"].set_index(["date", "contract"])
    night = night.reindex(day.index)
    has_n = night["open"].notna() & (night["volume"] > 0)
    out = day[["open", "high", "low", "close", "volume", "settle", "oi"]].copy()
    out.loc[has_n, "open"] = night.loc[has_n, "open"]
    out["high"] = np.fmax(day["high"], night["high"])
    out["low"] = np.fmin(day["low"], night["low"])
    out["volume"] = day["volume"].fillna(0) + night["volume"].fillna(0)
    return out


def build_continuous(raw: pd.DataFrame, adjust: str = "ratio", session: str = "day",
                     roll: str = "expiry", offset: int = 1):
    raw = raw[raw["kind"] == "month"]
    day_rows = raw[raw["session"] == "day"]
    held = held_contracts(day_rows, roll, offset)
    bars = _bars(raw, session)

    def px(d, c, col):
        try:
            return float(bars.at[(d, c), col])
        except KeyError:
            return np.nan

    def ref_price(d, c):
        """換月比價用：結算價優先，沒有就用收盤價。"""
        s = px(d, c, "settle")
        return s if np.isfinite(s) else px(d, c, "close")

    dates = list(held.index)
    recs, rolls = [], []
    for i, d in enumerate(dates):
        c = held[d]
        r = {"date": d, "contract": c}
        for k in ("open", "high", "low", "close", "settle", "volume", "oi"):
            r["raw_" + k] = px(d, c, k)
        # 報酬：一律同合約前後收盤
        if i > 0:
            prev = dates[i - 1]
            pc = px(prev, c, "close")
            if not np.isfinite(pc):                       # 新合約前一天沒成交，用結算價
                pc = px(prev, c, "settle")
            r["ret"] = np.log(r["raw_close"] / pc) if np.isfinite(pc) and pc > 0 else np.nan
            if held[prev] != c:
                old, new = held[prev], c
                po, pn = ref_price(prev, old), ref_price(prev, new)
                rolls.append({"roll_date": d, "last_old_day": prev, "from": old, "to": new,
                              "old_price": po, "new_price": pn, "gap_pts": pn - po,
                              "gap_pct": pn / po - 1 if po else np.nan})
                r["roll"] = True
        recs.append(r)
    cont = pd.DataFrame(recs).set_index("date")
    cont["roll"] = cont["roll"].fillna(False).astype(bool) if "roll" in cont else False
    roll_log = pd.DataFrame(rolls)

    # 回推調整：從最新往回累積
    cont["adj"] = 1.0 if adjust == "ratio" else 0.0
    if adjust in ("ratio", "diff") and not roll_log.empty:
        factor = 1.0 if adjust == "ratio" else 0.0
        adj = pd.Series(factor, index=cont.index)
        for _, rr in roll_log.sort_values("roll_date", ascending=False).iterrows():
            if not (np.isfinite(rr.old_price) and np.isfinite(rr.new_price)):
                print(f"  警告：{rr.last_old_day:%Y-%m-%d} 缺少換月比價，該次不調整")
                continue
            mask = cont.index < rr.roll_date
            if adjust == "ratio":
                adj[mask] *= rr.new_price / rr.old_price
            else:
                adj[mask] += rr.new_price - rr.old_price
        cont["adj"] = adj
        roll_log["cum_adj_before"] = [adj[adj.index < x].iloc[-1] if (adj.index < x).any() else np.nan
                                      for x in roll_log["roll_date"]]
    for k in ("open", "high", "low", "close", "settle"):
        cont[k] = cont["raw_" + k] * cont["adj"] if adjust == "ratio" else cont["raw_" + k] + cont["adj"]

    # 全市場量（所有月合約一般時段加總）
    tot = day_rows.groupby("date")[["volume", "oi"]].sum()
    cont["total_volume"] = tot["volume"]
    cont["total_oi"] = tot["oi"]
    cols = ["contract", "open", "high", "low", "close", "settle", "ret", "volume", "oi",
            "total_volume", "total_oi", "roll", "adj",
            "raw_open", "raw_high", "raw_low", "raw_close", "raw_settle"]
    cont["volume"], cont["oi"] = cont["raw_volume"], cont["raw_oi"]
    return cont[cols], roll_log


def naive_nearby(raw: pd.DataFrame) -> pd.Series:
    """不調整：每天直接取最近月收盤（結算日當天也取到期合約），用來對照跳空。"""
    day = raw[(raw["session"] == "day") & (raw["kind"] == "month")]
    return day.sort_values("contract").groupby("date")["close"].first()


def weekly_table(raw: pd.DataFrame) -> pd.DataFrame:
    """週小台：每天每個週合約的結算價、收盤、成交量與到期日（資料中最後出現日）。"""
    w = raw[(raw["kind"] == "week") & (raw["session"] == "day")].copy()
    if w.empty:
        return w
    exp = expiries(w)
    w["expiry"] = w["contract"].map(exp)
    return w[["date", "contract", "expiry", "settle", "close", "volume", "oi"]].sort_values(["date", "contract"])


def matching_future(option_contract: str) -> str | None:
    """選擇權合約 → 對應的小台合約代碼。週選 W → 同代碼週小台；月選 → 同月小台；F 系列 → 無。"""
    oc = option_contract.strip()
    if re.match(r"^\d{6}$", oc) or re.match(r"^\d{6}W\d$", oc):
        return oc
    return None


# ----------------------------------------------------------------------------
# 5. 檢查與圖
# ----------------------------------------------------------------------------
def check(cont: pd.DataFrame, roll_log: pd.DataFrame, raw: pd.DataFrame, adjust: str):
    print(f"\n期間 {cont.index.min():%Y-%m-%d} ~ {cont.index.max():%Y-%m-%d}，{len(cont)} 個交易日，換月 {len(roll_log)} 次")
    if not roll_log.empty:
        g = roll_log["gap_pts"]
        print(f"換月價差：平均 {g.mean():+.1f} 點，絕對值中位數 {g.abs().median():.0f} 點，最大 {g.abs().max():.0f} 點")
        by_m = roll_log.assign(m=roll_log["roll_date"].dt.month).groupby("m")["gap_pct"].mean() * 100
        print("各月份平均換月價差 (%)：" + "  ".join(f"{m}月 {v:+.2f}" for m, v in by_m.items()))
    # 調整後序列的報酬應等於合約內報酬
    if adjust == "ratio":
        r_adj = np.log(cont["close"] / cont["close"].shift(1))
        diff = (r_adj - cont["ret"]).abs().max()
        print(f"檢查：比例調整後的日報酬與合約內報酬最大差異 = {diff:.2e}（應接近 0）")
    naive = naive_nearby(raw).reindex(cont.index)
    r_naive = np.log(naive / naive.shift(1))
    fake = (r_naive - cont["ret"]).dropna()
    big = fake[fake.abs() > 0.0005]
    print(f"未調整序列的假報酬：{len(big)} 天超過 0.05%，最大 {fake.abs().max() * 100:.2f}%"
          f"（{fake.abs().idxmax():%Y-%m-%d}），累積誤差 {fake.sum() * 100:+.1f}%")
    print("  → 連續月序列已全部消除；這些假報酬都發生在換月前後")


def plot(cont: pd.DataFrame, roll_log: pd.DataFrame, raw: pd.DataFrame, out: str, years: int = 3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    # Colab/Linux：fonts-noto-cjk 的 .ttc 不一定被自動註冊，先手動加入
    for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
            except Exception:  # noqa: BLE001
                pass
    names = [f.name for f in font_manager.fontManager.ttflist]
    for name in ("Noto Sans CJK TC", "Microsoft JhengHei", "PingFang TC", "Heiti TC",
                 "Noto Sans CJK JP", "Noto Sans CJK SC"):
        if name in names:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    start = cont.index.max() - pd.DateOffset(years=years)
    c = cont[cont.index >= start]
    naive = naive_nearby(raw).reindex(c.index)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 2, 2]})
    ax = axes[0]
    ax.plot(c.index, naive, color="#bbb", lw=1, label="未調整（直接接近月）")
    ax.plot(c.index, c["close"], color="#1f5f7a", lw=1.4, label="連續月（換月調整）")
    for d in c.index[c["roll"]]:
        ax.axvline(d, color="#c8321e", lw=.5, alpha=.25)
    ax.set_title(f"期貨連續月，近 {years} 年（紅線＝換月日）")
    ax.legend(frameon=False)
    ax = axes[1]
    r_naive = np.log(naive / naive.shift(1)) * 100
    ax.plot(c.index, r_naive - c["ret"] * 100, color="#c8321e", lw=1)
    ax.set_title("未調整序列多算出來的假報酬 (%)：只有換月時不為 0")
    ax = axes[2]
    if not roll_log.empty:
        rl = roll_log[roll_log["roll_date"] >= start - pd.DateOffset(years=max(0, 10 - years))]
        colors = ["#c8321e" if v > 0 else "#16804a" for v in rl["gap_pts"]]
        ax.bar(rl["roll_date"], rl["gap_pts"], width=20, color=colors)
        ax.axhline(0, color="#333", lw=.8)
        ax.set_title("每次換月的價差（次月 − 近月，點）：紅＝正價差，綠＝逆價差")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"圖已存：{out}")


# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="台指期歷史資料與連續月")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--start", default="2001-04")
    d.add_argument("--end", default=date.today().strftime("%Y-%m"))
    d.add_argument("--cache", default="data/raw_mtx")
    d.add_argument("--product", default="MTX")
    d.add_argument("--pause", type=float, default=1.5)
    b = sub.add_parser("build")
    b.add_argument("--cache", default="data/raw_mtx")
    b.add_argument("--product", default="MTX")
    b.add_argument("--adjust", choices=["ratio", "diff", "none"], default="ratio")
    b.add_argument("--session", choices=["day", "full"], default="day")
    b.add_argument("--roll", choices=["expiry", "oi"], default="expiry")
    b.add_argument("--roll-offset", type=int, default=1)
    b.add_argument("--out", default="mtx_cont.csv")
    b.add_argument("--plot", default="mtx_cont.png")
    a = ap.parse_args(argv)

    if a.cmd == "download":
        download(a.start, a.end, a.cache, a.product, a.pause)
        return
    raw = load_cache(a.cache, a.product)
    cont, roll_log = build_continuous(raw, a.adjust, a.session, a.roll, a.roll_offset)
    cont.to_csv(a.out, float_format="%.4f")
    roll_log.to_csv(os.path.splitext(a.out)[0] + "_rolls.csv", index=False, float_format="%.4f")
    print(f"連續月已存：{a.out}；換月紀錄：{os.path.splitext(a.out)[0]}_rolls.csv")
    wk = weekly_table(raw)
    if not wk.empty:
        wk.to_csv(os.path.splitext(a.out)[0] + "_weekly.csv", index=False, float_format="%.4f")
        print(f"週合約結算價表：{os.path.splitext(a.out)[0]}_weekly.csv"
              f"（{wk['date'].min():%Y-%m-%d} 起，{wk['contract'].nunique()} 個週合約）")
    check(cont, roll_log, raw, a.adjust)
    if a.plot:
        plot(cont, roll_log, raw, a.plot)
    return cont, roll_log


if __name__ == "__main__":
    main()
