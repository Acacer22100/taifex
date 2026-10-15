"""
每日盤後報告（GitHub Actions 版）— 與 Colab 每日筆記本相同的計算。
產出（在 repo 根目錄）：report_latest.json、reports/、plan_log.csv、settle_log.csv、final_settle.csv
原始資料快取在 data/（由 Actions cache 保存，不提交到 repo）。
研究用途，不構成投資建議。
"""
import os
import sys
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.chdir(ROOT)

import taifex_futures as T  # noqa: E402
import taifex_report as R   # noqa: E402

for d in ("reports", "data/raw_mtx", "data/raw_txo"):
    os.makedirs(d, exist_ok=True)

# 1. 小台資料（第一次抓完整歷史，之後只補近 40 天）
FUT = "data/raw_mtx"
now = datetime.now(R.TPE)
first = not any(f.endswith(".csv") for f in os.listdir(FUT))
start = "2001-04" if first else (now - timedelta(days=40)).strftime("%Y-%m")
print("第一次執行，下載小台完整歷史（約 10 分鐘）…" if first else "補抓小台最近資料…", flush=True)
T.download(start, now.strftime("%Y-%m"), FUT, "MTX", pause=1.5)
fut_raw = T.load_cache(FUT, "MTX")
print("小台資料：", fut_raw.date.min().date(), "～", fut_raw.date.max().date(), flush=True)

# 2. 連續月
cont, rolls = T.build_continuous(fut_raw, "ratio", "day")

# 3. 最新兩個交易日的選擇權
opt_days = R.latest_two_days("data/raw_txo")
print("選擇權最新交易日：", [d.isoformat() for d, _ in opt_days], flush=True)

# 4. 報告、上週結算對照、系統判定追蹤
rep = R.make_report(opt_days, fut_raw, cont)
R.save_report(rep, ROOT)
finals = R.fetch_final_settlements(ROOT)
rep["scorecard"] = R.scorecard(ROOT, finals)
rep["plan_track"] = R.update_plan_log(ROOT, rep.get("plan"), finals)
R.save_report(rep, ROOT)
R.print_report(rep)

# 5. 選擇權原始檔只留最近 6 天（快取不要越長越大）
txo = sorted(f for f in os.listdir("data/raw_txo") if f.endswith(".csv"))
for f in txo[:-6]:
    os.remove(os.path.join("data/raw_txo", f))
print("\n完成：", rep["date"], flush=True)
