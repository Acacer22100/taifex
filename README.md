# 台指選擇權機率盤

用期交所盤後結算價，反推台指週選市場隱含的到期落點機率；對照同到期小台、期貨價量描述統計，並試算價差期望值。研究用途，不構成投資建議。

## 檔案

- `index.html`：網頁本體（單一檔案，含計算引擎與 10/08 範例資料）
- `report_latest.json`：每日盤後報告（由 Colab 產生或推送，網頁開啟時自動讀取）

## 上傳到 GitHub Pages

1. 在 GitHub 建立新的 repository（例如 `taifex`）。免費帳號的 GitHub Pages 需要 **Public** repo，網站與報告任何人都看得到。
2. 在 repo 頁面按 **Add file → Upload files**，上傳 `index.html` 與 `README.md`，按 Commit changes。
3. **Settings → Pages**：Source 選「Deploy from a branch」，Branch 選 `main`、資料夾 `/ (root)`，按 Save。
4. 約 1～2 分鐘後，網址是 `https://你的帳號.github.io/taifex/`。

## 每天更新報告

- **自動**：在 Colab 筆記本第 7 步填入 `GITHUB_REPO` 並設定 `GITHUB_TOKEN`，每天執行時會自動把 `report_latest.json` 推送到這個 repo。
- **手動**：把雲端硬碟 `taifex/report_latest.json` 上傳到 repo（覆蓋舊檔），或直接拖進網頁的檔案區。

## 與 Claude 版的差別

在 Claude 裡打開的版本可以直接讀你的 Google 雲端硬碟；GitHub 版讀不到雲端硬碟，改為讀取同一個 repo 裡的 `report_latest.json`。
