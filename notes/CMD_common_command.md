# Windows CMD 常用指令整理

> 適用於 Windows 命令提示字元 (cmd.exe)  
> 包含檔案總管相關指令（`explorer.exe`）

---

## 一、基本操作

| 指令 | 說明 | 範例 |
| ------ | ------ | ------ |
| `help` | 顯示所有指令說明 | `help` |
| `help 指令` | 顯示特定指令說明 | `help dir` |
| `cls` | 清除畫面 | `cls` |
| `exit` | 關閉 CMD | `exit` |
| `title` | 設定視窗標題 | `title 我的CMD` |
| `color` | 變更文字與背景顏色 | `color 0A`（黑底綠字） |
| `prompt` | 自訂提示符號 | `prompt $P$G` |

---

## 二、檔案與資料夾操作

| 指令 | 說明 | 範例 |
| ------ | ------ | ------ |
| `dir` | 列出目錄內容 | `dir` / `dir /w` / `dir /a` |
| `cd` | 切換目錄 | `cd Documents` / `cd ..` / `cd \` |
| `md` 或 `mkdir` | 建立資料夾 | `md 新資料夾` |
| `rd` 或 `rmdir` | 刪除空資料夾 | `rd 空資料夾` |
| `rd /s /q` | 強制刪除資料夾及其內容 | `rd /s /q 資料夾` |
| `copy` | 複製檔案 | `copy a.txt b.txt` |
| `xcopy` | 複製資料夾（含內容） | `xcopy 來源 目標 /e /i` |
| `robocopy` | 強大的複製工具 | `robocopy 來源 目標 /e` |
| `move` | 移動或重新命名 | `move a.txt D:\` |
| `ren` 或 `rename` | 重新命名 | `ren old.txt new.txt` |
| `del` | 刪除檔案 | `del *.tmp` / `del /f /q 檔案` |
| `type` | 顯示文字檔內容 | `type readme.txt` |
| `more` | 分頁顯示內容 | `type 大檔案.txt \| more` |

---

## 三、檔案總管相關（explorer.exe）

| 指令 | 說明 | 範例 |
| ------ | ------ | ------ |
| `explorer` | 開啟檔案總管 | `explorer` |
| `explorer .` | 開啟目前目錄 | `explorer .` |
| `explorer ..` | 開啟上一層目錄 | `explorer ..` |
| `explorer C:\` | 開啟指定磁碟機 | `explorer C:\` |
| `explorer C:\Users` | 開啟指定資料夾 | `explorer C:\Users` |
| `explorer /select,檔案路徑` | 開啟並選取特定檔案 | `explorer /select,C:\test.txt` |
| `explorer /e,路徑` | 以「資料夾樹」模式開啟 | `explorer /e,C:\Windows` |
| `explorer shell:RecycleBinFolder` | 開啟資源回收筒 | `explorer shell:RecycleBinFolder` |
| `explorer shell:Downloads` | 開啟下載資料夾 | `explorer shell:Downloads` |
| `explorer shell:Desktop` | 開啟桌面 | `explorer shell:Desktop` |
| `explorer shell:MyComputerFolder` | 開啟「此電腦」 | `explorer shell:MyComputerFolder` |

### 常用 shell: 路徑

```sh
explorer shell:Desktop
explorer shell:Downloads
explorer shell:Documents
explorer shell:Pictures
explorer shell:Music
explorer shell:Videos
explorer shell:RecycleBinFolder
explorer shell:Startup
explorer shell:SendTo
explorer shell:Fonts
explorer shell:AppData
```

---

## 四、系統與網路相關

| 指令 | 說明 | 範例 |
| ------ | ------ | ------ |
| `ipconfig` | 顯示 IP 設定 | `ipconfig` / `ipconfig /all` |
| `ipconfig /release` | 釋放 IP | `ipconfig /release` |
| `ipconfig /renew` | 重新取得 IP | `ipconfig /renew` |
| `ipconfig /flushdns` | 清除 DNS 快取 | `ipconfig /flushdns` |
| `ping` | 測試連線 | `ping google.com` / `ping -t 8.8.8.8` |
| `tracert` | 路由追蹤 | `tracert google.com` |
| `netstat` | 顯示網路連線狀態 | `netstat -ano` |
| `nslookup` | DNS 查詢 | `nslookup google.com` |
| `systeminfo` | 顯示系統詳細資訊 | `systeminfo` |
| `hostname` | 顯示電腦名稱 | `hostname` |
| `whoami` | 顯示目前使用者 | `whoami` |
| `tasklist` | 顯示執行中的程序 | `tasklist` |
| `taskkill` | 結束程序 | `taskkill /im notepad.exe` / `taskkill /pid 1234 /f` |
| `shutdown` | 關機、重新啟動 | `shutdown /s /t 0`（立即關機）<br>`shutdown /r /t 0`（立即重啟）<br>`shutdown /a`（取消關機） |

---

## 五、磁碟與磁碟機

| 指令 | 說明 | 範例 |
| ------ | ------ | ------ |
| `chkdsk` | 檢查磁碟 | `chkdsk C:` / `chkdsk C: /f` |
| `diskpart` | 磁碟分割工具 | `diskpart`（進入後再操作） |
| `format` | 格式化磁碟 | `format E: /fs:NTFS`（小心使用！） |
| `label` | 變更磁碟機標籤 | `label C: 系統碟` |
| `vol` | 顯示磁碟區序號與標籤 | `vol C:` |
| `fsutil` | 檔案系統工具 | `fsutil volume diskfree C:` |

---

## 六、實用技巧與組合指令

### 1. 快速開啟常用位置

```cmd
explorer shell:Downloads
explorer shell:Desktop
explorer %USERPROFILE%
explorer %APPDATA%
```

### 2. 批次重新命名（範例）

```cmd
ren *.txt *.bak
```

### 3. 找出佔用特定 Port 的程序

```cmd
netstat -ano | findstr :8080
```

### 4. 強制結束程序

```cmd
taskkill /f /im chrome.exe
```

### 5. 顯示隱藏檔案與系統檔

```cmd
dir /a
```

### 6. 建立文字檔並寫入內容

```cmd
echo 這是內容 > test.txt
echo 第二行 >> test.txt
```

### 7. 查看環境變數

```cmd
echo %PATH%
set
```

### 8. 以系統管理員權限執行（需在系統管理員 CMD 中）

```cmd
net session
```

---

## 七、常用快捷鍵（在 CMD 視窗中）

| 快捷鍵 | 功能 |
| -------- | ------ |
| `↑` / `↓` | 瀏覽歷史指令 |
| `Tab` | 自動完成檔名 / 資料夾名 |
| `Ctrl + C` | 中斷目前執行的指令 |
| `Ctrl + V` 或滑鼠右鍵 | 貼上 |
| `F7` | 顯示指令歷史清單 |
| `Alt + Enter` | 全螢幕切換 |

---

## 注意事項

1. 部分指令（如 `format`、`diskpart`、`rd /s /q`）具有破壞性，請小心使用。
2. 建議以「系統管理員」身分執行 CMD，才能使用完整功能。
3. Windows 10/11 建議改用 **Windows Terminal** 或 **PowerShell**，功能更強大。
4. `explorer.exe` 其實就是檔案總管的主程式，直接打 `explorer` 即可。

---

**檔案建立時間**：2026-08-26  
**適用系統**：Windows 10 / 11
