import pandas as pd
import aiosqlite
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import json
from cryptography.fernet import Fernet
from 設定檔 import SQLite資料夾, 市場清單
from 工具模組 import 錯誤記錄與自動修復, 監控硬體狀態並降級, validate_utility_input

# 配置日誌
log_dir = SQLite資料夾.parent / "日誌"
log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("搜尋結果查詢與繪圖分析模組")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
file_handler = TimedRotatingFileHandler(
    filename=log_dir / "search_results_logs",
    when="midnight",
    backupCount=30,
    encoding='utf-8'
)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# 錯誤碼定義
ERROR_CODES = {
    "E1001": "查詢失敗",
    "E1002": "繪圖分析失敗",
    "E1003": "硬體資源超限",
    "E1004": "數據加密失敗"
}

# 推播限制器
class PushLimiter:
    def __init__(self, max_pushes_per_minute=10):
        self.max_pushes = max_pushes_per_minute
        self.push_timestamps = deque(maxlen=60)
        self.cache_db = SQLite資料夾 / "push_cache.db"

    async def can_push(self):
        current_time = datetime.datetime.now()
        self.push_timestamps = deque([t for t in self.push_timestamps if (current_time - t).seconds < 60], maxlen=60)
        return len(self.push_timestamps) < self.max_pushes

    async def cache_message(self, message, market, timeframe, mode, priority="normal"):
        async with aiosqlite.connect(self.cache_db) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS push_cache (
                    id TEXT PRIMARY KEY,
                    市場 TEXT,
                    時間框架 TEXT,
                    模式 TEXT,
                    訊息 TEXT,
                    優先級 TEXT,
                    時間 TEXT
                )
            """)
            await conn.execute("INSERT INTO push_cache VALUES (?, ?, ?, ?, ?, ?, ?)",
                              (str(uuid.uuid4()), market, timeframe, mode, message, priority,
                               datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            await conn.commit()

    async def retry_cached_messages(self):
        async with aiosqlite.connect(self.cache_db) as conn:
            messages = await conn.execute_fetchall("SELECT 市場, 時間框架, 模式, 訊息, 優先級 FROM push_cache ORDER BY 優先級 DESC")
            for market, timeframe, mode, message, priority in messages:
                if await self.can_push():
                    await 發送通知(message, market, timeframe, mode)
                    await conn.execute("DELETE FROM push_cache WHERE 訊息 = ?", (message,))
                    await conn.commit()
                    self.push_timestamps.append(datetime.datetime.now())

push_limiter = PushLimiter()

async def encrypt_data(data):
    """加密查詢數據"""
    try:
        key_path = SQLite資料夾.parent / "secure_key.key"
        if not key_path.exists():
            key = Fernet.generate_key()
            with open(key_path, "wb") as key_file:
                key_file.write(key)
                
        with open(key_path, "rb") as key_file:
            key = key_file.read()
        cipher = Fernet(key)
        encrypted_data = cipher.encrypt(json.dumps(data, ensure_ascii=False).encode('utf-8'))
        
        logger.info("[資安] 查詢數據加密完成")
        await push_limiter.cache_message("[執行通知] 查詢數據加密完成", "多市場", "多框架", "搜尋結果", "normal")
        await push_limiter.retry_cached_messages()
        
        with open(log_dir / f"資安事件_{datetime.datetime.now().strftime('%Y%m%d')}.log", "a", encoding='utf-8') as f:
            f.write(f"[資安事件] 查詢數據加密\n時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        return encrypted_data
    except Exception as e:
        logger.error(f"[資安] 數據加密失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E1004：數據加密失敗 {e}", "多市場", "多框架", "搜尋結果", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E1004：數據加密失敗 {e}", "數據加密錯誤", "多市場", "多框架", "搜尋結果")
        return None

async def multi_market_search(limit=100, filter_market=None, filter_timeframe=None, custom_conditions=None, retry_count=0):
    """整合多市場查詢結果"""
    try:
        import time
        start_time = time.time()
        
        # 硬體監控
        batch_size, _ = await 監控硬體狀態並降級(32, 2)
        if batch_size < 8:
            logger.error(f"[多市場_多框架] 硬體資源超限，批次大小 {batch_size}")
            await push_limiter.cache_message(f"錯誤碼E1003：硬體資源超限，批次大小 {batch_size}", "多市場", "多框架", "搜尋結果", "high")
            await push_limiter.retry_cached_messages()
            return None

        # 查詢數據
        async with aiosqlite.connect(SQLite資料夾 / "performance_log.db") as conn:
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_市場_時間框架 ON 績效記錄 (市場, 時間框架, 時間)")
            query = "SELECT 市場, 時間框架, 最終資金, 報酬率, f1分數, sharpe, 最大回撤, 穩定性 FROM 績效記錄"
            params = []
            if filter_market or filter_timeframe or custom_conditions:
                query += " WHERE 1=1"
                if filter_market:
                    query += " AND 市場 = ?"
                    params.append(filter_market)
                if filter_timeframe:
                    query += " AND 時間框架 = ?"
                    params.append(filter_timeframe)
                if custom_conditions:
                    for cond in custom_conditions:
                        query += f" AND {cond}"
            query += " ORDER BY 時間 DESC LIMIT ?"
            params.append(limit * len(市場清單))
            df_performance = pd.DataFrame(await conn.execute_fetchall(query, params),
                                        columns=["市場", "時間框架", "最終資金", "報酬率", "f1分數", "sharpe", "最大回撤", "穩定性"])

        async with aiosqlite.connect(SQLite資料夾 / "訓練紀錄.db") as conn:
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_市場_時間框架 ON 訓練進度 (市場, 時間框架, 時間)")
            query = "SELECT 市場, 時間框架, 平均獎勵 AS 獎勵 FROM 訓練進度"
            params = []
            if filter_market or filter_timeframe or custom_conditions:
                query += " WHERE 1=1"
                if filter_market:
                    query += " AND 市場 = ?"
                    params.append(filter_market)
                if filter_timeframe:
                    query += " AND 時間框架 = ?"
                    params.append(filter_timeframe)
                if custom_conditions:
                    for cond in custom_conditions:
                        query += f" AND {cond}"
            query += " ORDER BY 時間 DESC LIMIT ?"
            params.append(limit * len(市場清單))
            df_training = pd.DataFrame(await conn.execute_fetchall(query, params),
                                     columns=["市場", "時間框架", "獎勵"])

        async with aiosqlite.connect(SQLite資料夾 / "檢查點紀錄.db") as conn:
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_市場_時間框架 ON 檢查點記錄 (市場, 時間框架, 時間)")
            query = "SELECT 市場, 時間框架, 參數 FROM 檢查點記錄"
            params = []
            if filter_market or filter_timeframe or custom_conditions:
                query += " WHERE 1=1"
                if filter_market:
                    query += " AND 市場 = ?"
                    params.append(filter_market)
                if filter_timeframe:
                    query += " AND 時間框架 = ?"
                    params.append(filter_timeframe)
                if custom_conditions:
                    for cond in custom_conditions:
                        query += f" AND {cond}"
            query += " ORDER BY 時間 DESC LIMIT ?"
            params.append(limit * len(市場清單))
            df_checkpoint = pd.DataFrame(await conn.execute_fetchall(query, params),
                                       columns=["市場", "時間框架", "參數"])

        # 合併數據
        df = df_performance.merge(df_training, on=["市場", "時間框架"], how="outer")
        df = df.merge(df_checkpoint, on=["市場", "時間框架"], how="outer")
        df = df[["市場", "時間框架", "報酬率", "f1分數", "sharpe", "最大回撤", "穩定性", "獎勵", "參數"]].dropna(subset=["報酬率", "f1分數", "sharpe", "最大回撤", "穩定性"])

        if df.empty:
            logger.error("[多市場_多框架] 無有效查詢數據")
            await push_limiter.cache_message("錯誤碼E1001：無有效查詢數據", "多市場", "多框架", "搜尋結果", "high")
            await push_limiter.retry_cached_messages()
            return None

        # 快取查詢結果
        cache_path = SQLite資料夾.parent / "cache" / f"search_cache_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez_compressed(cache_path, data=df.to_dict())
        await push_limiter.cache_message(f"【執行通知】查詢結果已快取至 {cache_path}", "多市場", "多框架", "搜尋結果", "normal")
        await push_limiter.retry_cached_messages()

        # 簡化報表
        df_report = df[["市場", "時間框架", "報酬率", "f1分數", "sharpe", "最大回撤", "穩定性"]]
        csv_path = SQLite資料夾.parent / "備份" / f"Multi_Market_Search_Results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_report.to_csv(csv_path, index=False, encoding='utf-8-sig')
        await push_limiter.cache_message(f"【執行通知】生成多市場查詢報表 {csv_path}", "多市場", "多框架", "搜尋結果", "normal")
        await push_limiter.retry_cached_messages()

        # 寫入異動歷程
        change_log_path = SQLite資料夾.parent / "dot" / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        change_log = {
            "UUID": str(uuid.uuid4()),
            "市場": "多市場",
            "時間框架": "多框架",
            "異動前值": "N/A",
            "異動後值": f"查詢記錄數={len(df)}",
            "異動原因": "多市場查詢",
            "時間戳": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        change_df = pd.DataFrame([change_log])
        if change_log_path.exists():
            existing_df = pd.read_excel(change_log_path)
            change_df = pd.concat([existing_df, change_df], ignore_index=True)
        change_df.to_excel(change_log_path, index=False, engine='openpyxl')

        return df
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[多市場_多框架] 查詢失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E1001：查詢失敗 重試{retry_count + 1}/5: {e}", "查詢錯誤", "多市場", "多框架", "搜尋結果")
            await asyncio.sleep(5)
            return await multi_market_search(limit, filter_market, filter_timeframe, custom_conditions, retry_count + 1)
        logger.error(f"[多市場_多框架] 查詢失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E1001：查詢失敗 {e}", "多市場", "多框架", "搜尋結果", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E1001：查詢失敗 {e}", "查詢錯誤", "多市場", "多框架", "搜尋結果")
        return None

async def long_term_trend_analysis(limit=100, filter_market=None, filter_timeframe=None, custom_conditions=None, retry_count=0):
    """長期趨勢分析"""
    try:
        import time
        start_time = time.time()
        
        # 硬體監控
        batch_size, _ = await 監控硬體狀態並降級(32, 2)
        if batch_size < 8:
            logger.error(f"[多市場_多框架] 硬體資源超限，批次大小 {batch_size}")
            await push_limiter.cache_message(f"錯誤碼E1003：硬體資源超限，批次大小 {batch_size}", "多市場", "多框架", "搜尋結果", "high")
            await push_limiter.retry_cached_messages()
            return None

        async with aiosqlite.connect(SQLite資料夾 / "performance_log.db") as conn:
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_市場_時間框架 ON 績效記錄 (市場, 時間框架, 時間)")
            query = "SELECT 市場, 時間框架, 報酬率, f1分數, sharpe, 最大回撤, 穩定性, 時間 FROM 績效記錄"
            params = []
            if filter_market or filter_timeframe or custom_conditions:
                query += " WHERE 1=1"
                if filter_market:
                    query += " AND 市場 = ?"
                    params.append(filter_market)
                if filter_timeframe:
                    query += " AND 時間框架 = ?"
                    params.append(filter_timeframe)
                if custom_conditions:
                    for cond in custom_conditions:
                        query += f" AND {cond}"
            query += " ORDER BY 時間 DESC LIMIT ?"
            params.append(limit * len(市場清單))
            df = pd.DataFrame(await conn.execute_fetchall(query, params),
                             columns=["市場", "時間框架", "報酬率", "f1分數", "sharpe", "最大回撤", "穩定性", "時間"])

        if df.empty:
            logger.error("[多市場_多框架] 無有效長期趨勢數據")
            await push_limiter.cache_message("錯誤碼E1002：無有效長期趨勢數據", "多市場", "多框架", "搜尋結果", "high")
            await push_limiter.retry_cached_messages()
            return None

        # 檢查穩定性
        grouped = df.groupby(["市場", "時間框架"]).agg({"報酬率": ["mean", "std"], "sharpe": "mean", "最大回撤": "mean", "穩定性": "mean"})
        grouped.columns = ["報酬率平均", "報酬率標準差", "sharpe平均", "最大回撤平均", "穩定性平均"]
        grouped = grouped.reset_index()
        weighted_std = 0.0
        total_weight = 0.0
        for _, row in grouped.iterrows():
            weight = 0.5 if row["市場"] == "BTCUSDT" and row["時間框架"] == "15m" else 0.5 / (len(市場清單) - 1)
            if row["sharpe平均"] < 1.5 or row["最大回撤平均"] > 0.25 or row["報酬率標準差"] > 0.1:
                logger.warning(f"[{row['市場']}_{row['時間框架']}] 穩定性未達標: Sharpe={row['sharpe平均']:.2f}, 最大回撤={row['最大回撤平均']:.2%}, 標準差={row['報酬率標準差']:.4f}")
                await push_limiter.cache_message(
                    f"錯誤碼E1002：穩定性未達標 {row['市場']}_{row['時間框架']} Sharpe={row['sharpe平均']:.2f}, 最大回撤={row['最大回撤平均']:.2%}, 標準差={row['報酬率標準差']:.4f}",
                    row["市場"], row["時間框架"], "搜尋結果", "high")
                await push_limiter.retry_cached_messages()
            weighted_std += row["報酬率標準差"] * weight
            total_weight += weight
        weighted_std = weighted_std / total_weight if total_weight > 0 else 0.0
        if weighted_std > 0.1:
            logger.warning(f"[多市場_多框架] 長期績效穩定性低，加權標準差: {weighted_std:.4f}")
            await push_limiter.cache_message(f"錯誤碼E1002：長期績效穩定性低，加權標準差 {weighted_std:.4f}", "多市場", "多框架", "搜尋結果", "high")
            await push_limiter.retry_cached_messages()
            # 暫停流程
            logger.info("[多市場_多框架] 穩定性低，暫停流程")
            await push_limiter.cache_message("【執行通知】穩定性低，暫停流程", "多市場", "多框架", "搜尋結果", "high")
            await push_limiter.retry_cached_messages()
            return None

        # 生成雙軸趨勢圖表（動態調整）
        fig = go.Figure()
        for (market, timeframe) in 市場清單:
            subset = df[(df["市場"] == market) & (df["時間框架"] == timeframe)]
            if not subset.empty:
                fig.add_trace(go.Scatter(
                    x=subset["時間"],
                    y=subset["報酬率"] * 100,
                    mode="lines+markers",
                    name=f"{market}_{timeframe} 報酬率",
                    line=dict(color="green" if subset["報酬率"].mean() > 0 else "red")
                ))
                fig.add_trace(go.Scatter(
                    x=subset["時間"],
                    y=subset["sharpe"],
                    mode="lines+markers",
                    name=f"{market}_{timeframe} Sharpe",
                    yaxis="y2",
                    line=dict(color="blue" if subset["sharpe"].mean() > 1.5 else "orange")
                ))
        fig.update_layout(
            title="多市場長期績效與Sharpe趨勢",
            xaxis_title="時間",
            yaxis_title="報酬率 (%)",
            yaxis2=dict(title="Sharpe", overlaying="y", side="right"),
            template="plotly_dark",
            hovermode="x unified",
            height=600,
            showlegend=True,
            updatemenus=[
                dict(
                    buttons=[
                        dict(label="全選", method="update", args=[{"visible": [True] * len(fig.data)}]),
                        dict(label="僅報酬率", method="update", args=[{"visible": [i % 2 == 0 for i in range(len(fig.data))]}]),
                        dict(label="僅Sharpe", method="update", args=[{"visible": [i % 2 == 1 for i in range(len(fig.data))]}])
                    ],
                    direction="down",
                    showactive=True
                )
            ]
        )
        plot_path = SQLite資料夾.parent / "圖片" / f"Long_Term_Search_Trend_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(plot_path)
        await push_limiter.cache_message(f"【執行通知】生成長期趨勢圖表 {plot_path}", "多市場", "多框架", "搜尋結果", "normal")
        await push_limiter.retry_cached_messages()

        # 寫入異動歷程
        change_log_path = SQLite資料夾.parent / "dot" / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        change_log = {
            "UUID": str(uuid.uuid4()),
            "市場": "多市場",
            "時間框架": "多框架",
            "異動前值": "N/A",
            "異動後值": f"圖表生成={plot_path}, 記錄數={len(df)}",
            "異動原因": "長期趨勢分析",
            "時間戳": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        change_df = pd.DataFrame([change_log])
        if change_log_path.exists():
            existing_df = pd.read_excel(change_log_path)
            change_df = pd.concat([existing_df, change_df], ignore_index=True)
        change_df.to_excel(change_log_path, index=False, engine='openpyxl')

        return grouped
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[多市場_多框架] 長期趨勢分析失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E1002：長期趨勢分析失敗 重試{retry_count + 1}/5: {e}", "趨勢分析錯誤", "多市場", "多框架", "搜尋結果")
            await asyncio.sleep(5)
            return await long_term_trend_analysis(limit, filter_market, filter_timeframe, custom_conditions, retry_count + 1)
        logger.error(f"[多市場_多框架] 長期趨勢分析失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E1002：長期趨勢分析失敗 {e}", "多市場", "多框架", "搜尋結果", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E1002：長期趨勢分析失敗 {e}", "趨勢分析錯誤", "多市場", "多框架", "搜尋結果")
        return None

async def parameter_impact_visualization(df, retry_count=0):
    """參數對績效影響視覺化"""
    try:
        import time
        start_time = time.time()
        
        if df is None or df.empty:
            logger.error("[多市場_多框架] 無有效參數影響數據")
            await push_limiter.cache_message("錯誤碼E1002：無有效參數影響數據", "多市場", "多框架", "搜尋結果", "high")
            await push_limiter.retry_cached_messages()
            return None

        # 提取參數與績效
        impact_data = []
        for _, row in df.iterrows():
            params = json.loads(row["參數"]) if isinstance(row["參數"], str) else row["參數"]
            weight = 0.5 if row["市場"] == "BTCUSDT" and row["時間框架"] == "15m" else 0.5 / (len(市場清單) - 1)
            for param_name, param_value in params.items():
                impact_data.append({
                    "市場": row["市場"],
                    "時間框架": row["時間框架"],
                    "參數名稱": param_name,
                    "參數值": param_value,
                    "報酬率": row["報酬率"] * weight,
                    "sharpe": row["sharpe"] * weight,
                    "最大回撤": row["最大回撤"] * weight,
                    "穩定性": row["穩定性"] * weight
                })

        df_impact = pd.DataFrame(impact_data)
        pivot = df_impact.pivot_table(values=["報酬率", "sharpe"], index="參數名稱", columns="參數值", aggfunc="mean")
        fig = go.Figure()
        for metric in ["報酬率", "sharpe"]:
            fig.add_trace(go.Heatmap(
                z=pivot[metric].values,
                x=pivot[metric].columns,
                y=pivot[metric].index,
                colorscale="Viridis" if metric == "報酬率" else "Plasma",
                name=metric,
                showscale=True
            ))
        fig.update_layout(
            title="參數對報酬率與Sharpe影響熱圖",
            xaxis_title="參數值",
            yaxis_title="參數名稱",
            template="plotly_dark",
            height=600,
            updatemenus=[
                dict(
                    buttons=[
                        dict(label="全選", method="update", args=[{"visible": [True] * len(fig.data)}]),
                        dict(label="僅報酬率", method="update", args=[{"visible": [i == 0 for i in range(len(fig.data))]}]),
                        dict(label="僅Sharpe", method="update", args=[{"visible": [i == 1 for i in range(len(fig.data))]}])
                    ],
                    direction="down",
                    showactive=True
                )
            ]
        )
        heatmap_path = SQLite資料夾.parent / "圖片" / f"Parameter_Impact_Heatmap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(heatmap_path)
        await push_limiter.cache_message(f"【執行通知】生成參數影響熱圖 {heatmap_path}", "多市場", "多框架", "搜尋結果", "normal")
        await push_limiter.retry_cached_messages()

        # 加密並儲存影響數據
        encrypted_data = await encrypt_data(df_impact.to_dict())
        if not encrypted_data:
            return None
        async with aiosqlite.connect(SQLite資料夾 / "performance_log.db") as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS 參數影響 (
                    id TEXT PRIMARY KEY,
                    市場 TEXT,
                    時間框架 TEXT,
                    參數名稱 TEXT,
                    參數值 REAL,
                    報酬率 REAL,
                    sharpe REAL,
                    最大回撤 REAL,
                    穩定性 REAL,
                    時間 TEXT
                )
            """)
            for _, row in df_impact.iterrows():
                await conn.execute("INSERT INTO 參數影響 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                 (str(uuid.uuid4()), row["市場"], row["時間框架"], row["參數名稱"],
                                  row["參數值"], row["報酬率"], row["sharpe"], row["最大回撤"], row["穩定性"],
                                  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            await conn.commit()

        # 寫入異動歷程
        change_log_path = SQLite資料夾.parent / "dot" / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        change_log = {
            "UUID": str(uuid.uuid4()),
            "市場": "多市場",
            "時間框架": "多框架",
            "異動前值": "N/A",
            "異動後值": f"熱圖生成={heatmap_path}, 記錄數={len(df_impact)}",
            "異動原因": "參數影響視覺化",
            "時間戳": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        change_df = pd.DataFrame([change_log])
        if change_log_path.exists():
            existing_df = pd.read_excel(change_log_path)
            change_df = pd.concat([existing_df, change_df], ignore_index=True)
        change_df.to_excel(change_log_path, index=False, engine='openpyxl')

        # 效率報告
        elapsed_time = time.time() - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = 0.0  # 無GPU操作
        efficiency_report = (
            f"【效率報告】參數影響視覺化耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, "多市場", "多框架", "搜尋結果", "normal")
        await push_limiter.retry_cached_messages()

        return df_impact
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[多市場_多框架] 參數影響視覺化失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E1002：參數影響視覺化失敗 重試{retry_count + 1}/5: {e}", "視覺化錯誤", "多市場", "多框架", "搜尋結果")
            await asyncio.sleep(5)
            return await parameter_impact_visualization(df, retry_count + 1)
        logger.error(f"[多市場_多框架] 參數影響視覺化失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E1002：參數影響視覺化失敗 {e}", "多市場", "多框架", "搜尋結果", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E1002：參數影響視覺化失敗 {e}", "視覺化錯誤", "多市場", "多框架", "搜尋結果")
        return None

async def main_search_and_plot(retry_count=0):
    """主搜尋與繪圖函數"""
    try:
        import time
        import psutil
        start_time = time.time()

        # 硬體監控
        batch_size, _ = await 監控硬體狀態並降級(32, 2)
        if batch_size < 8:
            logger.error(f"[多市場_多框架] 硬體資源超限，批次大小 {batch_size}")
            await push_limiter.cache_message(f"錯誤碼E1003：硬體資源超限，批次大小 {batch_size}", "多市場", "多框架", "搜尋結果", "high")
            await push_limiter.retry_cached_messages()
            return None

        # 多市場查詢
        df = await multi_market_search()
        if df is None:
            return None

        # 長期趨勢分析
        trend_data = await long_term_trend_analysis()
        if trend_data is None:
            return None

        # 參數影響視覺化
        impact_data = await parameter_impact_visualization(df)
        if impact_data is None:
            return None

        # 效率報告
        elapsed_time = time.time() - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = 0.0  # 無GPU操作
        efficiency_report = (
            f"【效率報告】搜尋與繪圖耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, "多市場", "多框架", "搜尋結果", "normal")
        await push_limiter.retry_cached_messages()

        return {"search_data": df, "trend_data": trend_data, "impact_data": impact_data}
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[多市場_多框架] 主搜尋與繪圖失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E1001：主搜尋與繪圖失敗 重試{retry_count + 1}/5: {e}", "主搜尋錯誤", "多市場", "多框架", "搜尋結果")
            await asyncio.sleep(5)
            return await main_search_and_plot(retry_count + 1)
        logger.error(f"[多市場_多框架] 主搜尋與繪圖失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E1001：主搜尋與繪圖失敗 {e}", "多市場", "多框架", "搜尋結果", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E1001：主搜尋與繪圖失敗 {e}", "主搜尋錯誤", "多市場", "多框架", "搜尋結果")
        return None