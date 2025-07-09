import asyncio
import json
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import datetime
import sqlite3
import uuid
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import aiohttp
import psutil
import torch
from cryptography.fernet import Fernet
from 設定檔 import SQLite資料夾, 市場清單, 訓練參數, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, 記錄參數異動, 訓練設備
from 工具模組 import 錯誤記錄與自動修復, 監控硬體狀態並降級, validate_utility_input
from 推播通知模組 import 發送錯誤訊息, 發送通知
from 交易環境模組 import multi_market_trading_env
from 測試網模組 import 測試網交易
from 獎勵計算模組 import calculate_multi_market_reward
from 績效分析模組 import parameter_sensitivity_analysis

# 配置日誌
log_dir = SQLite資料夾.parent / "日誌"
log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("最佳參數推送模組")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
file_handler = TimedRotatingFileHandler(
    filename=log_dir / "best_parameters_push_logs",
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
    "E801": "最佳參數推送失敗",
    "E802": "參數穩定性驗證失敗",
    "E803": "真倉模擬失敗",
    "E804": "參數加密失敗"
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

async def encrypt_params(params):
    """加密參數"""
    try:
        key_path = SQLite資料夾.parent / "secure_key.key"
        if not key_path.exists():
            key = Fernet.generate_key()
            with open(key_path, "wb") as key_file:
                key_file.write(key)
                
        with open(key_path, "rb") as key_file:
            key = key_file.read()
        cipher = Fernet(key)
        encrypted_params = cipher.encrypt(json.dumps(params, ensure_ascii=False).encode('utf-8'))
        
        logger.info("[資安] 參數加密完成")
        await push_limiter.cache_message("[執行通知] 參數加密完成", "多市場", "多框架", "最佳參數推送", "normal")
        await push_limiter.retry_cached_messages()
        
        with open(log_dir / f"資安事件_{datetime.datetime.now().strftime('%Y%m%d')}.log", "a", encoding='utf-8') as f:
            f.write(f"[資安事件] 參數加密\n時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        return encrypted_params
    except Exception as e:
        logger.error(f"[資安] 參數加密失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E804：參數加密失敗 {e}", "多市場", "多框架", "最佳參數推送", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E804：參數加密失敗 {e}", "參數加密錯誤", "多市場", "多框架", "最佳參數推送")
        return None

async def validate_best_parameters(best_params, market, timeframe):
    """驗證最佳參數格式與範圍"""
    try:
        required_keys = [
            "learning_rate", "dropout", "batch_size", "model_type", "window",
            "sma_period", "hma_period", "atr_period", "vhf_period", "pivot_period",
            "stop_loss", "take_profit", "trailing_stop", "trailing_take_profit",
            "breakeven_trigger", "single_loss_limit", "daily_loss_limit"
        ]
        if not await validate_utility_input(market, timeframe, mode="最佳參數推送"):
            return False

        if not all(key in best_params for key in required_keys):
            logger.error(f"[{market}_{timeframe}] 缺少必要參數: {required_keys}")
            await push_limiter.cache_message(f"錯誤碼E801：缺少必要參數 {required_keys}", market, timeframe, "最佳參數推送", "high")
            await push_limiter.retry_cached_messages()
            return False

        ranges = {
            "learning_rate": (1e-5, 1e-2),
            "dropout": (0.1, 0.5),
            "batch_size": [32, 64, 128, 256],
            "sma_period": (10, 200),
            "hma_period": (8, 30),
            "atr_period": (7, 30),
            "vhf_period": (14, 100),
            "pivot_period": (3, 20),
            "stop_loss": (0.01, 0.05),
            "take_profit": (0.02, 0.1),
            "trailing_stop": (0.01, 0.05),
            "trailing_take_profit": (0.02, 0.1),
            "breakeven_trigger": (0.01, 0.05),
            "single_loss_limit": (0.01, 0.03),
            "daily_loss_limit": (0.0, 0.1)
        }

        for key, value in best_params.items():
            if key in ranges:
                if isinstance(ranges[key], list):
                    if value not in ranges[key]:
                        logger.error(f"[{market}_{timeframe}] 無效參數 {key}: {value}")
                        await push_limiter.cache_message(f"錯誤碼E801：無效參數 {key}: {value}", market, timeframe, "最佳參數推送", "high")
                        await push_limiter.retry_cached_messages()
                        return False
                else:
                    min_val, max_val = ranges[key]
                    if not (min_val <= value <= max_val):
                        logger.error(f"[{market}_{timeframe}] 無效參數 {key}: {value}")
                        await push_limiter.cache_message(f"錯誤碼E801：無效參數 {key}: {value}", market, timeframe, "最佳參數推送", "high")
                        await push_limiter.retry_cached_messages()
                        return False

        return True
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 參數驗證失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E801：參數驗證失敗 {e}", market, timeframe, "最佳參數推送", "high")
        await push_limiter.retry_cached_messages()
        return False

async def verify_parameter_stability(params_history, market, timeframe):
    """驗證參數穩定性"""
    try:
        async with aiosqlite.connect(SQLite資料夾 / "檢查點紀錄.db") as conn:
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_市場_時間框架 ON 檢查點記錄 (市場, 時間框架, 時間)")
            df = await conn.execute_fetchall("SELECT 參數 FROM 檢查點記錄 WHERE 市場 = ? AND 時間框架 = ? ORDER BY 時間 DESC LIMIT 10",
                                            (market, timeframe))
            historical_params = [json.loads(row[0]) for row in df]

        stability_data = []
        for param_name in params_history.keys():
            values = [p.get(param_name, 0.0) for p in historical_params if param_name in p]
            if len(values) > 1:
                std = np.std(values)
                mean = np.mean(values)
                stability_data.append({
                    "參數名稱": param_name,
                    "平均值": mean,
                    "標準差": std
                })

        if not stability_data:
            logger.error(f"[{market}_{timeframe}] 無有效穩定性數據")
            await push_limiter.cache_message(f"錯誤碼E802：無有效穩定性數據", market, timeframe, "最佳參數推送", "high")
            await push_limiter.retry_cached_messages()
            return False

        df = pd.DataFrame(stability_data)
        unstable_params = df[df["標準差"] > 0.1]
        if not unstable_params.empty:
            logger.warning(f"[{market}_{timeframe}] 參數不穩定: {unstable_params.to_dict()}")
            await push_limiter.cache_message(f"錯誤碼E802：參數不穩定 {unstable_params.to_dict()}", market, timeframe, "最佳參數推送", "high")
            await push_limiter.retry_cached_messages()
            return False

        # 動態調整歷史數據量
        history_limit = 10 if len(historical_params) > 10 else len(historical_params)
        fig = go.Figure()
        for _, row in df.iterrows():
            fig.add_trace(go.Scatter(
                x=[market + "_" + timeframe],
                y=[row["標準差"]],
                mode="markers",
                name=row["參數名稱"]
            ))
        fig.update_layout(
            title=f"{market}_{timeframe} 參數穩定性趨勢",
            template="plotly_dark",
            xaxis_title="市場_時間框架",
            yaxis_title="標準差",
            height=600
        )
        plot_path = SQLite資料夾.parent / "圖片" / f"{market}_{timeframe}_Parameter_Stability_Trend_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(plot_path)
        await push_limiter.cache_message(f"【執行通知】生成參數穩定性圖表 {plot_path}", market, timeframe, "最佳參數推送", "normal")
        await push_limiter.retry_cached_messages()

        return True
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 參數穩定性驗證失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E802：參數穩定性驗證失敗 {e}", market, timeframe, "最佳參數推送", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E802：參數穩定性驗證失敗 {e}", "穩定性驗證錯誤", market, timeframe, "最佳參數推送")
        return False

async def simulate_real_trading(best_params, market_signal_mapping, retry_count=0):
    """模擬真倉交易，涵蓋多市場並優先驗證BTCUSDT_15m"""
    try:
        # 並行處理多市場模擬
        tasks = []
        for (market, timeframe) in 市場清單:
            signal_data = market_signal_mapping.get((market, timeframe), {"信號": [0], "價格": [0]})
            task = multi_market_trading_env(
                {(market, timeframe): signal_data},
                資產類型="虛擬貨幣" if market in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "BNBUSDT"] else "CFD",
                params_list={(market, timeframe): best_params}
            )
            tasks.append(task)
        multi_results_list = await asyncio.gather(*tasks)
        multi_results = {mkt_tf: result[0] for mkt_tf, result in zip(市場清單, multi_results_list) if result[0]}

        if not multi_results:
            logger.error("[多市場_多框架] 多市場模擬失敗")
            await push_limiter.cache_message(f"錯誤碼E803：多市場模擬失敗", "多市場", "多框架", "最佳參數推送", "high")
            await push_limiter.retry_cached_messages()
            return None, 0.0

        # 計算加權績效
        performance_data = []
        total_sharpe = 0.0
        valid_markets = 0
        for (market, timeframe), result in multi_results.items():
            if result is None:
                continue
            returns = [record["損益"] / 1000.0 for record in result["交易記錄"] if "損益" in record]
            sharpe = result.get("sharpe", 0.0)
            max_drawdown = result.get("最大回撤", 0.0)
            if sharpe < 1.5 or max_drawdown > 0.25:
                logger.warning(f"[{market}_{timeframe}] 績效未達標: Sharpe={sharpe:.2f}, Max Drawdown={max_drawdown:.2%}")
                await push_limiter.cache_message(f"錯誤碼E803：績效未達標 Sharpe={sharpe:.2f}, Max Drawdown={max_drawdown:.2%}", market, timeframe, "最佳參數推送", "high")
                await push_limiter.retry_cached_messages()
                continue
            weight = 0.5 if market == "BTCUSDT" and timeframe == "15m" else 0.5 / (len(市場清單) - 1)
            total_sharpe += sharpe * weight
            valid_markets += 1
            performance_data.append({
                "市場": market,
                "時間框架": timeframe,
                "報酬率": (result["最終資金"] - 1000.0) / 1000.0,
                "Sharpe": sharpe,
                "最大回撤": max_drawdown,
                "穩定性": np.std(returns) if returns else 0.0
            })

        # 避免過擬合：正則化懲罰
        sensitivity_data = await parameter_sensitivity_analysis(multi_results, {mkt: best_params for mkt in 市場清單})
        if sensitivity_data:
            df_sensitivity = pd.DataFrame(sensitivity_data)
            param_variability = df_sensitivity.groupby("參數名稱")["穩定性"].mean()
            if any(param_variability > 0.1):
                logger.warning(f"[多市場_多框架] 參數變異度過高，可能過擬合: {param_variability.to_dict()}")
                await push_limiter.cache_message(f"錯誤碼E803：參數變異度過高 {param_variability.to_dict()}", "多市場", "多框架", "最佳參數推送", "high")
                await push_limiter.retry_cached_messages()

        # BTCUSDT_15m 專屬驗證
        btc_result, btc_reward = await 測試網交易(
            params=best_params,
            市場="BTCUSDT",
            時間框架="15m",
            信號=market_signal_mapping.get(("BTCUSDT", "15m"), {}).get("信號", [0])[-1],
            價格=market_signal_mapping.get(("BTCUSDT", "15m"), {}).get("價格", [0])[-1]
        )
        if btc_result:
            btc_sharpe = btc_result.get("sharpe", 0.0)
            btc_max_drawdown = btc_result.get("最大回撤", 0.0)
            if btc_sharpe >= 1.5 and btc_max_drawdown <= 0.25:
                performance_data.append({
                    "市場": "BTCUSDT",
                    "時間框架": "15m",
                    "報酬率": (btc_result["最終資金"] - 1000.0) / 1000.0,
                    "Sharpe": btc_sharpe,
                    "最大回撤": btc_max_drawdown,
                    "穩定性": btc_result.get("穩定性", 0.0)
                })
            else:
                logger.warning(f"[BTCUSDT_15m] 測試網未達標: Sharpe={btc_sharpe:.2f}, Max Drawdown={btc_max_drawdown:.2%}")
                await push_limiter.cache_message(f"錯誤碼E803：BTCUSDT_15m 未達標 Sharpe={btc_sharpe:.2f}, Max Drawdown={btc_max_drawdown:.2%}", "BTCUSDT", "15m", "最佳參數推送", "high")
                await push_limiter.retry_cached_messages()

        # 生成多市場模擬報表
        df = pd.DataFrame(performance_data)
        if not df.empty:
            csv_path = SQLite資料夾.parent / "備份" / f"Multi_Market_Simulation_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            await push_limiter.cache_message(f"【執行通知】生成多市場模擬報表 {csv_path}", "多市場", "多框架", "最佳參數推送", "normal")
            await push_limiter.retry_cached_messages()

        return performance_data, total_sharpe / valid_markets if valid_markets > 0 else 0.0
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[多市場_多框架] 真倉模擬失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E803：真倉模擬失敗 重試{retry_count + 1}/5: {e}", "真倉模擬錯誤", "多市場", "多框架", "最佳參數推送")
            await asyncio.sleep(5)
            return await simulate_real_trading(best_params, market_signal_mapping, retry_count + 1)
        logger.error(f"[多市場_多框架] 真倉模擬失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E803：真倉模擬失敗 {e}", "多市場", "多框架", "最佳參數推送", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E803：真倉模擬失敗 {e}", "真倉模擬錯誤", "多市場", "多框架", "最佳參數推送")
        return None, 0.0

async def push_best_parameters(best_params, best_value, market_signal_mapping, prev_best_value=0.0, retry_count=0):
    """推送最佳參數至檢查點與Telegram"""
    try:
        import time
        start_time = time.time()

        # 硬體監控
        batch_size, _ = await 監控硬體狀態並降級(best_params.get("batch_size", 32), 2)
        if batch_size < 8:
            logger.error(f"[多市場_多框架] 硬體資源超限，批次大小 {batch_size}")
            await push_limiter.cache_message(f"錯誤碼E801：硬體資源超限，批次大小 {batch_size}", "多市場", "多框架", "最佳參數推送", "high")
            await push_limiter.retry_cached_messages()
            return False

        # 驗證最佳參數
        for market, timeframe in 市場清單:
            if not await validate_best_parameters(best_params, market, timeframe):
                return False

        # 參數穩定性驗證
        params_history = {key: [best_params[key]] for key in best_params}
        for market, timeframe in 市場清單:
            if not await verify_parameter_stability(params_history, market, timeframe):
                return False

        # 真倉模擬驗證
        performance_data, weighted_sharpe = await simulate_real_trading(best_params, market_signal_mapping)
        if performance_data is None:
            return False

        # 動態推送（獎勵變化 >0.1）
        if abs(best_value - prev_best_value) < 0.1:
            logger.info(f"[多市場_多框架] 獎勵變化 {abs(best_value - prev_best_value):.4f} 小於0.1，跳過推送")
            return False

        # 加密參數（GPU加速）
        if torch.cuda.is_available():
            params_tensor = torch.tensor(list(json.dumps(best_params, ensure_ascii=False).encode('utf-8')), device=訓練設備)
            key_path = SQLite資料夾.parent / "secure_key.key"
            if not key_path.exists():
                key = Fernet.generate_key()
                with open(key_path, "wb") as key_file:
                    key_file.write(key)
            with open(key_path, "rb") as key_file:
                key = key_file.read()
            cipher = Fernet(key)
            encrypted_params = cipher.encrypt(params_tensor.cpu().numpy().tobytes())
        else:
            key_path = SQLite資料夾.parent / "secure_key.key"
            if not key_path.exists():
                key = Fernet.generate_key()
                with open(key_path, "wb") as key_file:
                    key_file.write(key)
            with open(key_path, "rb") as key_file:
                key = key_file.read()
            cipher = Fernet(key)
            encrypted_params = cipher.encrypt(json.dumps(best_params, ensure_ascii=False).encode('utf-8'))
        if not encrypted_params:
            return False

        logger.info("[資安] 參數加密完成")
        await push_limiter.cache_message("[執行通知] 參數加密完成", "多市場", "多框架", "最佳參數推送", "normal")
        await push_limiter.retry_cached_messages()
        
        with open(log_dir / f"資安事件_{datetime.datetime.now().strftime('%Y%m%d')}.log", "a", encoding='utf-8') as f:
            f.write(f"[資安事件] 參數加密\n時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 儲存最佳參數至檢查點
        async with aiosqlite.connect(SQLite資料夾 / "檢查點紀錄.db") as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS 最佳參數記錄 (
                    id TEXT PRIMARY KEY,
                    市場 TEXT,
                    時間框架 TEXT,
                    參數 BLOB,
                    獎勵 REAL,
                    時間 TEXT
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_市場_時間框架 ON 最佳參數記錄 (市場, 時間框架, 時間)")
            await conn.execute("INSERT INTO 最佳參數記錄 VALUES (?, ?, ?, ?, ?, ?)",
                              (str(uuid.uuid4()), "多市場", "多框架", encrypted_params,
                               best_value, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            await conn.commit()

        # 寫入異動歷程
        change_log_path = SQLite資料夾.parent / "dot" / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        change_log = {
            "UUID": str(uuid.uuid4()),
            "市場": "多市場",
            "時間框架": "多框架",
            "異動前值": f"prev_reward={prev_best_value:.4f}",
            "異動後值": f"reward={best_value:.4f}, params={json.dumps(best_params, ensure_ascii=False)}",
            "異動原因": "最佳參數推送",
            "時間戳": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        change_df = pd.DataFrame([change_log])
        if change_log_path.exists():
            existing_df = pd.read_excel(change_log_path)
            change_df = pd.concat([existing_df, change_df], ignore_index=True)
        change_df.to_excel(change_log_path, index=False, engine='openpyxl')

        # Telegram 推送
        message = (
            f"【最佳參數推送】\n"
            f"市場: 多市場\n"
            f"時間框架: 多框架\n"
            f"獎勵: {best_value:.4f}\n"
            f"加權Sharpe: {weighted_sharpe:.4f}\n"
            f"參數: {json.dumps({k: v for k, v in best_params.items() if k != 'model_type'}, ensure_ascii=False)}"
        )
        async with aiohttp.ClientSession() as session:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    logger.error("[多市場_多框架] Telegram推送失敗")
                    await push_limiter.cache_message(f"錯誤碼E801：Telegram推送失敗", "多市場", "多框架", "最佳參數推送", "high")
                    await push_limiter.retry_cached_messages()
                    return False

        # 生成報表
        df = pd.DataFrame([{
            "市場": "多市場",
            "時間框架": "多框架",
            "參數": json.dumps(best_params, ensure_ascii=False),
            "獎勵": best_value,
            "加權Sharpe": weighted_sharpe
        }])
        csv_path = SQLite資料夾.parent / "備份" / f"Multi_Market_Best_Parameters_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        await push_limiter.cache_message(f"【執行通知】生成最佳參數報表 {csv_path}", "多市場", "多框架", "最佳參數推送", "normal")
        await push_limiter.retry_cached_messages()

        # 真倉適用性日誌
        async with aiosqlite.connect(SQLite資料夾 / "push_log.db") as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS 推送記錄 (
                    id TEXT PRIMARY KEY,
                    市場 TEXT,
                    時間框架 TEXT,
                    參數 BLOB,
                    獎勵 REAL,
                    加權Sharpe REAL,
                    時間 TEXT
                )
            """)
            await conn.execute("INSERT INTO 推送記錄 VALUES (?, ?, ?, ?, ?, ?, ?)",
                              (str(uuid.uuid4()), "多市場", "多框架", encrypted_params,
                               best_value, weighted_sharpe, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            await conn.commit()

        # 效率報告
        elapsed_time = time.time() - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】最佳參數推送耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, "多市場", "多框架", "最佳參數推送", "normal")
        await push_limiter.retry_cached_messages()

        return True
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[多市場_多框架] 最佳參數推送失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E801：最佳參數推送失敗 重試{retry_count + 1}/5: {e}", "推送錯誤", "多市場", "多框架", "最佳參數推送")
            await asyncio.sleep(5)
            return await push_best_parameters(best_params, best_value, market_signal_mapping, prev_best_value, retry_count + 1)
        logger.error(f"[多市場_多框架] 最佳參數推送失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E801：最佳參數推送失敗 {e}", "多市場", "多框架", "最佳參數推送", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E801：最佳參數推送失敗 {e}", "推送錯誤", "多市場", "多框架", "最佳參數推送")
        return False