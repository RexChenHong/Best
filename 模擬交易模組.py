import numpy as np
import torch
import time
import datetime
import logging
import uuid
import asyncio
import optuna
import pandas as pd
import aiosqlite
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from collections import deque
from cryptography.fernet import Fernet
from 設定檔 import 訓練設備, 快取資料夾, 槓桿比例, 手續費率, 最小下單手數, 點差, 點值, 訓練參數
from 工具模組 import 檢查交易時間, 記錄持倉狀態, 錯誤記錄與自動修復, 監控硬體狀態並降級, validate_utility_input
from 推播通知模組 import 發送下單通知, 發送倉位通知, 發送錯誤訊息, 發送通知
from 獎勵計算模組 import calculate_multi_market_reward

# 配置日誌
log_dir = 快取資料夾.parent / "日誌"
log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("模擬交易模組")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
file_handler = TimedRotatingFileHandler(
    filename=log_dir / "sim_trade_logs",
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
    "E101": "無效信號",
    "E103": "硬體資源超限",
    "E104": "風險參數優化失敗",
    "E105": "數據加密失敗"
}

# 推播限制器
class PushLimiter:
    def __init__(self, max_pushes_per_minute=10):
        self.max_pushes = max_pushes_per_minute
        self.push_timestamps = deque(maxlen=60)
        self.cache_db = 快取資料夾.parent / "SQLite" / "push_cache.db"

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

async def encrypt_trading_records(records):
    """加密交易記錄"""
    try:
        key_path = 快取資料夾.parent / "secure_key.key"
        if not key_path.exists():
            key = Fernet.generate_key()
            with open(key_path, "wb") as key_file:
                key_file.write(key)
                
        with open(key_path, "rb") as key_file:
            key = key_file.read()
        cipher = Fernet(key)
        encrypted_records = cipher.encrypt(json.dumps(records, ensure_ascii=False).encode('utf-8'))
        
        logger.info("[資安] 交易記錄加密完成")
        await push_limiter.cache_message("[執行通知] 交易記錄加密完成", "多市場", "多框架", "模擬交易", "normal")
        await push_limiter.retry_cached_messages()
        
        with open(log_dir / f"資安事件_{datetime.datetime.now().strftime('%Y%m%d')}.log", "a", encoding='utf-8') as f:
            f.write(f"[資安事件] 交易記錄加密\n時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        return encrypted_records
    except Exception as e:
        logger.error(f"[資安] 交易記錄加密失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E105：交易記錄加密失敗 {e}", "多市場", "多框架", "模擬交易", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E105：交易記錄加密失敗 {e}", "數據加密錯誤", "多市場", "多框架", "模擬交易")
        return None

async def validate_input(信號, 價格, 資產類型, 市場, 時間框架):
    """驗證輸入信號、價格與參數"""
    try:
        if not await validate_utility_input(市場, 時間框架, mode="模擬交易"):
            return False
        if not isinstance(信號, (list, np.ndarray, torch.Tensor)) or not all(s in [1.0, -1.0, 0.0] for s in np.array(信號).flatten()):
            logger.error(f"[{市場}_{時間框架}] 無效信號: {信號}")
            await push_limiter.cache_message(f"錯誤碼E101：無效信號 {信號}", 市場, 時間框架, "模擬交易", "high")
            await push_limiter.retry_cached_messages()
            return False
        if not isinstance(價格, (list, np.ndarray, torch.Tensor)) or not all(p > 0 for p in np.array(價格).flatten()):
            logger.error(f"[{市場}_{時間框架}] 無效價格: {價格}")
            await push_limiter.cache_message(f"錯誤碼E101：無效價格 {價格}", 市場, 時間框架, "模擬交易", "high")
            await push_limiter.retry_cached_messages()
            return False
        if 資產類型 not in ["CFD", "虛擬貨幣"]:
            logger.error(f"[{市場}_{時間框架}] 無效資產類型: {資產類型}")
            await push_limiter.cache_message(f"錯誤碼E101：無效資產類型 {資產類型}", 市場, 時間框架, "模擬交易", "high")
            await push_limiter.retry_cached_messages()
            return False
        if 市場 not in 槓桿比例:
            logger.error(f"[{市場}_{時間框架}] 無效市場: {市場}")
            await push_limiter.cache_message(f"錯誤碼E101：無效市場 {市場}", 市場, 時間框架, "模擬交易", "high")
            await push_limiter.retry_cached_messages()
            return False
        return True
    except Exception as e:
        logger.error(f"[{市場}_{時間框架}] 輸入驗證失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E101：輸入驗證失敗 {e}", 市場, 時間框架, "模擬交易", "high")
        await push_limiter.retry_cached_messages()
        return False

async def check_reverse_trade_violation(trading_records, market, timeframe):
    """檢查同K棒反手違規"""
    try:
        k_bar_timestamps = {}
        for record in trading_records:
            timestamp = record.get("時間", "")
            signal_strength = record.get("信號強度", 0.0)
            if timestamp:
                if timestamp in k_bar_timestamps:
                    k_bar_timestamps[timestamp] += 1
                    if k_bar_timestamps[timestamp] > 1 and signal_strength <= 0.9:
                        logger.warning(f"[{market}_{timeframe}] 同K棒多次反手違規: {timestamp}")
                        await push_limiter.cache_message(
                            f"錯誤碼E101：同K棒多次反手違規 時間={timestamp}, 信號強度={signal_strength:.2f}",
                            market, timeframe, "模擬交易", "high")
                        await push_limiter.retry_cached_messages()
                        async with aiosqlite.connect(快取資料夾.parent / "SQLite" / "risk_violation_log.db") as conn:
                            await conn.execute("""
                                CREATE TABLE IF NOT EXISTS 風險違規記錄 (
                                    id TEXT PRIMARY KEY,
                                    市場 TEXT,
                                    時間框架 TEXT,
                                    違規內容 TEXT,
                                    時間 TEXT
                                )
                            """)
                            await conn.execute("INSERT INTO 風險違規記錄 VALUES (?, ?, ?, ?, ?)",
                                             (str(uuid.uuid4()), market, timeframe, f"同K棒多次反手: {timestamp}",
                                              datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                            await conn.commit()
                        return False
                else:
                    k_bar_timestamps[timestamp] = 1
        return True
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 同K棒反手檢查失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E101：同K棒反手檢查失敗 {e}", market, timeframe, "模擬交易", "high")
        await push_limiter.retry_cached_messages()
        return False

async def optimize_risk_parameters(信號, 價格, 資產類型, 市場, 時間框架, batch_size, initial_params, retry_count=0):
    """使用 Optuna 優化風險參數"""
    async def objective(trial):
        try:
            params = initial_params.copy()
            params["single_loss_limit"] = trial.suggest_float("single_loss_limit", 0.01, 0.03)
            params["daily_loss_limit"] = trial.suggest_float("daily_loss_limit", 0.03, 0.07)
            params["stop_loss"] = trial.suggest_float("stop_loss", 0.01, 0.05)
            params["take_profit"] = trial.suggest_float("take_profit", 0.02, 0.1)
            params["trailing_stop"] = trial.suggest_float("trailing_stop", 0.01, 0.05)
            params["trailing_take_profit"] = trial.suggest_float("trailing_take_profit", 0.02, 0.1)
            params["breakeven_trigger"] = trial.suggest_float("breakeven_trigger", 0.01, 0.05)
            params["signal_threshold"] = trial.suggest_float("signal_threshold", 0.5, 0.95)

            # 模擬交易
            result = await 模擬交易(
                信號=信號,
                價格=價格,
                資產類型=資產類型,
                市場=市場,
                時間框架=時間框架,
                strong_signals=[params["signal_threshold"]],
                batch_size=1,  # 限定單一資金
                停損=params["stop_loss"],
                停利=params["take_profit"],
                平本觸發=params["breakeven_trigger"],
                移動停損=params["trailing_stop"],
                移動停利=params["trailing_take_profit"],
                單筆損失限制=params["single_loss_limit"],
                單日損失限制=params["daily_loss_limit"]
            )
            
            if result is None or result.get(市場) is None:
                return 0.0
            
            # 計算獎勵
            reward, _ = await calculate_multi_market_reward({(市場, 時間框架): result[市場]}, {(市場, 時間框架): params})
            return reward
        except Exception as e:
            logger.error(f"[{市場}_{時間框架}] 風險參數優化失敗: {e}")
            await push_limiter.cache_message(f"錯誤碼E104：風險參數優化失敗 {e}", 市場, 時間框架, "模擬交易", "high")
            await push_limiter.retry_cached_messages()
            return 0.0

    try:
        study = optuna.create_study(direction="maximize")
        if torch.cuda.is_available():
            study.optimize(lambda trial: asyncio.run(objective(trial)), n_trials=50, n_jobs=1)
        else:
            study.optimize(lambda trial: asyncio.run(objective(trial)), n_trials=50, n_jobs=1)
        best_params = initial_params.copy()
        best_params.update(study.best_params)
        
        # 加密並儲存優化結果
        encrypted_params = await encrypt_trading_records(best_params)
        if not encrypted_params:
            return initial_params
        async with aiosqlite.connect(快取資料夾.parent / "SQLite" / "風險優化記錄.db") as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS 風險優化記錄 (
                    id TEXT PRIMARY KEY,
                    市場 TEXT,
                    時間框架 TEXT,
                    風險參數 BLOB,
                    獎勵 REAL,
                    時間 TEXT
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_市場_時間框架 ON 風險優化記錄 (市場, 時間框架, 時間)")
            await conn.execute("INSERT INTO 風險優化記錄 VALUES (?, ?, ?, ?, ?, ?)",
                             (str(uuid.uuid4()), 市場, 時間框架, encrypted_params, study.best_value,
                              datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            await conn.commit()

        # 檢查穩定性
        async with aiosqlite.connect(快取資料夾.parent / "SQLite" / "風險優化記錄.db") as conn:
            df = await conn.execute_fetchall(
                "SELECT 風險參數 FROM 風險優化記錄 WHERE 市場 = ? AND 時間框架 = ? ORDER BY 時間 DESC LIMIT 10",
                (市場, 時間框架))
            historical_params = []
            key_path = 快取資料夾.parent / "secure_key.key"
            if key_path.exists():
                with open(key_path, "rb") as key_file:
                    cipher = Fernet(key_file.read())
                for row in df:
                    decrypted = cipher.decrypt(row[0]).decode('utf-8')
                    historical_params.append(json.loads(decrypted))
            if len(historical_params) > 1:
                df_params = pd.DataFrame(historical_params)
                std_single = df_params["single_loss_limit"].std()
                std_daily = df_params["daily_loss_limit"].std()
                if std_single > 0.1 or std_daily > 0.1:
                    logger.warning(f"[{市場}_{時間框架}] 風險參數穩定性低: 單筆={std_single:.4f}, 單日={std_daily:.4f}")
                    await push_limiter.cache_message(
                        f"錯誤碼E104：風險參數穩定性低 單筆={std_single:.4f}, 單日={std_daily:.4f}",
                        市場, 時間框架, "模擬交易", "high")
                    await push_limiter.retry_cached_messages()

        # 推播優化結果
        message = (
            f"【風險參數優化完成】\n"
            f"市場: {市場}_{時間框架}\n"
            f"獎勵: {study.best_value:.4f}\n"
            f"單筆損失限制: {best_params['single_loss_limit']:.4f}\n"
            f"單日損失限制: {best_params['daily_loss_limit']:.4f}\n"
            f"停損: {best_params['stop_loss']:.4f}\n"
            f"停利: {best_params['take_profit']:.4f}\n"
            f"移動停損: {best_params['trailing_stop']:.4f}\n"
            f"移動停利: {best_params['trailing_take_profit']:.4f}\n"
            f"平本觸發: {best_params['breakeven_trigger']:.4f}\n"
            f"信號閾值: {best_params['signal_threshold']:.4f}"
        )
        await push_limiter.cache_message(message, 市場, 時間框架, "模擬交易", "normal")
        await push_limiter.retry_cached_messages()
        return best_params
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[{市場}_{時間框架}] 風險參數優化失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E104：風險參數優化失敗 重試{retry_count + 1}/5: {e}", "風險參數優化錯誤", 市場, 時間框架, "模擬交易")
            await asyncio.sleep(5)
            return await optimize_risk_parameters(信號, 價格, 資產類型, 市場, 時間框架, batch_size, initial_params, retry_count + 1)
        logger.error(f"[{市場}_{時間框架}] 風險參數優化失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E104：風險參數優化失敗 {e}", 市場, 時間框架, "模擬交易", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E104：風險參數優化失敗 {e}", "風險參數優化錯誤", 市場, 時間框架, "模擬交易")
        return initial_params

async def handle_strong_signal(signal, position, price, avg_price, leverage, fee_rate, funds, trading_records, position_time, last_close_time, daily_loss, consecutive_losses, stop_loss_time, market, timeframe, asset_type, point_spread, single_loss_limit, daily_loss_limit, trailing_stop, trailing_take_profit, breakeven_trigger, signal_threshold):
    """處理強信號（>signal_threshold）即時反手邏輯"""
    try:
        current_time = time.time()
        trade_mask = (signal.abs() >= signal_threshold) & ((current_time - last_close_time >= 2.0) | (position == 0))
        if not trade_mask:
            return position, avg_price, funds, trading_records, position_time, last_close_time, daily_loss, consecutive_losses, stop_loss_time

        if position != 0:  # 平倉
            quantity = position
            close_price = price + point_spread if quantity > 0 else price - point_spread
            fee = abs(quantity) * close_price * fee_rate * 2
            profit = (close_price - avg_price) * quantity * leverage - fee if quantity > 0 else (avg_price - close_price) * quantity * leverage - fee
            if profit < 0 and abs(profit) >= 1000.0 * single_loss_limit:
                logger.error(f"[{market}_{timeframe}] 單筆損失超限: {profit:.2f}")
                await push_limiter.cache_message(f"錯誤碼E101：單筆損失超限: {profit:.2f}", market, timeframe, "模擬交易", "high")
                await push_limiter.retry_cached_messages()
                return position, avg_price, funds, trading_records, position_time, last_close_time, daily_loss, consecutive_losses, stop_loss_time
            funds += profit
            if profit < 0:
                daily_loss += abs(profit)
                consecutive_losses += 1
            else:
                consecutive_losses = 0
            trading_records.append({
                "id": str(uuid.uuid4()),
                "市場": market,
                "時間框架": timeframe,
                "類型": "強信號平倉",
                "價格": close_price.item(),
                "數量": quantity.item(),
                "損益": profit.item(),
                "時間": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "持倉時間": position_time.item(),
                "信號強度": signal.item()
            })
            await push_limiter.cache_message(
                f"【下單通知】強信號平倉: {market}_{timeframe}, 價格={close_price.item():.2f}, 數量={quantity.item():.2f}, 損益={profit.item():.2f}",
                market, timeframe, "模擬交易", "normal")
            await push_limiter.retry_cached_messages()
            position = 0
            avg_price = 0.0
            position_time = 0
            last_close_time = current_time

        # 開新倉
        quantity = 最小下單手數.get(market, 最小下單手數["default"]) if asset_type == "CFD" else (funds * 0.1 * leverage) / price
        quantity = quantity * signal.sign()
        fee = abs(quantity) * (price + point_spread if quantity > 0 else price - point_spread) * fee_rate * 2
        funds -= fee
        position = quantity
        avg_price = price
        trading_records.append({
            "id": str(uuid.uuid4()),
            "市場": market,
            "時間框架": timeframe,
            "類型": "強信號開倉",
            "價格": price.item(),
            "數量": quantity.item(),
            "損益": 0.0,
            "時間": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "持倉時間": 0,
            "信號強度": signal.item()
        })
        await push_limiter.cache_message(
            f"【下單通知】強信號開倉: {market}_{timeframe}, 價格={price.item():.2f}, 數量={quantity.item():.2f}, 資金={funds:.2f}",
            market, timeframe, "模擬交易", "normal")
        await push_limiter.retry_cached_messages()
        return position, avg_price, funds, trading_records, position_time, last_close_time, daily_loss, consecutive_losses, stop_loss_time
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 強信號處理失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E101：強信號處理失敗 {e}", market, timeframe, "模擬交易", "high")
        await push_limiter.retry_cached_messages()
        return position, avg_price, funds, trading_records, position_time, last_close_time, daily_loss, consecutive_losses, stop_loss_time

async def 模擬交易(
    信號,
    價格,
    資產類型,
    市場,
    時間框架,
    strong_signals=None,
    batch_size=1,  # 限定單一資金
    停損=0.02,
    停利=0.03,
    平本觸發=0.01,
    移動停損=0.01,
    移動停利=0.02,
    單筆損失限制=0.02,
    單日損失限制=0.05,
    retry_count=0
):
    """模擬單一市場交易，獨立資金管理，動態優化風險參數，初始資金1000 USDT/USD"""
    try:
        import time
        start_time = time.time()
        # 驗證輸入
        if not await validate_input(信號, 價格, 資產類型, 市場, 時間框架):
            logger.warning(f"[{市場}_{時間框架}] 無效輸入，跳過交易")
            return None

        # 動態優化風險參數
        initial_params = {
            "stop_loss": 停損,
            "take_profit": 停利,
            "breakeven_trigger": 平本觸發,
            "trailing_stop": 移動停損,
            "trailing_take_profit": 移動停利,
            "single_loss_limit": 單筆損失限制,
            "daily_loss_limit": 單日損失限制,
            "signal_threshold": strong_signals[0] if strong_signals else 0.9
        }
        optimized_params = await optimize_risk_parameters(信號, 價格, 資產類型, 市場, 時間框架, batch_size, initial_params)
        單筆損失限制 = optimized_params["single_loss_limit"]
        單日損失限制 = optimized_params["daily_loss_limit"]
        停損 = optimized_params["stop_loss"]
        停利 = optimized_params["take_profit"]
        移動停損 = optimized_params["trailing_stop"]
        移動停利 = optimized_params["trailing_take_profit"]
        平本觸發 = optimized_params["breakeven_trigger"]
        signal_threshold = optimized_params["signal_threshold"]

        # 查表取得參數 (槓桿固定不變)
        槓桿 = 槓桿比例.get(市場, 槓桿比例["default"] if 資產類型 == "CFD" else 槓桿比例["default_crypto"])
        手續費 = 手續費率.get(市場, 手續費率["default"] if 資產類型 == "CFD" else 手續費率["default_crypto"])
        訂單手數 = 最小下單手數.get(市場, 最小下單手數["default"]) if 資產類型 == "CFD" else None
        該點差 = 點差.get(市場, 點差["default"])
        該點值 = 點值.get(市場, 點值["default"])

        # 硬體監控與降級
        batch_size, _ = await 監控硬體狀態並降級(batch_size, 2)
        if batch_size < 1:  # 限定單一資金，batch_size最小為1
            logger.error(f"[{市場}_{時間框架}] 硬體資源超限，無法執行模擬")
            await push_limiter.cache_message(f"錯誤碼E103：硬體資源超限，批次大小 {batch_size}", 市場, 時間框架, "模擬交易", "high")
            await push_limiter.retry_cached_messages()
            return None

        # 初始化環境，單一資金1000 USDT/USD
        with torch.cuda.amp.autocast():
            初始資金 = 1000.0
            資金 = torch.tensor([初始資金], dtype=torch.float32, device=訓練設備)  # 單一資金
            持倉數量 = torch.tensor([0.0], dtype=torch.float32, device=訓練設備)
            平均入場價格 = torch.tensor([0.0], dtype=torch.float32, device=訓練設備)
            未實現盈虧 = torch.tensor([0.0], dtype=torch.float32, device=訓練設備)
            持倉時間 = torch.tensor([0], dtype=torch.int32, device=訓練設備)
            最後平倉時間 = torch.tensor([0.0], dtype=torch.float32, device=訓練設備)
            巔峰資金 = 資金.clone()
            最大回撤 = torch.tensor([0.0], dtype=torch.float32, device=訓練設備)
            交易記錄 = [[]]
            完成 = torch.tensor([False], dtype=torch.bool, device=訓練設備)
            日損失 = torch.tensor([0.0], dtype=torch.float32, device=訓練設備)
            連續虧損次數 = torch.tensor([0], dtype=torch.int32, device=訓練設備)
            停損時間 = torch.tensor([0.0], dtype=torch.float32, device=訓練設備)
            最高價格 = torch.tensor([0.0], dtype=torch.float32, device=訓練設備)
            最低價格 = torch.tensor([float('inf')], dtype=torch.float32, device=訓練設備)
            checkpoint = 0

            if 資產類型 == "CFD" and not await 檢查交易時間(市場):
                logger.warning(f"[{市場}_{時間框架}] 非交易時段，跳過模擬")
                return None

            for t in range(len(信號)):
                # 硬體檢查
                batch_size, _ = await 監控硬體狀態並降級(batch_size, 2)
                torch.cuda.empty_cache()

                # 交易時段檢查
                if 資產類型 == "CFD" and not await 檢查交易時間(市場):
                    logger.info(f"[{市場}_{時間框架}] 非交易時段，跳過交易")
                    continue

                當前價格 = torch.tensor(價格[t] if isinstance(價格, (list, np.ndarray)) else 價格, dtype=torch.float32, device=訓練設備)
                當前時間 = time.time()
                可交易_mask = (當前時間 - 最後平倉時間 >= 2.0) | (持倉數量 == 0)

                # 交易延遲測量
                交易開始時間 = time.time()

                # 持倉更新
                持倉_mask = 持倉數量 != 0
                未實現盈虧[持倉_mask] = (當前價格 - 平均入場價格[持倉_mask]) * 持倉數量[持倉_mask] * 槓桿
                持倉時間[持倉_mask] += 1
                維持率 = torch.where(
                    持倉_mask,
                    (資金 + 未實現盈虧) / (平均入場價格 * torch.abs(持倉數量) / 槓桿) * 100,
                    torch.tensor([100.0], device=訓練設備)
                )

                # 風控檢查
                if 維持率 < (50 if 資產類型 == "CFD" else 105):
                    logger.error(f"[{市場}_{時間框架}] 維持率低於 {(50 if 資產類型 == 'CFD' else 105)}%: {維持率.item():.2f}")
                    await push_limiter.cache_message(
                        f"錯誤碼E101：維持率低於{(50 if 資產類型 == 'CFD' else 105)}%: {維持率.item():.2f}, 持倉數量={持倉數量.item():.2f}, 未實現盈虧={未實現盈虧.item():.2f}",
                        市場, 時間框架, "模擬交易", "high")
                    await push_limiter.retry_cached_messages()
                    完成.fill_(True)
                    break
                巔峰資金 = torch.max(巔峰資金, 資金 + 未實現盈虧)
                最大回撤 = (巔峰資金 - (資金 + 未實現盈虧)) / 巔峰資金.clamp(min=1e-10)
                if 資金 <= 0 or 最大回撤 >= 0.25:
                    logger.error(f"[{市場}_{時間框架}] 爆倉: 資金 {資金.item():.2f}, 回撤 {最大回撤.item():.2%}")
                    await push_limiter.cache_message(
                        f"【重大異常：爆倉】市場={市場}_{時間框架}, 資金={資金.item():.2f}, 回撤={最大回撤.item():.2%}, 動作=停止流程",
                        市場, 時間框架, "模擬交易", "high")
                    await push_limiter.retry_cached_messages()
                    完成.fill_(True)
                    break
                if 日損失 >= 初始資金 * 單日損失限制:
                    logger.error(f"[{市場}_{時間框架}] 單日損失超限: {日損失.item():.2f}")
                    await push_limiter.cache_message(
                        f"錯誤碼E101：單日損失超限: {日損失.item():.2f}",
                        市場, 時間框架, "模擬交易", "high")
                    await push_limiter.retry_cached_messages()
                    完成.fill_(True)
                    break
                if 連續虧損次數 >= 3:
                    if 當前時間 - 停損時間 < 3600:
                        logger.warning(f"[{市場}_{時間框架}] 連續虧損超限，暫停交易 1 小時")
                        await push_limiter.cache_message(
                            f"錯誤碼E101：連續虧損超限，暫停交易 1 小時",
                            市場, 時間框架, "模擬交易", "high")
                        await push_limiter.retry_cached_messages()
                        完成.fill_(True)
                        break
                    停損時間 = torch.where(連續虧損次數 >= 3, torch.tensor(當前時間, device=訓練設備), 停損時間)

                # 強信號處理
                持倉數量, 平均入場價格, 資金, 交易記錄, 持倉時間, 最後平倉時間, 日損失, 連續虧損次數, 停損時間 = await handle_strong_signal(
                    torch.tensor(信號[t], dtype=torch.float32, device=訓練設備),
                    持倉數量, 當前價格, 平均入場價格, 槓桿, 手續費, 資金, 交易記錄[0],
                    持倉時間, 最後平倉時間, 日損失, 連續虧損次數, 停損時間, 市場, 時間框架, 資產類型,
                    該點差, 單筆損失限制, 單日損失限制, 移動停損, 移動停利, 平本觸發, signal_threshold
                )

                # 更新移動停損/停利
                if 持倉_mask:
                    最高價格[持倉_mask] = torch.max(最高價格[持倉_mask], 當前價格)
                    最低價格[持倉_mask] = torch.min(最低價格[持倉_mask], 當前價格)
                    if 持倉數量 > 0:  # 多單
                        if 當前價格 >= 平均入場價格 * (1 + 平本觸發):
                            停損價格 = 平均入場價格 + (當前價格 - 平均入場價格) * 移動停損
                            停利價格 = 平均入場價格 + (當前價格 - 平均入場價格) * 移動停利
                            if 當前價格 <= 停損價格:
                                成交價格 = 當前價格
                                手續費總 = 持倉數量 * 成交價格 * 手續費 * 2
                                損益 = (成交價格 - 平均入場價格) * 持倉數量 * 槓桿 - 手續費總
                                if 損益 < 0 and abs(損益) >= 初始資金 * 單筆損失限制:
                                    logger.error(f"[{市場}_{時間框架}] 移動停損單筆損失超限: {損益.item():.2f}")
                                    await push_limiter.cache_message(
                                        f"錯誤碼E101：移動停損單筆損失超限: {損益.item():.2f}",
                                        市場, 時間框架, "模擬交易", "high")
                                    await push_limiter.retry_cached_messages()
                                    完成.fill_(True)
                                    continue
                                資金 += 損益
                                if 損益 < 0:
                                    日損失 += abs(損益)
                                    連續虧損次數 += 1
                                else:
                                    連續虧損次數 = 0
                                交易記錄[0].append({
                                    "id": str(uuid.uuid4()),
                                    "市場": 市場,
                                    "時間框架": 時間框架,
                                    "類型": "移動停損平倉",
                                    "價格": 成交價格.item(),
                                    "數量": 持倉數量.item(),
                                    "損益": 損益.item(),
                                    "時間": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    "持倉時間": 持倉時間.item(),
                                    "信號強度": signal_threshold
                                })
                                await push_limiter.cache_message(
                                    f"【下單通知】移動停損平倉: {市場}_{時間框架}, 價格={成交價格.item():.2f}, 數量={持倉數量.item():.2f}, 損益={損益.item():.2f}",
                                    市場, 時間框架, "模擬交易", "normal")
                                await push_limiter.retry_cached_messages()
                                持倉數量 = 0
                                平均入場價格 = 0.0
                                未實現盈虧 = 0.0
                                持倉時間 = 0
                                最後平倉時間 = 當前時間
                                最高價格 = 0.0
                                最低價格 = float('inf')
                            elif 當前價格 >= 停利價格:
                                成交價格 = 當前價格
                                手續費總 = 持倉數量 * 成交價格 * 手續費 * 2
                                損益 = (成交價格 - 平均入場價格) * 持倉數量 * 槓桿 - 手續費總
                                資金 += 損益
                                if 損益 < 0:
                                    日損失 += abs(損益)
                                    連續虧損次數 += 1
                                else:
                                    連續虧損次數 = 0
                                交易記錄[0].append({
                                    "id": str(uuid.uuid4()),
                                    "市場": 市場,
                                    "時間框架": 時間框架,
                                    "類型": "移動停利平倉",
                                    "價格": 成交價格.item(),
                                    "數量": 持倉數量.item(),
                                    "損益": 損益.item(),
                                    "時間": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    "持倉時間": 持倉時間.item(),
                                    "信號強度": signal_threshold
                                })
                                await push_limiter.cache_message(
                                    f"【下單通知】移動停利平倉: {市場}_{時間框架}, 價格={成交價格.item():.2f}, 數量={持倉數量.item():.2f}, 損益={損益.item():.2f}",
                                    市場, 時間框架, "模擬交易", "normal")
                                await push_limiter.retry_cached_messages()
                                持倉數量 = 0
                                平均入場價格 = 0.0
                                未實現盈虧 = 0.0
                                持倉時間 = 0
                                最後平倉時間 = 當前時間
                                最高價格 = 0.0
                                最低價格 = float('inf')
                    else:  # 空單
                        if 當前價格 <= 平均入場價格 * (1 - 平本觸發):
                            停損價格 = 平均入場價格 - (平均入場價格 - 當前價格) * 移動停損
                            停利價格 = 平均入場價格 - (平均入場價格 - 當前價格) * 移動停利
                            if 當前價格 >= 停損價格:
                                成交價格 = 當前價格
                                手續費總 = abs(持倉數量) * 成交價格 * 手續費 * 2
                                損益 = (平均入場價格 - 成交價格) * abs(持倉數量) * 槓桿 - 手續費總
                                if 損益 < 0 and abs(損益) >= 初始資金 * 單筆損失限制:
                                    logger.error(f"[{市場}_{時間框架}] 移動停損單筆損失超限: {損益.item():.2f}")
                                    await push_limiter.cache_message(
                                        f"錯誤碼E101：移動停損單筆損失超限: {損益.item():.2f}",
                                        市場, 時間框架, "模擬交易", "high")
                                    await push_limiter.retry_cached_messages()
                                    完成.fill_(True)
                                    continue
                                資金 += 損益
                                if 損益 < 0:
                                    日損失 += abs(損益)
                                    連續虧損次數 += 1
                                else:
                                    連續虧損次數 = 0
                                交易記錄[0].append({
                                    "id": str(uuid.uuid4()),
                                    "市場": 市場,
                                    "時間框架": 時間框架,
                                    "類型": "移動停損平倉",
                                    "價格": 成交價格.item(),
                                    "數量": 持倉數量.item(),
                                    "損益": 損益.item(),
                                    "時間": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    "持倉時間": 持倉時間.item(),
                                    "信號強度": signal_threshold
                                })
                                await push_limiter.cache_message(
                                    f"【下單通知】移動停損平倉: {市場}_{時間框架}, 價格={成交價格.item():.2f}, 數量={持倉數量.item():.2f}, 損益={損益.item():.2f}",
                                    市場, 時間框架, "模擬交易", "normal")
                                await push_limiter.retry_cached_messages()
                                持倉數量 = 0
                                平均入場價格 = 0.0
                                未實現盈虧 = 0.0
                                持倉時間 = 0
                                最後平倉時間 = 當前時間
                                最高價格 = 0.0
                                最低價格 = float('inf')
                            elif 當前價格 <= 停利價格:
                                成交價格 = 當前價格
                                手續費總 = abs(持倉數量) * 成交價格 * 手續費 * 2
                                損益 = (平均入場價格 - 成交價格) * abs(持倉數量) * 槓桿 - 手續費總
                                資金 += 損益
                                if 損益 < 0:
                                    日損失 += abs(損益)
                                    連續虧損次數 += 1
                                else:
                                    連續虧損次數 = 0
                                交易記錄[0].append({
                                    "id": str(uuid.uuid4()),
                                    "市場": 市場,
                                    "時間框架": 時間框架,
                                    "類型": "移動停利平倉",
                                    "價格": 成交價格.item(),
                                    "數量": 持倉數量.item(),
                                    "損益": 損益.item(),
                                    "時間": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    "持倉時間": 持倉時間.item(),
                                    "信號強度": signal_threshold
                                })
                                await push_limiter.cache_message(
                                    f"【下單通知】移動停利平倉: {市場}_{時間框架}, 價格={成交價格.item():.2f}, 數量={持倉數量.item():.2f}, 損益={損益.item():.2f}",
                                    市場, 時間框架, "模擬交易", "normal")
                                await push_limiter.retry_cached_messages()
                                持倉數量 = 0
                                平均入場價格 = 0.0
                                未實現盈虧 = 0.0
                                持倉時間 = 0
                                最後平倉時間 = 當前時間
                                最高價格 = 0.0
                                最低價格 = float('inf')

                # T+1 市價交易邏輯
                if t < len(信號) - 1:
                    下一價格 = torch.tensor(價格[t + 1] if isinstance(價格, (list, np.ndarray)) else 價格, dtype=torch.float32, device=訓練設備)

                    # 買入處理
                    買入_mask = (信號[t] == 1.0) & 可交易_mask & (持倉數量 <= 0)
                    平倉_mask = 買入_mask & (持倉數量 < 0)
                    if 平倉_mask:
                        數量 = -持倉數量
                        成交價格 = 下一價格 + 該點差
                        手續費總 = 數量 * 成交價格 * 手續費 * 2
                        損益 = (平均入場價格 - 成交價格) * 數量 * 槓桿 - 手續費總
                        if 損益 < 0 and abs(損益) >= 初始資金 * 單筆損失限制:
                            logger.error(f"[{市場}_{時間框架}] 單筆損失超限: {損益.item():.2f}")
                            await push_limiter.cache_message(
                                f"錯誤碼E101：單筆損失超限: {損益.item():.2f}",
                                市場, 時間框架, "模擬交易", "high")
                            await push_limiter.retry_cached_messages()
                            完成.fill_(True)
                            continue
                        資金 += 損益
                        if 損益 < 0:
                            日損失 += abs(損益)
                            連續虧損次數 += 1
                        else:
                            連續虧損次數 = 0
                        交易記錄[0].append({
                            "id": str(uuid.uuid4()),
                            "市場": 市場,
                            "時間框架": 時間框架,
                            "類型": "買入平倉",
                            "價格": 成交價格.item(),
                            "數量": 數量.item(),
                            "損益": 損益.item(),
                            "時間": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "持倉時間": 持倉時間.item(),
                            "信號強度": signal_threshold
                        })
                        await push_limiter.cache_message(
                            f"【下單通知】買入平倉: {市場}_{時間框架}, 價格={成交價格.item():.2f}, 數量={數量.item():.2f}, 損益={損益.item():.2f}",
                            市場, 時間框架, "模擬交易", "normal")
                        await push_limiter.retry_cached_messages()
                        持倉數量 = 0
                        平均入場價格 = 0.0
                        未實現盈虧 = 0.0
                        持倉時間 = 0
                        最後平倉時間 = 當前時間
                        最高價格 = 0.0
                        最低價格 = float('inf')
                        await 記錄持倉狀態(市場, 時間框架, "模擬", 0.0, 0.0, 0.0)

                    if 買入_mask:
                        if 資產類型 == "CFD":
                            數量 = 訂單手數
                        else:
                            數量 = (資金 * 0.1 * 槓桿) / 下一價格
                        下單保證金 = 數量 * 下一價格 / 槓桿
                        可用保證金 = 資金 + 未實現盈虧 - 下單保證金
                        if 可用保證金 < 下單保證金:
                            logger.warning(f"[{市場}_{時間框架}] 可用保證金不足: {可用保證金:.2f}")
                            await push_limiter.cache_message(
                                f"錯誤碼E101：可用保證金不足: {可用保證金:.2f}",
                                市場, 時間框架, "模擬交易", "high")
                            await push_limiter.retry_cached_messages()
                            continue
                        手續費總 = 數量 * (下一價格 + 該點差) * 手續費 * 2
                        資金 -= 手續費總
                        持倉數量 = 數量
                        平均入場價格 = 下一價格
                        交易記錄[0].append({
                            "id": str(uuid.uuid4()),
                            "市場": 市場,
                            "時間框架": 時間框架,
                            "類型": "買",
                            "價格": 下一價格.item(),
                            "數量": 數量.item(),
                            "損益": 0.0,
                            "時間": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "持倉時間": 0,
                            "信號強度": signal_threshold
                        })
                        await 記錄持倉狀態(市場, 時間框架, "模擬", float(數量), float(下一價格), 0.0)
                        await push_limiter.cache_message(
                            f"【下單通知】買: {市場}_{時間框架}, 價格={下一價格.item():.2f}, 數量={數量.item():.2f}, 資金={資金.item():.2f}",
                            市場, 時間框架, "模擬交易", "normal")
                        await push_limiter.retry_cached_messages()

                    # 賣出處理
                    賣出_mask = (信號[t] == -1.0) & 可交易_mask & (持倉數量 >= 0)
                    平倉_mask = 賣出_mask & (持倉數量 > 0)
                    if 平倉_mask:
                        數量 = 持倉數量
                        成交價格 = 下一價格 - 該點差
                        手續費總 = abs(數量) * 成交價格 * 手續費 * 2
                        損益 = (成交價格 - 平均入場價格) * 數量 * 槓桿 - 手續費總
                        if 損益 < 0 and abs(損益) >= 初始資金 * 單筆損失限制:
                            logger.error(f"[{市場}_{時間框架}] 單筆損失超限: {損益.item():.2f}")
                            await push_limiter.cache_message(
                                f"錯誤碼E101：單筆損失超限: {損益.item():.2f}",
                                市場, 時間框架, "模擬交易", "high")
                            await push_limiter.retry_cached_messages()
                            完成.fill_(True)
                            continue
                        資金 += 損益
                        if 損益 < 0:
                            日損失 += abs(損益)
                            連續虧損次數 += 1
                        else:
                            連續虧損次數 = 0
                        交易記錄[0].append({
                            "id": str(uuid.uuid4()),
                            "市場": 市場,
                            "時間框架": 時間框架,
                            "類型": "賣出平倉",
                            "價格": 成交價格.item(),
                            "數量": 數量.item(),
                            "損益": 損益.item(),
                            "時間": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "持倉時間": 持倉時間.item(),
                            "信號強度": signal_threshold
                        })
                        await push_limiter.cache_message(
                            f"【下單通知】賣出平倉: {市場}_{時間框架}, 價格={成交價格.item():.2f}, 數量={數量.item():.2f}, 損益={損益.item():.2f}",
                            市場, 時間框架, "模擬交易", "normal")
                        await push_limiter.retry_cached_messages()
                        持倉數量 = 0
                        平均入場價格 = 0.0
                        未實現盈虧 = 0.0
                        持倉時間 = 0
                        最後平倉時間 = 當前時間
                        最高價格 = 0.0
                        最低價格 = float('inf')
                        await 記錄持倉狀態(市場, 時間框架, "模擬", 0.0, 0.0, 0.0)

                    if 賣出_mask:
                        if 資產類型 == "CFD":
                            數量 = -訂單手數
                        else:
                            數量 = -(資金 * 0.1 * 槓桿) / 下一價格
                        下單保證金 = abs(數量) * 下一價格 / 槓桿
                        可用保證金 = 資金 + 未實現盈虧 - 下單保證金
                        if 可用保證金 < 下單保證金:
                            logger.warning(f"[{市場}_{時間框架}] 可用保證金不足: {可用保證金:.2f}")
                            await push_limiter.cache_message(
                                f"錯誤碼E101：可用保證金不足: {可用保證金:.2f}",
                                市場, 時間框架, "模擬交易", "high")
                            await push_limiter.retry_cached_messages()
                            continue
                        手續費總 = abs(數量) * (下一價格 - 該點差) * 手續費 * 2
                        資金 -= 手續費總
                        持倉數量 = 數量
                        平均入場價格 = 下一價格
                        交易記錄[0].append({
                            "id": str(uuid.uuid4()),
                            "市場": 市場,
                            "時間框架": 時間框架,
                            "類型": "賣",
                            "價格": 下一價格.item(),
                            "數量": 數量.item(),
                            "損益": 0.0,
                            "時間": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "持倉時間": 0,
                            "信號強度": signal_threshold
                        })
                        await 記錄持倉狀態(市場, 時間框架, "模擬", float(數量), float(下一價格), 0.0)
                        await push_limiter.cache_message(
                            f"【下單通知】賣: {市場}_{時間框架}, 價格={下一價格.item():.2f}, 數量={數量.item():.2f}, 資金={資金.item():.2f}",
                            市場, 時間框架, "模擬交易", "normal")
                        await push_limiter.retry_cached_messages()

                # 停損/停利檢查
                if 持倉_mask:
                    if 持倉數量 > 0:  # 多單
                        if 當前價格 <= 平均入場價格 * (1 - 停損):
                            成交價格 = 當前價格
                            手續費總 = 持倉數量 * 成交價格 * 手續費 * 2
                            損益 = (成交價格 - 平均入場價格) * 持倉數量 * 槓桿 - 手續費總
                            if 損益 < 0 and abs(損益) >= 初始資金 * 單筆損失限制:
                                logger.error(f"[{市場}_{時間框架}] 停損單筆損失超限: {損益.item():.2f}")
                                await push_limiter.cache_message(
                                    f"錯誤碼E101：停損單筆損失超限: {損益.item():.2f}",
                                    市場, 時間框架, "模擬交易", "high")
                                await push_limiter.retry_cached_messages()
                                完成.fill_(True)
                                continue
                            資金 += 損益
                            if 損益 < 0:
                                日損失 += abs(損益)
                                連續虧損次數 += 1
                            else:
                                連續虧損次數 = 0
                            交易記錄[0].append({
                                "id": str(uuid.uuid4()),
                                "市場": 市場,
                                "時間框架": 時間框架,
                                "類型": "停損平倉",
                                "價格": 成交價格.item(),
                                "數量": 持倉數量.item(),
                                "損益": 損益.item(),
                                "時間": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                "持倉時間": 持倉時間.item(),
                                "信號強度": signal_threshold
                            })
                            await push_limiter.cache_message(
                                f"【下單通知】停損平倉: {市場}_{時間框架}, 價格={成交價格.item():.2f}, 數量={持倉數量.item():.2f}, 損益={損益.item():.2f}",
                                市場, 時間框架, "模擬交易", "normal")
                            await push_limiter.retry_cached_messages()
                            持倉數量 = 0
                            平均入場價格 = 0.0
                            未實現盈虧 = 0.0
                            持倉時間 = 0
                            最後平倉時間 = 當前時間
                            最高價格 = 0.0
                            最低價格 = float('inf')
                        elif 當前價格 >= 平均入場價格 * (1 + 停利):
                            成交價格 = 當前價格
                            手續費總 = 持倉數量 * 成交價格 * 手續費 * 2
                            損益 = (成交價格 - 平均入場價格) * 持倉數量 * 槓桿 - 手續費總
                            資金 += 損益
                            if 損益 < 0:
                                日損失 += abs(損益)
                                連續虧損次數 += 1
                            else:
                                連續虧損次數 = 0
                            交易記錄[0].append({
                                "id": str(uuid.uuid4()),
                                "市場": 市場,
                                "時間框架": 時間框架,
                                "類型": "停利平倉",
                                "價格": 成交價格.item(),
                                "數量": 持倉數量.item(),
                                "損益": 損益.item(),
                                "時間": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                "持倉時間": 持倉時間.item(),
                                "信號強度": signal_threshold
                            })
                            await push_limiter.cache_message(
                                f"【下單通知】停利平倉: {市場}_{時間框架}, 價格={成交價格.item():.2f}, 數量={持倉數量.item():.2f}, 損益={損益.item():.2f}",
                                市場, 時間框架, "模擬交易", "normal")
                            await push_limiter.retry_cached_messages()
                            持倉數量 = 0
                            平均入場價格 = 0.0
                            未實現盈虧 = 0.0
                            持倉時間 = 0
                            最後平倉時間 = 當前時間
                            最高價格 = 0.0
                            最低價格 = float('inf')

                # 交易延遲檢查
                交易結束時間 = time.time()
                延遲 = (交易結束時間 - 交易開始時間) * 1000
                if 延遲 > 100:
                    logger.warning(f"[{市場}_{時間框架}] 交易延遲超限: {延遲:.2f}ms")
                    await push_limiter.cache_message(
                        f"錯誤碼E101：交易延遲超限 {延遲:.2f}ms",
                        市場, 時間框架, "模擬交易", "high")
                    await push_limiter.retry_cached_messages()

            # 檢查同K棒反手違規
            if not await check_reverse_trade_violation(交易記錄[0], 市場, 時間框架):
                return None

            # 儲存交易結果
            result = {
                市場: {
                    "最終資金": 資金.item(),
                    "最大回撤": 最大回撤.item(),
                    "交易記錄": 交易記錄[0],
                    "連線狀態": "模擬模式",
                    "維持率": 維持率.item()
                }
            }

            # 快取交易結果
            cache_path = 快取資料夾 / f"simulation_cache_{市場}_{時間框架}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
            np.savez_compressed(cache_path, data=result)
            await push_limiter.cache_message(
                f"【執行通知】交易結果已快取至 {cache_path}",
                市場, 時間框架, "模擬交易", "normal")
            await push_limiter.retry_cached_messages()

            # 加密並儲存交易記錄
            encrypted_records = await encrypt_trading_records(result[市場]["交易記錄"])
            if not encrypted_records:
                return None
            async with aiosqlite.connect(快取資料夾.parent / "SQLite" / "模擬交易記錄.db") as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS 模擬交易記錄 (
                        id TEXT PRIMARY KEY,
                        市場 TEXT,
                        時間框架 TEXT,
                        最終資金 REAL,
                        最大回撤 REAL,
                        交易記錄 BLOB,
                        連線狀態 TEXT,
                        維持率 REAL,
                        時間 TEXT
                    )
                """)
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_市場_時間框架 ON 模擬交易記錄 (市場, 時間框架, 時間)")
                await conn.execute("INSERT INTO 模擬交易記錄 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                 (str(uuid.uuid4()), 市場, 時間框架, result[市場]["最終資金"],
                                  result[市場]["最大回撤"], encrypted_records, result[市場]["連線狀態"],
                                  result[市場]["維持率"], datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                await conn.commit()

            # 寫入異動歷程
            change_log_path = 快取資料夾.parent / "dot" / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
            change_log = {
                "UUID": str(uuid.uuid4()),
                "市場": 市場,
                "時間框架": 時間框架,
                "異動前值": "N/A",
                "異動後值": f"最終資金={result[市場]['最終資金']:.2f}, 最大回撤={result[市場]['最大回撤']:.2%}, 交易數={len(result[市場]['交易記錄'])}",
                "異動原因": "模擬交易",
                "時間戳": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            change_df = pd.DataFrame([change_log])
            if change_log_path.exists():
                existing_df = pd.read_excel(change_log_path)
                change_df = pd.concat([existing_df, change_df], ignore_index=True)
            change_df.to_excel(change_log_path, index=False, engine='openpyxl')

            # 推播交易結果
            message = (
                f"【模擬交易完成】\n"
                f"市場: {市場}_{時間框架}\n"
                f"最終資金: {result[市場]['最終資金']:.2f}\n"
                f"最大回撤: {result[市場]['最大回撤']:.2%}\n"
                f"交易數: {len(result[市場]['交易記錄'])}\n"
                f"維持率: {result[市場]['維持率']:.2f}%"
            )
            await push_limiter.cache_message(message, 市場, 時間框架, "模擬交易", "normal")
            await push_limiter.retry_cached_messages()

            # 效率報告
            elapsed_time = time.time() - start_time
            cpu_percent = psutil.cpu_percent()
            ram_percent = psutil.virtual_memory().percent
            gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
            efficiency_report = (
                f"【效率報告】模擬交易耗時：{elapsed_time:.2f}秒，"
                f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
                f"GPU：{gpu_util:.1f}%"
            )
            logger.info(efficiency_report)
            await push_limiter.cache_message(efficiency_report, 市場, 時間框架, "模擬交易", "normal")
            await push_limiter.retry_cached_messages()

            await 發送倉位通知(
                市場, 時間框架, "模擬",
                持倉數量.item(),
                平均入場價格.item(),
                未實現盈虧.item(),
                維持率.item()
            )
            return result

    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[{市場}_{時間框架}] 模擬交易失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E101：模擬交易失敗 重試{retry_count + 1}/5: {e}", "交易錯誤", 市場, 時間框架, "模擬交易")
            await asyncio.sleep(5)
            return await 模擬交易(
                信號, 價格, 資產類型, 市場, 時間框架, strong_signals, batch_size, 停損, 停利,
                平本觸發, 移動停損, 移動停利, 單筆損失限制, 單日損失限制, retry_count + 1
            )
        logger.error(f"[{市場}_{時間框架}] 模擬交易失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E101：模擬交易失敗 {e}", 市場, 時間框架, "模擬交易", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E101：模擬交易失敗 {e}", "交易錯誤", 市場, 時間框架, "模擬交易")
        return None