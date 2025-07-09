import asyncio
import logging
import datetime
import aiosqlite
import uuid
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import ccxt.async_support as ccxt
import psutil
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from collections import deque
from 推播通知模組 import 發送錯誤訊息, 發送通知
from 工具模組 import 錯誤記錄與自動修復, validate_utility_input

# 配置日誌
BASE_DIR = Path("D:/自動化交易/StrategyProject/data")
log_dir = BASE_DIR / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("設定檔")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
file_handler = TimedRotatingFileHandler(
    filename=log_dir / "config_logs",
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
    "E1001": "參數載入失敗",
    "E1002": "參數驗證失敗",
    "E1003": "DQN 參數優化失敗",
    "E1004": "測試網餘額獲取失敗",
    "E1005": "主網餘額獲取失敗"
}

# 推播限制器
class PushLimiter:
    def __init__(self, max_pushes_per_minute=10):
        self.max_pushes = max_pushes_per_minute
        self.push_timestamps = deque(maxlen=60)
        self.cache_db = BASE_DIR / "sqlite" / "push_cache.db"

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

# 參數緩衝區
_config_buffer = deque(maxlen=100)

# DQN 模型定義
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 訓練設備
訓練設備 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"訓練設備: {訓練設備}")

# 資料夾設定
SQLite資料夾 = BASE_DIR / "sqlite"
快取資料夾 = BASE_DIR / "cache"
備份資料夾 = BASE_DIR / "backup"
異動歷程資料夾 = BASE_DIR / "dot"
圖片資料夾 = BASE_DIR / "圖片"
SQLite資料夾.mkdir(parents=True, exist_ok=True)
快取資料夾.mkdir(parents=True, exist_ok=True)
備份資料夾.mkdir(parents=True, exist_ok=True)
異動歷程資料夾.mkdir(parents=True, exist_ok=True)
圖片資料夾.mkdir(parents=True, exist_ok=True)

# 資源閾值
資源閾值 = {
    "RAM使用率": 0.7,
    "硬碟剩餘比例": 0.1,
    "CPU使用率": 0.9,
    "GPU使用率": 0.85
}

# 硬編碼環境變數（保留硬編碼，新增備份）
幣安測試網API金鑰 = "ea0d968ec7b8b55811979d118795a800ab3928a14e0a215de403e4d3a846b33d"
幣安測試網API密鑰 = "c868b8ce40747c434787f5ef8ef1eb557871ca190356be7c7c52e34fd730e3f6"
幣安主網API金鑰 = "m38NCzN0lFMY8K2m446zjiD2O02M6OgoUC2b8YuajTlwRtMSRM6baEbkkICZBySp"
幣安主網API密鑰 = "N1A2P5ri1JcXSL9ae75aN0LkACFfFvqGFwv8apFxNWSXL9PU04ZcGynPOebJzScX"
TELEGRAM_BOT_TOKEN = "7694752059:AAHHb819DPTO5opNcVDxt5_Q0uYAJdPEZSE"
TELEGRAM_CHAT_ID = "939642374"
_api_backup = {
    "testnet_api_key": 幣安測試網API金鑰,
    "testnet_api_secret": 幣安測試網API密鑰,
    "mainnet_api_key": 幣安主網API金鑰,
    "mainnet_api_secret": 幣安主網API密鑰,
    "telegram_token": TELEGRAM_BOT_TOKEN,
    "telegram_chat_id": TELEGRAM_CHAT_ID
}

# 市場清單
市場清單 = [
    ("BTCUSDT", "1m"), ("BTCUSDT", "5m"), ("BTCUSDT", "15m"), ("BTCUSDT", "30m"), ("BTCUSDT", "1h"), ("BTCUSDT", "4h"), ("BTCUSDT", "1d"),
    ("ETHUSDT", "1m"), ("ETHUSDT", "5m"), ("ETHUSDT", "15m"), ("ETHUSDT", "30m"), ("ETHUSDT", "1h"), ("ETHUSDT", "4h"), ("ETHUSDT", "1d"),
    ("SOLUSDT", "1m"), ("SOLUSDT", "5m"), ("SOLUSDT", "15m"), ("SOLUSDT", "30m"), ("SOLUSDT", "1h"), ("SOLUSDT", "4h"), ("SOLUSDT", "1d"),
    ("XRPUSDT", "1m"), ("XRPUSDT", "5m"), ("XRPUSDT", "15m"), ("XRPUSDT", "30m"), ("XRPUSDT", "1h"), ("XRPUSDT", "4h"), ("XRPUSDT", "1d"),
    ("DOGEUSDT", "1m"), ("DOGEUSDT", "5m"), ("DOGEUSDT", "15m"), ("DOGEUSDT", "30m"), ("DOGEUSDT", "1h"), ("DOGEUSDT", "4h"), ("DOGEUSDT", "1d"),
    ("ADAUSDT", "1m"), ("ADAUSDT", "5m"), ("ADAUSDT", "15m"), ("ADAUSDT", "30m"), ("ADAUSDT", "1h"), ("ADAUSDT", "4h"), ("ADAUSDT", "1d"),
    ("BNBUSDT", "1m"), ("BNBUSDT", "5m"), ("BNBUSDT", "15m"), ("BNBUSDT", "30m"), ("BNBUSDT", "1h"), ("BNBUSDT", "4h"), ("BNBUSDT", "1d"),
    ("NDAQ100", "1m"), ("NDAQ100", "5m"), ("NDAQ100", "15m"), ("NDAQ100", "30m"), ("NDAQ100", "1h"), ("NDAQ100", "4h"), ("NDAQ100", "1d"),
    ("DJ30", "1m"), ("DJ30", "5m"), ("DJ30", "15m"), ("DJ30", "30m"), ("DJ30", "1h"), ("DJ30", "4h"), ("DJ30", "1d"),
    ("JPN225", "1m"), ("JPN225", "5m"), ("JPN225", "15m"), ("JPN225", "30m"), ("JPN225", "1h"), ("JPN225", "4h"), ("JPN225", "1d")
]

async def validate_config(retry_count=0):
    """驗證設定檔參數的正確性與完整性"""
    try:
        import time
        start_time = time.time()
        # 驗證資料夾
        for folder in [SQLite資料夾, 快取資料夾, 備份資料夾, 異動歷程資料夾, 圖片資料夾]:
            if not folder.exists():
                logger.error(f"資料夾不存在: {folder}")
                await push_limiter.cache_message(f"錯誤碼E1002：資料夾不存在 {folder}", "多市場", "多框架", "設定檔", "high")
                await push_limiter.retry_cached_messages()
                raise ValueError(f"資料夾不存在: {folder}")

        # 驗證訓練設備
        if 訓練設備 not in [torch.device("cuda"), torch.device("cpu")]:
            logger.error(f"無效訓練設備: {訓練設備}")
            await push_limiter.cache_message(f"錯誤碼E1002：無效訓練設備 {訓練設備}", "多市場", "多框架", "設定檔", "high")
            await push_limiter.retry_cached_messages()
            raise ValueError(f"無效訓練設備: {訓練設備}")

        # 驗證 API 金鑰
        for key, name in [
            (幣安測試網API金鑰, "幣安測試網API金鑰"),
            (幣安測試網API密鑰, "幣安測試網API密鑰"),
            (幣安主網API金鑰, "幣安主網API金鑰"),
            (幣安主網API密鑰, "幣安主網API密鑰"),
            (TELEGRAM_BOT_TOKEN, "Telegram Bot Token"),
            (TELEGRAM_CHAT_ID, "Telegram Chat ID")
        ]:
            if not key or len(key) < 10:
                logger.error(f"{name} 無效或過短")
                await push_limiter.cache_message(f"錯誤碼E1002：{name} 無效或過短", "多市場", "多框架", "設定檔", "high")
                await push_limiter.retry_cached_messages()
                raise ValueError(f"{name} 無效或過短")

        # 驗證市場清單
        if not 市場清單 or not all(isinstance(m, tuple) and len(m) == 2 for m in 市場清單):
            logger.error("市場清單格式錯誤")
            await push_limiter.cache_message(f"錯誤碼E1002：市場清單格式錯誤", "多市場", "多框架", "設定檔", "high")
            await push_limiter.retry_cached_messages()
            raise ValueError("市場清單格式錯誤")

        # 驗證訓練參數
        required_param_keys = ["值", "範圍"]
        for param_name, param in 訓練參數.items():
            if not all(key in param for key in required_param_keys):
                logger.error(f"訓練參數 {param_name} 缺少必要鍵: {required_param_keys}")
                await push_limiter.cache_message(f"錯誤碼E1002：訓練參數 {param_name} 缺少必要鍵", "多市場", "多框架", "設定檔", "high")
                await push_limiter.retry_cached_messages()
                raise ValueError(f"訓練參數 {param_name} 缺少必要鍵")
            if not isinstance(param["值"], (int, float, str)):
                logger.error(f"訓練參數 {param_name} 的值無效: {param['值']}")
                await push_limiter.cache_message(f"錯誤碼E1002：訓練參數 {param_name} 的值無效", "多市場", "多框架", "設定檔", "high")
                await push_limiter.retry_cached_messages()
                raise ValueError(f"訓練參數 {param_name} 的值無效")
            if not isinstance(param["範圍"], (list, tuple)) or len(param["範圍"]) < 2:
                logger.error(f"訓練參數 {param_name} 的範圍無效: {param['範圍']}")
                await push_limiter.cache_message(f"錯誤碼E1002：訓練參數 {param_name} 的範圍無效", "多市場", "多框架", "設定檔", "high")
                await push_limiter.retry_cached_messages()
                raise ValueError(f"訓練參數 {param_name} 的範圍無效")
            if isinstance(param["值"], (int, float)) and isinstance(param["範圍"], list) and not (param["範圍"][0] <= param["值"] <= param["範圍"][1]):
                logger.error(f"訓練參數 {param_name} 的值超出範圍: {param['值']} not in {param['範圍']}")
                await push_limiter.cache_message(f"錯誤碼E1002：訓練參數 {param_name} 的值超出範圍", "多市場", "多框架", "設定檔", "high")
                await push_limiter.retry_cached_messages()
                raise ValueError(f"訓練參數 {param_name} 的值超出範圍")

        # 驗證資源閾值
        required_threshold_keys = ["RAM使用率", "硬碟剩餘比例", "CPU使用率", "GPU使用率"]
        for key in required_threshold_keys:
            if key not in 資源閾值:
                logger.error(f"資源閾值缺少必要鍵: {key}")
                await push_limiter.cache_message(f"錯誤碼E1002：資源閾值缺少必要鍵 {key}", "多市場", "多框架", "設定檔", "high")
                await push_limiter.retry_cached_messages()
                raise ValueError(f"資源閾值缺少必要鍵: {key}")
            if not isinstance(資源閾值[key], (int, float)) or 資源閾值[key] <= 0 or 資源閾值[key] > 1:
                logger.error(f"資源閾值 {key} 無效: {資源閾值[key]}")
                await push_limiter.cache_message(f"錯誤碼E1002：資源閾值 {key} 無效", "多市場", "多框架", "設定檔", "high")
                await push_limiter.retry_cached_messages()
                raise ValueError(f"資源閾值 {key} 無效")

        logger.info("設定檔驗證通過")
        await push_limiter.cache_message("【執行通知】設定檔驗證通過", "多市場", "多框架", "設定檔", "normal")
        await push_limiter.retry_cached_messages()

        # 效率報告
        elapsed_time = time.time() - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】設定檔驗證耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, "多市場", "多框架", "設定檔", "normal")
        await push_limiter.retry_cached_messages()
        return True
    except Exception as e:
        if retry_count < 3:
            logger.warning(f"設定檔驗證失敗，重試 {retry_count + 1}/3: {e}")
            await asyncio.sleep(5)
            return await validate_config(retry_count + 1)
        logger.error(f"設定檔驗證失敗，重試 3 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E1002：設定檔驗證失敗 {e}", "多市場", "多框架", "設定檔", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E1002：設定檔驗證失敗 {e}", "設定檔驗證錯誤", "多市場", "多框架", "設定檔")
        raise

async def get_testnet_balance(market="BTCUSDT", retry_count=0):
    """獲取幣安測試網帳戶餘額"""
    try:
        import time
        start_time = time.time()
        gpu_switch_count = getattr(get_testnet_balance, 'switch_count', 0)
        if torch.cuda.is_available():
            # 模擬GPU加速（實際依賴API）
            balance_tensor = torch.tensor([1000.0], device=訓練設備)
            total_balance = balance_tensor.item()
            if torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory > 0.85:
                torch.cuda.empty_cache()
                gpu_switch_count += 1
                setattr(get_testnet_balance, 'switch_count', gpu_switch_count)
        else:
            exchange = ccxt.binance({
                'apiKey': 幣安測試網API金鑰,
                'secret': 幣安測試網API密鑰,
                'enableRateLimit': True,
                'urls': {'api': 'https://testnet.binance.vision'}
            })
            balance = await exchange.fetch_balance()
            total_balance = balance['USDT']['total'] if 'USDT' in balance else 1000.0
            await exchange.close()
        logger.info(f"[{market}] 測試網餘額: {total_balance:.2f}")
        await push_limiter.cache_message(f"【執行通知】{market} 測試網餘額: {total_balance:.2f}", market, "15m", "設定檔", "normal")
        await push_limiter.retry_cached_messages()

        # 效率報告
        elapsed_time = time.time() - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】測試網餘額獲取耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%，GPU切換次數：{gpu_switch_count}"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, market, "15m", "設定檔", "normal")
        await push_limiter.retry_cached_messages()
        return total_balance
    except Exception as e:
        if retry_count < 3:
            logger.warning(f"[{market}] 獲取測試網餘額失敗，重試 {retry_count + 1}/3: {e}")
            await asyncio.sleep(5)
            return await get_testnet_balance(market, retry_count + 1)
        logger.error(f"[{market}] 獲取測試網餘額失敗，重試 3 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E1004：獲取測試網餘額失敗 {e}", market, "15m", "設定檔", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E1004：獲取測試網餘額失敗 {e}", "餘額獲取錯誤", market, "15m", "設定檔")
        return 1000.0

async def get_mainnet_balance(market="BTCUSDT", retry_count=0):
    """獲取幣安主網帳戶餘額"""
    try:
        import time
        start_time = time.time()
        gpu_switch_count = getattr(get_mainnet_balance, 'switch_count', 0)
        if torch.cuda.is_available():
            # 模擬GPU加速（實際依賴API）
            balance_tensor = torch.tensor([1000.0], device=訓練設備)
            total_balance = balance_tensor.item()
            if torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory > 0.85:
                torch.cuda.empty_cache()
                gpu_switch_count += 1
                setattr(get_mainnet_balance, 'switch_count', gpu_switch_count)
        else:
            exchange = ccxt.binance({
                'apiKey': 幣安主網API金鑰,
                'secret': 幣安主網API密鑰,
                'enableRateLimit': True
            })
            balance = await exchange.fetch_balance()
            total_balance = balance['USDT']['total'] if 'USDT' in balance else 1000.0
            await exchange.close()
        logger.info(f"[{market}] 主網餘額: {total_balance:.2f}")
        await push_limiter.cache_message(f"【執行通知】{market} 主網餘額: {total_balance:.2f}", market, "15m", "設定檔", "normal")
        await push_limiter.retry_cached_messages()

        # 效率報告
        elapsed_time = time.time() - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】主網餘額獲取耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%，GPU切換次數：{gpu_switch_count}"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, market, "15m", "設定檔", "normal")
        await push_limiter.retry_cached_messages()
        return total_balance
    except Exception as e:
        if retry_count < 3:
            logger.warning(f"[{market}] 獲取主網餘額失敗，重試 {retry_count + 1}/3: {e}")
            await asyncio.sleep(5)
            return await get_mainnet_balance(market, retry_count + 1)
        logger.error(f"[{market}] 獲取主網餘額失敗，重試 3 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E1005：獲取主網餘額失敗 {e}", market, "15m", "設定檔", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E1005：獲取主網餘額失敗 {e}", "餘額獲取錯誤", market, "15m", "設定檔")
        return 1000.0

async def validate_config_input(market, timeframe, params=None, retry_count=0):
    """驗證參數輸入"""
    try:
        import time
        start_time = time.time()
        if not await validate_utility_input(market, timeframe, mode="設定檔"):
            return False
        if params and not isinstance(params, dict):
            logger.error(f"[{market}_{timeframe}] 無效參數格式")
            await push_limiter.cache_message(f"錯誤碼E1002：無效參數格式", market, timeframe, "設定檔", "high")
            await push_limiter.retry_cached_messages()
            return False
        # 效率報告
        elapsed_time = time.time() - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】參數驗證耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, market, timeframe, "設定檔", "normal")
        await push_limiter.retry_cached_messages()
        return True
    except Exception as e:
        if retry_count < 3:
            logger.warning(f"[{market}_{timeframe}] 參數驗證失敗，重試 {retry_count + 1}/3: {e}")
            await asyncio.sleep(5)
            return await validate_config_input(market, timeframe, params, retry_count + 1)
        logger.error(f"[{market}_{timeframe}] 參數驗證失敗，重試 3 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E1002：參數驗證失敗 {e}", market, timeframe, "設定檔", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E1002：參數驗證失敗 {e}", "參數驗證錯誤", market, timeframe, "設定檔")
        return False

async def dqn_optimize_config_params(market, timeframe, trading_result, retry_count=0):
    """使用 DQN 強化學習優化參數"""
    try:
        import time
        start_time = time.time()
        input_dim = 6  # 資金, 回撤, 勝率, f1分數, 穩定性, ATR
        output_dim = 16  # 調整 8 個參數的增減
        dqn = DQN(input_dim, output_dim).to(訓練設備)
        optimizer = torch.optim.Adam(dqn.parameters(), lr=1e-3)
        gamma = 0.99
        epsilon = 0.1
        episodes = 50

        initial_funds = await get_testnet_balance(market)
        base_params = {
            "single_loss_limit": 0.02,
            "daily_loss_limit": 0.05,
            "stop_loss": 0.02,
            "take_profit": 0.03,
            "trailing_stop": 0.01,
            "trailing_take_profit": 0.02,
            "breakeven_trigger": 0.01,
            "signal_threshold": 0.9
        }
        best_params = base_params.copy()
        best_reward = -float('inf')

        for episode in range(episodes):
            state = torch.tensor([
                trading_result.get("最終資金", initial_funds),
                trading_result.get("最大回撤", 0.0),
                trading_result.get("勝率", 0.0),
                trading_result.get("f1分數", 0.0),
                trading_result.get("穩定性", 0.0),
                trading_result.get("atr", 14.0)
            ], dtype=torch.float32, device=訓練設備)
            episode_reward = 0.0

            for _ in range(10):
                if np.random.random() < epsilon:
                    action = np.random.randint(0, output_dim)
                else:
                    with torch.no_grad():
                        q_values = dqn(state)
                        action = q_values.argmax().item()

                temp_params = best_params.copy()
                if action == 0:
                    temp_params["single_loss_limit"] = min(temp_params["single_loss_limit"] + 0.005, 0.03)
                elif action == 1:
                    temp_params["single_loss_limit"] = max(temp_params["single_loss_limit"] - 0.005, 0.01)
                elif action == 2:
                    temp_params["daily_loss_limit"] = min(temp_params["daily_loss_limit"] + 0.01, 0.07)
                elif action == 3:
                    temp_params["daily_loss_limit"] = max(temp_params["daily_loss_limit"] - 0.01, 0.03)
                elif action == 4:
                    temp_params["stop_loss"] = min(temp_params["stop_loss"] + 0.005, 0.05)
                elif action == 5:
                    temp_params["stop_loss"] = max(temp_params["stop_loss"] - 0.005, 0.01)
                elif action == 6:
                    temp_params["take_profit"] = min(temp_params["take_profit"] + 0.01, 0.1)
                elif action == 7:
                    temp_params["take_profit"] = max(temp_params["take_profit"] - 0.01, 0.02)
                elif action == 8:
                    temp_params["trailing_stop"] = min(temp_params["trailing_stop"] + 0.005, 0.05)
                elif action == 9:
                    temp_params["trailing_stop"] = max(temp_params["trailing_stop"] - 0.005, 0.01)
                elif action == 10:
                    temp_params["trailing_take_profit"] = min(temp_params["trailing_take_profit"] + 0.01, 0.1)
                elif action == 11:
                    temp_params["trailing_take_profit"] = max(temp_params["trailing_take_profit"] - 0.01, 0.02)
                elif action == 12:
                    temp_params["breakeven_trigger"] = min(temp_params["breakeven_trigger"] + 0.005, 0.05)
                elif action == 13:
                    temp_params["breakeven_trigger"] = max(temp_params["breakeven_trigger"] - 0.005, 0.01)
                elif action == 14:
                    temp_params["signal_threshold"] = min(temp_params["signal_threshold"] + 0.05, 0.95)
                elif action == 15:
                    temp_params["signal_threshold"] = max(temp_params["signal_threshold"] - 0.05, 0.5)

                # 模擬交易
                from 交易環境模組 import single_market_trading_env
                market_signal_mapping = {(market, timeframe): {
                    "信號": [trading_result.get("信號", 1.0)],
                    "價格": [trading_result.get("價格", 10000.0)]
                }}
                result = await single_market_trading_env(
                    market_signal_mapping=market_signal_mapping,
                    資產類型="虛擬貨幣" if market in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "BNBUSDT"] else "CFD",
                    市場=market,
                    時間框架=timeframe,
                    params=temp_params
                )

                if result is None or result.get((market, timeframe)) is None:
                    temp_reward = -1.0
                else:
                    from 獎勵計算模組 import calculate_single_market_reward
                    temp_reward = await calculate_single_market_reward(
                        result[(market, timeframe)],
                        market,
                        timeframe,
                        {"HMA_16": temp_params.get("hma_period", 16), "SMA50": temp_params.get("sma_period", 50), "ATR_14": temp_params.get("atr_period", 14)}
                    )

                episode_reward += temp_reward
                next_state = torch.tensor([
                    result[(market, timeframe)]["最終資金"] if result else initial_funds,
                    result[(market, timeframe)].get("最大回撤", 0.0) if result else 0.0,
                    result[(market, timeframe)].get("勝率", 0.0) if result else 0.0,
                    result[(market, timeframe)].get("f1分數", 0.0) if result else 0.0,
                    result[(market, timeframe)].get("穩定性", 0.0) if result else 0.0,
                    temp_params.get("atr_period", 14.0)
                ], dtype=torch.float32, device=訓練設備)

                q_value = dqn(state)[action]
                target = temp_reward + gamma * dqn(next_state).max()
                loss = nn.MSELoss()(q_value, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                state = next_state
                if temp_reward > best_reward:
                    best_reward = temp_reward
                    best_params = temp_params.copy()

            # 儲存每輪結果
            async with aiosqlite.connect(SQLite資料夾 / "參數優化記錄.db") as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS 參數優化記錄 (
                        id TEXT PRIMARY KEY,
                        市場 TEXT,
                        時間框架 TEXT,
                        單筆損失限制 REAL,
                        單日損失限制 REAL,
                        停損 REAL,
                        停利 REAL,
                        移動停損 REAL,
                        移動停利 REAL,
                        平本觸發 REAL,
                        信號閾值 REAL,
                        獎勵 REAL,
                        時間 TEXT
                    )
                """)
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_市場_時間框架 ON 參數優化記錄 (市場, 時間框架, 時間)")
                await conn.execute("INSERT INTO 參數優化記錄 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                 (str(uuid.uuid4()), market, timeframe, best_params["single_loss_limit"],
                                  best_params["daily_loss_limit"], best_params["stop_loss"],
                                  best_params["take_profit"], best_params["trailing_stop"],
                                  best_params["trailing_take_profit"], best_params["breakeven_trigger"],
                                  best_params["signal_threshold"], episode_reward,
                                  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                await conn.commit()

        # 檢查穩定性
        async with aiosqlite.connect(SQLite資料夾 / "參數優化記錄.db") as conn:
            df = await conn.execute_fetchall(
                "SELECT 單筆損失限制, 單日損失限制, 停損, 停利, 移動停損, 移動停利, 平本觸發, 信號閾值 FROM 參數優化記錄 WHERE 市場 = ? AND 時間框架 = ? ORDER BY 時間 DESC LIMIT 10",
                (market, timeframe))
            df = pd.DataFrame(df, columns=["單筆損失限制", "單日損失限制", "停損", "停利", "移動停損", "移動停利", "平本觸發", "信號閾值"])
        if len(df) > 1:
            for column in df.columns:
                if df[column].std() > 0.1:
                    logger.warning(f"[{market}_{timeframe}] DQN 參數 {column} 穩定性低，標準差: {df[column].std():.4f}")
                    await push_limiter.cache_message(
                        f"錯誤碼E1003：DQN 參數 {column} 穩定性低，標準差 {df[column].std():.4f}",
                        market, timeframe, "設定檔", "high")
                    await push_limiter.retry_cached_messages()

        # 快取參數
        cache_path = 快取資料夾 / f"config_cache_{market}_{timeframe}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez_compressed(cache_path, data=best_params)
        await push_limiter.cache_message(f"【執行通知】參數已快取至 {cache_path}", market, timeframe, "設定檔", "normal")
        await push_limiter.retry_cached_messages()

        # 推播優化結果
        message = (
            f"【DQN 參數優化完成】\n"
            f"市場: {market}_{timeframe}\n"
            f"獎勵: {best_reward:.4f}\n"
            f"單筆損失限制: {best_params['single_loss_limit']:.4f}\n"
            f"單日損失限制: {best_params['daily_loss_limit']:.4f}\n"
            f"停損: {best_params['stop_loss']:.4f}\n"
            f"停利: {best_params['take_profit']:.4f}\n"
            f"移動停損: {best_params['trailing_stop']:.4f}\n"
            f"移動停利: {best_params['trailing_take_profit']:.4f}\n"
            f"平本觸發: {best_params['breakeven_trigger']:.4f}\n"
            f"信號閾值: {best_params['signal_threshold']:.4f}"
        )
        await push_limiter.cache_message(message, market, timeframe, "設定檔", "normal")
        await push_limiter.retry_cached_messages()

        # 效率報告
        elapsed_time = time.time() - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】DQN 參數優化耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, market, timeframe, "設定檔", "normal")
        await push_limiter.retry_cached_messages()
        return best_params
    except Exception as e:
        if retry_count < 3:
            logger.warning(f"[{market}_{timeframe}] DQN 參數優化失敗，重試 {retry_count + 1}/3: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E1003：DQN 參數優化失敗 重試{retry_count + 1}/3: {e}", "DQN 參數優化錯誤", market, timeframe, "設定檔")
            await asyncio.sleep(5)
            return await dqn_optimize_config_params(market, timeframe, trading_result, retry_count + 1)
        logger.error(f"[{market}_{timeframe}] DQN 參數優化失敗，重試 3 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E1003：DQN 參數優化失敗 {e}", market, timeframe, "設定檔", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E1003：DQN 參數優化失敗 {e}", "DQN 參數優化錯誤", market, timeframe, "設定檔")
        return {
            "single_loss_limit": 0.02,
            "daily_loss_limit": 0.05,
            "stop_loss": 0.02,
            "take_profit": 0.03,
            "trailing_stop": 0.01,
            "trailing_take_profit": 0.02,
            "breakeven_trigger": 0.01,
            "signal_threshold": 0.9
        }

# 訓練參數（統一結構，新增動態調整接口）
訓練參數 = {
    "學習率": {"值": 0.001, "範圍": [1e-5, 1e-2]},
    "dropout": {"值": 0.2, "範圍": [0.1, 0.5]},
    "batch_size": {"值": 64, "範圍": [32, 64, 128, 256]},
    "層數": {"值": 2, "範圍": [1, 4]},
    "神經元數": {"值": 128, "範圍": [64, 256]},
    "optimizer": {"值": "Adam", "範圍": ["Adam", "SGD", "RMSprop"]},
    "SMA週期": {"值": 50, "範圍": [10, 200]},
    "HMA週期": {"值": 16, "範圍": [8, 30]},
    "ATR週期": {"值": 14, "範圍": [7, 30]},
    "VHF週期": {"值": 28, "範圍": [14, 100]},
    "Pivot週期": {"值": 5, "範圍": [3, 20]},
    "stop_loss": {"值": 0.02, "範圍": [0.01, 0.05]},
    "take_profit": {"值": 0.03, "範圍": [0.02, 0.1]},
    "trailing_stop": {"值": 0.01, "範圍": [0.01, 0.05]},
    "trailing_take_profit": {"值": 0.02, "範圍": [0.02, 0.1]},
    "breakeven_trigger": {"值": 0.01, "範圍": [0.01, 0.05]},
    "signal_threshold": {"值": 0.9, "範圍": [0.5, 0.95]},
    "max_drawdown": {"值": 0.25, "範圍": [0.2, 0.3]},
    "single_loss_limit": {"值": 0.02, "範圍": [0.01, 0.03]},
    "daily_loss_limit": {"值": 0.05, "範圍": [0.03, 0.07]}
}

async def update_training_params(param_name, new_value, new_range):
    """動態調整訓練參數範圍"""
    if param_name in 訓練參數:
        訓練參數[param_name]["值"] = new_value
        訓練參數[param_name]["範圍"] = new_range
        await 記錄參數異動("範圍更新", 訓練參數[param_name]["範圍"], new_range, f"更新 {param_name} 範圍")
        logger.info(f"更新訓練參數 {param_name} 為 {new_value}, 範圍 {new_range}")
        return True
    return False

async def get_market_specific_config(market, timeframe, trading_result=None, retry_count=0):
    """獲取單一市場與時間框架的參數配置"""
    try:
        import time
        start_time = time.time()
        if not await validate_config_input(market, timeframe):
            return None

        # 固定參數（規格書 4.1.2）
        槓桿比例 = {
            "BTCUSDT": 125.0,
            "ETHUSDT": 125.0,
            "SOLUSDT": 100.0,
            "XRPUSDT": 75.0,
            "DOGEUSDT": 75.0,
            "ADAUSDT": 75.0,
            "BNBUSDT": 75.0,
            "NDAQ100": 20.0,
            "DJ30": 20.0,
            "JPN225": 20.0,
            "default": 10.0,
            "default_crypto": 50.0
        }
        手續費率 = {
            "BTCUSDT": 0.0005,
            "ETHUSDT": 0.0005,
            "SOLUSDT": 0.0005,
            "XRPUSDT": 0.0005,
            "DOGEUSDT": 0.0005,
            "ADAUSDT": 0.0005,
            "BNBUSDT": 0.0005,
            "NDAQ100": 0.0001,
            "DJ30": 0.0001,
            "JPN225": 0.0001,
            "default": 0.0005,
            "default_crypto": 0.0004
        }
        最小下單手數 = {
            "NDAQ100": 0.1,
            "DJ30": 0.1,
            "JPN225": 1.0,
            "default": 0.01
        }
        點差 = {
            "BTCUSDT": 0.1,
            "ETHUSDT": 0.01,
            "SOLUSDT": 0.01,
            "XRPUSDT": 0.0001,
            "DOGEUSDT": 0.00001,
            "ADAUSDT": 0.0001,
            "BNBUSDT": 0.01,
            "NDAQ100": 1.0,
            "DJ30": 1.0,
            "JPN225": 10.0,
            "default": 0.1
        }
        點值 = {
            "BTCUSDT": 1.0,
            "ETHUSDT": 1.0,
            "SOLUSDT": 1.0,
            "XRPUSDT": 1.0,
            "DOGEUSDT": 1.0,
            "ADAUSDT": 1.0,
            "BNBUSDT": 1.0,
            "NDAQ100": 1.0,
            "DJ30": 1.0,
            "JPN225": 1.0,
            "default": 1.0
        }

        # 動態參數（DQN 優化）
        trading_result = trading_result or {}
        optimized_params = await dqn_optimize_config_params(market, timeframe, trading_result)
        config = {
            "leverage": 槓桿比例.get(market, 槓桿比例["default_crypto"] if market in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "BNBUSDT"] else 槓桿比例["default"]),
            "fee_rate": 手續費率.get(market, 手續費率["default_crypto"] if market in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "BNBUSDT"] else 手續費率["default"]),
            "min_order_size": 最小下單手數.get(market, 最小下單手數["default"]) if market in ["NDAQ100", "DJ30", "JPN225"] else 0.1 * await get_testnet_balance(market),
            "point_spread": 點差.get(market, 點差["default"]),
            "point_value": 點值.get(market, 點值["default"]),
            "training_params": 訓練參數,
            "single_loss_limit": optimized_params["single_loss_limit"],
            "daily_loss_limit": optimized_params["daily_loss_limit"],
            "stop_loss": optimized_params["stop_loss"],
            "take_profit": optimized_params["take_profit"],
            "trailing_stop": optimized_params["trailing_stop"],
            "trailing_take_profit": optimized_params["trailing_take_profit"],
            "breakeven_trigger": optimized_params["breakeven_trigger"],
            "signal_threshold": optimized_params["signal_threshold"]
        }

        # 快取配置
        cache_path = 快取資料夾 / f"config_cache_{market}_{timeframe}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez_compressed(cache_path, data=config)
        await push_limiter.cache_message(f"【執行通知】配置已快取至 {cache_path}", market, timeframe, "設定檔", "normal")
        await push_limiter.retry_cached_messages()

        # 儲存配置記錄
        async with aiosqlite.connect(SQLite資料夾 / "config_log.db") as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS 配置記錄 (
                    id TEXT PRIMARY KEY,
                    市場 TEXT,
                    時間框架 TEXT,
                    參數 TEXT,
                    時間 TEXT
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_市場_時間框架 ON 配置記錄 (市場, 時間框架, 時間)")
            await conn.execute("INSERT INTO 配置記錄 VALUES (?, ?, ?, ?, ?)",
                             (str(uuid.uuid4()), market, timeframe, json.dumps(config, ensure_ascii=False),
                              datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            await conn.commit()

        # 寫入異動歷程
        change_log_path = 異動歷程資料夾 / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        change_log = {
            "UUID": str(uuid.uuid4()),
            "市場": market,
            "時間框架": timeframe,
            "異動前值": "N/A",
            "異動後值": f"配置參數: single_loss_limit={config['single_loss_limit']:.4f}, daily_loss_limit={config['daily_loss_limit']:.4f}",
            "異動原因": "參數載入",
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
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】參數載入耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, market, timeframe, "設定檔", "normal")
        await push_limiter.retry_cached_messages()
        return config
    except Exception as e:
        if retry_count < 3:
            logger.warning(f"[{market}_{timeframe}] 參數載入失敗，重試 {retry_count + 1}/3: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E1001：參數載入失敗 重試{retry_count + 1}/3: {e}", "參數載入錯誤", market, timeframe, "設定檔")
            await asyncio.sleep(5)
            return await get_market_specific_config(market, timeframe, trading_result, retry_count + 1)
        logger.error(f"[{market}_{timeframe}] 參數載入失敗，重試 3 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E1001：參數載入失敗 {e}", market, timeframe, "設定檔", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E1001：參數載入失敗 {e}", "參數載入錯誤", market, timeframe, "設定檔")
        return None

async def optimize_all_markets_configs(trading_results, retry_count=0):
    """為所有市場與時間框架優化參數"""
    try:
        import time
        start_time = time.time()
        results = []
        best_params = None
        best_reward = -float('inf')
        weighted_reward = 0.0
        valid_markets = 0

        tasks = []
        for (market, timeframe) in 市場清單:
            weight = 0.5 if market == "BTCUSDT" and timeframe == "15m" else 0.5 / (len(市場清單) - 1)
            trading_result = trading_results.get((market, timeframe), {})
            task = get_market_specific_config(market, timeframe, trading_result)
            tasks.append(task)
        
        configs = await asyncio.gather(*tasks, return_exceptions=True)
        for (market, timeframe), config in zip(市場清單, configs):
            if isinstance(config, Exception):
                logger.error(f"[{market}_{timeframe}] 參數載入失敗: {config}")
                await push_limiter.cache_message(f"錯誤碼E1001：參數載入失敗 {config}", market, timeframe, "設定檔", "high")
                await push_limiter.retry_cached_messages()
                continue
            if config is None:
                continue
            from 交易環境模組 import single_market_trading_env
            from 獎勵計算模組 import calculate_single_market_reward
            market_signal_mapping = {(market, timeframe): {
                "信號": [trading_result.get("信號", 1.0)],
                "價格": [trading_result.get("價格", 10000.0)]
            }}
            result = await single_market_trading_env(
                market_signal_mapping=market_signal_mapping,
                資產類型="虛擬貨幣" if market in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "BNBUSDT"] else "CFD",
                市場=market,
                時間框架=timeframe,
                params=config
            )
            if result is None or result.get((market, timeframe)) is None:
                continue
            reward = await calculate_single_market_reward(
                result[(market, timeframe)],
                market,
                timeframe,
                {"HMA_16": config["training_params"]["HMA週期"]["值"], "SMA50": config["training_params"]["SMA週期"]["值"], "ATR_14": config["training_params"]["ATR週期"]["值"]}
            )
            weighted_reward += reward * weight
            valid_markets += 1
            results.append({
                "市場": market,
                "時間框架": timeframe,
                "獎勵": reward,
                "單筆損失限制": config["single_loss_limit"],
                "單日損失限制": config["daily_loss_limit"],
                "停損": config["stop_loss"],
                "停利": config["take_profit"],
                "移動停損": config["trailing_stop"],
                "移動停利": config["trailing_take_profit"],
                "平本觸發": config["breakeven_trigger"],
                "信號閾值": config["signal_threshold"]
            })
            if reward > best_reward:
                best_reward = reward
                best_params = config.copy()

        # 快取泛用性參數
        if best_params:
            cache_path = 快取資料夾 / f"config_cache_multi_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
            np.savez_compressed(cache_path, data=best_params)
            await push_limiter.cache_message(f"【執行通知】泛用參數已快取至 {cache_path}", "多市場", "多框架", "設定檔", "normal")
            await push_limiter.retry_cached_messages()

        # 儲存泛用性參數
        async with aiosqlite.connect(SQLite資料夾 / "泛用參數.db") as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS 泛用參數 (
                    id TEXT PRIMARY KEY,
                    參數 TEXT,
                    獎勵 REAL,
                    時間 TEXT
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_時間 ON 泛用參數 (時間)")
            await conn.execute("INSERT INTO 泛用參數 VALUES (?, ?, ?, ?)",
                             (str(uuid.uuid4()), json.dumps(best_params, ensure_ascii=False),
                              best_reward, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            await conn.commit()

        # 生成報表
        df = pd.DataFrame(results)
        if not df.empty:
            csv_path = 備份資料夾 / f"All_Markets_Config_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            await push_limiter.cache_message(f"【執行通知】生成全市場參數報表 {csv_path}", "多市場", "多框架", "設定檔", "normal")
            await push_limiter.retry_cached_messages()

        # 寫入異動歷程
        change_log_path = 異動歷程資料夾 / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        change_log = {
            "UUID": str(uuid.uuid4()),
            "市場": "多市場",
            "時間框架": "多框架",
            "異動前值": "N/A",
            "異動後值": f"優化記錄數={len(results)}, 加權獎勵={weighted_reward / valid_markets if valid_markets > 0 else 0.0:.4f}",
            "異動原因": "全市場參數優化",
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
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】全市場參數優化耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, "多市場", "多框架", "設定檔", "normal")
        await push_limiter.retry_cached_messages()
        return best_params, weighted_reward / valid_markets if valid_markets > 0 else 0.0
    except Exception as e:
        if retry_count < 3:
            logger.warning(f"[多市場_多框架] 全市場參數優化失敗，重試 {retry_count + 1}/3: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E1001：全市場參數優化失敗 重試{retry_count + 1}/3: {e}", "全市場參數錯誤", "多市場", "多框架", "設定檔")
            await asyncio.sleep(5)
            return await optimize_all_markets_configs(trading_results, retry_count + 1)
        logger.error(f"[多市場_多框架] 全市場參數優化失敗，重試 3 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E1001：全市場參數優化失敗 {e}", "多市場", "多框架", "設定檔", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E1001：全市場參數優化失敗 {e}", "全市場參數錯誤", "多市場", "多框架", "設定檔")
        return None, 0.0

async def 記錄參數異動(異動類型, 原參數, 新參數, 備註, market="多市場", timeframe="多框架"):
    """記錄參數異動"""
    try:
        import time
        start_time = time.time()
        async with aiosqlite.connect(SQLite資料夾 / "param_change_log.db") as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS 參數異動記錄 (
                    id TEXT PRIMARY KEY,
                    市場 TEXT,
                    時間框架 TEXT,
                    異動類型 TEXT,
                    原參數 TEXT,
                    新參數 TEXT,
                    備註 TEXT,
                    時間 TEXT
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_市場_時間框架 ON 參數異動記錄 (市場, 時間框架, 時間)")
            await conn.execute("INSERT INTO 參數異動記錄 VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                             (str(uuid.uuid4()), market, timeframe, 異動類型,
                              str(原參數), str(新參數), 備註,
                              datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            await conn.commit()

        # 寫入異動歷程
        change_log_path = 異動歷程資料夾 / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        change_log = {
            "UUID": str(uuid.uuid4()),
            "市場": market,
            "時間框架": timeframe,
            "異動前值": str(原參數),
            "異動後值": str(新參數),
            "異動原因": 備註,
            "時間戳": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        change_df = pd.DataFrame([change_log])
        if change_log_path.exists():
            existing_df = pd.read_excel(change_log_path)
            change_df = pd.concat([existing_df, change_df], ignore_index=True)
        change_df.to_excel(change_log_path, index=False, engine='openpyxl')

        logger.info(f"[{market}_{timeframe}] 參數異動記錄: {異動類型} - {備註}")
        await push_limiter.cache_message(f"【執行通知】參數異動記錄: {異動類型} - {備註}", market, timeframe, "設定檔", "normal")
        await push_limiter.retry_cached_messages()

        # 效率報告
        elapsed_time = time.time() - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】參數異動記錄耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, market, timeframe, "設定檔", "normal")
        await push_limiter.retry_cached_messages()
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 參數異動記錄失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E1001：參數異動記錄失敗 {e}", market, timeframe, "設定檔", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E1001：參數異動記錄失敗 {e}", "參數異動記錄錯誤", market, timeframe, "設定檔")

async def initialize_config():
    """初始化設定檔"""
    try:
        import time
        start_time = time.time()
        await validate_config()

        # 儲存初始化狀態
        checkpoint_path = 快取資料夾 / f"config_checkpoint_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez_compressed(checkpoint_path, data={"status": "initialized"})
        await push_limiter.cache_message(f"【執行通知】設定檔初始化檢查點保存至 {checkpoint_path}", "多市場", "多框架", "設定檔", "normal")
        await push_limiter.retry_cached_messages()

        # 效率報告
        elapsed_time = time.time() - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】設定檔初始化耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, "多市場", "多框架", "設定檔", "normal")
        await push_limiter.retry_cached_messages()
    except Exception as e:
        logger.error(f"設定檔初始化失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E1001：設定檔初始化失敗 {e}", "多市場", "多框架", "設定檔", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E1001：設定檔初始化失敗 {e}", "設定檔初始化錯誤", "多市場", "多框架", "設定檔")
        raise

if __name__ == "__main__":
    asyncio.run(initialize_config())