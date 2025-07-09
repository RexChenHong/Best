import asyncio
import aiosqlite
import logging
import datetime
import uuid
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import psutil
import pynvml
import os
import zstandard as zstd
import plotly.graph_objects as go
import aiohttp
import json
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from collections import deque
from cryptography.fernet import Fernet
from 設定檔 import SQLite資料夾, 市場清單, 訓練設備, 快取資料夾, 資源閾值
from 推播通知模組 import 發送錯誤訊息, 發送通知

# 配置日誌
log_dir = SQLite資料夾.parent / "日誌"
log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("工具模組")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
file_handler = TimedRotatingFileHandler(
    filename=log_dir / "utility_logs",
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
    "E901": "錯誤記錄失敗",
    "E902": "硬體監控失敗",
    "E903": "檔案路徑檢查失敗",
    "E904": "快取清理失敗",
    "E905": "DQN 工具參數優化失敗",
    "E906": "加密/解密失敗"
}

# 工具參數緩衝區
_utility_buffer = deque(maxlen=100)

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

async def encrypt_env_file():
    """加密 .env.txt 檔案"""
    try:
        key_path = SQLite資料夾.parent / "secure_key.key"
        env_path = SQLite資料夾.parent / ".env.txt"
        encrypted_path = SQLite資料夾.parent / ".env.encrypted"
        
        if not await 檢查檔案路徑(env_path):
            return False
            
        if not key_path.exists():
            key = Fernet.generate_key()
            with open(key_path, "wb") as key_file:
                key_file.write(key)
                
        with open(key_path, "rb") as key_file:
            key = key_file.read()
        cipher = Fernet(key)
        
        with open(env_path, "rb") as env_file:
            env_data = env_file.read()
        encrypted_data = cipher.encrypt(env_data)
        
        with open(encrypted_path, "wb") as enc_file:
            enc_file.write(encrypted_data)
            
        logger.info(f"[資安] .env.txt 加密完成: {encrypted_path}")
        await push_limiter.cache_message(f"【執行通知】.env.txt 加密完成: {encrypted_path}", "多市場", "多框架", "工具", "normal")
        await push_limiter.retry_cached_messages()
        
        with open(log_dir / f"資安事件_{datetime.datetime.now().strftime('%Y%m%d')}.log", "a", encoding='utf-8') as f:
            f.write(f"[資安事件] 加密 .env.txt\n時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        return True
    except Exception as e:
        logger.error(f"[資安] 加密失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E906：加密 .env.txt 失敗 {e}", "多市場", "多框架", "工具", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E906：加密 .env.txt 失敗 {e}", "加密錯誤", "多市場", "多框架", "工具")
        return False

async def decrypt_env_file():
    """解密 .env.encrypted 檔案"""
    try:
        key_path = SQLite資料夾.parent / "secure_key.key"
        encrypted_path = SQLite資料夾.parent / ".env.encrypted"
        
        if not await 檢查檔案路徑(encrypted_path) or not await 檢查檔案路徑(key_path):
            return None
            
        with open(key_path, "rb") as key_file:
            key = key_file.read()
        cipher = Fernet(key)
        
        with open(encrypted_path, "rb") as enc_file:
            encrypted_data = enc_file.read()
        decrypted_data = cipher.decrypt(encrypted_data).decode('utf-8')
        
        logger.info(f"[資安] .env.encrypted 解密完成")
        await push_limiter.cache_message(f"【執行通知】.env.encrypted 解密完成", "多市場", "多框架", "工具", "normal")
        await push_limiter.retry_cached_messages()
        
        with open(log_dir / f"資安事件_{datetime.datetime.now().strftime('%Y%m%d')}.log", "a", encoding='utf-8') as f:
            f.write(f"[資安事件] 解密 .env.encrypted\n時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        return decrypted_data
    except Exception as e:
        logger.error(f"[資安] 解密失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E906：解密 .env.encrypted 失敗 {e}", "多市場", "多框架", "工具", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E906：解密 .env.encrypted 失敗 {e}", "解密錯誤", "多市場", "多框架", "工具")
        return None

async def check_network_connectivity():
    """檢查網絡連線"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://www.google.com", timeout=5) as response:
                return response.status == 200
    except Exception as e:
        logger.error(f"網絡連線檢查失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E902：網絡連線檢查失敗 {e}", "多市場", "多框架", "工具", "normal")
        await push_limiter.retry_cached_messages()
        return False

async def validate_utility_input(market, timeframe, error_message=None, error_type=None, mode=None, params=None):
    """驗證工具函數輸入"""
    try:
        if market and market not in [mkt for mkt, _ in 市場清單]:
            logger.error(f"[{market}_{timeframe}] 無效市場: {market}")
            return False
        if timeframe and timeframe not in [tf for _, tf in 市場清單]:
            logger.error(f"[{market}_{timeframe}] 無效時間框架: {timeframe}")
            return False
        if error_message and not isinstance(error_message, str):
            logger.error(f"[{market}_{timeframe}] 無效錯誤訊息: {error_message}")
            return False
        if error_type and not isinstance(error_type, str):
            logger.error(f"[{market}_{timeframe}] 無效錯誤類型: {error_type}")
            return False
        if mode and mode not in ["模擬", "交易環境", "測試網", "超參數搜尋", "檢查點", "信號生成", "獎勵計算", "工具", "GUI"]:
            logger.error(f"[{market}_{timeframe}] 無效模式: {mode}")
            return False
        if params:
            if "retry_count" in params and (not isinstance(params["retry_count"], int) or not 3 <= params["retry_count"] <= 10):
                logger.error(f"[{market}_{timeframe}] 無效重試次數: {params['retry_count']}")
                return False
            if "cpu_threshold" in params and (not isinstance(params["cpu_threshold"], float) or not 50.0 <= params["cpu_threshold"] <= 90.0):
                logger.error(f"[{market}_{timeframe}] 無效CPU閾值: {params['cpu_threshold']}")
                return False
        return True
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 輸入驗證失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E901：輸入驗證失敗 {e}", market, timeframe, "工具", "high")
        await push_limiter.retry_cached_messages()
        return False

async def monitor_utility_stability(market, timeframe):
    """監控長期穩定性"""
    try:
        async with aiosqlite.connect(SQLite資料夾 / "錯誤紀錄.db") as conn:
            df = await conn.execute_fetchall("SELECT 重試次數, 時間 FROM 錯誤紀錄 WHERE 市場 = ? AND 時間框架 = ? ORDER BY 時間 DESC LIMIT 100",
                                            (market, timeframe))
            df = pd.DataFrame(df, columns=["重試次數", "時間"])
        if len(df) > 1:
            std_retry = df["重試次數"].std()
            if std_retry > 0.1:
                logger.warning(f"[{market}_{timeframe}] 錯誤頻率標準差過高: {std_retry:.4f}")
                await push_limiter.cache_message(f"錯誤碼E901：錯誤頻率標準差過高 {std_retry:.4f}", market, timeframe, "工具", "high")
                await push_limiter.retry_cached_messages()
                return False
        return True
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 穩定性監控失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E901：穩定性監控失敗 {e}", market, timeframe, "工具", "high")
        await push_limiter.retry_cached_messages()
        return False

async def dqn_optimize_utility_params(market, timeframe, system_state, retry_count=0):
    """使用DQN優化工具參數（重試次數與資源閾值）"""
    try:
        input_dim = 6  # CPU使用率, RAM使用率, GPU使用率, GPU溫度, 錯誤頻率, 市場波動
        output_dim = 4  # 調整重試次數, CPU閾值
        dqn = DQN(input_dim, output_dim).to(訓練設備)
        optimizer = torch.optim.Adam(dqn.parameters(), lr=1e-3)
        gamma = 0.99
        epsilon = 0.1
        episodes = 50

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        base_params = {
            "retry_count": 5,
            "cpu_threshold": 80.0
        }
        best_params = base_params.copy()
        best_reward = -float('inf')

        for episode in range(episodes):
            state = torch.tensor([
                system_state.get("cpu_percent", psutil.cpu_percent()),
                system_state.get("ram_percent", psutil.virtual_memory().percent),
                system_state.get("gpu_util", torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0),
                system_state.get("gpu_temp", gpu_temp),
                system_state.get("error_frequency", 0.0),
                system_state.get("market_volatility", 0.0)
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
                    temp_params["retry_count"] = min(temp_params["retry_count"] + 1, 10)
                elif action == 1:
                    temp_params["retry_count"] = max(temp_params["retry_count"] - 1, 3)
                elif action == 2:
                    temp_params["cpu_threshold"] = min(temp_params["cpu_threshold"] + 5.0, 90.0)
                elif action == 3:
                    temp_params["cpu_threshold"] = max(temp_params["cpu_threshold"] - 5.0, 50.0)

                simulated_load = (
                    state[0].item() * 0.3 +
                    state[1].item() * 0.3 +
                    state[2].item() * 0.2 +
                    state[3].item() * 0.1 +
                    state[4].item() * 0.1
                )
                reward = -simulated_load / 100.0
                if temp_params["retry_count"] > 7 or temp_params["cpu_threshold"] < 60.0 or state[3].item() > 80.0:
                    reward -= 0.5
                    if state[3].item() > 80.0:
                        await push_limiter.cache_message(f"錯誤碼E902：GPU溫度過高 {state[3].item()}°C", market, timeframe, "工具", "high")
                        await push_limiter.retry_cached_messages()

                episode_reward += reward
                next_state = torch.tensor([
                    state[0].item(),
                    state[1].item(),
                    state[2].item(),
                    state[3].item(),
                    state[4].item() * 0.9,
                    state[5].item()
                ], dtype=torch.float32, device=訓練設備)

                q_value = dqn(state)[action]
                target = reward + gamma * dqn(next_state).max()
                loss = nn.MSELoss()(q_value, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                state = next_state
                if reward > best_reward:
                    best_reward = reward
                    best_params = temp_params.copy()

            async with aiosqlite.connect(SQLite資料夾 / "工具優化記錄.db") as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS 工具優化記錄 (
                        id TEXT PRIMARY KEY,
                        市場 TEXT,
                        時間框架 TEXT,
                        重試次數 INTEGER,
                        CPU閾值 REAL,
                        獎勵 REAL,
                        時間 TEXT
                    )
                """)
                await conn.execute("INSERT INTO 工具優化記錄 VALUES (?, ?, ?, ?, ?, ?, ?)",
                                  (str(uuid.uuid4()), market, timeframe, best_params["retry_count"],
                                   best_params["cpu_threshold"], episode_reward,
                                   datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                await conn.commit()

        async with aiosqlite.connect(SQLite資料夾 / "工具優化記錄.db") as conn:
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_市場_時間框架 ON 工具優化記錄 (市場, 時間框架, 時間)")
            df = await conn.execute_fetchall("SELECT 重試次數, CPU閾值 FROM 工具優化記錄 WHERE 市場 = ? AND 時間框架 = ? ORDER BY 時間 DESC LIMIT 10",
                                            (market, timeframe))
            df = pd.DataFrame(df, columns=["重試次數", "CPU閾值"])
        if len(df) > 1:
            std_retry = df["重試次數"].std()
            std_cpu = df["CPU閾值"].std()
            if std_retry > 0.1 or std_cpu > 0.1:
                logger.warning(f"[{market}_{timeframe}] DQN 工具參數穩定性低: 重試={std_retry:.4f}, CPU={std_cpu:.4f}")
                await push_limiter.cache_message(f"錯誤碼E905：DQN 工具參數穩定性低 重試={std_retry:.4f}, CPU={std_cpu:.4f}", market, timeframe, "工具", "high")
                await push_limiter.retry_cached_messages()

        await push_limiter.cache_message(f"【通知】DQN 工具參數優化完成: {market}_{timeframe}\n重試次數: {best_params['retry_count']}\nCPU閾值: {best_params['cpu_threshold']:.2f}%\n獎勵: {best_reward:.4f}", market, timeframe, "工具", "normal")
        await push_limiter.retry_cached_messages()
        pynvml.nvmlShutdown()
        return best_params
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[{market}_{timeframe}] DQN 工具參數優化失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E905：DQN 工具參數優化失敗 重試{retry_count + 1}/5: {e}", "DQN 工具優化錯誤", market, timeframe, "工具")
            await asyncio.sleep(5)
            return await dqn_optimize_utility_params(market, timeframe, system_state, retry_count + 1)
        logger.error(f"[{market}_{timeframe}] DQN 工具參數優化失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E905：DQN 工具參數優化失敗 {e}", market, timeframe, "工具", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E905：DQN 工具參數優化失敗 {e}", "DQN 工具優化錯誤", market, timeframe, "工具")
        return {"retry_count": 5, "cpu_threshold": 80.0}

async def optimize_recovery_strategy(error_type, market, timeframe, system_state, retry_count=0):
    """使用DQN優化自動修復策略"""
    try:
        input_dim = 6  # CPU使用率, RAM使用率, GPU使用率, GPU溫度, 錯誤頻率, 市場波動
        output_dim = 5  # 重試, 切換備用模組, 重啟進程, 降低批次大小, 暫停非必要模組
        dqn = DQN(input_dim, output_dim).to(訓練設備)
        optimizer = torch.optim.Adam(dqn.parameters(), lr=1e-3)
        gamma = 0.99
        epsilon = 0.1
        episodes = 50

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        base_strategy = {"action": "retry", "retry_count": 5, "chunk_size": 50000}
        best_strategy = base_strategy.copy()
        best_reward = -float('inf')

        for episode in range(episodes):
            state = torch.tensor([
                system_state.get("cpu_percent", psutil.cpu_percent()),
                system_state.get("ram_percent", psutil.virtual_memory().percent),
                system_state.get("gpu_util", torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0),
                system_state.get("gpu_temp", gpu_temp),
                system_state.get("error_frequency", 0.0),
                system_state.get("market_volatility", 0.0)
            ], dtype=torch.float32, device=訓練設備)
            episode_reward = 0.0

            for _ in range(10):
                if np.random.random() < epsilon:
                    action = np.random.randint(0, output_dim)
                else:
                    with torch.no_grad():
                        q_values = dqn(state)
                        action = q_values.argmax().item()

                temp_strategy = best_strategy.copy()
                if action == 0:
                    temp_strategy["action"] = "retry"
                    temp_strategy["retry_count"] = min(temp_strategy["retry_count"] + 1, 10)
                elif action == 1:
                    temp_strategy["action"] = "switch_backup"
                elif action == 2:
                    temp_strategy["action"] = "restart_process"
                elif action == 3:
                    temp_strategy["action"] = "reduce_chunk_size"
                    temp_strategy["chunk_size"] = max(temp_strategy["chunk_size"] // 2, 5000)
                elif action == 4:
                    temp_strategy["action"] = "pause_non_essential"

                simulated_load = (
                    state[0].item() * 0.3 +
                    state[1].item() * 0.3 +
                    state[2].item() * 0.2 +
                    state[3].item() * 0.1 +
                    state[4].item() * 0.1
                )
                reward = -simulated_load / 100.0
                if temp_strategy["action"] == "retry" and temp_strategy["retry_count"] > 7:
                    reward -= 0.5
                if state[3].item() > 80.0:
                    reward -= 0.5
                    await push_limiter.cache_message(f"錯誤碼E902：GPU溫度過高 {state[3].item()}°C", market, timeframe, "工具", "high")
                    await push_limiter.retry_cached_messages()

                episode_reward += reward
                next_state = torch.tensor([
                    state[0].item(),
                    state[1].item(),
                    state[2].item(),
                    state[3].item(),
                    state[4].item() * 0.9,
                    state[5].item()
                ], dtype=torch.float32, device=訓練設備)

                q_value = dqn(state)[action]
                target = reward + gamma * dqn(next_state).max()
                loss = nn.MSELoss()(q_value, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                state = next_state
                if reward > best_reward:
                    best_reward = reward
                    best_strategy = temp_strategy.copy()

            async with aiosqlite.connect(SQLite資料夾 / "修復紀錄.db") as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS 修復紀錄 (
                        id TEXT PRIMARY KEY,
                        市場 TEXT,
                        時間框架 TEXT,
                        錯誤類型 TEXT,
                        修復動作 TEXT,
                        重試次數 INTEGER,
                        批次大小 INTEGER,
                        獎勵 REAL,
                        時間 TEXT
                    )
                """)
                await conn.execute("INSERT INTO 修復紀錄 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                  (str(uuid.uuid4()), market, timeframe, error_type,
                                   best_strategy["action"], best_strategy["retry_count"],
                                   best_strategy["chunk_size"], episode_reward,
                                   datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                await conn.commit()

        pynvml.nvmlShutdown()
        return best_strategy
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[{market}_{timeframe}] 修復策略優化失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E901：修復策略優化失敗 重試{retry_count + 1}/5: {e}", "修復策略錯誤", market, timeframe, "工具")
            await asyncio.sleep(5)
            return await optimize_recovery_strategy(error_type, market, timeframe, system_state, retry_count + 1)
        logger.error(f"[{market}_{timeframe}] 修復策略優化失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E901：修復策略優化失敗 {e}", market, timeframe, "工具", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E901：修復策略優化失敗 {e}", "修復策略錯誤", market, timeframe, "工具")
        return {"action": "retry", "retry_count": 5, "chunk_size": 50000}

async def render_resource_trend(market, timeframe, filter_market=None, filter_timeframe=None):
    """生成3D交互式資源使用圖表"""
    try:
        async with aiosqlite.connect(SQLite資料夾 / "資源監控.db") as conn:
            query = "SELECT CPU使用率, RAM使用率, GPU使用率, 時間, 市場, 時間框架 FROM 資源監控"
            params = []
            if filter_market or filter_timeframe:
                query += " WHERE 1=1"
                if filter_market:
                    query += " AND 市場 = ?"
                    params.append(filter_market)
                if filter_timeframe:
                    query += " AND 時間框架 = ?"
                    params.append(filter_timeframe)
            query += " ORDER BY 時間 DESC LIMIT 100"
            df = await conn.execute_fetchall(query, params)
            df = pd.DataFrame(df, columns=["CPU使用率", "RAM使用率", "GPU使用率", "時間", "市場", "時間框架"])
        if df.empty:
            logger.warning(f"[{market}_{timeframe}] 無資源監控數據")
            await push_limiter.cache_message(f"錯誤碼E902：無資源監控數據", market, timeframe, "工具", "normal")
            await push_limiter.retry_cached_messages()
            return None

        df['時間'] = pd.to_datetime(df['時間'])
        fig = go.Figure()
        for mkt, tf in df[['市場', '時間框架']].drop_duplicates().values:
            sub_df = df[(df['市場'] == mkt) & (df['時間框架'] == tf)]
            fig.add_trace(go.Scatter3d(
                x=sub_df['時間'], y=sub_df['CPU使用率'], z=sub_df['RAM使用率'],
                mode='lines+markers', name=f'{mkt}_{tf} CPU-RAM',
                line=dict(color='#00ffcc', width=4),
                marker=dict(size=5, color='#1e90ff')
            ))
            fig.add_trace(go.Scatter3d(
                x=sub_df['時間'], y=sub_df['GPU使用率'], z=sub_df['RAM使用率'],
                mode='lines+markers', name=f'{mkt}_{tf} GPU-RAM',
                line=dict(color='#ff4d4d', width=4),
                marker=dict(size=5, color='#ffcc00')
            ))
        fig.update_layout(
            title=f"{market}_{timeframe} 資源使用趨勢",
            scene=dict(
                xaxis_title="時間",
                yaxis_title="CPU/GPU使用率 (%)",
                zaxis_title="RAM使用率 (%)",
                xaxis=dict(backgroundcolor="#0d1117", gridcolor="#444"),
                yaxis=dict(backgroundcolor="#0d1117", gridcolor="#444"),
                zaxis=dict(backgroundcolor="#0d1117", gridcolor="#444")
            ),
            template="plotly_dark",
            height=600,
            showlegend=True
        )
        plot_path = SQLite資料夾.parent / "圖片" / f"{market}_{timeframe}_Resource_Trend_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.html"
        fig.write_html(plot_path)
        await push_limiter.cache_message(f"【執行通知】生成資源使用圖表 {plot_path}", market, timeframe, "工具", "normal")
        await push_limiter.retry_cached_messages()
        return plot_path
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 資源圖表生成失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E902：資源圖表生成失敗 {e}", market, timeframe, "工具", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E902：資源圖表生成失敗 {e}", "圖表生成錯誤", market, timeframe, "工具")
        return None

async def single_market_error_logging(error_message, error_type, market, timeframe, mode, params=None, system_state=None, retry_count=0):
    """單一市場錯誤記錄與自動修復"""
    try:
        import time
        start_time = time.time()
        if not await validate_utility_input(market, timeframe, error_message, error_type, mode, params):
            return False

        system_state = system_state or {
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": psutil.virtual_memory().percent,
            "gpu_util": torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0,
            "gpu_temp": pynvml.nvmlDeviceGetTemperature(pynvml.nvmlDeviceGetHandleByIndex(0), pynvml.NVML_TEMPERATURE_GPU) if torch.cuda.is_available() else 0,
            "error_frequency": 0.0,
            "market_volatility": 0.0
        }
        optimized_params = await dqn_optimize_utility_params(market, timeframe, system_state)
        max_retries = optimized_params["retry_count"]

        # 異步錯誤記錄
        async with aiosqlite.connect(SQLite資料夾 / "錯誤紀錄.db") as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS 錯誤紀錄 (
                    id TEXT PRIMARY KEY,
                    市場 TEXT,
                    時間框架 TEXT,
                    錯誤訊息 TEXT,
                    錯誤類型 TEXT,
                    模式 TEXT,
                    重試次數 INTEGER,
                    時間 TEXT
                )
            """)
            error_id = str(uuid.uuid4())
            await conn.execute("INSERT INTO 錯誤紀錄 VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                              (error_id, market, timeframe, error_message, error_type, mode, retry_count,
                               datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            await conn.commit()

        # 寫入異動歷程
        change_log_path = SQLite資料夾.parent / "dot" / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        change_log = {
            "UUID": error_id,
            "市場": market,
            "時間框架": timeframe,
            "異動前值": "N/A",
            "異動後值": error_message,
            "異動原因": f"錯誤記錄: {error_type}",
            "時間戳": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        change_df = pd.DataFrame([change_log])
        if change_log_path.exists():
            existing_df = pd.read_excel(change_log_path)
            change_df = pd.concat([existing_df, change_df], ignore_index=True)
        change_df.to_excel(change_log_path, index=False, engine='openpyxl')

        # 自動修復
        strategy = await optimize_recovery_strategy(error_type, market, timeframe, system_state)
        if strategy["action"] == "retry" and retry_count < max_retries:
            logger.info(f"[{market}_{timeframe}] 執行自動修復，重試 {retry_count + 1}/{max_retries}")
            return False
        elif strategy["action"] == "switch_backup":
            logger.info(f"[{market}_{timeframe}] 切換備用模組")
            await push_limiter.cache_message(f"【執行通知】切換備用模組: {market}_{timeframe}", market, timeframe, "工具", "normal")
            await push_limiter.retry_cached_messages()
        elif strategy["action"] == "restart_process":
            logger.info(f"[{market}_{timeframe}] 重啟進程")
            await push_limiter.cache_message(f"【執行通知】重啟進程: {market}_{timeframe}", market, timeframe, "工具", "normal")
            await push_limiter.retry_cached_messages()
        elif strategy["action"] == "reduce_chunk_size":
            logger.info(f"[{market}_{timeframe}] 降低批次大小至: {strategy['chunk_size']}")
            await push_limiter.cache_message(f"【執行通知】降低批次大小至 {strategy['chunk_size']}: {market}_{timeframe}", market, timeframe, "工具", "normal")
            await push_limiter.retry_cached_messages()
        elif strategy["action"] == "pause_non_essential":
            logger.info(f"[{market}_{timeframe}] 暫停非必要模組")
            await push_limiter.cache_message(f"【執行通知】暫停非必要模組: {market}_{timeframe}", market, timeframe, "工具", "normal")
            await push_limiter.retry_cached_messages()

        # 儲存錯誤記錄至緩衝區並檢查穩定性
        _utility_buffer.append({"retry_count": retry_count, "error_message": error_message, "error_type": error_type})
        if len(_utility_buffer) == _utility_buffer.maxlen:
            df_buffer = pd.DataFrame(list(_utility_buffer))
            error_freq = len(df_buffer[df_buffer["error_type"] == error_type]) / _utility_buffer.maxlen
            system_state["error_frequency"] = error_freq
            if error_freq > 0.1:
                logger.warning(f"[{market}_{timeframe}] {error_type} 錯誤頻率過高: {error_freq:.4f}")
                await push_limiter.cache_message(f"錯誤碼E901：{error_type} 錯誤頻率過高 {error_freq:.4f}", market, timeframe, "工具", "high")
                await push_limiter.retry_cached_messages()
            await monitor_utility_stability(market, timeframe)
            await render_resource_trend(market, timeframe)

        # 生成報表
        df = pd.DataFrame([{
            "市場": market,
            "時間框架": timeframe,
            "錯誤訊息": error_message,
            "錯誤類型": error_type,
            "模式": mode,
            "重試次數": retry_count,
            "時間": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }])
        if not df.empty:
            csv_path = SQLite資料夾.parent / "備份" / f"{market}_{timeframe}_Error_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            await push_limiter.cache_message(f"【執行通知】生成錯誤報表 {csv_path}", market, timeframe, "工具", "normal")
            await push_limiter.retry_cached_messages()

        # 效率報告
        elapsed_time = time.time() - start_time
        efficiency_report = (
            f"【效率報告】錯誤記錄與修復耗時：{elapsed_time:.2f}秒，"
            f"CPU：{system_state['cpu_percent']:.1f}%，RAM：{system_state['ram_percent']:.1f}%，"
            f"GPU：{system_state['gpu_util']:.1f}%，GPU溫度：{system_state['gpu_temp']}°C"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, market, timeframe, "工具", "normal")
        await push_limiter.retry_cached_messages()

        return True
    except Exception as e:
        if retry_count < max_retries:
            logger.warning(f"[{market}_{timeframe}] 錯誤記錄失敗，重試 {retry_count + 1}/{max_retries}: {e}")
            await asyncio.sleep(5)
            return await single_market_error_logging(error_message, error_type, market, timeframe, mode, params, system_state, retry_count + 1)
        logger.error(f"[{market}_{timeframe}] 錯誤記錄失敗，重試 {max_retries} 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E901：錯誤記錄失敗 {e}", market, timeframe, "工具", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E901：錯誤記錄失敗 {e}", "錯誤記錄錯誤", market, timeframe, "工具")
        return False

async def 錯誤記錄與自動修復(error_message, error_type, market=None, timeframe=None, mode=None, filepath=None, params=None, system_state=None):
    """錯誤記錄與自動修復接口"""
    try:
        if filepath:
            await 清理快取檔案(market, timeframe)
            torch.cuda.empty_cache()
            await push_limiter.cache_message(f"【執行通知】異常退出，清理快取並釋放GPU記憶體: {market}_{timeframe}", market, timeframe, "工具", "high")
            await push_limiter.retry_cached_messages()
        return await single_market_error_logging(error_message, error_type, market, timeframe, mode, params, system_state)
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 錯誤記錄接口失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E901：錯誤記錄接口失敗 {e}", market, timeframe, "工具", "high")
        await push_limiter.retry_cached_messages()
        return False

async def 監控硬體狀態並降級(chunk_size, max_proc, params=None, system_state=None, retry_count=0):
    """監控硬體狀態並動態降級"""
    try:
        import time
        start_time = time.time()
        system_state = system_state or {
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": psutil.virtual_memory().percent,
            "gpu_util": torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0,
            "gpu_temp": pynvml.nvmlDeviceGetTemperature(pynvml.nvmlDeviceGetHandleByIndex(0), pynvml.NVML_TEMPERATURE_GPU) if torch.cuda.is_available() else 0,
            "error_frequency": 0.0,
            "market_volatility": 0.0
        }
        disk_usage = psutil.disk_usage(str(快取資料夾))
        disk_free_ratio = disk_usage.free / disk_usage.total
        network_ok = await check_network_connectivity()

        optimized_params = await dqn_optimize_utility_params("default", "default", system_state)
        cpu_threshold = optimized_params["cpu_threshold"]

        if (system_state["cpu_percent"] > cpu_threshold or
            system_state["ram_percent"] > 資源閾值["RAM使用率"] * 100 or
            system_state["gpu_util"] > 資源閾值["GPU使用率"] * 100 or
            system_state["gpu_temp"] > 80.0 or
            disk_free_ratio < 0.1 or
            not network_ok):
            new_chunk_size = max(chunk_size // 2, 10000)
            new_max_proc = max(max_proc - 1, 1)
            gpu_switch_count = getattr(監控硬體狀態並降級, 'switch_count', 0) + 1
            setattr(監控硬體狀態並降級, 'switch_count', gpu_switch_count)
            if system_state["gpu_util"] > 資源閾值["GPU使用率"] * 100 or system_state["gpu_temp"] > 80.0:
                torch.cuda.empty_cache()
            logger.warning(f"[硬體監控] 資源超限: CPU={system_state['cpu_percent']:.1f}%, RAM={system_state['ram_percent']:.1f}%, GPU={system_state['gpu_util']:.1f}%, GPU溫度={system_state['gpu_temp']}°C, 磁碟剩餘={disk_free_ratio:.2%}, 網絡={'正常' if network_ok else '異常'}, 降級至 chunk_size={new_chunk_size}, max_proc={new_max_proc}, 切換次數={gpu_switch_count}")
            await push_limiter.cache_message(f"錯誤碼E902：資源超限 CPU={system_state['cpu_percent']:.1f}%, RAM={system_state['ram_percent']:.1f}%, GPU={system_state['gpu_util']:.1f}%, GPU溫度={system_state['gpu_temp']}°C, 磁碟={disk_free_ratio:.2%}, 網絡={'正常' if network_ok else '異常'}, 切換次數={gpu_switch_count}", "default", "default", "工具", "high")
            await push_limiter.retry_cached_messages()

            # 記錄資源監控
            async with aiosqlite.connect(SQLite資料夾 / "資源監控.db") as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS 資源監控 (
                        id TEXT PRIMARY KEY,
                        市場 TEXT,
                        時間框架 TEXT,
                        CPU使用率 REAL,
                        RAM使用率 REAL,
                        GPU使用率 REAL,
                        GPU溫度 REAL,
                        磁碟剩餘比例 REAL,
                        網絡狀態 BOOLEAN,
                        時間 TEXT
                    )
                """)
                await conn.execute("INSERT INTO 資源監控 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                  (str(uuid.uuid4()), "default", "default", system_state["cpu_percent"],
                                   system_state["ram_percent"], system_state["gpu_util"],
                                   system_state["gpu_temp"], disk_free_ratio, network_ok,
                                   datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                await conn.commit()

            # 寫入異動歷程
            change_log_path = SQLite資料夾.parent / "dot" / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
            change_log = {
                "UUID": str(uuid.uuid4()),
                "市場": "default",
                "時間框架": "default",
                "異動前值": f"chunk_size={chunk_size}, max_proc={max_proc}",
                "異動後值": f"chunk_size={new_chunk_size}, max_proc={new_max_proc}",
                "異動原因": "硬體資源超限",
                "時間戳": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            change_df = pd.DataFrame([change_log])
            if change_log_path.exists():
                existing_df = pd.read_excel(change_log_path)
                change_df = pd.concat([existing_df, change_df], ignore_index=True)
            change_df.to_excel(change_log_path, index=False, engine='openpyxl')

            return new_chunk_size, new_max_proc

        # 效率報告
        elapsed_time = time.time() - start_time
        efficiency_report = (
            f"【效率報告】硬體監控耗時：{elapsed_time:.2f}秒，"
            f"CPU：{system_state['cpu_percent']:.1f}%，RAM：{system_state['ram_percent']:.1f}%，"
            f"GPU：{system_state['gpu_util']:.1f}%，GPU溫度：{system_state['gpu_temp']}°C，磁碟：{disk_free_ratio:.2%}"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, "default", "default", "工具", "normal")
        await push_limiter.retry_cached_messages()
        return chunk_size, max_proc
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[硬體監控] 硬體監控失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E902：硬體監控失敗 重試{retry_count + 1}/5: {e}", "硬體監控錯誤", "default", "default", "工具")
            await asyncio.sleep(5)
            return await 監控硬體狀態並降級(chunk_size, max_proc, params, system_state, retry_count + 1)
        logger.error(f"[硬體監控] 硬體監控失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E902：硬體監控失敗 {e}", "default", "default", "工具", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E902：硬體監控失敗 {e}", "硬體監控錯誤", "default", "default", "工具")
        return chunk_size, max_proc

async def 檢查檔案路徑(filepath, market=None, timeframe=None, retry_count=0):
    """檢查檔案路徑有效性"""
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"[{market}_{timeframe}] 檔案路徑無效: {filepath}")
            await push_limiter.cache_message(f"錯誤碼E903：檔案路徑無效 {filepath}", market, timeframe, "工具", "high")
            await push_limiter.retry_cached_messages()
            return False
        if not filepath.is_file():
            logger.error(f"[{market}_{timeframe}] 非檔案: {filepath}")
            await push_limiter.cache_message(f"錯誤碼E903：非檔案 {filepath}", market, timeframe, "工具", "high")
            await push_limiter.retry_cached_messages()
            return False
        return True
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[{market}_{timeframe}] 檔案路徑檢查失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E903：檔案路徑檢查失敗 重試{retry_count + 1}/5: {e}", "檔案路徑錯誤", market, timeframe, "工具")
            await asyncio.sleep(5)
            return await 檢查檔案路徑(filepath, market, timeframe, retry_count + 1)
        logger.error(f"[{market}_{timeframe}] 檔案路徑檢查失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E903：檔案路徑檢查失敗 {e}", market, timeframe, "工具", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E903：檔案路徑檢查失敗 {e}", "檔案路徑錯誤", market, timeframe, "工具")
        return False

async def 清理快取檔案(market=None, timeframe=None, retry_count=0):
    """清理快取檔案"""
    try:
        import time
        start_time = time.time()
        disk_usage = psutil.disk_usage(str(快取資料夾))
        disk_free_ratio = disk_usage.free / disk_usage.total
        if disk_free_ratio < 0.1:
            cache_files = list(快取資料夾.glob("*.npz"))
            for file in cache_files:
                file_time = datetime.datetime.fromtimestamp(file.stat().st_mtime)
                if (datetime.datetime.now() - file_time).days > 7:
                    file.unlink()
                    logger.info(f"[{market}_{timeframe}] 清理快取檔案: {file}")
                    await push_limiter.cache_message(f"【執行通知】清理快取檔案: {file}", market, timeframe, "工具", "normal")
                    await push_limiter.retry_cached_messages()

            # 寫入異動歷程
            change_log_path = SQLite資料夾.parent / "dot" / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
            change_log = {
                "UUID": str(uuid.uuid4()),
                "市場": market or "default",
                "時間框架": timeframe or "default",
                "異動前值": f"磁碟剩餘={disk_free_ratio:.2%}",
                "異動後值": f"清理後磁碟剩餘={psutil.disk_usage(str(快取資料夾)).free / psutil.disk_usage(str(快取資料夾)).total:.2%}",
                "異動原因": "快取檔案清理",
                "時間戳": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            change_df = pd.DataFrame([change_log])
            if change_log_path.exists():
                existing_df = pd.read_excel(change_log_path)
                change_df = pd.concat([existing_df, change_df], ignore_index=True)
            change_df.to_excel(change_log_path, index=False, engine='openpyxl')

        # 效率報告
        elapsed_time = time.time() - start_time
        efficiency_report = (
            f"【效率報告】快取清理耗時：{elapsed_time:.2f}秒，"
            f"磁碟剩餘：{disk_free_ratio:.2%}"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, market, timeframe, "工具", "normal")
        await push_limiter.retry_cached_messages()
        return True
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[{market}_{timeframe}] 快取清理失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E904：快取清理失敗 重試{retry_count + 1}/5: {e}", "快取清理錯誤", market, timeframe, "工具")
            await asyncio.sleep(5)
            return await 清理快取檔案(market, timeframe, retry_count + 1)
        logger.error(f"[{market}_{timeframe}] 快取清理失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E904：快取清理失敗 {e}", market, timeframe, "工具", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E904：快取清理失敗 {e}", "快取清理錯誤", market, timeframe, "工具")
        return False

async def optimize_all_markets_utility_params(system_states, retry_count=0):
    """為所有市場與時間框架優化工具參數"""
    try:
        import time
        start_time = time.time()
        results = []
        best_params = None
        best_reward = -float('inf')

        tasks = []
        for (market, timeframe) in 市場清單:
            system_state = system_states.get((market, timeframe), {
                "cpu_percent": psutil.cpu_percent(),
                "ram_percent": psutil.virtual_memory().percent,
                "gpu_util": torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0,
                "gpu_temp": pynvml.nvmlDeviceGetTemperature(pynvml.nvmlDeviceGetHandleByIndex(0), pynvml.NVML_TEMPERATURE_GPU) if torch.cuda.is_available() else 0,
                "error_frequency": 0.0,
                "market_volatility": 0.0
            })
            task = dqn_optimize_utility_params(market, timeframe, system_state)
            tasks.append(task)
        
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        for (market, timeframe), result in zip(市場清單, results_list):
            if isinstance(result, Exception):
                logger.error(f"[{market}_{timeframe}] DQN 工具參數優化失敗: {result}")
                await push_limiter.cache_message(f"錯誤碼E905：DQN 工具參數優化失敗 {result}", market, timeframe, "工具", "high")
                await push_limiter.retry_cached_messages()
                results.append({"市場": market, "時間框架": timeframe, "重試次數": 5, "CPU閾值": 80.0, "獎勵": -float('inf')})
            else:
                optimized_params = result
                system_state = system_states.get((market, timeframe))
                reward = -(
                    system_state["cpu_percent"] * 0.3 +
                    system_state["ram_percent"] * 0.3 +
                    system_state["gpu_util"] * 0.2 +
                    system_state["gpu_temp"] * 0.1 +
                    system_state["error_frequency"] * 0.1
                ) / 100.0
                results.append({
                    "市場": market,
                    "時間框架": timeframe,
                    "重試次數": optimized_params["retry_count"],
                    "CPU閾值": optimized_params["cpu_threshold"],
                    "獎勵": reward
                })
                if reward > best_reward:
                    best_reward = reward
                    best_params = optimized_params.copy()

        async with aiosqlite.connect(SQLite資料夾 / "泛用工具參數.db") as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS 泛用工具參數 (
                    id TEXT PRIMARY KEY,
                    參數 TEXT,
                    獎勵 REAL,
                    時間 TEXT
                )
            """)
            await conn.execute("INSERT INTO 泛用工具參數 VALUES (?, ?, ?, ?)",
                              (str(uuid.uuid4()), json.dumps(best_params, ensure_ascii=False),
                               best_reward, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            await conn.commit()

        df = pd.DataFrame(results)
        if not df.empty:
            csv_path = SQLite資料夾.parent / "備份" / f"All_Markets_Utility_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            await push_limiter.cache_message(f"【執行通知】生成全市場工具報表 {csv_path}", "多市場", "多框架", "工具", "normal")
            await push_limiter.retry_cached_messages()

        # 寫入異動歷程
        change_log_path = SQLite資料夾.parent / "dot" / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        change_log = {
            "UUID": str(uuid.uuid4()),
            "市場": "多市場",
            "時間框架": "多框架",
            "異動前值": "N/A",
            "異動後值": json.dumps(best_params, ensure_ascii=False),
            "異動原因": "全市場工具參數優化",
            "時間戳": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        change_df = pd.DataFrame([change_log])
        if change_log_path.exists():
            existing_df = pd.read_excel(change_log_path)
            change_df = pd.concat([existing_df, change_df], ignore_index=True)
        change_df.to_excel(change_log_path, index=False, engine='openpyxl')

        elapsed_time = time.time() - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        gpu_temp = pynvml.nvmlDeviceGetTemperature(pynvml.nvmlDeviceGetHandleByIndex(0), pynvml.NVML_TEMPERATURE_GPU) if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】全市場工具優化耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%，GPU溫度：{gpu_temp}°C"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, "多市場", "多框架", "工具", "normal")
        await push_limiter.retry_cached_messages()

        return best_params, best_reward
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[多市場_多框架] 全市場工具優化失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E901：全市場工具優化失敗 重試{retry_count + 1}/5: {e}", "全市場工具錯誤", "多市場", "多框架", "工具")
            await asyncio.sleep(5)
            return await optimize_all_markets_utility_params(system_states, retry_count + 1)
        logger.error(f"[多市場_多框架] 全市場工具優化失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E901：全市場工具優化失敗 {e}", "多市場", "多框架", "工具", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E901：全市場工具優化失敗 {e}", "全市場工具錯誤", "多市場", "多框架", "工具")
        return None, 0.0