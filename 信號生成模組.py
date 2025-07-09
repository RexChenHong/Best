import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import asyncio
import aiosqlite
import uuid
import datetime
import plotly.graph_objects as go
import json
import psutil
import time
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from collections import deque
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from scipy.interpolate import interp1d
from 設定檔 import 訓練設備, SQLite資料夾, 市場清單, 訓練參數, 資源閾值, 點差, 點值
from 工具模組 import 錯誤記錄與自動修復, 監控硬體狀態並降級, validate_utility_input
from 推播通知模組 import 發送錯誤訊息, 發送通知
from 交易環境模組 import single_market_trading_env
from 獎勵計算模組 import calculate_single_market_reward

# 配置日誌
log_dir = SQLite資料夾.parent / "日誌"
log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("信號生成模組")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
file_handler = TimedRotatingFileHandler(
    filename=log_dir / "signal_generation_logs",
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
    "E501": "信號生成失敗",
    "E502": "信號穩定性驗證失敗",
    "E503": "硬體資源超限",
    "E504": "DQN 信號參數優化失敗",
    "E505": "異常信號過濾失敗"
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

    async def cache_message(self, message, market, timeframe, mode):
        async with aiosqlite.connect(self.cache_db) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS push_cache (
                    id TEXT PRIMARY KEY,
                    市場 TEXT,
                    時間框架 TEXT,
                    模式 TEXT,
                    訊息 TEXT,
                    時間 TEXT
                )
            """)
            await conn.execute("INSERT INTO push_cache VALUES (?, ?, ?, ?, ?, ?)",
                              (str(uuid.uuid4()), market, timeframe, mode, message,
                               datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            await conn.commit()

    async def retry_cached_messages(self):
        async with aiosqlite.connect(self.cache_db) as conn:
            messages = await conn.execute_fetchall("SELECT 市場, 時間框架, 模式, 訊息 FROM push_cache")
            for market, timeframe, mode, message in messages:
                if await self.can_push():
                    await 發送通知(message, market, timeframe, mode)
                    await conn.execute("DELETE FROM push_cache WHERE 訊息 = ?", (message,))
                    await conn.commit()
                    self.push_timestamps.append(datetime.datetime.now())

push_limiter = PushLimiter()

# 信號緩衝區
_signal_buffer = deque(maxlen=100)
checkpoint = 0

# 模型定義
class SignalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, layers=2, model_type="MLP"):
        super(SignalModel, self).__init__()
        self.model_type = model_type
        if model_type == "LSTM":
            self.lstm = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True, dropout=訓練參數["dropout"]["值"])
            self.fc = nn.Linear(hidden_dim, 3)  # 輸出: 買入(1), 賣出(-1), 無信號(0)
        else:  # MLP
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
                hidden_dim //= 2
            self.fc = nn.Linear(hidden_dim // 2, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(訓練參數["dropout"]["值"])

    def forward(self, x):
        if self.model_type == "LSTM":
            x, _ = self.lstm(x)
            x = x[:, -1, :]  # 取最後時間步
            x = self.fc(x)
        else:
            for layer in self.layers:
                x = self.relu(layer(x))
                x = self.dropout(x)
            x = self.fc(x)
        return torch.softmax(x, dim=-1)

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

async def validate_signal_input(data, market, timeframe):
    """驗證信號生成輸入數據"""
    try:
        if not await validate_utility_input(market, timeframe, mode="信號生成"):
            return False
        required_cols = ["open", "high", "low", "close", "timestamp", "HMA_16", "SMA50", "ATR_14", "VHF", "PivotHigh", "PivotLow"]
        if not isinstance(data, pd.DataFrame) or not all(col in data.columns for col in required_cols):
            logger.error(f"[{market}_{timeframe}] 缺少必要欄位: {required_cols}")
            await push_limiter.cache_message(f"錯誤碼E501：缺少必要欄位 {required_cols}", market, timeframe, "信號生成")
            await push_limiter.retry_cached_messages()
            return False
        if data.empty or data[required_cols].isnull().any().any():
            logger.error(f"[{market}_{timeframe}] 數據為空或存在缺失值")
            await push_limiter.cache_message(f"錯誤碼E501：數據為空或存在缺失值", market, timeframe, "信號生成")
            await push_limiter.retry_cached_messages()
            return False
        return True
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 輸入驗證失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E501：輸入驗證失敗 {e}", market, timeframe, "信號生成")
        await push_limiter.retry_cached_messages()
        return False

async def check_signal_stability(signals, market, timeframe, max_consecutive=10):
    """檢查信號穩定性"""
    try:
        signals = np.array(signals)
        consecutive_counts = []
        current_count = 1
        current_signal = signals[0] if len(signals) > 0 else 0
        for i in range(1, len(signals)):
            if signals[i] == current_signal:
                current_count += 1
            else:
                consecutive_counts.append(current_count)
                current_signal = signals[i]
                current_count = 1
        consecutive_counts.append(current_count)
        max_consecutive_count = max(consecutive_counts) if consecutive_counts else 0
        signal_distribution = {
            "buy": np.sum(signals == 1) / len(signals) if len(signals) > 0 else 0,
            "sell": np.sum(signals == -1) / len(signals) if len(signals) > 0 else 0,
            "hold": np.sum(signals == 0) / len(signals) if len(signals) > 0 else 0
        }
        if max_consecutive_count > max_consecutive:
            logger.warning(f"[{market}_{timeframe}] 信號穩定性低，最大連續信號數: {max_consecutive_count}")
            await push_limiter.cache_message(
                f"錯誤碼E502：信號穩定性低，最大連續信號數 {max_consecutive_count}", 
                market, timeframe, "信號生成")
            await push_limiter.retry_cached_messages()
            return False, signal_distribution
        return True, signal_distribution
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 信號穩定性檢查失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E502：信號穩定性檢查失敗: {e}", market, timeframe, "信號生成")
        await push_limiter.retry_cached_messages()
        return False, {}

async def filter_anomalous_signals(signals, data, market, timeframe):
    """過濾異常信號"""
    try:
        point_spread = 點差.get(market, 點差["default"])
        point_value = 點值.get(market, 點值["default"])
        atr = data['ATR_14'].mean() if 'ATR_14' in data.columns else 0.0
        threshold = point_spread * point_value * 5
        signals = np.array(signals)
        anomalous = np.abs(signals) > (1.0 + atr / data['close'].mean()) if atr > 0 else np.abs(signals) > 1.0
        anomaly_rate = np.mean(anomalous)
        if anomaly_rate > 0.05:
            logger.warning(f"[{market}_{timeframe}] 異常信號比例過高: {anomaly_rate:.4f}")
            await push_limiter.cache_message(
                f"錯誤碼E505：異常信號比例過高 {anomaly_rate:.4f}", 
                market, timeframe, "信號生成")
            await push_limiter.retry_cached_messages()
            signals[anomalous] = 0
            # 插值修復異常信號
            valid_mask = ~anomalous
            if valid_mask.sum() > 1:
                interp_func = interp1d(np.where(valid_mask)[0], signals[valid_mask], kind='linear', bounds_error=False, fill_value="extrapolate")
                signals[anomalous] = interp_func(np.where(anomalous)[0])
        return signals.tolist(), anomaly_rate
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 異常信號過濾失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E505：異常信號過濾失敗: {e}", market, timeframe, "信號生成")
        await push_limiter.retry_cached_messages()
        return signals, 0.0

async def train_signal_model(model, data, market, timeframe, params):
    """訓練信號模型（KFold交叉驗證）"""
    try:
        import time
        start_time = time.time()
        input_features = ["open", "high", "low", "close", "HMA_16", "SMA50", "ATR_14", "VHF", "PivotHigh", "PivotLow"]
        features = data[input_features].dropna()
        if len(features) == 0:
            logger.error(f"[{market}_{timeframe}] 無有效特徵數據")
            await push_limiter.cache_message(f"錯誤碼E501：無有效特徵數據", market, timeframe, "信號生成")
            await push_limiter.retry_cached_messages()
            return None

        # 假設目標信號（模擬數據）
        targets = np.random.randint(-1, 2, size=len(features))  # -1, 0, 1
        dataset = TensorDataset(
            torch.tensor(features.values, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.long) + 1  # 轉為 0, 1, 2
        )
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        optimizer = torch.optim.Adam(model.parameters(), lr=訓練參數["學習率"]["值"])
        loss_fn = nn.CrossEntropyLoss()
        training_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset))), 1):
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=params["batch_size"], shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=params["batch_size"], shuffle=False)

            for epoch in range(10):  # 每個 fold 訓練 10 次
                model.train()
                train_loss = 0.0
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(訓練設備), batch_y.to(訓練設備)
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = loss_fn(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(訓練設備), batch_y.to(訓練設備)
                        outputs = model(batch_x)
                        loss = loss_fn(outputs, batch_y)
                        val_loss += loss.item()

                training_results.append({
                    "fold": fold,
                    "epoch": epoch + 1,
                    "train_loss": train_loss / len(train_loader),
                    "val_loss": val_loss / len(val_loader)
                })

                if (epoch + 1) % (10 // 2) == 0:
                    await push_limiter.cache_message(
                        f"【進度通知】{market}_{timeframe} Fold {fold} Epoch {epoch + 1}/10: "
                        f"Train Loss={train_loss / len(train_loader):.4f}, Val Loss={val_loss / len(val_loader):.4f}",
                        market, timeframe, "信號生成")
                    await push_limiter.retry_cached_messages()

        # 儲存訓練結果
        async with aiosqlite.connect(SQLite資料夾 / "信號訓練記錄.db") as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS 信號訓練記錄 (
                    id TEXT PRIMARY KEY,
                    市場 TEXT,
                    時間框架 TEXT,
                    fold INTEGER,
                    epoch INTEGER,
                    train_loss REAL,
                    val_loss REAL,
                    時間 TEXT
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_市場_時間框架 ON 信號訓練記錄 (市場, 時間框架, 時間)")
            for result in training_results:
                await conn.execute("INSERT INTO 信號訓練記錄 VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                 (str(uuid.uuid4()), market, timeframe, result["fold"], result["epoch"],
                                  result["train_loss"], result["val_loss"],
                                  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                await conn.commit()

        # 寫入異動歷程
        change_log_path = SQLite資料夾.parent / "dot" / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        change_log = {
            "UUID": str(uuid.uuid4()),
            "市場": market,
            "時間框架": timeframe,
            "異動前值": "N/A",
            "異動後值": f"訓練記錄數={len(training_results)}, 平均Train Loss={np.mean([r['train_loss'] for r in training_results]):.4f}",
            "異動原因": "信號模型訓練",
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
            f"【效率報告】信號模型訓練耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, market, timeframe, "信號生成")
        await push_limiter.retry_cached_messages()
        return model
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 信號模型訓練失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E501：信號模型訓練失敗 {e}", market, timeframe, "信號生成")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E501：信號模型訓練失敗 {e}", "信號模型訓練錯誤", market, timeframe, "信號生成")
        return None

async def dqn_optimize_signal_params(data, market, timeframe, params, model, retry_count=0):
    """使用 DQN 強化學習優化信號參數"""
    try:
        import time
        start_time = time.time()
        input_dim = 7  # 資金, 回撤, 勝率, f1分數, 穩定性, ATR, HMA_16
        output_dim = 8  # 調整 hma_threshold, sma_threshold, signal_threshold (增/減)
        dqn = DQN(input_dim, output_dim).to(訓練設備)
        optimizer = torch.optim.Adam(dqn.parameters(), lr=訓練參數["學習率"]["值"])
        gamma = 0.99
        epsilon = 0.1
        episodes = 50

        initial_funds = params.get("initial_funds", 1000.0)
        best_params = params.copy()
        best_reward = -float('inf')

        for episode in range(episodes):
            state = torch.tensor([
                initial_funds,
                0.0,  # 初始回撤
                0.0,  # 初始勝率
                0.0,  # 初始 f1分數
                0.0,  # 初始穩定性
                data["ATR_14"].iloc[-1] if not data.empty else params.get("atr_period", 14),
                data["HMA_16"].iloc[-1] if not data.empty else params.get("hma_period", 16)
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
                    temp_params["hma_threshold"] = min(temp_params["hma_threshold"] + 0.05, 1.0)
                elif action == 1:
                    temp_params["hma_threshold"] = max(temp_params["hma_threshold"] - 0.05, 0.1)
                elif action == 2:
                    temp_params["sma_threshold"] = min(temp_params["sma_threshold"] + 0.05, 1.0)
                elif action == 3:
                    temp_params["sma_threshold"] = max(temp_params["sma_threshold"] - 0.05, 0.1)
                elif action == 4:
                    temp_params["signal_threshold"] = min(temp_params["signal_threshold"] + 0.05, 0.95)
                elif action == 5:
                    temp_params["signal_threshold"] = max(temp_params["signal_threshold"] - 0.05, 0.5)
                elif action == 6:
                    temp_params["atr_period"] = min(temp_params["atr_period"] + 1, 30)
                elif action == 7:
                    temp_params["atr_period"] = max(temp_params["atr_period"] - 1, 7)

                signals = await generate_single_market_signal(data, market, timeframe, temp_params, model)
                if signals is None:
                    temp_reward = -1.0
                else:
                    market_signal_mapping = {(market, timeframe): {"信號": signals, "價格": data["close"].values}}
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
                    data["ATR_14"].iloc[-1] if not data.empty else temp_params.get("atr_period", 14),
                    data["HMA_16"].iloc[-1] if not data.empty else temp_params.get("hma_period", 16)
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
            async with aiosqlite.connect(SQLite資料夾 / "信號優化記錄.db") as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS 信號優化記錄 (
                        id TEXT PRIMARY KEY,
                        市場 TEXT,
                        時間框架 TEXT,
                        hma_threshold REAL,
                        sma_threshold REAL,
                        signal_threshold REAL,
                        atr_period INTEGER,
                        獎勵 REAL,
                        時間 TEXT
                    )
                """)
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_市場_時間框架 ON 信號優化記錄 (市場, 時間框架, 時間)")
                await conn.execute("INSERT INTO 信號優化記錄 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                 (str(uuid.uuid4()), market, timeframe, best_params["hma_threshold"],
                                  best_params["sma_threshold"], best_params["signal_threshold"],
                                  best_params["atr_period"], episode_reward,
                                  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                await conn.commit()

        # 檢查穩定性
        async with aiosqlite.connect(SQLite資料夾 / "信號優化記錄.db") as conn:
            df = await conn.execute_fetchall(
                "SELECT hma_threshold, sma_threshold, signal_threshold, atr_period FROM 信號優化記錄 WHERE 市場 = ? AND 時間框架 = ? ORDER BY 時間 DESC LIMIT 10",
                (market, timeframe))
            df = pd.DataFrame(df, columns=["hma_threshold", "sma_threshold", "signal_threshold", "atr_period"])
        if len(df) > 1:
            for column in df.columns:
                if df[column].std() > 0.1:
                    logger.warning(f"[{market}_{timeframe}] DQN 信號參數 {column} 穩定性低，標準差: {df[column].std():.4f}")
                    await push_limiter.cache_message(
                        f"錯誤碼E504：DQN 信號參數 {column} 穩定性低，標準差 {df[column].std():.4f}",
                        market, timeframe, "信號生成")
                    await push_limiter.retry_cached_messages()

        # 寫入異動歷程
        change_log_path = SQLite資料夾.parent / "dot" / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        change_log = {
            "UUID": str(uuid.uuid4()),
            "市場": market,
            "時間框架": timeframe,
            "異動前值": f"原始參數={json.dumps(params, ensure_ascii=False)}",
            "異動後值": f"優化參數={json.dumps(best_params, ensure_ascii=False)}, 獎勵={best_reward:.4f}",
            "異動原因": "DQN 信號參數優化",
            "時間戳": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        change_df = pd.DataFrame([change_log])
        if change_log_path.exists():
            existing_df = pd.read_excel(change_log_path)
            change_df = pd.concat([existing_df, change_df], ignore_index=True)
        change_df.to_excel(change_log_path, index=False, engine='openpyxl')

        # 推播優化結果
        message = (
            f"【DQN 信號參數優化完成】\n"
            f"市場: {market}_{timeframe}\n"
            f"獎勵: {best_reward:.4f}\n"
            f"HMA閾值: {best_params['hma_threshold']:.4f}\n"
            f"SMA閾值: {best_params['sma_threshold']:.4f}\n"
            f"信號閾值: {best_params['signal_threshold']:.4f}\n"
            f"ATR週期: {best_params['atr_period']}"
        )
        await push_limiter.cache_message(message, market, timeframe, "信號生成")
        await push_limiter.retry_cached_messages()

        # 效率報告
        elapsed_time = time.time() - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】DQN 信號參數優化耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, market, timeframe, "信號生成")
        await push_limiter.retry_cached_messages()
        return best_params
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[{market}_{timeframe}] DQN 信號參數優化失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E504：DQN 信號參數優化失敗 重試{retry_count + 1}/5: {e}", "DQN 信號優化錯誤", market, timeframe, "信號生成")
            await asyncio.sleep(5)
            return await dqn_optimize_signal_params(data, market, timeframe, params, model, retry_count + 1)
        logger.error(f"[{market}_{timeframe}] DQN 信號參數優化失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E504：DQN 信號參數優化失敗 {e}", market, timeframe, "信號生成")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E504：DQN 信號參數優化失敗 {e}", "DQN 信號優化錯誤", market, timeframe, "信號生成")
        return params

async def generate_single_market_signal(data, market, timeframe, params=None, model=None, retry_count=0):
    """生成單一市場交易信號"""
    try:
        import time
        start_time = time.time()
        if not await validate_signal_input(data, market, timeframe):
            return None

        # 硬體監控
        batch_size = params.get("batch_size", 訓練參數["batch_size"]["值"]) if params else 訓練參數["batch_size"]["值"]
        batch_size, _ = await 監控硬體狀態並降級(batch_size, 2)
        if batch_size < 8:
            logger.error(f"[{market}_{timeframe}] 硬體資源超限，批次大小 {batch_size}")
            await push_limiter.cache_message(f"錯誤碼E503：硬體資源超限，批次大小 {batch_size}", market, timeframe, "信號生成")
            await push_limiter.retry_cached_messages()
            return None
        torch.cuda.empty_cache()

        # 參數初始化
        if params is None:
            params = {
                "initial_funds": 1000.0,
                "batch_size": 訓練參數["batch_size"]["值"],
                "hma_threshold": 訓練參數["signal_threshold"]["值"],
                "sma_threshold": 訓練參數["signal_threshold"]["值"],
                "sma_period": 訓練參數["SMA週期"]["值"],
                "hma_period": 訓練參數["HMA週期"]["值"],
                "atr_period": 訓練參數["ATR週期"]["值"],
                "vhf_period": 訓練參數["VHF週期"]["值"],
                "pivot_period": 訓練參數["Pivot週期"]["值"],
                "signal_threshold": 訓練參數["signal_threshold"]["值"],
                "single_loss_limit": 訓練參數["single_loss_limit"]["值"],
                "daily_loss_limit": 訓練參數["daily_loss_limit"]["值"],
                "custom_indicators": 訓練參數.get("custom_indicators", {})
            }

        # 訓練模型
        if model is None:
            input_dim = len(["open", "high", "low", "close", "HMA_16", "SMA50", "ATR_14", "VHF", "PivotHigh", "PivotLow"])
            model = SignalModel(input_dim, hidden_dim=訓練參數["神經元數"]["值"], layers=訓練參數["層數"]["值"], model_type=訓練參數["optimizer"]["值"]).to(訓練設備)
            model = await train_signal_model(model, data, market, timeframe, params)
            if model is None:
                return None

        # DQN 優化信號參數
        optimized_params = await dqn_optimize_signal_params(data, market, timeframe, params, model)
        params.update(optimized_params)

        # 數據準備
        input_features = ["open", "high", "low", "close", "HMA_16", "SMA50", "ATR_14", "VHF", "PivotHigh", "PivotLow"]
        features = data[input_features].dropna()
        if len(features) == 0:
            logger.error(f"[{market}_{timeframe}] 無有效特徵數據")
            await push_limiter.cache_message(f"錯誤碼E501：無有效特徵數據", market, timeframe, "信號生成")
            await push_limiter.retry_cached_messages()
            return None
        features_tensor = torch.tensor(features.values, dtype=torch.float32)

        # 動態信號規則
        signals = []
        signal_threshold = params["signal_threshold"]
        if market in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "BNBUSDT"]:
            # 加密貨幣：HMA與SMA交叉 + ATR確認
            for i in range(1, len(data)):
                if (data['HMA_16'].iloc[i] > data['SMA50'].iloc[i] * (1 + params["hma_threshold"]) and
                    data['ATR_14'].iloc[i] > data['ATR_14'].mean()):
                    signals.append(1.0)  # 買入
                elif (data['HMA_16'].iloc[i] < data['SMA50'].iloc[i] * (1 - params["hma_threshold"]) and
                      data['ATR_14'].iloc[i] > data['ATR_14'].mean()):
                    signals.append(-1.0)  # 賣出
                else:
                    signals.append(0.0)  # 持有
        else:
            # CFD：VHF與Pivot確認
            for i in range(1, len(data)):
                if (data['VHF'].iloc[i] > 0.5 and not pd.isna(data['PivotHigh'].iloc[i]) and
                    data['ATR_14'].iloc[i] > data['ATR_14'].mean()):
                    signals.append(-1.0)  # 賣出
                elif (data['VHF'].iloc[i] < 0.3 and not pd.isna(data['PivotLow'].iloc[i]) and
                      data['ATR_14'].iloc[i] > data['ATR_14'].mean()):
                    signals.append(1.0)  # 買入
                else:
                    signals.append(0.0)  # 持有

        # 自定義信號規則
        for name, config in params.get("custom_indicators", {}).items():
            if config.get("enabled", False):
                custom_signals = eval(config["function"])(data)
                signals = [max(s, cs) if s != 0 else cs for s, cs in zip(signals, custom_signals)]

        # 模型推理（批量處理）
        dataset = TensorDataset(features_tensor)
        loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=False)
        model_signals = []
        model.eval()
        gpu_switch_count = 0
        for batch in loader:
            batch = batch[0]
            with torch.no_grad():
                if torch.cuda.is_available() and torch.cuda.memory_allocated(訓練設備) < 資源閾值["GPU使用率"] * torch.cuda.get_device_properties(訓練設備).total_memory:
                    outputs = model(batch.to(訓練設備))
                else:
                    gpu_switch_count += 1
                    logger.warning(f"GPU記憶體超限，切換至CPU，當前切換次數: {gpu_switch_count}")
                    await push_limiter.cache_message(f"【通知】GPU記憶體超限，切換至CPU，當前切換次數: {gpu_switch_count}", market, timeframe, "信號生成")
                    await push_limiter.retry_cached_messages()
                    outputs = model(batch.cpu())
            model_signals.extend(torch.argmax(outputs, dim=1).cpu().numpy() - 1)  # 轉為 -1/0/1
        signals = [s if abs(s) <= signal_threshold else ms for s, ms in zip(signals, model_signals)]

        # 過濾異常信號
        signals, anomaly_rate = await filter_anomalous_signals(signals, data, market, timeframe)

        # 檢查信號穩定性
        is_stable, signal_distribution = await check_signal_stability(signals, market, timeframe)
        if not is_stable:
            return None

        # 快取信號
        cache_path = SQLite資料夾.parent / "cache" / f"signal_cache_{market}_{timeframe}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez_compressed(cache_path, signals=signals, distribution=signal_distribution, anomaly_rate=anomaly_rate)
        await push_limiter.cache_message(f"【執行通知】信號已快取至 {cache_path}", market, timeframe, "信號生成")
        await push_limiter.retry_cached_messages()

        # 儲存信號記錄
        _signal_buffer.append(signals[-1] if len(signals) > 0 else 0.0)
        async with aiosqlite.connect(SQLite資料夾 / "信號記錄.db") as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS 信號生成記錄 (
                    id TEXT PRIMARY KEY,
                    市場 TEXT,
                    時間框架 TEXT,
                    信號數量 INTEGER,
                    信號分佈 TEXT,
                    異常比例 REAL,
                    生成耗時 REAL,
                    時間 TEXT
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_市場_時間框架 ON 信號生成記錄 (市場, 時間框架, 時間)")
            await conn.execute("INSERT INTO 信號生成記錄 VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                             (str(uuid.uuid4()), market, timeframe, len(signals), json.dumps(signal_distribution),
                              anomaly_rate, time.time() - start_time, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS 信號記錄 (
                    id TEXT PRIMARY KEY,
                    市場 TEXT,
                    時間框架 TEXT,
                    信號 REAL,
                    時間 TEXT
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_信號_市場_時間框架 ON 信號記錄 (市場, 時間框架, 時間)")
            await conn.execute("INSERT INTO 信號記錄 VALUES (?, ?, ?, ?, ?)",
                             (str(uuid.uuid4()), market, timeframe, signals[-1] if len(signals) > 0 else 0.0,
                              datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            await conn.commit()

        # 生成穩定性圖表
        if len(_signal_buffer) == _signal_buffer.maxlen:
            signals_array = np.array(list(_signal_buffer))
            std_signal = np.std(signals_array)
            if std_signal > 0.1:
                logger.warning(f"[{market}_{timeframe}] 信號穩定性低，標準差: {std_signal:.4f}")
                await push_limiter.cache_message(
                    f"錯誤碼E502：信號穩定性低，標準差 {std_signal:.4f}", 
                    market, timeframe, "信號生成")
                await push_limiter.retry_cached_messages()

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=signals_array, mode="lines+markers", name="信號值"))
            fig.add_trace(go.Scatter(y=[np.mean(signals_array)] * len(signals_array), mode="lines", name="平均信號", line=dict(dash="dash")))
            fig.update_layout(
                title=f"{market}_{timeframe} 信號穩定性趨勢",
                xaxis_title="時間",
                yaxis_title="信號值",
                template="plotly_dark",
                height=600
            )
            plot_path = SQLite資料夾.parent / "圖片" / f"{market}_{timeframe}_Signal_Stability_Trend_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            fig.write_html(plot_path)
            await push_limiter.cache_message(f"【執行通知】生成信號穩定性圖表 {plot_path}", market, timeframe, "信號生成")
            await push_limiter.retry_cached_messages()

        # 生成報表
        df = pd.DataFrame([{
            "市場": market,
            "時間框架": timeframe,
            "信號": signals[-1] if len(signals) > 0 else 0.0,
            "HMA_16": data["HMA_16"].iloc[-1] if not data.empty else 0.0,
            "SMA50": data["SMA50"].iloc[-1] if not data.empty else 0.0,
            "ATR_14": data["ATR_14"].iloc[-1] if not data.empty else 0.0,
            "HMA閾值": params["hma_threshold"],
            "SMA閾值": params["sma_threshold"],
            "信號閾值": params["signal_threshold"],
            "異常比例": anomaly_rate,
            "信號分佈": signal_distribution
        }])
        if not df.empty:
            csv_path = SQLite資料夾.parent / "備份" / f"{market}_{timeframe}_Signal_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            await push_limiter.cache_message(f"【執行通知】生成信號報表 {csv_path}", market, timeframe, "信號生成")
            await push_limiter.retry_cached_messages()

        # 寫入異動歷程
        change_log_path = SQLite資料夾.parent / "dot" / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        change_log = {
            "UUID": str(uuid.uuid4()),
            "市場": market,
            "時間框架": timeframe,
            "異動前值": "N/A",
            "異動後值": f"信號數量={len(signals)}, 異常比例={anomaly_rate:.4f}, 分佈={json.dumps(signal_distribution)}",
            "異動原因": "信號生成",
            "時間戳": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        change_df = pd.DataFrame([change_log])
        if change_log_path.exists():
            existing_df = pd.read_excel(change_log_path)
            change_df = pd.concat([existing_df, change_df], ignore_index=True)
        change_df.to_excel(change_log_path, index=False, engine='openpyxl')

        # 推播信號結果
        message = (
            f"【信號生成完成】\n"
            f"市場: {market}_{timeframe}\n"
            f"信號數量: {len(signals)}\n"
            f"最新信號: {signals[-1] if len(signals) > 0 else 0.0}\n"
            f"分佈: {signal_distribution}\n"
            f"異常比例: {anomaly_rate:.4f}\n"
            f"耗時: {time.time() - start_time:.2f}秒"
        )
        await push_limiter.cache_message(message, market, timeframe, "信號生成")
        await push_limiter.retry_cached_messages()

        # 效率報告
        elapsed_time = time.time() - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】信號生成耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, market, timeframe, "信號生成")
        await push_limiter.retry_cached_messages()
        return signals
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[{market}_{timeframe}] 信號生成失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E501：信號生成失敗 重試{retry_count + 1}/5: {e}", "信號生成錯誤", market, timeframe, "信號生成")
            await asyncio.sleep(5)
            return await generate_single_market_signal(data, market, timeframe, params, model, retry_count + 1)
        logger.error(f"[{market}_{timeframe}] 信號生成失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E501：信號生成失敗 {e}", market, timeframe, "信號生成")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E501：信號生成失敗 {e}", "信號生成錯誤", market, timeframe, "信號生成")
        return None

async def generate_all_markets_signals(params_list, data_list, model=None, retry_count=0):
    """為所有市場與時間框架生成信號，尋找泛用性最佳參數"""
    try:
        import time
        start_time = time.time()
        results = []
        best_params = None
        best_reward = -float('inf')
        weighted_reward = 0.0
        valid_markets = 0

        for (market, timeframe) in 市場清單:
            weight = 0.5 if market == "BTCUSDT" and timeframe == "15m" else 0.5 / (len(市場清單) - 1)
            data = data_list.get((market, timeframe), pd.DataFrame())
            params = params_list.get((market, timeframe), params_list.get(("default", "default"), {}))
            signals = await generate_single_market_signal(data, market, timeframe, params, model)
            if signals is None:
                continue

            market_signal_mapping = {(market, timeframe): {"信號": signals, "價格": data["close"].values}}
            result = await single_market_trading_env(
                market_signal_mapping=market_signal_mapping,
                資產類型="虛擬貨幣" if market in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "BNBUSDT"] else "CFD",
                市場=market,
                時間框架=timeframe,
                params=params
            )
            if result is None or result.get((market, timeframe)) is None:
                continue

            reward = await calculate_single_market_reward(
                result[(market, timeframe)],
                market,
                timeframe,
                {"HMA_16": params.get("hma_period", 16), "SMA50": params.get("sma_period", 50), "ATR_14": params.get("atr_period", 14)}
            )
            weighted_reward += reward * weight
            valid_markets += 1
            results.append({
                "市場": market,
                "時間框架": timeframe,
                "信號": signals[-1] if signals else 0.0,
                "獎勵": reward,
                "最終資金": result[(market, timeframe)]["最終資金"],
                "最大回撤": result[(market, timeframe)]["最大回撤"],
                "f1分數": result[(market, timeframe)].get("f1分數", 0.0),
                "穩定性": result[(market, timeframe)].get("穩定性", 0.0)
            })
            if reward > best_reward:
                best_reward = reward
                best_params = params.copy()

        # 快取泛用性參數
        if best_params:
            cache_path = SQLite資料夾.parent / "cache" / f"signal_cache_multi_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
            np.savez_compressed(cache_path, params=best_params, reward=best_reward)
            await push_limiter.cache_message(f"【執行通知】泛用信號參數已快取至 {cache_path}", "多市場", "多框架", "信號生成")
            await push_limiter.retry_cached_messages()

        # 儲存泛用性參數
        async with aiosqlite.connect(SQLite資料夾 / "泛用信號參數.db") as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS 泛用信號參數 (
                    id TEXT PRIMARY KEY,
                    參數 TEXT,
                    獎勵 REAL,
                    時間 TEXT
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_時間 ON 泛用信號參數 (時間)")
            await conn.execute("INSERT INTO 泛用信號參數 VALUES (?, ?, ?, ?)",
                             (str(uuid.uuid4()), json.dumps(best_params, ensure_ascii=False),
                              best_reward, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            await conn.commit()

        # 生成報表
        df = pd.DataFrame(results)
        if not df.empty:
            csv_path = SQLite資料夾.parent / "備份" / f"All_Markets_Signal_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            await push_limiter.cache_message(f"【執行通知】生成全市場信號報表 {csv_path}", "多市場", "多框架", "信號生成")
            await push_limiter.retry_cached_messages()

        # 寫入異動歷程
        change_log_path = SQLite資料夾.parent / "dot" / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        change_log = {
            "UUID": str(uuid.uuid4()),
            "市場": "多市場",
            "時間框架": "多框架",
            "異動前值": "N/A",
            "異動後值": f"優化記錄數={len(results)}, 加權獎勵={weighted_reward / valid_markets if valid_markets > 0 else 0.0:.4f}",
            "異動原因": "全市場信號生成",
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
            f"【效率報告】全市場信號生成耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, "多市場", "多框架", "信號生成")
        await push_limiter.retry_cached_messages()

        return best_params, weighted_reward / valid_markets if valid_markets > 0 else 0.0
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[多市場_多框架] 全市場信號生成失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E501：全市場信號生成失敗 重試{retry_count + 1}/5: {e}", "全市場信號錯誤", "多市場", "多框架", "信號生成")
            await asyncio.sleep(5)
            return await generate_all_markets_signals(params_list, data_list, model, retry_count + 1)
        logger.error(f"[多市場_多框架] 全市場信號生成失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E501：全市場信號生成失敗 {e}", "多市場", "多框架", "信號生成")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E501：全市場信號生成失敗 {e}", "全市場信號錯誤", "多市場", "多框架", "信號生成")
        return None, 0.0

if __name__ == "__main__":
    asyncio.run(generate_all_markets_signals({}, {}))