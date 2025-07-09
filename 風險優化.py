import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from collections import deque
import datetime
import aiosqlite
import uuid
import asyncio
import plotly.graph_objects as go
from cryptography.fernet import Fernet
from 設定檔 import 訓練設備, SQLite資料夾, 市場清單, 訓練參數
from 工具模組 import 錯誤記錄與自動修復, 監控硬體狀態並降級, validate_utility_input
from 推播通知模組 import 發送錯誤訊息, 發送通知
from 模擬交易模組 import 模擬交易
from 獎勵計算模組 import calculate_multi_market_reward
import scipy.stats as stats

# 配置日誌
log_dir = SQLite資料夾.parent / "日誌"
log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("風險優化模組")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
file_handler = TimedRotatingFileHandler(
    filename=log_dir / "risk_optimization_logs",
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
    "E901": "風險優化失敗",
    "E902": "風險參數驗證失敗",
    "E903": "硬體資源超限",
    "E904": "數據加密失敗"
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

async def encrypt_risk_params(params):
    """加密風險參數"""
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
        
        logger.info("[資安] 風險參數加密完成")
        await push_limiter.cache_message("[執行通知] 風險參數加密完成", "多市場", "多框架", "風險優化")
        await push_limiter.retry_cached_messages()
        
        with open(log_dir / f"資安事件_{datetime.datetime.now().strftime('%Y%m%d')}.log", "a", encoding='utf-8') as f:
            f.write(f"[資安事件] 風險參數加密\n時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        return encrypted_params
    except Exception as e:
        logger.error(f"[資安] 風險參數加密失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E904：風險參數加密失敗 {e}", "多市場", "多框架", "風險優化")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E904：風險參數加密失敗 {e}", "數據加密錯誤", "多市場", "多框架", "風險優化")
        return None

async def validate_risk_input(trading_result, market, timeframe, params):
    """驗證風險優化輸入"""
    try:
        if not await validate_utility_input(market, timeframe, mode="風險優化"):
            return False
        required_keys = ["最終資金", "最大回撤", "交易記錄", "維持率"]
        if not all(key in trading_result for key in required_keys):
            logger.error(f"[{market}_{timeframe}] 缺少必要交易結果: {required_keys}")
            await push_limiter.cache_message(f"錯誤碼E901：缺少必要交易結果 {required_keys}", market, timeframe, "風險優化")
            await push_limiter.retry_cached_messages()
            return False
        if not isinstance(trading_result["最終資金"], (int, float)) or trading_result["最終資金"] < 0:
            logger.error(f"[{market}_{timeframe}] 無效最終資金: {trading_result['最終資金']}")
            await push_limiter.cache_message(f"錯誤碼E901：無效最終資金 {trading_result['最終資金']}", market, timeframe, "風險優化")
            await push_limiter.retry_cached_messages()
            return False
        if not isinstance(params, dict):
            logger.error(f"[{market}_{timeframe}] 無效參數格式: {params}")
            await push_limiter.cache_message(f"錯誤碼E901：無效參數格式 {params}", market, timeframe, "風險優化")
            await push_limiter.retry_cached_messages()
            return False
        required_params = ["stop_loss", "take_profit", "single_loss_limit", "trailing_stop", "trailing_take_profit", "breakeven_trigger", "signal_threshold"]
        if not all(key in params for key in required_params):
            logger.error(f"[{market}_{timeframe}] 缺少必要風險參數: {required_params}")
            await push_limiter.cache_message(f"錯誤碼E902：缺少必要風險參數 {required_params}", market, timeframe, "風險優化")
            await push_limiter.retry_cached_messages()
            return False
        ranges = {
            "stop_loss": (0.01, 0.05),
            "take_profit": (0.02, 0.1),
            "single_loss_limit": (0.01, 0.03),
            "trailing_stop": (0.01, 0.05),
            "trailing_take_profit": (0.02, 0.1),
            "breakeven_trigger": (0.01, 0.05),
            "signal_threshold": (0.5, 0.95)
        }
        for key, value in params.items():
            if key in ranges and (not ranges[key][0] <= value <= ranges[key][1]):
                logger.error(f"[{market}_{timeframe}] 無效風險參數 {key}: {value}")
                await push_limiter.cache_message(f"錯誤碼E902：無效風險參數 {key}: {value}", market, timeframe, "風險優化")
                await push_limiter.retry_cached_messages()
                return False
        return True
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 風險輸入驗證失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E901：風險輸入驗證失敗 {e}", market, timeframe, "風險優化")
        await push_limiter.retry_cached_messages()
        return False

async def check_reverse_trade_violation(trading_result, market, timeframe):
    """檢查同K棒反手違規"""
    try:
        trades = trading_result["交易記錄"]
        k_bar_timestamps = {}
        for record in trades:
            timestamp = record.get("時間戳", "")
            signal_strength = record.get("信號強度", 0.0)
            if timestamp:
                if timestamp in k_bar_timestamps:
                    k_bar_timestamps[timestamp] += 1
                    if k_bar_timestamps[timestamp] > 1 and signal_strength <= 0.9:
                        logger.warning(f"[{market}_{timeframe}] 同K棒多次反手違規: {timestamp}")
                        await push_limiter.cache_message(
                            f"錯誤碼E902：同K棒多次反手違規 時間={timestamp}, 信號強度={signal_strength:.2f}",
                            market, timeframe, "風險優化")
                        await push_limiter.retry_cached_messages()
                        async with aiosqlite.connect(SQLite資料夾 / "risk_violation_log.db") as conn:
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
        await push_limiter.cache_message(f"錯誤碼E902：同K棒反手檢查失敗 {e}", market, timeframe, "風險優化")
        await push_limiter.retry_cached_messages()
        return False

async def optimize_risk_parameters(trading_result, market, timeframe, params, retry_count=0):
    """優化風險參數"""
    try:
        import time
        start_time = time.time()
        if not await validate_risk_input(trading_result, market, timeframe, params):
            return None

        # 硬體監控
        batch_size, _ = await 監控硬體狀態並降級(params.get("batch_size", 32), 2)
        if batch_size < 8:
            logger.error(f"[{market}_{timeframe}] 硬體資源超限，批次大小 {batch_size}")
            await push_limiter.cache_message(f"錯誤碼E903：硬體資源超限，批次大小 {batch_size}", market, timeframe, "風險優化")
            await push_limiter.retry_cached_messages()
            return None
        torch.cuda.empty_cache()
        gpu_switch_count = 0

        # 檢查同K棒反手違規
        if not await check_reverse_trade_violation(trading_result, market, timeframe):
            return None

        # 提取交易數據
        initial_funds = 1000.0
        final_funds = trading_result["最終資金"]
        if final_funds <= 0:
            logger.error(f"[{market}_{timeframe}] 爆倉: 最終資金={final_funds}")
            await push_limiter.cache_message(
                f"【重大異常：爆倉】市場={market}_{timeframe}, 資金餘額={final_funds}, 動作=停止流程",
                market, timeframe, "風險優化")
            await push_limiter.retry_cached_messages()
            async with aiosqlite.connect(SQLite資料夾 / "risk_violation_log.db") as conn:
                await conn.execute("INSERT INTO 風險違規記錄 VALUES (?, ?, ?, ?, ?)",
                                 (str(uuid.uuid4()), market, timeframe, "爆倉", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                await conn.commit()
            return None
        returns = [record["損益"] / initial_funds for record in trading_result["交易記錄"] if "損益" in record]
        max_drawdown = trading_result["最大回撤"]
        volatility = np.std(returns) if returns else 0.0

        # DQN 優化
        input_dim = 6  # 最終資金, 最大回撤, 報酬率標準差, 信號強度, 交易次數, 市場波動
        output_dim = 7  # stop_loss, take_profit, single_loss_limit, trailing_stop, trailing_take_profit, breakeven_trigger, signal_threshold
        dqn = DQN(input_dim, output_dim).to(訓練設備)
        optimizer = torch.optim.Adam(dqn.parameters(), lr=1e-3)
        gamma = 0.99
        epsilon = 0.1
        episodes = 50

        risk_params_ranges = {
            "stop_loss": np.linspace(0.01, 0.05, 5),
            "take_profit": np.linspace(0.02, 0.1, 5),
            "single_loss_limit": np.linspace(0.01, 0.03, 5),
            "trailing_stop": np.linspace(0.01, 0.05, 5),
            "trailing_take_profit": np.linspace(0.02, 0.1, 5),
            "breakeven_trigger": np.linspace(0.01, 0.05, 5),
            "signal_threshold": np.linspace(0.5, 0.95, 5)
        }

        best_risk_params = params.copy()
        best_reward = -float('inf')
        optimization_data = []
        checkpoint = 0

        for episode in range(episodes):
            state = torch.tensor([
                final_funds / initial_funds,
                max_drawdown,
                np.std(returns) if returns else 0.0,
                np.mean([r.get("信號強度", 0.0) for r in trading_result["交易記錄"]]) if trading_result["交易記錄"] else 0.0,
                len(trading_result["交易記錄"]),
                volatility
            ], dtype=torch.float32, device=訓練設備)
            episode_reward = 0.0

            for step in range(10):
                if np.random.random() < epsilon:
                    action = np.random.randint(0, output_dim)
                else:
                    with torch.no_grad():
                        q_values = dqn(state)
                        action = q_values.argmax().item()

                temp_params = best_risk_params.copy()
                param_keys = list(risk_params_ranges.keys())
                if action < len(param_keys):
                    param = param_keys[action]
                    temp_params[param] = np.random.choice(risk_params_ranges[param])
                    # 根據歷史波動率動態調整停損
                    if param == "stop_loss" and volatility > 0.1:
                        temp_params[param] = min(temp_params[param] * (1 + volatility), 0.05)

                sim_result = await 模擬交易(
                    信號=[r.get("信號強度", 1.0) if r.get("信號強度", 0.0) > temp_params["signal_threshold"] else -1.0 for r in trading_result["交易記錄"]],
                    價格=[record["價格"] for record in trading_result["交易記錄"] if "價格" in record],
                    資產類型="虛擬貨幣",
                    市場=market,
                    時間框架=timeframe,
                    batch_size=batch_size,
                    停損=temp_params["stop_loss"],
                    停利=temp_params["take_profit"],
                    平本觸發=temp_params["breakeven_trigger"],
                    移動停損=temp_params["trailing_stop"],
                    移動停利=temp_params["trailing_take_profit"]
                )
                if sim_result and sim_result.get(market):
                    sim_reward, _ = await calculate_multi_market_reward({(market, timeframe): sim_result[market]}, {(market, timeframe): temp_params})
                    episode_reward += sim_reward
                    if sim_reward > best_reward:
                        best_reward = sim_reward
                        best_risk_params = temp_params.copy()
                else:
                    episode_reward += -1.0  # 懲罰無效模擬

                next_state = torch.tensor([
                    sim_result[market]["最終資金"] / initial_funds if sim_result and sim_result.get(market) else state[0].item(),
                    sim_result[market]["最大回撤"] if sim_result and sim_result.get(market) else state[1].item(),
                    np.std(returns) if returns else 0.0,
                    state[3].item(),
                    state[4].item(),
                    volatility
                ], dtype=torch.float32, device=訓練設備)

                q_value = dqn(state)[action]
                target = episode_reward + gamma * dqn(next_state).max()
                loss = nn.MSELoss()(q_value, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                state = next_state
                checkpoint = step
                if not await manage_resources():
                    logger.warning(f"[{market}_{timeframe}] 資源超限，中斷恢復至檢查點 {checkpoint}")
                    await push_limiter.cache_message(f"【通知】資源超限，中斷恢復至檢查點 {checkpoint}", market, timeframe, "風險優化")
                    await push_limiter.retry_cached_messages()
                    break

            optimization_data.append({
                "市場": market,
                "時間框架": timeframe,
                "stop_loss": best_risk_params["stop_loss"],
                "take_profit": best_risk_params["take_profit"],
                "single_loss_limit": best_risk_params["single_loss_limit"],
                "trailing_stop": best_risk_params["trailing_stop"],
                "trailing_take_profit": best_risk_params["trailing_take_profit"],
                "breakeven_trigger": best_risk_params["breakeven_trigger"],
                "signal_threshold": best_risk_params["signal_threshold"],
                "獎勵": episode_reward
            })

        # 加密並儲存優化記錄
        encrypted_params = await encrypt_risk_params(best_risk_params)
        if not encrypted_params:
            return None
        async with aiosqlite.connect(SQLite資料夾 / "risk_optimization_log.db") as conn:
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
                             (str(uuid.uuid4()), market, timeframe, encrypted_params, best_reward,
                              datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            await conn.commit()

        # 快取優化結果
        cache_path = 快取資料夾 / f"risk_cache_{market}_{timeframe}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez_compressed(cache_path, data=optimization_data)
        await push_limiter.cache_message(f"【執行通知】風險優化數據已快取至 {cache_path}", market, timeframe, "風險優化")
        await push_limiter.retry_cached_messages()

        # 生成風險優化圖表
        df = pd.DataFrame(optimization_data)
        if not df.empty:
            fig = go.Figure()
            for param in ["stop_loss", "take_profit", "trailing_stop", "trailing_take_profit", "signal_threshold"]:
                fig.add_trace(go.Scatter3d(
                    x=df[param], y=df["獎勵"], z=df["穩定性"] if "穩定性" in df else [0] * len(df),
                    mode="markers", name=param,
                    marker=dict(size=5)
                ))
            fig.update_layout(
                title=f"{market}_{timeframe} 風險參數優化",
                scene=dict(xaxis_title="參數值", yaxis_title="獎勵", zaxis_title="穩定性"),
                template="plotly_dark",
                height=600,
                showlegend=True
            )
            plot_path = SQLite資料夾.parent / "圖片" / f"{market}_{timeframe}_Risk_Optimization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            fig.write_html(plot_path)
            await push_limiter.cache_message(f"【執行通知】生成風險優化圖表 {plot_path}", market, timeframe, "風險優化")
            await push_limiter.retry_cached_messages()

        # 違規檢查
        violations = []
        if max_drawdown > 0.25:
            violations.append(f"最大回撤超限: {max_drawdown:.2%}")
        if any(abs(r) > best_risk_params["single_loss_limit"] for r in returns):
            violations.append(f"單筆損失超限: {min(returns):.4f}")
        if violations:
            logger.warning(f"[{market}_{timeframe}] 風險違規: {violations}")
            await push_limiter.cache_message(f"錯誤碼E902：風險違規 {violations}", market, timeframe, "風險優化")
            await push_limiter.retry_cached_messages()
            async with aiosqlite.connect(SQLite資料夾 / "risk_violation_log.db") as conn:
                await conn.execute("INSERT INTO 風險違規記錄 VALUES (?, ?, ?, ?, ?)",
                                 (str(uuid.uuid4()), market, timeframe, json.dumps(violations, ensure_ascii=False),
                                  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                await conn.commit()

        # 壓力測試
        stress_scenarios = [
            {"volatility_multiplier": 2.0, "market_drop": 0.3},  # 市場崩盤場景
            {"volatility_multiplier": 1.5, "market_drop": 0.1}   # 高波動場景
        ]
        for scenario in stress_scenarios:
            stress_result = await 模擬交易(
                信號=[r.get("信號強度", 1.0) if r.get("信號強度", 0.0) > best_risk_params["signal_threshold"] else -1.0 for r in trading_result["交易記錄"]],
                價格=[record["價格"] * (1 - scenario["market_drop"]) for record in trading_result["交易記錄"] if "價格" in record],
                資產類型="虛擬貨幣",
                市場=market,
                時間框架=timeframe,
                batch_size=batch_size,
                停損=best_risk_params["stop_loss"],
                停利=best_risk_params["take_profit"],
                平本觸發=best_risk_params["breakeven_trigger"],
                移動停損=best_risk_params["trailing_stop"],
                移動停利=best_risk_params["trailing_take_profit"]
            )
            if stress_result and stress_result.get(market) and stress_result[market]["最終資金"] <= 0:
                logger.warning(f"[{market}_{timeframe}] 壓力測試場景 {scenario} 爆倉")
                await push_limiter.cache_message(f"錯誤碼E902：壓力測試場景 {scenario} 爆倉", market, timeframe, "風險優化")
                await push_limiter.retry_cached_messages()

        # 寫入異動歷程
        change_log_path = SQLite資料夾.parent / "dot" / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        change_log = {
            "UUID": str(uuid.uuid4()),
            "市場": market,
            "時間框架": timeframe,
            "異動前值": f"原始參數={json.dumps(params, ensure_ascii=False)}",
            "異動後值": f"優化參數={json.dumps(best_risk_params, ensure_ascii=False)}, 獎勵={best_reward:.4f}",
            "異動原因": "風險參數優化",
            "時間戳": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        change_df = pd.DataFrame([change_log])
        if change_log_path.exists():
            existing_df = pd.read_excel(change_log_path)
            change_df = pd.concat([existing_df, change_df], ignore_index=True)
        change_df.to_excel(change_log_path, index=False, engine='openpyxl')

        # 推播優化結果
        message = (
            f"【風險優化完成】\n"
            f"市場: {market}_{timeframe}\n"
            f"獎勵: {best_reward:.4f}\n"
            f"風險參數: stop_loss={best_risk_params['stop_loss']:.4f}, "
            f"take_profit={best_risk_params['take_profit']:.4f}, "
            f"single_loss_limit={best_risk_params['single_loss_limit']:.4f}, "
            f"trailing_stop={best_risk_params['trailing_stop']:.4f}, "
            f"trailing_take_profit={best_risk_params['trailing_take_profit']:.4f}, "
            f"breakeven_trigger={best_risk_params['breakeven_trigger']:.4f}, "
            f"signal_threshold={best_risk_params['signal_threshold']:.4f}"
        )
        await push_limiter.cache_message(message, market, timeframe, "風險優化")
        await push_limiter.retry_cached_messages()

        # 效率報告
        elapsed_time = time.time() - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】風險優化耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, market, timeframe, "風險優化")
        await push_limiter.retry_cached_messages()

        return best_risk_params
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[{market}_{timeframe}] 風險優化失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E901：風險優化失敗 重試{retry_count + 1}/5: {e}", "風險優化錯誤", market, timeframe, "風險優化")
            await asyncio.sleep(5)
            return await optimize_risk_parameters(trading_result, market, timeframe, params, retry_count + 1)
        logger.error(f"[{market}_{timeframe}] 風險優化失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E901：風險優化失敗 {e}", market, timeframe, "風險優化")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E901：風險優化失敗 {e}", "風險優化錯誤", market, timeframe, "風險優化")
        return None

async def multi_market_risk_optimization(multi_trading_results, params_list, retry_count=0):
    """多市場風險優化"""
    try:
        import time
        start_time = time.time()
        optimized_params = {}
        tasks = []
        for (market, timeframe), result in multi_trading_results.items():
            if result is None:
                continue
            params = params_list.get((market, timeframe), params_list.get(("default", "default"), {}))
            tasks.append(optimize_risk_parameters(result, market, timeframe, params))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (market, timeframe), result in zip(multi_trading_results.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"[{market}_{timeframe}] 風險優化失敗: {result}")
                await push_limiter.cache_message(f"錯誤碼E901：風險優化失敗 {result}", market, timeframe, "風險優化")
                await push_limiter.retry_cached_messages()
                optimized_params[(market, timeframe)] = None
            else:
                optimized_params[(market, timeframe)] = result

        # 快取多市場優化結果
        cache_path = 快取資料夾 / f"risk_cache_multi_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez_compressed(cache_path, data=optimized_params)
        await push_limiter.cache_message(f"【執行通知】多市場風險優化數據已快取至 {cache_path}", "多市場", "多框架", "風險優化")
        await push_limiter.retry_cached_messages()

        # 生成多市場報表
        report_data = []
        for (market, timeframe), opt_params in optimized_params.items():
            if opt_params:
                report_data.append({
                    "市場": market,
                    "時間框架": timeframe,
                    "stop_loss": opt_params.get("stop_loss", 0.0),
                    "take_profit": opt_params.get("take_profit", 0.0),
                    "single_loss_limit": opt_params.get("single_loss_limit", 0.0),
                    "trailing_stop": opt_params.get("trailing_stop", 0.0),
                    "trailing_take_profit": opt_params.get("trailing_take_profit", 0.0),
                    "breakeven_trigger": opt_params.get("breakeven_trigger", 0.0),
                    "signal_threshold": opt_params.get("signal_threshold", 0.0)
                })
        df = pd.DataFrame(report_data)
        if not df.empty:
            csv_path = SQLite資料夾.parent / "備份" / f"Multi_Market_Risk_Optimization_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            await push_limiter.cache_message(f"【執行通知】生成多市場風險優化報表 {csv_path}", "多市場", "多框架", "風險優化")
            await push_limiter.retry_cached_messages()

        # 寫入異動歷程
        change_log_path = SQLite資料夾.parent / "dot" / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        change_log = {
            "UUID": str(uuid.uuid4()),
            "市場": "多市場",
            "時間框架": "多框架",
            "異動前值": "N/A",
            "異動後值": f"優化記錄數={len(report_data)}",
            "異動原因": "多市場風險優化",
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
            f"【效率報告】多市場風險優化耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, "多市場", "多框架", "風險優化")
        await push_limiter.retry_cached_messages()

        return optimized_params
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[多市場_多框架] 多市場風險優化失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E901：多市場風險優化失敗 重試{retry_count + 1}/5: {e}", "多市場風險優化錯誤", "多市場", "多框架", "風險優化")
            await asyncio.sleep(5)
            return await multi_market_risk_optimization(multi_trading_results, params_list, retry_count + 1)
        logger.error(f"[多市場_多框架] 多市場風險優化失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E901：多市場風險優化失敗 {e}", "多市場", "多框架", "風險優化")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E901：多市場風險優化失敗 {e}", "多市場風險優化錯誤", "多市場", "多框架", "風險優化")
        return None