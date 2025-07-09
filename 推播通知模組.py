import asyncio
import logging
import datetime
import sqlite3
import uuid
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go
import aiohttp
import psutil
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from collections import deque
from 設定檔 import SQLite資料夾, 市場清單, 訓練設備, 資源閾值, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_BACKUP_CHAT_ID
from 工具模組 import 錯誤記錄與自動修復, 監控硬體狀態並降級

# 配置日誌
log_dir = SQLite資料夾.parent / "日誌"
log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("推播通知模組")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
file_handler = TimedRotatingFileHandler(
    filename=log_dir / "notification_logs",
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
    "E801": "推播通知發送失敗",
    "E802": "推播穩定性驗證失敗",
    "E803": "硬體或網絡資源超限",
    "E804": "DQN 推播頻率優化失敗"
}

# 推播緩衝區
_notification_buffer = deque(maxlen=100)

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

async def check_network_connectivity():
    """檢查Telegram API網絡連線"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe", timeout=5) as response:
                if response.status != 200:
                    logger.warning(f"Telegram API連線失敗，狀態碼: {response.status}")
                    return False
                return True
    except Exception as e:
        logger.error(f"網絡連線檢查失敗: {e}")
        await 錯誤記錄與自動修復(f"錯誤碼E803：網絡連線檢查失敗 {e}", "網絡連線錯誤", "多市場", "多框架", "推播通知")
        return False

async def manage_notification_resources():
    """資源管理"""
    try:
        ram_ok = psutil.virtual_memory().percent < 資源閾值["RAM使用率"] * 100
        cpu_ok = psutil.cpu_percent() < 資源閾值["CPU使用率"] * 100
        network_ok = await check_network_connectivity()
        if torch.cuda.is_available():
            gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100
            gpu_ok = gpu_util < 資源閾值["GPU使用率"] * 100
            if not gpu_ok:
                torch.cuda.empty_cache()
                logger.warning(f"GPU使用率超限: {gpu_util:.2f}%")
        else:
            gpu_ok = True
        if not (ram_ok and cpu_ok and network_ok and gpu_ok):
            logger.warning(f"資源超限: RAM={psutil.virtual_memory().percent}%, CPU={psutil.cpu_percent()}%, 網絡={'正常' if network_ok else '異常'}, GPU={'正常' if gpu_ok else f'{gpu_util:.2f}%'}")
            await 發送錯誤訊息(f"錯誤碼E803：資源超限 RAM={psutil.virtual_memory().percent}%, CPU={psutil.cpu_percent()}%, 網絡={'正常' if network_ok else '異常'}, GPU={'正常' if gpu_ok else f'{gpu_util:.2f}%'}", "多市場", "多框架", "推播通知")
            return False
        return True
    except Exception as e:
        logger.error(f"資源管理失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E803：資源管理失敗 {e}", "多市場", "多框架", "推播通知")
        await 錯誤記錄與自動修復(f"錯誤碼E803：資源管理失敗 {e}", "資源管理錯誤", "多市場", "多框架", "推播通知")
        return False

async def validate_notification_input(message, market, timeframe, mode):
    """驗證推播輸入數據"""
    try:
        if not isinstance(message, str) or not message:
            logger.error(f"[{market}_{timeframe}] 無效推播訊息")
            return False
        if market and market not in [mkt for mkt, _ in 市場清單]:
            logger.error(f"[{market}_{timeframe}] 無效市場: {market}")
            return False
        if timeframe and timeframe not in [tf for _, tf in 市場清單]:
            logger.error(f"[{market}_{timeframe}] 無效時間框架: {timeframe}")
            return False
        if mode not in ["模擬", "交易環境", "測試網", "超參數搜尋", "檢查點", "信號生成", "獎勵計算", "GUI"]:
            logger.error(f"[{market}_{timeframe}] 無效模式: {mode}")
            return False
        return True
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 推播輸入驗證失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E801：推播輸入驗證失敗 {e}", market, timeframe, "推播通知")
        return False

async def send_telegram_message(chat_id, message, retry_count=0):
    """發送Telegram訊息"""
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": message[:4096],  # Telegram訊息長度限制
                "parse_mode": "Markdown"
            }
            async with session.post(url, json=payload, timeout=10) as response:
                if response.status == 429:  # 速率限制
                    logger.warning(f"Telegram API速率限制，狀態碼: {response.status}")
                    if retry_count < 5:
                        await asyncio.sleep(5)
                        return await send_telegram_message(chat_id, message, retry_count + 1)
                    return await send_telegram_message(TELEGRAM_BACKUP_CHAT_ID, message, 0)  # 切換備用頻道
                if response.status != 200:
                    logger.error(f"Telegram訊息發送失敗，狀態碼: {response.status}")
                    if retry_count < 5:
                        await asyncio.sleep(5)
                        return await send_telegram_message(chat_id, message, retry_count + 1)
                    return await send_telegram_message(TELEGRAM_BACKUP_CHAT_ID, message, 0)
                return True
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"Telegram訊息發送失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E801：Telegram訊息發送失敗 重試{retry_count + 1}/5: {e}", "推播通知錯誤", "多市場", "多框架", "推播通知")
            await asyncio.sleep(5)
            return await send_telegram_message(chat_id, message, retry_count + 1)
        logger.error(f"Telegram訊息發送失敗，重試 5 次無效: {e}")
        await 錯誤記錄與自動修復(f"錯誤碼E801：Telegram訊息發送失敗 {e}", "推播通知錯誤", "多市場", "多框架", "推播通知")
        return await send_telegram_message(TELEGRAM_BACKUP_CHAT_ID, message, 0)

async def optimize_notification_frequency(market, timeframe, system_state, retry_count=0):
    """使用DQN優化推播頻率"""
    try:
        input_dim = 5  # CPU使用率, RAM使用率, GPU使用率, 錯誤頻率, 市場波動
        output_dim = 4  # 調整錯誤通知間隔, 交易通知間隔
        dqn = DQN(input_dim, output_dim).to(訓練設備)
        optimizer = torch.optim.Adam(dqn.parameters(), lr=1e-3)
        gamma = 0.99
        epsilon = 0.1
        episodes = 50

        base_params = {
            "error_notification_interval": 60.0,  # 初始錯誤通知間隔（秒）
            "trade_notification_interval": 30.0   # 初始交易通知間隔（秒）
        }
        best_params = base_params.copy()
        best_reward = -float('inf')

        for episode in range(episodes):
            state = torch.tensor([
                system_state.get("cpu_percent", psutil.cpu_percent()),
                system_state.get("ram_percent", psutil.virtual_memory().percent),
                system_state.get("gpu_util", torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0),
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
                    temp_params["error_notification_interval"] = min(temp_params["error_notification_interval"] + 10.0, 60.0)
                elif action == 1:
                    temp_params["error_notification_interval"] = max(temp_params["error_notification_interval"] - 10.0, 1.0)
                elif action == 2:
                    temp_params["trade_notification_interval"] = min(temp_params["trade_notification_interval"] + 5.0, 60.0)
                elif action == 3:
                    temp_params["trade_notification_interval"] = max(temp_params["trade_notification_interval"] - 5.0, 1.0)

                simulated_load = (
                    state[0].item() * 0.4 +
                    state[1].item() * 0.3 +
                    state[2].item() * 0.2 +
                    state[3].item() * 0.1
                )
                reward = -simulated_load / 100.0
                if temp_params["error_notification_interval"] < 1.0 or temp_params["trade_notification_interval"] < 1.0:
                    reward -= 0.5

                episode_reward += reward
                next_state = torch.tensor([
                    state[0].item(),
                    state[1].item(),
                    state[2].item(),
                    state[3].item() * 0.9,
                    state[4].item()
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

            conn = sqlite3.connect(SQLite資料夾 / "推播優化記錄.db")
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS 推播優化記錄 (
                    id TEXT PRIMARY KEY,
                    市場 TEXT,
                    時間框架 TEXT,
                    錯誤通知間隔 REAL,
                    交易通知間隔 REAL,
                    獎勵 REAL,
                    時間 TEXT
                )
            """)
            c.execute("INSERT INTO 推播優化記錄 VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (str(uuid.uuid4()), market, timeframe, best_params["error_notification_interval"],
                       best_params["trade_notification_interval"], episode_reward,
                       datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
            conn.close()

        conn = sqlite3.connect(SQLite資料夾 / "推播優化記錄.db")
        df = pd.read_sql_query("SELECT 錯誤通知間隔, 交易通知間隔 FROM 推播優化記錄 WHERE 市場 = ? AND 時間框架 = ? ORDER BY 時間 DESC LIMIT 10",
                               conn, params=(market, timeframe))
        conn.close()
        if len(df) > 1:
            std_error = df["錯誤通知間隔"].std()
            std_trade = df["交易通知間隔"].std()
            if std_error > 0.1 or std_trade > 0.1:
                logger.warning(f"[{market}_{timeframe}] DQN 推播頻率穩定性低: 錯誤={std_error:.4f}, 交易={std_trade:.4f}")
                await 發送錯誤訊息(f"錯誤碼E804：DQN 推播頻率穩定性低 錯誤={std_error:.4f}, 交易={std_trade:.4f}", market, timeframe, "推播通知")

        await 發送通知(f"【通知】DQN 推播頻率優化完成: {market}_{timeframe}\n錯誤通知間隔: {best_params['error_notification_interval']:.2f}秒\n交易通知間隔: {best_params['trade_notification_interval']:.2f}秒\n獎勵: {best_reward:.4f}", market, timeframe, "推播通知")
        return best_params
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[{market}_{timeframe}] DQN 推播頻率優化失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E804：DQN 推播頻率優化失敗 重試{retry_count + 1}/5: {e}", "DQN 推播優化錯誤", market, timeframe, "推播通知")
            await asyncio.sleep(5)
            return await optimize_notification_frequency(market, timeframe, system_state, retry_count + 1)
        logger.error(f"[{market}_{timeframe}] DQN 推播頻率優化失敗，重試 5 次無效: {e}")
        await 發送錯誤訊息(f"錯誤碼E804：DQN 推播頻率優化失敗 {e}", market, timeframe, "推播通知")
        await 錯誤記錄與自動修復(f"錯誤碼E804：DQN 推播頻率優化失敗 {e}", "DQN 推播優化錯誤", market, timeframe, "推播通知")
        return {
            "error_notification_interval": 60.0,
            "trade_notification_interval": 30.0
        }

async def render_notification_trend(market, timeframe):
    """生成3D交互式通知趨勢圖表"""
    try:
        conn = sqlite3.connect(SQLite資料夾 / "通知紀錄.db")
        df = pd.read_sql_query("SELECT 時間, 模式 FROM 通知記錄 WHERE 市場 = ? AND 時間框架 = ? ORDER BY 時間 DESC LIMIT 100",
                               conn, params=(market, timeframe))
        conn.close()
        if df.empty:
            logger.warning(f"[{market}_{timeframe}] 無通知數據")
            await 發送錯誤訊息(f"錯誤碼E802：無通知數據", market, timeframe, "推播通知")
            return None

        df['時間'] = pd.to_datetime(df['時間'])
        mode_counts = df.groupby('模式').size().to_dict()
        fig = go.Figure()
        for mode in mode_counts:
            mode_df = df[df['模式'] == mode]
            fig.add_trace(go.Scatter3d(
                x=mode_df['時間'], y=[mode_counts[mode]] * len(mode_df), z=[mode] * len(mode_df),
                mode='lines+markers', name=mode,
                line=dict(color='#00ffcc' if mode == '交易環境' else '#ff4d4d', width=4),
                marker=dict(size=5, color='#1e90ff')
            ))
        fig.update_layout(
            title=f"{market}_{timeframe} 通知趨勢",
            scene=dict(
                xaxis_title="時間",
                yaxis_title="通知數",
                zaxis_title="模式",
                xaxis=dict(backgroundcolor="#0d1117", gridcolor="#444"),
                yaxis=dict(backgroundcolor="#0d1117", gridcolor="#444"),
                zaxis=dict(backgroundcolor="#0d1117", gridcolor="#444")
            ),
            template="plotly_dark",
            height=600
        )
        plot_path = SQLite資料夾.parent / "圖片" / f"{market}_{timeframe}_Notification_Trend_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.html"
        fig.write_html(plot_path)
        await 發送通知(f"【執行通知】生成通知趨勢圖表 {plot_path}", market, timeframe, "推播通知")
        
        # 保留母檔2D圖表
        fig_2d = go.Figure()
        fig_2d.add_trace(go.Scatter(y=[mode_counts.get(mode, 0) for mode in mode_counts], mode="lines+markers", name="通知數"))
        fig_2d.update_layout(title=f"{market}_{timeframe} 推播頻率穩定性趨勢", template="plotly_dark")
        plot_path_2d = SQLite資料夾.parent / "圖片" / f"{market}_{timeframe}_Notification_Stability_Trend_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.html"
        fig_2d.write_html(plot_path_2d)
        await 發送通知(f"【執行通知】生成推播頻率穩定性圖表 {plot_path_2d}", market, timeframe, "推播通知")
        return plot_path
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 通知趨勢圖表生成失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E802：通知趨勢圖表生成失敗 {e}", market, timeframe, "推播通知")
        await 錯誤記錄與自動修復(f"錯誤碼E802：通知趨勢圖表生成失敗 {e}", "圖表生成錯誤", market, timeframe, "推播通知")
        return None

async def single_market_notification(message, market, timeframe, mode, params=None, system_state=None, retry_count=0):
    """發送單一市場推播通知"""
    try:
        start_time = time.time()
        if not await validate_notification_input(message, market, timeframe, mode):
            return False

        if not await manage_notification_resources():
            return False

        system_state = system_state or {
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": psutil.virtual_memory().percent,
            "gpu_util": torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0,
            "error_frequency": 0.0,
            "market_volatility": 0.0
        }
        optimized_params = await optimize_notification_frequency(market, timeframe, system_state)
        error_interval = optimized_params["error_notification_interval"]
        trade_interval = optimized_params["trade_notification_interval"]

        notification_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f"[ID: {notification_id}] [{market}_{timeframe}] [{mode}] {message} [{timestamp}]"

        # Telegram推播
        success = await send_telegram_message(TELEGRAM_CHAT_ID, formatted_message)
        if not success:
            logger.warning(f"[{market}_{timeframe}] 主頻道推播失敗，切換備用頻道")
            success = await send_telegram_message(TELEGRAM_BACKUP_CHAT_ID, formatted_message)
            if not success:
                logger.error(f"[{market}_{timeframe}] 備用頻道推播失敗")
                await 錯誤記錄與自動修復(f"錯誤碼E801：備用頻道推播失敗", "推播通知錯誤", market, timeframe, "推播通知")
                # 模擬推播（保留母檔邏輯）
                logger.info(formatted_message)

        # 儲存通知記錄
        conn = sqlite3.connect(SQLite資料夾 / "通知紀錄.db")
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS 通知記錄 (
                id TEXT PRIMARY KEY,
                市場 TEXT,
                時間框架 TEXT,
                模式 TEXT,
                訊息 TEXT,
                時間 TEXT
            )
        """)
        c.execute("INSERT INTO 通知記錄 VALUES (?, ?, ?, ?, ?, ?)",
                  (notification_id, market, timeframe, mode, message, timestamp))
        conn.commit()
        conn.close()

        # 儲存推播頻率記錄
        _notification_buffer.append({
            "error_interval": error_interval,
            "trade_interval": trade_interval
        })
        if len(_notification_buffer) == _notification_buffer.maxlen:
            df_buffer = pd.DataFrame(list(_notification_buffer))
            std_error = df_buffer["error_interval"].std()
            std_trade = df_buffer["trade_interval"].std()
            if std_error > 0.1 or std_trade > 0.1:
                logger.warning(f"[{market}_{timeframe}] 推播頻率穩定性低: 錯誤={std_error:.4f}, 交易={std_trade:.4f}")
                await 發送錯誤訊息(f"錯誤碼E802：推播頻率穩定性低 錯誤={std_error:.4f}, 交易={std_trade:.4f}", market, timeframe, "推播通知")
            await render_notification_trend(market, timeframe)

        # 生成單市場報表
        df = pd.DataFrame([{
            "市場": market,
            "時間框架": timeframe,
            "模式": mode,
            "訊息": message,
            "錯誤通知間隔": error_interval,
            "交易通知間隔": trade_interval,
            "時間": timestamp
        }])
        if not df.empty:
            csv_path = SQLite資料夾.parent / "備份" / f"{market}_{timeframe}_Notification_Report_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            await 發送通知(f"【執行通知】生成單市場推播報表 {csv_path}", market, timeframe, "推播通知")

        # 效率報告
        elapsed_time = time.time() - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】單市場推播耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await 發送通知(efficiency_report, market, timeframe, "推播通知")

        return success
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[{market}_{timeframe}] 推播通知發送失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E801：推播通知發送失敗 重試{retry_count + 1}/5: {e}", "推播通知錯誤", market, timeframe, "推播通知")
            await asyncio.sleep(5)
            return await single_market_notification(message, market, timeframe, mode, params, system_state, retry_count + 1)
        logger.error(f"[{market}_{timeframe}] 推播通知發送失敗，重試 5 次無效: {e}")
        await 發送錯誤訊息(f"錯誤碼E801：推播通知發送失敗 {e}", market, timeframe, "推播通知")
        await 錯誤記錄與自動修復(f"錯誤碼E801：推播通知發送失敗 {e}", "推播通知錯誤", market, timeframe, "推播通知")
        return False

async def 發送下單通知(order_type, market, timeframe, price, quantity, profit, mode, params=None, system_state=None):
    """發送下單通知"""
    message = f"下單通知：{order_type} | 價格: {price:.2f} | 數量: {quantity:.4f} | 損益: {profit:.2f}"
    return await single_market_notification(message, market, timeframe, mode, params, system_state)

async def 發送倉位通知(market, timeframe, mode, position, avg_price, unrealized_pnl, margin_ratio, params=None, system_state=None):
    """發送倉位通知"""
    message = f"倉位通知：持倉: {position:.4f} | 平均價格: {avg_price:.2f} | 未實現損益: {unrealized_pnl:.2f} | 維持率: {margin_ratio:.2f}%"
    return await single_market_notification(message, market, timeframe, mode, params, system_state)

async def 發送錯誤訊息(error_message, market, timeframe, mode, params=None, system_state=None):
    """發送錯誤通知"""
    return await single_market_notification(error_message, market, timeframe, mode, params, system_state)

async def 發送通知(message, market, timeframe, mode, params=None, system_state=None):
    """發送一般通知"""
    return await single_market_notification(message, market, timeframe, mode, params, system_state)

async def optimize_all_markets_notifications(params_list, system_states):
    """為所有市場與時間框架優化推播參數"""
    try:
        start_time = time.time()
        results = []
        best_params = None
        best_reward = -float('inf')

        for (market, timeframe) in 市場清單:
            system_state = system_states.get((market, timeframe), {
                "cpu_percent": psutil.cpu_percent(),
                "ram_percent": psutil.virtual_memory().percent,
                "gpu_util": torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0,
                "error_frequency": 0.0,
                "market_volatility": 0.0
            })
            params = params_list.get((market, timeframe), params_list.get(("default", "default"), {}))
            optimized_params = await optimize_notification_frequency(market, timeframe, system_state)
            results.append({
                "市場": market,
                "時間框架": timeframe,
                "錯誤通知間隔": optimized_params["error_notification_interval"],
                "交易通知間隔": optimized_params["trade_notification_interval"],
                "獎勵": -(
                    system_state["cpu_percent"] * 0.4 +
                    system_state["ram_percent"] * 0.3 +
                    system_state["gpu_util"] * 0.2 +
                    system_state["error_frequency"] * 0.1
                ) / 100.0
            })
            if results[-1]["獎勵"] > best_reward:
                best_reward = results[-1]["獎勵"]
                best_params = optimized_params.copy()

        conn = sqlite3.connect(SQLite資料夾 / "泛用推播參數.db")
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS 泛用推播參數 (
                id TEXT PRIMARY KEY,
                參數 TEXT,
                獎勵 REAL,
                時間 TEXT
            )
        """)
        c.execute("INSERT INTO 泛用推播參數 VALUES (?, ?, ?, ?)",
                  (str(uuid.uuid4()), json.dumps(best_params, ensure_ascii=False),
                   best_reward, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()

        df = pd.DataFrame(results)
        if not df.empty:
            csv_path = SQLite資料夾.parent / "備份" / f"All_Markets_Notification_Report_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            await 發送通知(f"【執行通知】生成全市場推播報表 {csv_path}", "多市場", "多框架", "推播通知")

        elapsed_time = time.time() - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】全市場推播優化耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await 發送通知(efficiency_report, "多市場", "多框架", "推播通知")

        return best_params, best_reward
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[多市場_多框架] 全市場推播優化失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E801：全市場推播優化失敗 重試{retry_count + 1}/5: {e}", "全市場推播錯誤", "多市場", "多框架", "推播通知")
            await asyncio.sleep(5)
            return await optimize_all_markets_notifications(params_list, system_states, retry_count + 1)
        logger.error(f"[多市場_多框架] 全市場推播優化失敗，重試 5 次無效: {e}")
        await 發送錯誤訊息(f"錯誤碼E801：全市場推播優化失敗 {e}", "多市場", "多框架", "推播通知")
        await 錯誤記錄與自動修復(f"錯誤碼E801：全市場推播優化失敗 {e}", "全市場推播錯誤", "多市場", "多框架", "推播通知")
        return None, 0.0