import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import asyncio
import sqlite3
import uuid
import datetime
import plotly.graph_objects as go
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from collections import deque
from torch.utils.data import DataLoader, TensorDataset
from 設定檔 import 訓練設備, SQLite資料夾, 訓練參數, 市場清單, 記錄參數異動
from 工具模組 import 錯誤記錄與自動修復, 監控硬體狀態並降級
from 推播通知模組 import 發送錯誤訊息, 發送通知

# 配置日誌
log_dir = SQLite資料夾.parent / "日誌"
log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("獎勵計算模組")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
file_handler = TimedRotatingFileHandler(
    filename=log_dir / "reward_calculation_logs",
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
    "E201": "獎勵計算失敗",
    "E202": "風險指標異常",
    "E203": "硬體資源超限",
    "E204": "DQN 權重優化失敗"
}

# 獎勵緩衝區
_reward_buffer = deque(maxlen=100)

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

async def validate_input(trading_result, market, timeframe, technical_indicators):
    """驗證輸入數據"""
    try:
        required_keys = ["最終資金", "最大回撤", "交易記錄", "維持率"]
        if not all(key in trading_result for key in required_keys):
            logger.error(f"[{market}_{timeframe}] 缺少必要交易結果: {required_keys}")
            await 發送錯誤訊息(f"錯誤碼E201：缺少必要交易結果 {required_keys}", market, timeframe, "獎勵計算", "high")
            return False
        if not isinstance(trading_result["最終資金"], (int, float)) or trading_result["最終資金"] < 0:
            logger.error(f"[{market}_{timeframe}] 無效最終資金: {trading_result['最終資金']}")
            await 發送錯誤訊息(f"錯誤碼E201：無效最終資金 {trading_result['最終資金']}", market, timeframe, "獎勵計算", "high")
            return False
        if not isinstance(trading_result["最大回撤"], (int, float)) or trading_result["最大回撤"] < 0:
            logger.error(f"[{market}_{timeframe}] 無效最大回撤: {trading_result['最大回撤']}")
            await 發送錯誤訊息(f"錯誤碼E201：無效最大回撤 {trading_result['最大回撤']}", market, timeframe, "獎勵計算", "high")
            return False
        if not isinstance(trading_result["交易記錄"], list):
            logger.error(f"[{market}_{timeframe}] 無效交易記錄格式")
            await 發送錯誤訊息(f"錯誤碼E201：無效交易記錄格式", market, timeframe, "獎勵計算", "high")
            return False
        if not isinstance(technical_indicators, dict) or not all(key in technical_indicators for key in ["HMA_16", "SMA50", "ATR_14"]):
            logger.error(f"[{market}_{timeframe}] 缺少必要技術指標: {technical_indicators}")
            await 發送錯誤訊息(f"錯誤碼E201：缺少必要技術指標 {technical_indicators}", market, timeframe, "獎勵計算", "high")
            return False
        return True
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 輸入驗證失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E201：輸入驗證失敗 {e}", market, timeframe, "獎勵計算", "high")
        await 錯誤記錄與自動修復(f"錯誤碼E201：輸入驗證失敗 {e}", "輸入驗證錯誤", market, timeframe, "獎勵計算")
        return False

def calculate_var(returns, confidence_level=0.95):
    """計算VaR_95"""
    try:
        if len(returns) == 0:
            return 0.0
        sorted_returns = np.sort(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        return -sorted_returns[index] if index < len(sorted_returns) else 0.0
    except Exception as e:
        logger.error(f"VaR_95計算失敗: {e}")
        asyncio.run(發送錯誤訊息(f"錯誤碼E202：VaR_95計算失敗 {e}", None, None, "獎勵計算", "high"))
        return 0.0

def calculate_cvar(returns, confidence_level=0.95):
    """計算CVaR_95"""
    try:
        if len(returns) == 0:
            return 0.0
        sorted_returns = np.sort(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        cvar = -sorted_returns[:index].mean() if index > 0 else 0.0
        return cvar
    except Exception as e:
        logger.error(f"CVaR_95計算失敗: {e}")
        asyncio.run(發送錯誤訊息(f"錯誤碼E202：CVaR_95計算失敗 {e}", None, None, "獎勵計算", "high"))
        return 0.0

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """計算年化夏普比率"""
    try:
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        mean_return = np.mean(returns) * 252
        std_return = np.std(returns) * np.sqrt(252)
        sharpe = (mean_return - risk_free_rate) / std_return if std_return != 0 else 0.0
        return sharpe
    except Exception as e:
        logger.error(f"夏普比率計算失敗: {e}")
        asyncio.run(發送錯誤訊息(f"錯誤碼E202：夏普比率計算失敗 {e}", None, None, "獎勵計算", "high"))
        return 0.0

async def monitor_long_term_reward_stability():
    """長期獎勵穩定性監控，每10輪檢查多市場加權標準差"""
    try:
        conn = sqlite3.connect(SQLite資料夾 / "reward_log.db")
        df = pd.read_sql_query(
            "SELECT 市場, 時間框架, 獎勵, 資金報酬率, 最大回撤, f1分數, 穩定性, 時間 FROM 獎勵記錄 ORDER BY 時間 DESC LIMIT 100",
            conn)
        conn.close()
        if df.empty:
            logger.warning("[多市場_多框架] 無長期獎勵穩定性數據")
            await 發送錯誤訊息("錯誤碼E201：無長期獎勵穩定性數據", "多市場", "多框架", "獎勵計算", "high")
            return False

        weighted_std = 0.0
        total_weight = 0.0
        fig = go.Figure()
        for market, timeframe in 市場清單:
            subset = df[(df["市場"] == market) & (df["時間框架"] == timeframe)]
            if not subset.empty:
                returns = subset["資金報酬率"].values
                weight = 0.5 if market == "BTCUSDT" and timeframe == "15m" else 0.5 / (len(市場清單) - 1)
                weighted_std += np.std(returns) * weight if len(returns) > 1 else 0.0
                total_weight += weight
                fig.add_trace(go.Scatter(
                    x=subset["時間"],
                    y=subset["資金報酬率"] * 100,
                    mode="lines+markers",
                    name=f"{market}_{timeframe}",
                    line=dict(color="green" if subset["資金報酬率"].mean() > 0 else "red")
                ))
        weighted_std = weighted_std / total_weight if total_weight > 0 else 0.0
        if weighted_std > 0.1:
            logger.warning(f"[多市場_多框架] 長期獎勵穩定性低，加權標準差: {weighted_std:.4f}")
            await 發送錯誤訊息(f"錯誤碼E201：長期獎勵穩定性低，加權標準差 {weighted_std:.4f}", "多市場", "多框架", "獎勵計算", "high")

        fig.update_layout(
            title="多市場長期獎勵趨勢",
            xaxis_title="時間",
            yaxis_title="報酬率 (%)",
            template="plotly_dark",
            hovermode="x unified",
            updatemenus=[
                dict(
                    buttons=[
                        dict(label="全選", method="update", args=[{"visible": [True] * len(fig.data)}]),
                        dict(label="僅BTCUSDT_15m", method="update", args=[{"visible": [i == 0 for i in range(len(fig.data))]}])
                    ],
                    direction="down",
                    showactive=True
                )
            ]
        )
        plot_path = SQLite資料夾.parent / "圖片" / f"多市場_長期獎勵趨勢_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(plot_path)
        await 發送通知(f"【執行通知】生成多市場長期獎勵趨勢圖表 {plot_path}", "多市場", "多框架", "獎勵計算", "normal")
        return weighted_std <= 0.1
    except Exception as e:
        logger.error(f"[多市場_多框架] 長期獎勵穩定性監控失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E201：長期獎勵穩定性監控失敗 {e}", "多市場", "多框架", "獎勵計算", "high")
        await 錯誤記錄與自動修復(f"錯誤碼E201：長期獎勵穩定性監控失敗 {e}", "穩定性監控錯誤", "多市場", "多框架", "獎勵計算")
        return False

async def dqn_optimize_weights(market, timeframe, technical_indicators, trading_result, retry_count=0):
    """使用 DQN 強化學習優化權重"""
    try:
        input_dim = 6  # 資金, 回撤, 勝率, f1分數, 穩定性, ATR
        output_dim = 6  # 調整最終資金、最大回撤、f1分數、穩定性、維持率、單筆損失的權重
        dqn = DQN(input_dim, output_dim).to(訓練設備)
        optimizer = torch.optim.Adam(dqn.parameters(), lr=1e-3)
        gamma = 0.99
        epsilon = 0.1
        episodes = 50
        batch_size = 32

        initial_funds = trading_result["最終資金"]
        base_weights = {
            "最終資金": 0.5,
            "最大回撤": -0.3,
            "f1分數": 0.2,
            "穩定性": -0.2,
            "維持率": 0.05,
            "單筆損失": -0.2
        }
        best_weights = base_weights.copy()
        best_reward = -float('inf')

        # 準備批量數據
        states = torch.tensor([[
            trading_result["最終資金"],
            trading_result["最大回撤"],
            trading_result.get("勝率", 0.0),
            trading_result.get("f1分數", 0.0),
            trading_result.get("穩定性", 0.0),
            technical_indicators.get("ATR_14", 訓練參數["ATR週期"]["值"])
        ] for _ in range(batch_size)], dtype=torch.float32, device=訓練設備)
        dataset = TensorDataset(states)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for episode in range(episodes):
            episode_reward = 0.0
            for batch_state in dataloader:
                batch_state = batch_state[0]
                if np.random.random() < epsilon:
                    action = np.random.randint(0, output_dim)
                else:
                    with torch.no_grad():
                        q_values = dqn(batch_state[0])
                        action = q_values.argmax().item()

                # 執行動作
                temp_weights = best_weights.copy()
                weight_keys = list(temp_weights.keys())
                if action < len(weight_keys):
                    key = weight_keys[action]
                    temp_weights[key] = min(max(temp_weights[key] + 0.05 * (1 if action % 2 == 0 else -1), -0.5), 0.5)

                # 計算獎勵
                returns = [record["損益"] / initial_funds for record in trading_result["交易記錄"] if "損益" in record]
                資金報酬率 = (trading_result["最終資金"] - initial_funds) / initial_funds
                標準化資金報酬率 = max(min(資金報酬率, 2.0), -2.0)
                標準化回撤 = max(min(trading_result["最大回撤"], 0.5), 0.0)
                標準化f1分數 = max(min(trading_result.get("f1分數", 0.0), 1.0), 0.0)
                標準化穩定性 = max(min(trading_result.get("穩定性", 0.0), 0.1), 0.0)
                標準化維持率 = max(min((trading_result["維持率"] - 100) / 100, 1.0), -1.0)
                標準化單筆損失 = max(min(min(returns) if returns else 0.0, 0.1), -0.1)

                reward = (
                    temp_weights["最終資金"] * 標準化資金報酬率 +
                    temp_weights["最大回撤"] * 標準化回撤 +
                    temp_weights["f1分數"] * 標準化f1分數 +
                    temp_weights["穩定性"] * 標準化穩定性 +
                    temp_weights["維持率"] * 標準化維持率 +
                    temp_weights["單筆損失"] * 標準化單筆損失
                )

                episode_reward += reward
                next_state = torch.tensor([
                    trading_result["最終資金"],
                    trading_result["最大回撤"],
                    trading_result.get("勝率", 0.0),
                    trading_result.get("f1分數", 0.0),
                    trading_result.get("穩定性", 0.0),
                    technical_indicators.get("ATR_14", 訓練參數["ATR週期"]["值"])
                ], dtype=torch.float32, device=訓練設備)

                # 更新 DQN (GPU加速)
                if torch.cuda.is_available():
                    q_value = dqn(batch_state[0])[action].cuda()
                    target = (reward + gamma * dqn(next_state).max()).cuda()
                    loss = nn.MSELoss()(q_value, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    q_value = dqn(batch_state[0])[action]
                    target = reward + gamma * dqn(next_state).max()
                    loss = nn.MSELoss()(q_value, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if reward > best_reward:
                    best_reward = reward
                    best_weights = temp_weights.copy()

                torch.cuda.empty_cache()

            # 儲存每輪結果
            conn = sqlite3.connect(SQLite資料夾 / "權重優化記錄.db")
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS 權重優化記錄 (
                    id TEXT PRIMARY KEY,
                    市場 TEXT,
                    時間框架 TEXT,
                    最終資金權重 REAL,
                    最大回撤權重 REAL,
                    f1分數權重 REAL,
                    穩定性權重 REAL,
                    維持率權重 REAL,
                    單筆損失權重 REAL,
                    獎勵 REAL,
                    時間 TEXT
                )
            """)
            c.execute("INSERT INTO 權重優化記錄 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (str(uuid.uuid4()), market, timeframe, best_weights["最終資金"],
                       best_weights["最大回撤"], best_weights["f1分數"], best_weights["穩定性"],
                       best_weights["維持率"], best_weights["單筆損失"], episode_reward,
                       datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
            conn.close()

        # 檢查穩定性
        conn = sqlite3.connect(SQLite資料夾 / "權重優化記錄.db")
        df = pd.read_sql_query("SELECT 最終資金權重, 最大回撤權重, f1分數權重, 穩定性權重, 維持率權重, 單筆損失權重 FROM 權重優化記錄 WHERE 市場 = ? AND 時間框架 = ? ORDER BY 時間 DESC LIMIT 10",
                               conn, params=(market, timeframe))
        conn.close()
        if len(df) > 1:
            for column in df.columns:
                if df[column].std() > 0.1:
                    logger.warning(f"[{market}_{timeframe}] 權重 {column} 穩定性低，標準差: {df[column].std():.4f}")
                    await 發送錯誤訊息(f"錯誤碼E204：權重 {column} 穩定性低，標準差 {df[column].std():.4f}", market, timeframe, "獎勵計算", "high")
        
        await 記錄參數異動("動態權重", base_weights, best_weights, f"{market}_{timeframe} DQN調整")
        await 發送通知(f"【通知】DQN 權重優化完成: {market}_{timeframe}\n權重: {best_weights}\n獎勵: {best_reward:.4f}", market, timeframe, "獎勵計算", "normal")
        return best_weights
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[{market}_{timeframe}] DQN 權重優化失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E204：DQN 權重優化失敗 重試{retry_count + 1}/5: {e}", "DQN 權重優化錯誤", market, timeframe, "獎勵計算")
            await asyncio.sleep(5)
            return await dqn_optimize_weights(market, timeframe, technical_indicators, trading_result, retry_count + 1)
        logger.error(f"[{market}_{timeframe}] DQN 權重優化失敗，重試 5 次無效: {e}")
        await 發送錯誤訊息(f"錯誤碼E204：DQN 權重優化失敗 {e}\n動作：返回預設權重", market, timeframe, "獎勵計算", "high")
        await 錯誤記錄與自動修復(f"錯誤碼E204：DQN 權重優化失敗 {e}", "DQN 權重優化錯誤", market, timeframe, "獎勵計算")
        return {
            "最終資金": 0.5,
            "最大回撤": -0.3,
            "f1分數": 0.2,
            "穩定性": -0.2,
            "維持率": 0.05,
            "單筆損失": -0.2
        }

async def calculate_single_market_reward(trading_result, market="BTCUSDT", timeframe="15m", technical_indicators=None, custom_metrics=None, retry_count=0):
    """計算單一市場 DQN 獎勵，整合多維指標與動態權重，支持自定義指標"""
    try:
        start_time = time.time()
        if not await validate_input(trading_result, market, timeframe, technical_indicators):
            return 0.0

        # 硬體監控
        batch_size = 32
        batch_size, _ = await 監控硬體狀態並降級(batch_size, 2)
        if batch_size < 8:
            logger.error(f"[{market}_{timeframe}] 硬體資源超限，批次大小 {batch_size}")
            await 發送錯誤訊息(f"錯誤碼E203：硬體資源超限，批次大小 {batch_size}", market, timeframe, "獎勵計算", "high")
            await 錯誤記錄與自動修復(f"錯誤碼E203：硬體資源超限，批次大小 {batch_size}", "硬體資源錯誤", market, timeframe, "獎勵計算")
            return 0.0
        torch.cuda.empty_cache()

        # 提取交易數據
        initial_funds = 1000.0  # 固定初始資金
        final_funds = trading_result["最終資金"]
        max_drawdown = trading_result["最大回撤"]
        trading_records = trading_result["交易記錄"]
        margin_ratio = trading_result["維持率"]
        f1_score = trading_result.get("f1分數", 0.0)
        stability = trading_result.get("穩定性", 0.0)

        # 計算報酬率與損失
        returns = [record["損益"] / initial_funds for record in trading_records if "損益" in record]
        single_loss = min(returns) if returns else 0.0
        holding_time = sum(record["持倉時間"] for record in trading_records) / len(trading_records) if trading_records else 0

        # 風險指標
        var_95 = calculate_var(returns)
        cvar_95 = calculate_cvar(returns)
        sharpe_ratio = calculate_sharpe_ratio(returns)

        # 風險指標檢查
        if var_95 > 0.05 or cvar_95 > 0.05 or sharpe_ratio < 1.5 or max_drawdown > 0.25:
            logger.warning(f"[{market}_{timeframe}] 風險指標異常: VaR_95={var_95:.4f}, CVaR_95={cvar_95:.4f}, Sharpe={sharpe_ratio:.4f}, 最大回撤={max_drawdown:.2%}")
            await 發送錯誤訊息(f"錯誤碼E202：風險指標異常: VaR_95={var_95:.4f}, CVaR_95={cvar_95:.4f}, Sharpe={sharpe_ratio:.4f}, 最大回撤={max_drawdown:.2%}", market, timeframe, "獎勵計算", "high")
            await 錯誤記錄與自動修復(f"錯誤碼E202：風險指標異常", "風險指標錯誤", market, timeframe, "獎勵計算")

        # 動態權重
        weights = await dqn_optimize_weights(market, timeframe, technical_indicators, trading_result)

        # 標準化指標
        return_rate = (final_funds - initial_funds) / initial_funds
        standardized_return_rate = max(min(return_rate, 2.0), -2.0)
        standardized_drawdown = max(min(max_drawdown, 0.25), 0.0)
        standardized_f1_score = max(min(f1_score, 1.0), 0.0)
        standardized_stability = max(min(stability, 0.1), 0.0)
        standardized_margin_ratio = max(min((margin_ratio - 100) / 100, 1.0), -1.0)
        standardized_single_loss = max(min(single_loss, 0.1), -0.1)

        # 自定義指標處理
        custom_reward = 0.0
        if custom_metrics:
            for metric_name, metric_value in custom_metrics.items():
                if metric_name == "holding_time":
                    standardized_custom = max(min(holding_time / 100, 1.0), 0.0)  # 假設100為標準化上限
                elif metric_name == "win_rate":
                    win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0.0
                    standardized_custom = max(min(win_rate, 1.0), 0.0)
                else:
                    standardized_custom = max(min(metric_value, 1.0), 0.0)  # 默認標準化
                custom_reward += standardized_custom * 0.1  # 自定義指標權重

        # 計算獎勵
        reward = (
            weights["最終資金"] * standardized_return_rate +
            weights["最大回撤"] * standardized_drawdown +
            weights["f1分數"] * standardized_f1_score +
            weights["穩定性"] * standardized_stability +
            weights["維持率"] * standardized_margin_ratio +
            weights["單筆損失"] * standardized_single_loss +
            custom_reward
        )

        # 長期穩定性監控（每10輪）
        _reward_buffer.append(reward)
        if len(_reward_buffer) == _reward_buffer.maxlen:
            await monitor_long_term_reward_stability()

        # 儲存獎勵記錄
        conn = sqlite3.connect(SQLite資料夾 / "reward_log.db")
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS 獎勵記錄 (
                id TEXT PRIMARY KEY,
                市場 TEXT,
                時間框架 TEXT,
                獎勵 REAL,
                資金報酬率 REAL,
                最大回撤 REAL,
                f1分數 REAL,
                穩定性 REAL,
                維持率 REAL,
                單筆損失 REAL,
                var_95 REAL,
                cvar_95 REAL,
                sharpe_ratio REAL,
                時間 TEXT
            )
        """)
        c.execute("INSERT INTO 獎勵記錄 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (str(uuid.uuid4()), market, timeframe, reward, return_rate, max_drawdown, f1_score, stability,
                   margin_ratio, single_loss, var_95, cvar_95, sharpe_ratio,
                   datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()

        # 生成報表
        df = pd.DataFrame([{
            "市場": market,
            "時間框架": timeframe,
            "獎勵": reward,
            "最終資金": final_funds,
            "最大回撤": max_drawdown,
            "f1分數": f1_score,
            "穩定性": stability,
            "維持率": margin_ratio,
            "單筆損失": single_loss
        }])
        if not df.empty:
            csv_path = SQLite資料夾.parent / "備份" / f"{market}_{timeframe}_Reward_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            await 發送通知(f"【執行通知】生成獎勵報表 {csv_path}", market, timeframe, "獎勵計算", "normal")

        # 效率報告
        elapsed_time = time.time() - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】獎勵計算耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await 發送通知(efficiency_report, market, timeframe, "獎勵計算", "normal")

        return reward
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[{market}_{timeframe}] 獎勵計算失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E201：獎勵計算失敗 重試{retry_count + 1}/5: {e}", "獎勵計算錯誤", market, timeframe, "獎勵計算")
            await asyncio.sleep(5)
            return await calculate_single_market_reward(trading_result, market, timeframe, technical_indicators, custom_metrics, retry_count + 1)
        logger.error(f"[{market}_{timeframe}] 獎勵計算失敗，重試 5 次無效: {e}")
        await 發送錯誤訊息(f"錯誤碼E201：獎勵計算失敗 {e}\n動作：中止流程", market, timeframe, "獎勵計算", "high")
        await 錯誤記錄與自動修復(f"錯誤碼E201：獎勵計算失敗 {e}", "獎勵計算錯誤", market, timeframe, "獎勵計算")
        return 0.0

async def calculate_multi_market_reward(results, params_list):
    """計算多市場加權獎勵，支持並行計算"""
    try:
        total_reward = 0.0
        reward_data = []
        valid_markets = 0
        tasks = []
        for (market, timeframe), result in results.items():
            if result is None:
                continue
            technical_indicators = {
                "HMA_16": 訓練參數["HMA週期"]["值"],
                "SMA50": 訓練參數["SMA週期"]["值"],
                "ATR_14": 訓練參數["ATR週期"]["值"]
            }
            task = asyncio.create_task(calculate_single_market_reward(result, market, timeframe, technical_indicators))
            tasks.append((market, timeframe, task))

        for market, timeframe, task in tasks:
            reward = await task
            weight = 0.5 if market == "BTCUSDT" and timeframe == "15m" else 0.5 / (len(市場清單) - 1)
            total_reward += reward * weight
            valid_markets += 1
            reward_data.append({
                "市場": market,
                "時間框架": timeframe,
                "獎勵": reward,
                "最終資金": results[(market, timeframe)]["最終資金"],
                "最大回撤": results[(market, timeframe)]["最大回撤"],
                "f1分數": results[(market, timeframe)].get("f1分數", 0.0),
                "穩定性": results[(market, timeframe)].get("穩定性", 0.0)
            })

        if valid_markets == 0:
            logger.error("[多市場_多框架] 無有效獎勵數據")
            await 發送錯誤訊息("錯誤碼E201：無有效獎勵數據", "多市場", "多框架", "獎勵計算", "high")
            await 錯誤記錄與自動修復("錯誤碼E201：無有效獎勵數據", "多市場獎勵錯誤", "多市場", "多框架", "獎勵計算")
            return 0.0, {}

        # 生成報表
        df = pd.DataFrame(reward_data)
        if not df.empty:
            csv_path = SQLite資料夾.parent / "備份" / f"Multi_Market_Reward_Comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            await 發送通知(f"【執行通知】生成多市場獎勵比較報表 {csv_path}", "多市場", "多框架", "獎勵計算", "normal")

        return total_reward / valid_markets if valid_markets > 0 else 0.0, reward_data
    except Exception as e:
        logger.error(f"[多市場_多框架] 多市場獎勵計算失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E201：多市場獎勵計算失敗 {e}\n動作：中止流程", "多市場", "多框架", "獎勵計算", "high")
        await 錯誤記錄與自動修復(f"錯誤碼E201：多市場獎勵計算失敗 {e}", "多市場獎勵錯誤", "多市場", "多框架", "獎勵計算")
        return 0.0, {}