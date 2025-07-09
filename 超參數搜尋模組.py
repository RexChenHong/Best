import optuna
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
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from 設定檔 import 訓練設備, SQLite資料夾, 訓練參數, 市場清單, 記錄參數異動
from 工具模組 import 錯誤記錄與自動修復, 監控硬體狀態並降級
from 推播通知模組 import 發送錯誤訊息, 發送通知
from 交易環境模組 import single_market_trading_env
from 獎勵計算模組 import calculate_single_market_reward
from skopt import gp_minimize
import scipy.stats as stats

# 配置日誌
log_dir = SQLite資料夾.parent / "日誌"
log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("超參數搜尋模組")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
file_handler = TimedRotatingFileHandler(
    filename=log_dir / "hyperparameter_search_logs",
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
    "E301": "風險指標異常",
    "E302": "超參數搜尋失敗",
    "E303": "硬體資源超限",
    "E304": "KFold交叉驗證失敗"
}

# 模型結構定義
class MLP(nn.Module):
    def __init__(self, input_dim, num_layers, num_neurons, dropout_ratio):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, num_neurons))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_ratio))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_ratio))
        layers.append(nn.Linear(num_neurons, 3))  # Long, Short, No Action
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class CNN(nn.Module):
    def __init__(self, input_dim, num_layers, num_neurons, dropout_ratio):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, num_neurons, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_neurons, num_neurons, kernel_size=3, padding=1)
        self.fc = nn.Linear(num_neurons * input_dim, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_ratio)
        self.num_layers = num_layers

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, input_dim]
        x = self.relu(self.conv1(x))
        if self.num_layers > 1:
            x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class LSTM(nn.Module):
    def __init__(self, input_dim, num_layers, num_neurons, dropout_ratio):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, num_neurons, num_layers, batch_first=True, dropout=dropout_ratio if num_layers > 1 else 0)
        self.fc = nn.Linear(num_neurons, 3)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, input_dim]
        _, (hn, _) = self.lstm(x)
        x = self.dropout(hn[-1])
        x = self.fc(x)
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim, num_layers, num_neurons, dropout_ratio):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=num_neurons, dropout=dropout_ratio)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, 3)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, input_dim]
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x

class GNN(nn.Module):
    def __init__(self, input_dim, num_layers, num_neurons, dropout_ratio):
        super(GNN, self).__init__()
        self.conv1 = nn.Linear(input_dim, num_neurons)
        self.conv2 = nn.Linear(num_neurons, num_neurons)
        self.fc = nn.Linear(num_neurons, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_ratio)
        self.num_layers = num_layers

    def forward(self, x):
        x = self.relu(self.conv1(x))
        if self.num_layers > 1:
            x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.fc(x)
        return x

async def validate_input(market_signal_mapping, 資產類型, market, timeframe):
    """驗證輸入數據"""
    try:
        if not market_signal_mapping or (market, timeframe) not in market_signal_mapping:
            logger.error(f"[{market}_{timeframe}] 無效信號映射")
            await 發送錯誤訊息(f"錯誤碼E302：無效信號映射", market, timeframe, "超參數搜尋")
            return False
        if 資產類型 not in ["虛擬貨幣", "CFD"]:
            logger.error(f"[{market}_{timeframe}] 無效資產類型: {資產類型}")
            await 發送錯誤訊息(f"錯誤碼E302：無效資產類型 {資產類型}", market, timeframe, "超參數搜尋")
            return False
        if market not in [mkt for mkt, _ in 市場清單]:
            logger.error(f"[{market}_{timeframe}] 無效市場: {market}")
            await 發送錯誤訊息(f"錯誤碼E302：無效市場 {market}", market, timeframe, "超參數搜尋")
            return False
        signals = market_signal_mapping[(market, timeframe)]["信號"]
        prices = market_signal_mapping[(market, timeframe)]["價格"]
        if not signals or not prices:
            logger.error(f"[{market}_{timeframe}] 信號或價格數據為空")
            await 發送錯誤訊息(f"錯誤碼E302：信號或價格數據為空", market, timeframe, "超參數搜尋")
            return False
        return True
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 輸入驗證失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E302：輸入驗證失敗 {e}", market, timeframe, "超參數搜尋")
        await 錯誤記錄與自動修復(f"錯誤碼E302：輸入驗證失敗 {e}", "輸入驗證錯誤", market, timeframe, "超參數搜尋")
        return False

async def monitor_search_stability():
    """長期超參數搜尋穩定性監控，每10輪檢查加權標準差"""
    try:
        conn = sqlite3.connect(SQLite資料夾 / "最佳參數記錄.db")
        df = pd.read_sql_query(
            "SELECT 市場, 週期, 學習率, 批次大小, Dropout比率, 層數, 神經元數, Sharpe, 最大回撤, 記錄時間 FROM 最佳參數表 ORDER BY 記錄時間 DESC LIMIT 100",
            conn)
        conn.close()
        if df.empty:
            logger.warning("[多市場_多框架] 無長期穩定性數據")
            await 發送錯誤訊息("錯誤碼E302：無長期穩定性數據", "多市場", "多框架", "超參數搜尋")
            return False

        weighted_std = 0.0
        total_weight = 0.0
        fig = go.Figure()
        for market, timeframe in 市場清單:
            subset = df[(df["市場"] == market) & (df["週期"] == timeframe)]
            if not subset.empty:
                sharpe_std = subset["Sharpe"].std() if len(subset["Sharpe"]) > 1 else 0.0
                weight = 0.5 if market == "BTCUSDT" and timeframe == "15m" else 0.5 / (len(市場清單) - 1)
                weighted_std += sharpe_std * weight
                total_weight += weight
                fig.add_trace(go.Scatter(
                    x=subset["記錄時間"],
                    y=subset["Sharpe"],
                    mode="lines+markers",
                    name=f"{market}_{timeframe}",
                    line=dict(color="green" if subset["Sharpe"].mean() > 1.5 else "red")
                ))
        weighted_std = weighted_std / total_weight if total_weight > 0 else 0.0
        if weighted_std > 0.1 or df["Sharpe"].mean() < 1.5 or df["最大回撤"].max() > 0.25:
            logger.warning(f"[多市場_多框架] 超參數穩定性異常: 加權標準差={weighted_std:.4f}, 平均Sharpe={df['Sharpe'].mean():.4f}, 最大回撤={df['最大回撤'].max():.2%}")
            await 發送錯誤訊息(f"錯誤碼E301：超參數穩定性異常\n加權標準差={weighted_std:.4f}, 平均Sharpe={df['Sharpe'].mean():.4f}, 最大回撤={df['最大回撤'].max():.2%}", "多市場", "多框架", "超參數搜尋")
            await 錯誤記錄與自動修復(f"錯誤碼E301：超參數穩定性異常", "穩定性錯誤", "多市場", "多框架", "超參數搜尋")

        fig.update_layout(
            title="多市場超參數搜尋Sharpe趨勢",
            xaxis_title="時間",
            yaxis_title="Sharpe比率",
            template="plotly_dark",
            hovermode="x unified"
        )
        plot_path = SQLite資料夾.parent / "圖片" / f"多市場_超參數趨勢_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(plot_path)
        await 發送通知(f"【執行通知】生成多市場超參數趨勢圖表 {plot_path}", "多市場", "多框架", "超參數搜尋")
        return weighted_std <= 0.1
    except Exception as e:
        logger.error(f"[多市場_多框架] 長期穩定性監控失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E302：長期穩定性監控失敗 {e}", "多市場", "多框架", "超參數搜尋")
        await 錯誤記錄與自動修復(f"錯誤碼E302：長期穩定性監控失敗 {e}", "穩定性監控錯誤", "多市場", "多框架", "超參數搜尋")
        return False

async def kfold_cross_validation(signals, prices, params, model_type, input_dim, batch_size):
    """執行5折KFold交叉驗證"""
    try:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        train_losses = []
        val_losses = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(signals)):
            train_signals = signals[train_idx]
            train_prices = prices[train_idx]
            val_signals = signals[val_idx]
            val_prices = prices[val_idx]

            # 初始化模型
            if model_type == "MLP":
                model = MLP(input_dim, params["num_layers"], params["num_neurons"], params["dropout_ratio"]).to(訓練設備)
            elif model_type == "CNN":
                model = CNN(input_dim, params["num_layers"], params["num_neurons"], params["dropout_ratio"]).to(訓練設備)
            elif model_type == "LSTM":
                model = LSTM(input_dim, params["num_layers"], params["num_neurons"], params["dropout_ratio"]).to(訓練設備)
            elif model_type == "Transformer":
                model = Transformer(input_dim, params["num_layers"], params["num_neurons"], params["dropout_ratio"]).to(訓練設備)
            elif model_type == "GNN":
                model = GNN(input_dim, params["num_layers"], params["num_neurons"], params["dropout_ratio"]).to(訓練設備)
            else:
                logger.error(f"無效模型類型: {model_type}")
                return float('inf'), float('inf')

            # 初始化優化器
            if params["optimizer"] == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
            elif params["optimizer"] == "SGD":
                optimizer = torch.optim.SGD(model.parameters(), lr=params["learning_rate"])
            elif params["optimizer"] == "RMSprop":
                optimizer = torch.optim.RMSprop(model.parameters(), lr=params["learning_rate"])
            criterion = nn.CrossEntropyLoss()

            # 訓練數據
            train_dataset = TensorDataset(
                torch.tensor(train_signals, dtype=torch.float32, device=訓練設備),
                torch.tensor(train_prices, dtype=torch.float32, device=訓練設備)
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataset = TensorDataset(
                torch.tensor(val_signals, dtype=torch.float32, device=訓練設備),
                torch.tensor(val_prices, dtype=torch.float32, device=訓練設備)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # 訓練
            model.train()
            train_loss = 0.0
            for epoch in range(10):  # 假設10個epoch
                for batch_signals, _ in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_signals)
                    loss = criterion(outputs, torch.zeros(batch_signals.size(0), dtype=torch.long, device=訓練設備))
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)

            # 驗證
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_signals, _ in val_loader:
                    outputs = model(batch_signals)
                    loss = criterion(outputs, torch.zeros(batch_signals.size(0), dtype=torch.long, device=訓練設備))
                    val_loss += loss.item()
                val_loss /= len(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            torch.cuda.empty_cache()

        return sum(train_losses) / len(train_losses), sum(val_losses) / len(val_losses)
    except Exception as e:
        logger.error(f"KFold交叉驗證失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E304：KFold交叉驗證失敗 {e}", None, None, "超參數搜尋")
        await 錯誤記錄與自動修復(f"錯誤碼E304：KFold交叉驗證失敗 {e}", "KFold錯誤", None, None, "超參數搜尋")
        return float('inf'), float('inf')

async def objective(trial, market_signal_mapping, 資產類型, market, timeframe, batch_size):
    """Optuna試驗目標函數"""
    try:
        # 定義超參數搜尋範圍（規格書4.1.1）
        params = {
            "model_type": trial.suggest_categorical("model_type", ["MLP", "CNN", "LSTM", "Transformer", "GNN"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "dropout_ratio": trial.suggest_float("dropout_ratio", 0.1, 0.5),
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "num_neurons": trial.suggest_int("num_neurons", 64, 256),
            "optimizer": trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"]),
            "sma_enabled": trial.suggest_categorical("sma_enabled", [True, False]),
            "sma_period": trial.suggest_int("sma_period", 10, 200),
            "hma_period": trial.suggest_int("hma_period", 8, 30),
            "hma_weight": trial.suggest_float("hma_weight", 0.1, 1.0),
            "atr_period": trial.suggest_int("atr_period", 7, 30),
            "atr_weight": trial.suggest_float("atr_weight", 0.1, 1.0),
            "vhf_period": trial.suggest_int("vhf_period", 14, 100),
            "vhf_threshold": trial.suggest_float("vhf_threshold", 0.1, 1.0),
            "pivot_enabled": trial.suggest_categorical("pivot_enabled", [True, False]),
            "pivot_period": trial.suggest_int("pivot_period", 3, 20),
            "strong_signal_threshold": trial.suggest_float("strong_signal_threshold", 0.5, 0.95),
            "stop_loss": trial.suggest_float("stop_loss", 0.01, 0.05),
            "take_profit": trial.suggest_float("take_profit", 0.02, 0.1),
            "trailing_stop": trial.suggest_float("trailing_stop", 0.01, 0.05),
            "breakeven_trigger": trial.suggest_float("breakeven_trigger", 0.01, 0.05),
            "single_loss_limit": trial.suggest_float("single_loss_limit", 0.01, 0.03),
            "signal_weight": trial.suggest_float("signal_weight", 0.1, 1.0),
            "condition_threshold": trial.suggest_int("condition_threshold", 1, 5),
            "signal_threshold": trial.suggest_float("signal_threshold", 0.5, 0.9)
        }

        # KFold交叉驗證
        signals = np.array(market_signal_mapping[(market, timeframe)]["信號"])
        prices = np.array(market_signal_mapping[(market, timeframe)]["價格"])
        input_dim = signals.shape[1] if len(signals.shape) > 1 else 1
        train_loss, val_loss = await kfold_cross_validation(signals, prices, params, params["model_type"], input_dim, params["batch_size"])

        # 模擬交易
        result = await single_market_trading_env(
            market_signal_mapping=market_signal_mapping,
            資產類型=資產類型,
            市場=market,
            時間框架=timeframe,
            params=params,
            batch_size=batch_size
        )

        if result is None or (market, timeframe) not in result:
            logger.warning(f"[{market}_{timeframe}] 模擬交易失敗，返回低獎勵")
            return -1.0

        trading_result = result[(market, timeframe)]
        initial_funds = 1000.0  # 假設初始資金
        returns = [(record["損益"] / initial_funds) for record in trading_result["交易記錄"] if "損益" in record]
        max_drawdown = trading_result["最大回撤"]
        sharpe_ratio = calculate_sharpe_ratio(returns)

        # 風險指標檢查
        if max_drawdown > 0.25 or any(abs(r) > params["single_loss_limit"] for r in returns if r < 0):
            logger.warning(f"[{market}_{timeframe}] 風險指標異常: 最大回撤={max_drawdown:.2%}, 單筆損失超限")
            await 發送錯誤訊息(f"錯誤碼E301：風險指標異常\n最大回撤={max_drawdown:.2%}, 單筆損失超限", market, timeframe, "超參數搜尋")
            await 錯誤記錄與自動修復(f"錯誤碼E301：風險指標異常", "風險指標錯誤", market, timeframe, "超參數搜尋")
            return -1.0

        # 計算獎勵
        technical_indicators = {
            "HMA_16": params["hma_period"],
            "SMA50": params["sma_period"],
            "ATR_14": params["atr_period"]
        }
        reward = await calculate_single_market_reward(trading_result, market, timeframe, technical_indicators)

        # 儲存試驗結果
        conn = sqlite3.connect(SQLite資料夾 / "最佳參數記錄.db")
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS 最佳參數表 (
                UUID TEXT PRIMARY KEY,
                市場 TEXT,
                週期 TEXT,
                模型 TEXT,
                Dropout設定 BOOLEAN,
                學習率 REAL,
                批次大小 INTEGER,
                層數 INTEGER,
                神經元數 INTEGER,
                Optimizer TEXT,
                訓練Loss REAL,
                驗證Loss REAL,
                Sharpe REAL,
                最大回撤 REAL,
                記錄時間 TEXT
            )
        """)
        c.execute("INSERT INTO 最佳參數表 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (str(uuid.uuid4()), market, timeframe, params["model_type"], params["dropout_ratio"] > 0,
                   params["learning_rate"], params["batch_size"], params["num_layers"], params["num_neurons"],
                   params["optimizer"], train_loss, val_loss, sharpe_ratio, max_drawdown,
                   datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()

        trial.set_user_attr("max_drawdown", max_drawdown)
        trial.set_user_attr("train_loss", train_loss)
        trial.set_user_attr("val_loss", val_loss)
        return reward if sharpe_ratio >= 1.5 and max_drawdown <= 0.25 else -1.0
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 試驗目標函數失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E302：試驗目標函數失敗 {e}", market, timeframe, "超參數搜尋")
        await 錯誤記錄與自動修復(f"錯誤碼E302：試驗目標函數失敗 {e}", "試驗目標錯誤", market, timeframe, "超參數搜尋")
        return -1.0

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
        asyncio.run(發送錯誤訊息(f"錯誤碼E301：夏普比率計算失敗 {e}", None, None, "超參數搜尋"))
        return 0.0

async def hyperparameter_search(market_signal_mapping, 資產類型, market, timeframe, n_trials=100, retry_count=0):
    """執行超參數搜尋，支援多市場與KFold交叉驗證"""
    try:
        start_time = time.time()
        if not await validate_input(market_signal_mapping, 資產類型, market, timeframe):
            return None, 0.0

        # 硬體監控
        batch_size = 32
        batch_size, _ = await 監控硬體狀態並降級(batch_size, 2)
        if batch_size < 8:
            logger.error(f"[{market}_{timeframe}] 硬體資源超限，批次大小 {batch_size}")
            await 發送錯誤訊息(f"錯誤碼E303：硬體資源超限，批次大小 {batch_size}", market, timeframe, "超參數搜尋")
            await 錯誤記錄與自動修復(f"錯誤碼E303：硬體資源超限，批次大小 {batch_size}", "硬體資源錯誤", market, timeframe, "超參數搜尋")
            return None, 0.0
        torch.cuda.empty_cache()
        gpu_switch_count = 0

        # 準備批量數據
        signals = np.array(market_signal_mapping[(market, timeframe)]["信號"])
        prices = np.array(market_signal_mapping[(market, timeframe)]["價格"])
        dataset = TensorDataset(
            torch.tensor(signals, dtype=torch.float32, device=訓練設備),
            torch.tensor(prices, dtype=torch.float32, device=訓練設備)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # 創建Optuna研究
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: asyncio.run(objective(trial, market_signal_mapping, 資產類型, market, timeframe, batch_size)),
            n_trials=n_trials,
            n_jobs=1
        )

        # 貝葉斯優化
        def bayesian_objective(params):
            return -asyncio.run(objective(optuna.trial.FixedTrial(params), market_signal_mapping, 資產類型, market, timeframe, batch_size))
        space = [
            (1e-5, 1e-2, "log-uniform"),  # learning_rate
            (32, 256),                    # batch_size
            (0.1, 0.5),                   # dropout_ratio
            (1, 4),                       # num_layers
            (64, 256),                    # num_neurons
            (0, 1),                       # sma_enabled
            (10, 200),                    # sma_period
            (8, 30),                      # hma_period
            (0.1, 1.0),                   # hma_weight
            (7, 30),                      # atr_period
            (0.1, 1.0),                   # atr_weight
            (14, 100),                    # vhf_period
            (0.1, 1.0),                   # vhf_threshold
            (0, 1),                       # pivot_enabled
            (3, 20),                      # pivot_period
            (0.5, 0.95),                  # strong_signal_threshold
            (0.01, 0.05),                 # stop_loss
            (0.02, 0.1),                  # take_profit
            (0.01, 0.05),                 # trailing_stop
            (0.01, 0.05),                 # breakeven_trigger
            (0.01, 0.03),                 # single_loss_limit
            (0.1, 1.0),                   # signal_weight
            (1, 5),                       # condition_threshold
            (0.5, 0.9)                    # signal_threshold
        ]
        bayesian_result = gp_minimize(bayesian_objective, space, n_calls=20, random_state=42)
        bayesian_params = dict(zip(["learning_rate", "batch_size", "dropout_ratio", "num_layers", "num_neurons", "sma_enabled", "sma_period", "hma_period", "hma_weight", "atr_period", "atr_weight", "vhf_period", "vhf_threshold", "pivot_enabled", "pivot_period", "strong_signal_threshold", "stop_loss", "take_profit", "trailing_stop", "breakeven_trigger", "single_loss_limit", "signal_weight", "condition_threshold", "signal_threshold"], bayesian_result.x))
        bayesian_reward = -bayesian_result.fun

        # 選擇最佳結果
        best_params = study.best_params if study.best_value > bayesian_reward else bayesian_params
        best_reward = max(study.best_value, bayesian_reward)

        # 長期穩定性監控（每10輪）
        if study.best_trial.number % 10 == 0:
            await monitor_search_stability()

        # 儲存最佳參數
        conn = sqlite3.connect(SQLite資料夾 / "最佳參數記錄.db")
        c = conn.cursor()
        c.execute("INSERT INTO 最佳參數表 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (str(uuid.uuid4()), market, timeframe, best_params["model_type"], best_params["dropout_ratio"] > 0,
                   best_params["learning_rate"], best_params["batch_size"], best_params["num_layers"],
                   best_params["num_neurons"], best_params["optimizer"],
                   study.best_trial.user_attrs.get("train_loss", 1.0),
                   study.best_trial.user_attrs.get("val_loss", 1.0),
                   calculate_sharpe_ratio([t.value for t in study.trials if t.value is not None]),
                   study.best_trial.user_attrs.get("max_drawdown", 0.0),
                   datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()

        # 生成報表
        trials_df = pd.DataFrame([
            {
                "試驗編號": trial.number,
                "獎勵": trial.value,
                "模型類型": trial.params.get("model_type", "MLP"),
                "學習率": trial.params.get("learning_rate", 0.0),
                "批次大小": trial.params.get("batch_size", 0),
                "Dropout比率": trial.params.get("dropout_ratio", 0.0),
                "層數": trial.params.get("num_layers", 0),
                "神經元數": trial.params.get("num_neurons", 0),
                "訓練Loss": trial.user_attrs.get("train_loss", 1.0),
                "驗證Loss": trial.user_attrs.get("val_loss", 1.0),
                "最大回撤": trial.user_attrs.get("max_drawdown", 0.0),
                "時間": trial.datetime_complete.strftime('%Y-%m-%d %H:%M:%S') if trial.datetime_complete else ""
            }
            for trial in study.trials if trial.value is not None
        ])
        if not trials_df.empty:
            csv_path = SQLite資料夾.parent / "備份" / f"{market}_{timeframe}_Hyperparameter_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            trials_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            await 發送通知(f"【執行通知】生成超參數搜尋報表 {csv_path}", market, timeframe, "超參數搜尋")

        # 效率報告
        elapsed_time = time.time() - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】超參數搜尋耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await 發送通知(efficiency_report, market, timeframe, "超參數搜尋")

        await 記錄參數異動("超參數搜尋", {}, best_params, f"{market}_{timeframe} Optuna最佳參數")
        await 發送通知(
            f"【超參數搜尋完成】\n市場：{market}_{timeframe}\n最佳參數：{best_params}\nSharpe：{calculate_sharpe_ratio([t.value for t in study.trials if t.value is not None]):.2f}\n最大回撤：{study.best_trial.user_attrs.get('max_drawdown', 0.0):.2%}\n時間：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            market, timeframe, "超參數搜尋"
        )
        return best_params, best_reward
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[{market}_{timeframe}] 超參數搜尋失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E302：超參數搜尋失敗 重試{retry_count + 1}/5: {e}", "超參數搜尋錯誤", market, timeframe, "超參數搜尋")
            await asyncio.sleep(5)
            return await hyperparameter_search(market_signal_mapping, 資產類型, market, timeframe, n_trials, retry_count + 1)
        logger.error(f"[{market}_{timeframe}] 超參數搜尋失敗，重試 5 次無效: {e}")
        await 發送錯誤訊息(f"錯誤碼E302：超參數搜尋失敗 {e}\n動作：中止流程", market, timeframe, "超參數搜尋")
        await 錯誤記錄與自動修復(f"錯誤碼E302：超參數搜尋失敗 {e}", "超參數搜尋錯誤", market, timeframe, "超參數搜尋")
        return None, 0.0

async def multi_market_hyperparameter_search(market_signal_mapping, 資產類型, n_trials=100):
    """多市場超參數搜尋"""
    try:
        start_time = time.time()
        results = []
        best_params_all = {}
        total_reward = 0.0
        tasks = []
        for market, timeframe in 市場清單:
            task = hyperparameter_search(
                market_signal_mapping=market_signal_mapping,
                資產類型=資產類型,
                market=market,
                timeframe=timeframe,
                n_trials=n_trials
            )
            tasks.append(task)
        
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        for (market, timeframe), result in zip(市場清單, results_list):
            if isinstance(result, Exception):
                logger.error(f"[{market}_{timeframe}] 單市場超參數搜尋失敗: {result}")
                await 發送錯誤訊息(f"錯誤碼E302：單市場超參數搜尋失敗 {result}", market, timeframe, "超參數搜尋")
                best_params_all[(market, timeframe)] = None
            else:
                best_params, reward = result
                if best_params:
                    best_params_all[(market, timeframe)] = best_params
                    total_reward += reward
                    results.append({
                        "市場": market,
                        "時間框架": timeframe,
                        "最佳參數": best_params,
                        "獎勵": reward
                    })

        # 生成多市場報表
        df = pd.DataFrame(results)
        if not df.empty:
            csv_path = SQLite資料夾.parent / "備份" / f"Multi_Market_Hyperparameter_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            await 發送通知(f"【執行通知】生成多市場超參數搜尋報表 {csv_path}", "多市場", "多框架", "超參數搜尋")

        # 效率報告
        elapsed_time = time.time() - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】多市場超參數搜尋耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await 發送通知(efficiency_report, "多市場", "多框架", "超參數搜尋")

        return best_params_all, total_reward / len(results) if results else 0.0
    except Exception as e:
        logger.error(f"[多市場_多框架] 多市場超參數搜尋失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E302：多市場超參數搜尋失敗 {e}", "多市場", "多框架", "超參數搜尋")
        await 錯誤記錄與自動修復(f"錯誤碼E302：多市場超參數搜尋失敗 {e}", "多市場搜尋錯誤", "多市場", "多框架", "超參數搜尋")
        return {}, 0.0