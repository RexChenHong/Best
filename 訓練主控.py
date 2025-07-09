import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import asyncio
import logging
import datetime
import sqlite3
import uuid
import psutil
import shutil
from pathlib import Path
from collections import deque
from logging.handlers import TimedRotatingFileHandler
from 設定檔 import 訓練設備, SQLite資料夾, 市場清單, 訓練參數
from 資料預處理模組 import load_and_preprocess_data, 取得Kfold資料集
from 信號生成模組 import SignalModel
from 交易環境模組 import multi_market_trading_env
from 檢查點模組 import 儲存檢查點
from 超參數搜尋模組 import hyperparameter_search
from 推播通知模組 import 發送訓練進度檢查點, 發送錯誤訊息, 發送通知
from 工具模組 import 錯誤記錄與自動修復, 監控硬體狀態並降級
from 測試網模組 import single_market_testnet_trading
from 獎勵計算模組 import calculate_multi_market_reward
from GUI介面模組 import TradingGUI

# 配置日誌
log_dir = SQLite資料夾.parent / "日誌"
log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("訓練主控")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
file_handler = TimedRotatingFileHandler(
    filename=log_dir / "train_control_logs",
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
    "E401": "風險指標異常",
    "E402": "訓練流程失敗",
    "E403": "硬體資源超限",
    "E404": "資料加載失敗"
}

# 訓練緩衝區
training_buffer = deque(maxlen=100)
checkpoint = 0

async def validate_training_input(model_list, optimizer_list, data_loader_dict, params_list):
    """驗證訓練輸入"""
    try:
        if not model_list or not all(isinstance(model, nn.Module) for model in model_list):
            logger.error("無效模型列表")
            await 發送錯誤訊息("錯誤碼E404：無效模型列表", "多市場", "多框架", "訓練主控")
            return False
        if not optimizer_list or not all(isinstance(optimizer, optim.Optimizer) for optimizer in optimizer_list):
            logger.error("無效優化器列表")
            await 發送錯誤訊息("錯誤碼E404：無效優化器列表", "多市場", "多框架", "訓練主控")
            return False
        if not data_loader_dict or not all(isinstance(loader[0], torch.utils.data.DataLoader) for loader in data_loader_dict.values()):
            logger.error("無效數據載入器字典")
            await 發送錯誤訊息("錯誤碼E404：無效數據載入器字典", "多市場", "多框架", "訓練主控")
            return False
        return True
    except Exception as e:
        logger.error(f"訓練輸入驗證失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E404：訓練輸入驗證失敗 {e}", "多市場", "多框架", "訓練主控")
        await 錯誤記錄與自動修復(f"錯誤碼E404：訓練輸入驗證失敗 {e}", "輸入驗證錯誤", "多市場", "多框架", "訓練主控")
        return False

async def manage_resources():
    """管理資源並清理快取"""
    try:
        disk_usage = shutil.disk_usage(SQLite資料夾.parent)
        disk_free_ratio = disk_usage.free / disk_usage.total
        if disk_free_ratio < 0.1:
            logger.warning("磁碟空間不足，清理7天前快取")
            cache_dir = SQLite資料夾.parent / "快取"
            for file in cache_dir.glob("*.npz"):
                if (datetime.datetime.now() - datetime.datetime.fromtimestamp(file.stat().st_mtime)).days > 7:
                    file.unlink()
            await 發送通知(f"【通知】磁碟空間不足，清理7天前快取\n剩餘空間：{disk_free_ratio:.2%}", "多市場", "多框架", "訓練主控")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100
            cpu_ok = psutil.cpu_percent() < 資源閾值["CPU使用率"] * 100
            if gpu_util > 80 or not cpu_ok:
                gpu_switch_count = getattr(manage_resources, 'switch_count', 0) + 1
                setattr(manage_resources, 'switch_count', gpu_switch_count)
                logger.warning(f"資源超限（GPU記憶體 {gpu_util:.2f}% 或 CPU使用率），切換至CPU，當前切換次數: {gpu_switch_count}")
                await 發送通知(f"【通知】資源超限（GPU記憶體 {gpu_util:.2f}% 或 CPU使用率），切換至CPU，當前切換次數: {gpu_switch_count}", "多市場", "多框架", "訓練主控")
                return False
        return True
    except Exception as e:
        logger.error(f"資源管理失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E403：資源管理失敗 {e}", "多市場", "多框架", "訓練主控")
        await 錯誤記錄與自動修復(f"錯誤碼E403：資源管理失敗 {e}", "資源管理錯誤", "多市場", "多框架", "訓練主控")
        return False

async def check_risk_metrics(result, market, timeframe, initial_funds=1000.0):
    """檢查風險指標"""
    try:
        max_drawdown = result.get("最大回撤", 0.0)
        returns = [(record["損益"] / initial_funds) for record in result.get("交易記錄", []) if "損益" in record]
        single_loss = min(returns) if returns else 0.0
        sharpe_ratio = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)) if returns and np.std(returns) > 0 else 0.0
        stability = np.std(returns) if returns else 0.0

        if max_drawdown > 0.25 or single_loss < -0.03:
            logger.warning(f"[{market}_{timeframe}] 風險指標異常: 最大回撤={max_drawdown:.2%}, 單筆損失={single_loss:.4f}")
            await 發送錯誤訊息(f"錯誤碼E401：風險指標異常\n最大回撤={max_drawdown:.2%}, 單筆損失={single_loss:.4f}", market, timeframe, "訓練主控")
            await 錯誤記錄與自動修復(f"錯誤碼E401：風險指標異常", "風險指標錯誤", market, timeframe, "訓練主控")
            return False, sharpe_ratio, stability
        return True, sharpe_ratio, stability
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 風險指標檢查失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E401：風險指標檢查失敗 {e}", market, timeframe, "訓練主控")
        await 錯誤記錄與自動修復(f"錯誤碼E401：風險指標檢查失敗 {e}", "風險指標錯誤", market, timeframe, "訓練主控")
        return False, 0.0, 0.0

async def train_single_epoch(model, optimizer, train_loader, loss_fn, params, market, timeframe):
    """單市場單週期訓練"""
    try:
        model.train()
        total_loss = 0.0
        batch_count = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(訓練設備), target.to(訓練設備)
            optimizer.zero_grad()
            output = model(data)
            if output is None:
                logger.error(f"[{market}_{timeframe}] 模型輸出無效")
                await 發送錯誤訊息(f"錯誤碼E402：模型輸出無效", market, timeframe, "訓練主控")
                return None
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        return avg_loss
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 單週期訓練失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E402：單週期訓練失敗 {e}", market, timeframe, "訓練主控")
        await 錯誤記錄與自動修復(f"錯誤碼E402：單週期訓練失敗 {e}", "訓練錯誤", market, timeframe, "訓練主控")
        return None

async def validate_single_epoch(model, val_loader, loss_fn, market, timeframe):
    """單市場單週期驗證"""
    try:
        model.eval()
        total_loss = 0.0
        batch_count = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(訓練設備), target.to(訓練設備)
                output = model(data)
                if output is None:
                    logger.error(f"[{market}_{timeframe}] 模型驗證輸出無效")
                    await 發送錯誤訊息(f"錯誤碼E402：模型驗證輸出無效", market, timeframe, "訓練主控")
                    return None
                loss = loss_fn(output, target)
                total_loss += loss.item()
                batch_count += 1
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        return avg_loss
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 單週期驗證失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E402：單週期驗證失敗 {e}", market, timeframe, "訓練主控")
        await 錯誤記錄與自動修復(f"錯誤碼E402：單週期驗證失敗 {e}", "驗證錯誤", market, timeframe, "訓練主控")
        return None

async def infinite_loop_control(market_signal_mapping, 資產類型, max_iterations=10, retry_count=0):
    """無限輪結構控制"""
    try:
        iteration = 0
        best_params_all = {}
        total_reward = 0.0
        initial_funds = 1000.0
        conn = sqlite3.connect(SQLite資料夾 / "訓練記錄.db")
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS 訓練記錄 (
                UUID TEXT PRIMARY KEY,
                市場 TEXT,
                週期 TEXT,
                迭代次數 INTEGER,
                輪次 INTEGER,
                Sharpe REAL,
                最大回撤 REAL,
                穩定性 REAL,
                獎勵 REAL,
                參數 TEXT,
                記錄時間 TEXT
            )
        """)
        conn.commit()
        conn.close()

        # 初始化模型與優化器
        model_list = [SignalModel(input_size=10, hidden_size=128, dropout=0.2, model_type=model_type)
                      for model_type in ["MLP", "CNN", "LSTM", "Transformer"]]
        optimizer_list = [optim.Adam(model.parameters(), lr=訓練參數["學習率"]["值"]) for model in model_list]

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"[多市場_多框架] 開始第 {iteration}/{max_iterations} 次迭代")
            results = {}
            iteration_reward = 0.0

            # 資料加載與KFold
            data_loader_dict = {}
            for market, timeframe in 市場清單:
                data = await load_and_preprocess_data(market, timeframe)
                if data:
                    data_loader_dict[(market, timeframe)] = 取得Kfold資料集(data, K=5, batch_size=32)[0]
                else:
                    logger.warning(f"[{market}_{timeframe}] 資料加載失敗")
                    continue

            if not data_loader_dict:
                logger.error("[多市場_多框架] 無有效數據載入器")
                return None, 0.0

            # 資源管理
            if not await manage_resources():
                logger.error("[多市場_多框架] 資源管理失敗")
                return None, 0.0

            # 超參數搜尋
            best_params, reward = await hyperparameter_search(
                market_signal_mapping=market_signal_mapping,
                資產類型=資產類型,
                market="BTCUSDT",
                timeframe="15m",
                n_trials=100
            )
            if best_params is None:
                continue
            best_params_all = {(market, timeframe): best_params for market, timeframe in 市場清單}

            # 訓練循環
            loss_fn = nn.CrossEntropyLoss()
            best_val_loss = float('inf')
            training_data = []
            for epoch in range(1, 11):  # 假設10個epoch
                for (market, timeframe), (train_loader, val_loader) in data_loader_dict.items():
                    model = next((m for m in model_list if m.model_type == best_params.get("model_type", "MLP")), model_list[0])
                    optimizer = optimizer_list[model_list.index(model)]
                    params = best_params_all.get((market, timeframe), {})
                    batch_size = params.get("batch_size", 32)

                    # 訓練
                    train_loss = await train_single_epoch(model, optimizer, train_loader, loss_fn, params, market, timeframe)
                    if train_loss is None:
                        continue

                    # 驗證
                    val_loss = await validate_single_epoch(model, val_loader, loss_fn, market, timeframe)
                    if val_loss is None:
                        continue

                    # 記錄訓練進度
                    training_data.append({
                        "epoch": epoch,
                        "market": market,
                        "timeframe": timeframe,
                        "train_loss": train_loss,
                        "val_loss": val_loss
                    })
                    training_buffer.append({
                        "epoch": epoch,
                        "market": market,
                        "timeframe": timeframe,
                        "val_loss": val_loss
                    })

                    # 推播進度
                    await 發送訓練進度檢查點({
                        "epoch": epoch,
                        "epochs": 10,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "market": market,
                        "interval": timeframe,
                        "model_type": model.model_type
                    })

                    # 儲存檢查點
                    trading_result, _ = await multi_market_trading_env(
                        market_signal_mapping=market_signal_mapping,
                        資產類型=資產類型,
                        params_list={(market, timeframe): params}
                    )
                    if trading_result:
                        checkpoint = epoch
                        await 儲存檢查點(
                            model_list=[model],
                            optimizer_list=[optimizer],
                            period=epoch,
                            market=market,
                            timeframe=timeframe,
                            mode="訓練",
                            environment_state=trading_result.get((market, timeframe), {}),
                            reward=0.0
                        )

                    # 更新最佳參數
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params_all[(market, timeframe)] = params

            # 測試網驗證（僅BTCUSDT_15M）
            for (market, timeframe) in 市場清單:
                if market == "BTCUSDT" and timeframe == "15m":
                    model = next((m for m in model_list if m.model_type == best_params.get("model_type", "MLP")), model_list[0])
                    optimizer = optimizer_list[model_list.index(model)]
                    signals = market_signal_mapping.get((market, timeframe), {"信號": [0.0]})["信號"]
                    result, testnet_reward = await single_market_testnet_trading(
                        params=best_params_all.get((market, timeframe), {}),
                        market=market,
                        timeframe=timeframe,
                        signal=signals[-1] if signals else 0.0,
                        price=None,
                        model=model,
                        optimizer=optimizer
                    )
                    if result:
                        results[(market, timeframe)] = result
                        iteration_reward += testnet_reward

                        # 風險指標檢查
                        is_valid, sharpe_ratio, stability = await check_risk_metrics(result, market, timeframe, initial_funds)
                        if not is_valid:
                            continue

                        # 儲存迭代結果
                        conn = sqlite3.connect(SQLite資料夾 / "訓練記錄.db")
                        c = conn.cursor()
                        c.execute("INSERT INTO 訓練記錄 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                  (str(uuid.uuid4()), market, timeframe, iteration, iteration // 10 + 1,
                                   sharpe_ratio, result["最大回撤"], stability, testnet_reward,
                                   str(best_params_all[(market, timeframe)]), datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                        conn.commit()
                        conn.close()

            # 計算多市場獎勵
            total_reward += (await calculate_multi_market_reward(results, best_params_all))[0]

            # 每輪（10次迭代）檢查終止條件
            if iteration % 10 == 0:
                conn = sqlite3.connect(SQLite資料夾 / "訓練記錄.db")
                df = pd.read_sql_query(
                    "SELECT Sharpe, 最大回撤, 穩定性 FROM 訓練記錄 WHERE 市場 = 'BTCUSDT' AND 週期 = '15m' ORDER BY 記錄時間 DESC LIMIT 10",
                    conn)
                conn.close()
                if not df.empty:
                    sharpe_mean = df["Sharpe"].mean()
                    max_drawdown = df["最大回撤"].max()
                    stability = df["穩定性"].mean()
                    if sharpe_mean >= 1.5 and max_drawdown <= 0.25 and stability < 0.1:
                        logger.info(f"[多市場_多框架] 滿足終止條件: Sharpe={sharpe_mean:.2f}, 最大回撤={max_drawdown:.2%}, 穩定性={stability:.4f}")
                        await 發送通知(
                            f"【輪次完成】\n輪次：第 {iteration // 10}\n市場：BTCUSDT_15m\nSharpe：{sharpe_mean:.2f}\n最大回撤：{max_drawdown:.2%}\n穩定性：{stability:.4f}\n時間：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            "多市場", "多框架", "訓練主控"
                        )
                        return best_params_all, total_reward
                await 發送通知(
                    f"【輪次完成】\n輪次：第 {iteration // 10}\n市場：BTCUSDT_15m\nSharpe：{sharpe_mean:.2f}\n最大回撤：{max_drawdown:.2%}\n穩定性：{stability:.4f}\n時間：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "多市場", "多框架", "訓練主控"
                )

        return best_params_all, total_reward
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[多市場_多框架] 無限輪訓練失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E402：無限輪訓練失敗 重試{retry_count + 1}/5: {e}", "訓練流程錯誤", "多市場", "多框架", "訓練主控")
            await asyncio.sleep(5)
            return await infinite_loop_control(market_signal_mapping, 資產類型, max_iterations, retry_count + 1)
        logger.error(f"[多市場_多框架] 無限輪訓練失敗，重試 5 次無效: {e}")
        await 發送錯誤訊息(f"錯誤碼E402：無限輪訓練失敗 {e}\n動作：中止流程", "多市場", "多框架", "訓練主控")
        await 錯誤記錄與自動修復(f"錯誤碼E402：無限輪訓練失敗 {e}", "訓練流程錯誤", "多市場", "多框架", "訓練主控")
        return None, 0.0

async def train_control(market_signal_mapping, 資產類型="虛擬貨幣", retry_count=0):
    """訓練主控，協調閉環流程"""
    try:
        start_time = time.time()
        if not await validate_training_input([], [], {}, {}):  # 初始化時驗證
            return None, 0.0

        # 初始化 GUI
        gui = TradingGUI()
        gui.start_gui()

        # 資源管理
        if not await manage_resources():
            return None, 0.0

        # 資料加載與KFold
        data_loader_dict = {}
        for market, timeframe in 市場清單:
            data = await load_and_preprocess_data(market, timeframe)
            if data:
                data_loader_dict[(market, timeframe)] = 取得Kfold資料集(data, K=5, batch_size=32)[0]
            else:
                logger.warning(f"[{market}_{timeframe}] 資料加載失敗")
                await 發送錯誤訊息(f"錯誤碼E404：資料加載失敗", market, timeframe, "訓練主控")
                continue

        if not data_loader_dict:
            logger.error("[多市場_多框架] 無有效數據載入器")
            await 發送錯誤訊息(f"錯誤碼E404：無有效數據載入器", "多市場", "多框架", "訓練主控")
            return None, 0.0

        # 初始化模型與優化器
        model_list = [SignalModel(input_size=10, hidden_size=128, dropout=0.2, model_type=model_type)
                      for model_type in ["MLP", "CNN", "LSTM", "Transformer"]]
        optimizer_list = [optim.Adam(model.parameters(), lr=訓練參數["學習率"]["值"]) for model in model_list]

        # 超參數搜尋
        best_params_all = {}
        for market, timeframe in 市場清單:
            best_params, reward = await hyperparameter_search(
                market_signal_mapping=market_signal_mapping,
                資產類型=資產類型,
                market=market,
                timeframe=timeframe,
                n_trials=100
            )
            if best_params:
                best_params_all[(market, timeframe)] = best_params

        # 自定義訓練策略
        custom_strategies = 訓練參數.get("custom_strategies", {})
        for strategy_name, strategy_config in custom_strategies.items():
            if strategy_config.get("enabled", False):
                best_params_all.update(eval(strategy_config["function"])(market_signal_mapping, 資產類型))

        # 無限輪訓練
        best_params, total_reward = await infinite_loop_control(market_signal_mapping, 資產類型)

        if best_params is None:
            logger.error("[多市場_多框架] 訓練失敗，無有效參數")
            return None, 0.0

        # 最終測試網驗證
        results = {}
        for (market, timeframe) in 市場清單:
            model = next((m for m in model_list if m.model_type == best_params.get((market, timeframe), {}).get("model_type", "MLP")), model_list[0])
            optimizer = optimizer_list[model_list.index(model)]
            signals = market_signal_mapping.get((market, timeframe), {"信號": [0.0]})["信號"]
            result, testnet_reward = await single_market_testnet_trading(
                params=best_params.get((market, timeframe), {}),
                market=market,
                timeframe=timeframe,
                signal=signals[-1] if signals else 0.0,
                price=None,
                model=model,
                optimizer=optimizer
            )
            if result:
                results[(market, timeframe)] = result
                is_valid, sharpe_ratio, stability = await check_risk_metrics(result, market, timeframe)
                if is_valid:
                    conn = sqlite3.connect(SQLite資料夾 / "訓練記錄.db")
                    c = conn.cursor()
                    c.execute("INSERT INTO 訓練記錄 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                              (str(uuid.uuid4()), market, timeframe, 0, 0, sharpe_ratio, result["最大回撤"],
                               stability, testnet_reward, str(best_params.get((market, timeframe), {})),
                               datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                    conn.commit()
                    conn.close()

        # 生成最終報表
        df = pd.DataFrame(training_buffer)
        if not df.empty:
            csv_path = SQLite資料夾.parent / "備份" / f"Training_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            await 發送通知(f"【執行通知】生成訓練報表 {csv_path}", "多市場", "多框架", "訓練主控")

        # 效率報告
        elapsed_time = time.time() - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】訓練主控耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await 發送通知(efficiency_report, "多市場", "多框架", "訓練主控")

        return best_params, total_reward
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[多市場_多框架] 訓練主控失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E402：訓練主控失敗 重試{retry_count + 1}/5: {e}", "訓練流程錯誤", "多市場", "多框架", "訓練主控")
            await asyncio.sleep(5)
            return await train_control(market_signal_mapping, 資產類型, retry_count + 1)
        logger.error(f"[多市場_多框架] 訓練主控失敗，重試 5 次無效: {e}")
        await 發送錯誤訊息(f"錯誤碼E402：訓練主控失敗 {e}\n動作：中止流程", "多市場", "多框架", "訓練主控")
        await 錯誤記錄與自動修復(f"錯誤碼E402：訓練主控失敗 {e}", "訓練流程錯誤", "多市場", "多框架", "訓練主控")
        return None, 0.0

async def main_training_controller():
    """主訓練控制器"""
    try:
        start_time = time.time()
        # 初始化 GUI
        gui = TradingGUI()
        gui.start_gui()

        # 初始化模型與優化器
        model_list = [SignalModel(input_size=10, hidden_size=128, dropout=0.2, model_type=model_type)
                      for model_type in ["MLP", "CNN", "LSTM", "Transformer"]]
        optimizer_list = [optim.Adam(model.parameters(), lr=訓練參數["學習率"]["值"]) for model in model_list]

        # 準備數據與信號
        market_signal_mapping = {}
        for market, timeframe in 市場清單:
            data = await load_and_preprocess_data(market, timeframe)
            if data:
                signals = await SignalModel.generate_signals(market, timeframe)
                prices = await get_real_time_price(market, timeframe)
                market_signal_mapping[(market, timeframe)] = {"信號": signals, "價格": prices}

        # 執行訓練
        best_params, total_reward = await train_control(market_signal_mapping, 資產類型="虛擬貨幣")

        # 訓練結束報告
        elapsed_time = time.time() - start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】總訓練耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await 發送通知(efficiency_report, "多市場", "多框架", "訓練主控")

    except Exception as e:
        logger.error(f"[多市場_多框架] 主訓練控制器失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E402：主訓練控制器失敗 {e}", "多市場", "多框架", "訓練主控")
        await 錯誤記錄與自動修復(f"錯誤碼E402：主訓練控制器失敗 {e}", "主訓練錯誤", "多市場", "多框架", "訓練主控")

async def get_real_time_price(market, timeframe):
    """從測試網模組獲取真實價格"""
    try:
        from 測試網模組 import get_mainnet_price
        price, _ = await get_mainnet_price(market)
        if price is None:
            logger.error(f"[{market}_{timeframe}] 無法獲取真實價格")
            await 發送錯誤訊息(f"錯誤碼E404：無法獲取真實價格", market, timeframe, "訓練主控")
            return [1000.0]
        return [price]
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 獲取真實價格失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E404：獲取真實價格失敗 {e}", market, timeframe, "訓練主控")
        return [1000.0]

if __name__ == "__main__":
    asyncio.run(main_training_controller())