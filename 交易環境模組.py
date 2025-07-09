import numpy as np
import torch
import time
import datetime
import logging
import uuid
import asyncio
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from 設定檔 import (
    訓練設備, 快取資料夾, 槓桿比例, 手續費率, 最小下單手數, 點差, 點值, 訓練參數, 市場清單, 資源閾值
)
from 工具模組 import 檢查交易時間, 記錄持倉狀態, 錯誤記錄與自動修復, 監控硬體狀態並降級
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
    "E102": "同K棒多次反手",
    "E103": "硬體資源超限",
    "E104": "風險參數優化失敗",
    "E105": "爆倉",
    "E106": "長期穩定性異常"
}

def validate_input(信號, 價格, 資產類型, 市場, 時間框架):
    """驗證輸入信號、價格與參數，無效時回退至0.0"""
    try:
        if not isinstance(信號, (list, np.ndarray, torch.Tensor)) or not all(s in [1.0, -1.0, 0.0] for s in np.array(信號).flatten()):
            logger.error(f"[{市場}_{時間框架}] 無效信號: {信號}")
            asyncio.run(發送錯誤訊息(f"錯誤碼E101：無效信號 {信號}", 市場, 時間框架, "模擬交易"))
            return False
        if not isinstance(價格, (list, np.ndarray, torch.Tensor)) or not all(p > 0 for p in np.array(價格).flatten()):
            logger.error(f"[{市場}_{時間框架}] 無效價格: {價格}")
            asyncio.run(發送錯誤訊息(f"錯誤碼E101：無效價格 {價格}", 市場, 時間框架, "模擬交易"))
            return False
        if 資產類型 not in ["CFD", "虛擬貨幣"]:
            logger.error(f"[{市場}_{時間框架}] 無效資產類型: {資產類型}")
            asyncio.run(發送錯誤訊息(f"錯誤碼E101：無效資產類型 {資產類型}", 市場, 時間框架, "模擬交易"))
            return False
        if 市場 not in 槓桿比例:
            logger.error(f"[{市場}_{時間框架}] 無效市場: {市場}")
            asyncio.run(發送錯誤訊息(f"錯誤碼E101：無效市場 {市場}", 市場, 時間框架, "模擬交易"))
            return False
        return True
    except Exception as e:
        logger.error(f"[{市場}_{時間框架}] 輸入驗證失敗: {e}")
        asyncio.run(發送錯誤訊息(f"錯誤碼E101：輸入驗證失敗 {e}", 市場, 時間框架, "模擬交易"))
        return False

async def check_risk_and_blowout(資金, 持倉數量, 未實現盈虧, 平均入場價格, 槓桿, 初始資金, 單筆損失限制, 單日損失限制, 市場, 時間框架):
    """檢查風險與爆倉條件"""
    try:
        巔峰資金 = 資金.max()
        最大回撤 = (巔峰資金 - 資金) / 巔峰資金.clamp(min=1e-10)
        維持率 = torch.where(
            持倉數量 != 0,
            (資金 + 未實現盈虧) / (torch.abs(持倉數量) * 平均入場價格 / 槓桿) * 100,
            torch.tensor(100.0, device=訓練設備)
        )
        if torch.any(資金 <= 0):
            logger.error(f"[{市場}_{時間框架}] 爆倉: 資金 {資金.min().item():.2f}")
            await 發送錯誤訊息(f"錯誤碼E105：爆倉，資金 {資金.min().item():.2f}\n動作：立即停止流程", 市場, 時間框架, "模擬交易")
            return True, 最大回撤, 維持率
        if torch.any(最大回撤 >= 0.25):
            logger.error(f"[{市場}_{時間框架}] 最大回撤超限: {最大回撤.max().item():.2%}")
            await 發送錯誤訊息(f"錯誤碼E101：最大回撤超限 {最大回撤.max().item():.2%}", 市場, 時間框架, "模擬交易")
            return True, 最大回撤, 維持率
        if torch.any(維持率 < (50 if 資產類型 == "CFD" else 105)):
            logger.error(f"[{市場}_{時間框架}] 維持率低於 {(50 if 資產類型 == 'CFD' else 105)}%")
            await 發送錯誤訊息(f"錯誤碼E101：維持率低於 {(50 if 資產類型 == 'CFD' else 105)}%", 市場, 時間框架, "模擬交易")
            return True, 最大回撤, 維持率
        return False, 最大回撤, 維持率
    except Exception as e:
        logger.error(f"[{市場}_{時間框架}] 風險檢查失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E101：風險檢查失敗 {e}", 市場, 時間框架, "模擬交易")
        return True, torch.tensor(0.0, device=訓練設備), torch.tensor(100.0, device=訓練設備)

async def monitor_long_term_stability(市場, 時間框架, 資金, 交易記錄):
    """長期穩定性監控，每10輪檢查Sharpe、回撤、資金曲線標準差"""
    try:
        conn = sqlite3.connect(快取資料夾.parent / "SQLite" / "測試網交易結果.db")
        df = pd.read_sql_query(
            "SELECT 總收益率, Sharpe, 最大回撤 FROM 測試結果表 WHERE 市場 = ? AND 週期 = ? ORDER BY 記錄時間 DESC LIMIT 100",
            conn, params=(市場, 時間框架))
        conn.close()
        if df.empty:
            logger.warning(f"[{市場}_{時間框架}] 無長期穩定性數據")
            await 發送錯誤訊息(f"錯誤碼E106：無長期穩定性數據", 市場, 時間框架, "模擬交易")
            return False
        returns = df["總收益率"].values
        sharpe = df["Sharpe"].mean()
        max_drawdown = df["最大回撤"].max()
        std_returns = np.std(returns) if len(returns) > 1 else 0.0
        if sharpe < 1.5 or max_drawdown > 0.25 or std_returns > 0.1:
            logger.warning(f"[{市場}_{時間框架}] 長期穩定性異常: Sharpe={sharpe:.2f}, 最大回撤={max_drawdown:.2%}, 標準差={std_returns:.4f}")
            await 發送錯誤訊息(f"錯誤碼E106：長期穩定性異常\nSharpe={sharpe:.2f}, 最大回撤={max_drawdown:.2%}, 標準差={std_returns:.4f}", 市場, 時間框架, "模擬交易")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=range(len(returns)), y=returns, mode="lines+markers", name="報酬率"))
            fig.add_trace(go.Scatter(x=range(len(returns)), y=[np.mean(returns)] * len(returns), mode="lines", name="平均報酬率", line=dict(dash="dash")))
            fig.update_layout(title=f"{市場}_{時間框架} 長期績效趨勢", xaxis_title="時間", yaxis_title="報酬率 (%)", template="plotly_dark")
            plot_path = 快取資料夾.parent / "圖片" / f"{市場}_{時間框架}_長期趨勢_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            fig.write_html(plot_path)
            await 發送通知(f"【執行通知】生成長期趨勢圖表 {plot_path}", 市場, 時間框架, "模擬交易")
            return False
        return True
    except Exception as e:
        logger.error(f"[{市場}_{時間框架}] 長期穩定性監控失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E106：長期穩定性監控失敗 {e}", 市場, 時間框架, "模擬交易")
        return False

async def handle_strong_signal(signal, position, price, avg_price, leverage, fee_rate, batch_size, funds, trading_records, position_time, last_close_time, daily_loss, consecutive_losses, stop_loss_time, market, timeframe, asset_type, point_spread, single_loss_limit, daily_loss_limit, kbar_timestamp):
    """處理強信號（>0.9）即時反手邏輯，限制同K棒僅一次反手"""
    try:
        current_time = time.time()
        trade_mask = (signal.abs() > 0.9) & ((current_time - last_close_time >= 2.0) | (position == 0))
        kbar_mask = torch.ones_like(signal, dtype=torch.bool, device=訓練設備)
        if timeframe == "15m" and market == "BTCUSDT":
            kbar_mask = (kbar_timestamp // (15 * 60)) == (current_time // (15 * 60))
            if torch.any(trade_mask & kbar_mask & (last_close_time > kbar_timestamp - 15 * 60)):
                logger.error(f"[{market}_{timeframe}] 同K棒多次反手")
                await 發送錯誤訊息(f"錯誤碼E102：同K棒多次反手\n時間：{datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')}\n動作：限制後續反手", market, timeframe, "模擬交易")
                await 錯誤記錄與自動修復(f"錯誤碼E102：同K棒多次反手", "交易錯誤", market, timeframe, "模擬交易")
                return position, avg_price, funds, trading_records, position_time, last_close_time, daily_loss, consecutive_losses, stop_loss_time

        for idx in torch.where(trade_mask & kbar_mask)[0]:
            if position[idx] != 0:  # 平倉
                quantity = position[idx]
                close_price = price[idx] + point_spread if quantity > 0 else price[idx] - point_spread
                fee = abs(quantity) * close_price * fee_rate * 2
                profit = (close_price - avg_price[idx]) * quantity * leverage - fee if quantity > 0 else (avg_price[idx] - close_price) * quantity * leverage - fee
                if profit < 0 and abs(profit) >= 1000.0 * single_loss_limit:
                    logger.error(f"[{market}_{timeframe}] 單筆損失超限: {profit.item():.2f}")
                    await 發送錯誤訊息(f"錯誤碼E101：單筆損失超限: {profit.item():.2f}", market, timeframe, "模擬交易")
                    continue
                funds[idx] += profit
                if profit < 0:
                    daily_loss[idx] += abs(profit)
                    consecutive_losses[idx] += 1
                else:
                    consecutive_losses[idx] = 0
                trading_records[idx].append({
                    "id": str(uuid.uuid4()),
                    "市場": market,
                    "時間框架": timeframe,
                    "類型": "強信號平倉",
                    "價格": close_price.item(),
                    "數量": quantity.item(),
                    "損益": profit.item(),
                    "時間": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "持倉時間": position_time[idx].item()
                })
                await 發送下單通知("強信號平倉", market, timeframe, close_price.item(), quantity.item(), profit.item(), "模擬交易")
                position[idx] = 0
                avg_price[idx] = 0.0
                position_time[idx] = 0
                last_close_time[idx] = current_time

            # 開新倉
            quantity = 最小下單手數.get(market, 最小下單手數["default"]) if asset_type == "CFD" else (funds[idx] * 0.1 * leverage) / price[idx]
            quantity = quantity * signal[idx].sign()
            fee = abs(quantity) * (price[idx] + point_spread if quantity > 0 else price[idx] - point_spread) * fee_rate * 2
            funds[idx] -= fee
            position[idx] = quantity
            avg_price[idx] = price[idx]
            trading_records[idx].append({
                "id": str(uuid.uuid4()),
                "市場": market,
                "時間框架": timeframe,
                "類型": "強信號開倉",
                "價格": price[idx].item(),
                "數量": quantity.item(),
                "損益": 0.0,
                "時間": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "持倉時間": 0
            })
            await 發送下單通知("強信號開倉", market, timeframe, price[idx].item(), quantity.item(), funds[idx].item() * 0.1, "模擬交易")
        return position, avg_price, funds, trading_records, position_time, last_close_time, daily_loss, consecutive_losses, stop_loss_time
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 強信號處理失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E101：強信號處理失敗 {e}", market, timeframe, "模擬交易")
        return position, avg_price, funds, trading_records, position_time, last_close_time, daily_loss, consecutive_losses, stop_loss_time

async def single_market_trading_env(
    market_signal_mapping,
    資產類型,
    市場,
    時間框架,
    params,
    batch_size=32
):
    """模擬單一市場交易，獨立資金管理，動態優化風險參數"""
    try:
        # 驗證輸入
        signals = market_signal_mapping.get((市場, 時間框架), {}).get("信號", [])
        prices = market_signal_mapping.get((市場, 時間框架), {}).get("價格", [])
        if not validate_input(signals, prices, 資產類型, 市場, 時間框架):
            logger.warning(f"[{市場}_{時間框架}] 無效輸入，跳過交易")
            return None

        # 硬體監控與降級
        batch_size, _ = await 監控硬體狀態並降級(batch_size, 2)
        if batch_size < 8:
            logger.error(f"[{市場}_{時間框架}] 硬體資源超限，無法執行模擬")
            await 發送錯誤訊息(f"錯誤碼E103：硬體資源超限，批次大小 {batch_size}", 市場, 時間框架, "模擬交易")
            return None
        torch.cuda.empty_cache()

        # 固定槓桿與手續費
        leverage = 槓桿比例.get(市場, 槓桿比例["default"] if 資產類型 == "CFD" else 槓桿比例["default_crypto"])
        fee_rate = 手續費率["default"] if 資產類型 == "CFD" else 手續費率["default_crypto"]  # 固定手續費率
        point_spread = 點差.get(市場, 點差["default"])
        point_value = 點值.get(市場, 點值["default"])

        # 風險參數
        stop_loss = params.get("stop_loss", 0.02)
        take_profit = params.get("take_profit", 0.03)
        breakeven_trigger = params.get("breakeven_trigger", 0.01)
        trailing_stop = params.get("trailing_stop", 0.01)
        single_loss_limit = params.get("single_loss_limit", 0.02)
        daily_loss_limit = params.get("daily_loss_limit", 0.05)

        # 初始化環境，每市場獨立資金1000USDT/USD
        initial_funds = 1000.0
        funds = torch.full((batch_size,), initial_funds, dtype=torch.float32, device=訓練設備)
        position = torch.zeros(batch_size, dtype=torch.float32, device=訓練設備)
        avg_price = torch.zeros(batch_size, dtype=torch.float32, device=訓練設備)
        unrealized_pnl = torch.zeros(batch_size, dtype=torch.float32, device=訓練設備)
        position_time = torch.zeros(batch_size, dtype=torch.int32, device=訓練設備)
        last_close_time = torch.zeros(batch_size, dtype=torch.float32, device=訓練設備)
        peak_funds = funds.clone()
        max_drawdown = torch.zeros(batch_size, dtype=torch.float32, device=訓練設備)
        trading_records = [[] for _ in range(batch_size)]
        completed = torch.zeros(batch_size, dtype=torch.bool, device=訓練設備)
        daily_loss = torch.zeros(batch_size, dtype=torch.float32, device=訓練設備)
        consecutive_losses = torch.zeros(batch_size, dtype=torch.int32, device=訓練設備)
        stop_loss_time = torch.zeros(batch_size, dtype=torch.float32, device=訓練設備)

        # 交易時段檢查
        if 資產類型 == "CFD" and not await 檢查交易時間(市場):
            logger.warning(f"[{市場}_{時間框架}] 非交易時段，跳過模擬")
            return None

        # 使用 DataLoader 處理批量信號
        signal_tensor = torch.tensor(signals, dtype=torch.float32, device=訓練設備)
        price_tensor = torch.tensor(prices, dtype=torch.float32, device=訓練設備)
        kbar_timestamps = torch.tensor([time.time() - i * 900 for i in range(len(signals))], dtype=torch.float32, device=訓練設備)  # 假設15分鐘K棒
        dataset = TensorDataset(signal_tensor, price_tensor, kbar_timestamps)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for batch_idx, (batch_signals, batch_prices, batch_timestamps) in enumerate(dataloader):
            # 硬體檢查
            batch_size, _ = await 監控硬體狀態並降級(batch_size, 2)
            torch.cuda.empty_cache()

            # 交易時段檢查
            if 資產類型 == "CFD" and not await 檢查交易時間(市場):
                logger.info(f"[{市場}_{時間框架}] 非交易時段，跳過交易")
                continue

            current_price = batch_prices
            current_time = time.time()
            trade_mask = (current_time - last_close_time >= 2.0) | (position == 0)

            # 風險檢查
            is_blowout, max_drawdown, margin_ratio = await check_risk_and_blowout(
                funds, position, unrealized_pnl, avg_price, leverage, initial_funds,
                single_loss_limit, daily_loss_limit, 市場, 時間框架
            )
            if is_blowout:
                completed.fill_(True)
                break

            # 持倉更新
            hold_mask = position != 0
            unrealized_pnl[hold_mask] = (current_price[hold_mask] - avg_price[hold_mask]) * position[hold_mask] * leverage
            position_time[hold_mask] += 1
            await 發送倉位通知(市場, 時間框架, "模擬", position[0].item(), avg_price[0].item(), unrealized_pnl[0].item(), margin_ratio[0].item())

            # 強信號處理
            position, avg_price, funds, trading_records, position_time, last_close_time, daily_loss, consecutive_losses, stop_loss_time = await handle_strong_signal(
                batch_signals, position, current_price, avg_price, leverage, fee_rate, batch_size, funds,
                trading_records, position_time, last_close_time, daily_loss, consecutive_losses, stop_loss_time,
                市場, 時間框架, 資產類型, point_spread, single_loss_limit, daily_loss_limit, batch_timestamps
            )

            # T+1 市價交易邏輯
            for t in range(len(batch_signals) - 1):
                next_price = current_price[t + 1] if t + 1 < len(current_price) else current_price[t]

                # 買入處理
                buy_mask = (batch_signals[t] == 1.0) & trade_mask & (position <= 0)
                close_mask = buy_mask & (position < 0)
                for idx in torch.where(close_mask)[0]:
                    quantity = -position[idx]
                    trade_price = next_price + point_spread
                    fee = quantity * trade_price * fee_rate * 2
                    profit = (avg_price[idx] - trade_price) * quantity * leverage - fee
                    if profit < 0 and abs(profit) >= initial_funds * single_loss_limit:
                        logger.error(f"[{市場}_{時間框架}] 單筆損失超限: {profit.item():.2f}")
                        await 發送錯誤訊息(f"錯誤碼E101：單筆損失超限: {profit.item():.2f}", 市場, 時間框架, "模擬交易")
                        completed[idx] = True
                        continue
                    funds[idx] += profit
                    if profit < 0:
                        daily_loss[idx] += abs(profit)
                        consecutive_losses[idx] += 1
                    else:
                        consecutive_losses[idx] = 0
                    trading_records[idx].append({
                        "id": str(uuid.uuid4()),
                        "市場": 市場,
                        "時間框架": 時間框架,
                        "類型": "買入平倉",
                        "價格": trade_price.item(),
                        "數量": quantity.item(),
                        "損益": profit.item(),
                        "時間": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "持倉時間": position_time[idx].item()
                    })
                    await 發送下單通知("買入平倉", 市場, 時間框架, trade_price.item(), quantity.item(), profit.item(), "模擬交易")
                    position[idx] = 0
                    avg_price[idx] = 0.0
                    unrealized_pnl[idx] = 0.0
                    position_time[idx] = 0
                    last_close_time[idx] = current_time
                    await 記錄持倉狀態(市場, 時間框架, "模擬", 0.0, 0.0, 0.0)

                if buy_mask.any():
                    for idx in torch.where(buy_mask)[0]:
                        quantity = 最小下單手數.get(市場, 最小下單手數["default"]) if 資產類型 == "CFD" else (funds[idx] * 0.1 * leverage) / next_price
                        margin_required = quantity * next_price / leverage
                        margin_available = funds[idx] + unrealized_pnl[idx] - margin_required
                        if margin_available < margin_required:
                            logger.warning(f"[{市場}_{時間框架}] 可用保證金不足: {margin_available:.2f}")
                            await 發送錯誤訊息(f"錯誤碼E101：可用保證金不足: {margin_available:.2f}", 市場, 時間框架, "模擬交易")
                            continue
                        fee = quantity * (next_price + point_spread) * fee_rate * 2
                        funds[idx] -= fee
                        position[idx] = quantity
                        avg_price[idx] = next_price
                        trading_records[idx].append({
                            "id": str(uuid.uuid4()),
                            "市場": 市場,
                            "時間框架": 時間框架,
                            "類型": "買",
                            "價格": next_price.item(),
                            "數量": quantity.item(),
                            "損益": 0.0,
                            "時間": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "持倉時間": 0
                        })
                        await 記錄持倉狀態(市場, 時間框架, "模擬", float(quantity), float(next_price), 0.0)
                        await 發送下單通知("買", 市場, 時間框架, next_price.item(), quantity.item(), funds[idx].item() * 0.1, "模擬交易")

                # 賣出處理
                sell_mask = (batch_signals[t] == -1.0) & trade_mask & (position >= 0)
                close_mask = sell_mask & (position > 0)
                for idx in torch.where(close_mask)[0]:
                    quantity = position[idx]
                    trade_price = next_price - point_spread
                    fee = abs(quantity) * trade_price * fee_rate * 2
                    profit = (trade_price - avg_price[idx]) * quantity * leverage - fee
                    if profit < 0 and abs(profit) >= initial_funds * single_loss_limit:
                        logger.error(f"[{市場}_{時間框架}] 單筆損失超限: {profit.item():.2f}")
                        await 發送錯誤訊息(f"錯誤碼E101：單筆損失超限: {profit.item():.2f}", 市場, 時間框架, "模擬交易")
                        completed[idx] = True
                        continue
                    funds[idx] += profit
                    if profit < 0:
                        daily_loss[idx] += abs(profit)
                        consecutive_losses[idx] += 1
                    else:
                        consecutive_losses[idx] = 0
                    trading_records[idx].append({
                        "id": str(uuid.uuid4()),
                        "市場": 市場,
                        "時間框架": 時間框架,
                        "類型": "賣出平倉",
                        "價格": trade_price.item(),
                        "數量": quantity.item(),
                        "損益": profit.item(),
                        "時間": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "持倉時間": position_time[idx].item()
                    })
                    await 發送下單通知("賣出平倉", 市場, 時間框架, trade_price.item(), quantity.item(), profit.item(), "模擬交易")
                    position[idx] = 0
                    avg_price[idx] = 0.0
                    unrealized_pnl[idx] = 0.0
                    position_time[idx] = 0
                    last_close_time[idx] = current_time
                    await 記錄持倉狀態(市場, 時間框架, "模擬", 0.0, 0.0, 0.0)

                if sell_mask.any():
                    for idx in torch.where(sell_mask)[0]:
                        quantity = -最小下單手數.get(市場, 最小下單手數["default"]) if 資產類型 == "CFD" else -(funds[idx] * 0.1 * leverage) / next_price
                        margin_required = abs(quantity) * next_price / leverage
                        margin_available = funds[idx] + unrealized_pnl[idx] - margin_required
                        if margin_available < margin_required:
                            logger.warning(f"[{市場}_{時間框架}] 可用保證金不足: {margin_available:.2f}")
                            await 發送錯誤訊息(f"錯誤碼E101：可用保證金不足: {margin_available:.2f}", 市場, 時間框架, "模擬交易")
                            continue
                        fee = abs(quantity) * (next_price - point_spread) * fee_rate * 2
                        funds[idx] -= fee
                        position[idx] = quantity
                        avg_price[idx] = next_price
                        trading_records[idx].append({
                            "id": str(uuid.uuid4()),
                            "市場": 市場,
                            "時間框架": 時間框架,
                            "類型": "賣",
                            "價格": next_price.item(),
                            "數量": quantity.item(),
                            "損益": 0.0,
                            "時間": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "持倉時間": 0
                        })
                        await 記錄持倉狀態(市場, 時間框架, "模擬", float(quantity), float(next_price), 0.0)
                        await 發送下單通知("賣", 市場, 時間框架, next_price.item(), quantity.item(), funds[idx].item() * 0.1, "模擬交易")

                # 停損/停利檢查
                if hold_mask.any():
                    for idx in torch.where(hold_mask)[0]:
                        if position[idx] > 0:
                            if current_price[idx] <= avg_price[idx] * (1 - stop_loss):
                                trade_price = current_price[idx]
                                fee = position[idx] * trade_price * fee_rate * 2
                                profit = (trade_price - avg_price[idx]) * position[idx] * leverage - fee
                                if profit < 0 and abs(profit) >= initial_funds * single_loss_limit:
                                    logger.error(f"[{市場}_{時間框架}] 單筆損失超限: {profit.item():.2f}")
                                    await 發送錯誤訊息(f"錯誤碼E101：單筆損失超限: {profit.item():.2f}", 市場, 時間框架, "模擬交易")
                                    completed[idx] = True
                                    continue
                                funds[idx] += profit
                                if profit < 0:
                                    daily_loss[idx] += abs(profit)
                                    consecutive_losses[idx] += 1
                                else:
                                    consecutive_losses[idx] = 0
                                trading_records[idx].append({
                                    "id": str(uuid.uuid4()),
                                    "市場": 市場,
                                    "時間框架": 時間框架,
                                    "類型": "停損平倉",
                                    "價格": trade_price.item(),
                                    "數量": position[idx].item(),
                                    "損益": profit.item(),
                                    "時間": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    "持倉時間": position_time[idx].item()
                                })
                                await 發送下單通知("停損平倉", 市場, 時間框架, trade_price.item(), position[idx].item(), profit.item(), "模擬交易")
                                position[idx] = 0
                                avg_price[idx] = 0.0
                                unrealized_pnl[idx] = 0.0
                                position_time[idx] = 0
                                last_close_time[idx] = current_time
                            elif current_price[idx] >= avg_price[idx] * (1 + take_profit):
                                trade_price = current_price[idx]
                                fee = position[idx] * trade_price * fee_rate * 2
                                profit = (trade_price - avg_price[idx]) * position[idx] * leverage - fee
                                funds[idx] += profit
                                if profit < 0:
                                    daily_loss[idx] += abs(profit)
                                    consecutive_losses[idx] += 1
                                else:
                                    consecutive_losses[idx] = 0
                                trading_records[idx].append({
                                    "id": str(uuid.uuid4()),
                                    "市場": 市場,
                                    "時間框架": 時間框架,
                                    "類型": "停利平倉",
                                    "價格": trade_price.item(),
                                    "數量": position[idx].item(),
                                    "損益": profit.item(),
                                    "時間": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    "持倉時間": position_time[idx].item()
                                })
                                await 發送下單通知("停利平倉", 市場, 時間框架, trade_price.item(), position[idx].item(), profit.item(), "模擬交易")
                                position[idx] = 0
                                avg_price[idx] = 0.0
                                unrealized_pnl[idx] = 0.0
                                position_time[idx] = 0
                                last_close_time[idx] = current_time

                # 連續虧損與持倉時間檢查
                if torch.any(consecutive_losses >= 3):
                    if torch.any(current_time - stop_loss_time < 3600):
                        logger.warning(f"[{市場}_{時間框架}] 連續虧損超限，暫停交易 1 小時")
                        await 發送錯誤訊息(f"錯誤碼E101：連續虧損超限，暫停交易 1 小時", 市場, 時間框架, "模擬交易")
                        completed.fill_(True)
                        break
                    stop_loss_time = torch.where(consecutive_losses >= 3, torch.tensor(current_time, device=訓練設備), stop_loss_time)
                if torch.any(position_time >= 3600):
                    logger.warning(f"[{市場}_{時間框架}] 持倉時間超限，強制平倉")
                    await 發送錯誤訊息(f"錯誤碼E101：持倉時間超限，強制平倉", 市場, 時間框架, "模擬交易")
                    completed.fill_(True)
                    break

            if completed.all():
                break

        # 長期穩定性監控（每10輪）
        if len(trading_records[0]) > 0 and len(trading_records[0]) % 10 == 0:
            await monitor_long_term_stability(市場, 時間框架, funds[0].item(), trading_records[0])

        # 儲存結果
        result = {
            (市場, 時間框架): {
                "最終資金": funds.cpu().numpy()[0],
                "最大回撤": max_drawdown.cpu().numpy()[0],
                "交易記錄": trading_records[0],
                "連線狀態": "模擬模式",
                "維持率": margin_ratio.cpu().numpy()[0],
                "f1分數": sum(1 for r in trading_records[0] if r.get("損益", 0) > 0) / len(trading_records[0]) if trading_records[0] else 0.0,
                "穩定性": np.std([r.get("損益", 0) / initial_funds for r in trading_records[0]]) if trading_records[0] else 0.0
            }
        }
        conn = sqlite3.connect(快取資料夾.parent / "SQLite" / "測試網交易結果.db")
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS 測試結果表 (
                UUID TEXT PRIMARY KEY,
                市場 TEXT,
                週期 TEXT,
                模型 TEXT,
                使用參數_UUID TEXT,
                Sharpe REAL,
                最大回撤 REAL,
                總收益率 REAL,
                爆倉標記 BOOLEAN,
                交易次數 INTEGER,
                資金曲線檔案路徑 TEXT,
                記錄時間 TEXT
            )
        """)
        sharpe = (result[(市場, 時間框架)]["最終資金"] / initial_funds - 1) * 252 / (result[(市場, 時間框架)]["穩定性"] * np.sqrt(252)) if result[(市場, 時間框架)]["穩定性"] > 0 else 0.0
        c.execute("INSERT INTO 測試結果表 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (str(uuid.uuid4()), 市場, 時間框架, params.get("model_type", "MLP"), str(uuid.uuid4()),
                   sharpe, result[(市場, 時間框架)]["最大回撤"],
                   (result[(市場, 時間框架)]["最終資金"] - initial_funds) / initial_funds,
                   funds.cpu().numpy()[0] <= 0, len(trading_records[0]),
                   str(快取資料夾.parent / "圖片" / f"{市場}_{時間框架}_資金曲線_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"),
                   datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()

        # 生成資金曲線圖
        df = pd.DataFrame([{"資金": r.get("損益", 0) + initial_funds, "時間": r["時間"]} for r in trading_records[0]])
        if not df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["時間"], y=df["資金"], mode="lines+markers", name="資金曲線"))
            fig.update_layout(title=f"{市場}_{時間框架} 資金曲線", xaxis_title="時間", yaxis_title="資金", template="plotly_dark")
            plot_path = 快取資料夾.parent / "圖片" / f"{市場}_{時間框架}_資金曲線_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            fig.write_html(plot_path)
            await 發送通知(f"【執行通知】生成資金曲線圖表 {plot_path}", 市場, 時間框架, "模擬交易")

        # 效率報告
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        gpu_util = torch.cuda.memory_allocated(訓練設備) / torch.cuda.get_device_properties(訓練設備).total_memory * 100 if torch.cuda.is_available() else 0
        efficiency_report = (
            f"【效率報告】模擬交易耗時：{time.time() - current_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await 發送通知(efficiency_report, 市場, 時間框架, "模擬交易")

        return result

    except Exception as e:
        logger.error(f"[{市場}_{時間框架}] 模擬交易失敗: {e}")
        await 錯誤記錄與自動修復(
            錯誤訊息=f"錯誤碼E101：模擬交易失敗 {e}",
            錯誤類型="交易錯誤",
            市場=市場,
            時間框架=時間框架,
            模式="模擬"
        )
        await 發送錯誤訊息(f"錯誤碼E101：模擬交易失敗 {e}", 市場, 時間框架, "模擬交易")
        return None

async def multi_market_trading_env(market_signal_mapping, 資產類型, params_list):
    """多市場模擬交易環境"""
    try:
        results = {}
        total_reward = 0.0
        tasks = []
        for (market, timeframe) in 市場清單:
            task = single_market_trading_env(
                market_signal_mapping=market_signal_mapping,
                資產類型=資產類型,
                市場=market,
                時間框架=timeframe,
                params=params_list.get((market, timeframe), params_list.get(("default", "default"), {})),
                batch_size=params_list.get((market, timeframe), {}).get("batch_size", 32)
            )
            tasks.append(task)
        
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        for (market, timeframe), result in zip(市場清單, results_list):
            if isinstance(result, Exception):
                logger.error(f"[{market}_{timeframe}] 單市場交易失敗: {result}")
                await 發送錯誤訊息(f"錯誤碼E101：單市場交易失敗 {result}", market, timeframe, "模擬交易")
                results[(market, timeframe)] = None
            elif result and (market, timeframe) in result:
                results[(market, timeframe)] = result[(market, timeframe)]
                reward, _ = await calculate_multi_market_reward(
                    {(market, timeframe): result[(market, timeframe)]},
                    {(market, timeframe): params_list.get((market, timeframe), params_list.get(("default", "default"), {}))}
                )
                total_reward += reward
        return results, total_reward
    except Exception as e:
        logger.error(f"[多市場_多框架] 多市場模擬交易失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E101：多市場模擬交易失敗 {e}", "多市場", "多框架", "模擬交易")
        await 錯誤記錄與自動修復(f"錯誤碼E101：多市場模擬交易失敗 {e}", "多市場交易錯誤", "多市場", "多框架", "模擬交易")
        return None, 0.0