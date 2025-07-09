import torch
import numpy as np
import pandas as pd
import logging
from logging.handlers import QueueHandler, TimedRotatingFileHandler
from pathlib import Path
import datetime
import asyncio
import queue
import multiprocessing as mp
import hashlib
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from cryptography.fernet import Fernet
from 設定檔 import 訓練設備, 快取資料夾, 市場清單, 訓練參數
from 工具模組 import 錯誤記錄與自動修復, 監控硬體狀態並降級, validate_utility_input
from 推播通知模組 import 發送錯誤訊息, 發送通知
from 信號生成模組 import SignalModel
from 績效分析模組 import calculate_performance
from GUI介面模組 import TradingGUI

# 配置日誌
log_dir = 快取資料夾.parent / "日誌"
log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("批次切分與批次訓練")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
file_handler = TimedRotatingFileHandler(
    filename=log_dir / "batch_training_logs",
    when="midnight",
    backupCount=30,
    encoding='utf-8'
)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
log_queue = queue.Queue(-1)
queue_handler = QueueHandler(log_queue)
logger.addHandler(queue_handler)

# 錯誤碼定義
ERROR_CODES = {
    "E1201": "批次切分失敗",
    "E1202": "批次訓練失敗",
    "E1203": "硬體資源超限",
    "E1204": "數據加密失敗"
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

def log_listener(queue, log_file_handler, log_console_handler):
    """日誌監聽進程"""
    while True:
        try:
            record = queue.get()
            if record is None:
                break
            log_file_handler.handle(record)
            log_console_handler.handle(record)
        except Exception:
            break

def start_log_listener():
    """啟動日誌監聽"""
    listener = mp.Process(target=log_listener, args=(log_queue, file_handler, console_handler), daemon=True)
    listener.start()
    return listener

async def encrypt_dataset(dataset):
    """加密數據集"""
    try:
        key_path = 快取資料夾.parent / "secure_key.key"
        if not key_path.exists():
            key = Fernet.generate_key()
            with open(key_path, "wb") as key_file:
                key_file.write(key)
                
        with open(key_path, "rb") as key_file:
            key = key_file.read()
        cipher = Fernet(key)
        data_dict = {"data": dataset.tensors[0].numpy().tolist(), "targets": dataset.tensors[1].numpy().tolist()}
        encrypted_data = cipher.encrypt(json.dumps(data_dict, ensure_ascii=False).encode('utf-8'))
        
        logger.info("[資安] 數據集加密完成")
        await push_limiter.cache_message("[執行通知] 數據集加密完成", "多市場", "多框架", "批次訓練")
        await push_limiter.retry_cached_messages()
        
        with open(log_dir / f"資安事件_{datetime.datetime.now().strftime('%Y%m%d')}.log", "a", encoding='utf-8') as f:
            f.write(f"[資安事件] 數據集加密\n時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        return encrypted_data
    except Exception as e:
        logger.error(f"[資安] 數據集加密失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E1204：數據集加密失敗 {e}", "多市場", "多框架", "批次訓練")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E1204：數據集加密失敗 {e}", "數據加密錯誤", "多市場", "多框架", "批次訓練")
        return None

async def validate_batch_input(dataset, market, timeframe, params):
    """驗證批次輸入"""
    try:
        if not await validate_utility_input(market, timeframe, mode="批次訓練"):
            return False
        if not isinstance(dataset, torch.utils.data.Dataset):
            logger.error(f"[{market}_{timeframe}] 無效數據集格式")
            await push_limiter.cache_message(f"錯誤碼E1201：無效數據集格式", market, timeframe, "批次訓練")
            await push_limiter.retry_cached_messages()
            return False
        if not isinstance(params, dict):
            logger.error(f"[{market}_{timeframe}] 無效參數格式: {params}")
            await push_limiter.cache_message(f"錯誤碼E1201：無效參數格式 {params}", market, timeframe, "批次訓練")
            await push_limiter.retry_cached_messages()
            return False
        required_params = ["learning_rate", "dropout", "batch_size", "model_type", "window", "stop_loss", "take_profit"]
        if not all(key in params for key in required_params):
            logger.error(f"[{market}_{timeframe}] 缺少必要參數: {required_params}")
            await push_limiter.cache_message(f"錯誤碼E1201：缺少必要參數 {required_params}", market, timeframe, "批次訓練")
            await push_limiter.retry_cached_messages()
            return False
        ranges = {
            "learning_rate": (1e-5, 1e-2),
            "dropout": (0.1, 0.5),
            "batch_size": [32, 64, 128, 256],
            "window": (10, 200),
            "stop_loss": (0.01, 0.05),
            "take_profit": (0.02, 0.1)
        }
        for key, value in params.items():
            if key in ranges:
                if isinstance(ranges[key], list):
                    if value not in ranges[key]:
                        logger.error(f"[{market}_{timeframe}] 無效參數 {key}: {value}")
                        await push_limiter.cache_message(f"錯誤碼E1201：無效參數 {key}: {value}", market, timeframe, "批次訓練")
                        await push_limiter.retry_cached_messages()
                        return False
                else:
                    min_val, max_val = ranges[key]
                    if not (min_val <= value <= max_val):
                        logger.error(f"[{market}_{timeframe}] 無效參數 {key}: {value}")
                        await push_limiter.cache_message(f"錯誤碼E1201：無效參數 {key}: {value}", market, timeframe, "批次訓練")
                        await push_limiter.retry_cached_messages()
                        return False
        return True
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 批次輸入驗證失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E1201：批次輸入驗證失敗 {e}", market, timeframe, "批次訓練")
        await push_limiter.retry_cached_messages()
        return False

async def calculate_md5(data):
    """計算數據MD5"""
    try:
        md5_hash = hashlib.md5()
        data_np = data.tensors[0].numpy().tobytes()
        md5_hash.update(data_np)
        return md5_hash.hexdigest()
    except Exception as e:
        logger.error(f"MD5計算失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E1201：MD5計算失敗 {e}", "多市場", "多框架", "批次訓練")
        await push_limiter.retry_cached_messages()
        return None

async def prepare_batch_data(dataset, market, timeframe, window_size=60, batch_size=32, k_folds=5):
    """準備批次數據"""
    try:
        # MD5驗證
        data_md5 = await calculate_md5(dataset)
        if not data_md5:
            return []

        # 分割數據集
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

        # 儲存分割數據
        for subset, name in [(train_dataset, "train"), (val_dataset, "val"), (test_dataset, "test")]:
            cache_path = 快取資料夾 / f"{market}_{timeframe}_{name}.npz"
            np.savez_compressed(cache_path, data=subset.dataset.tensors[0][subset.indices].numpy(), targets=subset.dataset.tensors[1][subset.indices].numpy())
            await push_limiter.cache_message(f"【執行通知】數據分割已快取至 {cache_path}", market, timeframe, "批次訓練")
            await push_limiter.retry_cached_messages()

        # 寫入異動歷程
        change_log_path = 快取資料夾.parent / "dot" / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        change_log = {
            "UUID": str(uuid.uuid4()),
            "市場": market,
            "時間框架": timeframe,
            "異動前值": "N/A",
            "異動後值": f"分割數據: train={train_size}, val={val_size}, test={test_size}, MD5={data_md5}",
            "異動原因": "數據分割",
            "時間戳": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        change_df = pd.DataFrame([change_log])
        if change_log_path.exists():
            existing_df = pd.read_excel(change_log_path)
            change_df = pd.concat([existing_df, change_df], ignore_index=True)
        change_df.to_excel(change_log_path, index=False, engine='openpyxl')

        # KFold分割
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_data = []
        for train_idx, val_idx in kf.split(range(len(train_dataset))):
            train_subset = torch.utils.data.Subset(train_dataset, train_idx)
            val_subset = torch.utils.data.Subset(train_dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
            fold_data.append((train_loader, val_loader))
        return fold_data, test_dataset
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 批次數據準備失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E1201：批次數據準備失敗 {e}", market, timeframe, "批次訓練")
        await push_limiter.retry_cached_messages()
        return [], None

async def train_batch(model, optimizer, train_loader, loss_fn, market, timeframe, params, progress_callback=None):
    """單批次訓練"""
    try:
        model.train()
        total_loss = 0.0
        batch_count = 0
        total_batches = len(train_loader)
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(訓練設備), target.to(訓練設備)
            optimizer.zero_grad()
            output = model(data)
            if output is None:
                logger.error(f"[{market}_{timeframe}] 模型輸出無效")
                await push_limiter.cache_message(f"錯誤碼E1202：模型輸出無效", market, timeframe, "批次訓練")
                await push_limiter.retry_cached_messages()
                return None
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            if progress_callback and batch_count % (total_batches // 10) == 0:
                progress = (batch_count / total_batches) * 100
                await push_limiter.cache_message(f"【進度通知】{market}_{timeframe} 訓練進度: {progress:.1f}%", market, timeframe, "批次訓練")
                await push_limiter.retry_cached_messages()
                progress_callback(progress)
        return total_loss / batch_count if batch_count > 0 else 0.0
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 批次訓練失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E1202：批次訓練失敗 {e}", market, timeframe, "批次訓練")
        await push_limiter.retry_cached_messages()
        return None

async def validate_batch(model, val_loader, loss_fn, market, timeframe):
    """單批次驗證"""
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
                    await push_limiter.cache_message(f"錯誤碼E1202：模型驗證輸出無效", market, timeframe, "批次訓練")
                    await push_limiter.retry_cached_messages()
                    return None
                loss = loss_fn(output, target)
                total_loss += loss.item()
                batch_count += 1
        return total_loss / batch_count if batch_count > 0 else 0.0
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 批次驗證失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E1202：批次驗證失敗 {e}", market, timeframe, "批次訓練")
        await push_limiter.retry_cached_messages()
        return None

async def batch_training(dataset, market, timeframe, params, epochs=100, retry_count=0):
    """單市場批次訓練"""
    try:
        import time
        start_time = time.time()
        if not await validate_batch_input(dataset, market, timeframe, params):
            return None

        # 硬體監控
        batch_size, _ = await 監控硬體狀態並降級(params.get("batch_size", 32), 2)
        if batch_size < 8:
            logger.error(f"[{market}_{timeframe}] 硬體資源超限，批次大小 {batch_size}")
            await push_limiter.cache_message(f"錯誤碼E1203：硬體資源超限，批次大小 {batch_size}", market, timeframe, "批次訓練")
            await push_limiter.retry_cached_messages()
            return None
        torch.cuda.empty_cache()
        gpu_switch_count = 0

        # 加密數據集
        encrypted_dataset = await encrypt_dataset(dataset)
        if not encrypted_dataset:
            return None

        model = SignalModel(input_size=10, hidden_size=128, dropout=params.get("dropout", 0.2), model_type=params.get("model_type", "MLP"))
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get("learning_rate", 訓練參數["學習率"]["值"]))
        loss_fn = torch.nn.CrossEntropyLoss()
        fold_data, test_dataset = await prepare_batch_data(dataset, market, timeframe, window_size=params.get("window", 60), batch_size=batch_size, k_folds=5)

        training_results = []
        checkpoint_path = 快取資料夾 / f"{market}_{timeframe}_checkpoint.pt"
        for fold, (train_loader, val_loader) in enumerate(fold_data, 1):
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
            else:
                start_epoch = 1
            for epoch in range(start_epoch, epochs + 1):
                train_loss = await train_batch(model, optimizer, train_loader, loss_fn, market, timeframe, params,
                                              lambda p: TradingGUI.update_progress(p))
                if train_loss is None:
                    return None
                val_loss = await validate_batch(model, val_loader, loss_fn, market, timeframe)
                if val_loss is None:
                    return None

                # 模擬交易與績效分析
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                signals = [1.0 if torch.argmax(model(data.to(訓練設備)), dim=-1).item() == 0 else -1.0 if torch.argmax(model(data.to(訓練設備)), dim=-1).item() == 1 else 0.0
                           for data, _ in test_loader]
                trading_result = await 模擬交易(
                    信號=signals,
                    價格=[1000.0] * len(signals),
                    資產類型="虛擬貨幣",
                    市場=market,
                    時間框架=timeframe,
                    batch_size=batch_size,
                    停損=params.get("stop_loss", 0.02),
                    停利=params.get("take_profit", 0.03)
                )
                if trading_result and trading_result.get(market):
                    performance = await calculate_performance(trading_result[market], market, timeframe, params)
                    if performance:
                        training_results.append({
                            "fold": fold,
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "報酬率": performance["報酬率"],
                            "f1分數": performance["f1分數"],
                            "穩定性": performance["穩定性"],
                            "sharpe_ratio": performance["sharpe_ratio"]
                        })
                        if epoch % (epochs // 10) == 0:
                            await push_limiter.cache_message(
                                f"【進度通知】{market}_{timeframe} Fold {fold} Epoch {epoch}/{epochs}: "
                                f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Sharpe={performance['sharpe_ratio']:.2f}",
                                market, timeframe, "批次訓練")
                            await push_limiter.retry_cached_messages()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }, checkpoint_path)

        # 加密並儲存訓練結果
        encrypted_results = await encrypt_dataset(TensorDataset(
            torch.tensor([r["train_loss"] for r in training_results]),
            torch.tensor([r["val_loss"] for r in training_results])
        ))
        if not encrypted_results:
            return None
        async with aiosqlite.connect(快取資料夾.parent / "SQLite" / "batch_training_log.db") as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS 批次訓練記錄 (
                    id TEXT PRIMARY KEY,
                    市場 TEXT,
                    時間框架 TEXT,
                    fold INTEGER,
                    epoch INTEGER,
                    train_loss REAL,
                    val_loss REAL,
                    報酬率 REAL,
                    f1分數 REAL,
                    穩定性 REAL,
                    sharpe_ratio REAL,
                    時間 TEXT
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_市場_時間框架 ON 批次訓練記錄 (市場, 時間框架, 時間)")
            for result in training_results:
                await conn.execute("INSERT INTO 批次訓練記錄 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                 (str(uuid.uuid4()), market, timeframe, result["fold"], result["epoch"],
                                  result["train_loss"], result["val_loss"], result["報酬率"], result["f1分數"],
                                  result["穩定性"], result["sharpe_ratio"], datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            await conn.commit()

        # 快取訓練結果
        cache_path = 快取資料夾 / f"batch_cache_{market}_{timeframe}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez_compressed(cache_path, data=training_results)
        await push_limiter.cache_message(f"【執行通知】批次訓練數據已快取至 {cache_path}", market, timeframe, "批次訓練")
        await push_limiter.retry_cached_messages()

        # 生成報表
        df = pd.DataFrame(training_results)
        if not df.empty:
            csv_path = 快取資料夾.parent / "備份" / f"{market}_{timeframe}_Batch_Training_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            await push_limiter.cache_message(f"【執行通知】生成批次訓練報表 {csv_path}", market, timeframe, "批次訓練")
            await push_limiter.retry_cached_messages()

        # 寫入異動歷程
        change_log_path = 快取資料夾.parent / "dot" / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        change_log = {
            "UUID": str(uuid.uuid4()),
            "市場": market,
            "時間框架": timeframe,
            "異動前值": "N/A",
            "異動後值": f"訓練記錄數={len(training_results)}, 平均Sharpe={np.mean([r['sharpe_ratio'] for r in training_results]):.2f}",
            "異動原因": "批次訓練",
            "時間戳": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        change_df = pd.DataFrame([change_log])
        if change_log_path.exists():
            existing_df = pd.read_excel(change_log_path)
            change_df = pd.concat([existing_df, change_df], ignore_index=True)
        change_df.to_excel(change_log_path, index=False, engine='openpyxl')

        return training_results
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[{market}_{timeframe}] 批次訓練失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E1202：批次訓練失敗 重試{retry_count + 1}/5: {e}", "批次訓練錯誤", market, timeframe, "批次訓練")
            await asyncio.sleep(5)
            return await batch_training(dataset, market, timeframe, params, epochs, retry_count + 1)
        logger.error(f"[{market}_{timeframe}] 批次訓練失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E1202：批次訓練失敗 {e}", market, timeframe, "批次訓練")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E1202：批次訓練失敗 {e}", "批次訓練錯誤", market, timeframe, "批次訓練")
        return None

async def main_batch():
    """批次切分與訓練主函數"""
    try:
        import time
        import psutil
        start_time = time.time()
        listener = start_log_listener()
        gui = TradingGUI()
        gui_task = asyncio.create_task(asyncio.to_thread(gui.start_gui))

        dataset = TensorDataset(torch.randn(1000, 60, 10), torch.randint(0, 3, (1000,)))
        params = {
            "learning_rate": 訓練參數["學習率"]["值"],
            "dropout": 0.2,
            "batch_size": 32,
            "model_type": "MLP",
            "window": 60,
            "stop_loss": 訓練參數["stop_loss"]["值"],
            "take_profit": 訓練參數["take_profit"]["值"]
        }

        tasks = []
        weighted_sharpe = 0.0
        valid_markets = 0
        for market, timeframe in 市場清單:
            weight = 0.5 if market == "BTCUSDT" and timeframe == "15m" else 0.5 / (len(市場清單) - 1)
            result = await batch_training(dataset, market, timeframe, params, epochs=10)
            if result:
                avg_sharpe = np.mean([r["sharpe_ratio"] for r in result])
                weighted_sharpe += avg_sharpe * weight
                valid_markets += 1
                tasks.append(result)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_results = []
        for (market, timeframe), result in zip(市場清單, results):
            if isinstance(result, Exception):
                logger.error(f"[{market}_{timeframe}] 批次訓練失敗: {result}")
                await push_limiter.cache_message(f"錯誤碼E1202：批次訓練失敗 {result}", market, timeframe, "批次訓練")
                await push_limiter.retry_cached_messages()
            elif result:
                all_results.extend(result)

        # 生成總結報表
        df = pd.DataFrame(all_results)
        if not df.empty:
            csv_path = 快取資料夾.parent / "備份" / f"Multi_Market_Batch_Training_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            await push_limiter.cache_message(f"【執行通知】生成多市場批次訓練報表 {csv_path}", "多市場", "多框架", "批次訓練")
            await push_limiter.retry_cached_messages()

        # 寫入異動歷程
        change_log_path = 快取資料夾.parent / "dot" / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        change_log = {
            "UUID": str(uuid.uuid4()),
            "市場": "多市場",
            "時間框架": "多框架",
            "異動前值": "N/A",
            "異動後值": f"訓練記錄數={len(all_results)}, 加權Sharpe={weighted_sharpe / valid_markets if valid_markets > 0 else 0.0:.2f}",
            "異動原因": "多市場批次訓練",
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
            f"【效率報告】批次訓練耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, "多市場", "多框架", "批次訓練")
        await push_limiter.retry_cached_messages()

        log_queue.put(None)
        listener.terminate()
        gui.root.destroy()
        return all_results
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[多市場_多框架] 批次訓練主函數失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E1202：批次訓練主函數失敗 重試{retry_count + 1}/5: {e}", "批次訓練錯誤", "多市場", "多框架", "批次訓練")
            await asyncio.sleep(5)
            return await main_batch(retry_count + 1)
        logger.error(f"[多市場_多框架] 批次訓練主函數失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E1202：批次訓練主函數失敗 {e}", "多市場", "多框架", "批次訓練")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E1202：批次訓練主函數失敗 {e}", "批次訓練錯誤", "多市場", "多框架", "批次訓練")
        log_queue.put(None)
        listener.terminate()
        return None

if __name__ == "__main__":
    asyncio.run(main_batch())