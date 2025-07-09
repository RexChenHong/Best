import asyncio
import cupy as cp
import numpy as np
import torch
import pandas as pd
import datetime
import multiprocessing as mp
import glob
import logging
import queue
import gzip
import hashlib
import sqlite3
import uuid
from logging.handlers import TimedRotatingFileHandler, QueueHandler
from pathlib import Path
import chardet
import re
import psutil
import os
import time
from io import StringIO
from 設定檔 import 訓練設備, 快取資料夾, 訓練參數, 資源閾值, 市場清單, 點差, 點值
from 工具模組 import 錯誤記錄與自動修復, 檢查檔案路徑, 清理快取檔案, 監控硬體狀態並降級
from 推播通知模組 import 發送錯誤訊息, 發送通知
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
import scipy.interpolate as interp

# 參數配置
MAX_PROC = min(4, mp.cpu_count() - 1)
CHUNK_SIZE = 50000
MIN_CHUNK_SIZE = 10000
OVERLAP = 50
MAX_RAM_USE = 0.7 * psutil.virtual_memory().total
MAX_GPU_USE = 0.85
技術指標名單 = ["SMA50", "HMA_16", "ATR_14", "VHF_28", "PivotHigh", "PivotLow"]
CACHE_VERSION = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# 日誌配置
log_dir = 快取資料夾.parent / "日誌"
log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("資料預處理模組")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
file_handler = TimedRotatingFileHandler(
    filename=log_dir / "preprocess_logs",
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

# GPU/CPU切換計數器
gpu_switch_count = 0

def log_listener(queue, log_file_handler, log_console_handler):
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
    listener = mp.Process(target=log_listener, args=(log_queue, file_handler, console_handler), daemon=True)
    listener.start()
    return listener

def calculate_md5(file_path):
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"MD5計算失敗 [{file_path}]: {e}")
        asyncio.run(發送錯誤訊息(f"錯誤碼E501：MD5計算失敗 [{file_path}]: {e}", None, None, "資料預處理"))
        return None

def check_md5_consistency(source_path, cache_path):
    try:
        if not source_path.exists() or not cache_path.exists():
            logger.warning(f"MD5一致性檢查失敗：檔案不存在 [{source_path}, {cache_path}]")
            return False
        source_md5 = calculate_md5(source_path)
        cache_md5 = calculate_md5(cache_path)
        if source_md5 is None or cache_md5 is None:
            logger.error(f"MD5計算無效：{source_path}, {cache_path}")
            return False
        is_consistent = source_md5 == cache_md5
        conn = sqlite3.connect(快取資料夾.parent / "SQLite" / "預處理紀錄.db")
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS 預處理紀錄表 (
                UUID TEXT PRIMARY KEY,
                市場 TEXT,
                週期 TEXT,
                來源檔案名 TEXT,
                來源路徑 TEXT,
                筆數 INTEGER,
                欄位完整性 BOOLEAN,
                處理時間 REAL,
                GPU標記 BOOLEAN,
                來源MD5 TEXT,
                快取MD5 TEXT,
                檢查結果 BOOLEAN,
                快取檔名 TEXT,
                是否成功 BOOLEAN,
                重試次數 INTEGER,
                完成時間戳 TEXT,
                缺失率 TEXT,
                異常比例 TEXT,
                版本號 TEXT
            )
        """)
        uuid_val = str(uuid.uuid4())
        market, timeframe = source_path.stem.split("_")[:2]
        c.execute("INSERT INTO 預處理紀錄表 (UUID, 市場, 週期, 來源檔案名, 來源路徑, 檢查結果, 來源MD5, 快取MD5, 完成時間戳, 版本號) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (uuid_val, market, timeframe, source_path.name, str(source_path), is_consistent, source_md5, cache_md5, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), CACHE_VERSION))
        conn.commit()
        conn.close()
        if not is_consistent:
            logger.warning(f"MD5不一致：{source_path} vs {cache_path}")
            asyncio.run(發送錯誤訊息(f"錯誤碼E501：MD5不一致 [{source_path}]", market, timeframe, "資料預處理"))
        return is_consistent
    except Exception as e:
        logger.error(f"MD5一致性檢查失敗 [{source_path}]: {e}")
        asyncio.run(發送錯誤訊息(f"錯誤碼E501：MD5一致性檢查失敗 [{source_path}]: {e}", None, None, "資料預處理"))
        asyncio.run(錯誤記錄與自動修復(f"錯誤碼E501：MD5一致性檢查失敗: {e}", "MD5檢查錯誤", None, None, "資料預處理"))
        return False

async def cleanup_expired_cache():
    try:
        cache_files = list(快取資料夾.glob("*.npz"))
        current_time = datetime.datetime.now()
        for cache_file in cache_files:
            file_mtime = datetime.datetime.fromtimestamp(cache_file.stat().st_mtime)
            if (current_time - file_mtime).days > 7:
                cache_file.unlink()
                logger.info(f"清理過期快取: {cache_file}")
                await 發送通知(f"【通知】清理過期快取: {cache_file}", None, None, "資料預處理")
        return True
    except Exception as e:
        logger.error(f"快取清理失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E502：快取清理失敗: {e}", None, None, "資料預處理")
        await 錯誤記錄與自動修復(f"錯誤碼E502：快取清理失敗: {e}", "快取清理錯誤", None, None, "資料預處理")
        return False

def normalize_column_name(col):
    try:
        col = str(col).encode("utf-8").decode("utf-8-sig", errors="ignore")
        col = col.replace('\ufeff', '').replace('\r', '').replace('\n', '').replace('\t', '')
        col = col.replace('\u3000', '').replace(' ', '').replace('_', '').replace('-', '')
        col = col.lower()
        col = re.sub('[^a-z0-9\u4e00-\u9fff]', '', col)
        return col.strip()
    except Exception as e:
        logger.error(f"欄位名稱標準化失敗: {e}")
        return col

def smart_column_map(columns):
    mapping_dict = {
        "open": ["open", "開", "開盤", "開盤價", "開市", "始値", "オープン"],
        "high": ["high", "高", "最高", "最高價", "高値", "ハイ"],
        "low": ["low", "低", "最低", "最低價", "低値", "ロー"],
        "close": ["close", "收", "收盤", "收盤價", "結束", "終値", "クロース"],
        "timestamp": ["timestamp", "datetime", "time", "date", "日期", "時間", "成交時間", "時刻", "時間戳", "datetimeutc", "時間(utc)", "date(utc)"]
    }
    reverse_map = {normalize_column_name(v): std for std, variants in mapping_dict.items() for v in variants}
    return {col: reverse_map.get(normalize_column_name(col), col) for col in columns}

def guess_time_column(columns):
    candidates = ["timestamp", "datetime", "time", "date", "日期", "時間", "成交時間", "時刻", "時間戳", "datetimeutc", "時間(utc)", "date(utc)"]
    normalized_cols = [normalize_column_name(col) for col in columns]
    for cand in candidates:
        if normalize_column_name(cand) in normalized_cols:
            return columns[normalized_cols.index(normalize_column_name(cand))]
    return None

def detect_encoding(filepath, sample_size=10000):
    try:
        with open(filepath, 'rb') as f:
            rawdata = f.read(sample_size)
        result = chardet.detect(rawdata)
        encoding = result['encoding'] if result and result['encoding'] else 'utf-8'
        if encoding not in ['utf-8', 'utf-8-sig', 'ascii', 'big5', 'gb2312']:
            logger.warning(f"檢測到非標準編碼: {encoding}，轉換為 utf-8")
            encoding = 'utf-8'
            asyncio.run(發送錯誤訊息(f"錯誤碼E502：非標準編碼 {encoding}，轉換為 utf-8 [{filepath}]", None, None, "資料預處理"))
        return encoding
    except Exception as e:
        logger.error(f"檔案編碼檢測失敗 [{filepath}]: {e}")
        asyncio.run(發送錯誤訊息(f"錯誤碼E502：檔案編碼檢測失敗 [{filepath}]: {e}", None, None, "資料預處理"))
        return 'utf-8'

def load_zst_csv(file_path):
    try:
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        with open(file_path, 'rb') as f:
            stream_reader = dctx.stream_reader(f)
            text = stream_reader.read().decode('utf-8-sig', errors='replace')
            df = pd.read_csv(StringIO(text))
        required_cols = ['open', 'high', 'low', 'close', 'timestamp']
        col_map = smart_column_map(df.columns)
        df = df.rename(columns=col_map)
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"缺少必要欄位: {missing_cols} [{file_path}]")
            asyncio.run(發送錯誤訊息(f"錯誤碼E502：缺少必要欄位 {missing_cols} [{file_path}]", None, None, "資料預處理"))
            return None
        return df
    except Exception as e:
        logger.error(f"讀取 Zstandard 檔案失敗 [{file_path}]: {e}")
        asyncio.run(發送錯誤訊息(f"錯誤碼E502：讀取 Zstandard 檔案失敗 [{file_path}]: {e}", None, None, "資料預處理"))
        asyncio.run(錯誤記錄與自動修復(f"錯誤碼E502：讀取 Zstandard 檔案失敗: {e}", "檔案讀取錯誤", None, None, "資料預處理", str(file_path)))
        return None

def parse_datetime_column(df, col):
    try:
        df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
        return df[df[col].notnull()].sort_values(by=col).reset_index(drop=True)
    except Exception as e:
        logger.error(f"時間欄位解析失敗 [{col}]: {e}")
        asyncio.run(發送錯誤訊息(f"錯誤碼E502：時間欄位解析失敗 [{col}]: {e}", None, None, "資料預處理"))
        asyncio.run(錯誤記錄與自動修復(f"錯誤碼E502：時間欄位解析失敗: {e}", "時間解析錯誤", None, None, "資料預處理"))
        return df

def check_time_continuity(df, timeframe, file_path):
    try:
        if 'timestamp' not in df.columns:
            return True
        time_diffs = df['timestamp'].diff().dt.total_seconds().dropna()
        expected_interval = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}.get(timeframe, 900)
        gaps = time_diffs[time_diffs > expected_interval * 1.5]
        if len(gaps) > 0:
            logger.warning(f"時間序列不連續 [{file_path}]，檢測到 {len(gaps)} 個間隙")
            asyncio.run(發送錯誤訊息(f"錯誤碼E502：時間序列不連續，檢測到 {len(gaps)} 個間隙 [{file_path}]", None, None, "資料預處理"))
            return False
        return True
    except Exception as e:
        logger.error(f"時間連續性檢查失敗 [{file_path}]: {e}")
        asyncio.run(發送錯誤訊息(f"錯誤碼E502：時間連續性檢查失敗 [{file_path}]: {e}", None, None, "資料預處理"))
        return False

def ensure_all_columns(df, required_cols):
    try:
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
        return df
    except Exception as e:
        logger.error(f"欄位補全失敗: {e}")
        asyncio.run(發送錯誤訊息(f"錯誤碼E502：欄位補全失敗: {e}", None, None, "資料預處理"))
        return df

def check_and_log_anomalies(df, log_writer, file_path):
    critical_cols = ['open', 'high', 'low', 'close', 'timestamp']
    problem = False
    missing_rates = {}
    anomaly_rates = {}
    market = Path(file_path).stem.split("_")[0]
    point_spread = 點差.get(market, 點差["default"])
    point_value = 點值.get(market, 點值["default"])
    for col in critical_cols:
        try:
            missing_rate = df[col].isna().mean()
            missing_rates[col] = missing_rate
            if col in df.columns and df[col].nunique() < 5:
                log_writer.write(f"❗ 欄位[{col}] 變異度極低，資料可能異常：{file_path}\n")
                anomaly_rates[col] = 1.0
                asyncio.run(發送錯誤訊息(f"錯誤碼E502：欄位[{col}] 變異度極低 [{file_path}]", None, None, "資料預處理"))
                problem = True
            if col in ['open', 'high', 'low', 'close'] and col in df.columns:
                if (df[col] <= 0).mean() > 0.05:
                    log_writer.write(f"❗ 欄位[{col}] 非正值超過5%：{file_path}\n")
                    anomaly_rates[col] = (df[col] <= 0).mean()
                    asyncio.run(發送錯誤訊息(f"錯誤碼E502：欄位[{col}] 非正值超過5% [{file_path}]", None, None, "資料預處理"))
                    problem = True
                if len(df[col]) > 100:
                    mean_price = df[col][:100].mean()
                    atr = df['ATR_14'][:100].mean() if 'ATR_14' in df.columns else 0.0
                    threshold = mean_price + max(3 * atr, point_spread * point_value * 5)
                    anomaly_rate = (df[col] > threshold).mean()
                    if anomaly_rate > 0:
                        log_writer.write(f"❗ 欄位[{col}] 異常值超過均值+max(3*ATR, 5*點差*點值)：{file_path}\n")
                        anomaly_rates[col] = anomaly_rate
                        asyncio.run(發送錯誤訊息(f"錯誤碼E502：欄位[{col}] 異常值超過均值+max(3*ATR, 5*點差*點值) [{file_path}]", None, None, "資料預處理"))
                        problem = True
        except Exception as e:
            logger.error(f"資料檢查失敗 [{file_path}]: {e}")
            asyncio.run(發送錯誤訊息(f"錯誤碼E502：資料檢查失敗 [{file_path}]: {e}", None, None, "資料預處理"))
            problem = True
    if problem:
        log_writer.write(f"❌ 資料檔案[{file_path}] 關鍵欄位異常，建議檢查！\n")
        asyncio.run(發送錯誤訊息(f"錯誤碼E502：資料檔案[{file_path}] 關鍵欄位異常", None, None, "資料預處理"))
    conn = sqlite3.connect(快取資料夾.parent / "SQLite" / "預處理紀錄.db")
    c = conn.cursor()
    uuid_val = str(uuid.uuid4())
    market, timeframe = Path(file_path).stem.split("_")[:2]
    c.execute("UPDATE 預處理紀錄表 SET 缺失率 = ?, 異常比例 = ?, UUID = ? WHERE 來源檔案名 = ?",
              (str(missing_rates), str(anomaly_rates), uuid_val, Path(file_path).name))
    conn.commit()
    conn.close()
    asyncio.run(發送通知(f"【資料完整性】檔案: {file_path}\n缺失率: {missing_rates}\n異常比例: {anomaly_rates}", market, timeframe, "資料預處理"))
    if problem:
        # 自動修復：插值異常值或捨棄
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns and anomaly_rates.get(col, 0) > 0:
                valid_mask = df[col].notna() & (df[col] > 0) & (df[col] <= threshold)
                if valid_mask.sum() > 0:
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
                    df[col] = np.where(df[col] > threshold, df[col].mean(), df[col])
                    log_writer.write(f"✅ 欄位[{col}] 異常值修復完成\n")
    return not problem

def try_gpu_indicator(fn, *args, **kwargs):
    global gpu_switch_count
    try:
        if torch.cuda.is_available():
            gpu_prop = torch.cuda.get_device_properties(訓練設備)
            gpu_max = int(gpu_prop.total_memory * MAX_GPU_USE)
            gpu_memory = torch.cuda.memory_allocated(訓練設備)
            cpu_ok = psutil.cpu_percent() < 資源閾值["CPU使用率"] * 100
            if gpu_memory < gpu_max and cpu_ok:
                return fn(*args, **kwargs)
            else:
                gpu_switch_count += 1
                logger.warning(f"資源超限（GPU記憶體 {gpu_memory/1024/1024:.2f}MB/{gpu_max/1024/1024:.2f}MB 或 CPU使用率），切換至CPU，當前切換次數: {gpu_switch_count}")
                asyncio.run(發送通知(f"【通知】資源超限（GPU記憶體 {gpu_memory/1024/1024:.2f}MB/{gpu_max/1024/1024:.2f}MB 或 CPU使用率），切換至CPU，當前切換次數: {gpu_switch_count}", None, None, "資料預處理"))
                torch.cuda.empty_cache()
        np_args = [cp.asnumpy(a) if isinstance(a, cp.ndarray) else a for a in args]
        return fn(*np_args, **kwargs)
    except Exception as e:
        gpu_switch_count += 1
        logger.error(f"技術指標計算失敗，切換至CPU，當前切換次數: {gpu_switch_count}: {e}")
        asyncio.run(發送錯誤訊息(f"錯誤碼E502：技術指標計算失敗，切換至CPU: {e}", None, None, "資料預處理"))
        np_args = [cp.asnumpy(a) if isinstance(a, cp.ndarray) else a for a in args]
        return fn(*np_args, **kwargs)

def calc_sma(arr, window):
    try:
        if isinstance(arr, cp.ndarray):
            kernel = cp.ones(window, dtype=cp.float32) / window
            sma = cp.convolve(arr, kernel, mode='valid')
            nanpad = cp.full(window-1, cp.nan, dtype=cp.float32)
            return cp.concatenate([nanpad, sma])
        else:
            sma = np.convolve(arr, np.ones(window, dtype=np.float32) / window, mode='valid')
            nanpad = np.full(window-1, np.nan, dtype=np.float32)
            return np.concatenate([nanpad, sma])
    except Exception as e:
        logger.error(f"SMA計算錯誤: {e}")
        raise RuntimeError(f"SMA計算錯誤: {e}")

def calc_hma(arr, window):
    try:
        half = int(window / 2)
        sqrtw = int(np.sqrt(window))
        wma1 = calc_wma(arr, half)
        wma2 = calc_wma(arr, window)
        raw = 2 * wma1 - wma2
        return calc_wma(raw, sqrtw)
    except Exception as e:
        logger.error(f"HMA計算錯誤: {e}")
        raise RuntimeError(f"HMA計算錯誤: {e}")

def calc_wma(arr, window):
    try:
        if isinstance(arr, cp.ndarray):
            weights = cp.arange(1, window+1)
            wma = cp.full_like(arr, cp.nan, dtype=cp.float32)
            for i in range(window-1, len(arr)):
                wma[i] = cp.dot(arr[i-window+1:i+1], weights) / weights.sum()
            return wma
        else:
            weights = np.arange(1, window+1)
            wma = np.full_like(arr, np.nan, dtype=np.float32)
            for i in range(window-1, len(arr)):
                wma[i] = np.dot(arr[i-window+1:i+1], weights) / weights.sum()
            return wma
    except Exception as e:
        logger.error(f"WMA計算錯誤: {e}")
        raise RuntimeError(f"WMA計算錯誤: {e}")

def calc_atr(high, low, close, window):
    try:
        if isinstance(high, cp.ndarray):
            prev_close = cp.concatenate([cp.array([close[0]]), close[:-1]])
            tr = cp.maximum(high - low, cp.maximum(cp.abs(high - prev_close), cp.abs(low - prev_close)))
            return calc_sma(tr, window)
        else:
            prev_close = np.concatenate([np.array([close[0]]), close[:-1]])
            tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
            return calc_sma(tr, window)
    except Exception as e:
        logger.error(f"ATR計算錯誤: {e}")
        raise RuntimeError(f"ATR計算錯誤: {e}")

def calc_vhf(close, window):
    try:
        if isinstance(close, cp.ndarray):
            vhf = cp.full_like(close, cp.nan, dtype=cp.float32)
            for i in range(window-1, len(close)):
                maxp = cp.max(close[i-window+1:i+1])
                minp = cp.min(close[i-window+1:i+1])
                denominator = cp.sum(cp.abs(close[i-window+2:i+1] - close[i-window+1:i]))
                vhf[i] = (maxp - minp) / denominator if denominator != 0 else 0
            return vhf
        else:
            vhf = np.full_like(close, np.nan, dtype=np.float32)
            for i in range(window-1, len(close)):
                maxp = np.max(close[i-window+1:i+1])
                minp = np.min(close[i-window+1:i+1])
                denominator = np.sum(np.abs(close[i-window+2:i+1] - close[i-window+1:i]))
                vhf[i] = (maxp - minp) / denominator if denominator != 0 else 0
            return vhf
    except Exception as e:
        logger.error(f"VHF計算錯誤: {e}")
        raise RuntimeError(f"VHF計算錯誤: {e}")

def calc_pivot_high(arr, window):
    try:
        if isinstance(arr, cp.ndarray):
            pivot = cp.full_like(arr, cp.nan, dtype=cp.float32)
            for i in range(window, len(arr)-window):
                center = arr[i]
                if cp.all(center >= arr[i-window:i+window+1]):
                    pivot[i] = center
            return pivot
        else:
            pivot = np.full_like(arr, np.nan, dtype=np.float32)
            for i in range(window, len(arr)-window):
                center = arr[i]
                if np.all(center >= arr[i-window:i+window+1]):
                    pivot[i] = center
            return pivot
    except Exception as e:
        logger.error(f"PivotHigh計算錯誤: {e}")
        raise RuntimeError(f"PivotHigh計算錯誤: {e}")

def calc_pivot_low(arr, window):
    try:
        if isinstance(arr, cp.ndarray):
            pivot = cp.full_like(arr, cp.nan, dtype=cp.float32)
            for i in range(window, len(arr)-window):
                center = arr[i]
                if cp.all(center <= arr[i-window:i+window+1]):
                    pivot[i] = center
            return pivot
        else:
            pivot = np.full_like(arr, np.nan, dtype=np.float32)
            for i in range(window, len(arr)-window):
                center = arr[i]
                if np.all(center <= arr[i-window:i+window+1]):
                    pivot[i] = center
            return pivot
    except Exception as e:
        logger.error(f"PivotLow計算錯誤: {e}")
        raise RuntimeError(f"PivotLow計算錯誤: {e}")

def torch_tensorize(df, device=訓練設備):
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df) == 0:
            logger.error("無數值欄位可轉換為 tensor")
            asyncio.run(發送錯誤訊息("錯誤碼E502：無數值欄位可轉換為 tensor", None, None, "資料預處理"))
            return None
        arr = cp.asarray(numeric_df.values, dtype=cp.float32)
        tensor = torch.as_tensor(arr, device=device)
        return tensor
    except Exception as e:
        logger.error(f"tensor 轉換錯誤: {e}")
        asyncio.run(發送錯誤訊息(f"錯誤碼E502：tensor 轉換錯誤: {e}", None, None, "資料預處理"))
        asyncio.run(錯誤記錄與自動修復(f"錯誤碼E502：tensor 轉換錯誤: {e}", "tensor 轉換錯誤", None, None, "資料預處理"))
        return None

def save_cache(file_path, tensor, columns, meta=None):
    try:
        cache_path = 快取資料夾 / f"{Path(file_path).stem}_已處理資料_v{CACHE_VERSION}.npz"
        if tensor is not None:
            np.savez_compressed(cache_path, tensor=tensor.cpu().numpy(), columns=np.array(columns), meta=meta)
            logger.info(f"已快取: {file_path} → {cache_path}")
            conn = sqlite3.connect(快取資料夾.parent / "SQLite" / "預處理紀錄.db")
            c = conn.cursor()
            uuid_val = str(uuid.uuid4())
            market, timeframe = Path(file_path).stem.split("_")[:2]
            c.execute("UPDATE 預處理紀錄表 SET 快取檔名 = ?, 是否成功 = ?, 筆數 = ?, 欄位完整性 = ?, 版本號 = ? WHERE 來源檔案名 = ?",
                      (str(cache_path), True, meta["rows"], True, CACHE_VERSION, Path(file_path).name))
            conn.commit()
            conn.close()
            return cache_path
        else:
            logger.error(f"無法儲存快取，tensor 為 None: {file_path}")
            asyncio.run(發送錯誤訊息(f"錯誤碼E502：無法儲存快取，tensor 為 None: {file_path}", None, None, "資料預處理"))
            return None
    except Exception as e:
        logger.error(f"快取儲存失敗 [{file_path}]: {e}")
        asyncio.run(發送錯誤訊息(f"錯誤碼E502：快取儲存失敗 [{file_path}]: {e}", None, None, "資料預處理"))
        asyncio.run(錯誤記錄與自動修復(f"錯誤碼E502：快取儲存失敗: {e}", "快取儲存錯誤", None, None, "資料預處理", str(file_path)))
        return None

def check_system_resource():
    try:
        ram_ok = psutil.virtual_memory().used < MAX_RAM_USE
        cpu_ok = psutil.cpu_percent() < 資源閾值["CPU使用率"] * 100
        disk = psutil.disk_usage(str(快取資料夾))
        disk_ok = (disk.free / disk.total) * 100 > 資源閾值["硬碟剩餘比例"]
        if torch.cuda.is_available():
            gpu_prop = torch.cuda.get_device_properties(訓練設備)
            gpu_max = int(gpu_prop.total_memory * MAX_GPU_USE)
            gpu_ok = torch.cuda.memory_allocated(訓練設備) < gpu_max
        else:
            gpu_ok = True
        return ram_ok and cpu_ok and disk_ok and gpu_ok
    except Exception as e:
        logger.error(f"資源檢查失敗: {e}")
        asyncio.run(發送錯誤訊息(f"錯誤碼E502：資源檢查失敗: {e}", None, None, "資料預處理"))
        return False

def generate_technical_indicators(df, periods=None):
    try:
        if periods is None:
            periods = {
                "SMA": 訓練參數["SMA週期"]["值"] if 訓練參數["SMA週期"]["啟用"] else 50,
                "HMA": 訓練參數["HMA週期"]["值"] if 訓練參數["HMA週期"]["啟用"] else 16,
                "ATR": 訓練參數["ATR週期"]["值"] if 訓練參數["ATR週期"]["啟用"] else 14,
                "VHF": 訓練參數["VHF週期"]["值"] if 訓練參數["VHF週期"]["啟用"] else 28,
                "Pivot": 訓練參數["Pivot週期"]["值"] if 訓練參數["Pivot週期"]["啟用"] else 5
            }
        custom_indicators = 訓練參數.get("custom_indicators", {})
        arr_close = cp.asarray(df['close'].values, dtype=cp.float32)
        arr_high = cp.asarray(df['high'].values, dtype=cp.float32)
        arr_low = cp.asarray(df['low'].values, dtype=cp.float32)
        if 訓練參數["SMA週期"]["啟用"]:
            df['SMA50'] = try_gpu_indicator(calc_sma, arr_close, periods["SMA"])
        if 訓練參數["HMA週期"]["啟用"]:
            df['HMA_16'] = try_gpu_indicator(calc_hma, arr_close, periods["HMA"])
        if 訓練參數["ATR週期"]["啟用"]:
            df['ATR_14'] = try_gpu_indicator(calc_atr, arr_high, arr_low, arr_close, periods["ATR"])
        if 訓練參數["VHF週期"]["啟用"]:
            df['VHF_28'] = try_gpu_indicator(calc_vhf, arr_close, periods["VHF"])
        if 訓練參數["Pivot週期"]["啟用"]:
            df['PivotHigh'] = try_gpu_indicator(calc_pivot_high, arr_high, periods["Pivot"])
            df['PivotLow'] = try_gpu_indicator(calc_pivot_low, arr_low, periods["Pivot"])
        for name, config in custom_indicators.items():
            if config.get("enabled", False):
                df[name] = try_gpu_indicator(eval(config["function"]), arr_close, config.get("window", 14))
        conn = sqlite3.connect(快取資料夾.parent / "SQLite" / "預處理紀錄.db")
        c = conn.cursor()
        uuid_val = str(uuid.uuid4())
        market, timeframe = Path(file_path).stem.split("_")[:2]
        c.execute("INSERT INTO 預處理紀錄表 (UUID, 市場, 週期, 技術指標) VALUES (?, ?, ?, ?)",
                  (uuid_val, market, timeframe, str({**periods, **{k: v.get("window", 14) for k, v in custom_indicators.items()}})))
        conn.commit()
        conn.close()
        return df
    except Exception as e:
        logger.error(f"技術指標運算失敗: {e}")
        asyncio.run(發送錯誤訊息(f"錯誤碼E502：技術指標運算失敗: {e}", None, None, "資料預處理"))
        asyncio.run(錯誤記錄與自動修復(f"錯誤碼E502：技術指標運算失敗: {e}", "指標運算錯誤", None, None, "資料預處理"))
        return df

async def adjust_chunk_size(current_chunk_size):
    try:
        memory = psutil.virtual_memory()
        ram_percent = memory.percent
        disk = psutil.disk_usage(str(快取資料夾))
        disk_free_percent = (disk.free / disk.total) * 100
        cpu_ok = psutil.cpu_percent() < 資源閾值["CPU使用率"] * 100
        gpu_ok = True
        if torch.cuda.is_available():
            gpu_prop = torch.cuda.get_device_properties(訓練設備)
            gpu_max = int(gpu_prop.total_memory * MAX_GPU_USE)
            gpu_ok = torch.cuda.memory_allocated(訓練設備) < gpu_max
        if ram_percent > 資源閾值["RAM使用率"] or disk_free_percent < 資源閾值["硬碟剩餘比例"] or not cpu_ok or not gpu_ok:
            new_size = max(MIN_CHUNK_SIZE, current_chunk_size // 2)
            logger.info(f"資源超限: RAM={ram_percent}%, 硬碟剩餘={disk_free_percent}%, CPU={'正常' if cpu_ok else '超載'}, GPU={'正常' if gpu_ok else '超載'}，調整 CHUNK_SIZE 至 {new_size}")
            await 發送通知(f"【通知】資源超限: RAM={ram_percent}%, 硬碟剩餘={disk_free_percent}%, CPU={'正常' if cpu_ok else '超載'}, GPU={'正常' if gpu_ok else '超載'}，調整 CHUNK_SIZE 至 {new_size}", None, None, "資料預處理")
            return new_size
        return current_chunk_size
    except Exception as e:
        logger.error(f"CHUNK_SIZE 調整失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E502：CHUNK_SIZE 調整失敗: {e}", None, None, "資料預處理")
        return current_chunk_size

def split_datasets(df, file_path):
    try:
        train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
        train_size = int(len(df) * train_ratio)
        val_size = int(len(df) * val_ratio)
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
        market, timeframe = Path(file_path).stem.split("_")[:2]
        train_path = 快取資料夾 / f"{market}_{timeframe}_train_v{CACHE_VERSION}.npz"
        val_path = 快取資料夾 / f"{market}_{timeframe}_val_v{CACHE_VERSION}.npz"
        test_path = 快取資料夾 / f"{market}_{timeframe}_test_v{CACHE_VERSION}.npz"
        for subset, path in [(train_df, train_path), (val_df, val_path), (test_df, test_path)]:
            tensor = torch_tensorize(subset, device='cpu')
            if tensor is not None:
                np.savez_compressed(path, tensor=tensor.cpu().numpy(), columns=np.array(subset.columns))
        conn = sqlite3.connect(快取資料夾.parent / "SQLite" / "預處理紀錄.db")
        c = conn.cursor()
        uuid_val = str(uuid.uuid4())
        c.execute("INSERT INTO 預處理紀錄表 (UUID, 市場, 週期, 訓練集筆數, 驗證集筆數, 測試集筆數, 版本號) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (uuid_val, market, timeframe, len(train_df), len(val_df), len(test_df), CACHE_VERSION))
        conn.commit()
        conn.close()
        return {"train": train_path, "val": val_path, "test": test_path}
    except Exception as e:
        logger.error(f"數據集分割失敗 [{file_path}]: {e}")
        asyncio.run(發送錯誤訊息(f"錯誤碼E502：數據集分割失敗 [{file_path}]: {e}", None, None, "資料預處理"))
        asyncio.run(錯誤記錄與自動修復(f"錯誤碼E502：數據集分割失敗: {e}", "數據集分割錯誤", None, None, "資料預處理", str(file_path)))
        return None

async def process_single_file(file_path, log_writer, periods=None, retry_count=0):
    try:
        start_time = time.time()
        file_path = Path(file_path)
        if not await 檢查檔案路徑(file_path):
            logger.error(f"無效檔案路徑: {file_path}")
            await 發送錯誤訊息(f"錯誤碼E502：無效檔案路徑: {file_path}", None, None, "資料預處理")
            return None
        cache_path = 快取資料夾 / f"{file_path.stem}_已處理資料_v{CACHE_VERSION}.npz"
        if cache_path.exists() and check_md5_consistency(file_path, cache_path):
            logger.info(f"MD5一致，跳過處理: {file_path}")
            await 發送通知(f"【通知】MD5一致，跳過處理: {file_path}", None, None, "資料預處理")
            return cache_path
        required_cols = ['open', 'high', 'low', 'close', 'timestamp'] + 技術指標名單
        if file_path.suffix.lower() == ".zst":
            df_source = load_zst_csv(file_path)
        elif file_path.suffix.lower() == ".gz":
            with gzip.open(file_path, "rt", encoding="utf-8-sig") as f:
                df_source = pd.read_csv(f)
        else:
            encoding = detect_encoding(file_path)
            df_source = pd.read_csv(file_path, encoding=encoding)
        if df_source is None:
            return None
        available_cols = df_source.columns.str.lower().str.strip()
        time_col = guess_time_column(available_cols)
        if not time_col:
            logger.error(f"無可用時間欄位: {df_source.columns.tolist()}")
            await 發送錯誤訊息(f"錯誤碼E502：無可用時間欄位: {df_source.columns.tolist()}", None, None, "資料預處理")
            return None
        df_source = parse_datetime_column(df_source, time_col)
        market, timeframe = Path(file_path).stem.split("_")[:2]
        if not check_time_continuity(df_source, timeframe, file_path):
            return None
        total_rows = len(df_source)
        chunk_thresh = 500000
        all_chunks = []
        current_chunk_size = CHUNK_SIZE
        checkpoint = 0
        if total_rows > chunk_thresh:
            chunker = pd.read_csv(file_path, encoding=detect_encoding(file_path), chunksize=current_chunk_size)
            last_overlap = None
            for chunk_idx, chunk in enumerate(chunker):
                if not check_system_resource():
                    current_chunk_size = await adjust_chunk_size(current_chunk_size)
                    chunker = pd.read_csv(file_path, encoding=detect_encoding(file_path), chunksize=current_chunk_size)
                    torch.cuda.empty_cache()
                    await asyncio.sleep(0.005)
                if last_overlap is not None:
                    chunk = pd.concat([last_overlap, chunk], ignore_index=True)
                chunk = await process_chunk(chunk, required_cols, periods)
                if chunk is None:
                    continue
                last_overlap = chunk.iloc[-OVERLAP:].copy()
                all_chunks.append(chunk.iloc[OVERLAP:].reset_index(drop=True) if len(all_chunks) > 0 else chunk.reset_index(drop=True))
                checkpoint = chunk_idx + 1
                del chunk
                torch.cuda.empty_cache()
            if all_chunks:
                df_final = pd.concat(all_chunks, ignore_index=True)
            else:
                logger.error(f"無有效分塊數據: {file_path}, 恢復點: {checkpoint}")
                await 發送錯誤訊息(f"錯誤碼E502：無有效分塊數據: {file_path}, 恢復點: {checkpoint}", None, None, "資料預處理")
                return None
        else:
            df_final = await process_chunk(df_source, required_cols, periods)
            if df_final is None:
                return None
        if not check_and_log_anomalies(df_final, log_writer, file_path):
            return None
        dataset_paths = split_datasets(df_final, file_path)
        if dataset_paths is None:
            return None
        tensor = torch_tensorize(df_final, device='cpu')
        meta = {"cols": df_final.columns.tolist(), "rows": len(df_final), "train_path": str(dataset_paths["train"]), "val_path": str(dataset_paths["val"]), "test_path": str(dataset_paths["test"])}
        path = save_cache(file_path, tensor, df_final.columns, meta=meta)
        elapsed_time = time.time() - start_time
        if path:
            await 發送通知(
                f"【資料預處理完成】\nUUID: {uuid_val}\n市場: {market}\n週期: {timeframe}\n筆數: {meta['rows']}\nGPU: {torch.cuda.is_available()}\n快取: {path}\n處理時間: {elapsed_time:.2f}秒\n時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                market, timeframe, "資料預處理"
            )
        return dataset_paths
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"檔案處理失敗，重試 {retry_count + 1}/5: {e}, 恢復點: {checkpoint}")
            await 錯誤記錄與自動修復(f"錯誤碼E502：檔案處理失敗 重試{retry_count + 1}/5: {e}, 恢復點: {checkpoint}", "檔案處理錯誤", None, None, "資料預處理", str(file_path))
            await asyncio.sleep(5)
            return await process_single_file(file_path, log_writer, periods, retry_count + 1)
        logger.error(f"檔案處理失敗，重試 5 次無效 [{file_path}]: {e}, 恢復點: {checkpoint}")
        await 發送錯誤訊息(f"錯誤碼E502：檔案處理失敗，重試 5 次無效 [{file_path}]: {e}, 恢復點: {checkpoint}", None, None, "資料預處理")
        await 錯誤記錄與自動修復(f"錯誤碼E502：檔案處理失敗: {e}, 恢復點: {checkpoint}", "檔案處理錯誤", None, None, "資料預處理", str(file_path))
        return None

async def process_chunk(chunk, required_cols, periods=None, retry_count=0):
    try:
        col_map = smart_column_map(chunk.columns)
        chunk = chunk.rename(columns=col_map)
        chunk = ensure_all_columns(chunk, ['open', 'high', 'low', 'close', 'timestamp'])
        chunk = generate_technical_indicators(chunk, periods)
        chunk = ensure_all_columns(chunk, required_cols)
        if 'timestamp' in chunk.columns and not np.issubdtype(chunk['timestamp'].dtype, np.datetime64):
            chunk = parse_datetime_column(chunk, 'timestamp')
        chunk = chunk[required_cols]
        return chunk
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"分塊處理失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E502：分塊處理失敗 重試{retry_count + 1}/5: {e}", "分塊處理錯誤", None, None, "資料預處理")
            await asyncio.sleep(5)
            return await process_chunk(chunk, required_cols, periods, retry_count + 1)
        logger.error(f"分塊處理失敗，重試 5 次無效: {e}")
        await 發送錯誤訊息(f"錯誤碼E502：分塊處理失敗，重試 5 次無效: {e}", None, None, "資料預處理")
        await 錯誤記錄與自動修復(f"錯誤碼E502：分塊處理失敗: {e}", "分塊處理錯誤", None, None, "資料預處理")
        return None

async def load_and_preprocess_data(market, timeframe, periods=None, retry_count=0):
    try:
        start_time = time.time()
        file_path = 快取資料夾.parent / f"訓練資料/{market}_{timeframe}.csv"
        if not file_path.exists():
            file_path = 快取資料夾.parent / f"訓練資料/{market}_{timeframe}.zst"
        if not file_path.exists():
            file_path = 快取資料夾.parent / f"訓練資料/{market}_{timeframe}.gz"
        if not await 檢查檔案路徑(file_path):
            logger.error(f"[{market}_{timeframe}] 無效檔案路徑: {file_path}")
            await 發送錯誤訊息(f"錯誤碼E502：無效檔案路徑: {file_path}", market, timeframe, "資料預處理")
            return None
        log_path = log_dir / f"資料預處理日誌_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with open(log_path, 'w', encoding='utf-8') as log_writer:
            dataset_paths = await process_single_file(file_path, log_writer, periods)
        if dataset_paths:
            elapsed_time = time.time() - start_time
            logger.info(f"[{market}_{timeframe}] 數據預處理完成，耗時: {elapsed_time:.2f}秒")
            return dataset_paths
        return None
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[{market}_{timeframe}] 數據預處理失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E502：數據預處理失敗 重試{retry_count + 1}/5: {e}", "數據預處理錯誤", market, timeframe, "資料預處理")
            await asyncio.sleep(5)
            return await load_and_preprocess_data(market, timeframe, periods, retry_count + 1)
        logger.error(f"[{market}_{timeframe}] 數據預處理失敗，重試 5 次無效: {e}")
        await 發送錯誤訊息(f"錯誤碼E502：數據預處理失敗，重試 5 次無效: {e}", market, timeframe, "資料預處理")
        await 錯誤記錄與自動修復(f"錯誤碼E502：數據預處理失敗: {e}", "數據預處理錯誤", market, timeframe, "資料預處理")
        return None

async def main():
    try:
        listener = start_log_listener()
        await cleanup_expired_cache()
        zst_files = list(快取資料夾.parent.glob("訓練資料/*.zst")) + list(快取資料夾.parent.glob("訓練資料/*.gz")) + list(快取資料夾.parent.glob("訓練資料/*.csv"))
        total_files = len(zst_files)
        if total_files == 0:
            logger.error("未找到任何 .zst、.gz 或 .csv 檔案")
            await 發送錯誤訊息("錯誤碼E502：未找到任何 .zst、.gz 或 .csv 檔案", None, None, "資料預處理")
            log_queue.put(None)
            listener.terminate()
            return
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = log_dir / f"資料預處理日誌_{now}.log"
        current_proc = MAX_PROC
        with open(log_path, 'w', encoding='utf-8') as log:
            log.write(f"📁 共發現檔案數量：{total_files}\n")
            with mp.Pool(processes=current_proc) as pool:
                for idx, file_path in enumerate(zst_files, 1):
                    if not check_system_resource():
                        _, current_proc = await 監控硬體狀態並降級(CHUNK_SIZE, current_proc, priority=1)
                        pool.close()
                        pool.join()
                        pool = mp.Pool(processes=current_proc)
                    log.write(f"[{idx}/{total_files}] 🕒 處理中：{file_path.name}，使用進程數：{current_proc}\n")
                    pool.apply_async(process_single_file, args=(file_path, log))
                pool.close()
                pool.join()
            log.write(f"✅ 預處理完成，詳細記錄見：{log_path}\n")
            await 發送通知(f"【通知】預處理完成，詳細記錄: {log_path}，總進程數: {current_proc}", None, None, "資料預處理")
        log_queue.put(None)
        listener.terminate()
    except Exception as e:
        logger.error(f"主流程失敗: {e}")
        await 發送錯誤訊息(f"錯誤碼E502：資料預處理主流程失敗: {e}", None, None, "資料預處理")
        await 錯誤記錄與自動修復(f"錯誤碼E502：主流程失敗: {e}", "主流程錯誤", None, None, "資料預處理")
        log_queue.put(None)
        listener.terminate()

def 取得Kfold資料集(dataset, K=5, shuffle=True, random_state=42, batch_size=128):
    try:
        kf = KFold(n_splits=K, shuffle=shuffle, random_state=random_state)
        fold_data = []
        for train_idx, val_idx in kf.split(range(len(dataset))):
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
            fold_data.append((train_loader, val_loader))
        return fold_data
    except Exception as e:
        logger.error(f"KFold 資料集生成失敗: {e}")
        asyncio.run(發送錯誤訊息(f"錯誤碼E502：KFold 資料集生成失敗: {e}", None, None, "資料預處理"))
        asyncio.run(錯誤記錄與自動修復(f"錯誤碼E502：KFold 資料集生成失敗: {e}", "KFold 錯誤", None, None, "資料預處理"))
        return []

def 滑動窗口資料生成(X, y, window_size=60, step_size=1):
    try:
        Xs, ys = [], []
        N = len(X)
        for i in range(0, N - window_size + 1, step_size):
            Xs.append(X[i:i+window_size])
            ys.append(y[i+window_size-1])
        return np.array(Xs), np.array(ys)
    except Exception as e:
        logger.error(f"滑動窗口資料生成失敗: {e}")
        asyncio.run(發送錯誤訊息(f"錯誤碼E502：滑動窗口資料生成失敗: {e}", None, None, "資料預處理"))
        asyncio.run(錯誤記錄與自動修復(f"錯誤碼E502：滑動窗口資料生成失敗: {e}", "滑動窗口錯誤", None, None, "資料預處理"))
        return np.array([]), np.array([])

if __name__ == "__main__":
    asyncio.run(main())