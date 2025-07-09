import torch
import numpy as np
import pandas as pd
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from collections import deque
import datetime
import aiosqlite
import uuid
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import precision_recall_fscore_support
from cryptography.fernet import Fernet
from 設定檔 import 訓練設備, SQLite資料夾, 訓練參數, 市場清單
from 工具模組 import 錯誤記錄與自動修復, 監控硬體狀態並降級, validate_utility_input
from 推播通知模組 import 發送錯誤訊息, 發送通知

# 配置日誌
log_dir = SQLite資料夾.parent / "日誌"
log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("績效分析模組")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
file_handler = TimedRotatingFileHandler(
    filename=log_dir / "performance_analysis_logs",
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
    "E501": "績效分析失敗",
    "E502": "風險指標異常",
    "E503": "硬體資源超限",
    "E504": "數據加密失敗"
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

# 績效緩衝區
performance_buffer = deque(maxlen=100)

async def encrypt_performance_data(data):
    """加密績效數據"""
    try:
        key_path = SQLite資料夾.parent / "secure_key.key"
        if not key_path.exists():
            key = Fernet.generate_key()
            with open(key_path, "wb") as key_file:
                key_file.write(key)
                
        with open(key_path, "rb") as key_file:
            key = key_file.read()
        cipher = Fernet(key)
        encrypted_data = cipher.encrypt(json.dumps(data, ensure_ascii=False).encode('utf-8'))
        
        logger.info("[資安] 績效數據加密完成")
        await push_limiter.cache_message("[執行通知] 績效數據加密完成", "多市場", "多框架", "績效分析", "normal")
        await push_limiter.retry_cached_messages()
        
        with open(log_dir / f"資安事件_{datetime.datetime.now().strftime('%Y%m%d')}.log", "a", encoding='utf-8') as f:
            f.write(f"[資安事件] 績效數據加密\n時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        return encrypted_data
    except Exception as e:
        logger.error(f"[資安] 數據加密失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E504：數據加密失敗 {e}", "多市場", "多框架", "績效分析", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E504：數據加密失敗 {e}", "數據加密錯誤", "多市場", "多框架", "績效分析")
        return None

async def validate_input(trading_result, market, timeframe, params):
    """驗證輸入數據"""
    try:
        if not await validate_utility_input(market, timeframe, mode="績效分析"):
            return False
        required_keys = ["最終資金", "最大回撤", "交易記錄", "維持率"]
        if not all(key in trading_result for key in required_keys):
            logger.error(f"[{market}_{timeframe}] 缺少必要交易結果: {required_keys}")
            await push_limiter.cache_message(f"錯誤碼E501：缺少必要交易結果 {required_keys}", market, timeframe, "績效分析", "high")
            await push_limiter.retry_cached_messages()
            return False
        if not isinstance(trading_result["最終資金"], (int, float)) or trading_result["最終資金"] < 0:
            logger.error(f"[{market}_{timeframe}] 無效最終資金: {trading_result['最終資金']}")
            await push_limiter.cache_message(f"錯誤碼E501：無效最終資金 {trading_result['最終資金']}", market, timeframe, "績效分析", "high")
            await push_limiter.retry_cached_messages()
            return False
        if not isinstance(trading_result["最大回撤"], (int, float)) or trading_result["最大回撤"] < 0:
            logger.error(f"[{market}_{timeframe}] 無效最大回撤: {trading_result['最大回撤']}")
            await push_limiter.cache_message(f"錯誤碼E501：無效最大回撤 {trading_result['最大回撤']}", market, timeframe, "績效分析", "high")
            await push_limiter.retry_cached_messages()
            return False
        if not isinstance(trading_result["交易記錄"], list):
            logger.error(f"[{market}_{timeframe}] 無效交易記錄格式")
            await push_limiter.cache_message(f"錯誤碼E501：無效交易記錄格式", market, timeframe, "績效分析", "high")
            await push_limiter.retry_cached_messages()
            return False
        if not isinstance(params, dict):
            logger.error(f"[{market}_{timeframe}] 無效參數格式: {params}")
            await push_limiter.cache_message(f"錯誤碼E501：無效參數格式 {params}", market, timeframe, "績效分析", "high")
            await push_limiter.retry_cached_messages()
            return False
        return True
    except Exception as e:
        logger.error(f"[{market}_{timeframe}] 輸入驗證失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E501：輸入驗證失敗 {e}", market, timeframe, "績效分析", "high")
        await push_limiter.retry_cached_messages()
        return False

async def calculate_f1_score(trading_records):
    """計算 F1 分數"""
    try:
        true_labels = [1 if record["損益"] > 0 else 0 for record in trading_records if "損益" in record]
        pred_labels = [1 if record["類型"] in ["買", "賣"] else 0 for record in trading_records if "類型" in record]
        if len(true_labels) == 0 or len(pred_labels) == 0:
            return 0.0
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary', zero_division=0)
        return f1
    except Exception as e:
        logger.error(f"F1分數計算失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E502：F1分數計算失敗 {e}", None, None, "績效分析", "high")
        await push_limiter.retry_cached_messages()
        return 0.0

async def calculate_var(returns, confidence_level=0.95):
    """計算VaR_95"""
    try:
        if len(returns) == 0:
            return 0.0
        sorted_returns = np.sort(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        return -sorted_returns[index] if index < len(sorted_returns) else 0.0
    except Exception as e:
        logger.error(f"VaR_95計算失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E502：VaR_95計算失敗 {e}", None, None, "績效分析", "high")
        await push_limiter.retry_cached_messages()
        return 0.0

async def calculate_cvar(returns, confidence_level=0.95):
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
        await push_limiter.cache_message(f"錯誤碼E502：CVaR_95計算失敗 {e}", None, None, "績效分析", "high")
        await push_limiter.retry_cached_messages()
        return 0.0

async def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """計算年化夏普比率"""
    try:
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        mean_return = np.mean(returns) * 252  # 年化
        std_return = np.std(returns) * np.sqrt(252)  # 年化波動率
        sharpe = (mean_return - risk_free_rate) / std_return if std_return != 0 else 0.0
        return sharpe
    except Exception as e:
        logger.error(f"夏普比率計算失敗: {e}")
        await push_limiter.cache_message(f"錯誤碼E502：夏普比率計算失敗 {e}", None, None, "績效分析", "high")
        await push_limiter.retry_cached_messages()
        return 0.0

async def calculate_performance(trading_result, market, timeframe, params, custom_metrics=None, retry_count=0):
    """計算單市場績效，支持自定義指標"""
    try:
        import time
        start_time = time.time()
        if not await validate_input(trading_result, market, timeframe, params):
            return None

        # 硬體監控
        batch_size, _ = await 監控硬體狀態並降級(32, 2)
        if batch_size < 8:
            logger.error(f"[{market}_{timeframe}] 硬體資源超限，批次大小 {batch_size}")
            await push_limiter.cache_message(f"錯誤碼E503：硬體資源超限，批次大小 {batch_size}", market, timeframe, "績效分析", "high")
            await push_limiter.retry_cached_messages()
            return None
        torch.cuda.empty_cache()

        # 提取交易數據
        初始資金 = 1000.0
        最終資金 = trading_result["最終資金"]
        最大回撤 = trading_result["最大回撤"]
        交易記錄 = trading_result["交易記錄"]
        維持率 = trading_result["維持率"]

        # 計算績效指標
        returns = [record["損益"] / 初始資金 for record in 交易記錄 if "損益" in record]
        交易次數 = len(交易記錄)
        勝率 = len([r for r in returns if r > 0]) / 交易次數 if 交易次數 > 0 else 0.0
        f1分數 = await calculate_f1_score(交易記錄)
        穩定性 = np.std(returns) if returns else 0.0
        var_95 = await calculate_var(returns)
        cvar_95 = await calculate_cvar(returns)
        sharpe_ratio = await calculate_sharpe_ratio(returns)
        平均持倉時間 = sum(record["持倉時間"] for record in 交易記錄) / 交易次數 if 交易次數 > 0 else 0
        單筆損失 = min(returns) if returns else 0.0

        # 自定義指標處理
        custom_performance = {}
        if custom_metrics:
            for metric_name, metric_func in custom_metrics.items():
                custom_value = await metric_func(交易記錄, returns)
                custom_performance[metric_name] = custom_value

        # 穩定性與風險檢查
        if 穩定性 > 0.1 or sharpe_ratio < 1.5 or 最大回撤 > 0.25:
            logger.warning(f"[{market}_{timeframe}] 穩定性或風險未達標: 標準差={穩定性:.4f}, Sharpe={sharpe_ratio:.2f}, 最大回撤={最大回撤:.2%}")
            await push_limiter.cache_message(
                f"錯誤碼E502：穩定性或風險未達標: 標準差={穩定性:.4f}, Sharpe={sharpe_ratio:.2f}, 最大回撤={最大回撤:.2%}",
                market, timeframe, "績效分析", "high")
            await push_limiter.retry_cached_messages()

        if 單筆損失 < -params.get("single_loss_limit", 訓練參數.get("single_loss_limit", {"值": 0.02})["值"]):
            logger.warning(f"[{market}_{timeframe}] 單筆損失超限: {單筆損失:.4f}")
            await push_limiter.cache_message(f"錯誤碼E502：單筆損失超限: {單筆損失:.4f}", market, timeframe, "績效分析", "high")
            await push_limiter.retry_cached_messages()

        # 績效結果
        performance = {
            "最終資金": 最終資金,
            "報酬率": (最終資金 - 初始資金) / 初始資金,
            "最大回撤": 最大回撤,
            "勝率": 勝率,
            "f1分數": f1分數,
            "穩定性": 穩定性,
            "維持率": 維持率,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "sharpe_ratio": sharpe_ratio,
            "交易次數": 交易次數,
            "平均持倉時間": 平均持倉時間,
            "單筆損失": 單筆損失,
            **custom_performance
        }

        # 加密並儲存績效記錄
        encrypted_performance = await encrypt_performance_data(performance)
        if not encrypted_performance:
            return None
        async with aiosqlite.connect(SQLite資料夾 / "performance_log.db") as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS 績效記錄 (
                    id TEXT PRIMARY KEY,
                    市場 TEXT,
                    時間框架 TEXT,
                    最終資金 REAL,
                    報酬率 REAL,
                    最大回撤 REAL,
                    勝率 REAL,
                    f1分數 REAL,
                    穩定性 REAL,
                    維持率 REAL,
                    var_95 REAL,
                    cvar_95 REAL,
                    sharpe_ratio REAL,
                    交易次數 INTEGER,
                    平均持倉時間 REAL,
                    單筆損失 REAL,
                    參數 BLOB,
                    時間 TEXT
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_市場_時間框架 ON 績效記錄 (市場, 時間框架, 時間)")
            await conn.execute("INSERT INTO 績效記錄 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                             (str(uuid.uuid4()), market, timeframe, 最終資金, performance["報酬率"], 最大回撤,
                              勝率, f1分數, 穩定性, 維持率, var_95, cvar_95, sharpe_ratio, 交易次數, 平均持倉時間,
                              單筆損失, encrypted_performance, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            await conn.commit()

        # 風險指標推播
        if var_95 > 0.05 or cvar_95 > 0.05 or sharpe_ratio < 1.5:
            logger.warning(f"[{market}_{timeframe}] 風險指標異常: VaR_95={var_95:.4f}, CVaR_95={cvar_95:.4f}, Sharpe={sharpe_ratio:.4f}")
            await push_limiter.cache_message(
                f"錯誤碼E502：風險指標異常: VaR_95={var_95:.4f}, CVaR_95={cvar_95:.4f}, Sharpe={sharpe_ratio:.4f}",
                market, timeframe, "績效分析", "high")
            await push_limiter.retry_cached_messages()

        # 長期績效分析
        performance_buffer.append(performance["報酬率"])
        if len(performance_buffer) == performance_buffer.maxlen:
            performances = np.array(list(performance_buffer))
            mean_performance = performances.mean()
            std_performance = performances.std()
            async with aiosqlite.connect(SQLite資料夾 / "performance_log.db") as conn:
                await conn.execute("INSERT INTO 績效記錄 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                 (str(uuid.uuid4()), market, timeframe, 0.0, mean_performance, 0.0, 0.0, 0.0,
                                  std_performance, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, json.dumps(params, ensure_ascii=False),
                                  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                await conn.commit()
            if std_performance > 0.1:
                logger.warning(f"[{market}_{timeframe}] 長期績效穩定性低，標準差: {std_performance:.4f}")
                await push_limiter.cache_message(
                    f"錯誤碼E502：長期績效穩定性低，標準差 {std_performance:.4f}",
                    market, timeframe, "績效分析", "high")
                await push_limiter.retry_cached_messages()
                # 暫停流程
                logger.info(f"[{market}_{timeframe}] 穩定性低，暫停流程")
                await push_limiter.cache_message("【執行通知】穩定性低，暫停流程", market, timeframe, "績效分析", "high")
                await push_limiter.retry_cached_messages()
                return None

            # 長期績效趨勢圖表
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=list(performance_buffer),
                mode="lines+markers",
                name="報酬率",
                line=dict(color="green" if mean_performance > 0 else "red")
            ))
            fig.add_trace(go.Scatter(
                y=[mean_performance] * len(performance_buffer),
                mode="lines",
                name="平均報酬率",
                line=dict(dash="dash", color="blue")
            ))
            fig.add_trace(go.Scatter(
                y=[performance["sharpe_ratio"]] * len(performance_buffer),
                mode="lines",
                name="Sharpe比率",
                yaxis="y2",
                line=dict(color="orange" if performance["sharpe_ratio"] < 1.5 else "cyan")
            ))
            fig.update_layout(
                title=f"{market}_{timeframe} 長期績效與Sharpe趨勢",
                xaxis_title="時間",
                yaxis_title="報酬率 (%)",
                yaxis2=dict(title="Sharpe比率", overlaying="y", side="right"),
                template="plotly_dark",
                height=600,
                showlegend=True,
                updatemenus=[
                    dict(
                        buttons=[
                            dict(label="全選", method="update", args=[{"visible": [True] * len(fig.data)}]),
                            dict(label="僅報酬率", method="update", args=[{"visible": [i == 0 for i in range(len(fig.data))]}]),
                            dict(label="僅Sharpe", method="update", args=[{"visible": [i == 2 for i in range(len(fig.data))]}])
                        ],
                        direction="down",
                        showactive=True
                    )
                ]
            )
            plot_path = SQLite資料夾.parent / "圖片" / f"{market}_{timeframe}_Long_Term_Performance_Trend_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            fig.write_html(plot_path)
            await push_limiter.cache_message(f"【執行通知】生成長期績效趨勢圖表 {plot_path}", market, timeframe, "績效分析", "normal")
            await push_limiter.retry_cached_messages()

        # 快取績效數據
        cache_path = 快取資料夾 / f"performance_cache_{market}_{timeframe}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez_compressed(cache_path, data=performance)
        await push_limiter.cache_message(f"【執行通知】績效數據已快取至 {cache_path}", market, timeframe, "績效分析", "normal")
        await push_limiter.retry_cached_messages()

        # 生成報表
        df = pd.DataFrame([performance])
        csv_path = SQLite資料夾.parent / "備份" / f"{market}_{timeframe}_Performance_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        await push_limiter.cache_message(f"【執行通知】生成績效報表 {csv_path}", market, timeframe, "績效分析", "normal")
        await push_limiter.retry_cached_messages()

        # 寫入異動歷程
        change_log_path = SQLite資料夾.parent / "dot" / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        change_log = {
            "UUID": str(uuid.uuid4()),
            "市場": market,
            "時間框架": timeframe,
            "異動前值": "N/A",
            "異動後值": f"報酬率={performance['報酬率']:.4f}, Sharpe={sharpe_ratio:.2f}, 最大回撤={最大回撤:.2%}",
            "異動原因": "單市場績效分析",
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
            f"【效率報告】績效分析耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, market, timeframe, "績效分析", "normal")
        await push_limiter.retry_cached_messages()

        return performance
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[{market}_{timeframe}] 績效分析失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E501：績效分析失敗 重試{retry_count + 1}/5: {e}", "績效分析錯誤", market, timeframe, "績效分析")
            await asyncio.sleep(5)
            return await calculate_performance(trading_result, market, timeframe, params, custom_metrics, retry_count + 1)
        logger.error(f"[{market}_{timeframe}] 績效分析失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E501：績效分析失敗 {e}", market, timeframe, "績效分析", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E501：績效分析失敗 {e}", "績效分析錯誤", market, timeframe, "績效分析")
        return None

async def multi_market_performance_analysis(multi_trading_results, params_list, retry_count=0):
    """多市場加權績效分析，支持並行計算"""
    try:
        import time
        start_time = time.time()
        performance_data = []
        total_sharpe = 0.0
        valid_markets = 0
        tasks = []
        for (market, timeframe), result in multi_trading_results.items():
            if result is None:
                continue
            params = params_list.get((market, timeframe), params_list.get(("default", "default"), {}))
            task = asyncio.create_task(calculate_performance(result, market, timeframe, params))
            tasks.append((market, timeframe, task))

        for market, timeframe, task in tasks:
            performance = await task
            if performance:
                weight = 0.5 if market == "BTCUSDT" and timeframe == "15m" else 0.5 / (len(市場清單) - 1)
                total_sharpe += performance["sharpe_ratio"] * weight
                valid_markets += 1
                performance_data.append({
                    "市場": market,
                    "時間框架": timeframe,
                    "最終資金": performance["最終資金"],
                    "報酬率": performance["報酬率"],
                    "最大回撤": performance["最大回撤"],
                    "勝率": performance["勝率"],
                    "f1分數": performance["f1分數"],
                    "穩定性": performance["穩定性"],
                    "var_95": performance["var_95"],
                    "cvar_95": performance["cvar_95"],
                    "sharpe_ratio": performance["sharpe_ratio"],
                    "交易次數": performance["交易次數"],
                    "平均持倉時間": performance["平均持倉時間"],
                    "單筆損失": performance["單筆損失"]
                })

        if valid_markets == 0:
            logger.error("[多市場_多框架] 無有效績效數據")
            await push_limiter.cache_message("錯誤碼E501：無有效績效數據", "多市場", "多框架", "績效分析", "high")
            await push_limiter.retry_cached_messages()
            return None

        # 快取多市場績效數據
        cache_path = 快取資料夾 / f"performance_cache_multi_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez_compressed(cache_path, data=performance_data)
        await push_limiter.cache_message(f"【執行通知】多市場績效數據已快取至 {cache_path}", "多市場", "多框架", "績效分析", "normal")
        await push_limiter.retry_cached_messages()

        # 生成多市場報表
        df = pd.DataFrame(performance_data)
        if not df.empty:
            csv_path = SQLite資料夾.parent / "備份" / f"Multi_Market_Performance_Comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            await push_limiter.cache_message(f"【執行通知】生成多市場績效比較報表 {csv_path}", "多市場", "多框架", "績效分析", "normal")
            await push_limiter.retry_cached_messages()

        # 寫入異動歷程
        change_log_path = SQLite資料夾.parent / "dot" / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        change_log = {
            "UUID": str(uuid.uuid4()),
            "市場": "多市場",
            "時間框架": "多框架",
            "異動前值": "N/A",
            "異動後值": f"加權Sharpe={total_sharpe / valid_markets if valid_markets > 0 else 0.0:.2f}, 記錄數={len(performance_data)}",
            "異動原因": "多市場績效分析",
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
            f"【效率報告】多市場績效分析耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, "多市場", "多框架", "績效分析", "normal")
        await push_limiter.retry_cached_messages()

        return {"加權sharpe": total_sharpe / valid_markets if valid_markets > 0 else 0.0, "數據": performance_data}
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[多市場_多框架] 多市場績效分析失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E501：多市場績效分析失敗 重試{retry_count + 1}/5: {e}", "多市場績效錯誤", "多市場", "多框架", "績效分析")
            await asyncio.sleep(5)
            return await multi_market_performance_analysis(multi_trading_results, params_list, retry_count + 1)
        logger.error(f"[多市場_多框架] 多市場績效分析失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E501：多市場績效分析失敗 {e}", "多市場", "多框架", "績效分析", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E501：多市場績效分析失敗 {e}", "多市場績效錯誤", "多市場", "多框架", "績效分析")
        return None

async def parameter_sensitivity_analysis(trading_results, params_list, filter_market=None, filter_timeframe=None, custom_metrics=None, retry_count=0):
    """參數敏感性分析，支持自定義指標"""
    try:
        import time
        start_time = time.time()
        sensitivity_data = []
        for (market, timeframe), result in trading_results.items():
            if result is None:
                continue
            if filter_market and market != filter_market:
                continue
            if filter_timeframe and timeframe != filter_timeframe:
                continue
            params = params_list.get((market, timeframe), params_list.get(("default", "default"), {}))
            performance = await calculate_performance(result, market, timeframe, params, custom_metrics)
            if performance:
                weight = 0.5 if market == "BTCUSDT" and timeframe == "15m" else 0.5 / (len(市場清單) - 1)
                for param_name, param_value in params.items():
                    sensitivity_entry = {
                        "市場": market,
                        "時間框架": timeframe,
                        "參數名稱": param_name,
                        "參數值": param_value,
                        "報酬率": performance["報酬率"] * weight,
                        "f1分數": performance["f1分數"] * weight,
                        "var_95": performance["var_95"] * weight,
                        "cvar_95": performance["cvar_95"] * weight,
                        "sharpe_ratio": performance["sharpe_ratio"] * weight,
                        "穩定性": performance["穩定性"] * weight
                    }
                    if custom_metrics:
                        for metric_name in custom_metrics.keys():
                            sensitivity_entry[metric_name] = performance.get(metric_name, 0.0) * weight
                    sensitivity_data.append(sensitivity_entry)

        if not sensitivity_data:
            logger.error("[多市場_多框架] 無有效敏感性數據")
            await push_limiter.cache_message("錯誤碼E501：無有效敏感性數據", "多市場", "多框架", "績效分析", "high")
            await push_limiter.retry_cached_messages()
            return None

        # 生成雙軸敏感性熱圖
        df = pd.DataFrame(sensitivity_data)
        for metric in ["var_95", "cvar_95"] + (list(custom_metrics.keys()) if custom_metrics else []):
            pivot = df.pivot_table(values=metric, index="參數名稱", columns="參數值", aggfunc="mean")
            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="Viridis" if metric in ["var_95", "cvar_95"] else "Plasma",
                name=metric,
                showscale=True
            ))
            fig.add_trace(go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="Viridis" if metric in ["var_95", "cvar_95"] else "Plasma",
                name=f"{metric} (右軸)",
                zaxis="z2",
                showscale=False
            ))
            fig.update_layout(
                title=f"參數對{metric}影響熱圖",
                xaxis_title="參數值",
                yaxis_title="參數名稱",
                template="plotly_dark",
                height=600,
                showlegend=True,
                updatemenus=[
                    dict(
                        buttons=[
                            dict(label="全選", method="update", args=[{"visible": [True] * len(fig.data)}]),
                            dict(label="僅主指標", method="update", args=[{"visible": [i % 2 == 0 for i in range(len(fig.data))]}])
                        ],
                        direction="down",
                        showactive=True
                    )
                ]
            )
            heatmap_path = SQLite資料夾.parent / "圖片" / f"Parameter_Sensitivity_{metric}_Heatmap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            fig.write_html(heatmap_path)
            await push_limiter.cache_message(f"【執行通知】生成參數敏感性熱圖 {heatmap_path}", "多市場", "多框架", "績效分析", "normal")
            await push_limiter.retry_cached_messages()

        # 加密並儲存敏感性數據
        encrypted_data = await encrypt_performance_data(sensitivity_data)
        if not encrypted_data:
            return None
        async with aiosqlite.connect(SQLite資料夾 / "performance_log.db") as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS 敏感性記錄 (
                    id TEXT PRIMARY KEY,
                    市場 TEXT,
                    時間框架 TEXT,
                    參數名稱 TEXT,
                    參數值 REAL,
                    報酬率 REAL,
                    f1分數 REAL,
                    var_95 REAL,
                    cvar_95 REAL,
                    sharpe_ratio REAL,
                    穩定性 REAL,
                    時間 TEXT
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_市場_時間框架 ON 敏感性記錄 (市場, 時間框架, 時間)")
            for data in sensitivity_data:
                await conn.execute("INSERT INTO 敏感性記錄 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                 (str(uuid.uuid4()), data["市場"], data["時間框架"], data["參數名稱"], data["參數值"],
                                  data["報酬率"], data["f1分數"], data["var_95"], data["cvar_95"], data["sharpe_ratio"],
                                  data["穩定性"], datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            await conn.commit()

        # 快取敏感性數據
        cache_path = 快取資料夾 / f"sensitivity_cache_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez_compressed(cache_path, data=sensitivity_data)
        await push_limiter.cache_message(f"【執行通知】敏感性數據已快取至 {cache_path}", "多市場", "多框架", "績效分析", "normal")
        await push_limiter.retry_cached_messages()

        # 寫入異動歷程
        change_log_path = SQLite資料夾.parent / "dot" / f"異動歷程_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        change_log = {
            "UUID": str(uuid.uuid4()),
            "市場": "多市場",
            "時間框架": "多框架",
            "異動前值": "N/A",
            "異動後值": f"熱圖生成={heatmap_path}, 記錄數={len(sensitivity_data)}",
            "異動原因": "參數敏感性分析",
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
            f"【效率報告】參數敏感性分析耗時：{elapsed_time:.2f}秒，"
            f"CPU：{cpu_percent:.1f}%，RAM：{ram_percent:.1f}%，"
            f"GPU：{gpu_util:.1f}%"
        )
        logger.info(efficiency_report)
        await push_limiter.cache_message(efficiency_report, "多市場", "多框架", "績效分析", "normal")
        await push_limiter.retry_cached_messages()

        return sensitivity_data
    except Exception as e:
        if retry_count < 5:
            logger.warning(f"[多市場_多框架] 參數敏感性分析失敗，重試 {retry_count + 1}/5: {e}")
            await 錯誤記錄與自動修復(f"錯誤碼E501：參數敏感性分析失敗 重試{retry_count + 1}/5: {e}", "敏感性分析錯誤", "多市場", "多框架", "績效分析")
            await asyncio.sleep(5)
            return await parameter_sensitivity_analysis(trading_results, params_list, filter_market, filter_timeframe, custom_metrics, retry_count + 1)
        logger.error(f"[多市場_多框架] 參數敏感性分析失敗，重試 5 次無效: {e}")
        await push_limiter.cache_message(f"錯誤碼E501：參數敏感性分析失敗 {e}", "多市場", "多框架", "績效分析", "high")
        await push_limiter.retry_cached_messages()
        await 錯誤記錄與自動修復(f"錯誤碼E501：參數敏感性分析失敗 {e}", "敏感性分析錯誤", "多市場", "多框架", "績效分析")
        return None