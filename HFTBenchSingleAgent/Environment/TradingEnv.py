from datetime import datetime, timedelta
from collections import defaultdict
import json
from multiprocessing import Process, Manager

class TradeMarket:
    def __init__(
        self,
        json_files,
        stock_names,
        step_size_sec: int = 1, ### minimal observation/trading time
        history_points: int = 20, ### history points for each stock
        history_min_gap_sec: int = 60, ### minimal gap between two history points
        decay_window: float = 1.5 ### decay window for linear decay
    ):
        self.market_data = {}
        self.agents_trade_in_second = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.sim_time = None
        self.real_time = None
        self.trade_count = defaultdict(int)
        self.step_size = step_size_sec
        self.history_points = history_points
        self.history_min_gap_sec = history_min_gap_sec
        self.current_date = None
        self.time_cursor = None
        self.trade_time_delay = []
        self.daily_avg_prices = {}  # store the daily average price of each stock
        self.decay_window = decay_window # 2 second for linear decay

        stock_times = []

        for file_path, stock_name in zip(json_files, stock_names):
            with open(file_path, "r") as f:
                raw_data = json.load(f)
            self.market_data[stock_name] = {}
            self.daily_avg_prices[stock_name] = {}
            stock_time_set = set()

            for date, time_data in raw_data.items():
                self.market_data[stock_name][date] = {}
                daily_highs = []
                daily_lows = []
                
                for timestamp, ohlc in time_data.items():
                    self.market_data[stock_name][date][timestamp] = {
                        "high": ohlc["high"],
                        "low": ohlc["low"]
                    }
                    daily_highs.append(ohlc["high"])
                    daily_lows.append(ohlc["low"])
                    dt = datetime.strptime(f"{date} {timestamp}", "%Y-%m-%d %H:%M:%S")
                    stock_time_set.add(dt)
                
                # calculate the daily average price
                if daily_highs and daily_lows:
                    avg_high = sum(daily_highs) / len(daily_highs)
                    avg_low = sum(daily_lows) / len(daily_lows)
                    self.daily_avg_prices[stock_name] = (avg_high + avg_low) / 2
                    

            stock_times.append(stock_time_set)

        # Construct the common timestamps for all stocks
        common_times = set.intersection(*stock_times)
        self.common_timestamps = sorted(list(common_times))
        self.current_index = 0

        if self.common_timestamps:
            self.current_date = self.common_timestamps[0].strftime("%Y-%m-%d")
            self.time_cursor = self.common_timestamps[0]
        else:
            raise ValueError("No overlapping timestamps across stocks.")

    def _apply_linear_decay(self, high, low, delay, stock_name):
        avg = self.daily_avg_prices[stock_name]
        
        # 如果延迟大于等于decay_window(2秒)，价格完全收敛到平均价格
        if delay >= self.decay_window:
            return avg, avg
        
        # 如果延迟为0，返回原始价格
        if delay <= 0:
            return high, low
        
        # 线性插值：delay在0到decay_window之间时的线性变化
        # alpha = 0 时保持原价格，alpha = 1 时完全变为平均价格
        alpha = delay / self.decay_window
        
        new_high = high * (1 - alpha) + avg * alpha
        new_low = low * (1 - alpha) + avg * alpha
        
        return new_high, new_low

    def get_current_price(self, dt: datetime, delay = 0):
        date_str = dt.strftime("%Y-%m-%d")
        sec_str = dt.strftime("%H:%M:%S")
        price_dict = {}
        for stock, data in self.market_data.items():
            if sec_str not in data[date_str]:
                price_dict[stock] = {"buy": None, "sell": None}
                continue
            entry = data[date_str][sec_str]
            high, low = entry["high"], entry["low"]
            
            # 应用线性衰减
            if delay > 0:
                adj_high, adj_low = self._apply_linear_decay(high, low, delay, stock_name = stock)
                price_dict[stock] = {"buy": adj_low, "sell": adj_high}
            else:
                price_dict[stock] = {"buy": low, "sell": high}
            
        return price_dict

    def get_next_time(self):
        while self.current_index < len(self.common_timestamps):
            next_time = self.common_timestamps[self.current_index]
            if self.sim_time is None or (next_time - self.sim_time).total_seconds() >= self.step_size:
                if self.sim_time is not None and len(self.trade_time_delay) > 0:
                    if next_time - self.sim_time < timedelta(seconds=self.trade_time_delay[0]):
                        ### trade delay is not over, skip this time step
                        self.current_index += 1
                        continue
                self.time_cursor = next_time
                self.sim_time = next_time
                self.current_index += 1
                return next_time
            self.current_index += 1
        return None  # simulation end

    def run_simulation(self, agents: list):
        print("[⏳] Waiting for all agents to load...")
        for a in agents:
            a.init_model()
        print("[✅] All agents loaded. Starting simulation.")

        while True:
            dt = self.get_next_time()
            if dt is None:
                print("[🔚] Reached end of timestamp list.")
                break

            self.trade_count[dt] = 0
            prices_snapshot = self.get_current_price(dt)
            triggering_agents = [a for a in agents if a.should_call_llm(prices_snapshot)]
            self.agents_trade_in_second[dt] = len(triggering_agents)

            if not triggering_agents:
                continue

            self.real_time = datetime.now()
            ####TODO currently only one agent is allowed to trade at a time
            decisions, delay = agents[0].decide_trades(prices_snapshot)
            agents[0].apply_trade(decisions, self, dt, delay)

            if self.time_cursor.strftime("%H:%M:%S") > "15:30:00":
                print("[🕒] Simulation cutoff reached at 15:30.")
                break

    def get_daily_avg_prices(self, stock_name=None, date=None):
        """
        获取每日平均价格数据
        
        Args:
            stock_name: 股票名称，如果为None则返回所有股票
            date: 日期字符串(YYYY-MM-DD)，如果为None则返回所有日期
            
        Returns:
            dict: 包含平均价格信息的字典
        """
        if stock_name is None:
            if date is None:
                return self.daily_avg_prices
            else:
                return {stock: prices.get(date, {}) for stock, prices in self.daily_avg_prices.items()}
        else:
            if date is None:
                return self.daily_avg_prices.get(stock_name, {})
            else:
                return self.daily_avg_prices.get(stock_name, {}).get(date, {})
    
    def print_daily_summary(self):
        """打印所有股票的每日平均价格汇总"""
        print("\n[📈] 每日平均价格汇总:")
        print("=" * 80)
        for stock_name, dates_data in self.daily_avg_prices.items():
            print(f"\n股票: {stock_name}")
            print("-" * 60)
            for date, price_data in dates_data.items():
                print(f"  {date}: 平均最高价={price_data['avg_high']:.4f}, "
                      f"平均最低价={price_data['avg_low']:.4f}, "
                      f"日均价={price_data['avg_price']:.4f}")
        print("=" * 80)
