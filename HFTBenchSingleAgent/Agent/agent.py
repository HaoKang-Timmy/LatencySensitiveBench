from dataclasses import dataclass
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from vllm import SamplingParams, LLM
from dataclasses import dataclass
import torch
import time

@dataclass
class AgentConfig:
    model_name: str
    max_input_len: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    do_sample: bool
    device: str
    profit_threshold: float
    cash: int
    port: int
    hostname: str
    serve_type: str


class TradingAgent:
    def __init__(self, config: AgentConfig, stock_list: list):
        self.config = config
        self.stock_list = stock_list
        self.cash = 10000
        self.holdings = {s: {"shares": 0, "avg_price": 0.0} for s in stock_list}
        self.last_decision_time = None
        # self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.serve_type = config.serve_type
    def init_model(self):
        if self.serve_type == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name).to(self.config.device)
        elif self.serve_type == "vllm" or self.serve_type == "sglang":
            from openai import OpenAI
            self.client = OpenAI(
                base_url=f"{self.config.hostname}:{self.config.port}/v1",
                api_key="None",
            )
            

    def should_call_llm(self, prices: dict):
        for stock in self.stock_list:
            shares = self.holdings[stock]["shares"]
            avg_price = self.holdings[stock]["avg_price"]
            price = prices.get(stock, {})
            high = price.get("sell")
            low = price.get("buy")
            if shares == 0:
                return True
            if high and low:
                diff = max(abs(high - avg_price), abs(low - avg_price)) / avg_price * 100
                if diff >= self.config.profit_threshold:
                    return True
        return False

    def get_history_stats(self, dt: datetime, num_points: int = None):
        if num_points is None:
            num_points = self.env.history_points
        min_gap = self.env.step_size if not hasattr(self.env, "history_min_gap_sec") else self.env.history_min_gap_sec
        history = {}

        for stock in self.stock_list:
            history[stock] = []
            last_added_time = None

            date_str = dt.strftime("%Y-%m-%d")
            market_data = self.env.market_data[stock].get(date_str, {})
            # 转换为 datetime 列表
            timestamps = sorted([
                datetime.strptime(f"{date_str} {ts}", "%Y-%m-%d %H:%M:%S")
                for ts in market_data.keys()
                if datetime.strptime(f"{date_str} {ts}", "%Y-%m-%d %H:%M:%S") < dt
            ], reverse=True)

            for t in timestamps:
                if last_added_time is None or (last_added_time - t).total_seconds() >= min_gap:
                    entry = market_data.get(t.strftime("%H:%M:%S"))
                    if entry:
                        avg = (entry["low"] + entry["high"]) / 2
                        history[stock].append((t.strftime("%H:%M:%S"), {"price": avg}))
                        last_added_time = t
                if len(history[stock]) >= num_points:
                    break

        return history

    def get_max_quantity(self, prices: dict):
        max_q = {}
        for stock in self.stock_list:
            price = prices.get(stock, {}).get("buy")
            if price and price > 0:
                max_q[stock] = int(self.cash // price)
            else:
                max_q[stock] = 0
        return max_q

    def construct_prompt(self, dt: datetime, prices: dict, history_stats: dict, max_quantity: dict) -> dict:
        system_prompt = (
            "You are a high-frequency trading agent. Your task is to make optimal buy/sell decisions "
            "for each stock based on current and recent market signals, holdings, and constraints. "
            "You must close all positions before 15:30:00. Maximize profit before that time.\n\n"
            "Trading Guide:\n"
            "- Your goal is to maximize profit **within the same day**.\n"
            "- Use recent historical average price trends to infer short-term movements.\n"
            "- If the average price has been rising and current price is still low, consider buying.\n"
            "- If you already hold stocks and the current sell price is higher than your average cost, consider selling to lock profit.\n"
            "- ⚠️ If the difference between the current sell and buy price is large, it indicates a rare profit opportunity — do not miss it.\n"
            "- Avoid buying or selling too much at once unless profit signals are strong. Keep trades proportional to your cash.\n"
            "- If you've already made sufficient profit today, it's okay to **avoid further trades** to secure gains.\n"
            "- Always prepare to exit all positions by 15:30:00.\n"
            "- You may skip trading if there's no clear advantage."
        )

        time_left = (datetime.strptime("15:30:00", "%H:%M:%S") - dt).seconds
        user_prompt = (
            f"Current simulation time: {dt.strftime('%H:%M:%S')}\n"
            f"Time left before close: {time_left} seconds\n"
            f"Available cash: ${self.cash:.2f}\n\n"
        )

        for stock in self.stock_list:
            hold = self.holdings[stock]
            current = prices[stock]
            buy_price = current.get('buy')
            sell_price = current.get('sell')

            if buy_price is None or sell_price is None:
                continue  # skip this stock

            avg_price = hold['avg_price']
            shares = hold['shares']

            buy_profit = ((avg_price - buy_price) / avg_price * 100) if avg_price > 0 else 0
            sell_profit = ((sell_price - avg_price) / avg_price * 100) if avg_price > 0 else 0

            user_prompt += (
                f"[{stock}]\n"
                f"  Current Buy: ${buy_price:.2f}, Buy Profit Margin: {buy_profit:.2f}%\n"
                f"  Current Sell: ${sell_price:.2f}, Sell Profit Margin: {sell_profit:.2f}%\n"
                f"  Holdings: {shares} shares @ ${avg_price:.2f}\n"
                f"  Max Buyable Qty: {max_quantity[stock]}\n\n"
            )

            user_prompt += "  Historical (time-ordered):\n"
            for t, stats in history_stats[stock]:
                price = stats["price"]
                if avg_price > 0:
                    margin = (price - avg_price) / avg_price * 100
                    user_prompt += f"    {t} → Avg = ${price:.2f}, Margin = {margin:.2f}%\n"
                else:
                    user_prompt += f"    {t} → Avg = ${price:.2f}\n"
            user_prompt += "\n"

        example = "{\n" + ",\n".join([f'  \"{s}\": XX' for s in self.stock_list]) + "\n}"
        user_prompt += (
            "Please respond ONLY with a dict like the following:\n"
            f"{example}\n"
        )
        
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    def run_hf_inference(self, prompt_text: str) -> str:
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.config.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
            )
        # Remove prefill: only return newly generated text
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
    def run_client_inference(self, prompt_text: str) -> str:
        # sampling_decode = SamplingParams(max_tokens=self.config.max_new_tokens)
        # response = self.model.generate(
        #     prompt_text,
        #     sampling_params = sampling_decode,
        # )
        # return response[0].outputs[0].text

        if "Qwen3" in self.config.model_name:
            extra_body = {
                "max_tokens": self.config.max_new_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        else:
            extra_body = {
                "max_tokens": self.config.max_new_tokens,
            }
        completion = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=prompt_text,
                stream=False,
                extra_body=extra_body
            )
        text = completion.choices[0].message.content
        return text
    def extract_trade_decision(self, response: str) -> dict:
        decisions = {s: 0 for s in self.stock_list}  # default to 0

        for stock in self.stock_list:
            # Look for '"AAPL": 10' or "AAPL": -5 (possibly with spaces)
            match = re.search(rf'"{stock}"\s*:\s*(-?\d+)', response)
            if match:
                try:
                    decisions[stock] = int(match.group(1))
                except:
                    decisions[stock] = 0  # fallback to 0 if malformed number
            else:
                decisions[stock] = 0  # fallback if not matched

        return decisions

    def inference(self, text: str):

        if self.serve_type == "huggingface":
            text = self.tokenizer.apply_chat_template(text)
            infernce_result = self.run_hf_inference(text)
        elif self.serve_type == "vllm" or self.serve_type == "sglang":
            infernce_result = self.run_client_inference(text)
        extracted_decision = self.extract_trade_decision(infernce_result)

        return extracted_decision
    def decide_trades(self, prices: dict):
        dt = self.env.sim_time
        history_stats = self.get_history_stats(dt)
        max_quantity = self.get_max_quantity(prices)
        text = self.construct_prompt(dt, prices, history_stats, max_quantity)
        # text = self.tokenizer.apply_chat_template(prompt)
        start = time.time()
        decisions = self.inference(text)
        torch.cuda.synchronize()
        end = time.time()
        # print(prompt["user"])  # Debug only
        # return {s: 0 for s in self.stock_list}  # placeholder
        return decisions, end - start

    def apply_trade(self, decisions: dict, env, dt: datetime, delay):
        price_view = env.get_current_price(dt, delay)
        env.trade_count[dt] += 1
        for stock, delta in decisions.items():
            p = price_view.get(stock, {})
            buy_price = p.get("buy")
            sell_price = p.get("sell")
            if delta > 0 and buy_price:
                cost = delta * buy_price
                if cost <= self.cash:
                    prev = self.holdings[stock]
                    total_cost = prev["shares"] * prev["avg_price"] + cost
                    new_shares = prev["shares"] + delta
                    new_avg = total_cost / new_shares
                    self.holdings[stock] = {"shares": new_shares, "avg_price": new_avg}
                    self.cash -= cost
            elif delta < 0 and sell_price:
                to_sell = min(-delta, self.holdings[stock]["shares"])
                self.cash += to_sell * sell_price
                self.holdings[stock]["shares"] -= to_sell
        self.last_decision_time = dt
