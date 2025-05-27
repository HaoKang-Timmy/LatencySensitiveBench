from datetime import datetime
import os
from get_simulation_config import get_simulation_config
from Agent.agent import AgentConfig, TradingAgent
from Environment import TradeMarket

if __name__ == "__main__":
    SIM_CONFIG = get_simulation_config()

    stock_names = SIM_CONFIG["stock_names"]
    date_str = SIM_CONFIG["date_str"]
    json_dir = SIM_CONFIG["json_dir"]
    json_files = [os.path.join(json_dir, f"ohlc_{stock}_{date_str}.json") for stock in stock_names]

    env = TradeMarket(
        json_files=json_files,
        stock_names=stock_names,
        step_size_sec=SIM_CONFIG["step_size_sec"],
        history_points=SIM_CONFIG["history_points"],
        history_min_gap_sec=SIM_CONFIG["history_min_gap_sec"]
    )

    agent_configs = [
        AgentConfig(
            model_name=SIM_CONFIG["model_names"][i],
            max_input_len=SIM_CONFIG["max_input_len"],
            max_new_tokens=SIM_CONFIG["max_new_tokens"],
            temperature=SIM_CONFIG["temperature"],
            top_p=SIM_CONFIG["top_p"],
            top_k=SIM_CONFIG["top_k"],
            do_sample=SIM_CONFIG["do_sample"],
            serve_type=SIM_CONFIG["serve_type"],
            device=SIM_CONFIG["device_list"][i],
            profit_threshold=SIM_CONFIG["profit_threshold"],
            cash=SIM_CONFIG["cash"],
            port=SIM_CONFIG["port"],
            hostname=SIM_CONFIG["hostname"]
        )
        for i in range(SIM_CONFIG["agent_count"])
    ]

    # agents = [TradingAgent(config=cfg, stock_list=stock_names) for cfg in agent_configs]
    agents = []
    for i, cfg in enumerate(agent_configs):
        agents.append(TradingAgent(config=cfg, stock_list=stock_names))
    for agent in agents:
        agent.env = env

    print(f"[🚀] Starting simulation at {datetime.now().strftime('%H:%M:%S')}")
    env.run_simulation(agents)
    print(f"[✅] Simulation completed at {datetime.now().strftime('%H:%M:%S')}")
