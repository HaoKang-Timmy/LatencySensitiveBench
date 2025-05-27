
import argparse

def get_simulation_config():
    parser = argparse.ArgumentParser(description="Simulation configuration for multi-agent trading.")

    parser.add_argument("--history_min_gap_sec", type=int, default=60)
    parser.add_argument("--step_size_sec", type=int, default=1)
    parser.add_argument("--history_points", type=int, default=20)
    parser.add_argument("--model_names", type=str, nargs="+", help="List of model names, one per agent.")
    parser.add_argument("--max_input_len", type=int, default=2000)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--cash", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--serve_type", type=str, default="huggingface", choices=["huggingface", "vllm", "sglang"])
    parser.add_argument("--profit_threshold", type=float, default=0.1)
    parser.add_argument("--stock_names", type=str, nargs="+", default=["AMZN", "NVDA"])
    parser.add_argument("--date_str", type=str, default="2024-08-05")
    parser.add_argument("--json_dir", type=str, default="./data/")
    parser.add_argument("--agent_count", type=int, required=True)
    parser.add_argument("--device_list", type=str, nargs="+", help="List of CUDA devices, one per agent.", default=["cuda:0", "cuda:1"])
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--hostname", type=str, default="http://localhost")

    args = parser.parse_args()
    config = vars(args)

    assert len(config["model_names"]) == config["agent_count"], "Number of model names must equal agent count"
    assert len(config["device_list"]) == config["agent_count"], "Number of devices must equal agent count"

    return config
