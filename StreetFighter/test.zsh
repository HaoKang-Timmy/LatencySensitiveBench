GAME_PATH="/fsx-alignment/dedicated-fsx-data-repo-alignment-us-west-2/home/zenhk/.game"


# diambra -r "$GAME_PATH" run -l python3 run_api.py --serving-choice api --agent1 openai:gpt-4o-mini --agent2 openai:gpt-4o-mini 





diambra -r "$GAME_PATH" run -l python3 run_api.py --serving-choice sglang --agent1 Qwen/Qwen3-8B --agent2 Qwen/Qwen3-4B --port1 8002 --port2 8003