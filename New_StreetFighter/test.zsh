GAME_PATH="/home/zenhk/.game"


diambra -r "$GAME_PATH" run -l python3 run_api.py --serving-choice api --agent1 openai:gpt-4o-mini --agent2 openai:gpt-4o-mini 





diambra -r /home/zenhk/.game run -l python3 run_api.py --serving-choice vllm --agent1 Qwen/Qwen3-4B --agent2 Qwen/Qwen3-4B