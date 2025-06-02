# GAME_PATH="/fsx-alignment/dedicated-fsx-data-repo-alignment-us-west-2/home/zenhk/.game"


# # diambra -r "$GAME_PATH" run -l python3 run_api.py --serving-choice api --agent1 openai:gpt-4o-mini --agent2 openai:gpt-4o-mini 





# diambra -r "$GAME_PATH" run -l python3 run_api.py --serving-choice sglang --agent1 Qwen/Qwen3-4B --agent2 Qwen/Qwen3-8B --port1 8002 --port2 8003


GAME_PATH="/fsx-alignment/dedicated-fsx-data-repo-alignment-us-west-2/home/zenhk/.game"


# for i in {1..40}; do
#     echo "===== Running experiment $i ====="
#     diambra -r "$GAME_PATH" run -l python3 run_api.py \
#         --serving-choice sglang \
#         --agent1 Qwen/Qwen3-4B \
#         --agent2 Qwen/Qwen3-4B \
#         --port1 8002 \
#         --port2 8004 \
#         --logdir "4vs4_8bit.log"

#     echo "===== Finished experiment $i ====="
#     echo ""
# done



for i in {1..40}; do
    echo "===== Running experiment $i ====="
    diambra -r "$GAME_PATH" run -l python3 run_api.py \
        --serving-choice sglang \
        --agent1 Qwen/Qwen3-8B \
        --agent2 Qwen/Qwen3-8B \
        --port1 8002 \
        --port2 8003 \
        --logdir "8vs8_8bit.log"

    echo "===== Finished experiment $i ====="
    echo ""
done

for i in {1..40}; do
    echo "===== Running experiment $i ====="
    diambra -r "$GAME_PATH" run -l python3 run_api.py \
        --serving-choice sglang \
        --agent1 Qwen/Qwen3-14B \
        --agent2 Qwen/Qwen3-14B \
        --port1 8004 \
        --port2 8005 \
        --logdir "14vs14_8bit.log"

    echo "===== Finished experiment $i ====="
    echo ""
done


for i in {1..40}; do
    echo "===== Running experiment $i ====="
    diambra -r "$GAME_PATH" run -l python3 run_api.py \
        --serving-choice sglang \
        --agent1 Qwen/Qwen3-8B \
        --agent2 Qwen/Qwen3-14B \
        --port1 8002 \
        --port2 8003 \
        --logdir "4vs14.log"

    echo "===== Finished experiment $i ====="
    echo ""
done
