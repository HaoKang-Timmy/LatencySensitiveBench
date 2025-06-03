# Frontend-DiambraEngine
Diambra platform is a reinforcement learning environment. See [here](https://docs.diambra.ai/) for more details and implementation of other games

`./runapi.py` starts the game simulation environments in docker.

`./GameEngine/Engine/game.py` initialize the agent and define the game loop.

`./GameEngine/Agent/player.py` a upper class that contains agent and other important items.

`./GameEngine/Agent/robot.py` define the agent action, observation and other functions.

The agent are executed in `multiprocess` in python. Different from `Thread`, it allows true parallelizations among players.

# Backend- LLM Agent design and serving
Here we support three kinds of LLM model serving, huggingface transformer, vllm and sglang

`./GameEngine/Agent/prompt.py` designs the few shot prompt capable for most of opensource models.

`./GameEngine/Agent/llm_serving.py` supports remote api calls

`./GameEngine/Agent/agent.py` init the client and models for LLM agent serving.

# Profile mutiple agent elo rankings
## Serve agents with vllm and sglang

Run two models in sglang/vllm. See `./server_launch_player1.py` and `./server_launch_player2.py`.
Then start `./test.zsh` to start the competition.

See [vllm](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html) and [sglang](https://docs.sglang.ai/backend/openai_api_completions.html) for more details.
## Generate ranking
Our ranking is based on bradley-terry model.

First run `generate_csv.py`, then run `generate_bradley_terry.py`.

## Results on H100

We also provide Bradley-Terry model strength rankings based on comprehensive battle results. The Bradley-Terry model uses maximum likelihood estimation to determine optimal rankings from pairwise comparisons. We are using Qwen3 model zoo.

| Rank | Model Size | Bitwidth | Bradley-Terry Score | Win-Loss Record |
|------|------------|----------|-------------------|-----------------|
| 1    | 14B_8bit   | 8        | **0.824**         | 25W-15L         |
| 2    | 32       | 16       | **0.549**         | 30W-10L         |
| 3    | 14B        | 16       | **0.315**         | 80W-59L         |
| 4    | 4B_8bit    | 8        | **0.299**         | 32W-8L          |
| 5    | 32B        | 16       | **0.114**         | 18W-22L         |
| 6    | 8B         | 16       | **0.025**         | 58W-34L         |
| 7    | 8B_8bit    | 8        | **-0.487**        | 15W-25L         |
| 8    | 32B_8bit  | 8        | **-0.549**        | 10W-30L         |
| 9    | 4B         | 16       | **-1.089**        | 24W-89L         |

**Note:** Higher Bradley-Terry scores indicate stronger models. The score represents the log-odds of winning against an average opponent. Models with positive scores are above average, while negative scores indicate below-average performance.

# Self define agent
The function calling is at `call_llm_local` in `./GameEngine/Agent/robot.py`.

The model initialization of vllm and sglang is outside of the gaming, but for huggingface it is in `init_local_model` of `./GameEngine/Agent/robot.py`.

I suggest set `--serving-choice` to `huggingface` and change the two function above to design your own agent. But remember, if the initialization of agent takes more than 1 min, testing results can be unstable. 