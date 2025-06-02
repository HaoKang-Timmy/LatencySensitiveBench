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
## How to serve agents with vllm and sglang
Run two models in sglang/vllm. See `./server_launch_player1.py` and `./server_launch_player2.py`.
Then start `./test.zsh` to start the competition

# How to self define agent and test
The function calling is at `call_llm_local` in `./GameEngine/Agent/robot.py`.

The model initialization of vllm and sglang is outside of the gaming, but for huggingface it is in `init_local_model` of `./GameEngine/Agent/robot.py`.

I suggest set `--serving-choice` to `huggingface` and change the two function above to design your own agent. But remember, if the initialization of agent takes more than 1 min, testing results can be unstable. 