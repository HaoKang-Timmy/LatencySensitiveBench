# LSB: Latency Sensitive Benchmarks for LLM Agents.
**[Paper (Win Fast or Lose Slow)](https://arxiv.org/abs/2505.19481) | [Website(Competitve Agents)](https://haokang-timmy.github.io/CompetitiveAgents/)**
<p align="left"><img width="100%" src="./Figs/FiguresTogether.png"></p><br/>
<!-- <p align="right"><img width="40%" src="./Figs/TimeSensitiveTask.png"></p><br/>
<div style="display: flex; justify-content: center; align-items: center; gap: 2%;">
  <img src="./Figs/CompetitiveAgents.png" style="max-height: 200px; height: auto; width: auto;">
  <img src="./Figs/TimeSensitiveTask.png" style="max-height: 200px; height: auto; width: auto;">
</div>
<div style="width: 80%; margin: auto; display: flex; justify-content: center; align-items: center; gap: 2%;">
  <img src="./Figs/CompetitiveAgents.png" style="max-height: 200px; height: auto; width: auto;">
  <img src="./Figs/TimeSensitiveTask.png" style="max-height: 200px; height: auto; width: auto;">
</div> -->

Latency Sensitive Benchmarks (LSB) are specifically designed to evaluate LLM Agents in realistic, latency-sensitive scenarios such as competitive games and high-frequency trading. In these tasks, **both latency and accuracy** jointly determine the final reward (e.g., game win rate or trading yield). Unlike previous benchmarks, LSB introduces two novel tasks that not only assess the intelligence of LLM agents, but also rigorously evaluate the efficiency of the underlying serving systems and algorithms. By integrating latency, accuracy, and real-world reward into a unified framework, LSB pioneers a new direction for benchmarking—encouraging the development of efficient, adaptive, and latency-aware LLM systems and algorithms. We hope our benchmarks and findings inspire the community to move beyond accuracy-centric evaluation and to build LLM solutions that truly excel in real-world, time-critical applications. We invite you to try LSB and join us in advancing this exciting frontier!
## Key Features


- **Diverse Benchmarks:** LSB offers two cutting-edge benchmarks， competitive gaming (StreetFighter) and high frequency trading backtesting system, capturing the essence of real-world, latency-sensitive tasks.
- **Flexible Agent Deployment:** Provides LLM agent implementations that support local, remote, and API-based serving, enabling comprehensive evaluation across different system architectures.
- **System-Aware Evaluation:** Highlights how agent performance varies with different serving systems and hardware configurations, offering actionable insights for both algorithm and system optimization.

Experience how LSB can help you benchmark and improve your LLM agents in truly challenging, real-time environments!
## Visualization
Competitive Games: Faster one wins!!!!
https://github.com/user-attachments/assets/c69571df-f109-4e92-9e60-60a9cd7933f2
## Contents
- [Installation](#installation)
- [QuickStart](#quick-start)
- [Evaluation&Trade-off](#evaluation)
- [Self-define Agent](#self-define-agent)
- [References](#References)

## Installation
### StreetFighter
1. Diambra
```
pip install diambra diambra-arena
```
2. Install [huggingface](https://huggingface.co/docs/transformers/installation#:~:text=pip%20install%20transformers), [vllm](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html), [sglang](https://docs.sglang.ai/start/install.html).

3. Install other relavant envs
```
pip install loguru llama_index dotenv gymnasium rich openai
```
4. Register your diambra account at [here](https://www.diambra.ai/)
5. Install StreetFighter kernel at [here](https://wowroms.com/en/roms/mame/street-fighter-iii-3rd-strike-fight-for-the-futur-japan-clone/106255.html). And put the zip file(do not unzip it) at $GAME_PATH(wherever you like).
### HFTBench
Install [huggingface](https://huggingface.co/docs/transformers/installation#:~:text=pip%20install%20transformers), [vllm](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html), [sglang](https://docs.sglang.ai/start/install.html).
Install other relavant envs
```
pip install loguru llama_index dotenv gymnasium rich openai
```

## Quick Start
### StreetFighter
change $GAME_PATH to the root path of where you put the zip file.
```
python3 diambra -r $GAME_PATH --serving-choice huggingface --agent1 Qwen/Qwen3-4B --agent2 Qwen/Qwen3-8B --logdir "test.log" --device1 cuda:0 --device2 cuda:1
```
### HFTBench
```
python3 Simulation.py --agent_count 1 --device_list cuda:0 
```
## Evaluation
Here we provide results on two RTX5090. More results on H100 are comming soon.
### StreetFighter

## Self-define Agent


sudo yum install libsndfile1