import datetime
import os
import random
import traceback
from threading import Thread
from typing import List, Optional
from loguru import logger

from .robot import Robot, TextRobot
from .observer import *
from .socket_config import SocketConfig
from .share_episode import Episode
import time



class Player:
    nickname: str
    model: str
    robot: Optional[Robot] = None
    serving_method: str
    temperature: float = 0.7
    device: str = "cuda"


    def verify_provider_name(self):
        if self.model.startswith("openai"):
            assert (
                os.environ.get("OPENAI_API_KEY") is not None
            ), "OpenAI API key not set"
        if self.model.startswith("mistral"):
            assert (
                os.environ.get("MISTRAL_API_KEY") is not None
            ), "Mistral API key not set"
        if self.model.startswith("cerebras"):
            assert (
                os.environ.get("CEREBRAS_API_KEY") is not None
            ), "Cerebras API key not set"
        if self.model.startswith("remote"):
            return
        
        
        
class Player1(Player):
    def __init__(
        self,
        nickname: str,
        model: str,
        robot_type: str = "text",
        temperature: float = 0.7,
        host: str = "localhost",
        port: int = 38001,
        max_client: int = 1,
        serving_method: str = "remote",
        device: str = "cuda",
        api_key: str = "123321",
    ):
        self.nickname = nickname
        self.model = model
        self.robot_type = robot_type
        self.temperature = temperature
        if serving_method == "remote":
            socket_config = SocketConfig(
                Host=host,
                Port=port,
                max_client=max_client,
            )
        else:
            socket_config = None
        self.robot = TextRobot(
            action_space=None,
            character="Ken",
            side=0,
            character_color=KEN_RED,
            ennemy_color=KEN_GREEN,
            only_punch=os.getenv("TEST_MODE", False),
            temperature=self.temperature,
            sleepy=False,
            model=self.model,
            player_nb=1,
            socket_config=socket_config,
            serving_method=serving_method,
            device=device,
            host = host,
            port = port,
            api_key = api_key,
            )
        self.verify_provider_name()
        
        
class Player2(Player):
    def __init__(
        self,
        nickname: str,
        model: str,
        robot_type: str = "text",
        temperature: float = 0.7,
        delay: float = 0.0,
        host: str = "localhost",
        port: int = 38001,
        max_client: int = 1,
        serving_method: str = "remote",
        device: str = "cuda",
        api_key: str = "123321",
        ):
        self.nickname = nickname
        self.model = model
        self.robot_type = robot_type
        self.temperature = temperature
        self.delay = delay
        if serving_method == "remote":
            socket_config = SocketConfig(
                Host=host,
                Port=port,
                max_client=max_client,
            )
        else:
            socket_config = None
        self.robot = TextRobot(
            action_space=None,
            character="Ken",
            side=1,
            character_color=KEN_GREEN,
            ennemy_color=KEN_RED,
            temperature=self.temperature,
            sleepy=os.getenv("TEST_MODE", False),
            model=self.model,
            player_nb=2,
            delay=self.delay,
            socket_config=socket_config,
            serving_method=serving_method,
            device=device,
            host = host,
            port = port,
            api_key = api_key,
        )
        self.verify_provider_name()
        
        
class PlanAndAct(Thread):
    def __init__(self, game, episode: Episode, delayed:float = 0.0, serving_type = None):
        self.running = True
        self.game = game
        self.episode = episode
        self.delayed = delayed
        self.serving_type = serving_type

        Thread.__init__(self, daemon=True)
        
        
class PlanAndActPlayer1(PlanAndAct):
    def run(self) -> None:
        player_1 = []
        connect_flag = False
        while True:
            if self.running:
                if connect_flag is False and self.serving_type == "remote":
                    
                    self.game.player_1.robot.connect_socket()
                    connect_flag = True
                    
                if "agent_0" not in self.game.actions:
                    # Plan
                    # print("start acting Player 1")
                    start = time.time()
                    self.game.player_1.robot.plan()
                    # Act
                    self.game.actions["agent_0"] = self.game.player_1.robot.act()
                    # Observe the environment
                    self.game.player_1.robot.observe(
                        self.game.observation, self.game.actions, self.game.reward
                    )
                    end = time.time()
                    period_time = end - start
                    player_1.append(period_time)
            else:
                # return player_1
                print(f"Player 1 time list: {player_1}")
                print(f"Player 1 average time: {sum(player_1) / len(player_1)}")
                logger.info(
                    f"Player 1 time list: {player_1}"
                )
                logger.info(
                    f"Player 1 average time: {sum(player_1) / len(player_1)}"
                )
                break
            
            
class PlanAndActPlayer2(PlanAndAct):
    def run(self) -> None:
        player_2 = []
        connect_flag = False
        while True:
            if self.running:
                if connect_flag is False and self.serving_type == "remote":
                    self.game.player_2.robot.connect_socket()
                    connect_flag = True

                if "agent_1" not in self.game.actions:
                    
                    start = time.time()
                    # Plan
                    self.game.player_2.robot.plan()
                    # Act
                    self.game.actions["agent_1"] = self.game.player_2.robot.act()
                    # Observe the environment
                    self.game.player_2.robot.observe(
                        self.game.observation, self.game.actions, -self.game.reward
                    )
                    end = time.time()
                    period_time = end - start
                    player_2.append(period_time)
                    time.sleep(self.delayed * period_time)
            else:
                # return player_2
                print(f"Player 2 time list: {player_2}")
                print(f"Player 2 average time: {sum(player_2) / len(player_2)}")
                logger.info(
                    f"Player 2 time list: {player_2}"
                )
                logger.info(
                    f"Player 2 average time: {sum(player_2) / len(player_2)}"
                )
                break
            
def agent_loop(agent_id: str, player, shared):
    connect_flag = False
    time_list = []
    if player.robot.serving_method is not "remote" or player.robot.serving_method is not "api":
        # player.robot.init_local_model()
        player.robot.init_client()
    if agent_id == "agent_0":
        shared["model_prepare_0"] = True
    else:
        shared["model_prepare_1"] = True
    while not shared["start"]:
        time.sleep(0.01)
    player.robot.observe(shared["observation"], {}, 0.0)
    while not shared["done"]:

        actions = shared["actions"]
        if agent_id not in actions:
            if not connect_flag and player.robot.serving_method == "remote":
                player.robot.connect_socket()
                connect_flag = True
            start = time.time()
            player.robot.plan()
            actions[agent_id] = player.robot.act()
            end = time.time()
            time_list.append(end - start)
            
        obs = shared.get("observation")
        reward = shared.get("reward")
        action = shared["actions"].get(agent_id, 0)  # fallback to 0
        if obs is not None:
            player.robot.observe(obs, {agent_id: action}, reward)
        time.sleep(0.001)
