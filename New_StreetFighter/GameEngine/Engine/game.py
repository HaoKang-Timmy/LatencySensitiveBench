import datetime
import os
import random
import traceback
from threading import Thread
from typing import List, Optional
from loguru import logger

from GameEngine.Agent import KEN_GREEN, KEN_RED, TextRobot, SocketConfig, Player1, Player2, PlanAndActPlayer1, PlanAndActPlayer2,Episode
from GameEngine.Agent.config import MODELS
from diambra.arena import (
    EnvironmentSettingsMultiAgent,
    RecordingSettings,
    SpaceTypes,
    make,
)
from rich import print

from GameEngine.Agent.robot import Robot
import time

class Game:
    player_1: Player1
    player_2: Player2

    render: Optional[bool] = False
    splash_screen: Optional[bool] = False
    save_game: Optional[bool] = False
    characters: Optional[List[str]] = ["Ken", "Ken"]
    outfits: Optional[List[int]] = [1, 3]
    frame_shape: Optional[List[int]] = [0, 0, 0]
    seed: Optional[int] = 42
    settings: EnvironmentSettingsMultiAgent = None  # Settings of the game
    env = None  # Environment of the game
    
    def __init__(
        self,
        player_1: Player1,
        player_2: Player2,
        render: bool = False,
        save_game: bool = False,
        splash_screen: bool = False,
        characters: List[str] = ["Ken", "Ken"],
        super_arts: List[int] = [3, 3],
        outfits: List[int] = [1, 3],
        frame_shape: List[int] = [0, 0, 0],
        seed: int = 42,
    ):
        self.render = render
        self.splash_screen = splash_screen
        self.save_game = save_game
        self.characters = characters
        self.super_arts = super_arts
        self.outfits = outfits
        self.frame_shape = frame_shape
        self.seed = seed
        self.settings = self._init_settings()
        self.env = self._init_env(self.settings)
        self.observation, self.info = self.env.reset(seed=self.seed)

        self.player_1 = player_1
        self.player_2 = player_2
        
    def _init_settings(self) -> EnvironmentSettingsMultiAgent:
        """
        Initializes the settings for the game.
        """
        settings = EnvironmentSettingsMultiAgent(
            render_mode="rgb_array",
            splash_screen=self.splash_screen,
        )

        settings.action_space = (SpaceTypes.DISCRETE, SpaceTypes.DISCRETE)
        settings.characters = self.characters
        settings.outfits = self.outfits
        settings.frame_shape = self.frame_shape
        settings.super_art = self.super_arts

        return settings
    def _init_recorder(self) -> RecordingSettings:
        """
        Initializes the recorder for the game.
        """
        if not self.save_game:
            return None
        # Recording settings in root directory
        root_dir = os.path.dirname(os.path.abspath(__file__))
        game_id = "sfiii3n"
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        recording_settings = RecordingSettings()
        recording_settings.dataset_path = os.path.join(
            root_dir, "diambra/episode_recording", game_id, "-", timestamp
        )
        recording_settings.username = "llm-colosseum"

        return recording_settings
    
    def _init_env(self, settings: EnvironmentSettingsMultiAgent):
        """
        Initializes the environment for the game.
        """
        render_mode = "human" if self.render else "rgb_array"
        recorder_settings = self._init_recorder()
        if self.save_game:
            return make(
                "sfiii3n",
                settings,
                render_mode=render_mode,
                episode_recording_settings=recorder_settings,
            )
        return make("sfiii3n", settings, render_mode=render_mode)
    
    def _save(self):
        """
        Save the game state.
        """
        pass
    
    def _determine_winner(self, episode: Episode):
        p1_health = self.observation["P1"]["health"][0]
        p2_health = self.observation["P2"]["health"][0]
        if p1_health > p2_health:
            episode.player_1_won = True
        elif p2_health > p1_health:
            episode.player_1_won = False
        else:
            return "Draw"
        
        
    def run(self):
        try:
            self.actions = {
                "agent_0": 0,
                "agent_1": 0,
            }
            self.reward = 0.0
            
            self.player_1.robot.observe(self.observation, {}, 0.0)
            self.player_2.robot.observe(self.observation, {}, 0.0)
            episode = Episode(player_1=self.player_1, player_2=self.player_2)
            player1_thread = PlanAndActPlayer1(game=self, episode=episode, serving_type=self.player_1.robot.serving_method)
            
            player2_thread = PlanAndActPlayer2(game=self, episode=episode, serving_type=self.player_2.robot.serving_method)
            player1_thread.start()
            player2_thread.start()
            logger.info(
                f"Game started between {self.player_1.nickname} and {self.player_2.nickname}"
            )
            print("------game simluation starting----------")
            while True:
                if self.render:
                    self.env.render()
                    
                actions = self.actions
                time.sleep(0.001)
                
                if "agent_0" not in actions:
                    actions["agent_0"] = 0
                if "agent_1" not in actions:
                    actions["agent_1"] = 0
                    
                observation, reward, terminated, truncated, info = self.env.step(
                    actions
                )
                
                if "agent_0" in self.actions:
                    del actions["agent_0"]
                if "agent_1" in self.actions:
                    del actions["agent_1"]
                    
                self.observation = observation
                self.reward += reward
                
                p1_wins = observation["P1"]["wins"][0]
                p2_wins = observation["P2"]["wins"][0]
                
                if p1_wins == 1 or p2_wins == 1:
                    player1_thread.running = False
                    player2_thread.running = False
                    
                    episode.player_1_won = p1_wins == 1
                    
                    if episode.player_1_won:
                        print(
                            f"[red] Player1 {self.player_1.robot.model} '{self.player_1.nickname}' won!"
                        )
                        logger.info(f"[red]Player1 {self.player_1.robot.model} '{self.player_1.nickname}' won!")
                    else:
                        print(
                            f"[green] Player2 {self.player_2.robot.model} {self.player_2.nickname} won!"
                        )
                        logger.info(f"[green] Player2 {self.player_2.robot.model} {self.player_2.nickname} won!")
                        
                    # episode.save()
                    self.env.close()
                    
                    player1_time_list = player1_thread.join()
                    player2_time_list = player2_thread.join()
                    
                    return episode.player_1_won
                
        except Exception as e:
            # self.env.close()
            print(f"Exception: {e}")
            traceback.print_exception(limit=10)
            traceback.print_tb(limit=40)
            if self.player_1 is None:
                self.controller.stop()
            self.env.close()
        try:
            if self.player_1 is None:
                self.controller.stop()
            self.env.close()
        except Exception as e:
            pass  # Ignore the exception
        return 0
    