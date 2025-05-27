from .robot import Robot
from typing import List, Optional
from agent import KEN_GREEN, KEN_RED, TextRobot, VisionRobot, TextLocalRobot, SocketConfig
import os
class Player:
    nickname: str
    model: str
    robot: Optional[Robot] = None
    temperature: float = 0.7
    
    def verify_provider_name(self):
        if self.model.startswith("openai"):
            assert (
                os.environ.get("OPENAI_API_KEY") is not None
            ), "OpenAI API key not set"
            
class Player1(Player):
    def __init__(self, nickname: str,
                 model: str,
                 host: str,
                 port: int,
                 serving_method: str,
                 temperature: float = 0.7,
                 max_client: int = 1,):
        self.nickname = nickname
        self.model = model
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
            )
        print(f"[red] Player 1 using: {self.model}")
        self.verify_provider_name()
        
class Player2(Player):
    def __init__(self, nickname: str,
        model: str,
        host: str,
        port: int,
        serving_method: str,
        temperature: float = 0.7,
        max_client: int = 1,):
        self.nickname = nickname
        self.model = model
        self.temperature = temperature
        if serving_method == "remote":
            socket_config = SocketConfig(
                Host=host,
                Port=port,
                max_client=max_client,
            )
        
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
            socket_config=socket_config,
            serving_method=serving_method,
        )
        print(f"[green] Player 2 using: {self.model}")
        self.verify_provider_name()
        
        
        
class PlanAndAct(Thread):
    def __init__(self, game: Game, episode: Episode, delayed:float = 0.0, serving_type = None):
        self.running = True
        self.game = game
        self.episode = episode
        self.delayed = delayed
        self.serving_type = serving_type

        Thread.__init__(self, daemon=True)
        # atexit.register(self.stop)