import os
import io
import random
import re
import time
from collections import defaultdict
from typing import Any, Dict, Generator, List, Literal, Optional

import numpy as np
import base64

from gymnasium import spaces
from loguru import logger
from rich import print

from .config import (
    INDEX_TO_MOVE,
    META_INSTRUCTIONS,
    META_INSTRUCTIONS_WITH_LOWER,
    MOVES,
    NB_FRAME_WAIT,
    X_SIZE,
    Y_SIZE,
)
from .prompt import *
from .llm_serving import get_client


from .socket_config import SocketConfig
from .observer import detect_position_from_color
import abc

from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.llms import ChatMessage, ChatResponse

def get_device_id(device):

    if isinstance(device, int):
        return str(device)
    match = re.search(r":(\d+)", str(device))
    if match:
        return match.group(1)
    match2 = re.search(r"(\d+)", str(device))
    if match2:
        return match2.group(1)
    return "0"  


class Robot(metaclass=abc.ABCMeta):
    observations: List[Optional[Dict[str, Any]]]  # memory
    next_steps: List[int]  # action plan
    actions: dict  # actions of the agents during a step of the game
    # actions of the agents during the previous step of the game
    previous_actions: Dict[str, List[int]]
    reward: float  # reward of the agent

    action_space: spaces.Space
    character: Optional[str] = None  # character name
    side: int  # side of the stage where playing: 0 = left, 1 = right
    current_direction: Literal["Left", "Right"]  # current direction facing
    sleepy: Optional[bool] = False  # if the robot is sleepy
    only_punch: Optional[bool] = False  # if the robot only punch
    temperature: float = 0.7  # temperature of the language model

    model: str  # model of the robot

    super_bar_own: int
    player_nb: int  # player number # device to use for the model

    def __init__(
        self,
        action_space: spaces.Space,
        character: str,
        side: int,
        character_color: list,
        ennemy_color: list,
        sleepy: bool = False,
        only_punch: bool = False,
        temperature: float = 0.7,
        model: str = "mistral:mistral-large-latest",
        player_nb: int = 0,  # 0 means not specified    
        delay: float = 0.0,
        local_model = None,
        tokenizer = None,
        serving_method = None,
        socket_config: SocketConfig = None,
        device: str = "cuda",
        host: str = "http://localhost",
        port: int = 8001,
        api_key: str = "None",
    ):
        self.action_space = action_space
        self.character = character
        if side == 0:
            self.current_direction = "Right"
        elif side == 1:
            self.current_direction = "Left"

        self.observations = []
        self.next_steps = []
        self.character_color = character_color
        self.ennemy_color = ennemy_color
        self.side = side
        self.sleepy = sleepy
        self.only_punch = only_punch
        self.temperature = temperature
        self.model = model
        self.previous_actions = defaultdict(list)
        self.actions = {}
        self.player_nb = player_nb
        self.delay = delay
        self.local_model = local_model 
        self.tokenizer = tokenizer
        self.serving_method = serving_method
        self.socket_config = socket_config
        self.device = device
        self.host = host
        self.port = port
        self.api_key = api_key
    
    def act(self) -> int:
        """
        At each game frame, we execute the first action in the list of next steps.

        An action is an integer from 0 to 18, where 0 is no action.

        See the MOVES dictionary for the mapping of actions to moves.
        """
        if not self.next_steps or len(self.next_steps) == 0:
            return 0  # No move

        if self.sleepy:
            return 0

        if self.only_punch:
            # Do a Hadouken
            if self.current_direction == "Right":
                self.next_steps.extend(
                    [
                        MOVES["Down"],
                        MOVES["Right+Down"],
                        MOVES["Right"],
                        MOVES["High Punch"],
                    ]
                )
            elif self.current_direction == "Left":
                self.next_steps.extend(
                    [
                        MOVES["Down"],
                        MOVES["Down+Left"],
                        MOVES["Left"],
                        MOVES["High Punch"],
                    ]
                )

        next_step = self.next_steps.pop(0)

        return next_step

    def plan(self) -> None:
        """
        The robot will plan its next steps by calling this method.

        In SF3, moves are based on combos, which are list of actions that must be executed in a sequence.

        Moves of Ken
        https://www.eventhubs.com/guides/2009/may/11/ken-street-fighter-3-third-strike-character-guide/

        Moves of Ryu
        https://www.eventhubs.com/guides/2008/may/09/ryu-street-fighter-3-third-strike-character-guide/
        """

        # If we already have a next step, we don't need to plan
        if len(self.next_steps) > 0:
            return
        # Call the LLM to get the next steps
        start = time.time()
        next_steps_from_llm = self.get_moves_from_llm()
        end = time.time()
        logger.info(f"Time to get moves from LLM: {end - start}s")
        next_buttons_to_press = [
            button
            for combo in next_steps_from_llm
            for button in META_INSTRUCTIONS_WITH_LOWER[combo][
                self.current_direction.lower()
            ]
            # We add a wait time after each button press
            + [0] * NB_FRAME_WAIT
        ]
        self.next_steps.extend(next_buttons_to_press)
        
    def get_moves_from_llm(
        self,
    ) -> List[str]:
        """
        Get a list of moves from the language model.
        """

        # Filter the moves that are not in the list of moves
        invalid_moves = []
        valid_moves = []

        # If we are in the test environment, we don't want to call the LLM
        if os.getenv("DISABLE_LLM", "False") == "True":
            # Choose a random int from the list of moves
            logger.debug("DISABLE_LLM is True, returning a random move")
            return [random.choice(list(MOVES.values()))]
        if self.serving_method == "api":
            while len(valid_moves) == 0:
                llm_stream = self.call_llm()

                # adding support for streaming the response
                # this should make the players faster!

                llm_response = ""
                
                for r in llm_stream:
                    # print(r.delta, end="")
                    
                    llm_response += r.delta
                    # print("resp",llm_response)
                    

                    # The response is a bullet point list of moves. Use regex
                    matches = re.findall(r"- ([\w ]+)", llm_response)
                    moves = ["".join(match) for match in matches]
                    invalid_moves = []
                    valid_moves = []

                    for move in moves:
                        cleaned_move_name = move.strip().lower()
                        if cleaned_move_name in META_INSTRUCTIONS_WITH_LOWER.keys():
                            if self.player_nb == 1:
                                print(
                                    f"[red] Player {self.player_nb} move: {cleaned_move_name}"
                                )
                                logger.info(f"[red] Player {self.player_nb} move: {cleaned_move_name}")
                            elif self.player_nb == 2:
                                print(
                                    f"[green] Player {self.player_nb} move: {cleaned_move_name}"
                                )
                                logger.info(f"[green] Player {self.player_nb} move: {cleaned_move_name}")
                            valid_moves.append(cleaned_move_name)
                        else:
                            logger.debug(f"Invalid completion: {move}")
                            logger.info(f"Invalid completion: {move}")
                            logger.debug(f"Cleaned move name: {cleaned_move_name}")
                            invalid_moves.append(move)

                    if len(invalid_moves) > 1:
                        logger.warning(f"Many invalid moves: {invalid_moves}")
                        logger.info(f"Many invalid moves: {invalid_moves}")

                logger.debug(f"Next moves: {valid_moves}")
                return valid_moves
        elif self.serving_method == "huggingface" or self.serving_method == "vllm" or self.serving_method == "sglang":
            result = self.call_llm_local()
            matches = re.findall(r"- ([\w ]+)", result)
            moves = ["".join(match) for match in matches]
            invalid_moves = []
            valid_moves = []
            for move in moves:
                cleaned_move_name = move.strip().lower()
                if cleaned_move_name in META_INSTRUCTIONS_WITH_LOWER.keys():
                    if self.player_nb == 1:
                        print(
                                f"[red] Player {self.player_nb} move: {cleaned_move_name}"
                            )
                    elif self.player_nb == 2:
                            print(
                                f"[green] Player {self.player_nb} move: {cleaned_move_name}"
                            )
                    valid_moves.append(cleaned_move_name)
                else:
                    logger.debug(f"Invalid completion: {move}")
                    logger.debug(f"Cleaned move name: {cleaned_move_name}")
                    invalid_moves.append(move)
                if len(invalid_moves) > 1:
                    logger.warning(f"Many invalid moves: {invalid_moves}")
                logger.debug(f"Next moves: {valid_moves}")
                logger.info(f"Next moves: {valid_moves}")
            return valid_moves
            
        return []

    
    @abc.abstractmethod
    def call_llm(
        self,
        max_tokens: int = 100,
        top_p: float = 1.0,
    ) -> (
        Generator[ChatResponse, None, None] | Generator[CompletionResponse, None, None]
    ):
        """
        Make an API call to the language model.

        Edit this method to change the behavior of the robot!

        This should return a streaming response. The response should be a list of ChatResponse objects.
        Look into Llamaindex and make sure streaming is on.
        """
        raise NotImplementedError("call_llm method must be implemented")

    @abc.abstractmethod
    def observe(self, observation: dict, actions: dict, reward: float):
        """
        The robot will observe the environment by calling this method.

        The latest observations are at the end of the list.
        """
        # By default, we don't observe anything.
        pass
    
class TextRobot(Robot):
    def init_local_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if self.serving_method == "huggingface":
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.local_model = AutoModelForCausalLM.from_pretrained(self.model, device_map=self.device)
        if self.serving_method == "vllm":
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = get_device_id(self.device)

            from vllm import LLM, SamplingParams
            self.sampling_params = SamplingParams()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.local_model = LLM(model=self.model)
        if self.serving_method == "sglang":
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = get_device_id(self.device)
            from vllm import LLM, SamplingParams
            import sglang as sgl
            self.sampling_params = {"temperature": 0.0}
            self.local_model = sgl.SGLang(model=self.model)
    def init_client(self):
        from openai import OpenAI
        if self.serving_method == "vllm" or self.serving_method == "sglang":
            print("url:", f"{self.host}:{self.port}/v1")
            self.client = OpenAI(
                base_url = f"{self.host}:{self.port}/v1",
                api_key = self.api_key,
            )
        
    def observe(self, observation: dict, actions: dict, reward: float):
        """
        The robot will observe the environment by calling this method.

        The latest observations are at the end of the list.
        """
        observation["character_position"] = detect_position_from_color(
            observation, self.character_color
        )
        observation["ennemy_position"] = detect_position_from_color(
            observation, self.ennemy_color
        )
        
        self.observations.append(observation)
        # we delete the oldest observation if we have more than 10 observations
        if len(self.observations) > 10:
            self.observations.pop(0)
        self.reward = reward

        if actions.get("agent_0") is not None and actions.get("agent_0") != 0:
            self.previous_actions["agent_0"].append(actions["agent_0"])
        if actions.get("agent_1") is not None and actions.get("agent_1") != 0:
            self.previous_actions["agent_1"].append(actions["agent_1"])

        for key, value in actions.items():
            if len(self.previous_actions[key]) > 10:
                self.previous_actions[key].pop(0)

        # Keep track of the current direction by checking the position of the character
        # and the ennemy
        character_position = observation.get("character_position")
        ennemy_position = observation.get("ennemy_position")
        if (
            character_position is not None
            and ennemy_position is not None
            and len(character_position) == 2
            and len(ennemy_position) == 2
        ):
            if character_position[0] < ennemy_position[0]:
                self.current_direction = "Right"
            else:
                self.current_direction = "Left"
                
    def context_prompt(self) -> str:
        """
        Return a str of the context

        "The observation for you is Left"
        "The observation for the opponent is Left+Up"
        "The action history is Up"
        """

        # Create the position prompt
        side = self.side
        obs_own = self.observations[-1]["character_position"]
        obs_opp = self.observations[-1]["ennemy_position"]
        super_bar_own = self.observations[-1]["P" + str(side + 1)]["super_bar"][0]

        if obs_own is not None and obs_opp is not None:
            relative_position = np.array(obs_own) - np.array(obs_opp)
            normalized_relative_position = [
                relative_position[0] / X_SIZE,
                relative_position[1] / Y_SIZE,
            ]
        else:
            normalized_relative_position = [0.3, 0]

        position_prompt = ""
        if abs(normalized_relative_position[0]) > 0.1:
            position_prompt += (
                "You are very far from the opponent. Move closer to the opponent."
            )
            if normalized_relative_position[0] < 0:
                position_prompt += "Your opponent is on the right."
            else:
                position_prompt += "Your opponent is on the left."

        else:
            position_prompt += "You are close to the opponent. You should attack him."

        power_prompt = ""
        if super_bar_own >= 30:
            power_prompt = "You can now use a powerfull move. The names of the powerful moves are: Megafireball, Super attack 2."
        if super_bar_own >= 120 or super_bar_own == 0:
            power_prompt = "You can now only use very powerfull moves. The names of the very powerful moves are: Super attack 3, Super attack 4"
        #### disable power_prompt
        power_prompt = ""
        # Create the last action prompt
        last_action_prompt = ""
        if len(self.previous_actions.keys()) >= 0:
            act_own_list = self.previous_actions["agent_" + str(side)]
            act_opp_list = self.previous_actions["agent_" + str(abs(1 - side))]

            if len(act_own_list) == 0:
                act_own = 0
            else:
                act_own = act_own_list[-1]
            if len(act_opp_list) == 0:
                act_opp = 0
            else:
                act_opp = act_opp_list[-1]

            str_act_own = INDEX_TO_MOVE[act_own]
            str_act_opp = INDEX_TO_MOVE[act_opp]

            last_action_prompt += f"Your last action was {str_act_own}. The opponent's last action was {str_act_opp}."

        reward = self.reward

        # Create the score prompt
        score_prompt = ""
        if reward > 0:
            score_prompt += "You are winning. Keep attacking the opponent."
        elif reward < 0:
            score_prompt += (
                "You are losing. Continue to attack the opponent but don't get hit."
            )

        # Assemble everything
        context = f"""{position_prompt}
{power_prompt}
{last_action_prompt}
Your current score is {reward}. {score_prompt}
To increase your score, move toward the opponent and attack the opponent. To prevent your score from decreasing, don't get hit by the opponent.
"""

        return context
    
    def call_llm(
        self,
        max_tokens: int = 100,
        top_p: float = 1.0,
    ) -> Generator[ChatResponse, None, None]:
        """
        Make an API call to the language model.

        Edit this method to change the behavior of the robot!
        """

        # Generate the prompts
        move_list = "- " + "\n - ".join([move for move in META_INSTRUCTIONS])


        client = get_client(self.model, temperature=self.temperature)

        messages = [
            ChatMessage(role="system", content=BACKGROUND(self.character) + HINT_KEN()),
            ChatMessage(role="user", content=PROMPT_KEN(move_list, self.context_prompt()) + "\nYour Response:\n"),
        ]

        resp = client.stream_chat(messages)

        # logger.debug(f"LLM call to {self.model}: {system_prompt}")
        # logger.debug(f"LLM call to {self.model}: {time.time() - start_time}s")

        return resp
    def call_llm_local(
        self,
        max_tokens: int = 100,
        top_p: float = 1.0,
    ):
        # print(f"call local llm")
        move_list = "- " + "\n - ".join([move for move in META_INSTRUCTIONS])
        prompt = [
            {"role": "system", "content": BACKGROUND(self.character) + HINT_KEN()},
            {
                "role": "user",
                "content": PROMPT_KEN(move_list, self.context_prompt())
                + "\nYour Response:\n",
            },
        ]
        if self.serving_method == "huggingface":
            if "Qwen3" in self.model:
                prompt = self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    enable_thinking=False,
                )
            else:
                prompt = self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False
                )
        if self.serving_method == "huggingface":
            prompt_encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            # Generate the response
            response = self.local_model.generate(
                **prompt_encoded,
                max_new_tokens=max_tokens,
                top_p=top_p,
            )
            # response 直接是Tensor，形状为(batch_size, seq_len)
            prompt_length = prompt_encoded["input_ids"].shape[-1]
            # 只取新生成的部分
            generated_ids = response[0][prompt_length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        elif self.serving_method == "vllm" or self.serving_method == "sglang":
            # vllm 只支持字符串prompt
            # outputs = self.local_model.generate(
            #     prompt,
            #     sampling_params=self.sampling_params
            # )
            # print("client ask")
            if "Qwen3" in self.model:
                extra_body = {
                    "max_tokens": max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            else:
                extra_body = {
                    "max_tokens": max_tokens,
                }
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                stream=False,
                extra_body=extra_body
            )
            text = completion.choices[0].message.content

        # print("-----------response of model:", text)
        return text