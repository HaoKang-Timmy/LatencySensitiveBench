import sys
from dotenv import load_dotenv
from GameEngine import Game, Player1, Player2
# from Agent import Player1, Player2
from loguru import logger
import argparse
# prepare for env and logger
load_dotenv()
logger.remove()
logger.add(sys.stdout, level="INFO")

def main():
    parser = argparse.ArgumentParser(description="settings of the game and agent")
    parser.add_argument(
        "--serving-choice",
        type=str,
        choices=["vllm", "remote", "api","sglang","huggingface"],
        default="local",
        help="Serving mode: local, remote, or api"
    )
    parser.add_argument(
        "--agent1",
        type=str,
        help="agent1 model choice"
    )
    parser.add_argument(
        "--agent2",
        type=str,
        help="agent2 model choice"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="debug.log",
        help="Path to the log file"
    )
    parser.add_argument(
        "--hostname1",
        type=str,
        default="http://localhost",
        help="Hostname for the remote server"
    )
    parser.add_argument(
        "--hostname2",
        type=str,
        default="http://localhost",
        help="Hostname for the remote server"
    )
    parser.add_argument(
        "--port1",
        type=int,
        default=8001,
        help="Port for the remote server"
    )
    parser.add_argument(
        "--port2",
        type=int,
        default=8002,
        help="Port for the remote server"
    )
    parser.add_argument(
        "--device1",
        type=str,
        default="cuda:0",
        help="Device to use"
    )
    parser.add_argument(
        "--device2",
        type=str,
        default="cuda:1",
        help="Device to use"
    )
    
    args = parser.parse_args()
    logger.add(args.logdir, level="INFO", rotation="100 MB", encoding="utf-8")
    game = Game(
        render=True,
        player_1=Player1(
            nickname="Agent1",
            model=args.agent1,
            host= args.hostname1,
            port=args.port1,
            serving_method=args.serving_choice,
            device=args.device1,
        ),
        player_2=Player2(
            nickname="Agent2",
            model=args.agent1,
            host= args.hostname2,
            port=args.port2,
            serving_method=args.serving_choice,
            device=args.device2,
        ),
    )
    game.run()




if __name__ == "__main__":
    main()