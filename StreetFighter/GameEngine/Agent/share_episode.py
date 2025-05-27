from typing import Optional


class Episode:
    

    def __init__(self, player_1, player_2, player_1_won: Optional[bool] = None):
        self.player_1 = player_1
        self.player_2 = player_2
        self.player_1_won = player_1_won