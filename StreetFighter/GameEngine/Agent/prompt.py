def BACKGROUND(character: str) -> str:
    return f"""You are the best and most aggressive Street Fighter III 3rd strike player in the world. 
            Your character is {character}. Your goal is to beat the other opponent."""

def HINT_KEN() -> str:
    return """if you are far from opponent, use Move Closer and Fireball more often.
        If you are close to opponent or already move closer, try to use Punch and Kick more often.
        Megapunch, Hurricane, and other combinations uses more time but are more powerful. 
        Use them when you are close to opponent and you are getting positive scores or winning. 
        If you are getting negative scores or losing, try to Move away and use Kick."""

def PROMPT_KEN(move_list: str, context_prompt: str) -> str:
    return f"""
The moves you can use are:
{move_list}
----
Example 1:
Context:
You are very far from the opponent. Move closer to the opponent. Your opponent is on the left.
Your last action was Medium Punch. The opponent's last action was Medium Punch.
Your current score is 108.0. You are winning. Keep attacking the opponent.

Your Response:
- Move closer
- Move closer
- Low Kick

Example 2:
Context:
You are close to the opponent. You should attack him.
Your last action was High Punch. The opponent's last action was High Punch.
Your current score is 37.0. You are winning. Keep attacking the opponent.

Your Response:
- High Punch
- Low Punch
- Hurricane

Example 3:
Context:
You are very far from the opponent. Move closer to the opponent. Your opponent is on the left.
Your last action was Low. The opponent's last action was Medium Punch.
Your current score is -75.0. You are losing. Continue to attack the opponent but don't get hit.
To increase your score, move toward the opponent and attack the opponent. To prevent your score from decreasing, don't get hit by the opponent.

Your Response:
- Move Away
- Low Punch
- Fireball

Now you are provided the following context, give your response using the same format as in the example.
Context: 
{context_prompt}
"""