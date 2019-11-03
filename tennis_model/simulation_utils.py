import numpy as np

"""
Utilities for simulating outcomes of tennis match - used to test deterministic solution
"""


def sim_game_outcome(point_win_prob):
    """
    Simulation of single game. 
    
    NOTE - not deterministic!

    Args:
        point_win_prob (float) prob. of player 0 winning single point

    Returns:
        0 or 1 incating player 0/1 won game (int).
    """
    # Intialise scores
    score = [0, 0]

    while True:
        # Increment points totals based on random point outcome
        point_outcome = np.random.choice(
            a=[0, 1], p=[point_win_prob, 1.0 - point_win_prob]
        )
        score[point_outcome] += 1
        # Return if game over, otherwise simulate next point.
        if score[0] >= 4 and score[0] - score[1] >= 2:
            return 1
        elif score[1] >= 4 and score[1] - score[0] >= 2:
            return 0


def confidence_interval(sample, k=2.54):
    """
    Calcates upper and lower bound of confidence interval of binary sample
    
    Args: 
        sample (list) list of binary values
        k (int) size of confidence interval (default is 2.54 = 99% 2 sided)

    Returns
        (tuple) upper and lower bound (floats)
    """
    n = len(sample)
    sample_mean = sum(sample) / n
    sample_standard_error = ((sample_mean * (1 - sample_mean)) / n) ** 0.5
    return (
        sample_mean - (k * sample_standard_error),
        sample_mean + (k * sample_standard_error),
    )


def sim_outcome(sim_game_fn, sim_game_args, target_score):
    """
    Simulation of single set. 
    
    Note - not deterministic
        
    Args:
        sim_game_fn - function: simulates single game, returns 0 / 1 depending on winner

        sim_game_arge - dict: named arguments of the 'sim_game' function

        target_score: list , e.g. [6,1] would find probability of player 0 winning 6-1.
    
    Returns:
        0,1 indicating success/failure in achieving target score
    """
    # Start set at 0,0
    game_score = [0, 0]
    while True:
        # Increment games using sim_game function
        game_outcome = sim_game_fn(**sim_game_args)

        game_score[game_outcome] += 1

        # If game count is exactly equal to target, return 1
        if game_score[0] == target_score[0] and game_score[1] == target_score[1]:
            return 1

        # If player scores already exceed target, return 0
        elif game_score[0] > target_score[0] or game_score[1] > target_score[1]:
            return 0

        # If game is already over, return 0
        elif max(game_score) >= 6 and abs(game_score[0] - game_score[1]) >= 2:
            return 0
