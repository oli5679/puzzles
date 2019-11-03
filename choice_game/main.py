import numpy as np


def mountain_count(points):
    """
    Calculate number of mountain peaks in sequence of points
    NOTE - plateau not counted as a peak

    Args:
        points (iterable): Sequence of numbers

    Returns:
        number_of_peaks (integer): The number of peaks in the mountain
    """
    points = np.array(points)
    differences = points[1:] - points[:-1]
    peaks = (
        (first > 0 and second < 0)
        for (first, second) in zip(differences[:-1], differences[1:])
    )
    return sum(peaks)


class GameSolver:
    """
    Finds Nash equilibrium of game where players sequentially choose n values 
    from sequence. 
    After choice:
        - items are removed from sequence, and their total is attributed to player
        - choice is removed from choice-set
        - other player chooses

    Parameters:
        choices (tuple): choices availible to 1st player
        opponent_choices (tuple): choices availibe to 2nd player
        selection_sequence (tuple): sequence of payoffs from choices
        ​
    Attributes:
        find_optimal (method): finds optimal strategies for all subgames, including main game
        
    Example: 
        sequence of [1,2,10,20] and choice set of [1,2,3]
        choice of 3 leads to:
            Value of 13 attributed to player,  
            Choice set of [1,2]
            Other player now chooses
    """

    def __init__(self, choices, opponent_choices, selection_sequence):
        self.choices = tuple(choices)
        self.opponent_choices = tuple(opponent_choices)
        # Pad with zeros, if selection_sequence not as long as total availible choices
        self.selection_sequence = selection_sequence + tuple(
            [0] * (len(choices) + len(opponent_choices))
        )
        self._values_cache = {}
        self._strategies_cache = {}

    def _solve(self, choices, opponent_choices, selection_sequence):
        """
        Finds optimal strategy for given game-state

        Arguments:
            choices (tuple): choices availible to player with current move
            opponent_choices (tuple): choices availible to player with next move
            selection_sequence (tuple): sequence of remaining payoffs from choices

        Returns:
            payoffs (tuple):
                (payoff_1 (int): payoff of current player,
                payoff_2 (int): payoff of rival player)

        Side-effects:
            caches optimal strategy value and choice in:
                _values_cache 
                _strategies_cache
        """
        cache_key = (choices, opponent_choices, selection_sequence)
        if self._values_cache.get(cache_key):
            return self._values_cache[cache_key]
        else:
            if len(choices) == 1 and len(opponent_choices) == 0:
                payoffs = (sum(selection_sequence[0 : choices[0]]), 0)
                self._strategies_cache[cache_key] = [choices[0]]
            else:
                max_payoff = -1e10
                for i, c in enumerate(choices):
                    turn_payoff = sum(selection_sequence[0:c])
                    remaining_sequence = selection_sequence[c:]
                    remaining_choices = choices[:i] + choices[i + 1 :]
                    opponent_payoff, remaining_payoff = self._solve(
                        opponent_choices, remaining_choices, remaining_sequence
                    )
                    player_payoff = turn_payoff + remaining_payoff
                    if player_payoff > max_payoff:
                        payoffs = (player_payoff, opponent_payoff)
                        self._strategies_cache[cache_key] = (
                            [c]
                            + self._strategies_cache[
                                (
                                    opponent_choices,
                                    remaining_choices,
                                    remaining_sequence,
                                )
                            ]
                        )

            self._values_cache[
                (choices, opponent_choices, selection_sequence)
            ] = payoffs
            return payoffs

    def find_optimal(self):
        """
        Finds optimal strategies for all subgames, including main game

        Returns:
            optimal_strategy (list): list of optimal choices for main game
        """
        self._solve(self.choices, self.opponent_choices, self.selection_sequence)
        return self._strategies_cache[
            (self.choices, self.opponent_choices, self.selection_sequence)
        ]
