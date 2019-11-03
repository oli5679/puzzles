import numpy as np
from tqdm import tqdm
import scipy.stats as ss
from copy import copy


class OutcomeProbability:
    """    
    Attributes:
        point_win_prob (float): Chance of player A winning single game
        target_set_score (tuple) Score target to calculate probability for:
                    int: games won by player A
                    int: games won by player B
        
    Methods:
        win_game_prob - Calculates prob of player A winning single game, from given point score
        outcome_prob - Calculates prob of reaching target score, from given score
    """

    def __init__(self, point_win_prob, target_set_score):
        self.point_win_prob = point_win_prob
        self.target_set_score = target_set_score

    def _deuce_win_prob(self):
        """
        Calculates prob. of player A winning single game from score deuce

        Args:
            point_win_prob (float) prob. of player A winning single point
        
        Returns:
            probability of winning game (float)
        """

        num = self.point_win_prob ** 2
        denom = 1 - (2 * self.point_win_prob * (1 - self.point_win_prob))
        return num / denom

    def win_game_prob(self, point_score=(0, 0)):
        """
        Calculates prob. of player A winning single game from any score
        
        Args:
            point_score (tuple)
                    int: points won by player A
                    int: points won by player B
        
        Returns: probability of winning game (float)
        """

        # Base case one - player A has won
        if point_score[0] - point_score[1] >= 2 and point_score[0] >= 4:
            return 1.0

        # Base case two - player B has won
        elif point_score[1] - point_score[0] >= 2 and point_score[1] >= 4:
            return 0.0

        # Base case three - players have reached deuce
        # Can solve for all 3 combinations of next 2 points and rearrange:
        elif (
            point_score[0] >= 3
            and point_score[1] >= 3
            and point_score[0] == point_score[1]
        ):
            return self._deuce_win_prob()

        # Recursive step
        else:
            # Player zero's winning chance conditional on winning
            win_val = self.win_game_prob(
                point_score=(point_score[0] + 1, point_score[1])
            )

            # Player zero's winning chance conditional on winning
            lose_val = self.win_game_prob(
                point_score=(point_score[0], point_score[1] + 1)
            )

            # Average, weighted by player A's winning/losing prob on this point
            return (win_val * self.point_win_prob) + (
                lose_val * (1 - self.point_win_prob)
            )

    def outcome_prob(self, game_score=(0, 0)):
        """
        Calculates prob. of reaching target_outcome, given:
            current score (games) 
            and player winning chance (game)?
        
        NOTE, not designed to deal with tiebreaks!
        
        Args:
            game_score (tuple)
                int: games won by player A 
                int: games won by player B 
                            
        Returns:
            prob. of achieving outcome (float)
        """
        target_outcome = self.target_set_score
        # Have we exactly reached the target?
        if game_score[0] == target_outcome[0] and game_score[1] == target_outcome[1]:
            return 1.0

        # Does one player already exceed the target?
        elif game_score[0] > target_outcome[0] or game_score[1] > target_outcome[1]:
            return 0.0

        # Has one player already won (and we have not exactly reached the target)
        # Note, this is why you can't use binomial formula, with 6C10
        elif max(game_score) >= 6 and abs(game_score[0] - game_score[1]) >= 2:
            return 0.0

        # Recursive step
        else:
            # Chance of reaching outcome conditional on first player winning
            win_val = self.outcome_prob(game_score=(game_score[0] + 1, game_score[1]))

            # Chance of reaching outcome conditional on first player winning
            lose_val = self.outcome_prob(game_score=(game_score[0], game_score[1] + 1))

            game_win = self.win_game_prob((0, 0))
            # Weighted sum
            return (win_val * game_win) + (lose_val * (1 - game_win))


class Bayesian_Game_Simulation:
    """    
    Attributes:
        prior: beta distribution parameters for player A's point winning
        prob. (list). Ratio = prior mean, sum = prior weight
        target_score (list) target of the simulation
        point_score (list) current score in points of current game
        game_score (list) current score in games of set
    
    Methods:
        sim_single_point - simulates a point, and updates the score and proir
        sim_outcome - simulates set, and returns 0/1 if outcome is achieved
    
    """

    def __init__(self, prior, target_score, point_score, game_score):
        self.prior = prior
        self.target_score = target_score
        self.point_score = point_score
        self.game_score = game_score

    def sim_single_point(self):
        """
        Simulates a single point - outcome is weighted by prior of players'
        winning chances
        
        Args:
            self.prior
            
            
        Returns:
            point outcome (int) 1/0 indicates player B/0 won
        """
        prior_prob = self.prior[0] / (self.prior[0] + self.prior[1])
        return np.random.choice(a=[0, 1], p=(prior_prob, 1.0 - prior_prob))

    def update_scores_and_prior(self, point_outcome):
        """
        Updates prior, and game score
        
        Args:
            point_outcome (int)
            
        Side-effects:
            updates self.prior, self.game_score and self.point_score
            
        Returns:
            None
        """
        self.prior[point_outcome] += 1
        self.point_score[point_outcome] += 1

        above_4_points = self.point_score[point_outcome] >= 4
        ahead_by_2 = (
            self.point_score[point_outcome] - self.point_score[1 - point_outcome]
        ) >= 2
        if above_4_points and ahead_by_2:
            self.point_score = [0, 0]
            self.game_score[point_outcome] += 1

    def sim_outcome(self):
        """
        Simulates a set, checking if score matches target
        
        Args:
            self
            
        Returns:
            1/0 if simulation failed/succeeded in reaching target
        """
        while True:
            # Increment score using sim_single_point method
            point_outcome = self.sim_single_point()
            self.update_scores_and_prior(point_outcome)

            # If game count is exactly equal to target, return 1
            if (
                self.game_score[0] == self.target_score[0]
                and self.game_score[1] == self.target_score[1]
            ):
                return 1

            # If player scores already exceed target, return 0
            elif (
                self.game_score[0] > self.target_score[0]
                or self.game_score[1] > self.target_score[1]
            ):
                return 0

            # If game is already over, return 0
            elif (
                max(self.game_score) >= 6
                and abs(self.game_score[0] - self.game_score[1]) >= 2
            ):
                return 0
