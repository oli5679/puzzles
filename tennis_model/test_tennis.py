import pytest
import tennis
import simulation_utils
import scipy.stats as ss


def test_init():
    gs = tennis.OutcomeProbability(point_win_prob=0.2, target_set_score=(3, 2))


def test_game_win_prob():
    for prob in [0.2, 0.45, 0.32, 0.88]:
        gs = tennis.OutcomeProbability(point_win_prob=prob, target_set_score=(5, 1))
        win_game_prob = gs.win_game_prob()
        game_sims = [simulation_utils.sim_game_outcome(prob) for _ in range(10000)]
        lower_bound_game, upper_bound_game, = simulation_utils.confidence_interval(
            game_sims
        )
        assert upper_bound_game > win_game_prob and lower_bound_game < win_game_prob


def test_outcome_prob():
    for prob, target in [(0.3, (4, 1)), (0.22, (3, 2)), (0.85, (4, 0)), (0.13, (1, 4))]:
        outcome_sims = [
            simulation_utils.sim_outcome(
                sim_game_fn=simulation_utils.sim_game_outcome,
                sim_game_args={"point_win_prob": 1.0 - prob},
                target_score=target,
            )
            for _ in range(10000)
        ]
        gs = tennis.OutcomeProbability(point_win_prob=prob, target_set_score=target)
        outcome_prob = gs.outcome_prob()
        lower_bound_outcome, upper_bound_outcome = simulation_utils.confidence_interval(
            outcome_sims
        )
        assert upper_bound_outcome > outcome_prob and lower_bound_outcome < outcome_prob


def test_outcome_prob_binomial():

    gs = tennis.OutcomeProbability(point_win_prob=0.35, target_set_score=(6, 3))

    win_game_prob = gs.win_game_prob()
    outcome_prob = gs.outcome_prob()
    set_score = ss.binom.pmf(n=9, k=6, p=win_game_prob) - (
        ss.binom.pmf(n=8, k=6, p=win_game_prob) * (1 - win_game_prob)
    )
    assert outcome_prob == pytest.approx(set_score)
