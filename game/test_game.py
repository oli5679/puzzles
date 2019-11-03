import main

def test_upslope():
    assert main.mountain_count([1, 2, 3, 4]) == 0
    assert main.mountain_count([4, 3, 2, 1]) == 0
    assert main.mountain_count([1, 1, 1, 1]) == 0


def test_single_slope():
    assert main.mountain_count([1, 2, 4, 1]) == 1
    assert main.mountain_count([0, 0, 1, 2, 4, 1, 2]) == 1


def test_valley():
    assert main.mountain_count([1, 0, -1]) == 0
    assert main.mountain_count([1, 0, -1, -1, 1]) == 0


def test_simple_strategy():
    sol = main.GameSolver(
        choices=(1, 2), opponent_choices=(0, 0), selection_sequence=(1, 1)
    )
    sol.find_optimal() == [2, 0, 1, 0]


def test_patient_strategy():
    sol = main.GameSolver(
        choices=(1, 2), opponent_choices=(2, 2), selection_sequence=(1, 1, 1, 1, 10)
    )
    sol.find_optimal() == [1, 2, 2, 2]


def test_defensive_strategy():
    sol = main.GameSolver(
        choices=(1, 1), opponent_choices=(1, 2), selection_sequence=(1, 1, -20, 1, 1)
    )
    sol.find_optimal() == [1, 1, 1, 2]


def test_hand_solved_strategy():
    sol = main.GameSolver(
        choices=(1, 1), opponent_choices=(1, 2), selection_sequence=(1, 1, -20, 1, 1)
    )
    sol.find_optimal() == [1, 1, 1, 2]
