import random
import unittest

from src.game.board import Board, GameState, KNOWN_MINE
from src.solver import Solver


def make_board(rows, cols, mines, revealed):
    """Build a Board with a fixed mine layout for deterministic tests.

    mines:    set of (r, c) coordinates that hold mines
    revealed: dict {(r, c): player_value} of already-revealed cells
    """
    b = Board(rows, cols, len(mines), rng=random.Random(0))
    for r, c in mines:
        b._mine_grid[r][c] = True
    b._generated = True
    for r in range(rows):
        for c in range(cols):
            if (r, c) in mines:
                b._adj_counts[r][c] = -1
            else:
                b._adj_counts[r][c] = sum(
                    1 for nr, nc in b.neighbors(r, c) if (nr, nc) in mines
                )
    for (r, c), v in revealed.items():
        b._player_grid[r][c] = v
        b._revealed_count += 1
    if b._revealed_count == b.total_cells - b.mine_count:
        b._game_state = GameState.WON
    return b


class TestSinglePoint(unittest.TestCase):
    def test_reveal_safe_neighbors(self):
        # (1,1) needs 2 mines. (0,1) and (1,0) are already known mines,
        # so every other hidden neighbor must be safe.
        b = make_board(3, 3, mines={(0, 1), (1, 0)}, revealed={(1, 1): 2})
        b.mark_known_mine(0, 1)
        b.mark_known_mine(1, 0)

        s = Solver()
        changed = s._single_point(b, b.frontier())

        self.assertTrue(changed)
        for r, c in [(0, 0), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]:
            self.assertTrue(b.is_revealed(r, c), f"({r},{c}) should be revealed")

    def test_mark_all_neighbors_as_mines(self):
        # Corner (0,0) needs 3 mines; its only 3 neighbors are all hidden
        # and all are mines -> remaining == hidden count, mark them all.
        b = make_board(3, 3, mines={(0, 1), (1, 0), (1, 1)}, revealed={(0, 0): 3})

        s = Solver()
        changed = s._single_point(b, b.frontier())

        self.assertTrue(changed)
        for r, c in [(0, 1), (1, 0), (1, 1)]:
            self.assertTrue(b.is_known_mine(r, c), f"({r},{c}) should be a known mine")

    def test_no_change_when_ambiguous(self):
        # (1,1) needs 1 mine among 3 hidden neighbors - unresolvable.
        b = make_board(3, 3, mines={(2, 2)}, revealed={(1, 1): 1})

        s = Solver()
        changed = s._single_point(b, b.frontier())

        self.assertFalse(changed)
        # Nothing should have been revealed or marked
        for r in range(3):
            for c in range(3):
                if (r, c) != (1, 1):
                    self.assertTrue(b.is_hidden(r, c))


class TestSubsetRule(unittest.TestCase):
    def test_mark_extra_cells_as_mines(self):
        mines = {(1, 1), (1, 2), (2, 0), (2, 2)}
        b = make_board(3, 3, mines=mines, revealed={
            (0, 0): 1, (0, 1): 2, (0, 2): 2, (2, 1): 4,
        })

        s = Solver()
        changed = s._subset_rule(b, b.frontier())

        self.assertTrue(changed)
        for r, c in [(2, 0), (2, 2), (1, 2)]:
            self.assertTrue(b.is_known_mine(r, c), f"({r},{c}) should be a known mine")

    def test_reveal_extra_cells_as_safe(self):
        mines = {(1, 1), (1, 2)}
        b = make_board(3, 3, mines=mines, revealed={
            (0, 0): 1, (0, 1): 2, (0, 2): 2, (2, 1): 2,
        })

        s = Solver()
        changed = s._subset_rule(b, b.frontier())

        self.assertTrue(changed)
        self.assertTrue(b.is_revealed(2, 0), "(2,0) should be revealed")
        self.assertTrue(b.is_revealed(2, 2), "(2,2) should be revealed")
        self.assertTrue(b.is_known_mine(1, 2), "(1,2) should remain a known mine")


class TestSolve(unittest.TestCase):
    def test_returns_won_when_deduction_clears_board(self):
        # 3x3 with two mines in the far corners. A manual reveal of the top
        # region leaves 3 hidden cells; single-point deduction resolves all.
        b = make_board(3, 3, mines={(2, 0), (2, 2)}, revealed={
            (0, 0): 0, (0, 1): 0, (0, 2): 0,
            (1, 0): 1, (1, 1): 2, (1, 2): 1,
        })

        s = Solver()
        result = s.solve(b)

        self.assertEqual(result, "won")
        self.assertTrue(b.won)

    def test_returns_stuck_when_guess_needed(self):
        # One mine among three hidden neighbors - deduction cannot proceed.
        b = make_board(3, 3, mines={(1, 1)}, revealed={(0, 0): 1})

        s = Solver()
        result = s.solve(b)

        self.assertEqual(result, "stuck")
        self.assertFalse(b.won)
        self.assertFalse(b.lost)

    def test_returns_lost_when_forced_reveal_hits_mine(self):
        # Inconsistent board state: (0,0) claims 0 adjacent mines, but (1,1)
        # is a mine. A forced cascade reveals it and loses. This must never
        # happen in practice, but the solver should surface it cleanly.
        b = make_board(3, 3, mines={(1, 1)}, revealed={(0, 0): 0})
        b._adj_counts[0][0] = 0  # lie: real count is 1

        s = Solver()
        result = s.solve(b)

        self.assertEqual(result, "lost")
        self.assertTrue(b.lost)

    def test_returns_stuck_on_already_won_board(self):
        b = make_board(3, 3, mines={(2, 2)}, revealed={
            (0, 0): 0, (0, 1): 0, (0, 2): 0,
            (1, 0): 0, (1, 1): 0, (1, 2): 1,
            (2, 0): 1, (2, 1): 1,
        })
        b._game_state = GameState.WON

        s = Solver()
        result = s.solve(b)

        self.assertEqual(result, "won")

    def test_solve_survives_board_winning_mid_pass(self):
        # A board that is already won, but the solver doesn't know it yet. The solver
        # should not crash or loop infinitely, and should return "won".
        b = Board(8, 8, 12, rng=random.Random(5017))
        b.reveal(4, 4)

        s = Solver()
        result = s.solve(b)

        self.assertEqual(result, "won")


if __name__ == "__main__":
    unittest.main()