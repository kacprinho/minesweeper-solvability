import random
import time
import unittest
from itertools import combinations

from src.game.board import Board, GameState
from src.solver import ProbabilityEngine


def make_board(rows, cols, mines, revealed):
    """Build a Board with a fixed mine layout for deterministic tests."""
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


def brute_force_probabilities(board: Board) -> dict[tuple[int, int], float]:
    """Independent, slow-but-obviously-correct reference implementation.

    Enumerates every way to choose `remaining_total_mines` hidden cells as
    mines out of ALL hidden cells on the board, keeps only the choices
    consistent with every revealed clue, and returns each cell's empirical
    frequency of being a mine across the valid choices. Used to cross-check
    the exact combinatorial engine without relying on any hand-derived
    expected numbers.
    """
    hidden = [
        (r, c)
        for r in range(board.rows)
        for c in range(board.cols)
        if board.is_hidden(r, c)
    ]
    remaining = board.remaining_total_mines()
    frontier = board.frontier()

    valid_configs = []
    for combo in combinations(hidden, remaining):
        mine_set = set(combo)
        consistent = True
        for cell in frontier:
            neighbors = board.unrevealed_neighbors(*cell)
            needed = board.adj_remaining_mines(*cell)
            if sum(1 for n in neighbors if n in mine_set) != needed:
                consistent = False
                break
        if consistent:
            valid_configs.append(mine_set)

    counts = {cell: 0 for cell in hidden}
    for config in valid_configs:
        for cell in config:
            counts[cell] += 1

    total = len(valid_configs)
    return {cell: counts[cell] / total for cell in hidden}


# ----------------------------------------------------------------------
# Shared board builders
# ----------------------------------------------------------------------

def build_three_component_board() -> Board:
    """6x6 board with three widely-separated corners, each an isolated
    'exactly 1 mine among 3 cells' constraint, so none of them touch."""
    mines = {(1, 1), (1, 4), (4, 1)}
    revealed = {(0, 0): 1, (0, 5): 1, (5, 0): 1}
    return make_board(6, 6, mines, revealed)


def build_coupled_components_board() -> Board:
    """1x12 board with two overlapping constraints forming one 3-cell
    component with a flexible (not fixed) mine count, a second simple
    2-cell component, and four genuinely unconstrained filler cells."""
    mines = {(0, 0), (0, 4), (0, 8)}  # a, c, d
    revealed = {(0, 1): 1, (0, 3): 1, (0, 9): 1}  # X, Y, Z
    return make_board(1, 12, mines, revealed)


def build_overlap_forces_safe_board() -> Board:
    """2x3 board where two corner clues share two of their three cells,
    forcing each clue's unique cell to be safe once combined, even though
    neither clue proves that on its own."""
    mines = {(0, 1)}  # x
    revealed = {(0, 0): 1, (0, 2): 1}  # A, B
    return make_board(2, 3, mines, revealed)


def build_known_mine_board() -> Board:
    """4x4 board: one corner forces all 3 of its neighbours to be mines,
    a second corner has one ambiguous mine among 3 cells, and 8 cells are
    genuinely unconstrained."""
    mines = {(0, 1), (1, 0), (1, 1), (2, 2)}
    revealed = {(0, 0): 3, (3, 3): 1}
    return make_board(4, 4, mines, revealed)


class TestMultiComponentCoupling(unittest.TestCase):
    # Checks that a component in the middle of the components list (not
    # first or last) is combined correctly with the components both before
    # and after it, not just with an edge component where one side of the
    # prefix/suffix combination is trivial.
    def test_middle_component_combines_with_both_sides(self):
        b = build_three_component_board()
        probs = ProbabilityEngine().probabilities(b)

        constrained_cells = [
            (0, 1), (1, 0), (1, 1),
            (0, 4), (1, 4), (1, 5),
            (4, 0), (4, 1), (5, 1),
        ]
        for cell in constrained_cells:
            self.assertAlmostEqual(probs[cell], 1 / 3, places=12)

        unconstrained_sample = [(0, 2), (2, 2), (3, 3), (5, 4)]
        for cell in unconstrained_sample:
            self.assertEqual(probs[cell], 0.0)

        self.assertAlmostEqual(sum(probs.values()), 3.0, places=12)

    # Checks that two components whose own local mine counts are not fixed
    # to a single value still influence each other correctly through the
    # shared global remaining-mine budget, rather than being treated as if
    # each had an unlimited, independent mine supply.
    def test_components_with_flexible_mine_counts_are_coupled(self):
        b = build_coupled_components_board()
        probs = ProbabilityEngine().probabilities(b)

        self.assertAlmostEqual(probs[(0, 0)], 1 / 5, places=12)   # a
        self.assertAlmostEqual(probs[(0, 2)], 4 / 5, places=12)   # b
        self.assertAlmostEqual(probs[(0, 4)], 1 / 5, places=12)   # c
        self.assertAlmostEqual(probs[(0, 8)], 1 / 2, places=12)   # d
        self.assertAlmostEqual(probs[(0, 10)], 1 / 2, places=12)  # e

        fillers = [(0, 5), (0, 6), (0, 7), (0, 11)]
        for cell in fillers:
            self.assertAlmostEqual(probs[cell], 1 / 5, places=12)

        self.assertAlmostEqual(sum(probs.values()), 3.0, places=12)


class TestComponentSeparation(unittest.TestCase):
    # Checks that two constraints sharing a cell get merged into a single
    # component (even though they don't share every cell), while a
    # separate pair of constraints with no shared cells stays in its own,
    # independent component.
    def test_components_merge_transitively_and_separate_independent_groups(self):
        b = build_coupled_components_board()
        components = ProbabilityEngine()._components(b)

        cell_sets = {frozenset(cells) for cells, _cons in components}
        expected = {
            frozenset({(0, 0), (0, 2), (0, 4)}),
            frozenset({(0, 8), (0, 10)}),
        }
        self.assertEqual(cell_sets, expected)


class TestFullPipelineDeduction(unittest.TestCase):
    # Checks that combining two overlapping revealed clues through the full
    # pipeline (frontier -> components -> enumeration -> weights) can force
    # a cell safe that neither individual clue could resolve on its own.
    def test_overlapping_clues_force_unique_cells_safe(self):
        b = build_overlap_forces_safe_board()
        probs = ProbabilityEngine().probabilities(b)

        self.assertAlmostEqual(probs[(0, 1)], 0.5, places=12)   # x, shared
        self.assertAlmostEqual(probs[(1, 1)], 0.5, places=12)   # z, shared
        self.assertEqual(probs[(1, 0)], 0.0)                    # y, forced safe
        self.assertEqual(probs[(1, 2)], 0.0)                    # w, forced safe

        self.assertAlmostEqual(sum(probs.values()), 1.0, places=12)

class TestKnownMineAccounting(unittest.TestCase):
    # Checks that mines marked as known are correctly excluded from later
    # probability computations and reflected in the remaining mine budget,
    # in isolation from any other board changes.
    def test_known_mines_reduce_remaining_budget(self):
        b = build_known_mine_board()
        engine = ProbabilityEngine()

        self.assertEqual(b.remaining_total_mines(), 4)

        # Mark A's three forced mines directly, without calling resolve(),
        # so nothing else on the board changes and B's region is
        # unaffected by any newly-revealed cells leaking extra information.
        for cell in [(0, 1), (1, 0), (1, 1)]:
            b.mark_known_mine(*cell)

        self.assertEqual(b.remaining_total_mines(), 1)

        probs = engine.probabilities(b)
        for cell in [(2, 2), (2, 3), (3, 2)]:
            self.assertAlmostEqual(probs[cell], 1 / 3, places=12)

        fillers = [(0, 2), (0, 3), (1, 2), (1, 3), (2, 0), (2, 1), (3, 0), (3, 1)]
        for cell in fillers:
            self.assertEqual(probs[cell], 0.0)

class TestBruteForceCrossCheck(unittest.TestCase):
    # Checks that the engine's exact probabilities match an independent
    # brute-force enumeration over every possible mine placement, for a
    # simple single-constraint board. Doesn't rely on any hand-derived
    # constant, only on the two methods agreeing with each other.
    def test_brute_force_matches_engine_single_constraint(self):
        b = make_board(3, 3, mines={(1, 1), (2, 2)}, revealed={(0, 0): 1})

        engine_probs = ProbabilityEngine().probabilities(b)
        brute_probs = brute_force_probabilities(b)

        self.assertEqual(engine_probs.keys(), brute_probs.keys())
        for cell in engine_probs:
            self.assertAlmostEqual(engine_probs[cell], brute_probs[cell], places=12)

    # Checks the same brute-force agreement for two independent components
    # sharing a single global mine budget.
    def test_brute_force_matches_engine_two_components(self):
        b = make_board(
            4, 4, mines={(1, 1), (2, 2)}, revealed={(0, 0): 1, (3, 3): 1}
        )

        engine_probs = ProbabilityEngine().probabilities(b)
        brute_probs = brute_force_probabilities(b)

        self.assertEqual(engine_probs.keys(), brute_probs.keys())
        for cell in engine_probs:
            self.assertAlmostEqual(engine_probs[cell], brute_probs[cell], places=12)

    # Checks the same brute-force agreement for the harder coupled-component
    # scenario, where local mine counts per component are not individually
    # fixed to a single value.
    def test_brute_force_matches_engine_coupled_components(self):
        b = build_coupled_components_board()

        engine_probs = ProbabilityEngine().probabilities(b)
        brute_probs = brute_force_probabilities(b)

        self.assertEqual(engine_probs.keys(), brute_probs.keys())
        for cell in engine_probs:
            self.assertAlmostEqual(engine_probs[cell], brute_probs[cell], places=12)


class TestEnumerationPerformance(unittest.TestCase):
    # Checks that enumeration on a larger connected component (well beyond
    # the 3-5 cell examples used elsewhere) finishes quickly and finds the
    # correct number of valid configurations, since 30x30 boards can
    # produce much bigger frontier components than the small hand-built
    # test boards do.
    def test_large_chain_component_is_fast_and_correct(self):
        n = 19
        cells = list(range(n))
        # Each pair of consecutive cells must contain exactly one mine,
        # which forces a strictly alternating pattern determined entirely
        # by the first cell's value: exactly two valid configurations exist
        # no matter how long the chain is.
        cons = [((i, i + 1), 1) for i in range(n - 1)]

        start = time.perf_counter()
        size_dist, _ones = ProbabilityEngine()._enumerate_component(cells, cons)
        elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 2.0, "chain enumeration took too long")
        self.assertEqual(sum(size_dist), 2)


if __name__ == "__main__":
    unittest.main()