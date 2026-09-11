import random
import unittest

from src.game.board import Board, GameState, HIDDEN, KNOWN_MINE, DIRECTIONS


class TestBoardInit(unittest.TestCase):
    # Check that board initialises correctly
    def test_dimensions(self):
        b = Board(9, 9, 10)
        self.assertEqual(b.rows, 9)
        self.assertEqual(b.cols, 9)
        self.assertEqual(b.mine_count, 10)
        self.assertEqual(b.total_cells, 81)

    # Check that a board with zero rows or columns raises an error
    def test_invalid_board_size_zero(self):
        with self.assertRaises(ValueError):
            Board(0, 5, 1)
        with self.assertRaises(ValueError):
            Board(5, 0, 1)

    # Check that a board with negative rows or columns raises an error
    def test_invalid_board_size_negative(self):
        with self.assertRaises(ValueError):
            Board(-1, 5, 1)
        with self.assertRaises(ValueError):
            Board(5, -1, 1)

    # Check that too many mines raises an error
    def test_invalid_mine_count_exceeds_cells(self):
        with self.assertRaises(ValueError):
            Board(3, 3, 10)

    # Check that a negative mine count raises an error
    def test_invalid_mine_count_negative(self):
        with self.assertRaises(ValueError):
            Board(5, 5, -1)

    # Check that a board with zero mines is considered won after the first reveal
    def test_zero_mines(self):
        b = Board(5, 5, 0)
        self.assertFalse(b.won)
        b.reveal(2, 2)
        self.assertTrue(b.won)


class TestFirstClickSafety(unittest.TestCase):
    # Check that the first click is always safe and reveals a zero cell
    def _check_first_click(self, rows, cols, mines, first_row, first_col):
        rng = random.Random(42)
        b = Board(rows, cols, mines, rng=rng)
        b.reveal(first_row, first_col)
        self.assertFalse(b.is_mine(first_row, first_col))
        self.assertFalse(b.is_hidden(first_row, first_col))
        self.assertTrue(b.is_revealed(first_row, first_col))
        self.assertEqual(b.player_value(first_row, first_col), 0)

    # Check first click in various positions on different board sizes
    def test_corner_click_9x9(self):
        self._check_first_click(9, 9, 10, 0, 0)

    def test_center_click_9x9(self):
        self._check_first_click(9, 9, 10, 4, 4)

    def test_edge_click_16x16(self):
        self._check_first_click(16, 16, 40, 0, 8)

    def test_corner_click_30x30(self):
        self._check_first_click(30, 30, 270, 0, 0)

    def test_center_click_30x30(self):
        self._check_first_click(30, 30, 270, 15, 15)

    # Check that the safe zone around the first click does not contain any mines
    def test_safe_zone(self):
        rng = random.Random(7)
        b = Board(9, 9, 10, rng=rng)
        b.reveal(4, 4)
        for dr, dc in DIRECTIONS:
            r, c = 4 + dr, 4 + dc
            self.assertFalse(b.is_mine(r, c), f"mine at ({r},{c}) in safe zone")


class TestMinePlacement(unittest.TestCase):
    # Check that the number of mines placed matches the specified mine count
    def test_exact_mine_count(self):
        rng = random.Random(1)
        b = Board(9, 9, 10, rng=rng)
        b.reveal(0, 0)
        mines = sum(
            1 for r in range(9) for c in range(9) if b.is_mine(r, c)
        )
        self.assertEqual(mines, 10)

    # Check that the adjacent mine counts are correct for each revealed cell
    def test_adjacent_mines_correct(self):
        rng = random.Random(2)
        b = Board(5, 5, 5, rng=rng)
        b.reveal(0, 0)
        for r in range(5):
            for c in range(5):
                if b.is_mine(r, c) or not b.is_revealed(r, c):
                    continue
                computed = b.player_value(r, c)
                actual = sum(
                    1
                    for nr, nc in b.neighbors(r, c)
                    if b.is_mine(nr, nc)
                )
                self.assertEqual(computed, actual, f"mismatch at ({r},{c})")


class TestReveal(unittest.TestCase):
    # Check that on the first click, flood fill works
    def test_reveal_first_click_flood_fills(self):
        rng = random.Random(10)
        b = Board(5, 5, 1, rng=rng)
        # One mine at far corner, center should be all zeros
        b.reveal(2, 2)
        # Flood-fill should reveal a large region from center
        self.assertGreater(b.revealed_count, 1)

    # Check that revealing a mine returns False and sets the game to lost
    def test_reveal_mine_returns_false(self):
        rng = random.Random(12)
        b = Board(5, 5, 10, rng=rng)
        b.reveal(0, 0)  
        # Find a mine to click
        mine_found = False
        for r in range(5):
            for c in range(5):
                if b.is_mine(r, c):
                    self.assertFalse(b.reveal(r, c))
                    mine_found = True
                    break
            if mine_found:
                break
        self.assertTrue(b.game_over, "no mines placed")

    # Check that revealing a cell out of bounds raises an IndexError
    def test_reveal_out_of_bounds(self):
        b = Board(5, 5, 5)
        with self.assertRaises(IndexError):
            b.reveal(-1, 0)
        with self.assertRaises(IndexError):
            b.reveal(5, 0)

    # Check that revealing an already revealed cell does not change the revealed count
    def test_reveal_already_revealed(self):
        rng = random.Random(13)
        b = Board(5, 5, 1, rng=rng)
        b.reveal(0, 0)
        count_before = b.revealed_count
        b.reveal(0, 0)
        self.assertEqual(b.revealed_count, count_before)


class TestFloodFill(unittest.TestCase):
    # Check that flood fill reveals full board when there are zero mines
    def test_full_flood_on_zero_mines(self):
        b = Board(10, 10, 0)
        b.reveal(5, 5)
        self.assertTrue(b.won)
        self.assertEqual(b.revealed_count, 100)

    # Check that flood fill only reveals zero cells
    def test_flood_stops_at_numbers(self):
        rng = random.Random(20)
        b = Board(5, 5, 5, rng=rng)
        b.reveal(2, 2)
        # Some cells should still be hidden (mines or cells adjacent to mines)
        hidden = sum(
            1 for r in range(5) for c in range(5) if b.is_hidden(r, c)
        )
        self.assertGreater(hidden, 0)

    # Check that the flood fill works correctly at the board edges
    def test_flood_to_board_edge(self):
        b = Board(5, 5, 0)
        b.reveal(0, 0)
        self.assertTrue(b.won)
        self.assertEqual(b.revealed_count, 25)


class TestKnownMines(unittest.TestCase):
    # Check that marking a cell as a known mine works and can be queried
    def test_mark_and_query(self):
        b = Board(5, 5, 5)
        self.assertFalse(b.is_known_mine(2, 3))
        b.mark_known_mine(2, 3)
        self.assertTrue(b.is_known_mine(2, 3))

    # Check that marking a known mine does not reveal it and does not change the player's view
    def test_known_mine_not_revealed(self):
        rng = random.Random(30)
        b = Board(5, 5, 5, rng=rng)
        b.reveal(0, 0)
        b.mark_known_mine(4, 4)
        self.assertTrue(b.is_known_mine(4, 4))
        self.assertFalse(b.is_revealed(4, 4))
        self.assertEqual(b.player_value(4, 4), -1)

    # Check that marking a revealed cell raises a ValueError
    def test_mark_revealed_cell_raises(self):
        rng = random.Random(92)
        b = Board(5, 5, 5, rng=rng)
        b.reveal(0, 0)
        with self.assertRaises(ValueError):
            b.mark_known_mine(0, 0)

    # Check that marking the same cell twice raises a ValueError
    def test_mark_same_cell_twice_raises(self):
        b = Board(5, 5, 5)
        b.mark_known_mine(2, 2)
        with self.assertRaises(ValueError):
            b.mark_known_mine(2, 2)


class TestRemainingMines(unittest.TestCase):
    # Check that the remaining adjacent mines count is correct when no known mines are present
    def test_remaining_mines_no_known(self):
        rng = random.Random(40)
        b = Board(5, 5, 5, rng=rng)
        b.reveal(0, 0)
        for r in range(5):
            for c in range(5):
                if b.is_revealed(r, c) and b.player_value(r, c) > 0:
                    self.assertEqual(
                        b.adj_remaining_mines(r, c),
                        b.player_value(r, c),
                    )
                    break
            else:
                continue
            break

    # Check that the remaining adjacent mines count decreases when a known mine is marked
    def test_remaining_mines_with_known(self):
        rng = random.Random(41)
        b = Board(5, 5, 5, rng=rng)
        b.reveal(0, 0)
        for r in range(5):
            for c in range(5):
                if b.is_revealed(r, c) and b.player_value(r, c) > 0:
                    before = b.adj_remaining_mines(r, c)
                    # Mark one hidden neighbor as known mine
                    for nr, nc in b.unrevealed_neighbors(r, c):
                        b.mark_known_mine(nr, nc)
                        break
                    self.assertEqual(b.adj_remaining_mines(r, c), before - 1)
                    return


class TestUnrevealedNeighbors(unittest.TestCase):
    # Check that the unrevealed neighbors function returns the correct number of hidden neighbors
    def test_unrevealed_count_before_reveal(self):
        b = Board(5, 5, 5)
        # Before any reveal, all neighbors are unrevealed
        self.assertEqual(len(b.unrevealed_neighbors(2, 2)), 8)

class TestNeighbors(unittest.TestCase):
    # Check that the neighbors function returns the correct number of neighbors for different positions
    def test_center_has_8_neighbors(self):
        b = Board(10, 10, 0)
        self.assertEqual(len(b.neighbors(5, 5)), 8)

    def test_corner_has_3_neighbors(self):
        b = Board(10, 10, 0)
        self.assertEqual(len(b.neighbors(0, 0)), 3)

    def test_edge_has_5_neighbors(self):
        b = Board(10, 10, 0)
        self.assertEqual(len(b.neighbors(0, 5)), 5)


class TestWon(unittest.TestCase):
    # Check that the game state is tracked correctly
    def test_not_won_initially(self):
        b = Board(5, 5, 5)
        self.assertFalse(b.won)

    def test_won_when_all_safe_cells_revealed(self):
        b = Board(3, 3, 0)
        b.reveal(1, 1)
        self.assertTrue(b.won)

    def test_won_requires_all_safe_cells(self):
        rng = random.Random(60)
        b = Board(5, 5, 5, rng=rng)
        b.reveal(2, 2)
        self.assertFalse(b.won)


class TestLost(unittest.TestCase):
    # Check that the game state is tracked correctly
    def test_not_lost_initially(self):
        b = Board(5, 5, 5)
        self.assertFalse(b.lost)

    def test_lost_when_revealing_mine(self):
        rng = random.Random(70)
        b = Board(5, 5, 10, rng=rng)
        b.reveal(0, 0)
        for r in range(5):
            for c in range(5):
                if b.is_mine(r, c):
                    b.reveal(r, c)
                    self.assertTrue(b.lost)
                    return

    def test_not_lost_on_safe_reveal(self):
        rng = random.Random(71)
        b = Board(5, 5, 5, rng=rng)
        b.reveal(0, 0)
        self.assertFalse(b.lost)


class TestGameOver(unittest.TestCase):
    # Check that the game state is tracked correctly
    def test_game_over_initially_false(self):
        b = Board(5, 5, 5)
        self.assertFalse(b.game_over)

    def test_game_over_on_win(self):
        b = Board(3, 3, 0)
        b.reveal(1, 1)
        self.assertTrue(b.game_over)

    def test_game_over_on_loss(self):
        rng = random.Random(80)
        b = Board(5, 5, 10, rng=rng)
        b.reveal(0, 0)
        for r in range(5):
            for c in range(5):
                if b.is_mine(r, c):
                    b.reveal(r, c)
                    self.assertTrue(b.game_over)
                    return


class TestGameState(unittest.TestCase):
    # Check that the game state is tracked correctly
    def test_game_state_ongoing_initially(self):
        b = Board(5, 5, 5)
        self.assertEqual(b.game_state, GameState.ONGOING)

    def test_game_state_won(self):
        b = Board(3, 3, 0)
        b.reveal(1, 1)
        self.assertEqual(b.game_state, GameState.WON)

    def test_game_state_lost(self):
        rng = random.Random(81)
        b = Board(5, 5, 10, rng=rng)
        b.reveal(0, 0)
        for r in range(5):
            for c in range(5):
                if b.is_mine(r, c):
                    b.reveal(r, c)
                    self.assertEqual(b.game_state, GameState.LOST)
                    return


class TestRemainingTotalMines(unittest.TestCase):
    # Check that the universal remaining mine count is tracked correctly
    def test_initial_total(self):
        b = Board(5, 5, 10)
        self.assertEqual(b.remaining_total_mines(), 10)

    def test_decreases_after_marking(self):
        b = Board(5, 5, 10)
        b.mark_known_mine(0, 0)
        self.assertEqual(b.remaining_total_mines(), 9)

    def test_multiple_marks(self):
        b = Board(5, 5, 10)
        b.mark_known_mine(0, 0)
        b.mark_known_mine(1, 1)
        b.mark_known_mine(2, 2)
        self.assertEqual(b.remaining_total_mines(), 7)


class TestCountUnrevealedNeighbors(unittest.TestCase):
    # Check that the count of unrevealed neighbors is correct before and after reveals
    def test_before_reveal(self):
        b = Board(5, 5, 5)
        self.assertEqual(b.count_unrevealed_neighbors(2, 2), 8)

    def test_after_flood_fill(self):
        b = Board(5, 5, 0)
        b.reveal(2, 2)
        self.assertEqual(b.count_unrevealed_neighbors(2, 2), 0)

    def test_corner_before_reveal(self):
        b = Board(5, 5, 5)
        self.assertEqual(b.count_unrevealed_neighbors(0, 0), 3)

    def test_matches_unrevealed_neighbors_length(self):
        rng = random.Random(82)
        b = Board(5, 5, 5, rng=rng)
        b.reveal(0, 0)
        for r in range(5):
            for c in range(5):
                if b.is_revealed(r, c):
                    count = b.count_unrevealed_neighbors(r, c)
                    length = len(b.unrevealed_neighbors(r, c))
                    self.assertEqual(count, length)


class TestFrontier(unittest.TestCase):
    # Check that the frontier is empty before any reveal
    def test_empty_before_reveal(self):
        b = Board(5, 5, 5)
        self.assertEqual(b.frontier(), [])

    # Check that the frontier is correctly populated after a reveal
    def test_frontier_after_reveal(self):
        rng = random.Random(83)
        b = Board(9, 9, 10, rng=rng)
        b.reveal(4, 4)
        f = b.frontier()
        self.assertGreater(len(f), 0)
        for r, c in f:
            self.assertTrue(b.is_revealed(r, c))
            self.assertTrue(
                any(b.is_hidden(nr, nc) for nr, nc in b.neighbors(r, c))
            )

    # Check that the frontier is empty after a full flood fill
    def test_frontier_empty_after_full_flood(self):
        b = Board(5, 5, 0)
        b.reveal(2, 2)
        self.assertEqual(b.frontier(), [])

class TestRevealAfterLoss(unittest.TestCase):
    # Check that revealing a cell after the game is lost raises a RuntimeError
    def test_reveal_raises_after_loss(self):
        rng = random.Random(91)
        b = Board(5, 5, 10, rng=rng)
        b.reveal(0, 0)
        for r in range(5):
            for c in range(5):
                if b.is_mine(r, c):
                    b.reveal(r, c)
                    break
            else:
                continue
            break
        self.assertTrue(b.lost)
        with self.assertRaises(RuntimeError):
            b.reveal(1, 1)

class TestPlayerValue(unittest.TestCase):
    # Check that the player_value function returns the correct values for hidden, revealed, and known mine cells
    def test_hidden_cell(self):
        b = Board(5, 5, 5)
        self.assertEqual(b.player_value(2, 2), HIDDEN)

    def test_known_mine_cell(self):
        b = Board(5, 5, 5)
        b.mark_known_mine(1, 1)
        self.assertEqual(b.player_value(1, 1), KNOWN_MINE)


if __name__ == "__main__":
    unittest.main()
