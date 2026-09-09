import random
from collections import deque
from enum import Enum

# Config
DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]
HIDDEN = -2
KNOWN_MINE = -1


class GameState(Enum):
    ONGOING = "Ongoing"
    WON = "Won"
    LOST = "Lost"


class Board:
    """Minesweeper board representation and core operations.

    Two-grid architecture:
      _mine_grid:  bool grid — ground truth mine locations
      _player_grid: int grid — everything the solver/player sees:
          -2 = unrevealed, -1 = known mine (solver flagged), 0-8 = revealed
    """
    # ------------------------------------------------------------------
    # Board Setup
    # ------------------------------------------------------------------

    def __init__(self, rows: int, cols: int, mine_count: int, rng: random.Random | None = None) -> None:
        # Error checks
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be > 0")
        if mine_count < 0:
            raise ValueError("mine_count must be >= 0")
        total = rows * cols
        if mine_count > total:
            raise ValueError("mine_count exceeds total cells")

        self._rows = rows
        self._cols = cols
        self._mine_count = mine_count
        self._rng = rng or random.Random()

        # Mine locations, generated on first click
        self._mine_grid: list[list[bool]] = [[False] * cols for _ in range(rows)]
        # Player-visible state: -2=hidden, -1=known mine, 0-8=revealed adj count
        self._player_grid: list[list[int]] = [[HIDDEN] * cols for _ in range(rows)]

        # Game state tracking
        self._revealed_count = 0
        self._known_mines: set[tuple[int, int]] = set()
        self._generated = False
        self._game_state = GameState.ONGOING

        # Pre-compute neighbor lists
        self._adj: list[list[list[tuple[int, int]]]] = [
            [
                [
                    (r + dr, c + dc)
                    for dr, dc in DIRECTIONS
                    if 0 <= r + dr < rows and 0 <= c + dc < cols
                ]
                for c in range(cols)
            ]
            for r in range(rows)
        ]

        # Pre-compute adjacency counts (filled on generation)
        self._adj_counts: list[list[int]] = [[0] * cols for _ in range(rows)]

    # ------------------------------------------------------------------
    # Board generation (called on first click)
    # ------------------------------------------------------------------

    def _generate(self, safe_row: int, safe_col: int) -> None:
        """Place mines uniformly at random, excluding the safe zone."""
        safe: set[tuple[int, int]] = {(safe_row, safe_col)}
        for nr, nc in self._adj[safe_row][safe_col]:
            safe.add((nr, nc))

        candidates = [
            (r, c)
            for r in range(self._rows)
            for c in range(self._cols)
            if (r, c) not in safe
        ]

        self._rng.shuffle(candidates)
        for r, c in candidates[: self._mine_count]:
            self._mine_grid[r][c] = True

        for r in range(self._rows):
            for c in range(self._cols):
                if self._mine_grid[r][c]:
                    self._adj_counts[r][c] = -1
                    continue
                self._adj_counts[r][c] = sum(
                    1 for nr, nc in self._adj[r][c] if self._mine_grid[nr][nc]
                )

        self._generated = True

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def reveal(self, row: int, col: int) -> bool:
        """
        Reveal a cell. Returns False if the cell is a mine (game over).

        On the very first call the board is generated with this cell
        guaranteed to be a zero-cell.
        """
        if not (0 <= row < self._rows and 0 <= col < self._cols):
            raise IndexError(f"({row}, {col}) out of bounds")

        if self._game_state != GameState.ONGOING:
            raise RuntimeError("Cannot reveal after game is over")

        if not self._generated:
            self._generate(row, col)

        elif self._mine_grid[row][col]:
            self._game_state = GameState.LOST
            return False

        if self._player_grid[row][col] != HIDDEN:
            return True

        # Reveal this cell
        self._player_grid[row][col] = self._adj_counts[row][col]
        self._revealed_count += 1

        # BFS flood-fill for zero-cells
        if self._adj_counts[row][col] == 0:
            queue: deque[tuple[int, int]] = deque()
            queue.append((row, col))

            while queue:
                r, c = queue.popleft()
                if self._adj_counts[r][c] == 0:
                    for nr, nc in self._adj[r][c]:
                        if self._player_grid[nr][nc] == HIDDEN:
                            self._player_grid[nr][nc] = self._adj_counts[nr][nc]
                            self._revealed_count += 1
                            if self._adj_counts[nr][nc] == 0:
                                queue.append((nr, nc))

        if self._revealed_count == self._rows * self._cols - self._mine_count:
            self._game_state = GameState.WON
        return True

    # ------------------------------------------------------------------
    # Known-mine tracking (used by solver)
    # ------------------------------------------------------------------

    def mark_known_mine(self, row: int, col: int) -> None:
        """Mark a cell as a known mine (solver use only)."""
        if not (0 <= row < self._rows and 0 <= col < self._cols):
            raise IndexError(f"({row}, {col}) out of bounds")
        if self._player_grid[row][col] != HIDDEN:
            raise ValueError(f"Cell ({row}, {col}) is not hidden")
        if (row, col) in self._known_mines:
            raise ValueError(f"Cell ({row}, {col}) is already marked as known mine")
        self._player_grid[row][col] = KNOWN_MINE
        self._known_mines.add((row, col))

    def is_known_mine(self, row: int, col: int) -> bool:
        return (row, col) in self._known_mines

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    @property
    def mine_count(self) -> int:
        return self._mine_count

    @property
    def total_cells(self) -> int:
        return self._rows * self._cols

    @property
    def revealed_count(self) -> int:
        return self._revealed_count

    @property
    def game_state(self) -> GameState:
        return self._game_state

    @property
    def won(self) -> bool:
        return self._game_state == GameState.WON

    @property
    def lost(self) -> bool:
        return self._game_state == GameState.LOST

    @property
    def game_over(self) -> bool:
        return self._game_state != GameState.ONGOING

    # USE WITH CARE — only for simulation purposes
    def is_mine(self, row: int, col: int) -> bool:
        return self._mine_grid[row][col]

    def is_revealed(self, row: int, col: int) -> bool:
        return self._player_grid[row][col] >= 0

    def is_hidden(self, row: int, col: int) -> bool:
        return self._player_grid[row][col] == HIDDEN

    def player_value(self, row: int, col: int) -> int:
        """Raw value of the player grid at (row, col)."""
        return self._player_grid[row][col]

    def neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        return list(self._adj[row][col])

    def adj_mines_num(self, row: int, col: int) -> int:
        """Adjacent mine count. Raises ValueError if cell is not revealed."""
        if self._player_grid[row][col] < 0:
            raise ValueError("Cell must be revealed to query adjacent mines")
        return self._adj_counts[row][col]

    def adj_remaining_mines(self, row: int, col: int) -> int:
        """Mines adjacent to (row, col) that are not yet marked known."""
        return self._adj_counts[row][col] - sum(
            1 for nr, nc in self._adj[row][col] if (nr, nc) in self._known_mines
        )

    def remaining_total_mines(self) -> int:
        """Total number of unknown mines remaining on the board."""
        return self._mine_count - len(self._known_mines)

    def unrevealed_neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        return [
            (nr, nc)
            for nr, nc in self._adj[row][col]
            if self._player_grid[nr][nc] == HIDDEN
        ]

    def count_unrevealed_neighbors(self, row: int, col: int) -> int:
        return sum(
            1 for nr, nc in self._adj[row][col]
            if self._player_grid[nr][nc] == HIDDEN
        )

    def frontier(self) -> list[tuple[int, int]]:
        """All hidden cells adjacent to at least one revealed cell."""
        return [
            (r, c)
            for r in range(self._rows)
            for c in range(self._cols)
            if self._player_grid[r][c] >= 0
            and any(
                self._player_grid[nr][nc] == HIDDEN
                for nr, nc in self._adj[r][c]
            )
        ]

def print_board(board: Board, show_mines: bool = False):
    """Print the board state in a user-friendly format."""
    header = "    " + "  ".join(str(c) for c in range(board.cols))
    print(header)
    print("    " + "---" * board.cols)
    for r in range(board.rows):
        row_str = f"{r:>2} |"
        for c in range(board.cols):
            v = board.player_value(r, c)
            if v == KNOWN_MINE:
                row_str += " * "
            elif v == HIDDEN:
                row_str += " . "
            else:
                row_str += f" {v} "
        print(row_str)