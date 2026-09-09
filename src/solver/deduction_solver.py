from src.game.board import Board


class Solver:
    """
    Deduction solver using given patterns.

    Applies forced-move patterns to a board until no pattern can make
    further progress. Returns a status string describing
    how the run ended.
    """

    def solve(self, board: Board) -> str:
        """
        Check for forced moves and apply them until no more can be found.

        Returns:
            "won"   if every safe cell was revealed by forced moves
            "stuck" if our patterns have been exhausted and the board isn't cleared 
            "lost"  if a forced reveal hit a mine (should not happen)
        """
        changed = True
        while changed and not board.game_over:
            changed = False
            frontier = board.frontier()
            if self._single_point(board, frontier):
                changed = True
            if self._subset_rule(board, frontier):
                changed = True

        if board.won:
            return "won"
        if board.lost:
            return "lost"
        return "stuck"

    # ------------------------------------------------------------------
    # Patterns
    # ------------------------------------------------------------------

    def _single_point(self, board: Board, frontier: list[tuple[int, int]]) -> bool:
        """
        Basic constraint deductions

        For each revealed cell:
          - remaining adjacent mines == 0
                -> all hidden neighbors are safe, reveal them
          - remaining adjacent mines == len(hidden neighbors)
                -> all hidden neighbors are mines, mark them
        """
        changed = False
        #Loop through all frontier cells and apply the single point rule
        for r, c in frontier:
            if not board.is_revealed(r, c):
                continue
            remaining = board.adj_remaining_mines(r, c)
            hidden = board.unrevealed_neighbors(r, c)
            # If the number of remaining mines is zero, reveal all hidden neighbors.
            if remaining == 0 and hidden:
                for nr, nc in hidden:
                    board.reveal(nr, nc)
                changed = True
            # If the number of remaining mines equals the number of hidden neighbors, mark all hidden neighbors as mines.
            elif remaining == len(hidden) and hidden:
                for nr, nc in hidden:
                    board.mark_known_mine(nr, nc)
                changed = True
        return changed

    def _subset_rule(self, board: Board, frontier: list[tuple[int, int]]) -> bool:
        """
        Containment deduction: if one cell's hidden neighbors are a subset of
        another's, the extra cells must carry the difference in mines.

        If the extra cells are the same number as the difference in remaining mines,
        they are all mines, so mark them.
        If the difference in remaining mines is zero, they are all safe, so reveal them.
        """
        changed = False

        # Loop through all pairs of frontier cells and get remaining mines and hidden neighbors for each cell
        for i, (r1, c1) in enumerate(frontier):
            a = board.adj_remaining_mines(r1, c1)
            sa = set(board.unrevealed_neighbors(r1, c1))
            if a <= 0 or not sa:
                continue

            for r2, c2 in frontier[i + 1:]:
                b = board.adj_remaining_mines(r2, c2)
                if b <= 0:
                    continue
                sb = set(board.unrevealed_neighbors(r2, c2))
                if not sb:
                    continue

                #Check a is a subset of b
                if sa.issubset(sb):
                    extra = sb - sa
                    if extra:
                        need = b - a
                        #Check if number of extra cells is equal to the difference in remaining mines
                        if need == len(extra):
                            for nr, nc in extra:
                                board.mark_known_mine(nr, nc)
                            changed = True
                        #Check if the difference in remaining mines is zero, then reveal all extra cells
                        elif need == 0:
                            for nr, nc in extra:
                                board.reveal(nr, nc)
                                if board.lost:
                                    break
                            changed = True

                #Check b is a subset of a
                elif sb.issubset(sa):
                    extra = sa - sb
                    if extra:
                        need = a - b
                        if need == len(extra):
                            for nr, nc in extra:
                                board.mark_known_mine(nr, nc)
                            changed = True
                        elif need == 0:
                            for nr, nc in extra:
                                board.reveal(nr, nc)
                                if board.lost:
                                    break
                            changed = True
                            
        return changed