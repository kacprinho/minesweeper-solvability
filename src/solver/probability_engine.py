import math

from src.game.board import Board


class ProbabilityEngine:
    """
    Calculate mine probabilities for hidden cells.

    All mine placements are enumerated per component and combined with the
    global mine count. P=0 cells are guaranteed safe, P=1 cells are
    guaranteed mines, so we can resolve these. Otherwise
    we can reveal the cell with the lowest probability of being a mine as a "best guess".
    """

    def probabilities(self, board: Board) -> dict[tuple[int, int], float]:
        """
        Return {cell: probability the cell holds a mine} for every hidden cell.
        """
        weights = self._mine_weights(board)
        return {cell: num / den for cell, (num, den) in weights.items()}

    def resolve(self, board: Board) -> bool:
        """
        Reveal every P=0 cell and mark every P=1 cell.

        Returns:
            True if at least one cell was revealed or marked.
        """
        probabilities = self.probabilities(board)
        moved = False

        # Mark every certain mine first, using the probabilities computed
        # from the board's state *before* any of these changes are applied.
        for (r, c), prob in probabilities.items():
            if board.game_over:
                break
            if prob == 1 and board.is_hidden(r, c):
                board.mark_known_mine(r, c)
                moved = True

        # Then reveal every certainly-safe cell. Still using the same
        # pre-computed probabilities dict, since marking known mines above does
        # not change which cells were already proven safe.
        for (r, c), prob in probabilities.items():
            if board.game_over:
                break
            if prob == 0 and board.is_hidden(r, c):
                board.reveal(r, c)
                moved = True
        return moved

    def best_guess(self, board: Board) -> tuple[int, int]:
        """
        Return the hidden cell with the lowest probability of being a mine.

        If multiple cells tie for lowest probability, one is chosen
        arbitrarily. If no hidden cells remain, raises ValueError.
        """
        probabilities = self.probabilities(board)
        best_cell = min(probabilities, key=probabilities.get)
        return best_cell

    def _mine_weights(self, board: Board) -> dict[tuple[int, int], tuple[int, int]]:
        """
        Return {cell: (mine_weight, total_weight)} for every hidden cell.

        Rational probabilities are kept as integer pairs so P=0/P=1
        decisions are exact. The total weight is the number of mine
        placements consistent with the revealed clues and the global mine
        count.
        """

        # Split the board's constraints into independent groups (components),
        # each of which can be enumerated on its own.
        components = self._components(board)

        # For each component, run enumeration to get the mine count distribution and the per-cell mine counts.
        component_mine_count_dists = []
        component_cell_mine_counts = []
        for cells, cons in components:
            mine_count_dist, cell_mine_counts = self._enumerate_component(cells, cons)
            component_mine_count_dists.append(mine_count_dist)
            component_cell_mine_counts.append(cell_mine_counts)

        # Collect every hidden cell on the board.
        hidden_cells = [
            (row, col)
            for row in range(board.rows)
            for col in range(board.cols)
            if board.is_hidden(row, col)
        ]

        # Cells belonging to some component are "constrained" (they touch at
        # least one frontier clue). Anything hidden but not in any component
        # is "unconstrained", meaning it forms an implicit extra component whose
        # cells are all interchangeable (any subset can hold mines).
        constrained_cells = set()
        for cells, _ in components:
            constrained_cells.update(cells)
        unconstrained_cells = [cell for cell in hidden_cells if cell not in constrained_cells]
        num_unconstrained = len(unconstrained_cells)
        remaining_mines = board.remaining_total_mines()

        # Convolve every component's mine-count distribution together to get
        # the joint distribution across all constrained cells: combined[ t] =
        # number of ways the constrained region as a whole uses t mines.
        combined_constrained_dist = [1]
        for dist in component_mine_count_dists:
            combined_constrained_dist = self._convolve(combined_constrained_dist, dist)

        # For every possible split of remaining
        # mines between "constrained cells use t" and "unconstrained cells
        # use remaining_mines - t", count the ways and sum them all up.
        # This is the total number of globally valid mine placements.
        denominator = 0
        for t, count in enumerate(combined_constrained_dist):
            denominator += count * self._choose(num_unconstrained, remaining_mines - t)
        if denominator == 0:
            raise ValueError("no mine placement is consistent with the board")

        weights: dict[tuple[int, int], tuple[int, int]] = {}

        if components:
            # Prefix/suffix convolutions let us cheaply get "every component
            # except this one" without re-enumerating anything: pref[i] is
            # components 0..i combined, suff[i] is components i..end combined.
            num_components = len(components)
            prefix_dists = [None] * num_components
            running = [1]
            for i in range(num_components):
                running = self._convolve(running, component_mine_count_dists[i])
                prefix_dists[i] = running
            suffix_dists = [None] * num_components
            running = [1]
            for i in range(num_components - 1, -1, -1):
                running = self._convolve(component_mine_count_dists[i], running)
                suffix_dists[i] = running

            for i, (cells, _cons) in enumerate(components):
                for cell in cells:
                    # Distribution of "how many mines do all OTHER
                    # components use", obtained by convolving everything
                    # before component i with everything after it.
                    left = prefix_dists[i - 1] if i > 0 else [1]
                    right = suffix_dists[i + 1] if i + 1 < num_components else [1]
                    other_components_dist = self._convolve(left, right)

                    # Sum over every way to split the remaining mine budget:
                    # t mines from this cell's own component (only counting
                    # configurations where the cell IS a mine), s mines from
                    # every other component, and the rest from unconstrained
                    # cells.
                    numerator = 0
                    for t, count_with_cell_as_mine in enumerate(component_cell_mine_counts[i][cell]):
                        if count_with_cell_as_mine == 0:
                            continue
                        inner_sum = 0
                        for s, other_ways in enumerate(other_components_dist):
                            if other_ways == 0:
                                continue
                            inner_sum += other_ways * self._choose(
                                num_unconstrained, remaining_mines - t - s
                            )
                        numerator += count_with_cell_as_mine * inner_sum
                    weights[cell] = (numerator, denominator)

        if unconstrained_cells:
            # By symmetry every unconstrained cell has the same probability:
            # fix one specific unconstrained cell as a mine, then distribute
            # the remaining (remaining_mines - t - 1) mines among the other
            # (num_unconstrained - 1) unconstrained cells.
            numerator_unconstrained = 0
            for t, count in enumerate(combined_constrained_dist):
                if count == 0:
                    continue
                numerator_unconstrained += count * self._choose(
                    num_unconstrained - 1, remaining_mines - t - 1
                )
            for cell in unconstrained_cells:
                weights[cell] = (numerator_unconstrained, denominator)

        return weights

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    def _components(self, board: Board) -> list[tuple[list, list]]:
        """
        Split constraints into independent groups via union-find on cells.
        This essentially works by grouping every hidden cell that appears together in at least one constraint, 
        then collecting all constraints touching any of those cells into the same component.
        """
        parent = {}

        # Returns the root of the union-find tree for x, creating a new root if x is not yet in the structure. 
        # Path compression is applied to flatten the tree for efficiency.
        def find(x):
            # Create a new root for x if it doesn't exist yet, then follow parent links to find the root.
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]  
                x = parent[x]
            return x

        # Merges the union-find trees of a and b, making the root of b point to the root of a.
        def union(a, b):
            # Find the roots of a and b, then make the root of b point to the root of a if they are different.
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        # Loop through frontier cells and build one constraint cell: the
        # set of hidden cells it borders, plus how many mines must still be
        # found among them.
        frontier_cells = board.frontier()
        constraints = []
        for cell in frontier_cells:
            constraints.append((frozenset(board.unrevealed_neighbors(*cell)), board.adj_remaining_mines(*cell)))

        # First pass: union every hidden cell that appears together in the
        # same constraint.
        for hidden_cells_in_constraint, _ in constraints:
            # Pick one cell as the "representative" and union it with every other cell in the same constraint. 
            # This ensures that all cells in the same constraint are connected in the union-find structure.
            first_cell = next(iter(hidden_cells_in_constraint))
            for other_cell in hidden_cells_in_constraint:
                union(first_cell, other_cell)

        # Second pass: group constraints by which root their
        # cells now resolve to. Each group of constraints is an independent component.
        grouped = {}
        for hidden_cells_in_constraint, remaining in constraints:
            # Find the root of one of the hidden cells in the constraint, 
            # and use it as the key to group constraints together.
            root = find(next(iter(hidden_cells_in_constraint)))
            entry = grouped.setdefault(root, {"cells": set(), "cons": []})
            entry["cells"].update(hidden_cells_in_constraint)
            entry["cons"].append((hidden_cells_in_constraint, remaining))

        # Convert each component's constraints from raw cell coordinates to
        # local indices into a sortedcell list — this is what allows us to
        # perform enumeration without needing to know the global board coordinates.
        components = []
        for entry in grouped.values():
            # Sort the cells in the component to have a consistent order, 
            # and create a mapping from cell coordinates to their index in the sorted list.
            cells = sorted(entry["cells"])
            index_of = {cell: i for i, cell in enumerate(cells)}
            local_cons = [
                (tuple(sorted(index_of[cell] for cell in cell_set)), remaining)
                for cell_set, remaining in entry["cons"]
            ]
            components.append((cells, local_cons))

        return components

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def _enumerate_component(self, cells, cons):
        """
        Recursively enumerate all possible mine placements for one component.

        Args:
            cells: the cells involved in this component
            cons:  (indices, remaining_mines) constraint pairs

        Returns:
            size_dist: list which represents the number of valid placements using t mines, where size_dist[t] = count
            ones: {cell: list} where ones[cell][t] = placements using t mines
                  in which that cell holds a mine
        """
        num_cells = len(cells)
        num_constraints = len(cons)

        # Record which constraints each cell participates in, so we can
        # later reorder cells by how constrained they are.
        cell_constraints = [[] for _ in range(num_cells)]
        constraint_targets = [c for _, c in cons]
        for c, (idxs, _) in enumerate(cons):
            for idx in idxs:
                cell_constraints[idx].append(c)

        # Reorder cells so the most constrained ones (touching the most
        # constraints) get decided first
        order = sorted(range(num_cells), key=lambda v: (-len(cell_constraints[v]), v))
        new_position_of = {old: new for new, old in enumerate(order)}
        cons_reordered = [
            (tuple(sorted(new_position_of[i] for i in idxs)), r)
            for idxs, r in cons
        ]

        # Rebuild cell_constraints against the new ordering, and precompute,
        # for every constraint and every decision depth v, how many of that
        # constraint's cells are still undecided once cells 0..v-1 are
        # assigned. This lets the search reject impossible branches instantly.
        cell_constraints = [[] for _ in range(num_cells)]
        undecided_count_at_depth = []
        for c, (idxs, r) in enumerate(cons_reordered):
            suffix_counts = [0] * (num_cells + 1)
            j = 0
            for v in range(num_cells):
                while j < len(idxs) and idxs[j] < v:
                    j += 1
                suffix_counts[v] = len(idxs) - j
            undecided_count_at_depth.append(suffix_counts)
            for idx in idxs:
                cell_constraints[idx].append(c)

        size_dist = [0] * (num_cells + 1)
        ones = {cell: [0] * (num_cells + 1) for cell in cells}
        assignment = [0] * num_cells
        mines_used_per_constraint = [0] * num_constraints

        def search(v):
            if v == num_cells:
                # All cells decided — verify every constraint is
                # exactly satisfied (earlier pruning only ruled out
                # impossible branches, it doesn't guarantee equality).
                for c in range(num_constraints):
                    if mines_used_per_constraint[c] != constraint_targets[c]:
                        return
                total_mines = sum(assignment)
                size_dist[total_mines] += 1
                for i in range(num_cells):
                    if assignment[i]:
                        # order[i] maps the search's reordered position
                        # back to the original cell index.
                        ones[cells[order[i]]][total_mines] += 1
                return

            # Reject early if any constraint is already broken, or
            # can no longer possibly be satisfied even in the best case.
            for c in range(num_constraints):
                if mines_used_per_constraint[c] > constraint_targets[c]:
                    return
                if mines_used_per_constraint[c] + undecided_count_at_depth[c][v] < constraint_targets[c]:
                    return

            # Try this cell as safe (0) then as a mine (1).
            for value in (0, 1):
                assignment[v] = value
                if value:
                    for c in cell_constraints[v]:
                        mines_used_per_constraint[c] += 1
                search(v + 1)
                if value:
                    for c in cell_constraints[v]:
                        mines_used_per_constraint[c] -= 1

        search(0)
        return size_dist, ones

    # ------------------------------------------------------------------
    # Arithmetic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _choose(n: int, k: int) -> int:
        """math.comb with out-of-range values returning 0 instead of raising."""
        if k < 0 or k > n:
            return 0
        return math.comb(n, k)

    @staticmethod
    def _convolve(a: list[int], b: list[int]) -> list[int]:
        """Convolve two mine-count distributions."""
        result = [0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            if ai == 0:
                continue
            for j, bj in enumerate(b):
                if bj:
                    result[i + j] += ai * bj
        return result