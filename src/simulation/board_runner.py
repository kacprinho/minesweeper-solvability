from src.game.board import Board, print_board
from src.solver.deduction_solver import Solver
from src.solver.probability_engine import ProbabilityEngine
from src.simulation.schema import GameRecord, GuessRecord
import random

def play_board(seed) -> tuple[GameRecord, list[GuessRecord]]:
    """
    Plays a game of Minesweeper using the given seed and returns the game record and guess records.
    """
    # Create a board with the given seed
    board = Board(30,30,270,rng=random.Random(seed))
    
    # Initialize the solver
    solver = Solver()
    
    # Initialize the probability engine
    probability_engine = ProbabilityEngine()

    # Generate first click coordinates and reveal the cell
    first_click_row, first_click_col = random.randint(0, board.rows - 1), random.randint(0, board.cols - 1)
    board.reveal(first_click_row, first_click_col)

    # Play the game
    guess_count = 0
    guess_records = []
    product_win_probability = 1.0
    min_guess_prob = float('inf')
    max_guess_prob = float('-inf')
    avg_guess_prob = 0.0
    
    while not board.game_over:
        solver.solve(board)
        probability_engine.resolve(board)

        if guess_count == 0:
            cells_revealed_before_first_guess = board.revealed_count

        best_guess_cell, best_guess_prob = probability_engine.best_guess(board)
        guess_count += 1
        guess_record = GuessRecord(
            game_id=str(seed),
            guess_location=best_guess_cell,
            is_edge=best_guess_cell[0] == 0 or best_guess_cell[0] == board.rows - 1 or 
            best_guess_cell[1] == 0 or best_guess_cell[1] == board.cols - 1,
            is_corner=best_guess_cell[0] == 0 and best_guess_cell[1] == 0 or 
            best_guess_cell[0] == 0 and best_guess_cell[1] == board.cols - 1 or 
            best_guess_cell[0] == board.rows - 1 and best_guess_cell[1] == 0 or 
            best_guess_cell[0] == board.rows - 1 and best_guess_cell[1] == board.cols - 1,
            guess_number=guess_count,
            guess_prob=best_guess_prob,
            guess_result="mine" if board.is_mine(best_guess_cell[0], best_guess_cell[1]) else "safe",
            component_size=probability_engine.component_size(board, best_guess_cell),
            num_valid_configs=probability_engine.num_valid_configs(board, best_guess_cell),
            mines_remaining_at_guess=board.remaining_total_mines()
        )

        if best_guess_cell is None:
            break  # No valid guesses left
        guess_row, guess_col = best_guess_cell
        board.reveal(guess_row, guess_col)
        guess_records.append(guess_record)

        product_win_probability *= (1-best_guess_prob)
        min_guess_prob = min(min_guess_prob, best_guess_prob)
        max_guess_prob = max(max_guess_prob, best_guess_prob)
        avg_guess_prob = ((avg_guess_prob * (guess_count - 1)) + best_guess_prob) / guess_count

    # Create the game record
    game_record = GameRecord(
        game_id=str(seed),
        num_cols=board.cols,
        num_rows=board.rows,
        mine_count=board.mine_count,
        remaining_mines=board.remaining_total_mines(),
        num_guesses=guess_count,
        success_prob=product_win_probability,
        min_guess_prob=min_guess_prob,
        max_guess_prob=max_guess_prob,
        avg_guess_prob=avg_guess_prob,
        win_no_guess=True if guess_count == 0 else False,
        win=board.won,
        first_click=(first_click_row, first_click_col),
        cells_revealed_before_first_guess=cells_revealed_before_first_guess
    )

    return game_record, guess_records
