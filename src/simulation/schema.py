from dataclasses import dataclass

"""
Game record schema which stores features of a given board.
game_id: Unique identifier for the game which will be the random seed used to generate the board
num_cols: Number of columns in the board
num_rows: Number of rows in the board
mine_count: Total number of mines in the board
remaining_mines: Number of mines remaining on the board at the end of the game
num_guesses: Number of guesses made in the game
success_prob: Probability of success given the guesses made in the game 
(calculated as the product of (1 - probability of mine) for each guess made)
min_guess_prob: The lowest probability in a given game of a guess being a mine
max_guess_prob: The highest probability in a given game of a guess being a mine
avg_guess_prob: The average probability in a given game of a guess being a mine
win_no_guess: Boolean indicating whether the game was fully solvable without any guesses
win: Boolean indicating whether the game was won
first_click: The coordinates of the first click made in the game
cells_revealed_before_first_guess: The number of cells revealed before the first guess was made
"""
@dataclass
class GameRecord:
    game_id: str
    num_cols: int
    num_rows: int
    mine_count: int
    remaining_mines: int
    num_guesses: int
    success_prob: float
    min_guess_prob: float
    max_guess_prob: float
    avg_guess_prob: float
    win_no_guess: bool
    win: bool
    first_click: tuple[int, int]
    cells_revealed_before_first_guess: int

"""
Guess Record schema which stores features of a given guess made in a game.
game_id: Unique identifier for the game which will be the random seed used to generate the board
guess_location: The coordinates of the guess made in the game
is_edge: Boolean indicating whether the guess was made on an edge of the board
is_corner: Boolean indicating whether the guess was made on a corner of the board
guess_number: The number of the guess made in the game (1-indexed)
guess_prob: The probability of the guess being a mine
guess_result: The result of the guess made in the game ("safe" or "mine")
component_size: The size of the connected component of hidden cells that the guess was made in
num_valid_configs: The number of valid configurations of mines in the connected component of hidden cells that the guess was made in
mines_remaining_at_guess: The number of mines remaining on the board at the time of the guess
"""
@dataclass
class GuessRecord:
    game_id: str
    guess_location: tuple[int, int]
    is_edge: bool
    is_corner: bool
    guess_number: int
    guess_prob: float
    guess_result: str  # "safe" or "mine"
    component_size: int
    num_valid_configs: int
    mines_remaining_at_guess: int
    