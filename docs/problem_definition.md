# Defining the problem

## 1. The main question

With this project, the main question we are asking is the following:

**For a 30×30 Minesweeper board with 270 mines, how much luck (RNG) is required to win, and is it possible to win without any?**

The main purpose of answering this is to see whether it is realistic for a human player to complete this board. As someone who has tried to complete this for over 4 months and only gotten as far as 70 mines left, this will be either motivation to carry on or a sign to accept defeat!

We need to keep in mind that each board is different, and so for a specific board the answer to our question might vary. This means we can split the question up into 2 subproblems.

### The "is it possible to win without luck?" part

The first is the problem of determining whether a given board is **solvable**. What we mean by solvable will be explained below, but the point of this is that we need to be able to figure out if we can win the game without making any guesses. However, since we are working with such a high mine density (over *30%*), it will most likely be very rare to come across a lot of these boards, if any. This is where the second part comes in.

### The "how much luck is needed?" part

Once we are able to decide that a board isn't solvable, the second, more difficult subproblem is calculating the aggregate win probability under **optimal play**. This also means that we need to be able to play optimally in each run, which is a problem in itself which we will get to later. The win probability will essentially be our measure of the luck required, and aggregating it will give us an average (the answer to our question).

## 2. The plan

The outline of what we are going to do in this project is as follows:

- **Game engine:** 
    - Generate a random board which fits certain solvability criteria
    - Implement standard game rules such as reveal and flood fill
    - Be able to detect win/loss, i.e. check if a mine is revealed
    - Configurable mine count & board size to make testing on smaller boards easier
    - Board representation so we can analyse specific boards of interest
    - Set various conventions such as first click rules
- **Deduction solver:**
    - Use known rules and patterns to solve a board up to a point without any guessing
    - Stop when no more reveals can be made without guessing
    - Test on smaller board to validate its correctness
- **Probability engine:**
    - Extract the ambiguous regions and break them down into *independent* chunks
    - Figure out every possible combination of mines for each chunk
    - Weigh in the mine count to calculate probability that each ambiguous cell is a mine
- **Simulation runner:**
    - Generate a board and use the deduction solver to get to an ambiguous point in the board
    - Randomise the first click so we get an even sample of starts
    - Log statistics for each decision point as raw data
    - Guess the lowest probability cell at each point to represent a normal user trying to complete the board
    - Log the results for each board as raw data
    - Repeat for numerous boards and run in parallel to save time
- **Initial EDA:**
    - Answer the core question by calculating statistics and plotting distributions
- **Feature Extraction:**
    - Turn each decision point into usable data with multiple features
- **Deeper analysis:**
    - Create a model to see if early features of the board can predict future guess count or difficulty
    - Run hypothesis tests to see whether patterns in different regions differ significantly in guess risk
    - Group certain ambiguous patterns together using clustering
    - Rerun simulations at different mine densities to see how difficult this configuration really is
- **Report**
    - Answer the core question using our initial EDA
    - Give further insights into strategy and reasoning using the deeper analysis

## 3. Game rules & Conventions

- For each board, the mine count and board size will be decided beforehand.
- First click will always be a **guaranteed zero cell** meaning that board generation has to come after the first click. 
- Mine placement will be on a uniform random basis, while fitting the first click constraint.
- The game follows standard 8-directional adjacency rules, meaning that cells that touch on the diagonal still count as adjacent.
- The game will only feature the following two mechanics:
    - **Reveal**
    - **Flood-fill**
- There will be no flag implementation, since it is simply a UI convenience, it doesn't give any extra functionality to the game.
- When simulating, we will randomise the cell which is clicked first so we can get an even spread and use this as a feature for analysis.

## 4. Definitions

- **Forced move**: A move where we can figure out the mine arrangement purely from the current information
- **Guess**: a decision point where deduction is exhausted and so there are multiple possible mune configurations, meaning a choice has to be made
- **Solvable** (project-specific meaning): a board is solvable if and only if a complete sequence of *forced moves alone* exists that clears every non-mine cell, given the fixed opening reveal. (Note: this is a stronger claim than "some sequence of clicks wins it" — a board a lucky player could win by guessing does NOT count as solvable under this definition)
- **Optimal play**: Complete all possible forced moves, then at each guess point, choose the cell with the lowest computed mine probability (in guesses with equal probability, choose randomly)
- **Aggregate**: A statistic combined/summarized across many simulated boards (e.g. "average win rate across 10,000 games"), as opposed to a single board's result
- **Adjacent**: If two mines are touching in any way (up, down, left, right or diagonally), they are adjacent
- **Frontier**: A frontier cell is any revealed number cell that has at least one unrevealed neighbour cell.
- **Component**: A chunk of cells which can be focused on separately from the board, but not necessarily *independently* (clarified in below section)

## 5. Some clarifications

- It is entirely possible that a board could be solvable and not solvable, depending on where you make the first click. For our purposes, we will only check solvability for the randomly selected first click in that instance. This is because in reality, most players won't replay a board and select a different start click - you have already seen part of the board from the first run and so this could be considered as cheating.
- Since extracted components are not strictly independent from each other (global mine count constraints them),they are coupled; probabilities are computed by convolving mine count distributions for each component (including a virtual component for non-frontier cells which is simply a mine count that is at most the number of these cells) and condtioning on the global mine count.
- In this project, we are **NOT** modelling an actual player. This means we aren't taking into account, click speed, human error or flagging strategy. Here we are purely focusing on win rate under optimal play.

## 6. Future plans

After reporting my findings, there are a few ways in which I could extend these into a larger project:
    - A training website which will present the user with various patterns and give advise on probabilities and what the best move is in each situation, backed by the real data gathered in this project.
    - An RL model which we can train to see how it would perform on this board and compare to our data to see if it can outperform the calculated win rate or if it simply gets lucky.
    - Collecting data for multiple mine densities beyond our defined board to see how difficult our target really is.

## 7. Things to resolve later

Here I am making a list of issues that we may encounter later on so that we can keep track of how they will be resolved:

- How do we manage very large components which will take too long to enumerate?