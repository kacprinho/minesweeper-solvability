"""
Batch Simulation Pipeline

Plays a given number of boards in parallel, giving each board up to 10
seconds to finish by default. A board that times out is retried with a different
first-click cell, up to 3 total tries by default, before being skipped
and logged. Results are written to chunked parquet files as they
accumulate, so a crash or interrupt won't lose all progress. 
The output directory will contain:

    - games/ : parquet files containing game records
    - decisions/ : parquet files containing guess records
    - skipped_boards.csv : CSV file logging skipped boards and the reason
"""

import multiprocessing as mp
import os
import random
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from queue import Empty

import pandas as pd

from src.simulation.board_runner import play_board 


def _play_one_attempt(seed, result_queue):
    """Runs in its own process. Puts a result tuple on the queue rather
    than returning anything, since this is the target of a Process, not
    a function call we can get a return value from directly."""
    try:
        game_record, decision_records = play_board(seed)
        result_queue.put(("ok", game_record, decision_records))
    except Exception as exc: 
        result_queue.put(("error", str(exc)))


def _flush_chunk(games_buffer, decisions_buffer, out_dir, chunk_idx):
    games_dir = Path(out_dir) / "games"
    decisions_dir = Path(out_dir) / "decisions"
    games_dir.mkdir(parents=True, exist_ok=True)
    decisions_dir.mkdir(parents=True, exist_ok=True)

    games_df = pd.DataFrame([asdict(g) for g in games_buffer])
    games_df.to_parquet(games_dir / f"games_{chunk_idx:05d}.parquet", index=False)

    if decisions_buffer:
        decisions_df = pd.DataFrame([asdict(d) for d in decisions_buffer])
        decisions_df.to_parquet(decisions_dir / f"decisions_{chunk_idx:05d}.parquet", index=False)


def _log_skipped(seed, reason, out_dir):
    path = Path(out_dir) / "skipped_boards.csv"
    write_header = not path.exists()
    with open(path, "a") as f:
        if write_header:
            f.write("seed,reason\n")
        f.write(f"{seed},{reason}\n")


def run_batch(
    num_games: int,
    num_workers: int | None = None,
    chunk_size: int = 500,
    per_game_timeout: float = 10.0,
    max_attempts: int = 3,
    board_size: tuple[int, int] = (30, 30),
    out_dir: str = "../data/raw",
    print_every: int = 50,
) -> None:
    """
    Run a batch of Minesweeper games in parallel, with a timeout and retry mechanism for each game.
    """
    num_workers = num_workers or (os.cpu_count() or 4)
    rng = random.Random()

    pending = deque(range(num_games))
    attempts_used: dict[int, int] = {}
    active: list[dict] = []

    games_buffer, decisions_buffer = [], []
    chunk_idx = 0
    completed = 0
    skipped = 0
    start_time = time.perf_counter()

    def launch(seed):
        q = mp.Queue()
        p = mp.Process(target=_play_one_attempt, args=(seed, q))
        p.start()
        active.append(
            {"process": p, "queue": q, "seed": seed, "start": time.perf_counter()}
        )

    print(
        f"Starting {num_games} games, {num_workers} workers, "
        f"{per_game_timeout}s/attempt, max {max_attempts} attempts/board",
        flush=True,
    )

    try:
        while pending or active:
            # Keep every worker slot busy
            while pending and len(active) < num_workers:
                seed = pending.popleft()
                attempts_used[seed] = 1
                launch(seed) 

            still_active = []
            for slot in active:
                seed = slot["seed"]
                try:
                    result = slot["queue"].get_nowait()
                except Empty:
                    if time.perf_counter() - slot["start"] > per_game_timeout:
                        result = ("timeout",)
                    else:
                        still_active.append(slot)
                        continue

                if slot["process"].is_alive():
                    slot["process"].terminate()
                slot["process"].join()

                if result[0] == "ok":
                    _, game_record, decision_records = result
                    games_buffer.append(game_record)
                    decisions_buffer.extend(decision_records)
                    completed += 1

                    if completed % print_every == 0:
                        elapsed = time.perf_counter() - start_time
                        print(
                            f"[{elapsed:8.1f}s] completed {completed}/{num_games} "
                            f"(skipped {skipped})",
                            flush=True,
                        )

                    if len(games_buffer) >= chunk_size:
                        _flush_chunk(games_buffer, decisions_buffer, out_dir, chunk_idx)
                        games_buffer, decisions_buffer = [], []
                        chunk_idx += 1
                else:
                    reason = result[0] if result[0] != "error" else f"error: {result[1]}"
                    if attempts_used[seed] < max_attempts:
                        attempts_used[seed] += 1
                        launch(seed)
                    else:
                        skipped += 1
                        print(
                            f"board {seed}: skipped after {attempts_used[seed]} attempts "
                            f"({reason})",
                            flush=True,
                        )
                        _log_skipped(seed, reason, out_dir)

            active = still_active
            time.sleep(0.05)

    finally:
        if games_buffer:
            _flush_chunk(games_buffer, decisions_buffer, out_dir, chunk_idx)
        elapsed = time.perf_counter() - start_time
        print(
            f"\nDone: {completed} completed, {skipped} skipped, "
            f"{elapsed / 60:.1f} min total",
            flush=True,
        )