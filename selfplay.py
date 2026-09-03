"""Generate Texel-tuning training data by self-play.

Plays games with the *actual* engine's search (so sampled positions are ones
the engine really reaches), at a shallow fixed depth so generating tens of
thousands of games is cheap. Opening diversity comes from random moves for
the first few plies plus a small chance of a random move later on; without
that every game collapses onto the same few principal lines.

Usage:
    python selfplay.py --games 2000 --out data/games_0.jsonl --depth 5

Run several of these in parallel (different --out files, different --seed)
to use all your cores -- this step is offline, it does not run under the
tournament's 1-core/2GB limit.

IMPORTANT: edit AGENT_MODULE below (or pass --agent-module) to match the
filename of your engine, minus the .py extension. The engine file is expected
to be importable (i.e. on sys.path / in the same directory as this script).
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path

import chess

AGENT_MODULE = "agent"  # <-- change this if your engine file has another name


def load_agent(name: str):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    mod = importlib.import_module(name)
    # Self-play does not go through get_move(), so pondering is never started
    # anyway -- but disable it explicitly in case anything else in the module
    # triggers it, since a background thread per game is pure overhead here.
    mod.PONDER_ENABLED = False
    return mod


def play_game(
    agent,
    searcher,
    depth: int,
    random_plies: int,
    noise_prob: float,
    max_plies: int,
    skip_plies: int,
    sample_every: int,
    rng: random.Random,
):
    """Play one self-play game. Returns (list_of_fens, white_result)."""
    board = chess.Board()
    searcher.new_game()
    sampled: list[str] = []
    ply = 0

    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        legal = list(board.legal_moves)
        if not legal:
            break

        use_random = ply < random_plies or rng.random() < noise_prob
        if use_random:
            move = rng.choice(legal)
        else:
            try:
                move, _score, _depth_done = searcher.go(
                    board, hard_ms=10_000.0, soft_ms=10_000.0, max_depth=depth
                )
            except Exception:
                move = None
            if move is None or move not in legal:
                move = rng.choice(legal)

        if (
            ply >= skip_plies
            and (ply - skip_plies) % sample_every == 0
            and not board.is_check()
        ):
            sampled.append(board.fen())

        board.push(move)
        ply += 1

    if board.is_checkmate():
        # Side that just moved delivered mate; the side to move now lost.
        white_result = 0.0 if board.turn == chess.WHITE else 1.0
    elif board.is_game_over(claim_draw=True):
        white_result = 0.5
    else:
        # Hit max_plies without a decision -- treat as a draw rather than
        # guessing; adjudicating on eval would bias the very data we're
        # trying to fit.
        white_result = 0.5

    return sampled, white_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--depth", type=int, default=5, help="fixed search depth per move")
    ap.add_argument("--random-plies", type=int, default=8, help="fully random opening plies")
    ap.add_argument("--noise-prob", type=float, default=0.03, help="chance of a random move later in the game")
    ap.add_argument("--max-plies", type=int, default=220)
    ap.add_argument("--skip-plies", type=int, default=10, help="don't sample the opening")
    ap.add_argument("--sample-every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agent-module", type=str, default=AGENT_MODULE)
    args = ap.parse_args()

    agent = load_agent(args.agent_module)
    searcher = agent.Searcher()
    rng = random.Random(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_positions = 0
    t0 = time.monotonic()
    with out_path.open("w") as f:
        for g in range(args.games):
            fens, result = play_game(
                agent, searcher,
                depth=args.depth,
                random_plies=args.random_plies,
                noise_prob=args.noise_prob,
                max_plies=args.max_plies,
                skip_plies=args.skip_plies,
                sample_every=args.sample_every,
                rng=rng,
            )
            for fen in fens:
                f.write(json.dumps({"fen": fen, "result": result}) + "\n")
                n_positions += 1

            if (g + 1) % 20 == 0:
                elapsed = time.monotonic() - t0
                print(
                    f"[{elapsed:6.1f}s] game {g + 1}/{args.games}, "
                    f"{n_positions} positions so far",
                    file=sys.stderr,
                )

    print(f"done: {args.games} games, {n_positions} positions -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
