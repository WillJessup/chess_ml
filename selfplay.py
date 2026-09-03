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
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import chess

AGENT_MODULE = "agent"

def load_agent(name: str):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    mod = importlib.import_module(name)
    mod.PONDER_ENABLED = False
    return mod

def play_game(args_tuple):
    """Play one self-play game with early adjudication."""
    agent_module, depth, random_plies, noise_prob, max_plies, skip_plies, sample_every, seed = args_tuple
    
    agent = load_agent(agent_module)
    searcher = agent.Searcher()
    rng = random.Random(seed)
    
    board = chess.Board()
    searcher.new_game()
    sampled = []
    ply = 0
    white_scores = []

    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        legal = list(board.legal_moves)
        if not legal:
            break

        use_random = ply < random_plies or rng.random() < noise_prob
        if use_random:
            move = rng.choice(legal)
        else:
            try:
                move, score, _ = searcher.go(board, hard_ms=10_000.0, soft_ms=10_000.0, max_depth=depth)
                # Convert side-to-move score to White's perspective for stable tracking
                white_score = score if board.turn == chess.WHITE else -score
                white_scores.append(white_score)
            except Exception:
                move = None
                
            if move is None or move not in legal:
                move = rng.choice(legal)

        # Early Adjudication: Stop playing if the game is completely over or dead drawn
        if len(white_scores) >= 10:
            recent = white_scores[-10:]
            if all(s > 800 for s in recent):
                return sampled, 1.0  # White is completely winning
            if all(s < -800 for s in recent):
                return sampled, 0.0  # Black is completely winning
            if all(abs(s) < 15 for s in recent):
                return sampled, 0.5  # Dead draw

        if ply >= skip_plies and (ply - skip_plies) % sample_every == 0 and not board.is_check():
            sampled.append(board.fen())

        board.push(move)
        ply += 1

    if board.is_checkmate():
        return sampled, 0.0 if board.turn == chess.WHITE else 1.0
    
    return sampled, 0.5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--random-plies", type=int, default=8)
    ap.add_argument("--noise-prob", type=float, default=0.03)
    ap.add_argument("--max-plies", type=int, default=220)
    ap.add_argument("--skip-plies", type=int, default=10)
    ap.add_argument("--sample-every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agent-module", type=str, default=AGENT_MODULE)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use all cores minus 1 to keep the OS responsive
    cores = max(1, os.cpu_count() - 1)
    tasks = [
        (args.agent_module, args.depth, args.random_plies, args.noise_prob, 
         args.max_plies, args.skip_plies, args.sample_every, args.seed + i)
        for i in range(args.games)
    ]

    n_positions = 0
    t0 = time.monotonic()
    
    with out_path.open("w") as f:
        with ProcessPoolExecutor(max_workers=cores) as executor:
            for g, (fens, result) in enumerate(executor.map(play_game, tasks)):
                for fen in fens:
                    f.write(json.dumps({"fen": fen, "result": result}) + "\n")
                    n_positions += 1
                
                if (g + 1) % 20 == 0:
                    elapsed = time.monotonic() - t0
                    print(f"[{elapsed:6.1f}s] game {g + 1}/{args.games}, {n_positions} positions", file=sys.stderr)

if __name__ == "__main__":
    main()