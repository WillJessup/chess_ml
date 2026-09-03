"""Run this FIRST, before generating any self-play data.

This sandbox has no network access to install python-chess, so none of this
pipeline has been executed against a real chess.Board -- only read carefully
against the engine's `_evaluate_jit`/`_passers_jit` logic. Run this script in
your own environment and eyeball the output before trusting the tuner.

    pip install python-chess numpy
    python sanity_check.py
"""

import chess
import numpy as np

import feature_extractor as fe


def check_starting_position():
    board = chess.Board()
    params = fe.default_params()
    score = fe.eval_cp(board, params)
    print(f"starting position eval: {score:.2f}  (expected exactly {12.0:.2f} = tempo, "
          f"since the position is symmetric and phase=24)")
    R, mg_f, eg_f, phase, wtm = fe.extract(board)
    print(f"  phase={phase} (expected 24), mg_frozen={mg_f}, eg_frozen={eg_f} (expected 0, 0)")
    print(f"  nonzero R entries: {int(np.count_nonzero(R))} (expected 0)")
    assert phase == 24
    assert abs(mg_f) < 1e-9 and abs(eg_f) < 1e-9
    assert np.count_nonzero(R) == 0
    assert abs(score - 12.0) < 1e-6
    print("  OK\n")


def check_passed_pawn():
    # White pawn on a7, one step from queening, completely unopposed and
    # unsupported; kings tucked away. Should show up as PASSED_OFF[6] (rank
    # index 6 = 7th rank) with support=0 (no white piece nearby / defends it).
    board = chess.Board("8/P6k/8/8/8/8/7K/8 w - - 0 1")
    R, mg_f, eg_f, phase, wtm = fe.extract(board)
    idx = fe._PASSED_OFF + 6
    print(f"lone a7 pawn: R[passed_rank=6] = {R[idx]} (expected 1.0)")
    print(f"  R[support_bonus] = {R[fe._SUPPORT_IDX]} (expected 0, king too far, no pawn defender)")
    print(f"  R[blockade_penalty] = {R[fe._BLOCKADE_IDX]} (expected 0, a8 is empty)")
    assert R[idx] == 1.0
    print("  OK\n")


def check_blockade():
    # Same pawn, but with a black piece sitting on a8 (the stop square).
    board = chess.Board("n7/P6k/8/8/8/8/7K/8 w - - 0 1")
    R, mg_f, eg_f, phase, wtm = fe.extract(board)
    print(f"blockaded a7 pawn: R[blockade_penalty] = {R[fe._BLOCKADE_IDX]} (expected 1.0)")
    assert R[fe._BLOCKADE_IDX] == 1.0
    print("  OK\n")


def check_doubled_isolated():
    # White pawns on a2 and a3 (doubled), h2 isolated (no pawn on g or h... wait
    # give it a neighbour-free file): a2/a3 doubled, and c2 isolated (no pawns
    # on b or d files).
    board = chess.Board("7k/8/8/8/8/8/P1P5/K7 w - - 0 1")
    # a2 + c2: neither doubled (different files) nor adjacent -> both isolated.
    R, *_ = fe.extract(board)
    print(f"a2+c2 pawns: R[isolated] = {R[fe._ISOLATED_IDX]} (expected 2.0), "
          f"R[doubled] = {R[fe._DOUBLED_IDX]} (expected 0.0)")
    assert R[fe._ISOLATED_IDX] == 2.0
    assert R[fe._DOUBLED_IDX] == 0.0

    board2 = chess.Board("7k/8/8/8/8/P7/P7/K7 w - - 0 1")
    R2, *_ = fe.extract(board2)
    print(f"a2+a3 pawns: R[doubled] = {R2[fe._DOUBLED_IDX]} (expected 1.0), "
          f"R[isolated] = {R2[fe._ISOLATED_IDX]} (expected 2.0, no b-file pawns either)")
    assert R2[fe._DOUBLED_IDX] == 1.0
    print("  OK\n")


def check_rook_files():
    # White rook on d1, no pawns on d-file for white, black has a pawn on d6
    # (semi-open for white), and white rook on e1 fully open (no pawns at all
    # on the e-file).
    board = chess.Board("4k3/3p4/8/8/8/8/8/3RR2K w - - 0 1")
    R, *_ = fe.extract(board)
    print(f"rooks d1(semi)+e1(open): R[rook_semi] = {R[fe._ROOK_SEMI_IDX]} (expected 1.0), "
          f"R[rook_open] = {R[fe._ROOK_OPEN_IDX]} (expected 1.0)")
    assert R[fe._ROOK_SEMI_IDX] == 1.0
    assert R[fe._ROOK_OPEN_IDX] == 1.0
    print("  OK\n")


if __name__ == "__main__":
    check_starting_position()
    check_passed_pawn()
    check_blockade()
    check_doubled_isolated()
    check_rook_files()
    print("All checks passed. Safe to move on to self-play generation.")
