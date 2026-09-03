"""AI Chessathon agent.

Interface required by the platform:
    get_move(fen: str, time_left_ms: int) -> str    # UCI, e.g. "e2e4" / "e7e8q"

Design
------
* Iterative-deepening negamax with alpha-beta and principal-variation search
* Transposition table (persisted between moves)
* Quiescence search with check evasions and delta pruning
* Move ordering: TT move -> MVV/LVA captures -> promotions -> killers -> history
* Null-move pruning, reverse futility, late-move reductions/pruning
* Tapered PeSTO piece-square evaluation
* Time management with hard and soft deadlines, plus a panic mode
* Polyglot opening book lookup before search

Everything the platform calls is wrapped so that an unexpected error or a slow
search can never produce an illegal move or a clock overrun: the worst case is
that a shallow but legal move is returned.
"""

import math
import operator
import os
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Hashable
from typing import Any

# numba caches compiled functions next to the source file by default. The
# platform filesystem is read-only outside /tmp, so that write fails there and
# every process pays a full recompile. Pointing the cache at the temp dir makes
# the first process compile and every later one load in well under a second
# (1.85 s -> 0.65 s on a cold start). numba reads this at import time, so it has
# to be set before the import below. The ordering here is load-bearing: moving
# this line under the imports silently disables the cache.
os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.gettempdir())

import chess
import chess.polyglot
import numpy as np
from numba import b1, int32, njit, uint64

"""Piece-square tables (PeSTO, tapered midgame/endgame).

Tables are written in *visual* order: index 0 is a8, index 63 is h1.
python-chess uses a1 = 0, so:
    white piece on square s  ->  table[s ^ 56]
    black piece on square s  ->  table[s]         (vertical mirror)
"""

MG_VALUE = {1: 82, 2: 337, 3: 365, 4: 477, 5: 1025, 6: 0}
EG_VALUE = {1: 94, 2: 281, 3: 297, 4: 512, 5: 936, 6: 0}

# Game-phase weight per piece type (total 24 at the starting position).
PHASE_INC = {1: 0, 2: 1, 3: 1, 4: 2, 5: 4, 6: 0}

MG_PAWN = [
     0,   0,   0,   0,   0,   0,   0,   0,
    98, 134,  61,  95,  68, 126,  34, -11,
    -6,   7,  26,  31,  65,  56,  25, -20,
   -14,  13,   6,  21,  23,  12,  17, -23,
   -27,  -2,  -5,  12,  17,   6,  10, -25,
   -26,  -4,  -4, -10,   3,   3,  33, -12,
   -35,  -1, -20, -23, -15,  24,  38, -22,
     0,   0,   0,   0,   0,   0,   0,   0,
]

EG_PAWN = [
     0,   0,   0,   0,   0,   0,   0,   0,
   178, 173, 158, 134, 147, 132, 165, 187,
    94, 100,  85,  67,  56,  53,  82,  84,
    32,  24,  13,   5,  -2,   4,  17,  17,
    13,   9,  -3,  -7,  -7,  -8,   3,  -1,
     4,   7,  -6,   1,   0,  -5,  -1,  -8,
    13,   8,   8,  10,  13,   0,   2,  -7,
     0,   0,   0,   0,   0,   0,   0,   0,
]

MG_KNIGHT = [
   -167, -89, -34, -49,  61, -97, -15, -107,
    -73, -41,  72,  36,  23,  62,   7,  -17,
    -47,  60,  37,  65,  84, 129,  73,   44,
     -9,  17,  19,  53,  37,  69,  18,   22,
    -13,   4,  16,  13,  28,  19,  21,   -8,
    -23,  -9,  12,  10,  19,  17,  25,  -16,
    -29, -53, -12,  -3,  -1,  18, -14,  -19,
   -105, -21, -58, -33, -17, -28, -19,  -23,
]

EG_KNIGHT = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
]

MG_BISHOP = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
]

EG_BISHOP = [
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
]

MG_ROOK = [
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
]

EG_ROOK = [
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20,
]

MG_QUEEN = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
]

EG_QUEEN = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
]

MG_KING = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
]

EG_KING = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
]

_MG_RAW = {1: MG_PAWN, 2: MG_KNIGHT, 3: MG_BISHOP, 4: MG_ROOK, 5: MG_QUEEN, 6: MG_KING}
_EG_RAW = {1: EG_PAWN, 2: EG_KNIGHT, 3: EG_BISHOP, 4: EG_ROOK, 5: EG_QUEEN, 6: EG_KING}

# Piece value baked into the table so eval is a single lookup per piece.
# Index 0 is an unused placeholder: python-chess piece types start at 1.
MG_TABLE: list[list[int]] = [[0] * 64]
EG_TABLE: list[list[int]] = [[0] * 64]
for _pt in range(1, 7):
    MG_TABLE.append([MG_VALUE[_pt] + v for v in _MG_RAW[_pt]])
    EG_TABLE.append([EG_VALUE[_pt] + v for v in _EG_RAW[_pt]])

# Numpy Arrays for Numba
mg_table_np = np.array(MG_TABLE, dtype=np.int32)
eg_table_np = np.array(EG_TABLE, dtype=np.int32)
phase_inc_np = np.array([0, 0, 1, 1, 2, 4, 0], dtype=np.int32)

PASSED_MG = [0, 0, 2, 5, 12, 22, 40, 0]
PASSED_EG = [0, 10, 18, 34, 60, 100, 150, 0]
passed_mg_np = np.array(PASSED_MG, dtype=np.int32)
passed_eg_np = np.array(PASSED_EG, dtype=np.int32)

_KP_SCALE = [0, 0, 0, 1, 2, 3, 4, 0]
kp_scale_np = np.array(_KP_SCALE, dtype=np.int32)

KING_DIST: list[list[int]] = [
    [max(abs((a & 7) - (b & 7)), abs((a >> 3) - (b >> 3))) for b in range(64)]
    for a in range(64)
]
king_dist_np = np.array(KING_DIST, dtype=np.int32)

# Numba Bitboard Constants
BB_ALL = np.uint64(0xFFFFFFFFFFFFFFFF)
NOT_FILE_A = np.uint64(0xFEFEFEFEFEFEFEFE)
NOT_FILE_H = np.uint64(0x7F7F7F7F7F7F7F7F)
OUTPOST_RANKS_W = np.uint64(0x0000FFFFFF000000)
OUTPOST_RANKS_B = np.uint64(0x000000FFFFFF0000)


# =============================================================================
# Search
# =============================================================================

# Late-move reduction depths. The step function this replaces (1, then 2 at
# move 8, then 3 at depth 6 and move 14) reduces a move at depth 20 exactly as
# hard as at depth 4, which is far too timid deep and too eager shallow. The
# log formula is what every modern engine uses.
_LMR: list[list[int]] = [[0] * 64 for _ in range(64)]
for _d in range(1, 64):
    for _i in range(1, 64):
        _LMR[_d][_i] = int(0.5 + math.log(_d) * math.log(_i) / 2.5)

INF = 1 << 30
MATE = 1000000
MATE_BOUND = MATE - 4096
MAX_PLY = 96

# Move-ordering constants
TT_BONUS = 1 << 22
CAPTURE_BONUS = 1 << 20
KILLER1_BONUS = (1 << 19) + 2
KILLER2_BONUS = (1 << 19) + 1

# Most-valuable-victim / least-valuable-attacker table
SEE_VALUE = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 20000, None: 100}

# Sort key for (score, move) pairs. C-level, and never touches the Move.
_SCORE_OF = operator.itemgetter(0)

# Transposition table entry flags
EXACT, UPPER, LOWER = 0, 1, 2

# Measured: a table filled to 500,000 buckets costs 59 MB on top of a 164 MB
# baseline. The platform allows 2 GB, so 500k was using 11% of the budget while
# a single move at the real control (120 s + 0.5 s) already writes 25-30k
# entries -- a 40-move game wants well over a million. 2M buckets costs about
# 240 MB and covers a whole game without evicting.
TT_MAX_ENTRIES = 2_000_000

# Iterative deepening normally starts at depth 1. The shallow iterations are
# cheap and they order the deep ones, but with a comfortable budget, jumping
# straight to depth 4 reaches about one ply further inside the same time
# (measured d9 -> d10 at a 120+1 budget). Do not raise this: at depth 7 the
# first iteration has no move ordering at all, so alpha-beta cannot prune and
# it comes out both shallower and slower than starting at 1.
START_DEPTH = 4
# ...but only when there is time to finish that first iteration. If it does not
# finish there is no searched move at all and the engine plays whatever move
# generation produced first -- Bxf7+, in two of five test positions at a 120 ms
# deadline. Depth 1 always finishes; depth 4 does not. Below this hard budget,
# iterate from 1 as before.
START_DEPTH_MIN_MS = 1500
# Keep the transposition table across moves. This was previously True, because
# a repetition or fifty-move draw scores 0 only on the path that reached it;
# those scores propagated into parent entries, the table is keyed by position
# alone, and on a later move the engine read back "this is a draw" for a
# position that no longer was. It rotted until the engine played Qxd5?? in a
# real game.
#
# Clearing the whole table every move fixed that by throwing away everything,
# including the 81% of entries that were fine -- the measured TT hit rate at
# interior nodes was only 19%. The narrower fix is to never *store* the tainted
# entries in the first place: see `_taint` in search(). With that in place the
# table can safely persist, which is what every real engine does.
TT_CLEAR_EACH_MOVE = False

WHITE = chess.WHITE
BLACK = chess.BLACK


class _Timeout(Exception):
    """Raised internally when the move deadline is reached."""


# numba signatures, named so each decorator fits on one line. numba ships no
# type stubs, so mypy sees @njit as untyped and drops the annotations below it;
# the narrow ignore on each decorator keeps the rest of the file strict-clean.
# A bitboard crosses the JIT boundary as a plain Python int and is a numpy
# uint64 inside the compiled body. Annotating it `int` is wrong on one side of
# that line and made mypy reject every numpy operation in these functions; numba
# takes its types from the explicit signatures below regardless.
Bitboard = Any

# Pawn shield. The evaluation has had no king-safety term of any kind -- PeSTO's
# king table was the whole of it -- so nothing opposed advancing the pawns in
# front of our own castled king. Whenever g4 gained a tempo or a table point,
# the score said yes, because it could not see the hole being opened.
#
# Counted, not modelled: how many of our own pawns stand on the two ranks in
# front of the king, across its file and the two beside it. A pawn pushed to the
# fourth rank has left the zone and counts as missing, which is the case the
# observation was about. Midgame only, and only while the enemy queen is on --
# in an endgame the king belongs in front of its pawns, not behind them.
SHIELD_PER_MISSING = 20
# Divide a passed pawn's bonus by this when an enemy piece stands on its stop
# square. Measured from game useCaJ4C: a black d3 pawn with a white knight on d2
# collected PASSED_EG[5] = 100 in full, and the eval read +325 in a position
# worth 0 (see claude/blockaded-passer-blind-spot.md). The bonus was identical
# whether the knight sat on d2 or on g1, so the evaluation could not see a
# blockade at all. Three rather than two because the pawn in that game never
# advanced a square and was eventually just lost.
BLOCKADE_DIVISOR = 3
_SHIELD_SIG = int32(uint64, int32, b1)

_PASSERS_SIG = uint64(uint64, uint64, b1)
_POPCOUNT_SIG = int32(uint64)
_EVAL_SIG = int32(
    uint64, uint64, uint64, uint64, uint64, uint64,
    uint64, uint64, uint64, uint64, uint64, uint64, b1,
)


@njit(_SHIELD_SIG, cache=True)
def _shield_missing(own_pawns: Bitboard, king_sq: int, white: bool) -> int:
    """How many shield squares in front of the king lack one of our pawns."""
    kf = king_sq & 7
    kr = king_sq >> 3
    lo = kf - 1 if kf > 0 else 0
    hi = kf + 1 if kf < 7 else 7
    # The two ranks ahead of the king, clipped to the board.
    r1 = kr + 1 if white else kr - 1
    r2 = kr + 2 if white else kr - 2
    missing = 0
    for f in range(lo, hi + 1):
        near = 0 <= r1 <= 7 and (own_pawns >> np.uint64(r1 * 8 + f)) & np.uint64(1)
        far = 0 <= r2 <= 7 and (own_pawns >> np.uint64(r2 * 8 + f)) & np.uint64(1)
        if not (near or far):
            missing += 1
    return missing


@njit(_PASSERS_SIG, cache=True)
def _passers_jit(own: Bitboard, enemy: Bitboard, white: bool) -> Bitboard:
    if white:
        span = enemy >> np.uint64(8)
        span |= span >> np.uint64(8)
        span |= span >> np.uint64(16)
        span |= span >> np.uint64(32)
    else:
        span = enemy << np.uint64(8)
        span |= span << np.uint64(8)
        span |= span << np.uint64(16)
        span |= span << np.uint64(32)
    span |= ((span & NOT_FILE_H) << np.uint64(1)) | ((span & NOT_FILE_A) >> np.uint64(1))
    return own & ~span

@njit(_POPCOUNT_SIG, cache=True)
def popcount(n: Bitboard) -> int:
    count = 0
    while n:
        n &= n - np.uint64(1)
        count += 1
    return count

@njit(_EVAL_SIG, cache=True)
def _evaluate_jit(
    wp: Bitboard, wn: Bitboard, wb: Bitboard, wr: Bitboard, wq: Bitboard, wk: Bitboard,
    bp: Bitboard, bn: Bitboard, bb: Bitboard, br: Bitboard, bq: Bitboard, bk: Bitboard,
    is_white_turn: bool,
) -> int:
    mg = 0
    eg = 0
    phase = 0

    white_pieces = (wp, wn, wb, wr, wq, wk)
    black_pieces = (bp, bn, bb, br, bq, bk)

    wk_sq = -1
    bk_sq = -1

    for pt in range(6):
        w = white_pieces[pt]
        while w:
            lsb = w & -w
            sq = int(np.log2(float(lsb)))
            w ^= lsb
            idx = sq ^ 56
            mg += mg_table_np[pt + 1, idx]
            eg += eg_table_np[pt + 1, idx]
            phase += phase_inc_np[pt + 1]
            if pt == 5:
                wk_sq = sq

        b = black_pieces[pt]
        while b:
            lsb = b & -b
            sq = int(np.log2(float(lsb)))
            b ^= lsb
            mg -= mg_table_np[pt + 1, sq]
            eg -= eg_table_np[pt + 1, sq]
            phase += phase_inc_np[pt + 1]
            if pt == 5:
                bk_sq = sq

    w_pawn_attacks = ((wp & NOT_FILE_A) << np.uint64(7)) | ((wp & NOT_FILE_H) << np.uint64(9))
    b_pawn_attacks = ((bp & NOT_FILE_H) >> np.uint64(7)) | ((bp & NOT_FILE_A) >> np.uint64(9))

    # Passed Pawn Scaling (Halve bonus if unsupported/isolated, or blockaded)
    if wp or bp:
        w_occ = wp | wn | wb | wr | wq | wk
        b_occ = bp | bn | bb | br | bq | bk
        w = _passers_jit(wp, bp, True)
        while w:
            lsb = w & -w
            sq = int(np.log2(float(lsb)))
            w ^= lsb
            r = sq >> 3
            
            is_supported = (lsb & w_pawn_attacks) or (wk_sq >= 0 and king_dist_np[wk_sq, sq] <= 2)
            mg_bonus = passed_mg_np[r]
            eg_bonus = passed_eg_np[r]
            if not is_supported:
                mg_bonus //= 2
                eg_bonus //= 2
            # Blockade: an enemy piece standing on the stop square. See
            # BLOCKADE_DIVISOR -- a pawn that cannot advance is not a passer in
            # any useful sense, and this is the one case where the bonus is
            # both largest and most wrong.
            stop = sq + 8
            if stop < 64 and (b_occ >> np.uint64(stop)) & np.uint64(1):
                mg_bonus //= BLOCKADE_DIVISOR
                eg_bonus //= BLOCKADE_DIVISOR

            mg += mg_bonus
            eg += eg_bonus
            if stop < 64 and wk_sq >= 0 and bk_sq >= 0:
                eg += (2 * king_dist_np[bk_sq, stop] - king_dist_np[wk_sq, stop]) * kp_scale_np[r]

        b = _passers_jit(bp, wp, False)
        while b:
            lsb = b & -b
            sq = int(np.log2(float(lsb)))
            b ^= lsb
            r = 7 - (sq >> 3)
            
            is_supported = (lsb & b_pawn_attacks) or (bk_sq >= 0 and king_dist_np[bk_sq, sq] <= 2)
            mg_bonus = passed_mg_np[r]
            eg_bonus = passed_eg_np[r]
            if not is_supported:
                mg_bonus //= 2
                eg_bonus //= 2
            stop = sq - 8
            if stop >= 0 and (w_occ >> np.uint64(stop)) & np.uint64(1):
                mg_bonus //= BLOCKADE_DIVISOR
                eg_bonus //= BLOCKADE_DIVISOR

            mg -= mg_bonus
            eg -= eg_bonus
            if stop >= 0 and wk_sq >= 0 and bk_sq >= 0:
                eg -= (2 * king_dist_np[wk_sq, stop] - king_dist_np[bk_sq, stop]) * kp_scale_np[r]

    b_front_span = bp >> np.uint64(8)
    b_front_span |= b_front_span >> np.uint64(8)
    b_front_span |= b_front_span >> np.uint64(16)
    b_front_span |= b_front_span >> np.uint64(32)
    b_attack_span = (
        b_front_span
        | ((b_front_span & NOT_FILE_A) >> np.uint64(1))
        | ((b_front_span & NOT_FILE_H) << np.uint64(1))
    )
    
    w_front_span = wp << np.uint64(8)
    w_front_span |= w_front_span << np.uint64(8)
    w_front_span |= w_front_span << np.uint64(16)
    w_front_span |= w_front_span << np.uint64(32)
    w_attack_span = (
        w_front_span
        | ((w_front_span & NOT_FILE_A) >> np.uint64(1))
        | ((w_front_span & NOT_FILE_H) << np.uint64(1))
    )

    w_outposts = w_pawn_attacks & (~b_attack_span) & OUTPOST_RANKS_W
    b_outposts = b_pawn_attacks & (~w_attack_span) & OUTPOST_RANKS_B

    w_ko_count = popcount(wn & w_outposts)
    if w_ko_count:
        bonus = w_ko_count * 30
        mg += bonus
        eg += bonus

    b_ko_count = popcount(bn & b_outposts)
    if b_ko_count:
        bonus = b_ko_count * 30
        mg -= bonus
        eg -= bonus

    if wk_sq >= 0:
        wk_file = wk_sq & 7
        enemy_pawn_storm = 0
        p = bp
        while p:
            lsb = p & -p
            p_sq = int(np.log2(float(lsb)))
            p_file = p_sq & 7
            p_rank = p_sq >> 3
            if abs(p_file - wk_file) <= 1 and p_rank <= 4:
                enemy_pawn_storm += (5 - p_rank) * 12
            p ^= lsb
        mg -= enemy_pawn_storm

    if bk_sq >= 0:
        bk_file = bk_sq & 7
        enemy_pawn_storm_w = 0
        p = wp
        while p:
            lsb = p & -p
            p_sq = int(np.log2(float(lsb)))
            p_file = p_sq & 7
            p_rank = p_sq >> 3
            if abs(p_file - bk_file) <= 1 and p_rank >= 3:
                enemy_pawn_storm_w += p_rank * 12
            p ^= lsb
        mg += enemy_pawn_storm_w

    u = wp | (wp << np.uint64(8))
    u |= u << np.uint64(16)
    u |= u << np.uint64(32)
    w_files = u | (u >> np.uint64(8))
    w_files |= w_files >> np.uint64(16)
    w_files |= w_files >> np.uint64(32)

    v = bp | (bp >> np.uint64(8))
    v |= v >> np.uint64(16)
    v |= v >> np.uint64(32)
    b_files = v | (v << np.uint64(8))
    b_files |= b_files << np.uint64(16)
    b_files |= b_files << np.uint64(32)

    if wp:
        d = popcount(wp & (u << np.uint64(8)))
        adj = ((w_files & NOT_FILE_H) << np.uint64(1)) | ((w_files & NOT_FILE_A) >> np.uint64(1))
        i = popcount(wp & ~adj)
        mg -= 8 * d + 12 * i
        eg -= 18 * d + 16 * i

    if bp:
        d = popcount(bp & (v >> np.uint64(8)))
        adj = ((b_files & NOT_FILE_H) << np.uint64(1)) | ((b_files & NOT_FILE_A) >> np.uint64(1))
        i = popcount(bp & ~adj)
        mg += 8 * d + 12 * i
        eg += 18 * d + 16 * i

    w_rooks_alone = wr & ~w_files
    if w_rooks_alone:
        semi = popcount(w_rooks_alone & b_files)
        opn = popcount(w_rooks_alone & ~b_files)
        mg += 26 * opn + 12 * semi
        eg += 14 * opn + 8 * semi

    b_rooks_alone = br & ~b_files
    if b_rooks_alone:
        semi = popcount(b_rooks_alone & w_files)
        opn = popcount(b_rooks_alone & ~w_files)
        mg -= 26 * opn + 12 * semi
        eg -= 14 * opn + 8 * semi

    if phase <= 8 and wk_sq >= 0 and bk_sq >= 0:
        w_center_dist = abs((wk_sq & 7) - 3.5) + abs((wk_sq >> 3) - 3.5)
        b_center_dist = abs((bk_sq & 7) - 3.5) + abs((bk_sq >> 3) - 3.5)
        eg += int((7.0 - w_center_dist) * 4)
        eg -= int((7.0 - b_center_dist) * 4)

    if phase <= 6 and (wp or bp):
        near = 0
        p = bp
        while p:
            lsb = p & -p
            sq = int(np.log2(float(lsb)))
            p ^= lsb
            near -= king_dist_np[wk_sq, sq]
        p = wp
        while p:
            lsb = p & -p
            sq = int(np.log2(float(lsb)))
            p ^= lsb
            near += king_dist_np[bk_sq, sq]
        eg += 3 * near

    if popcount(wb) >= 2:
        mg += 22
        eg += 40
    if popcount(bb) >= 2:
        mg -= 22
        eg -= 40

    # Pawn shield, midgame only, and only against a queen that can use the hole.
    if bq != np.uint64(0) and wk != np.uint64(0):
        wk_sq = 0
        t = wk
        while t > np.uint64(1):
            t >>= np.uint64(1)
            wk_sq += 1
        if (wk_sq & 7) <= 2 or (wk_sq & 7) >= 5:     # a castled-ish king only
            mg -= _shield_missing(wp, wk_sq, True) * SHIELD_PER_MISSING
    if wq != np.uint64(0) and bk != np.uint64(0):
        bk_sq = 0
        t = bk
        while t > np.uint64(1):
            t >>= np.uint64(1)
            bk_sq += 1
        if (bk_sq & 7) <= 2 or (bk_sq & 7) >= 5:
            mg += _shield_missing(bp, bk_sq, False) * SHIELD_PER_MISSING

    if phase > 24:
        phase = 24
    score = (mg * phase + eg * (24 - phase)) // 24

    if is_white_turn:
        return score + 12
    return -score + 12




def evaluate(board: chess.Board) -> int:
    """Tapered PeSTO evaluation, in centipawns, from the side to move's view.

    The arithmetic lives in `_evaluate_jit`, compiled by numba (which the
    platform preinstalls). Measured against the pure-Python version this
    replaces: 7.3 us -> 1.63 us on the middlegame position, while also scoring
    outposts, doubled and isolated pawns, rook files and the bishop pair, none
    of which the Python version had.

    The bitboards are passed as plain Python ints. Wrapping each one in
    `np.uint64()` first -- as the engine this came from does -- costs 4.8 us of
    the 6.4 us that version spends, for nothing: numba coerces the ints itself.
    Twelve object constructions per evaluation was three quarters of the call.
    """
    white = board.occupied_co[WHITE]
    black = board.occupied_co[BLACK]
    return int(_evaluate_jit(
        board.pawns & white, board.knights & white, board.bishops & white,
        board.rooks & white, board.queens & white, board.kings & white,
        board.pawns & black, board.knights & black, board.bishops & black,
        board.rooks & black, board.queens & black, board.kings & black,
        board.turn,
    ))


def _warm_jit() -> None:
    """Force compilation during import, not on the first move.

    Import gets a 60 s budget before the clock starts; compilation is ~3.5 s.
    Left to happen lazily it would land on the first move instead, and with
    `cache=True` unavailable here -- the platform filesystem is read-only
    outside /tmp -- it would land there on every single game.
    """
    evaluate(chess.Board())


class Searcher:
    def __init__(self) -> None:
        # bucket index (full_key % TT_MAX_ENTRIES) -> (full_key, depth, flag,
        # value, best move). Bounded by construction, so it never needs a
        # hard clear; a new entry either fills an empty bucket, replaces a
        # shallower entry, or replaces a different position's stale entry
        # sharing the same bucket (depth-preferred, always-replace-on-key-miss).
        self.tt: dict[int, tuple[int, int, int, int, chess.Move | None]] = {}
        self.killers: list[list[chess.Move | None]] = [
            [None, None] for _ in range(MAX_PLY + 2)
        ]
        self.history: dict[tuple[chess.Color, int, int], int] = {}
        self.nodes = 0
        self.deadline = 0.0
        self.soft_deadline = 0.0
        self.seldepth = 0
        # positions on the current search path, plus this game's history
        self.path: dict[Hashable, int] = {}
        # Set by search() to describe the value it just returned: True when that
        # value came from a repetition or fifty-move draw, which is true only of
        # the path that reached it and must never enter a position-keyed table.
        # Read by the caller immediately after the recursive call returns.
        self._taint = False
        # Set from another thread to abandon a ponder search immediately. Checked
        # in _checkup, which runs every 256 nodes, so a running ponder yields
        # within microseconds of the opponent's move arriving.
        self.stop_flag = False

    # -- helpers ------------------------------------------------------------

    def new_game(self) -> None:
        """Forget everything learned from a previous game.

        The platform starts a fresh process per game, so this never fires there.
        A long-lived driver such as the Lichess bot reuses one `_SEARCHER` across
        games, and with the table no longer cleared between moves it is otherwise
        never cleared at all: measured over nine consecutive games in one process
        the table grew to 446k entries and RSS to 309 MB, climbing throughout.
        Speed was unaffected, so this is about bounding memory, not latency --
        but there is no reason to hold another game's table either way.
        """
        self.tt.clear()
        self.history.clear()
        self.path.clear()
        for pair in self.killers:
            pair[0] = pair[1] = None

    def _checkup(self) -> None:
        if self.stop_flag or time.monotonic() >= self.deadline:
            raise _Timeout

    def _has_non_pawn_material(self, board: chess.Board) -> bool:
        side = board.occupied_co[board.turn]
        return bool(side & (board.knights | board.bishops | board.rooks | board.queens))

    def _order_key(
        self, board: chess.Board, move: chess.Move, tt_move: chess.Move | None, ply: int
    ) -> int:
        if move == tt_move:
            return TT_BONUS

        to_sq = move.to_square
        victim = board.piece_type_at(to_sq)
        if victim is not None or board.is_en_passant(move):
            attacker = board.piece_type_at(move.from_square) or 1
            score = CAPTURE_BONUS + SEE_VALUE[victim] * 16 - SEE_VALUE[attacker]
            if move.promotion:
                score += SEE_VALUE[move.promotion] * 8
            return score

        if move.promotion:
            return CAPTURE_BONUS + SEE_VALUE[move.promotion] * 8

        k = self.killers[ply]
        if move == k[0]:
            return KILLER1_BONUS
        if move == k[1]:
            return KILLER2_BONUS

        return self.history.get((board.turn, move.from_square, to_sq), 0)

    def _sorted_moves(
        self, board: chess.Board, moves: list[chess.Move], tt_move: chess.Move | None, ply: int
    ) -> list[tuple[int, bool, chess.Move]]:
        """Score and order `moves`. Ordering is identical to `_order_key`'s.

        This is `_order_key` unrolled into a single loop. The parts that do not
        vary per move -- the killer pair, the history dict, the side to move, the
        bound methods -- are hoisted into locals instead of being re-fetched for
        each of ~35 moves at every node. `_order_key` as a sort callback was 11.5%
        of runtime.

        Two things beyond the hoisting:

        * En passant is checked with `to_sq == ep` rather than
          `board.is_en_passant(move)`. That call was being made for every quiet
          move in the search -- 976k times in a 16s profile -- to catch a case that
          needs a non-None `ep_square` to be possible at all.
        * `is_quiet` falls out of the same tests, so the search loop no longer
          needs its own `board.is_capture(move)` call per move.
        """
        history = self.history
        turn = board.turn
        piece_type_at = board.piece_type_at
        ep = board.ep_square
        # Only a pawn can capture en passant, and only onto ep_square.
        ep_from = board.pawns & board.occupied_co[turn] if ep is not None else 0

        # Move.__eq__ was 1.26M calls / 0.55s in the profile, almost all of it the
        # tt_move and killer tests below. Unpacking the three into ints lets the
        # comparison happen inline on values already in hand -- `to_sq` is fetched
        # anyway, and it short-circuits first for the ~97% of moves that miss.
        # to_square of -1 can never match, which is how "no such move" is spelled.
        tt_to = tt_move.to_square if tt_move is not None else -1
        tt_from = tt_move.from_square if tt_move is not None else -1
        tt_promo = tt_move.promotion if tt_move is not None else None
        k0, k1 = self.killers[ply]
        k0_to = k0.to_square if k0 is not None else -1
        k0_from = k0.from_square if k0 is not None else -1
        k0_promo = k0.promotion if k0 is not None else None
        k1_to = k1.to_square if k1 is not None else -1
        k1_from = k1.from_square if k1 is not None else -1
        k1_promo = k1.promotion if k1 is not None else None

        # `is_quiet` is carried explicitly rather than inferred from the score:
        # history values accumulate depth*depth per cutoff and can climb past
        # KILLER/CAPTURE_BONUS over a long game, which would silently reclassify a
        # quiet move as a capture and disable both late-move pruning and the killer
        # update for it.
        scored: list[tuple[int, bool, chess.Move]] = []
        add = scored.append
        for move in moves:
            to_sq = move.to_square
            from_sq = move.from_square
            promotion = move.promotion

            # Quiet is decided by the board, never by which slot the move matched,
            # so a quiet TT move still updates killers and history when it cuts.
            victim = piece_type_at(to_sq)
            if victim is not None:
                quiet = False
                score = (
                    CAPTURE_BONUS
                    + SEE_VALUE[victim] * 16
                    - SEE_VALUE[piece_type_at(from_sq) or 1]
                )
                if promotion:
                    score += SEE_VALUE[promotion] * 8
            elif to_sq == ep and ep_from >> from_sq & 1:
                # Pawn takes pawn; SEE_VALUE[None] is 100, matching _order_key.
                quiet = False
                score = CAPTURE_BONUS + 100 * 16 - 100
            elif promotion:
                quiet = False
                score = CAPTURE_BONUS + SEE_VALUE[promotion] * 8
            elif to_sq == k0_to and from_sq == k0_from and promotion == k0_promo:
                quiet = True
                score = KILLER1_BONUS
            elif to_sq == k1_to and from_sq == k1_from and promotion == k1_promo:
                quiet = True
                score = KILLER2_BONUS
            else:
                quiet = True
                score = history.get((turn, from_sq, to_sq), 0)

            if to_sq == tt_to and from_sq == tt_from and promotion == tt_promo:
                score = TT_BONUS
            add((score, quiet, move))

        # itemgetter(0) keeps the comparison in C and never compares the Move
        # objects, which are not orderable.
        scored.sort(key=_SCORE_OF, reverse=True)
        return scored

    def _store(
        self,
        key: int,
        depth: int,
        flag: int,
        value: int,
        move: chess.Move | None,
        ply: int,
    ) -> None:
        if value > MATE_BOUND:
            value += ply
        elif value < -MATE_BOUND:
            value -= ply
        idx = key % TT_MAX_ENTRIES
        prev = self.tt.get(idx)
        # Depth-preferred: keep the deeper entry for the same position.
        # Always replace when the bucket holds a different position (key
        # mismatch), so a stale entry can't permanently squat on a slot.
        if prev is None or prev[0] != key or prev[1] <= depth:
            self.tt[idx] = (key, depth, flag, value, move)

    # -- quiescence ---------------------------------------------------------

    def qsearch(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        if not self.nodes & 255:
            self._checkup()

        if ply > self.seldepth:
            self.seldepth = ply
        if ply >= MAX_PLY:
            return evaluate(board)

        in_check = board.is_check()

        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply
            best = -INF
        else:
            stand = evaluate(board)
            if stand >= beta:
                return stand
            if stand > alpha:
                alpha = stand
            best = stand
            moves = list(board.generate_legal_captures())
            # Queen promotions are tactical too, but only bother generating the
            # full move list when a pawn is actually one rank away.
            promo_rank = chess.BB_RANK_7 if board.turn == WHITE else chess.BB_RANK_2
            if board.pawns & board.occupied_co[board.turn] & promo_rank:
                for m in board.legal_moves:
                    if m.promotion == 5 and board.piece_type_at(m.to_square) is None:
                        moves.append(m)

        # Same scorer as the main search, with no TT move to favour. Quiescence
        # was still going through _order_key, which meant a Move.__eq__ against
        # None and an is_en_passant() call for every move it looked at.
        ordered = self._sorted_moves(board, moves, None, ply)

        for _score, _quiet, move in ordered:
            if not in_check:
                # Skip captures that lose material on their face: a cheaper piece
                # taking a more valuable one onto a defended square. Measured on
                # three positions at depth 5, 18,395 of 46,240 captures offered in
                # quiescence (39.8%) are this shape, and quiescence is 63% of all
                # search time -- the largest single pool in the profile.
                #
                # This is the cheap test, not a full static exchange evaluation:
                # it is wrong when the defender is pinned, or when we have a
                # second attacker behind the first. Those lose a little tactical
                # sight in exchange for a much smaller tree. A real SEE would be
                # more accurate and, in Python, cost more than it saves.
                to_sq = move.to_square
                victim = board.piece_type_at(to_sq)
                if victim is not None:
                    attacker = board.piece_type_at(move.from_square)
                    if (
                        attacker is not None
                        and SEE_VALUE[victim] < SEE_VALUE[attacker]
                        and board.is_attacked_by(not board.turn, to_sq)
                    ):
                        continue

                # Delta pruning: even winning this material would not raise alpha.
                gain = SEE_VALUE[victim] if victim is not None else 100
                if move.promotion:
                    gain += SEE_VALUE[move.promotion]
                if best + gain + 200 < alpha:
                    continue

            board.push(move)
            score = -self.qsearch(board, -beta, -alpha, ply + 1)
            board.pop()

            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break

        return best

    # -- main search --------------------------------------------------------

    def search(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        can_null: bool = True,
    ) -> int:
        self.nodes += 1
        if not self.nodes & 255:
            self._checkup()

        if ply >= MAX_PLY:
            self._taint = False
            return evaluate(board)

        # Position key, computed once and reused for the draw/repetition check,
        # the TT hash, and the search-path bookkeeping below.
        tkey = board._transposition_key()

        # Draw detections available from a FEN-only interface.
        #
        # Two of these three depend on how we got here rather than on the
        # position, and the transposition key does not include either the move
        # history or the halfmove clock. Their scores are therefore correct only
        # on this path, and _taint marks them so no ancestor stores them.
        # Insufficient material is a property of the position alone, so it is
        # safe to keep and cache.
        if ply > 0:
            if board.halfmove_clock >= 100:
                self._taint = True
                return 0
            if board.occupied.bit_count() <= 4 and board.is_insufficient_material():
                self._taint = False
                return 0
            if self.path.get(tkey, 0) >= 1:
                self._taint = True
                return 0

        in_check = board.is_check()
        if in_check:
            depth += 1  # check extension

        if depth <= 0:
            # qsearch has no repetition or fifty-move detection, so nothing it
            # returns is path-dependent.
            self._taint = False
            return self.qsearch(board, alpha, beta, ply)

        alpha_orig = alpha
        key = hash(tkey)
        tt_move: chess.Move | None = None
        entry = self.tt.get(key % TT_MAX_ENTRIES)
        if entry is not None and entry[0] == key:
            _e_key, e_depth, e_flag, e_val, e_move = entry
            tt_move = e_move
            if e_depth >= depth and ply > 0:
                val = e_val
                if val > MATE_BOUND:
                    val -= ply
                elif val < -MATE_BOUND:
                    val += ply
                if e_flag == EXACT or (e_flag == LOWER and val >= beta) or (
                    e_flag == UPPER and val <= alpha
                ):
                    # Clean by construction: tainted values are never stored.
                    self._taint = False
                    return val
            if tt_move is not None and not board.is_legal(tt_move):
                tt_move = None

        # Internal iterative reduction: with no TT move at high depth this node
        # is unordered, so a full-depth search here mostly pays to discover the
        # move ordering a shallower search would have found for a fraction of
        # the cost. Spend one ply less and let the TT entry it writes order the
        # re-search.
        if tt_move is None and depth >= 4:
            depth -= 1

        static: int | None = None

        # Reverse futility / static null-move pruning
        if not in_check and depth <= 4 and abs(beta) < MATE_BOUND:
            static = evaluate(board)
            if static - 85 * depth >= beta:
                self._taint = False
                return static

        # Null-move pruning
        if (
            can_null
            and not in_check
            and depth >= 3
            and ply > 0
            and abs(beta) < MATE_BOUND
            and self._has_non_pawn_material(board)
        ):
            if static is None:
                static = evaluate(board)
            if static >= beta:
                r = 2 + (depth > 6)
                board.push(chess.Move.null())
                score = -self.search(board, depth - 1 - r, -beta, -beta + 1, ply + 1, False)
                null_taint = self._taint
                board.pop()
                if score >= beta:
                    if score > MATE_BOUND:
                        score = beta
                    # The cutoff value came from that subtree, taint and all.
                    self._taint = null_taint
                    return score

        moves = list(board.legal_moves)
        if not moves:
            self._taint = False
            return -MATE + ply if in_check else 0

        ordered = self._sorted_moves(board, moves, tt_move, ply)

        best = -INF
        best_move: chess.Move | None = None
        best_taint = False
        key0 = tkey
        self.path[key0] = self.path.get(key0, 0) + 1

        try:
            for i, (_score, is_quiet, move) in enumerate(ordered):
                # Late move pruning on shallow quiet nodes
                if (
                    is_quiet
                    and not in_check
                    and depth <= 3
                    and i > 6 + depth * 6
                    and best > -MATE_BOUND
                ):
                    continue

                # Forward futility pruning: near the horizon, a quiet move that
                # cannot plausibly lift the static score to alpha is not worth a
                # node. Distinct from the reverse futility above, which prunes
                # the whole node when the score is already too good.
                if (
                    is_quiet
                    and not in_check
                    and depth <= 2
                    and i > 0
                    and static is not None
                    and best > -MATE_BOUND
                    and static + 110 * depth + 90 <= alpha
                ):
                    continue

                board.push(move)

                reduction = 0
                if is_quiet and depth >= 3 and i >= 3 and not board.is_check():
                    reduction = _LMR[depth if depth < 64 else 63][i if i < 64 else 63]
                    # Never reduce into quiescence: the point of a reduced search
                    # is a cheap refutation, not a different kind of search.
                    if reduction >= depth - 1:
                        reduction = depth - 2
                    if reduction < 0:
                        reduction = 0

                if i == 0:
                    score = -self.search(board, depth - 1, -beta, -alpha, ply + 1)
                    move_taint = self._taint
                else:
                    score = -self.search(
                        board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1
                    )
                    move_taint = self._taint
                    if score > alpha and reduction:
                        score = -self.search(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                        move_taint = self._taint
                    if alpha < score < beta:
                        score = -self.search(board, depth - 1, -beta, -alpha, ply + 1)
                        move_taint = self._taint

                board.pop()

                if score > best:
                    best = score
                    best_move = move
                    # Only the move that set `best` decides whether this node's
                    # value is path-dependent. Taking the taint of every child
                    # searched would be safe but far too broad: repetition fires
                    # often in shuffling lines, and it would suppress most stores.
                    best_taint = move_taint
                    if score > alpha:
                        alpha = score
                        if alpha >= beta:
                            if is_quiet:
                                k = self.killers[ply]
                                if k[0] != move:
                                    k[1] = k[0]
                                    k[0] = move
                                hk = (board.turn, move.from_square, move.to_square)
                                self.history[hk] = self.history.get(hk, 0) + depth * depth
                            break
        finally:
            c = self.path.get(key0, 1) - 1
            if c <= 0:
                self.path.pop(key0, None)
            else:
                self.path[key0] = c

        if best >= beta:
            flag = LOWER
        elif best > alpha_orig:
            flag = EXACT
        else:
            flag = UPPER
        # The one line this whole mechanism exists for: a value that is only true
        # of the path we took never enters a table keyed by position alone.
        if not best_taint:
            self._store(key, depth, flag, best, best_move, ply)

        self._taint = best_taint
        return best

    # -- root ---------------------------------------------------------------

    def go(
        self,
        board: chess.Board,
        hard_ms: float,
        soft_ms: float,
        max_depth: int = MAX_PLY,
    ) -> tuple[chess.Move | None, int, int]:
        now = time.monotonic()
        self.deadline = now + hard_ms / 1000.0
        self.soft_deadline = now + soft_ms / 1000.0
        self.nodes = 0
        self.seldepth = 0

        if TT_CLEAR_EACH_MOVE:
            self.tt.clear()

        if self.history:
            self.history = {k: v >> 1 for k, v in self.history.items() if v >= 2}

        self.path = dict(_GAME_HIST)

        moves = list(board.legal_moves)
        if not moves:
            return None, 0, 0
        if len(moves) == 1:
            return moves[0], 0, 1

        root_stack = len(board.move_stack)

        best_move: chess.Move | None = moves[0]
        best_score = 0
        depth_done = 0
        prev_scores: dict[chess.Move, int] = {}
        self.path[board._transposition_key()] = 1

        start = START_DEPTH if hard_ms >= START_DEPTH_MIN_MS else 1
        for depth in range(start, max_depth + 1):
            if depth > start and time.monotonic() >= self.soft_deadline:
                break

            moves.sort(
                key=lambda m: (
                    2 << 30 if m == best_move else prev_scores.get(m, -INF)
                ),
                reverse=True,
            )

            alpha, beta = -INF, INF
            if depth >= 7 and depth_done:
                window = 70
                alpha = best_score - window
                beta = best_score + window

            try:
                while True:
                    iter_best: chess.Move | None = None
                    iter_score = -INF
                    a = alpha
                    failed = False

                    for i, move in enumerate(moves):
                        board.push(move)
                        if i == 0:
                            score = -self.search(board, depth - 1, -beta, -a, 1)
                        else:
                            score = -self.search(board, depth - 1, -a - 1, -a, 1)
                            if a < score < beta:
                                score = -self.search(board, depth - 1, -beta, -a, 1)
                        board.pop()

                        prev_scores[move] = score

                        if score > iter_score:
                            iter_score = score
                            iter_best = move
                            if score > a:
                                a = score
                        if score >= beta:
                            failed = True
                            break

                    if failed:  # fail high
                        beta = min(INF, beta + 200 + (beta - alpha))
                        continue
                    if iter_score <= alpha and alpha != -INF:  # fail low
                        alpha = max(-INF, alpha - 200 - (beta - alpha))
                        beta = INF
                        continue
                    break

                best_move = iter_best
                best_score = iter_score
                depth_done = depth

                # Clean, UCI-style summary per completed depth instead of per-move spam, specifying the side the score is for
                side_to_move = "white" if board.turn else "black"
                print(
                    f"agent: info depth {depth} score cp {best_score} (for {side_to_move}) bestmove {best_move.uci() if best_move else 'none'}",
                    file=sys.stderr
                )

                if abs(best_score) > MATE_BOUND:
                    break  # forced mate found, no need to search deeper

            except _Timeout:
                while len(board.move_stack) > root_stack:
                    board.pop()
                break

        while len(board.move_stack) > root_stack:
            board.pop()
        return best_move, best_score, depth_done
# =============================================================================
# Opening book
# =============================================================================

# Polyglot (.bin) opening book, looked up before search on every move. Ship the
# .bin file alongside this script; if it is missing or malformed the engine
# silently falls back to normal search, so this is safe to leave enabled even
# if the file does not make it into a given deployment.
#
# Opened eagerly at import time rather than lazily on first use: the platform
# grants a fixed init budget before the clock starts, so the mmap/parse cost
# (a few ms for a book this size) should be paid there, not out of move 1's
# time budget.
_BOOK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "book.bin")


def _open_book() -> chess.polyglot.MemoryMappedReader | None:
    """Open the book, or carry on without one.

    Say so on stderr either way. A book that fails to load is invisible
    otherwise -- the engine just plays search moves and looks fine -- and that
    is exactly how a book corrupted by git's CRLF translation went unnoticed.
    A valid book is a whole number of 16-byte records; anything else is damage.
    """
    try:
        size = os.path.getsize(_BOOK_PATH)
    except OSError:
        print(f"agent: no opening book at {_BOOK_PATH}, searching from move 1",
              file=sys.stderr)
        return None
    if size % 16:
        print(f"agent: {_BOOK_PATH} is {size} bytes, not a multiple of 16 -- "
              "corrupt book, ignoring it", file=sys.stderr)
        return None
    try:
        reader = chess.polyglot.open_reader(_BOOK_PATH)
    except Exception as exc:
        print(f"agent: could not open {_BOOK_PATH}: {exc!r}", file=sys.stderr)
        return None
    print(f"agent: opening book loaded, {size // 16} entries", file=sys.stderr)
    return reader


_BOOK_READER: chess.polyglot.MemoryMappedReader | None = _open_book()


def _book_move(board: chess.Board) -> chess.Move | None:
    """Return a weighted-random book move for this position, or None if the
    position is not in the book (or the book is unavailable/corrupt)."""
    if _BOOK_READER is None:
        return None
    try:
        entry = _BOOK_READER.weighted_choice(board)
    except Exception:
        return None
    move = entry.move
    # Defensive: never trust the book blindly against the actual position.
    if move not in board.legal_moves:
        return None
    return move


# =============================================================================
# Entry point
# =============================================================================

_warm_jit()
_SEARCHER = Searcher()

# ---------------------------------------------------------------------------
# Game history
#
# The platform hands us a bare FEN with no move list, so the engine cannot see
# repetitions on its own and will happily shuffle a winning position into a
# threefold draw. We rebuild the history ourselves across calls: every position
# we are asked about, plus every position we create with our own move.
#
# Clearing on halfmove_clock == 0 is exact rather than a heuristic: no position
# can repeat across an irreversible move (a capture or a pawn push). It also
# resets the table automatically at the start of a new game.
# ---------------------------------------------------------------------------

_GAME_HIST: dict[Hashable, int] = {}
_LAST_PLY = -10


def _note_position(board: chess.Board) -> None:
    try:
        if board.halfmove_clock == 0:
            _GAME_HIST.clear()
        key = board._transposition_key()
        _GAME_HIST[key] = _GAME_HIST.get(key, 0) + 1
    except Exception:
        pass


def _note_root(board: chess.Board) -> None:
    """Record the position we were asked about, discarding the table first if
    this FEN does not continue the game we were following.

    Between two of our turns the ply count must advance by exactly two. Anything
    else means a new game, a new match, or an analysis position, and carrying
    the old history over would invent repetitions that never happened.
    """
    global _LAST_PLY
    try:
        ply = 2 * (board.fullmove_number - 1) + (0 if board.turn else 1)
        if ply != _LAST_PLY + 2:
            _GAME_HIST.clear()
            _SEARCHER.new_game()
        _LAST_PLY = ply
    except Exception:
        _GAME_HIST.clear()
        _SEARCHER.new_game()
    _note_position(board)

# Reserve for serialisation / server overhead, in milliseconds.
OVERHEAD_MS = 120
INCREMENT_MS = 500
# Below this clock the interpreter overhead alone dominates, so searching at
# all risks flagging: play the fallback immediately instead.
PANIC_FLOOR_MS = 250

# Time-management knobs. These defaults are the ones validated for the
# competition's 120s + 0.5s control; leave them alone for the submission.
# A driver (for example the Lichess bot) can reassign them per game to suit a
# faster control, where spending a thirtieth of the clock on one move is far
# too slow to watch.
MOVES_REMAINING = 30.0   # the budget assumes this many moves are still to play
HARD_MULTIPLIER = 3.0
MAX_FRACTION = 0.40


def _budget(time_left_ms: int) -> tuple[float, float]:
    """Return (hard_ms, soft_ms) for this move."""
    left = time_left_ms - OVERHEAD_MS
    if left <= 0:
        return 1.0, 1.0

    # Panic mode: almost out of clock, move nearly instantly.
    if left < 1500:
        t = max(15.0, left * 0.12)
        return t, t * 0.5

    soft = left / MOVES_REMAINING + INCREMENT_MS * 0.70
    hard = soft * HARD_MULTIPLIER

    cap = left * MAX_FRACTION
    if hard > cap:
        hard = cap
    if soft > hard * 0.55:
        soft = hard * 0.55
    return hard, soft


def _safe_fallback(board: chess.Board) -> str:
    """A move to play when the search is unavailable.

    This runs when the search raised, or when the clock is too low to search at
    all, so it has to be cheap. It must also not be reckless: an earlier version
    simply grabbed the most valuable capture available, which happily took a
    defended pawn with the queen. Here each move is scored as what it wins minus
    what it leaves hanging on the destination square, which is crude but enough
    to avoid handing over a piece.
    """
    them = not board.turn
    best: chess.Move | None = None
    best_score = -1 << 30

    for move in board.legal_moves:
        victim = board.piece_type_at(move.to_square)
        gain = SEE_VALUE[victim] if victim is not None else 0
        mover = board.piece_type_at(move.from_square) or 1

        board.push(move)
        mate = board.is_checkmate()
        # If they can recapture on that square, assume we lose the piece.
        hanging = board.is_attacked_by(them, move.to_square)
        risk = 0 if mate else (SEE_VALUE[mover] if hanging else 0)
        board.pop()

        score = gain - risk + (1 << 20 if mate else 0)
        if score > best_score:
            best_score = score
            best = move

    return best.uci() if best is not None else "0000"


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation for the given FEN position.

    Never raises, never returns an illegal move, never overruns the clock.
    """
    # First thing, before anything reads the table or the game history: the
    # opponent has moved, so the background search is now searching a tree we
    # will never enter. Stopping here also means the foreground search below
    # gets the core to itself.
    _stop_ponder()

    fallback = "0000"
    try:
        board = chess.Board(fen)
    except Exception:
        return fallback

    try:
        if not any(board.legal_moves):
            return fallback
        fallback = _safe_fallback(board)
    except Exception as exc:
        print(f"agent: fallback selection failed: {exc!r}", file=sys.stderr)
        return fallback

    try:
        tl = int(time_left_ms)
    except Exception:
        tl = 1000

    _note_root(board)

    if tl <= PANIC_FLOOR_MS:
        _note_after(board, fallback)
        _start_ponder(board, fallback, tl)
        return fallback

    # Opening book: costs essentially no time, so try it regardless of clock,
    # ahead of the panic-floor fallback above and before spending any search
    # budget. Falls straight through to search if out of book.
    try:
        book_move = _book_move(board)
    except Exception:
        book_move = None
    if book_move is not None:
        chosen = str(book_move.uci())
        _note_after(board, chosen)
        _start_ponder(board, chosen, tl)
        return chosen

    chosen = fallback
    try:
        hard, soft = _budget(tl)
        move, _score, _depth = _SEARCHER.go(board, hard, soft)
        if move is not None and board.is_legal(move):
            chosen = str(move.uci())
        elif move is not None:
            # Unreachable now that go() rewinds the board, but this is exactly
            # how a discarded search result hid for a whole game once.
            print(f"agent: searched move {move} rejected, playing {chosen}",
                  file=sys.stderr)
    except Exception:
        # Never let a search bug forfeit the game, but never hide it either:
        # stdout is redirected to stderr by the runner, so this reaches the logs.
        traceback.print_exc(file=sys.stderr)
        print(f"agent: search failed on {fen!r}, playing {chosen}", file=sys.stderr)

    _note_after(board, chosen)
    _start_ponder(board, chosen, tl)
    return chosen


def _note_after(board: chess.Board, uci: str) -> None:
    """Record the position our own move creates, so the next call's history is
    complete even though we never see that position in a FEN."""
    try:
        board.push(chess.Move.from_uci(uci))
        _note_position(board)
        board.pop()
    except Exception:
        pass


# =============================================================================
# Pondering
# =============================================================================
#
# Between our moves the process sits idle: harness/sandbox.py keeps one
# long-lived process per agent and blocks reading its stdout, and the referee
# charges our clock only for the wall time inside get_move(). So every second
# the opponent spends thinking is a second of our CPU going to waste.
#
# What we ponder is the position *after our own move*, with the opponent to
# move. The usual scheme guesses the opponent's reply and searches that, which
# is wrong whenever the guess is wrong and needs bookkeeping to notice. Rooting
# one ply higher can never be wrong: whatever they play, the position we are
# next asked about is a child of the tree we just searched, and the entries
# that tree left in the transposition table are exactly the ones the next
# search wants. The table is shared with the main searcher and survives across
# moves (TT_CLEAR_EACH_MOVE is False), so the work carries over directly.
#
# Everything else is kept separate. The ponder searcher is its own Searcher
# instance with its own killers, history, path and node counter; only `tt` is
# the same object. Sharing the whole searcher would let a background search
# overwrite the deadline and node count of a foreground one if the two ever
# overlapped, which is precisely the failure that is hardest to reproduce.
#
# The safety rules, in order of how much damage getting them wrong does:
#
# * The thread is a daemon, so a hung ponder can never keep the process alive
#   past the end of a game.
# * The join is bounded. An unbounded join means one wedged background search
#   forfeits the game on time; a bounded one means we play on regardless, and
#   the worst case is a few hundred milliseconds of shared CPU.
# * The ponder search carries its own hard deadline as a backstop, so even a
#   lost stop flag ends it.
# * A new ponder never starts while an old thread is alive.

PONDER_ENABLED = True
# Backstop deadline for a ponder search, in seconds. Reached only if the driver
# never asks for another move; a real opponent interrupts long before this.
PONDER_MAX_S = 45.0
# How long to wait for a ponder to notice the stop flag. _checkup runs every 256
# nodes, so this is thousands of times the expected latency -- it exists to bound
# the pathological case, not the normal one.
PONDER_JOIN_S = 0.5
# Below this clock, don't ponder at all.
#
# Stopping a ponder costs a flat ~6 ms: _checkup runs every 256 nodes and the
# background search runs at roughly 44k nodes/s, so that is one checkup interval
# plus a GIL handoff. Irrelevant against a 4-second move, but fuzzing at a 30 ms
# budget it turned the worst-case clock use from 3% into 48% -- a fixed cost
# eating a budget it is the same size as. Pondering has nothing to offer down
# there anyway: the next search is too short to spend the table it fills.
PONDER_MIN_CLOCK_MS = 3000

_PONDER = Searcher()
# Share the table and nothing else. This is a rebind of the attribute, so both
# searchers hold a reference to one dict; new_game() clears it for both, which
# is what we want.
_PONDER.tt = _SEARCHER.tt
_PONDER_THREAD: threading.Thread | None = None


def _ponder_worker(board: chess.Board) -> None:
    # A background search must never take the process down with it: if it fails
    # we simply lose the free work, and the foreground search is unharmed. It
    # still says so on stderr rather than vanishing -- a ponder that has been
    # throwing on every move all game looks exactly like one that is working.
    try:
        _PONDER.go(board, PONDER_MAX_S * 1000.0, PONDER_MAX_S * 1000.0)
    except Exception as exc:
        print(f"agent: ponder failed: {exc!r}", file=sys.stderr)


def _stop_ponder() -> None:
    """Ask any running ponder to stop and wait a bounded time for it."""
    global _PONDER_THREAD
    thread = _PONDER_THREAD
    if thread is None:
        return
    _PONDER.stop_flag = True
    try:
        thread.join(PONDER_JOIN_S)
        if thread.is_alive():
            # Leave stop_flag set and the handle in place: _start_ponder refuses
            # to launch another while this one is alive, so the two can never
            # pile up. The straggler will exit on its own next _checkup.
            print("agent: ponder did not stop within "
                  f"{PONDER_JOIN_S}s, continuing anyway", file=sys.stderr)
            return
    except Exception:
        pass
    _PONDER_THREAD = None


def _start_ponder(board: chess.Board, uci: str, time_left_ms: int) -> None:
    """Search the position our move creates, in the background, until stopped."""
    global _PONDER_THREAD
    if not PONDER_ENABLED or _PONDER_THREAD is not None:
        return
    if time_left_ms < PONDER_MIN_CLOCK_MS:
        return
    try:
        after = board.copy(stack=False)
        after.push(chess.Move.from_uci(uci))
        if after.is_game_over(claim_draw=False):
            return
        _PONDER.stop_flag = False
        thread = threading.Thread(
            target=_ponder_worker, args=(after,), name="ponder", daemon=True
        )
        _PONDER_THREAD = thread
        thread.start()
    except Exception:
        _PONDER_THREAD = None