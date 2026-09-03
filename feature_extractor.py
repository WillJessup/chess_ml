"""Feature extraction for Texel-style tuning of the engine's eval.

Reimplements the *tunable* part of `_evaluate_jit` in plain Python (no numba,
no numpy dependency in the hot path) as a set of linear features, so that

    eval(position) ~= tunable_features(position) . tunable_weights
                       + frozen_terms(position)

`frozen_terms` covers everything the tuner does NOT touch (king shield, pawn
storm, king centralisation/proximity in the endgame, the passed-pawn
king-distance term) computed with the constants currently shipped in the
engine. It is included as a fixed offset so the tuner isn't forced to distort
the tunable terms to compensate for effects it can't see.

Feature layout (see PARAM_LAYOUT below) mirrors the tapered mg/eg structure:
every tunable quantity gets an independent midgame weight and endgame weight,
combined the same way the engine already combines them:

    total = (mg_score * phase + eg_score * (24 - phase)) // 24

except here mg_score / eg_score are themselves linear in the weights, so the
whole thing is one big linear regression, tapering included.

This module has ONE external dependency: `python-chess`. It deliberately does
not import the agent module, so it can be developed/tested independently of
numba, the opening book, etc.
"""

from __future__ import annotations

import numpy as np
import chess

WHITE, BLACK = chess.WHITE, chess.BLACK
MASK64 = (1 << 64) - 1

NOT_FILE_A = MASK64 ^ chess.BB_FILE_A
NOT_FILE_H = MASK64 ^ chess.BB_FILE_H


def _shl(x: int, n: int) -> int:
    return (x << n) & MASK64


def _bits(bb: int):
    while bb:
        lsb = bb & -bb
        yield lsb.bit_length() - 1
        bb ^= lsb


def _popcount(bb: int) -> int:
    return bb.bit_count()


# ---------------------------------------------------------------------------
# Starting-point constants, copied verbatim from the shipped engine. These are
# ONLY used to build the initial parameter vector (so tuning starts from what
# is currently on the board, not from zero) and, for the frozen terms, as the
# permanent fixed values.
# ---------------------------------------------------------------------------

MG_VALUE = {1: 82, 2: 337, 3: 365, 4: 477, 5: 1025, 6: 0}
EG_VALUE = {1: 94, 2: 281, 3: 297, 4: 512, 5: 936, 6: 0}
PHASE_INC = {1: 0, 2: 1, 3: 1, 4: 2, 5: 4, 6: 0}

MG_PAWN = [0,0,0,0,0,0,0,0,98,134,61,95,68,126,34,-11,-6,7,26,31,65,56,25,-20,
    -14,13,6,21,23,12,17,-23,-27,-2,-5,12,17,6,10,-25,-26,-4,-4,-10,3,3,33,-12,
    -35,-1,-20,-23,-15,24,38,-22,0,0,0,0,0,0,0,0]
EG_PAWN = [0,0,0,0,0,0,0,0,178,173,158,134,147,132,165,187,94,100,85,67,56,53,
    82,84,32,24,13,5,-2,4,17,17,13,9,-3,-7,-7,-8,3,-1,4,7,-6,1,0,-5,-1,-8,13,8,
    8,10,13,0,2,-7,0,0,0,0,0,0,0,0]
MG_KNIGHT = [-167,-89,-34,-49,61,-97,-15,-107,-73,-41,72,36,23,62,7,-17,-47,60,
    37,65,84,129,73,44,-9,17,19,53,37,69,18,22,-13,4,16,13,28,19,21,-8,-23,-9,
    12,10,19,17,25,-16,-29,-53,-12,-3,-1,18,-14,-19,-105,-21,-58,-33,-17,-28,
    -19,-23]
EG_KNIGHT = [-58,-38,-13,-28,-31,-27,-63,-99,-25,-8,-25,-2,-9,-25,-24,-52,-24,
    -20,10,9,-1,-9,-19,-41,-17,3,22,22,22,11,8,-18,-18,-6,16,25,16,17,4,-18,-23,
    -3,-1,15,10,-3,-20,-22,-42,-20,-10,-5,-2,-20,-23,-44,-29,-51,-23,-15,-22,
    -18,-50,-64]
MG_BISHOP = [-29,4,-82,-37,-25,-42,7,-8,-26,16,-18,-13,30,59,18,-47,-16,37,43,
    40,35,50,37,-2,-4,5,19,50,37,37,7,-2,-6,13,13,26,34,12,10,4,0,15,15,15,14,
    27,18,10,4,15,16,0,7,21,33,1,-33,-3,-14,-21,-13,-12,-39,-21]
EG_BISHOP = [-14,-21,-11,-8,-7,-9,-17,-24,-8,-4,7,-12,-3,-13,-4,-14,2,-8,0,-1,
    -2,6,0,4,-3,9,12,9,14,10,3,2,-6,3,13,19,7,10,-3,-9,-12,-3,8,10,13,3,-7,-15,
    -14,-18,-7,-1,4,-9,-15,-27,-23,-9,-23,-5,-9,-16,-5,-17]
MG_ROOK = [32,42,32,51,63,9,31,43,27,32,58,62,80,67,26,44,-5,19,26,36,17,45,61,
    16,-24,-11,7,26,24,35,-8,-20,-36,-26,-12,-1,9,-7,6,-23,-45,-25,-16,-17,3,0,
    -5,-33,-44,-16,-20,-9,-1,11,-6,-71,-19,-13,1,17,16,7,-37,-26]
EG_ROOK = [13,10,18,15,12,12,8,5,11,13,13,11,-3,3,8,3,7,7,7,5,4,-3,-5,-3,4,3,13,
    1,2,1,-1,2,3,5,8,4,-5,-6,-8,-11,-4,0,-5,-1,-7,-12,-8,-16,-6,-6,0,2,-9,-9,-11,
    -3,-9,2,3,-1,-5,-13,4,-20]
MG_QUEEN = [-28,0,29,12,59,44,43,45,-24,-39,-5,1,-16,57,28,54,-13,-17,7,8,29,56,
    47,57,-27,-27,-16,-16,-1,17,-2,1,-9,-26,-9,-10,-2,-4,3,-3,-14,2,-11,-2,-5,2,
    14,5,-35,-8,11,2,8,15,-3,1,-1,-18,-9,10,-15,-25,-31,-50]
EG_QUEEN = [-9,22,22,27,27,19,10,20,-17,20,32,41,58,25,30,0,-20,6,9,49,47,35,19,
    9,3,22,24,45,57,40,57,36,-18,28,19,47,31,34,39,23,-16,-27,15,6,9,17,10,5,-22,
    -23,-30,-16,-16,-23,-36,-32,-33,-28,-22,-43,-5,-32,-20,-41]
MG_KING = [-65,23,16,-15,-56,-34,2,13,29,-1,-20,-7,-8,-4,-38,-29,-9,24,2,-16,-20,
    6,22,-22,-17,-20,-12,-27,-30,-25,-14,-36,-49,-1,-27,-39,-46,-44,-33,-51,-14,
    -14,-22,-46,-44,-30,-15,-27,1,7,-8,-64,-43,-16,9,8,-15,36,12,-54,8,-28,24,14]
EG_KING = [-74,-35,-18,-18,-11,15,4,-17,-12,17,14,17,17,38,23,11,10,17,23,15,20,
    45,44,13,-8,22,24,27,26,33,26,3,-18,-4,21,24,27,23,9,-11,-19,-3,11,21,23,16,
    7,-9,-27,-11,4,13,14,4,-5,-17,-53,-34,-21,-11,-28,-14,-24,-43]

_MG_RAW = {1: MG_PAWN, 2: MG_KNIGHT, 3: MG_BISHOP, 4: MG_ROOK, 5: MG_QUEEN, 6: MG_KING}
_EG_RAW = {1: EG_PAWN, 2: EG_KNIGHT, 3: EG_BISHOP, 4: EG_ROOK, 5: EG_QUEEN, 6: EG_KING}

PASSED_MG = [0, 0, 2, 5, 12, 22, 40, 0]
PASSED_EG = [0, 10, 18, 34, 60, 100, 150, 0]
KP_SCALE = [0, 0, 0, 1, 2, 3, 4, 0]          # frozen (not tuned)
SHIELD_PER_MISSING = 20                       # frozen
BLOCKADE_DIVISOR = 3                          # informs starting guess only

# ---------------------------------------------------------------------------
# Tunable parameter layout: everything below gets an independent mg weight and
# an independent eg weight, EXCEPT `tempo` which is untapered (added flat,
# exactly like the engine's current "+12" applied to whoever is to move).
# ---------------------------------------------------------------------------

PST_DIM = 6 * 64            # (piece_type 1..6) x (square 0..63)
R_NAMES = (
    ["pst"] * PST_DIM
    + ["passed_rank"] * 8
    + ["support_bonus"]
    + ["blockade_penalty"]
    + ["doubled"]
    + ["isolated"]
    + ["rook_open"]
    + ["rook_semi"]
    + ["outpost"]
    + ["bishop_pair"]
)
R_DIM = len(R_NAMES)  # 384 + 8 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 400

# Index helpers into the R vector
_PST_OFF = 0
_PASSED_OFF = PST_DIM
_SUPPORT_IDX = PST_DIM + 8
_BLOCKADE_IDX = _SUPPORT_IDX + 1
_DOUBLED_IDX = _BLOCKADE_IDX + 1
_ISOLATED_IDX = _DOUBLED_IDX + 1
_ROOK_OPEN_IDX = _ISOLATED_IDX + 1
_ROOK_SEMI_IDX = _ROOK_OPEN_IDX + 1
_OUTPOST_IDX = _ROOK_SEMI_IDX + 1
_BISHOP_PAIR_IDX = _OUTPOST_IDX + 1
assert _BISHOP_PAIR_IDX == R_DIM - 1

# Full packed parameter vector: [mg_weights(R_DIM), eg_weights(R_DIM), tempo]
PARAM_DIM = 2 * R_DIM + 1


def default_params() -> np.ndarray:
    """Initial parameter vector matching the values currently shipped."""
    mg = np.zeros(R_DIM, dtype=np.float64)
    eg = np.zeros(R_DIM, dtype=np.float64)

    for pt in range(1, 7):
        base_mg = MG_VALUE[pt]
        base_eg = EG_VALUE[pt]
        for idx in range(64):
            mg[_PST_OFF + (pt - 1) * 64 + idx] = base_mg + _MG_RAW[pt][idx]
            eg[_PST_OFF + (pt - 1) * 64 + idx] = base_eg + _EG_RAW[pt][idx]

    for r in range(8):
        mg[_PASSED_OFF + r] = PASSED_MG[r]
        eg[_PASSED_OFF + r] = PASSED_EG[r]

    # These four are re-parameterisations of multiplicative effects in the
    # shipped eval (halve if unsupported, divide by 3 if blockaded). There is
    # no exact equivalent additive constant, so these are rough starting
    # guesses only -- the tuner will move them.
    avg_mg = sum(PASSED_MG) / 8
    avg_eg = sum(PASSED_EG) / 8
    mg[_SUPPORT_IDX], eg[_SUPPORT_IDX] = avg_mg * 0.25, avg_eg * 0.25
    mg[_BLOCKADE_IDX], eg[_BLOCKADE_IDX] = -avg_mg * 0.5, -avg_eg * 0.5

    mg[_DOUBLED_IDX], eg[_DOUBLED_IDX] = -8, -18
    mg[_ISOLATED_IDX], eg[_ISOLATED_IDX] = -12, -16
    mg[_ROOK_OPEN_IDX], eg[_ROOK_OPEN_IDX] = 26, 14
    mg[_ROOK_SEMI_IDX], eg[_ROOK_SEMI_IDX] = 12, 8
    mg[_OUTPOST_IDX], eg[_OUTPOST_IDX] = 30, 30
    mg[_BISHOP_PAIR_IDX], eg[_BISHOP_PAIR_IDX] = 22, 40

    tempo = np.array([12.0])
    return np.concatenate([mg, eg, tempo])


def unpack(params: np.ndarray):
    mg = params[:R_DIM]
    eg = params[R_DIM:2 * R_DIM]
    tempo = params[2 * R_DIM]
    return mg, eg, tempo


# ---------------------------------------------------------------------------
# Passed-pawn detection, doubled/isolated files, outposts -- ported from
# _passers_jit / _evaluate_jit.
# ---------------------------------------------------------------------------


def _front_fill_up(bb: int) -> int:
    u = bb | _shl(bb, 8)
    u |= _shl(u, 16)
    u |= _shl(u, 32)
    return u


def _passers(own: int, enemy: int, white: bool) -> int:
    if white:
        span = enemy >> 8
        span |= span >> 8
        span |= span >> 16
        span |= span >> 32
    else:
        span = _shl(enemy, 8)
        span |= _shl(span, 8)
        span |= _shl(span, 16)
        span |= _shl(span, 32)
    span |= _shl(span & NOT_FILE_H, 1) | ((span & NOT_FILE_A) >> 1)
    return own & ~span & MASK64


def extract(board: chess.Board):
    """Return (R, mg_frozen, eg_frozen, phase, white_to_move).

    R is a float64 vector of length R_DIM (white-minus-black raw feature
    counts, NOT yet weighted). mg_frozen/eg_frozen are plain floats: the
    contribution of terms this tuner does not touch, using the engine's
    current fixed constants for them.
    """
    R = np.zeros(R_DIM, dtype=np.float64)
    mg_frozen = 0.0
    eg_frozen = 0.0
    phase = 0

    white = board.occupied_co[WHITE]
    black = board.occupied_co[BLACK]
    wp, wn, wb_, wr, wq, wk = (
        board.pawns & white, board.knights & white, board.bishops & white,
        board.rooks & white, board.queens & white, board.kings & white,
    )
    bp, bn, bb_, br, bq, bk = (
        board.pawns & black, board.knights & black, board.bishops & black,
        board.rooks & black, board.queens & black, board.kings & black,
    )
    white_pieces = (wp, wn, wb_, wr, wq, wk)
    black_pieces = (bp, bn, bb_, br, bq, bk)

    wk_sq = bk_sq = -1
    for i in range(6):
        pt = i + 1
        for sq in _bits(white_pieces[i]):
            idx = sq ^ 56
            R[_PST_OFF + i * 64 + idx] += 1
            phase += PHASE_INC[pt]
            if pt == 6:
                wk_sq = sq
        for sq in _bits(black_pieces[i]):
            R[_PST_OFF + i * 64 + sq] -= 1
            phase += PHASE_INC[pt]
            if pt == 6:
                bk_sq = sq

    w_pawn_attacks = _shl(wp & NOT_FILE_A, 7) | _shl(wp & NOT_FILE_H, 9)
    b_pawn_attacks = (bp & NOT_FILE_H) >> 7 | (bp & NOT_FILE_A) >> 9

    w_occ = wp | wn | wb_ | wr | wq | wk
    b_occ = bp | bn | bb_ | br | bq | bk

    if wp or bp:
        for sq in _bits(_passers(wp, bp, True)):
            r = sq >> 3
            lsb = 1 << sq
            supported = bool(lsb & w_pawn_attacks) or (
                wk_sq >= 0 and chess.square_distance(wk_sq, sq) <= 2
            )
            R[_PASSED_OFF + r] += 1
            if supported:
                R[_SUPPORT_IDX] += 1
            stop = sq + 8
            blockaded = stop < 64 and bool((b_occ >> stop) & 1)
            if blockaded:
                R[_BLOCKADE_IDX] += 1
            if stop < 64 and wk_sq >= 0 and bk_sq >= 0:
                term = 2 * chess.square_distance(bk_sq, stop) - chess.square_distance(wk_sq, stop)
                eg_frozen += term * KP_SCALE[r]

        for sq in _bits(_passers(bp, wp, False)):
            r = 7 - (sq >> 3)
            lsb = 1 << sq
            supported = bool(lsb & b_pawn_attacks) or (
                bk_sq >= 0 and chess.square_distance(bk_sq, sq) <= 2
            )
            R[_PASSED_OFF + r] -= 1
            if supported:
                R[_SUPPORT_IDX] -= 1
            stop = sq - 8
            blockaded = stop >= 0 and bool((w_occ >> stop) & 1)
            if blockaded:
                R[_BLOCKADE_IDX] -= 1
            if stop >= 0 and wk_sq >= 0 and bk_sq >= 0:
                term = 2 * chess.square_distance(wk_sq, stop) - chess.square_distance(bk_sq, stop)
                eg_frozen -= term * KP_SCALE[r]

    # Pawn attack spans (for outposts)
    b_front_span = bp >> 8
    b_front_span |= b_front_span >> 8
    b_front_span |= b_front_span >> 16
    b_front_span |= b_front_span >> 32
    b_attack_span = b_front_span | ((b_front_span & NOT_FILE_A) >> 1) | _shl(b_front_span & NOT_FILE_H, 1)

    w_front_span = _shl(wp, 8)
    w_front_span |= _shl(w_front_span, 8)
    w_front_span |= _shl(w_front_span, 16)
    w_front_span |= _shl(w_front_span, 32)
    w_attack_span = w_front_span | ((w_front_span & NOT_FILE_A) >> 1) | _shl(w_front_span & NOT_FILE_H, 1)

    OUTPOST_RANKS_W = 0x0000FFFFFF000000
    OUTPOST_RANKS_B = 0x000000FFFFFF0000
    w_outposts = w_pawn_attacks & ~b_attack_span & OUTPOST_RANKS_W & MASK64
    b_outposts = b_pawn_attacks & ~w_attack_span & OUTPOST_RANKS_B & MASK64
    R[_OUTPOST_IDX] += _popcount(wn & w_outposts)
    R[_OUTPOST_IDX] -= _popcount(bn & b_outposts)

    # Frozen: king shield (mg only, only vs. enemy queen, castled-ish king)
    if bq and wk:
        if (wk_sq & 7) <= 2 or (wk_sq & 7) >= 5:
            mg_frozen -= _shield_missing(wp, wk_sq, True) * SHIELD_PER_MISSING
    if wq and bk:
        if (bk_sq & 7) <= 2 or (bk_sq & 7) >= 5:
            mg_frozen += _shield_missing(bp, bk_sq, False) * SHIELD_PER_MISSING

    # Frozen: pawn storm (mg only)
    if wk_sq >= 0:
        wk_file = wk_sq & 7
        storm = 0
        for sq in _bits(bp):
            f, rnk = sq & 7, sq >> 3
            if abs(f - wk_file) <= 1 and rnk <= 4:
                storm += (5 - rnk) * 12
        mg_frozen -= storm
    if bk_sq >= 0:
        bk_file = bk_sq & 7
        storm = 0
        for sq in _bits(wp):
            f, rnk = sq & 7, sq >> 3
            if abs(f - bk_file) <= 1 and rnk >= 3:
                storm += rnk * 12
        mg_frozen += storm

    # Doubled / isolated pawns
    if wp:
        u = _front_fill_up(wp)
        w_files = u | (u >> 8)
        w_files |= w_files >> 16
        w_files |= w_files >> 32
        d = _popcount(wp & _shl(u, 8))
        adj = _shl(w_files & NOT_FILE_H, 1) | ((w_files & NOT_FILE_A) >> 1)
        i = _popcount(wp & ~adj & MASK64)
        R[_DOUBLED_IDX] += d
        R[_ISOLATED_IDX] += i
    else:
        w_files = 0

    if bp:
        v = bp | (bp >> 8)
        v |= v >> 16
        v |= v >> 32
        b_files = v | _shl(v, 8)
        b_files |= _shl(b_files, 16)
        b_files |= _shl(b_files, 32)
        d = _popcount(bp & (v >> 8))
        adj = _shl(b_files & NOT_FILE_H, 1) | ((b_files & NOT_FILE_A) >> 1)
        i = _popcount(bp & ~adj & MASK64)
        R[_DOUBLED_IDX] -= d
        R[_ISOLATED_IDX] -= i
    else:
        b_files = 0

    # Rook files
    w_rooks_alone = wr & ~w_files & MASK64
    if w_rooks_alone:
        R[_ROOK_SEMI_IDX] += _popcount(w_rooks_alone & b_files)
        R[_ROOK_OPEN_IDX] += _popcount(w_rooks_alone & ~b_files & MASK64)
    b_rooks_alone = br & ~b_files & MASK64
    if b_rooks_alone:
        R[_ROOK_SEMI_IDX] -= _popcount(b_rooks_alone & w_files)
        R[_ROOK_OPEN_IDX] -= _popcount(b_rooks_alone & ~w_files & MASK64)

    # Frozen: king centralisation / king-pawn proximity in the deep endgame
    if phase <= 8 and wk_sq >= 0 and bk_sq >= 0:
        w_center = abs((wk_sq & 7) - 3.5) + abs((wk_sq >> 3) - 3.5)
        b_center = abs((bk_sq & 7) - 3.5) + abs((bk_sq >> 3) - 3.5)
        eg_frozen += int((7.0 - w_center) * 4)
        eg_frozen -= int((7.0 - b_center) * 4)
    if phase <= 6 and (wp or bp) and wk_sq >= 0 and bk_sq >= 0:
        near = 0
        for sq in _bits(bp):
            near -= chess.square_distance(wk_sq, sq)
        for sq in _bits(wp):
            near += chess.square_distance(bk_sq, sq)
        eg_frozen += 3 * near

    # Bishop pair
    if _popcount(wb_) >= 2:
        R[_BISHOP_PAIR_IDX] += 1
    if _popcount(bb_) >= 2:
        R[_BISHOP_PAIR_IDX] -= 1

    if phase > 24:
        phase = 24

    return R, mg_frozen, eg_frozen, phase, board.turn == WHITE


def _shield_missing(own_pawns: int, king_sq: int, white: bool) -> int:
    kf, kr = king_sq & 7, king_sq >> 3
    lo, hi = max(kf - 1, 0), min(kf + 1, 7)
    r1 = kr + 1 if white else kr - 1
    r2 = kr + 2 if white else kr - 2
    missing = 0
    for f in range(lo, hi + 1):
        near = 0 <= r1 <= 7 and bool((own_pawns >> (r1 * 8 + f)) & 1)
        far = 0 <= r2 <= 7 and bool((own_pawns >> (r2 * 8 + f)) & 1)
        if not (near or far):
            missing += 1
    return missing


def eval_cp(board: chess.Board, params: np.ndarray) -> float:
    """Full evaluation (tunable + frozen terms), from the side-to-move's view.

    Same sign convention as the engine's `evaluate()`: positive is good for
    whoever is to move. Used by the tuner and by the self-check script.
    """
    R, mg_frozen, eg_frozen, phase, white_to_move = extract(board)
    mg_w, eg_w, tempo = unpack(params)
    mg_total = float(R @ mg_w) + mg_frozen
    eg_total = float(R @ eg_w) + eg_frozen
    score = (mg_total * phase + eg_total * (24 - phase)) / 24.0
    score = score if white_to_move else -score
    return score + float(tempo)
