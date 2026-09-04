"""AI Chessathon agent.

Interface required by the platform:
    get_move(fen: str, time_left_ms: int) -> str    # UCI, e.g. "e2e4" / "e7e8q"
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

# Import our tuned variables dynamically
from weights import *


# Game-phase weight per piece type (total 24 at the starting position).
PHASE_INC = {1: 0, 2: 1, 3: 1, 4: 2, 5: 4, 6: 0}
phase_inc_np = np.array([0, 0, 1, 1, 2, 4, 0], dtype=np.int32)

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

# Most-valuable-victim / least-valuable-attacker table (Index 0 acts as the None fallback)
SEE_VALUE = (100, 100, 320, 330, 500, 900, 20000)

_SCORE_OF = operator.itemgetter(0)
EXACT, UPPER, LOWER = 0, 1, 2
TT_MAX_ENTRIES = 2_000_000
START_DEPTH = 4
START_DEPTH_MIN_MS = 1500
TT_CLEAR_EACH_MOVE = False

WHITE = chess.WHITE
BLACK = chess.BLACK

class _Timeout(Exception):
    pass

Bitboard = Any

SHIELD_PER_MISSING = 20

_SHIELD_SIG = int32(uint64, int32, b1)
_PASSERS_SIG = uint64(uint64, uint64, b1)
_POPCOUNT_SIG = int32(uint64)
_EVAL_SIG = int32(
    uint64, uint64, uint64, uint64, uint64, uint64,
    uint64, uint64, uint64, uint64, uint64, uint64, b1,
)

@njit(_SHIELD_SIG, cache=True, fastmath=True, boundscheck=False)
def _shield_missing(own_pawns: Bitboard, king_sq: int, white: bool) -> int:
    kf = king_sq & 7
    kr = king_sq >> 3
    lo = kf - 1 if kf > 0 else 0
    hi = kf + 1 if kf < 7 else 7
    r1 = kr + 1 if white else kr - 1
    r2 = kr + 2 if white else kr - 2
    missing = 0
    for f in range(lo, hi + 1):
        near = 0 <= r1 <= 7 and (own_pawns >> np.uint64(r1 * 8 + f)) & np.uint64(1)
        far = 0 <= r2 <= 7 and (own_pawns >> np.uint64(r2 * 8 + f)) & np.uint64(1)
        if not (near or far):
            missing += 1
    return missing

@njit(_PASSERS_SIG, cache=True, fastmath=True, boundscheck=False)
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

@njit(_POPCOUNT_SIG, cache=True, fastmath=True, boundscheck=False)
def popcount(n: Bitboard) -> int:
    count = 0
    while n:
        n &= n - np.uint64(1)
        count += 1
    return count

@njit(_EVAL_SIG, cache=True, fastmath=True, boundscheck=False, nogil=True)
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
            
            if is_supported:
                mg_bonus += SUPPORT_BONUS_MG
                eg_bonus += SUPPORT_BONUS_EG
            
            stop = sq + 8
            if stop < 64 and (b_occ >> np.uint64(stop)) & np.uint64(1):
                mg_bonus += BLOCKADE_PENALTY_MG
                eg_bonus += BLOCKADE_PENALTY_EG

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
            
            if is_supported:
                mg_bonus += SUPPORT_BONUS_MG
                eg_bonus += SUPPORT_BONUS_EG
            
            stop = sq - 8
            if stop >= 0 and (w_occ >> np.uint64(stop)) & np.uint64(1):
                mg_bonus += BLOCKADE_PENALTY_MG
                eg_bonus += BLOCKADE_PENALTY_EG

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
        mg += w_ko_count * OUTPOST_MG
        eg += w_ko_count * OUTPOST_EG

    b_ko_count = popcount(bn & b_outposts)
    if b_ko_count:
        mg -= b_ko_count * OUTPOST_MG
        eg -= b_ko_count * OUTPOST_EG

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
        mg += DOUBLED_MG * d + ISOLATED_MG * i
        eg += DOUBLED_EG * d + ISOLATED_EG * i

    if bp:
        d = popcount(bp & (v >> np.uint64(8)))
        adj = ((b_files & NOT_FILE_H) << np.uint64(1)) | ((b_files & NOT_FILE_A) >> np.uint64(1))
        i = popcount(bp & ~adj)
        mg -= DOUBLED_MG * d + ISOLATED_MG * i
        eg -= DOUBLED_EG * d + ISOLATED_EG * i

    w_rooks_alone = wr & ~w_files
    if w_rooks_alone:
        semi = popcount(w_rooks_alone & b_files)
        opn = popcount(w_rooks_alone & ~b_files)
        mg += ROOK_OPEN_MG * opn + ROOK_SEMI_MG * semi
        eg += ROOK_OPEN_EG * opn + ROOK_SEMI_EG * semi

    b_rooks_alone = br & ~b_files
    if b_rooks_alone:
        semi = popcount(b_rooks_alone & w_files)
        opn = popcount(b_rooks_alone & ~w_files)
        mg -= ROOK_OPEN_MG * opn + ROOK_SEMI_MG * semi
        eg -= ROOK_OPEN_EG * opn + ROOK_SEMI_EG * semi

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
        mg += BISHOP_PAIR_MG
        eg += BISHOP_PAIR_EG
    if popcount(bb) >= 2:
        mg -= BISHOP_PAIR_MG
        eg -= BISHOP_PAIR_EG

    if bq != np.uint64(0) and wk != np.uint64(0):
        wk_sq = 0
        t = wk
        while t > np.uint64(1):
            t >>= np.uint64(1)
            wk_sq += 1
        if (wk_sq & 7) <= 2 or (wk_sq & 7) >= 5:     
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
        return score + TEMPO
    return -score + TEMPO


def evaluate(board: chess.Board) -> int:
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
    evaluate(chess.Board())


class Searcher:
    def __init__(self) -> None:
        self.tt: dict[int, tuple[int, int, int, int, chess.Move | None]] = {}
        self.killers: list[list[chess.Move | None]] = [
            [None, None] for _ in range(MAX_PLY + 2)
        ]
        self.history: dict[tuple[chess.Color, int, int], int] = {}
        self.nodes = 0
        self.deadline = 0.0
        self.soft_deadline = 0.0
        self.seldepth = 0
        self.path: dict[Hashable, int] = {}
        self._taint = False
        self.stop_flag = False

    def new_game(self) -> None:
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
            score = CAPTURE_BONUS + SEE_VALUE[victim if victim is not None else 0] * 16 - SEE_VALUE[attacker]
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
        history = self.history
        turn = board.turn
        piece_type_at = board.piece_type_at
        ep = board.ep_square
        ep_from = board.pawns & board.occupied_co[turn] if ep is not None else 0

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

        scored: list[tuple[int, bool, chess.Move]] = []
        add = scored.append
        for move in moves:
            to_sq = move.to_square
            from_sq = move.from_square
            promotion = move.promotion

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
        if prev is None or prev[0] != key or prev[1] <= depth:
            self.tt[idx] = (key, depth, flag, value, move)

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
            promo_rank = chess.BB_RANK_7 if board.turn == WHITE else chess.BB_RANK_2
            if board.pawns & board.occupied_co[board.turn] & promo_rank:
                for m in board.legal_moves:
                    if m.promotion == 5 and board.piece_type_at(m.to_square) is None:
                        moves.append(m)

        ordered = self._sorted_moves(board, moves, None, ply)

        for _score, _quiet, move in ordered:
            if not in_check:
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

        tkey = board._transposition_key()

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
            depth += 1  

        if depth <= 0:
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
                    self._taint = False
                    return val
            if tt_move is not None and not board.is_legal(tt_move):
                tt_move = None

        if tt_move is None and depth >= 4:
            depth -= 1

        static: int | None = None

        if not in_check and depth <= 4 and abs(beta) < MATE_BOUND:
            static = evaluate(board)
            if static - 85 * depth >= beta:
                self._taint = False
                return static

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
                if (
                    is_quiet
                    and not in_check
                    and depth <= 3
                    and i > 6 + depth * 6
                    and best > -MATE_BOUND
                ):
                    continue

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
            
        if not best_taint:
            self._store(key, depth, flag, best, best_move, ply)

        self._taint = best_taint
        return best

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

                    if failed:  
                        beta = min(INF, beta + 200 + (beta - alpha))
                        continue
                    if iter_score <= alpha and alpha != -INF:  
                        alpha = max(-INF, alpha - 200 - (beta - alpha))
                        beta = INF
                        continue
                    break

                best_move = iter_best
                best_score = iter_score
                depth_done = depth

                side_to_move = "white" if board.turn else "black"
                print(
                    f"agent: info depth {depth} score cp {best_score} (for {side_to_move}) bestmove {best_move.uci() if best_move else 'none'}",
                    file=sys.stderr
                )

                if abs(best_score) > MATE_BOUND:
                    break  

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

_BOOK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "book.bin")

def _open_book() -> chess.polyglot.MemoryMappedReader | None:
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
    if _BOOK_READER is None:
        return None
    try:
        entry = _BOOK_READER.weighted_choice(board)
    except Exception:
        return None
    move = entry.move
    if move not in board.legal_moves:
        return None
    return move


# =============================================================================
# Entry point
# =============================================================================

_warm_jit()
_SEARCHER = Searcher()

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

OVERHEAD_MS = 120
INCREMENT_MS = 500
PANIC_FLOOR_MS = 250
MOVES_REMAINING = 30.0   
HARD_MULTIPLIER = 3.0
MAX_FRACTION = 0.40

def _budget(time_left_ms: int) -> tuple[float, float]:
    left = time_left_ms - OVERHEAD_MS
    if left <= 0:
        return 1.0, 1.0
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
    them = not board.turn
    best: chess.Move | None = None
    best_score = -1 << 30

    for move in board.legal_moves:
        victim = board.piece_type_at(move.to_square)
        gain = SEE_VALUE[victim if victim is not None else 0]
        mover = board.piece_type_at(move.from_square) or 1

        board.push(move)
        mate = board.is_checkmate()
        hanging = board.is_attacked_by(them, move.to_square)
        risk = 0 if mate else (SEE_VALUE[mover] if hanging else 0)
        board.pop()

        score = gain - risk + (1 << 20 if mate else 0)
        if score > best_score:
            best_score = score
            best = move

    return best.uci() if best is not None else "0000"


def get_move(fen: str, time_left_ms: int) -> str:
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
            print(f"agent: searched move {move} rejected, playing {chosen}",
                  file=sys.stderr)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        print(f"agent: search failed on {fen!r}, playing {chosen}", file=sys.stderr)

    _note_after(board, chosen)
    _start_ponder(board, chosen, tl)
    return chosen


def _note_after(board: chess.Board, uci: str) -> None:
    try:
        board.push(chess.Move.from_uci(uci))
        _note_position(board)
        board.pop()
    except Exception:
        pass


# =============================================================================
# Pondering
# =============================================================================

PONDER_ENABLED = True
PONDER_MAX_S = 45.0
PONDER_JOIN_S = 0.5
PONDER_MIN_CLOCK_MS = 3000

_PONDER = Searcher()
_PONDER.tt = _SEARCHER.tt
_PONDER_THREAD: threading.Thread | None = None

def _ponder_worker(board: chess.Board) -> None:
    try:
        _PONDER.go(board, PONDER_MAX_S * 1000.0, PONDER_MAX_S * 1000.0)
    except Exception as exc:
        print(f"agent: ponder failed: {exc!r}", file=sys.stderr)


def _stop_ponder() -> None:
    global _PONDER_THREAD
    thread = _PONDER_THREAD
    if thread is None:
        return
    _PONDER.stop_flag = True
    try:
        thread.join(PONDER_JOIN_S)
        if thread.is_alive():
            print("agent: ponder did not stop within "
                  f"{PONDER_JOIN_S}s, continuing anyway", file=sys.stderr)
            return
    except Exception:
        pass
    _PONDER_THREAD = None


def _start_ponder(board: chess.Board, uci: str, time_left_ms: int) -> None:
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