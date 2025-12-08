import numpy as np
import random
import copy
import multiprocessing
import math
import sys
import time

# --- CONSTANTS ---
ROW_COUNT = 6
COLUMN_COUNT = 7
EMPTY = 0
PLAYER_1_PIECE = 1
PLAYER_2_PIECE = 2
WINDOW_LENGTH = 4

# --- GENETIC ALGORITHM SETTINGS ---
POPULATION_SIZE = 32
GENERATIONS = 20
OPPONENTS_PER_GEN = 5
TOURNAMENT_DEPTH = 6
TIME_LIMIT = 0.5


# --- GAME ENGINE ---
def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT))


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == 0


def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r


def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if board[ROW_COUNT - 1][col] == 0:
            valid_locations.append(col)
    return valid_locations


def winning_move(board, piece):
    # Check horizontal
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][
                c + 3] == piece:
                return True
    # Check vertical
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][
                c] == piece:
                return True
    # Check positive diagonal
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][
                c + 3] == piece:
                return True
    # Check negative diagonal
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][
                c + 3] == piece:
                return True
    return False


def is_terminal_node(board):
    return winning_move(board, PLAYER_1_PIECE) or winning_move(board, PLAYER_2_PIECE) or len(
        get_valid_locations(board)) == 0


# --- FAST HEURISTIC EVALUATION ---
def evaluate_window(window, piece, weights):
    score = 0
    opp_piece = PLAYER_1_PIECE if piece == PLAYER_2_PIECE else PLAYER_2_PIECE

    my_count = window.count(piece)
    empty_count = window.count(EMPTY)
    opp_count = window.count(opp_piece)

    if my_count == 4:
        score += weights['W_WIN']
    elif my_count == 3 and empty_count == 1:
        score += weights['W_THREE']
    elif my_count == 2 and empty_count == 2:
        score += weights['W_TWO']

    if opp_count == 3 and empty_count == 1:
        score -= weights['W_BLOCK']

    return score


def score_position(board, piece, weights):
    score = 0

    # Center Column
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * weights['W_CENTER']

    # Horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece, weights)

    # Vertical
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece, weights)

    # Positive Diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece, weights)

    # Negative Diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece, weights)

    return score


# --- FAST ENGINE: MINIMAX + MEMORY ---
def minimax(board, depth, alpha, beta, maximizingPlayer, piece, weights, tt):
    board_key = board.tobytes()

    # 1. Transposition Table Lookup
    if board_key in tt:
        stored_score, stored_flag, stored_depth, stored_move = tt[board_key]
        if stored_depth >= depth:
            if stored_flag == 'EXACT':
                return stored_move, stored_score
            elif stored_flag == 'LOWER':
                alpha = max(alpha, stored_score)
            elif stored_flag == 'UPPER':
                beta = min(beta, stored_score)
            if alpha >= beta: return stored_move, stored_score

    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    opp_piece = PLAYER_1_PIECE if piece == PLAYER_2_PIECE else PLAYER_2_PIECE

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, piece):
                return (None, 100000000)
            elif winning_move(board, opp_piece):
                return (None, -100000000)
            else:
                return (None, 0)
        else:
            return (None, score_position(board, piece, weights))

    # 2. Move Ordering (Try previous best move first)
    if board_key in tt:
        best_prev_move = tt[board_key][3]
        if best_prev_move in valid_locations:
            valid_locations.insert(0, valid_locations.pop(valid_locations.index(best_prev_move)))
    else:
        # Heuristic sort: Center out
        center = COLUMN_COUNT // 2
        valid_locations.sort(key=lambda x: abs(x - center))

    best_move = random.choice(valid_locations)

    if maximizingPlayer:
        value = -math.inf
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, piece)
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, piece, weights, tt)[1]
            if new_score > value:
                value = new_score
                best_move = col
            alpha = max(alpha, value)
            if alpha >= beta: break
    else:
        value = math.inf
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, opp_piece)
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, piece, weights, tt)[1]
            if new_score < value:
                value = new_score
                best_move = col
            beta = min(beta, value)
            if alpha >= beta: break

    # 3. Store in Table
    flag = 'EXACT'
    if value <= alpha:
        flag = 'UPPER'
    elif value >= beta:
        flag = 'LOWER'
    tt[board_key] = (value, flag, depth, best_move)

    return best_move, value


# --- ITERATIVE DEEPENING AGENT ---
def get_best_move(board, piece, weights, max_depth, tt):
    start_time = time.time()
    best_col = random.choice(get_valid_locations(board))

    for d in range(1, max_depth + 1):
        if time.time() - start_time > TIME_LIMIT:
            break
        col, score = minimax(board, d, -math.inf, math.inf, True, piece, weights, tt)
        best_col = col
        if score > 90000000:  # Found forced win
            break
    return best_col


# --- MULTIPROCESSING WORKER ---
def play_match(args):
    g1, g2 = args
    g1_score = 0
    g2_score = 0

    # Each bot gets its own Transposition Table for the duration of the match
    # This allows Iterative Deepening to work efficiently
    tt_p1 = {}
    tt_p2 = {}

    # Game 1: g1 is Player 1
    board = create_board()
    turn = 0
    while not is_terminal_node(board):
        if turn == 0:
            col = get_best_move(board, PLAYER_1_PIECE, g1['weights'], TOURNAMENT_DEPTH, tt_p1)
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, PLAYER_1_PIECE)
            if winning_move(board, PLAYER_1_PIECE): g1_score += 1; break
        else:
            col = get_best_move(board, PLAYER_2_PIECE, g2['weights'], TOURNAMENT_DEPTH, tt_p2)
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, PLAYER_2_PIECE)
            if winning_move(board, PLAYER_2_PIECE): g2_score += 1; break

        turn = (turn + 1) % 2
        if len(get_valid_locations(board)) == 0: g1_score += 0.5; g2_score += 0.5; break

    # Reset TTs for fair second game
    tt_p1.clear()
    tt_p2.clear()

    # Game 2: g2 is Player 1
    board = create_board()
    turn = 0  # g2 goes first
    while not is_terminal_node(board):
        if turn == 0:
            col = get_best_move(board, PLAYER_1_PIECE, g2['weights'], TOURNAMENT_DEPTH, tt_p2)
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, PLAYER_1_PIECE)
            if winning_move(board, PLAYER_1_PIECE): g2_score += 1; break
        else:
            col = get_best_move(board, PLAYER_2_PIECE, g1['weights'], TOURNAMENT_DEPTH, tt_p1)
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, PLAYER_2_PIECE)
            if winning_move(board, PLAYER_2_PIECE): g1_score += 1; break

        turn = (turn + 1) % 2
        if len(get_valid_locations(board)) == 0: g1_score += 0.5; g2_score += 0.5; break

    return (g1['id'], g1_score), (g2['id'], g2_score)


# --- GENETIC HELPERS ---
def create_initial_population(size):
    population = []
    for i in range(size):
        weights = {
            'W_CENTER': random.randint(0, 10),
            'W_WIN': 1000000,  # Fixed High Value
            'W_THREE': random.randint(1, 20),
            'W_TWO': random.randint(1, 10),
            'W_BLOCK': random.randint(1, 100)
        }
        population.append({'id': i, 'weights': weights, 'score': 0})
    return population


def mutate(weights):
    new_weights = weights.copy()
    trait = random.choice(['W_CENTER', 'W_THREE', 'W_TWO', 'W_BLOCK'])
    change = random.choice([-5, -2, -1, 1, 2, 5])
    new_weights[trait] = max(0, new_weights[trait] + change)
    return new_weights


# --- MAIN DRIVER ---
if __name__ == "__main__":
    multiprocessing.freeze_support()

    print(f"--- FAST GENETIC OPTIMIZATION ---")
    print(f"Algorithm: Iterative Deepening + Transposition Table")
    print(f"Depth Limit: {TOURNAMENT_DEPTH} | Time Limit: {TIME_LIMIT}s")
    print(f"Cores: {multiprocessing.cpu_count()}")

    population = create_initial_population(POPULATION_SIZE)

    for gen in range(GENERATIONS):
        print(f"\nGENERATION {gen + 1}/{GENERATIONS}")

        # 1. Create Matchups
        matchups = []
        indices = list(range(POPULATION_SIZE))
        for _ in range(OPPONENTS_PER_GEN):
            random.shuffle(indices)
            for i in range(0, POPULATION_SIZE, 2):
                p1 = population[indices[i]]
                p2 = population[indices[i + 1]]
                matchups.append((p1, p2))

        # 2. Run in Parallel
        with multiprocessing.Pool() as pool:
            results = pool.map(play_match, matchups)

        # 3. Score Aggregation
        score_map = {p['id']: 0 for p in population}
        for r in results:
            (id1, s1), (id2, s2) = r
            score_map[id1] += s1
            score_map[id2] += s2

        for p in population:
            p['score'] = score_map[p['id']]

        # 4. Selection
        population.sort(key=lambda x: x['score'], reverse=True)
        top_half = population[:POPULATION_SIZE // 2]
        best = top_half[0]

        print(f"Best Bot: {best['weights']} (Score: {best['score']}/{OPPONENTS_PER_GEN * 2})")

        # 5. Reproduction
        next_gen = []
        for parent in top_half:
            parent['score'] = 0
            next_gen.append(parent)

            child_weights = mutate(parent['weights'])
            new_id = len(next_gen) + ((gen + 1) * 100000)  # Robust ID generation
            child = {'id': new_id, 'weights': child_weights, 'score': 0}
            next_gen.append(child)

        population = next_gen

    print("\n--- DONE ---")
    print(f"Champion Weights: {population[0]['weights']}")