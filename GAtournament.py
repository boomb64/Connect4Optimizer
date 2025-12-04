import numpy as np
import random
import copy
import multiprocessing
import time
import sys

# --- CONSTANTS ---
ROW_COUNT = 6
COLUMN_COUNT = 7
EMPTY = 0
PLAYER_1_PIECE = 1
PLAYER_2_PIECE = 2
WINDOW_LENGTH = 4

# --- GENETIC ALGORITHM SETTINGS ---
POPULATION_SIZE = 32  # Must be even number
GENERATIONS = 10  # How many times to evolve
GAMES_PER_MATCHUP = 2  # Low number for speed (1 as P1, 1 as P2)
SEARCH_DEPTH = 7  # RECOMMENDATION: Train at Depth 4, Verify at Depth 7


# Depth 7 is too slow for training (hours vs minutes)

# --- GAME ENGINE (HEADLESS & FAST) ---
def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == 0


def get_next_open_row(board, col):
    # Fast numpy lookup
    return np.where(board[:, col] == 0)[0][0]


def check_win(board, piece):
    # Horizontal
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][
                c + 3] == piece:
                return True
    # Vertical
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][
                c] == piece:
                return True
    # Diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][
                c + 3] == piece:
                return True
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][
                c + 3] == piece:
                return True
    return False


def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if board[ROW_COUNT - 1][col] == 0:
            valid_locations.append(col)
    return valid_locations


def is_terminal_node(board):
    return check_win(board, PLAYER_1_PIECE) or check_win(board, PLAYER_2_PIECE) or len(get_valid_locations(board)) == 0


# --- AI LOGIC ---
def evaluate_window(window, piece, weights):
    score = 0
    opp_piece = PLAYER_1_PIECE if piece == PLAYER_2_PIECE else PLAYER_2_PIECE

    my_count = np.count_nonzero(window == piece)
    empty_count = np.count_nonzero(window == EMPTY)
    opp_count = np.count_nonzero(window == opp_piece)

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

    # Center Column Score
    center_array = board[:, COLUMN_COUNT // 2]
    center_count = np.count_nonzero(center_array == piece)
    score += center_count * weights['W_CENTER']

    # Horizontal
    for r in range(ROW_COUNT):
        row_array = board[r, :]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece, weights)

    # Vertical
    for c in range(COLUMN_COUNT):
        col_array = board[:, c]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece, weights)

    # Positive Diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = np.array([board[r + i][c + i] for i in range(WINDOW_LENGTH)])
            score += evaluate_window(window, piece, weights)

    # Negative Diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = np.array([board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)])
            score += evaluate_window(window, piece, weights)

    return score


def minimax(board, depth, alpha, beta, maximizingPlayer, piece, weights):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    opp_piece = PLAYER_1_PIECE if piece == PLAYER_2_PIECE else PLAYER_2_PIECE

    if depth == 0 or is_terminal:
        if is_terminal:
            if check_win(board, piece):
                return (None, 10000000)
            elif check_win(board, opp_piece):
                return (None, -10000000)
            else:
                return (None, 0)
        else:
            return (None, score_position(board, piece, weights))

    if maximizingPlayer:
        value = -np.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            b_copy[row][col] = piece
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, piece, weights)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta: break
        return column, value
    else:
        value = np.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            b_copy[row][col] = opp_piece
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, piece, weights)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta: break
        return column, value


# --- WORKER FUNCTION FOR MULTIPROCESSING ---
def play_match(args):
    """
    Plays a set of games between two genomes.
    Returns: (id_1, score_1), (id_2, score_2)
    """
    g1, g2 = args
    g1_score = 0
    g2_score = 0

    # Game 1: g1 goes first
    board = create_board()
    turn = 0
    while not is_terminal_node(board):
        if turn == 0:
            col, _ = minimax(board, SEARCH_DEPTH, -np.inf, np.inf, True, PLAYER_1_PIECE, g1['weights'])
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, PLAYER_1_PIECE)
            if check_win(board, PLAYER_1_PIECE): g1_score += 1; break
        else:
            col, _ = minimax(board, SEARCH_DEPTH, -np.inf, np.inf, True, PLAYER_2_PIECE, g2['weights'])
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, PLAYER_2_PIECE)
            if check_win(board, PLAYER_2_PIECE): g2_score += 1; break
        turn = (turn + 1) % 2

        # Check Draw
        if len(get_valid_locations(board)) == 0:
            g1_score += 0.5;
            g2_score += 0.5;
            break

    # Game 2: g2 goes first (Swap sides)
    board = create_board()
    turn = 1  # Player 2 (who is now g1 playing as P2 piece)
    while not is_terminal_node(board):
        if turn == 0:  # g2 is P1
            col, _ = minimax(board, SEARCH_DEPTH, -np.inf, np.inf, True, PLAYER_1_PIECE, g2['weights'])
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, PLAYER_1_PIECE)
            if check_win(board, PLAYER_1_PIECE): g2_score += 1; break
        else:  # g1 is P2
            col, _ = minimax(board, SEARCH_DEPTH, -np.inf, np.inf, True, PLAYER_2_PIECE, g1['weights'])
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, PLAYER_2_PIECE)
            if check_win(board, PLAYER_2_PIECE): g1_score += 1; break
        turn = (turn + 1) % 2
        if len(get_valid_locations(board)) == 0:
            g1_score += 0.5;
            g2_score += 0.5;
            break

    return (g1['id'], g1_score), (g2['id'], g2_score)


# --- GENETIC ALGORITHM HELPERS ---
def create_initial_population(size):
    population = []
    for i in range(size):
        # Random weights to start
        weights = {
            'W_CENTER': random.randint(0, 10),
            'W_WIN': 100,  # Keep this fixed mostly
            'W_THREE': random.randint(1, 20),
            'W_TWO': random.randint(1, 10),
            'W_BLOCK': random.randint(1, 100)  # Big variance here
        }
        population.append({'id': i, 'weights': weights, 'score': 0})
    return population


def mutate(weights):
    new_weights = weights.copy()
    # Mutate one random trait
    trait = random.choice(['W_CENTER', 'W_THREE', 'W_TWO', 'W_BLOCK'])
    change = random.choice([-2, -1, 1, 2, 5, -5])
    new_weights[trait] = max(0, new_weights[trait] + change)  # Ensure positive
    return new_weights


# --- MAIN DRIVER ---
# --- UPDATED MAIN DRIVER FOR STABILITY ---
if __name__ == "__main__":
    multiprocessing.freeze_support()

    # SETUP: INCREASED ACCURACY
    OPPONENTS_PER_GEN = 5  # Each bot plays 5 random opponents now (instead of 1)
    POPULATION_SIZE = 32  # Keep this size
    GENERATIONS = 15  # Slightly longer to allow convergence

    print(f"--- GENETIC OPTIMIZATION STARTING ---")
    print(f"Population: {POPULATION_SIZE} | Depth: {SEARCH_DEPTH} | Cores: {multiprocessing.cpu_count()}")
    print(f"Strategy: Each bot plays {OPPONENTS_PER_GEN} opponents per generation to reduce luck.")

    population = create_initial_population(POPULATION_SIZE)

    for gen in range(GENERATIONS):
        print(f"\nGENERATION {gen + 1}/{GENERATIONS}")

        # 1. Create Matchups (The Gauntlet)
        # Every bot plays 'OPPONENTS_PER_GEN' random other bots
        matchups = []
        indices = list(range(POPULATION_SIZE))

        for _ in range(OPPONENTS_PER_GEN):
            random.shuffle(indices)
            # Create pairs from the shuffled list
            for i in range(0, POPULATION_SIZE, 2):
                p1 = population[indices[i]]
                p2 = population[indices[i + 1]]
                matchups.append((p1, p2))

        # 2. Run Matches in Parallel
        with multiprocessing.Pool() as pool:
            results = pool.map(play_match, matchups)

        # 3. Update Scores (Accumulate over all games)
        score_map = {p['id']: 0 for p in population}

        for r in results:
            (id1, s1), (id2, s2) = r
            score_map[id1] += s1
            score_map[id2] += s2

        for p in population:
            # Update score (resetting previous gen score, keeping only this gen's performance)
            p['score'] = score_map[p['id']]

        # 4. Selection
        population.sort(key=lambda x: x['score'], reverse=True)
        top_half = population[:POPULATION_SIZE // 2]

        # Print the "Alpha" of this generation
        best = top_half[0]
        print(f"Best Bot: {best['weights']} (Score: {best['score']}/{OPPONENTS_PER_GEN * 2})")

<<<<<<< HEAD
        # ... inside the main loop ...

=======
>>>>>>> c5956db919b7a7e8d386c2b2cb36b6562ac47789
        # 5. Reproduction
        next_gen = []
        for parent in top_half:
            # Elite Preservation (The parent survives unchanged)
            parent['score'] = 0
            next_gen.append(parent)

            # Mutation (Child is born)
            child_weights = mutate(parent['weights'])

            # --- FIX: USE A ROBUST ID GENERATOR ---
            # We use the length of next_gen + a huge offset based on generation + 1
            # This guarantees children never share IDs with parents
            new_id = len(next_gen) + ((gen + 1) * 10000)

            child = {'id': new_id, 'weights': child_weights, 'score': 0}
            next_gen.append(child)

        population = next_gen

    print("\n--- OPTIMIZATION COMPLETE ---")
    print("Top 3 Converged Configurations:")
    for i in range(3):
        print(f"#{i + 1}: {population[i]['weights']}")