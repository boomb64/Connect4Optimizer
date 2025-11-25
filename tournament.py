import numpy as np
import random
import copy
import sys

# --- CONSTANTS ---
ROW_COUNT = 6
COLUMN_COUNT = 7
EMPTY = 0
PLAYER_1_PIECE = 1
PLAYER_2_PIECE = 2
WINDOW_LENGTH = 4

# --- CONFIGURATION ---
# Lower depth = faster tournament.
# Depth 2 is fast (seconds). Depth 4 is standard (minutes).
TOURNAMENT_DEPTH = 2
GAMES_PER_MATCHUP = 10  # How many times each pair plays (for statistical significance)


class Bot:
    def __init__(self, name, weights):
        self.name = name
        self.weights = weights
        self.points = 0  # Tournament Score (Win=1, Draw=0.5)
        self.wins = 0
        self.losses = 0
        self.draws = 0


# --- GAME ENGINE (HEADLESS) ---
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
    # Check diagonals
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


def is_terminal_node(board):
    return winning_move(board, PLAYER_1_PIECE) or winning_move(board, PLAYER_2_PIECE) or len(
        get_valid_locations(board)) == 0


def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations


# --- DYNAMIC AI ENGINE ---
def evaluate_window(window, piece, weights):
    score = 0
    opp_piece = PLAYER_1_PIECE
    if piece == PLAYER_1_PIECE:
        opp_piece = PLAYER_2_PIECE

    # Use the passed 'weights' dictionary
    if window.count(piece) == 4:
        score += weights['W_WIN']
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += weights['W_THREE']
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += weights['W_TWO']

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
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

    # Diagonals
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece, weights)
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece, weights)

    return score


def minimax(board, depth, alpha, beta, maximizingPlayer, piece, weights):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    # Identify Opponent Piece
    opp_piece = PLAYER_1_PIECE if piece == PLAYER_2_PIECE else PLAYER_2_PIECE

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, piece):
                return (None, 10000000)
            elif winning_move(board, opp_piece):
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
            drop_piece(b_copy, row, col, piece)
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
            drop_piece(b_copy, row, col, opp_piece)
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, piece, weights)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta: break
        return column, value


# --- TOURNAMENT LOGIC ---
def play_game(bot1, bot2):
    board = create_board()
    game_over = False
    turn = random.randint(0, 1)  # Randomize who goes first

    while not game_over:
        if len(get_valid_locations(board)) == 0:
            return "DRAW"

        if turn == 0:  # Bot 1
            col, score = minimax(board, TOURNAMENT_DEPTH, -np.inf, np.inf, True, PLAYER_1_PIECE, bot1.weights)
            if col is None: return "DRAW"
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, PLAYER_1_PIECE)
            if winning_move(board, PLAYER_1_PIECE): return bot1.name
            turn = 1

        else:  # Bot 2
            col, score = minimax(board, TOURNAMENT_DEPTH, -np.inf, np.inf, True, PLAYER_2_PIECE, bot2.weights)
            if col is None: return "DRAW"
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, PLAYER_2_PIECE)
            if winning_move(board, PLAYER_2_PIECE): return bot2.name
            turn = 0


def run_tournament():
    # 1. DEFINE YOUR CONTESTANTS
    bots = [
        Bot("Balanced", {'W_CENTER': 3, 'W_WIN': 100, 'W_THREE': 5, 'W_TWO': 2, 'W_BLOCK': 4}),
        Bot("Aggressive", {'W_CENTER': 3, 'W_WIN': 100, 'W_THREE': 10, 'W_TWO': 5, 'W_BLOCK': 1}),  # High offense
        Bot("Defensive", {'W_CENTER': 3, 'W_WIN': 100, 'W_THREE': 2, 'W_TWO': 1, 'W_BLOCK': 100}),  # High defense
        Bot("CenterHog", {'W_CENTER': 10, 'W_WIN': 100, 'W_THREE': 5, 'W_TWO': 2, 'W_BLOCK': 4}),  # Loves the middle
        Bot("Erratic", {'W_CENTER': 0, 'W_WIN': 100, 'W_THREE': 5, 'W_TWO': 2, 'W_BLOCK': 4})
        # Doesn't care about center
    ]

    print(f"--- STARTING TOURNAMENT ({GAMES_PER_MATCHUP} games per matchup) ---")

    # Round Robin
    for i in range(len(bots)):
        for j in range(i + 1, len(bots)):
            b1 = bots[i]
            b2 = bots[j]
            print(f"Matchup: {b1.name} vs {b2.name}...", end="")

            for _ in range(GAMES_PER_MATCHUP):
                winner = play_game(b1, b2)
                if winner == b1.name:
                    b1.wins += 1
                    b1.points += 1
                    b2.losses += 1
                elif winner == b2.name:
                    b2.wins += 1
                    b2.points += 1
                    b1.losses += 1
                else:
                    b1.draws += 1
                    b2.draws += 1
                    b1.points += 0.5
                    b2.points += 0.5
            print(" Done.")

    # Results
    print("\n--- FINAL STANDINGS ---")
    bots.sort(key=lambda x: x.points, reverse=True)
    print(f"{'BOT NAME':<15} {'POINTS':<8} {'WINS':<6} {'LOSS':<6} {'DRAWS':<6}")
    print("-" * 50)
    for b in bots:
        print(f"{b.name:<15} {b.points:<8} {b.wins:<6} {b.losses:<6} {b.draws:<6}")


if __name__ == "__main__":
    run_tournament()