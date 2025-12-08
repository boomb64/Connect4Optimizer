import numpy as np
import math
import random
import sys

# --- CONFIGURATION & CONSTANTS ---
ROW_COUNT = 6
COLUMN_COUNT = 7

# 0 = Greedy Agent (Depth 1), 1 = Optimization AI (Depth 4)
PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4

# SIMULATION SETTINGS
TOTAL_GAMES = 20
MINIMAX_DEPTH = 4  # The "Deep Thinker"

# --- OPTIMIZATION WEIGHTS ---
# Both agents use these weights, so it is a fair test of Intelligence vs. Speed
W_CENTER = 3
W_WIN = 100
W_THREE = 1
W_TWO = 8
W_BLOCK = 29


# --- GAME LOGIC FUNCTIONS ---
def create_board():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    return board


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == 0


def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r


def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][
                c + 3] == piece:
                return True

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][
                c] == piece:
                return True

    # Check positively sloped diaganols
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][
                c + 3] == piece:
                return True

    # Check negatively sloped diaganols
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][
                c + 3] == piece:
                return True


# --- OPTIMIZATION ENGINE (Evaluation Function) ---
def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opp_piece = AI_PIECE

    # OFFENSE
    if window.count(piece) == 4:
        score += W_WIN
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += W_THREE
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += W_TWO

    # DEFENSE
    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= W_BLOCK

    return score


def score_position(board, piece):
    score = 0
    # Score Center Column
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * W_CENTER

    # Score Horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Score Vertical
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Score Positive Sloped Diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score


def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0


def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations


# --- GREEDY AGENT (DEPTH 1) ---
def pick_best_move(board, piece):
    valid_locations = get_valid_locations(board)
    best_score = -math.inf
    best_col = random.choice(valid_locations)

    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, piece)
        score = score_position(temp_board, piece)

        if score > best_score:
            best_score = score
            best_col = col

    return best_col


# --- MINIMAX AGENT (DEPTH 4) ---
def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, 100000000000000)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -10000000000000)
            else:
                return (None, 0)
        else:
            return (None, score_position(board, AI_PIECE))

    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value

    else:  # Minimizing Player
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value


# --- MAIN EXECUTION (HEADLESS SIMULATION) ---
if __name__ == "__main__":
    ai_wins = 0
    greedy_wins = 0
    draws = 0

    print(f"--- STARTING VERIFICATION: AI (Depth {MINIMAX_DEPTH}) vs GREEDY (Depth 1) ---")
    print(f"Games to Play: {TOTAL_GAMES}")
    print("Running...")

    for game in range(TOTAL_GAMES):
        board = create_board()
        game_over = False
        turn = random.randint(PLAYER, AI)  # Randomize who starts

        while not game_over:
            # Check for draw
            if len(get_valid_locations(board)) == 0:
                draws += 1
                break

            if turn == PLAYER:
                # --- GREEDY AGENT LOGIC (Depth 1) ---
                col = pick_best_move(board, PLAYER_PIECE)

                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, PLAYER_PIECE)

                    if winning_move(board, PLAYER_PIECE):
                        greedy_wins += 1
                        game_over = True

                    turn += 1
                    turn = turn % 2

            elif turn == AI:
                # --- MINIMAX AGENT LOGIC (Depth 4) ---
                col, minimax_score = minimax(board, MINIMAX_DEPTH, -math.inf, math.inf, True)

                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, AI_PIECE)

                    if winning_move(board, AI_PIECE):
                        ai_wins += 1
                        game_over = True

                    turn += 1
                    turn = turn % 2

        print(f"Game {game + 1}/{TOTAL_GAMES} Finished. (AI: {ai_wins}, Greedy: {greedy_wins})")

    print("\n--- FINAL RESULTS ---")
    print(f"Total Games: {TOTAL_GAMES}")
    print(f"AI Wins (Depth 4):     {ai_wins} ({(ai_wins / TOTAL_GAMES) * 100}%)")
    print(f"Greedy Wins (Depth 1): {greedy_wins}")
    print(f"Draws:                 {draws}")