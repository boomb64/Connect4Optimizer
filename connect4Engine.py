import numpy as np
import pygame
import sys
import math
import random
import time

# --- CONFIGURATION & CONSTANTS ---
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4

# --- OPTIMIZATION WEIGHTS (CHAMPION GENOME) ---
# Updated W_WIN to 1,000,000 to ensure it overrides all defensive scores
W_CENTER = 3
W_WIN = 1000000
W_THREE = 1
W_TWO = 8
W_BLOCK = 29

# --- ALGORITHM SETTINGS ---
MAX_DEPTH = 9  # The AI will try to reach this depth

# --- TRANSPOSITION TABLE (MEMORY) ---
# Key: Board Hash, Value: (score, flag, depth, best_move)
TRANSPOSITION_TABLE = {}

# --- PYGAME SETUP ---
SQUARESIZE = 100
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE
size = (width, height)
RADIUS = int(SQUARESIZE / 2 - 5)


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


def print_board(board):
    print(np.flip(board, 0))


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


# --- HEURISTIC EVALUATION ---
def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opp_piece = AI_PIECE

    if window.count(piece) == 4:
        score += W_WIN
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += W_THREE
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += W_TWO

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= W_BLOCK

    return score


def score_position(board, piece):
    score = 0
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * W_CENTER

    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

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


# --- ADVANCED AI ENGINE (MINIMAX + MEMORY) ---
def minimax(board, depth, alpha, beta, maximizingPlayer):
    # 1. TRANSPOSITION TABLE LOOKUP
    # Convert board to bytes to use as a dictionary key (fast hashing)
    board_key = board.tobytes()

    if board_key in TRANSPOSITION_TABLE:
        stored_score, stored_flag, stored_depth, stored_move = TRANSPOSITION_TABLE[board_key]
        # Only use stored result if it was searched at a depth >= current depth
        if stored_depth >= depth:
            if stored_flag == 'EXACT':
                return stored_move, stored_score
            elif stored_flag == 'LOWERBOUND':
                alpha = max(alpha, stored_score)
            elif stored_flag == 'UPPERBOUND':
                beta = min(beta, stored_score)

            if alpha >= beta:
                return stored_move, stored_score

    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, 1000000000)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -1000000000)
            else:
                return (None, 0)
        else:
            return (None, score_position(board, AI_PIECE))

    # 2. MOVE ORDERING
    # If we have a 'best move' from a previous shallow search in the table, try it first!
    if board_key in TRANSPOSITION_TABLE:
        best_prev_move = TRANSPOSITION_TABLE[board_key][3]
        if best_prev_move in valid_locations:
            valid_locations.insert(0, valid_locations.pop(valid_locations.index(best_prev_move)))
    else:
        # Simple heuristic ordering: try center columns first
        center = COLUMN_COUNT // 2
        valid_locations.sort(key=lambda x: abs(x - center))

    best_move = random.choice(valid_locations)

    if maximizingPlayer:
        value = -math.inf
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, False)[1]

            if new_score > value:
                value = new_score
                best_move = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    else:
        value = math.inf
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, True)[1]

            if new_score < value:
                value = new_score
                best_move = col
            beta = min(beta, value)
            if alpha >= beta:
                break

    # 3. STORE RESULT IN TABLE
    flag = 'EXACT'
    if value <= alpha:
        flag = 'UPPERBOUND'
    elif value >= beta:
        flag = 'LOWERBOUND'

    TRANSPOSITION_TABLE[board_key] = (value, flag, depth, best_move)
    return best_move, value


# --- ITERATIVE DEEPENING WRAPPER ---
def iterative_deepening(board, max_depth, time_limit):
    start_time = time.time()
    best_col = None

    # Iterate from Depth 1 up to MAX_DEPTH
    for d in range(1, max_depth + 1):
        # Check time limit (optional safety)
        if time.time() - start_time > time_limit:
            break

        # Run Minimax for this depth
        # Because we share the TRANSPOSITION_TABLE, this run is faster
        col, score = minimax(board, d, -math.inf, math.inf, True)

        best_col = col

        # If we found a forced win, stop searching!
        if score > 900000000:
            break

    return best_col


# --- GUI FUNCTIONS ---
def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (int(c * SQUARESIZE + SQUARESIZE / 2),
                                               int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == PLAYER_PIECE:
                pygame.draw.circle(screen, RED, (int(c * SQUARESIZE + SQUARESIZE / 2),
                                                 height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
            elif board[r][c] == AI_PIECE:
                pygame.draw.circle(screen, YELLOW, (int(c * SQUARESIZE + SQUARESIZE / 2),
                                                    height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    pygame.display.update()


# --- MAIN EXECUTION ---
board = create_board()
print_board(board)
game_over = False
turn = random.randint(PLAYER, AI)

pygame.init()
screen = pygame.display.set_mode(size)
draw_board(board)
pygame.display.update()
pygame.display.set_caption("Connect 4 - Iterative Deepening AI")

myfont = pygame.font.SysFont("monospace", 75)

while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEMOTION:
            pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
            posx = event.pos[0]
            if turn == PLAYER:
                pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)), RADIUS)
            pygame.display.update()

        if event.type == pygame.MOUSEBUTTONDOWN:
            pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
            if turn == PLAYER:
                posx = event.pos[0]
                col = int(math.floor(posx / SQUARESIZE))

                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, PLAYER_PIECE)

                    if winning_move(board, PLAYER_PIECE):
                        label = myfont.render("Player 1 Wins!!", 1, RED)
                        screen.blit(label, (40, 10))
                        game_over = True

                    turn += 1
                    turn = turn % 2
                    draw_board(board)

    # AI TURN
    if turn == AI and not game_over:
        # Call the new Iterative Deepening function
        # It will try to reach MAX_DEPTH but stop if it takes > 2 seconds
        col = iterative_deepening(board, max_depth=MAX_DEPTH, time_limit=5.0)

        if is_valid_location(board, col):
            pygame.time.wait(200)  # Small visual delay
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, AI_PIECE)

            if winning_move(board, AI_PIECE):
                label = myfont.render("AI Wins!!", 1, YELLOW)
                screen.blit(label, (40, 10))
                game_over = True

            draw_board(board)
            turn += 1
            turn = turn % 2

    if game_over:
        pygame.time.wait(5000)