import numpy as np
import sys

# --- CONFIGURATION ---
ROW_COUNT = 6
COLUMN_COUNT = 7


class Connect4Env:
    def __init__(self):
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT))
        self.game_over = False
        self.turn = 0  # 0 = Player 1, 1 = Player 2

    def drop_piece(self, row, col, piece):
        self.board[row][col] = piece

    def is_valid_location(self, col):
        # Returns True if the top row of the column is empty
        return self.board[ROW_COUNT - 1][col] == 0

    def get_next_open_row(self, col):
        # Gravity logic: Find the lowest empty slot
        for r in range(ROW_COUNT):
            if self.board[r][col] == 0:
                return r

    def print_board(self):
        # Flip the board so row 0 is at the bottom
        print(np.flip(self.board, 0))
        print(" 0  1  2  3  4  5  6 ")  # Column guides
        print("---------------------")

    def winning_move(self, piece):
        # Check horizontal locations
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT):
                if self.board[r][c] == piece and self.board[r][c + 1] == piece and self.board[r][c + 2] == piece and \
                        self.board[r][c + 3] == piece:
                    return True

        # Check vertical locations
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT - 3):
                if self.board[r][c] == piece and self.board[r + 1][c] == piece and self.board[r + 2][c] == piece and \
                        self.board[r + 3][c] == piece:
                    return True

        # Check positively sloped diagonals
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT - 3):
                if self.board[r][c] == piece and self.board[r + 1][c + 1] == piece and self.board[r + 2][
                    c + 2] == piece and self.board[r + 3][c + 3] == piece:
                    return True

        # Check negatively sloped diagonals
        for c in range(COLUMN_COUNT - 3):
            for r in range(3, ROW_COUNT):
                if self.board[r][c] == piece and self.board[r - 1][c + 1] == piece and self.board[r - 2][
                    c + 2] == piece and self.board[r - 3][c + 3] == piece:
                    return True
        return False


# --- MAIN GAME LOOP ---
if __name__ == "__main__":
    env = Connect4Env()
    env.print_board()

    while not env.game_over:
        # Ask for Player Input
        current_player = (env.turn % 2) + 1  # converts 0/1 to Player 1/2

        try:
            selection = input(f"Player {current_player} turn (0-6): ")

            # Allow user to quit easily
            if selection.lower() == 'q':
                sys.exit()

            col = int(selection)

            # Validation Check
            if col < 0 or col > 6:
                print("Selection must be between 0 and 6.")
                continue

            if env.is_valid_location(col):
                row = env.get_next_open_row(col)
                env.drop_piece(row, col, current_player)

                if env.winning_move(current_player):
                    env.print_board()
                    print(f"PLAYER {current_player} WINS!!")
                    env.game_over = True
                else:
                    env.print_board()
                    env.turn += 1
            else:
                print("Column full! Choose another.")

        except ValueError:
            print("Please enter a valid integer.")