import chess
import chess.pgn
import numpy as np

from .tflite_model import TfLiteModel

class BoardState:
    def __init__(self, board):
        self.board = board
        self.game = chess.pgn.Game()
        self.isFirstMove = True
        self.node = 0
        self.prev_state = np.array([[ 1,  1,  1,  1,  1,  1,  1,  1],  # Rank A
                                    [ 1,  1,  1,  1,  1,  1,  1,  1],  # Rank B
                                    [ 0,  0,  0,  0,  0,  0,  0,  0],  # Rank C
                                    [ 0,  0,  0,  0,  0,  0,  0,  0],  # Rank D
                                    [ 0,  0,  0,  0,  0,  0,  0,  0],  # Rank E
                                    [ 0,  0,  0,  0,  0,  0,  0,  0],  # Rank F
                                    [-1, -1, -1, -1, -1, -1, -1, -1],  # Rank G
                                    [-1, -1, -1, -1, -1, -1, -1, -1]]) # Rank H
        self.move_num = 1
        self.current_state = self.prev_state
        self.tflite_model = TfLiteModel()

    def getBoardStateDiff(self, raw_frame):
        self.current_state = self.getBoardStateFromImage(raw_frame)
        state_diff = self.current_state - self.prev_state
        if self.board.turn == chess.BLACK:
            state_diff = -state_diff
        return state_diff

    def update(self, move):
        if self.isFirstMove:
            self.node = self.game.add_variation(move)
            self.isFirstMove = False
        else:
            self.node = self.node.add_variation(move)
        self.board.push(move)
        self.prev_state = self.current_state

    def printMove(self, san_move):
        if (self.board.turn == chess.WHITE):
            print("{:3}.   {:7}".format(self.move_num, san_move), end='', flush=True)
        else:
            print("{:7}".format(san_move))
            self.move_num = self.move_num + 1

    def getBoardStateFromImage(self, raw_frame):
        state_temp = np.array([[ 0,  0,  0,  0,  0,  0,  0,  0],
                               [ 0,  0,  0,  0,  0,  0,  0,  0],
                               [ 0,  0,  0,  0,  0,  0,  0,  0],
                               [ 0,  0,  0,  0,  0,  0,  0,  0],
                               [ 0,  0,  0,  0,  0,  0,  0,  0],
                               [ 0,  0,  0,  0,  0,  0,  0,  0],
                               [ 0,  0,  0,  0,  0,  0,  0,  0],
                               [ 0,  0,  0,  0,  0,  0,  0,  0]])

        # Size (HxWxD) = (400x400x3)
        img400x400 = raw_frame[40:440, 120:520]

        # Split current frame to 64 individual squares
        # Infer piece color in each square by using TF Lite model
        for i in range(0, 351, 50):
            for j in range(0, 351, 50):
                # Add N dimension to be (NxHxWxD) = (1x50x50x3)
                one_square = np.expand_dims(img400x400[i:i+50, j:j+50, :], axis=0)

                # Classify piece color in the individual square
                state_temp[i//50, j//50] = self.tflite_model.classifySquare(one_square)

        # Rotate 90-degrees clockwise thrice if White is on right side
        # TODO: # Rotate 90-degrees clockwise once if White is on left side
        state_temp = np.rot90(m=state_temp, k=3)

        # Flip in up-down direction to match initial_state
        current_state = np.flipud(state_temp)
        print(current_state)
        return current_state