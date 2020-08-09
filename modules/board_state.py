import chess
import chess.pgn
import numpy as np

import modules.constants as const
import modules.tflite_model as tflite_model

class BoardState:
    def __init__(self, board):
        self.board = board
        self.game = chess.pgn.Game()
        self.tflite_model = tflite_model.TfLiteModel()
        self.prev_state = np.array(const.INITIAL_BOARD_STATE)
        self.current_state = np.array(const.INITIAL_BOARD_STATE)
        self.move_count = 1

    def getBoardStateDiff(self, raw_frame):
        self.current_state = self.getBoardStateFromImage(raw_frame)
        state_diff = self.current_state - self.prev_state
        if self.board.turn == chess.BLACK:
            state_diff = -state_diff
        return state_diff

    def getBoardStateFromImage(self, raw_frame):
        state_temp = np.array(const.ZERO_BOARD_STATE)

        # Size (HxWxD) = (400x400x3)
        img400x400 = raw_frame[const.ROI_TOP_SIDE:const.ROI_BOTTOM_SIDE, const.ROI_LEFT_SIDE:const.ROI_RIGHT_SIDE]

        # Split current frame to 64 individual squares
        # Infer piece color in each square by using TF Lite model
        for i in range(0, const.EIGHTH_RANK, const.ONE_SQUARE_SIZE):
            for j in range(0, const.EIGHTH_FILE, const.ONE_SQUARE_SIZE):
                # Add N dimension to be (NxHxWxD) = (1x50x50x3)
                one_square = np.expand_dims(img400x400[i:i+const.ONE_SQUARE_SIZE, j:j+const.ONE_SQUARE_SIZE, :], axis=0)

                # Classify piece color in the individual square
                state_temp[i//const.ONE_SQUARE_SIZE, j//const.ONE_SQUARE_SIZE] = self.tflite_model.classifySquare(one_square)

        # Rotate 90-degrees clockwise thrice if White is on right side
        # TODO: # Rotate 90-degrees clockwise once if White is on left side
        state_temp = np.rot90(m=state_temp, k=3)

        # Flip in up-down direction to match initial_state
        current_state = np.flipud(state_temp)

        return current_state

    def printMove(self, san_move):
        if (self.board.turn == chess.WHITE):
            print("{:3}.   {:7}".format(self.move_count, san_move), end='', flush=True)
        else:
            print("{:7}".format(san_move))
            self.move_count = self.move_count + 1

    def update(self, move):
        if self.move_count == 1 and self.board.turn == chess.WHITE:
            self.node = self.game.add_variation(move)
        else:
            self.node = self.node.add_variation(move)
        self.board.push(move)
        self.prev_state = self.current_state