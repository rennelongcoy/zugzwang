import chess
import chess.pgn
import numpy as np

class BoardState:
    def __init__(self):
        self.board = chess.Board()
        self.game = chess.pgn.Game()
        self.isStartOfGame = True
        self.node = 0

    def getBoardStateDiff(self, current_state, prev_state):
        state_diff = current_state - prev_state
        if self.board.turn == chess.BLACK:
            state_diff = -state_diff
        return state_diff

    def getStartSquare(self, state_diff):
        start_point = np.unravel_index(state_diff.argmin(), state_diff.shape)
        # Notation: (rank, file)
        # print("start_point = " + str(start_point))
        start_square = chess.square(start_point[1], start_point[0])
        # print("start_square = " + str(chess.square_name(start_square)))
        return start_square

    def getDestSquare(self, state_diff):
        dest_point = np.unravel_index(state_diff.argmax(), state_diff.shape)
        # Notation: (rank, file)
        # print("dest_point = " + str(dest_point))
        dest_square = chess.square(dest_point[1], dest_point[0])
        # print("dest_square = " + str(chess.square_name(dest_square)))
        return dest_square

    def convertToChessMove(self, start_square, dest_square):
        move = chess.Move(start_square, dest_square)
        uci_move = chess.Move.uci(move)
        # print("UCI move = " + uci_move)
        san_move = self.board.san(move)
        # print("SAN move = " + san_move)
        return (move, uci_move, san_move)

    def executeMove(self, move):
        if self.isStartOfGame:
            self.node = self.game.add_variation(move)
            self.isStartOfGame = False
        else:
            self.node = self.node.add_variation(move)
        self.board.push(move)

    def isWhitesTurn(self):
        return self.board.turn == chess.WHITE