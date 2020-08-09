import chess
import chess.pgn
import numpy as np
import tflite_runtime
import tflite_runtime.interpreter as tflite

class BoardState:
    def __init__(self):
        self.board = chess.Board()
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
        self.interpreter = tflite.Interpreter(model_path="/zugzwang/model/model.tflite")
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        #print(input_details[0]['dtype'])
        self.count = self.input_details[0]['shape'][0] # Only 1 image to be input
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.depth = self.input_details[0]['shape'][3]
        #print("Expected input count  = " + str(count))
        #print("Expected input height = " + str(height))
        #print("Expected input width  = " + str(width))
        #print("Expected input depth  = " + str(depth))

        #print(output_details[0]['dtype'])
        self.rows = self.output_details[0]['shape'][0]
        self.cols = self.output_details[0]['shape'][1]
        #print("Expected output rows = " + str(rows))
        #print("Expected output cols  = " + str(cols))

    def setCurrentState(self, current_state):
        self.current_state = current_state

    def setPrevState(self, prev_state):
        self.prev_state = prev_state

    def getBoardStateDiff(self):
        state_diff = self.current_state - self.prev_state
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
        if self.isFirstMove:
            self.node = self.game.add_variation(move)
            self.isFirstMove = False
        else:
            self.node = self.node.add_variation(move)
        self.board.push(move)

    def isWhitesTurn(self):
        return self.board.turn == chess.WHITE

    def getStateFromImage(self, raw_sample_frame):
        temp = np.array([[ 0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0],
                        [ 0,  0,  0,  0,  0,  0,  0,  0]])

        # Size (HxWxD) = (50x50x3)
        cv_img_400x400 = raw_sample_frame[40:440, 120:520]

        # Split current frame to 8x8 individual squares
        for i in range(0, 351, 50):
            for j in range(0, 351, 50):
                # Add N dimension to be (NxHxWxD) = (1x50x50x3)
                one_square = np.expand_dims(cv_img_400x400[i:i+50, j:j+50, :], axis=0)

                # Infer piece color in each square by using TF Lite model (start from Rank A to Rank H)
                self.interpreter.set_tensor(self.input_details[0]['index'], one_square)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                results = np.round(np.squeeze(output_data))

                # Map neural network softmax output to corresponding class
                if results[0] == 1:
                    temp[i//50, j//50] = -1 # black
                elif results[1] == 1:
                    temp[i//50, j//50] = 0  # empty
                elif results[2] == 1:
                    temp[i//50, j//50] = 1  # white

        # Rotate 90-degrees clockwise thrice if White is on right side
        # TODO: # Rotate 90-degrees clockwise once if White is on left side
        temp = np.rot90(m=temp, k=3)

        # Flip in up-down direction to match initial_state
        current_state = np.flipud(temp)

        return current_state