# Python setup
# Turn on Camera
# Empty board calibration (find corners)
# Start game (capture photo)
# Infer piece positions per row
# Compare board states
# Determine the move made
# Record the move
# Proceed with next turn until game over

import chess
import chess.pgn
import chess.svg
import numpy as np
import cairosvg
import cv2
import io
import PIL
import tflite_runtime
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details[0]['dtype'])
count = input_details[0]['shape'][0] # Only 1 image to be input
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
depth = input_details[0]['shape'][3]
print("Expected input count  = " + str(count))
print("Expected input height = " + str(height))
print("Expected input width  = " + str(width))
print("Expected input depth  = " + str(depth))

print(output_details[0]['dtype'])
rows = output_details[0]['shape'][0]
cols = output_details[0]['shape'][1]
print("Expected output rows = " + str(rows))
print("Expected output cols  = " + str(cols))

cap = cv2.VideoCapture(0)

print("zugzwang v0.01")
print("chess.__version__          = " + chess.__version__)
print("numpy.__version__          = " + np.__version__)
print("cairosvg.__version__       = " + cairosvg.__version__)
print("cv2.__version__            = " + cv2.__version__)
print("PIL.__version__            = " + PIL.__version__)
print("tflite_runtime.__version__ = " + tflite_runtime.__version__)

# ChessGame class
board = chess.Board()
game = chess.pgn.Game()
game.headers["Event"] = "Event"
game.headers["Site"] = "Site"
game.headers["Date"] = "Date"
game.headers["Round"] = "Round #"
game.headers["White"] = "White"
game.headers["Black"] = "Black"

# Global variables
isStartOfGame = True
node = 0

# Populate by row from A-rank (1st row) to H-rank
initial_state = np.array([[ 1,  1,  1,  1,  1,  1,  1,  1],
                          [ 1,  1,  1,  1,  1,  1,  1,  1],
                          [ 0,  0,  0,  0,  0,  0,  0,  0],
                          [ 0,  0,  0,  0,  0,  0,  0,  0],
                          [ 0,  0,  0,  0,  0,  0,  0,  0],
                          [ 0,  0,  0,  0,  0,  0,  0,  0],
                          [-1, -1, -1, -1, -1, -1, -1, -1],
                          [-1, -1, -1, -1, -1, -1, -1, -1]])

# Should be from BoardState class
def getBoardStateDiff(current_state, prev_state):
  global board
  state_diff = current_state - prev_state
  if board.turn == chess.BLACK:
    state_diff = -state_diff
  return state_diff

def getStartSquare(state_diff):
  start_point = np.unravel_index(state_diff.argmin(), state_diff.shape)
  #print("start_point = " + str(start_point)) # Notation: (rank, file)
  start_square = chess.square(start_point[1], start_point[0])
  print("start_square = " + str(chess.square_name(start_square)))
  return start_square

def getDestSquare(state_diff):
  dest_point = np.unravel_index(state_diff.argmax(), state_diff.shape)
  #print("dest_point = " + str(dest_point)) # Notation: (rank, file)
  dest_square = chess.square(dest_point[1], dest_point[0])
  print("dest_square = " + str(chess.square_name(dest_square)))
  return dest_square

def convertToChessMove(start_square, dest_square):
  move = chess.Move(start_square, dest_square)
  uci_move = chess.Move.uci(move)
  print("UCI move = " + uci_move)
  san_move = board.san(move)
  print("SAN move = " + san_move)
  return move

def executeMove(move):
  global isStartOfGame
  global node
  if isStartOfGame:
    node = game.add_variation(move)
    isStartOfGame = False
  else:
    node = node.add_variation(move)
  board.push(move)

# White's turn
#print("Turn = " + ("White" if board.turn else "Black"))

# Move 1
# Sample move of White, populate by row from White side as 1st row
# Below is d2d4 move
# Should be from BoardState class
# TODO: Infer from video frame by using TF Lite model
'''prev_state = initial_state
current_state = np.array([[ 1,  1,  1,  1,  1,  1,  1,  1],
                          [ 1,  1,  1,  0,  1,  1,  1,  1],
                          [ 0,  0,  0,  0,  0,  0,  0,  0],
                          [ 0,  0,  0,  1,  0,  0,  0,  0],
                          [ 0,  0,  0,  0,  0,  0,  0,  0],
                          [ 0,  0,  0,  0,  0,  0,  0,  0],
                          [-1, -1, -1, -1, -1, -1, -1, -1],
                          [-1, -1, -1, -1, -1, -1, -1, -1]])

state_diff = getBoardStateDiff(current_state, prev_state)
start_square = getStartSquare(state_diff)
dest_square = getDestSquare(state_diff)
move = convertToChessMove(start_square, dest_square)
executeMove(move)
print("Turn = " + ("White" if board.turn else "Black"))'''

# Move 2
# Sample move of White, populate by row from White side as 1st row
# Below is Nc6 move
# TODO: Infer from video frame by using TF Lite model
'''prev_state = current_state
current_state = np.array([[ 1,  1,  1,  1,  1,  1,  1,  1],
                          [ 1,  1,  1,  0,  1,  1,  1,  1],
                          [ 0,  0,  0,  0,  0,  0,  0,  0],
                          [ 0,  0,  0,  1,  0,  0,  0,  0],
                          [ 0,  0,  0,  0,  0,  0,  0,  0],
                          [ 0,  0, -1,  0,  0,  0,  0,  0],
                          [-1, -1, -1, -1, -1, -1, -1, -1],
                          [-1,  0, -1, -1, -1, -1, -1, -1]])

state_diff = getBoardStateDiff(current_state, prev_state)
start_square = getStartSquare(state_diff)
dest_square = getDestSquare(state_diff)
move = convertToChessMove(start_square, dest_square)
executeMove(move)
print("Turn = " + ("White" if board.turn else "Black"))'''

print("Turn = " + ("White" if board.turn else "Black"))
print("Press 'esc' to quit. Press ' ' to make a move.")
prev_state = initial_state

temp = np.array([[ 0,  0,  0,  0,  0,  0,  0,  0],
                          [ 0,  0,  0,  0,  0,  0,  0,  0],
                          [ 0,  0,  0,  0,  0,  0,  0,  0],
                          [ 0,  0,  0,  0,  0,  0,  0,  0],
                          [ 0,  0,  0,  0,  0,  0,  0,  0],
                          [ 0,  0,  0,  0,  0,  0,  0,  0],
                          [ 0,  0,  0,  0,  0,  0,  0,  0],
                          [ 0,  0,  0,  0,  0,  0,  0,  0]])

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    raw_sample_frame = frame.copy().astype(np.float32)
    #print(type(raw_sample_frame))

    # Overlay a red 400x400 square to match with real-world Chess board dimension
    frame_overlay = cv2.rectangle(frame, (120, 40), (520 - 1, 440 - 1), (0, 0, 255), 1)

    # Display the resulting frame
    cv2.imshow('Zugzwang', frame_overlay)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord(' '):
        # Size (HxWxD) = (50x50x3)
        cv_img_400x400 = raw_sample_frame[40:440, 120:520]
        # Split current frame to 8x8 individual squares
        for i in range(0, 351, 50):
            for j in range(0, 351, 50):
                # Add N dim to be (NxHxWxD) = (1x50x50x3)
                one_square = np.expand_dims(cv_img_400x400[i:i+50, j:j+50, :], axis=0)

                # Infer piece color in each square by using TF Lite model (start from Rank A to Rank H)
                interpreter.set_tensor(input_details[0]['index'], one_square)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                results = np.round(np.squeeze(output_data))
                #print(results)
                if results[0] == 1:
                    temp[i//50, j//50] = -1 # black
                elif results[1] == 1:
                    temp[i//50, j//50] = 0  # empty
                elif results[2] == 1:
                    temp[i//50, j//50] = 1  # white
        print(temp)
        # Rotate 90-degrees clockwise thrice
        temp = np.rot90(m=temp, k=3)
        print(temp)
        # Flip in up-down direction to match initial_state
        current_state = np.flipud(temp)
        print(current_state)

        state_diff = getBoardStateDiff(current_state, prev_state)
        start_square = getStartSquare(state_diff)
        dest_square = getDestSquare(state_diff)
        move = convertToChessMove(start_square, dest_square)
        executeMove(move)
        prev_state = current_state

        # Below is d2d4 move
        '''current_state = np.array([[ 1,  1,  1,  1,  1,  1,  1,  1],  # Rank A
                                  [ 1,  1,  1,  0,  1,  1,  1,  1],  # Rank B
                                  [ 0,  0,  0,  0,  0,  0,  0,  0],  # Rank C
                                  [ 0,  0,  0,  1,  0,  0,  0,  0],  # Rank D
                                  [ 0,  0,  0,  0,  0,  0,  0,  0],  # Rank E
                                  [ 0,  0,  0,  0,  0,  0,  0,  0],  # Rank F
                                  [-1, -1, -1, -1, -1, -1, -1, -1],  # Rank G
                                  [-1, -1, -1, -1, -1, -1, -1, -1]]) # Rank H

        state_diff = getBoardStateDiff(current_state, prev_state)
        start_square = getStartSquare(state_diff)
        dest_square = getDestSquare(state_diff)
        move = convertToChessMove(start_square, dest_square)
        executeMove(move)
        prev_state = current_state'''
        print("Turn = " + ("White" if board.turn else "Black"))
        print("Press 'esc' to quit. Press ' ' to make a move.")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()