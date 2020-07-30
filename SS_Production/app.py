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

cap = cv2.VideoCapture(0)

print("zugzwang v0.01")
print("chess.__version__    = " + chess.__version__)
print("numpy.__version__    = " + np.__version__)
print("cairosvg.__version__ = " + cairosvg.__version__)
print("cv2.__version__      = " + cv2.__version__)
print("PIL.__version__      = " + PIL.__version__)

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
print("Turn = " + ("White" if board.turn else "Black"))

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
print("Press 'q' to quit. Press ' ' to make a move.")
prev_state = initial_state
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    raw_sample_frame = frame.copy()

    # Overlay a red 400x400 square to match with real-world Chess board dimension
    frame_overlay = cv2.rectangle(frame, (121, 41), (520, 440), (0, 0, 255), 1)

    # Display the resulting frame
    cv2.imshow('Data Gathering', frame_overlay)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        # TODO: Split current frame to 8x8 individual squares
        # TODO: Infer piece in each square by using TF Lite model (start from Rank A to Rank H)
        # Below is d2d4 move
        current_state = np.array([[ 1,  1,  1,  1,  1,  1,  1,  1],  # Rank A
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
        prev_state = current_state
        print("Turn = " + ("White" if board.turn else "Black"))
        print("Press 'q' to quit. Press ' ' to make a move.")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()