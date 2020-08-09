import chess
import numpy as np
import cv2

import modules.board_move as board_move
import modules.board_state as board_state

print("zugzwang v0.01")

if __name__ == "__main__":
    board = chess.Board()
    boardState = board_state.BoardState(board)
    moveHandler = board_move.MoveHandler(board)

    videoCapture = cv2.VideoCapture(0)

    print("Press 'esc' to quit. Press ' ' to make a move.")
    print("Chess Game Record:")
    while(True):
        # Capture frame-by-frame
        ret, frame = videoCapture.read()
        raw_frame = np.float32(frame.copy())

        # Overlay a red 400x400 square to match with real-world Chess board dimension
        frame_overlay = cv2.rectangle(frame, (120, 40), (520 - 1, 440 - 1), (0, 0, 255), 1)

        # Display the resulting frame
        cv2.imshow('Zugzwang v0.01', frame_overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: # Esc key
            break
        elif key == ord(' '): # Space bar
            # Calculate difference between current and previous board
            state_diff = boardState.getBoardStateDiff(raw_frame)

            # Convert state diff to move
            (move, uci_move, san_move) = moveHandler.getMoveFromStateDiff(state_diff)

            # Print move in SAN notation
            boardState.printMove(san_move)

            # Update BoardState internal record
            boardState.update(move)

    # Turn off camera capture when game is over
    print("\n")
    videoCapture.release()
    cv2.destroyAllWindows()