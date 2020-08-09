import numpy as np
import cv2

from modules import boardstate

print("zugzwang v0.01")
#print("numpy.__version__          = " + np.__version__)
#print("cv2.__version__            = " + cv2.__version__)
#print("tflite_runtime.__version__ = " + tflite_runtime.__version__)

if __name__ == "__main__":
    boardState = boardstate.BoardState()

    videoCapture = cv2.VideoCapture(0)

    print("Press 'esc' to quit. Press ' ' to make a move.")
    print("Chess Game Record:")
    move_num = 1
    while(True):
        # Capture frame-by-frame
        ret, frame = videoCapture.read()
        raw_sample_frame = np.float32(frame.copy())

        # Overlay a red 400x400 square to match with real-world Chess board dimension
        frame_overlay = cv2.rectangle(frame, (120, 40), (520 - 1, 440 - 1), (0, 0, 255), 1)

        # Display the resulting frame
        cv2.imshow('Zugzwang', frame_overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: # Esc key
            break
        elif key == ord(' '): # Space bar
            current_state = boardState.getStateFromImage(raw_sample_frame)

            # Calculate performed move
            boardState.setCurrentState(current_state)
            state_diff = boardState.getBoardStateDiff()
            start_square = boardState.getStartSquare(state_diff)
            dest_square = boardState.getDestSquare(state_diff)
            (move, uci_move, san_move) = boardState.convertToChessMove(start_square, dest_square)

            # Print move in SAN notation
            if (boardState.isWhitesTurn()):
                print("{:3}.   {:7}".format(move_num, san_move), end='', flush=True)
            else:
                print("{:7}".format(san_move))
                move_num = move_num + 1

            # Update BoardState
            boardState.executeMove(move)

            # Update previous state
            boardState.setPrevState(current_state)

    # Turn off camera capture when game is over
    print("\n")
    videoCapture.release()
    cv2.destroyAllWindows()