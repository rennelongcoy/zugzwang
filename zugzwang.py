import numpy as np
import cv2
import tflite_runtime
import tflite_runtime.interpreter as tflite

from modules import boardstate

if __name__ == "__main__":
    boardState = boardstate.BoardState()

    interpreter = tflite.Interpreter(model_path="model/model.tflite")
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    #print(input_details[0]['dtype'])
    count = input_details[0]['shape'][0] # Only 1 image to be input
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    depth = input_details[0]['shape'][3]
    #print("Expected input count  = " + str(count))
    #print("Expected input height = " + str(height))
    #print("Expected input width  = " + str(width))
    #print("Expected input depth  = " + str(depth))
    
    #print(output_details[0]['dtype'])
    rows = output_details[0]['shape'][0]
    cols = output_details[0]['shape'][1]
    #print("Expected output rows = " + str(rows))
    #print("Expected output cols  = " + str(cols))
    
    videoCapture = cv2.VideoCapture(0)
    
    #print("zugzwang v0.01")
    #print("chess.__version__          = " + chess.__version__)
    #print("numpy.__version__          = " + np.__version__)
    #print("cv2.__version__            = " + cv2.__version__)
    #print("tflite_runtime.__version__ = " + tflite_runtime.__version__)

    # Populate by row from A-rank (1st row) to H-rank
    initial_state = np.array([[ 1,  1,  1,  1,  1,  1,  1,  1],
                            [ 1,  1,  1,  1,  1,  1,  1,  1],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0],
                            [-1, -1, -1, -1, -1, -1, -1, -1],
                            [-1, -1, -1, -1, -1, -1, -1, -1]])

    prev_state = initial_state
    move_num = 1

    print("Press 'esc' to quit. Press ' ' to make a move.")
    print("Chess Game Record:")

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
                    interpreter.set_tensor(input_details[0]['index'], one_square)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])
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

            # Calculate performed move
            state_diff = boardState.getBoardStateDiff(current_state, prev_state)
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
            prev_state = current_state

    # Turn off camera capture when game is over
    print("\n")
    videoCapture.release()
    cv2.destroyAllWindows()