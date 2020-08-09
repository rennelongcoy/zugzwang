import chess
import chess.pgn
import numpy as np
import tflite_runtime
import tflite_runtime.interpreter as tflite

class TfLiteModel:
    def __init__(self):
        self.interpreter = tflite.Interpreter(model_path="/zugzwang/training/model.tflite")
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

    def classifySquare(self, one_square):
        # Infer piece color in provided Chess square
        self.interpreter.set_tensor(self.input_details[0]['index'], one_square)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        results = np.round(np.squeeze(output_data))

        # Map neural network softmax output to corresponding class
        if results[0] == 1:
            prediction = -1 # black
        elif results[1] == 1:
            prediction = 0  # empty
        elif results[2] == 1:
            prediction = 1  # white

        return prediction