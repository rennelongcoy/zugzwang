import numpy as np
import tflite_runtime.interpreter as tflite

class TfLiteModel:
    def __init__(self):
        self.interpreter = tflite.Interpreter(model_path="/zugzwang/training/model.tflite")
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(self.input_details[0]['dtype'])
        self.count = self.input_details[0]['shape'][0] # Only 1 image to be input
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.depth = self.input_details[0]['shape'][3]
        print("Expected input count  (N) = " + str(self.count))
        print("Expected input height (H) = " + str(self.height))
        print("Expected input width  (W) = " + str(self.width))
        print("Expected input depth  (D) = " + str(self.depth))

        print(self.output_details[0]['dtype'])
        self.rows = self.output_details[0]['shape'][0]
        self.cols = self.output_details[0]['shape'][1]
        print("Expected output rows = " + str(self.rows))
        print("Expected output cols = " + str(self.cols))

    def classifySquare(self, one_square):
        # Infer piece color in provided Chess square
        self.interpreter.set_tensor(self.input_details[0]['index'], one_square)
        self.interpreter.invoke()
        softmax_output = self.interpreter.get_tensor(self.output_details[0]['index'])
        softmax_output = np.round(np.squeeze(softmax_output))

        # Map neural network softmax output to corresponding class
        if softmax_output[0] == 1:
            prediction = -1 # black
        elif softmax_output[1] == 1:
            prediction = 0  # empty
        elif softmax_output[2] == 1:
            prediction = 1  # white

        return prediction