# TODO: Refactor to classes

import numpy as np
import cv2
import os

cap = cv2.VideoCapture(0)

# Important directories
model_root_dir    = "/zugzwang/SS_DataAcquisition/"
raw_dataset_dir   = model_root_dir + "01_raw_samples/"  # Contains uncategorized samples
#training_data_dir = model_root_dir + "02_training_data/" # Contains categorized sample split to training and validation sets
#model_files_dir   = model_root_dir + "03_model_files/"   # Contains fitted Keras models and matching TF Lite models

try:
    os.mkdir(raw_dataset_dir)
    os.chmod(raw_dataset_dir, 0o777)
except OSError as error:
    print(error)

'''try:
    os.mkdir(training_data_dir)
    os.chmod(training_data_dir, 0o777)
except OSError as error:
    print(error)

try:
    os.mkdir(model_files_dir)
    os.chmod(model_files_dir, 0o777)
except OSError as error:
    print(error)'''

#dataset_num = 1 # TODO: make as commandline argument. Specify to which dataset to add frame to.
#dataset_dir = raw_dataset_dir + str(dataset_num) + "/"

sample_num = 1

print("Press 'q' to quit. Press ' ' to capture data.")
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
        #print(frame[40:440, 120:520].shape)
        #print("Space bar pressed.")
        # Take sample data No. xx
        raw_sample_name = str(sample_num) + "-original.jpg"
        #raw_sample_dir = dataset_dir + str(sample_num) + "/"
        raw_sample_dir = raw_dataset_dir + str(sample_num) + "/"
        '''try:
            os.mkdir(dataset_dir)
            os.chmod(dataset_dir, 0o777)
        except OSError as error:
            print(error)'''
        try:
            os.mkdir(raw_sample_dir)
            os.chmod(raw_sample_dir, 0o777)
        except OSError as error:
            print(error)
        raw_sample_file = raw_sample_dir + raw_sample_name
        raw_overlay_file = raw_sample_dir + str(sample_num) + "-with-overlay.jpg"
        #print("dataset_dir     = " + dataset_dir)
        #print("raw_sample_dir  = " + raw_sample_dir)
        #print("raw_sample_file = " + raw_sample_file)
        #print("raw_overlay_file = " + raw_overlay_file)

        # Store the original image
        cv2.imwrite(raw_sample_file, raw_sample_frame)

        # Store the original image with red overlay
        cv2.imwrite(raw_overlay_file, frame_overlay)

        cv_img_400x400 = raw_sample_frame[40:440, 120:520]
        for i in range(0, 351, 50):
            for j in range(0, 351, 50):
                  #print('cv_img_400x400[' + str(i) + ':' + str(i+50) + ', ' + str(j) + ':' + str(j+50) + ']')
                  cv_save = cv_img_400x400[i:i+50, j:j+50, :]
                  cv2.imwrite(raw_sample_dir + str(sample_num) + '-' + str(round(i/50)) + '-' + str(round(j/50)) + '.jpg', cv_save)
        sample_num = sample_num + 1
        print("Press 'q' to quit. Press ' ' to capture data.")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()