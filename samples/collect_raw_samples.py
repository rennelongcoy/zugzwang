# TODO: Refactor to classes

import numpy as np
import cv2
import os

cap = cv2.VideoCapture(0)

# Location of samples
samples_root_dir = "/zugzwang/samples/"
raw_dataset_dir  = samples_root_dir + "acquired_samples/" # Contains uncategorized samples

try:
    os.mkdir(raw_dataset_dir)
    os.chmod(raw_dataset_dir, 0o777)
except OSError as error:
    print(error)

sample_num = 1

print("Press 'esc' to quit. Press ' ' to capture data.")
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    raw_sample_frame = frame.copy()

    # Overlay a red 400x400 square to match with real-world Chess board dimension
    frame_overlay = cv2.rectangle(frame, (121, 41), (520, 440), (0, 0, 255), 1)

    # Display the resulting frame
    cv2.imshow('Data Gathering', frame_overlay)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: # Esc key
        break
    elif key == ord(' '): # Space bar
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
        print("Acquired sample stored at: " + raw_sample_dir)
        print("Press 'esc' to quit. Press ' ' to capture data.")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()