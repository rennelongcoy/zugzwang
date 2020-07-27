import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv_img = cv2.rectangle(frame, (121, 41), (520, 440), (0, 0, 255), 1)

    # Display the resulting frame
    #cv2.imshow('frame',gray)
    cv2.imshow('frame',cv_img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        print(cv_img[40:440, 120:520].shape)
        print("Space bar pressed.")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()