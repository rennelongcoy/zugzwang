import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Overlay red 400x400 square to match with real-world Chess board dimension
    frame_overlay = cv2.rectangle(frame, (121, 41), (520, 440), (0, 0, 255), 1)

    # Display the resulting frame
    cv2.imshow('frame', frame_overlay)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        print(frame[40:440, 120:520].shape)
        print("Space bar pressed.")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()