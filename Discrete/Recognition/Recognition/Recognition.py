import custom_utils as utls
import cv2
import sys
from PIL import Image as im
import matplotlib.pyplot as plt


faceCascade = cv2.CascadeClassifier('{0}/Resources/haarcascade_frontalface_default.xml'.format(utls.DATA_DIR))

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=utls.IMG_SHAPE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

    if cv2.waitKey(1) == ord('c'):
        #basis, origin_values, origin_vectors, mid_image = utls.get_original_data()
        for face in faces:
            plt.imshow(face)
            plt.show()

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()