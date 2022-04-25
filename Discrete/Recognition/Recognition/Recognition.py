from custom_utils import PCA, DATA_DIR
import cv2


faceCascade = cv2.CascadeClassifier('{0}/Resources/haarcascade_frontalface_default.xml'.format(DATA_DIR))

video_capture = cv2.VideoCapture(0)

pca_model = PCA()

move_window = 30

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(300, 225)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-move_window), (x+w, y+h-move_window), (0, 255, 0), 2)
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

    if cv2.waitKey(1) == 13:
        for (x, y, w, h) in faces:
            screenshot = frame[y-move_window+2:y-move_window-2+h, x+2:x-2+w].copy()
            print('Screenshot was captured')
            cv2.imshow('Screenshot', screenshot)
            pca_model.recognise(screenshot)

video_capture.release()
cv2.destroyAllWindows()
