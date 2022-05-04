from custom_utils import PCA, DATA_DIR, IMG_SHAPE
import cv2


faceCascade = cv2.CascadeClassifier('{0}/Resources/haarcascade_frontalface_default.xml'.format(DATA_DIR))

video_capture = cv2.VideoCapture(0)

pca_model = PCA()

x_step = 0
y_step = 60
border_thikness = 2

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    frame = cv2.resize(frame, (1000, 800))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=IMG_SHAPE
    )

    for (x, y, w, h) in faces:
        y0 = y-y_step
        y1 = y+y_step+h
        cv2.rectangle(frame, (x, y0), (x+w, y1), (0, 255, 0), border_thikness)
        screenshot = frame[y0+border_thikness:y1-border_thikness, x+border_thikness:x-border_thikness+w].copy()
        pca_model.recognise(screenshot)
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break
            

video_capture.release()
cv2.destroyAllWindows()
