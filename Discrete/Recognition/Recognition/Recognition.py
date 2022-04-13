import custom_utils as utls
import cv2


faceCascade = cv2.CascadeClassifier('{0}/Resources/haarcascade_frontalface_default.xml'.format(utls.DATA_DIR))

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(100, 75)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

    if cv2.waitKey(1) == 13:
        for (x, y, w, h) in faces:
            screenshot = frame[y+1:y+1+h, x+1:x+1+w].copy()
            print('Screenshot was captured')
            cv2.imshow('Screenshot', screenshot)
            utls.recognise(screenshot)

video_capture.release()
cv2.destroyAllWindows()
