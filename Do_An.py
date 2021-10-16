import os

import cv2
import numpy as np
from keras.models import load_model
from pygame import mixer

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl = ["yawn", "no_yawn", "Closed", "Open"]
IMG_SIZE = 224


def prepare(filepath, face_cas=face):
    # img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img_array = filepath / 255
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


model = load_model('drowiness_new6 (2).h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]

while (True):
    ret, frame = cap.read(cv2.IMREAD_COLOR)
    height, width = frame.shape[:2]
    faces = face.detectMultiScale(frame, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(frame)
    right_eye = reye.detectMultiScale(frame)
    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)
    prediction = model.predict([prepare(frame)])
    f_pre = np.argmax(prediction)
    print(f_pre)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        prediction = model.predict([prepare(r_eye)])
        r_pre = np.argmax(prediction)
        if (r_pre == 2):
            lbl = 'Closed'
        if (r_pre == 3):
            lbl = 'open'
        break
    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        prediction = model.predict([prepare(l_eye)])
        l_pre = np.argmax(prediction)
        if (l_pre == 2):
            lbl = 'Closed'
        if (l_pre == 3):
            lbl = 'open'
        break
    if ((r_pre == 2 and l_pre == 2 or f_pre==0)or(r_pre == 2 and l_pre == 2 and f_pre==0)):
        score = score + 1
        cv2.putText(frame, "Drowsy", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score = score - 1
        cv2.putText(frame, "No_drowsy", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if (score < 0):
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (200, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if (score > 15):
        # person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()

        except:  # isplaying = False
            pass
        if (thicc < 16):
            thicc = thicc + 2
        else:
            thicc = thicc - 2
            if (thicc < 2):
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
