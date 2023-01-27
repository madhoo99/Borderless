import cv2
import numpy as np
import pyautogui


screensize = pyautogui.size()

w = screensize[0]
h = screensize[1]

channels = 3

#initialise blank frame same size as screen
img = np.zeros((h,w,channels), dtype=np.uint8)

cap = cv2.VideoCapture("arrows/end.mp4")
vs = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # frame = cv2.resize(frame, (800,800))
    # img[int(h/2-400):int(h/2+400), int(w/2-400):int(w/2+400)] = frame

    if ret:
        frame = cv2.resize(frame, (800,800))
        img[int(h/2-400):int(h/2+400), int(w/2-400):int(w/2+400)] = frame
        cv2.imshow("image",  frame)
    
    else:
        break

    # ret, video = vs.read()
    # video = cv2.resize(video, screensize)

    # video = cv2.addWeighted(img, 0.5, video, 1.0, 0)
    # cv2.imshow("Say Hello", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
