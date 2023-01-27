import numpy as np
import cv2
import pyautogui
import config

#camera
print('[INFO] starting video stream...')
vs = cv2.VideoCapture(config.camera)

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

screensize = pyautogui.size()

w = screensize[0] - 200
h = screensize[1]

channels = 3

#initialise blank frame same size as screen
img = np.zeros((h,w,channels), dtype=np.uint8)


start = cv2.VideoCapture('arrows/start.mp4')
end = cv2.VideoCapture('arrows/end.mp4')


while True:
    ret, startframe = start.read()
    
    if ret:
        startframe = cv2.resize(startframe, (800,800))
        img[int(h/2-400):int(h/2+400), int(w/2-400):int(w/2+400)] = startframe

        #resize videostream
        ret, video = vs.read() 
        video = cv2.resize(video, (w,h))

        # Detect the faces
        gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4) 

        if faces != (): #if a face is present in frame
            ret, endframe = end.read()

            if ret:
                endframe = cv2.resize(endframe, (800,800))
                img[int(h/2-400):int(h/2+400), int(w/2-400):int(w/2+400)] = endframe #overlay end animation
            
                video = cv2.addWeighted(img, 0.5, video, 1.0, 0)
                cv2.imshow("Say Hello", video)

            else:
                exec(open("arucodetector.py").read()) # run aruco detection 

            
        video = cv2.addWeighted(img, 0.5, video, 1.0, 0) #overlay start animation
        cv2.imshow("Say Hello", video)     

    
    else:
        start.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


start.release()
cv2.destroyAllWindows()