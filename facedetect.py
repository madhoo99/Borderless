import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if faces != ():
        print(True)
    else:
        print(False)
    
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.circle(img, (x,y), 4, (255,0,255), -1)

    # Display
    cv2.imshow('img', img)
    
    key = cv2.waitKey(1) & 0xFF
    #if the 'q' key was pressed, break from the loop
    if key == ord('q'):
        break

#cleanup
cap.release()
cv2.destroyAllWindows()
    