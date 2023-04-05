from datetime import datetime
from multiprocessing import Pool, Value, Process
import cv2
import numpy as np
from imutils.video import VideoStream
import argparse
import time
import sys
import pyautogui
import math

import pytz
import config
from dateutil.relativedelta import relativedelta
import keyboard

state = Value('i', 0)

def talker_thread(state):
    i = 0
    while i < 5:
        time.sleep(5)
        i += 1
        state.value = i

def aruco_thread(state):
    #construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", type=str,
                    default="DICT_4X4_50", #4x4 is the best
                    help="type of ArUco tag to detect")
    args = vars(ap.parse_args())

    #get screen res
    # size = pyautogui.size()
    # w = size[0] - 200
    # h = size[1]


    #define names of each possible ArUco tag OpenCV supports
    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50, #fastest, looking out for 50/1000
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }

    #verify that the supplied Aruco marker exists and is supported by OpenCV
    if ARUCO_DICT.get(args["type"], None) is None:
        print("[INFO] Aruco tag of '{}' is not supported".format(args["type"]))
        sys.exit(0)

    #load the Aruco Dictionary and grab the Aruco parameters
    print("[INFO] detecting '{}' tags...".format(args["type"]))
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
    arucoParams = cv2.aruco.DetectorParameters_create()

    #read and initialise overlay pngs
    img0 = cv2.imread("comments/comment0.png")
    img1 = cv2.imread("comments/comment1.png")
    img2 = cv2.imread("comments/comment2.png")
    img3 = cv2.imread("comments/comment3.png")

    #read and initialise emoji png
    emoji4 = cv2.imread("emoji/emoji4.png")
    emoji5 = cv2.imread("emoji/emoji5.png")
    emoji6 = cv2.imread("emoji/emoji6.png")


    #initialize the video stream and allow the camera sensor to warm up
    print('[INFO] starting video stream...')
    #vs = VideoStream(src=config.camera).start()
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    time.sleep(2.0) #time for camera to warm up

    #initialise arrow animations
    # startarrow = cv2.VideoCapture('arrows/start.mp4')
    # endarrow = cv2.VideoCapture('arrows/end.mp4')

    # Load the cascade
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    flag=1

    scale =  1.35 #1.55
    width = 200
    height = 800

    #loop over frames from video stream
    while True:
        #time.sleep(0.01)

        curr_time = datetime.now(pytz.utc)

        # if start_time + relativedelta(seconds=duration) > curr_time:
        if keyboard.is_pressed('p'):
            scale = float(input('Enter scale, current {}: '.format(str(scale))) or str(scale))
            width = int(input('Enter width trim start, current {}: '.format(str(width))) or str(width))
            height = int(input('Enter height trim start, current {}: '.format(str(height))) or str(height))
            # duration = int(input('Enter duration: ') or str(duration))
            
            # start_time = datetime.now(pytz.utc)

        #grab the frame from the threaded video stream and resize to screen resolution
        #frame = vs.read()
        # frame = cv2.resize(frame, (w,h))

        
        

        r, frame = cap.read()

        #print('Resolution: ' + str(frame.shape[0] + ' x ' + str(frame.shape[1])))
        
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        #frame = zoom(3, frame)

        w = frame.shape[1]
        h = frame.shape[0]
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        #print(frame.shape[1])
        print(state.value)

        # success with asus webcam
        # frame = frame[200:, :(frame.shape[1] - 1150)]

        frame = frame[width:, :(frame.shape[1] - height)]
        
        # print(w, h)


        #detect Aruco markers in the input frame
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters = arucoParams)

        # Detect the faces
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        #verify at least one aruco marker was detected

        if len(corners) > 0:

            #flatten the aruco IDs list
            ids = ids.flatten()

            #loop over the detected aruco corners
            for (markerCorner, markerID) in zip(corners, ids):

                #extraxt the marker corners (top-left, top-right, bottom-right, bottom-left order)
                corners = markerCorner.reshape((4,2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners

                #convert each of the (x-y) coordinate pairs to integers
                #so we can draw the coordinates using Open CV drawing function
                topRight    = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft  = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft     = (int(topLeft[0]), int(topLeft[1]))

                length = int(math.dist(topRight, bottomRight))
                
                #tag the comments and emojis to the specific markerIDs
                if markerID==0:
                    img = img0
                    flag = 1
                    imgl = 400
                elif markerID==1:
                    img = img1
                    flag = 1
                    imgl = 400
                elif markerID==2:
                    img = img2
                    flag = 1
                    imgl = 400
                elif markerID==3:
                    img = img3
                    flag = 1 
                    imgl = 400
                elif markerID==4:
                    img = emoji4
                    flag = 1
                    imgl = 200
                elif markerID==5:
                    img = emoji5
                    flag = 1
                    imgl = 200
                elif markerID==6:
                    img = emoji6
                    flag = 1
                    imgl = 200
                else:
                    flag = 0

                    #draw a box for other detected markers
                    cv2.line(frame, topLeft, topRight, (0,255,0), 2)
                    cv2.line(frame, topRight, bottomRight, (0,255,0), 2)
                    cv2.line(frame, bottomRight, bottomLeft, (0,255,0), 2)
                    cv2.line(frame, bottomLeft, topLeft, (0,255,0), 2)

                if flag == 1:

                    #set size of png
                    imgl2 = int(imgl/2)

                    img = cv2.resize(img, (imgl,imgl))
                    # img_corners = np.float32([[0,0],[imgl,0], [imgl,imgl], [0,imgl]])

                    #coordinates of a square twice the size of marker
                    topRight1    = (int(topRight[0]) + int(length), int(topRight[1]) - int(length))
                    bottomRight1 = (int(bottomRight[0]) + int(length), int(bottomRight[1]) + int(length))
                    bottomLeft1  = (int(bottomLeft[0]) - int(length), int(bottomLeft[1]) + int(length))
                    topLeft1     = (int(topLeft[0]) - int(length) , int(topLeft[1]) - int(length))
                    
                    aruco_corners = np.array([topLeft1, topRight1, bottomRight1, bottomLeft1])

                    # center (x,y) coordinates of the aruco marker 
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)

                #draw the aruco marker ID on the frame
                # cv2.putText(frame, "Hello World",
                #            (cX,cY), #(topLeft[0], topLeft[1]-15),
                #            cv2.FONT_HERSHEY_SIMPLEX,
                #            0.5, (0,255,0), 2)

                # calculate homography and warp perspective
                # matrix, status = cv2.findHomography(img_corners, aruco_corners)
                # frame1 = cv2.warpPerspective(img, matrix, (frame.shape[1], frame.shape[0]))

                # frame = cv2.fillConvexPoly(frame, aruco_corners, (255,255,255))
                # frame += frame1
                # frame = cv2.addWeighted(frame1, 1.0, frame, 1.0, 0)

                    #blank frame to place image on
                    frameImg = np.zeros([h, w, 3], dtype=np.uint8)
                    
                    if cX > imgl2 and cY > imgl2 and cX < w-imgl2 and cY < h -imgl2:
                        frame = cv2.circle(frame, (cX,cY-3), int(imgl2-2), (0 ,0, 0), -1) 
                        frameImg[cY-imgl2:cY+imgl2, cX-imgl2:cX+imgl2] = img

                    # frame = cv2.addWeighted(frame, 1.0, frameImg, 1.0, 0)
                    frame += frameImg
            

        #show the output frame
        cv2.imshow("Say Hello", frame)
        key = cv2.waitKey(1) & 0xFF

        #if the 'q' key was pressed, break from the loop
        if key == ord('q'):
            break

    #cleanup
    cv2.destroyAllWindows()
    cap.stop()

def init_worker():
    global state
    

if __name__=='__main__':
    
    pool = Pool(processes=2)
    # r1 = pool.apply_async(aruco_thread, [state])
    # r2 = pool.apply_async(talker_thread, [state])
    x = Process(target=aruco_thread, args=(state,))
    y = Process(target=talker_thread, args=(state,))

    x.start()
    y.start()
    # pool.close()
    # pool.join()
    x.join()
    y.join()
    print('end')