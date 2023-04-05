from datetime import datetime
from multiprocessing import Pool, Value, Process, Array
import cv2
import numpy as np
from imutils.video import VideoStream
import argparse
import time
import sys
import pyautogui
import math
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import requests 

import pytz
import config
from dateutil.relativedelta import relativedelta
import keyboard

state = Value('i', 1)
stateOther = Value('i', 1)
nickname = Array('c', b'')
nicknameOther = Array('c', b'')
drawing = Array('c', b'')
drawingOther = Array('c', b'')
description = Array('c', b'')
descriptionOther = Array('c', b'')
emoji = Array('c', b'')
emojiOther = Array('c', b'')

urlId = Array('c', b'')
cX = Value('i', 0)
cY = Value('i', 0)
cXOther = Value('i', 0)
cYOther = Value('i', 0)
stage = Value('i', 1)

# state1, state2, drawing1, drawing2, emoji1, emoji2, all provided at the same time, but some may be null values if not updated by user yet.

#retrieve data from the app and set the states accordingly in openCV

def sender_thread(stage, urlId, cX, cY):
    while True:
        url = 'https://borderless-backend.herokuapp.com/openCVData'
        data = {'id': urlId.value.decode('utf-8'), 'cX': cX.value, 'cY': cY.value}

        response = requests.post(url, json = data)

        time.sleep(0.5)


def talker_thread(stage, urlId, state, stateOther, nickname, nicknameOther, drawing, drawingOther, description,
                  descriptionOther, emoji, emojiOther, cXOther, cYOther):
    while True:
        url_id = requests.get('https://borderless-backend.herokuapp.com/QR').json() # Get unique URL and ID (string of numbers after '?id=')
        # print(url_id)
        urlId.value = url_id['id'].encode('utf-8')

        url = 'https://borderless-backend.herokuapp.com/openCVData?id='
        url += urlId.value.decode('utf-8')
        # print(url)

        first = True

        while True:
            data = requests.get(url).json()
            state.value = data['state']
            stateOther.value = data['stateOther']
            nickname.value = data['nickname'].encode('utf-8')
            nicknameOther.value = data['nicknameOther'].encode('utf-8')
            drawing.value = data['drawing'].encode('utf-8')
            drawingOther.value = data['drawingOther'].encode('utf-8')
            description.value = data['description'].encode('utf-8')
            descriptionOther.value = data['descriptionOther'].encode('utf-8')
            emoji.value = data['emoji'].encode('utf-8')
            emojiOther.value = data['emojiOther'].encode('utf-8')
            cXOther.value = data['cXOther']
            cYOther.value = data['cYOther']

            if not first and state.value == 0 and stateOther.value == 0:
                break

            first = False

            time.sleep(1)

def getStage(state, stateOther):
    if state in [0, 1, 2]:
        return 1
    if state == 3 and state > stateOther:
        return 2
    if state == 3:
        return 3
    
    if state == 4 and stateOther == 4:
        return 4
    if state == 5 and stateOther == 4:
        return 5
    if state == 4 and stateOther == 5:
        return 6
    if state == 5 and stateOther == 5:
        return 7
    
    return 8

#display qr code for people to scan
def stage0(frame):
    pass


#start frame 1 - welcome message     // stage = 0, 1, 2
def stage1(frame):
    cv2.putText(frame, 'Welcome! Press start on device to begin.', (50,50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

#start frame 2 - waiting for other user   // stage = 3, stage > stateOther
def stage2(frame):
    cv2.putText(frame, 'Waiting for other player...', (50,100),
                cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

#prompt                                   // stage = 3
def stage3(frame):
    cv2.putText(frame, 'Draw something that reminds you of your childhood.', (50,50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

# both drawing                                      // state = 4, stateOther = 4
# user finished, userOther has not                  // state = 5, stateOther = 4
# user has not finished, userOther has finished     // state = 4, stateOther = 5
# both finished                                     // state, stateOther = 5


#drawing - user is drawing message         
def stage4(frame):
    pass

#drawing - display of drawings on ar markers
def stage5(frame, cX, cY, imgl2):
    
    cv2.putText(frame, 'Draw something that reminds you of your childhood.', (50,50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
    
    #1. convert retrieved drawings' dataURLs to image file
    
    # decode the base64 string
    file = open('fyp test codes/file1.txt')
    base64_string = file.read()
    file.close()

    #pad the string with '=' to make the length a multiple of 4
    while len(base64_string) % 4 != 0:
        base64_string += "="

    # Remove the "data:image/png;base64," prefix from the string
    base64_string = base64_string.replace("data:image/png;base64,", "")

    image_data = base64.b64decode(base64_string)

    # open the image using PIL
    drawing = Image.open(io.BytesIO(image_data))

    # save the image as a PNG file
    drawing.save("output.png", "PNG")

    # load and initialise output png
    img = cv2.imread('output.png')
    img = cv2.resize(img, (imgl2*2,imgl2*2))

    w = frame.shape[1]
    h = frame.shape[0]

    #blank frame to place image on
    frameImg = np.zeros([h, w, 3], dtype=np.uint8)
                        
    if cX > imgl2 and cY > imgl2 and cX < w-imgl2 and cY < h -imgl2:
        frame = cv2.circle(frame, (cX,cY-3), int(imgl2-2), (255, 255, 255), -1) 
        frameImg[cY-imgl2:cY+imgl2, cX-imgl2:cX+imgl2] = img
        # cv2.imshow('image frame', frameImg)

    # frame = cv2.addWeighted(frame, 1.0, frameImg, 1.0, 0)
    frame += frameImg

# display react prompt
def stage6(frame):
    cv2.putText(frame, 'Draw something that reminds you of your childhood.', (50,50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
    cv2.putText(frame, "React to the other user's drawing!", (50,100),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 2)
    cv2.putText(frame, "Feel free to use gestures, facial expressions, or emojis.", (50,130),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 2)
    

# display emojis
def stage7(frame):
    # Load the font file
    font_file = 'fyp test codes\seguiemj.ttf'
    font_size = 150
    font = ImageFont.truetype(font_file, font_size)

    # Create a new RGBA image with a transparent background
    width, height = 200, 200
    image = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    #Retrieve emoji text from backend
    text = '😀'

    # Get the bounding box of the text
    bbox = font.getbbox(text)

    # Create a new image with dimensions based on the bounding box
    image = Image.new("RGBA", (bbox[2] - bbox[0], bbox[3] - bbox[1]), (255, 255, 255, 0))

    # Draw the text on the image
    draw = ImageDraw.Draw(image)
    draw.text((0, -bbox[1]), text, font=font, embedded_color=True)

    # Save the image to disk
    image.save('colored_emoji.png')

    #load and initialise emoji image in openCV
    emj = cv2.imread('colored_emoji.png', cv2.IMREAD_UNCHANGED).astype('uint8')

    # x1, y1 = bbox[0], bbox[1]
    # x2, y2 = bbox[2], bbox[1]
    # x3, y3 = bbox[2], bbox[3]
    # x4, y4 = bbox[0], bbox[3]
    # cv2.circle(emj, (x1,y1), 5, (255,0,0), -1 )
    # cv2.circle(emj, (x2,y2), 5, (0,255,0), -1 )
    # cv2.circle(emj, (x3,y3), 5, (0,255,255), -1 )
    # cv2.circle(emj, (x4,y4), 5, (255,0,255), -1 )

    # cv2.imshow('emoji bbox', emj)

    emoji_w = bbox[2] - bbox[0]
    emoji_h = bbox[3] - bbox[1]


    w = frame.shape[1]
    h = frame.shape[0]

    #convert video feed to rgba
    height, width, channels = frame.shape
    rgba_frame = np.zeros((height, width, 4), dtype=np.uint8)
    rgba_frame[:,:,0:3] = frame

    #blank frame to place image on
    # frameImg = np.zeros([h, w, 3], dtype=np.uint8)

    #initialise a blank rgba frame
    channels = 4  # RGBA
    frameImg = np.zeros((h, w, channels), dtype=np.uint8)
    frameImg[:,:,3] = 0

    #display emoji on top left corner of window
    frameImg[0:emoji_h, 0:emoji_w] = emj
    # cv2.imshow('emoji frame', frameImg)

    # rgba_frame += frameImg
    frame = cv2.addWeighted(rgba_frame, 1.0, frameImg, 0.5, 0)
    cv2.imshow('emoji overlay',frame)


# display thank you message
def stage8(frame):
    cv2.putText(frame, 'Thank you for playing!', (50,50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
    cv2.putText(frame, 'Check out the archive wall outside.', (50,100),
                cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

#loop that is running the aruco program
def aruco_thread(stage, urlId, state, stateOther, nickname, nicknameOther, drawing, drawingOther, description,
                  descriptionOther, emoji, emojiOther, cXOther, cYOther, cX, cY):
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

    flag=1

    scale =  1.35 #1.55
    width = 200
    height = 800

    #loop over frames from video stream
    while True:

        # if start_time + relativedelta(seconds=duration) > curr_time:
        if keyboard.is_pressed('p'):
            sys.stdin = open(0)
            # scale = float(input('Enter scale, current {}: '.format(str(scale))) or str(scale))
            # width = int(input('Enter width trim start, current {}: '.format(str(width))) or str(width))
            # height = int(input('Enter height trim start, current {}: '.format(str(height))) or str(height))
            stage.value = int(input('Enter stage value: ') or str(stage.value))
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
        # print(stage.value)

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
        cX, cY = 0,0
        imgl2=1

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

                if flag == 1: #relevant aruco markers showing up on screen

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

                    # cX and cY would need to be global variables with shared memory
                    # center (x,y) coordinates of the aruco marker 
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)

                    frame = cv2.circle(frame, (cX,cY-3), int(imgl2-2), (255, 255, 255), -1) #display a white circle on aruco marker by default
                
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
                    # w = frame.shape[1]
                    # h = frame.shape[0]

                    # #blank frame to place image on
                    # frameImg = np.zeros([h, w, 3], dtype=np.uint8)
                    
                    # if cX > imgl2 and cY > imgl2 and cX < w-imgl2 and cY < h -imgl2:
                    #     # frame = cv2.circle(frame, (cX,cY-3), int(imgl2-2), (255, 255, 255), -1) 
                    #     frameImg[cY-imgl2:cY+imgl2, cX-imgl2:cX+imgl2] = img

                    # frame = cv2.addWeighted(frame, 1.0, frameImg, 1.0, 0)
                    # frame += frameImg
        
        if stage.value ==1:
            stage1(frame)
        elif stage.value == 2:
            stage2(frame)
        elif stage.value ==3:
            stage3(frame)
        elif stage.value==4:
            stage4(frame)   
        elif stage.value ==5:
            stage5(frame, cX, cY, imgl2)
        elif stage.value ==6:
            stage6(frame)
        elif stage.value ==7:
            stage7(frame)
        elif stage.value==8:
            stage8(frame)

        #show the output frame
        cv2.imshow("Say Hello", frame)

        key = cv2.waitKey(1) & 0xFF

        #if the 'q' key was pressed, break from the loop
        if key == ord('q'):
            break

    #cleanup
    cv2.destroyAllWindows()
    cap.stop()

    

if __name__=='__main__':
    
    arucoThread = Process(target=aruco_thread, args=(stage, urlId, state, stateOther, nickname, nicknameOther, drawing, drawingOther, description,
                  descriptionOther, emoji, emojiOther, cXOther, cYOther, cX, cY))
    talkerThread = Process(target=talker_thread, args=(stage, urlId, state, stateOther, nickname, nicknameOther, drawing, drawingOther, description,
                  descriptionOther, emoji, emojiOther, cXOther, cYOther))
    senderThread = Process(target=sender_thread, args=(state, urlId, cX, cY))

    arucoThread.start()
    talkerThread.start()
    senderThread.start()
    # pool.close()
    # pool.join()
    arucoThread.join()
    talkerThread.join()
    senderThread.join()
    print('end')