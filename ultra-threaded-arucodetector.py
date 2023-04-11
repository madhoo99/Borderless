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
import qrcode as QR

import pytz
import config
from dateutil.relativedelta import relativedelta
import keyboard

stringSize = 1000000

state = Value('i', 1)
stateOther = Value('i', 1)
nickname = Array('c', stringSize)
nicknameOther = Array('c', stringSize)
drawing = Array('c', stringSize)
drawingOther = Array('c', stringSize)
description = Array('c', stringSize)
descriptionOther = Array('c', stringSize)
emoji = Array('c', stringSize)
emojiOther = Array('c', stringSize)

urlId = Array('c', stringSize)
urlIdOther = Array('c', stringSize)
cX = Value('i', 0)
cY = Value('i', 0)
cXOther = Value('i', 0)
cYOther = Value('i', 0)
isDrawingReady = Value('b', False)
isDrawingReadyOther = Value('b', False)
stage = Value('i', 1)

reset = Value('b', False)


#retrieve data from the app and set the states accordingly in openCV

def sender_thread(urlId, cX, cY):
    while True:
        url = 'https://borderless-backend.herokuapp.com/setcXcY'
        data = {'id': urlId.value.decode('utf-8'), 'cX': cX.value, 'cY': cY.value}

        response = requests.post(url, json = data)

        time.sleep(0.1)

def talker_thread_light(urlId, urlIdOther, state, stateOther, cXOther, cYOther, emoji, emojiOther, isDrawingReady, isDrawingReadyOther,
                        reset):
    
    while True:
        try:
            url_id = requests.get('https://borderless-backend.herokuapp.com/QR').json() # Get unique URL and ID (string of numbers after '?id=')
            # print(url_id)
            urlId.value = url_id['id'].encode('utf-8')

            url = 'https://borderless-backend.herokuapp.com/openCVDataLight' + '?id=' + urlId.value.decode('utf-8')

            first = True
            
            while True:
            
                response = requests.get(url).json()
                data = response['data']
                state.value = data['state']
                stateOther.value = data['stateOther']
                cXOther.value = data['cXOther']
                cYOther.value = data['cYOther']
                emoji.value = data['emoji'].encode('utf-8')
                emojiOther.value = data['emojiOther'].encode('utf-8')
                isDrawingReady.value = data['isDrawingReady']
                isDrawingReadyOther.value = data['isDrawingReadyOther']
                urlIdOther.value = data['urlIdOther'].encode('utf-8')
                reset.value = data['reset']

                # if not first and state.value == 0 and stateOther.value == 0:
                #     break

                if reset.value:
                    print('breaks')
                    break

                if state.value > 0 or stateOther.value > 0:
                    first = False

                time.sleep(0.1)
        except Exception as e:
            print(e)

def nickname_get_thread(urlId, nickname, nicknameOther):
    time.sleep(2)
    url = 'https://borderless-backend.herokuapp.com/nickname' + '?id=' + urlId.value.decode('utf-8')
    
    response = requests.get(url).json()
    data = response['data']
    nickname.value = data['nickname'].encode('utf-8')
    nicknameOther.value = data['nicknameOther'].encode('utf-8')

def drawing_get_thread(urlId, drawing, description):
    while description.value.decode('utf-8') == '':
        print('I am repeatedly querying le DB')
        time.sleep(1)
        url = 'https://borderless-backend.herokuapp.com/drawing' + '?id=' + urlId.value.decode('utf-8')
        
        response = requests.get(url).json()
        data = response['data']
        drawing.value = data['drawing'].encode('utf-8')
        description.value = data['description'].encode('utf-8')
    print('drawign received')

def drawing_save_thread(drawing, fileName):
    print('drawing being saved')
    base64_string = drawing

    #pad the string with '=' to make the length a multiple of 4
    while len(base64_string) % 4 != 0:
        base64_string += "="

    # Remove the "data:image/png;base64," prefix from the string
    base64_string = base64_string.replace("data:image/png;base64,", "")

    image_data = base64.b64decode(base64_string)

    # open the image using PIL
    drawing = Image.open(io.BytesIO(image_data))
    drawing = np.array(drawing)

    for row in drawing:
        for pixel in row:
            if pixel[3] !=0:
                pixel[0] = 128
                pixel[1] = 128
                pixel[2] = 128

    drawing = Image.fromarray(drawing)

    # save the image as a PNG file
    drawing.save(fileName, "PNG")

    print('done')

def talker_thread(stage, urlId, state, stateOther, nickname, nicknameOther, drawing, drawingOther, description,
                  descriptionOther, emoji, emojiOther, cXOther, cYOther):

    while True:
        url_id = requests.get('https://borderless-backend.herokuapp.com/QR').json() # Get unique URL and ID (string of numbers after '?id=')
        # print(url_id)
        urlId.value = url_id['id'].encode('utf-8')

        url = 'https://borderless-backend.herokuapp.com/openCVData?id='
        url += urlId.value.decode('utf-8')

        first = True

        while True:
            data = requests.get(url).json()['data'] 
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

            if state.value > 0 or stateOther.value > 0:
                first = False

            time.sleep(0.1)

def getStage(state, stateOther):
    if stateOther in [0, 1, 2]: # welcome message, nickname, drawing prompt
        return 1
    if stateOther == 3 and stateOther > state:  
        return 2
    if (stateOther == 3 or stateOther == 4) or (stateOther == 5 and state != 5):
        return 3
    if stateOther in [5,6] and state in [5,6]:
        return 4
    if stateOther in [7,8,9]:
        return 5
    
    return -1

#start frame 1 - welcome message and display QR code    // state = 0, 1, 2
def stage1(frame, urlIdOther):
    # print('I am in stage 1.')

    cv2.putText(frame, 'Welcome! Press start on device to begin.', (50,50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
    
    URL = 'https://borderless-frontend-new.herokuapp.com/home?id=' #Add your URL
    URL += urlIdOther
    
    # print(URL)

    code = QR.make(URL)
    code.save('QR.png')
    
    url_img = cv2.imread('QR.png')
    imgl2 = 200

    url_img = cv2.resize(url_img, (imgl2*2,imgl2*2))

    w = frame.shape[1]
    h = frame.shape[0]

    #blank frame to place image on
    frameImg = np.zeros([h, w, 3], dtype=np.uint8)

    centerX = int(frameImg.shape[1]/2)
    centerY = int(frameImg.shape[0]/2)
    cv2.rectangle(frame, (centerX - imgl2, centerY + imgl2), (centerX + imgl2, centerY - imgl2), (0, 0, 0), -1)

    frameImg[int(centerY- imgl2):int(centerY+ imgl2), int(centerX- imgl2):int(centerX+ imgl2)] = url_img

    # frame = cv2.addWeighted(frame, 1.0, frameImg, 1.0, 0)
    frame += frameImg

    

#start frame 2 - waiting for other user   // stage = 3, stage > stateOther
def stage2(frame):
    # print('I am in stage 2.')
    cv2.putText(frame, 'Waiting for other player...', (50,100),
                cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

# both drawing                                      // state = 4, stateOther = 4
# user finished, userOther has not                  // state = 5, stateOther = 4
# user has not finished, userOther has finished     // state = 4, stateOther = 5
# both finished                                     // state, stateOther = 5

def stage3(frame, cX, cY, imgl2, corners, 
           drawing, drawingOther, nickname, nicknameOther,
           cXOther, cYOther, saveDrawing, saveDrawingOther):
    cv2.putText(frame, 'Draw something that reminds you of your childhood.', (50,50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
    
    saveDrawing, saveDrawingOther = append_drawing(frame, cX, cY, imgl2, corners, 
           drawing, drawingOther, nickname, nicknameOther,
           cXOther, cYOther, saveDrawing, saveDrawingOther)
    
    return saveDrawing, saveDrawingOther


#drawing - display of drawings on ar markers
def append_drawing(frame, cX, cY, imgl2, corners, 
           drawing, drawingOther, nickname, nicknameOther,
           cXOther, cYOther, saveDrawing, saveDrawingOther):

    # print('I am in stage 3.')

    #1. If drawing exists, convert retrieved drawings' dataURLs to image file, else display 'nickname is drawing' message
    # tagged to cX, cY

    if drawing != '':
        try:
            fileName = 'output.png'
            if saveDrawing:
                saveDrawingThread = Process(target=drawing_save_thread, args=(drawing, fileName))
                saveDrawingThread.start()
                saveDrawing = False

            # load and initialise output png
            img = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
            # print(img.shape)

            img = cv2.resize(img, (imgl2*2,imgl2*2))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            w = frame.shape[1]
            h = frame.shape[0]

            #blank frame to place image on
            frameImg = np.zeros([h, w, 3], dtype=np.uint8)
            

                              
            if cX > imgl2 and cY > imgl2 and cX < w-imgl2 and cY < h -imgl2:
                frame = cv2.circle(frame, (cX,cY-3), int(imgl2-2), (255, 255, 255), -1) 
                frameImg[cY-imgl2:cY+imgl2, cX-imgl2:cX+imgl2] = img
            # cv2.imshow('image frame', frameImg)

            frame += frameImg

        except: 
            pass

        # frame = cv2.addWeighted(frame, 1.0, frameImg, 1.0, 0)
    else:

        w = frame.shape[1]
        h = frame.shape[0]

        if corners> 0: #only if marker is detected
            try:
                if cX > imgl2 and cY > imgl2 and cX < w-imgl2 and cY < h -imgl2:
                    frame = cv2.circle(frame, (cX,cY-3), int(imgl2-2), (255, 255, 255), -1) 
                    cv2.putText(frame, nickname, (cX-80, cY-20), 
                                cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
                    cv2.putText(frame, 'is drawing...', (cX-100, cY+10), 
                                cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
            except:
                pass

    #2. If drawingOther exists, display on other bubble. Else, display 'nickname' is drawing message
    # tagged to cX other, cY other

    if drawingOther != '':
        try:
            fileName = 'outputother.png'
            if saveDrawingOther:
                saveDrawingThread = Process(target=drawing_save_thread, args=(drawingOther, fileName))
                saveDrawingThread.start()
                saveDrawingOther = False

            # load and initialise output png
            img = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
            # print(img.shape)

            img = cv2.resize(img, (imgl2*2,imgl2*2))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            w = frame.shape[1]
            h = frame.shape[0]

            #blank frame to place image on
            frameImg = np.zeros([h, w, 3], dtype=np.uint8)
               
            if cXOther > imgl2 and cYOther > imgl2 and cXOther < w-imgl2 and cYOther < h -imgl2:
                frame = cv2.circle(frame, (cXOther,cYOther-3), int(imgl2-2), (255, 255, 255), -1) 
                frameImg[cYOther-imgl2:cYOther+imgl2, cXOther-imgl2:cXOther+imgl2] = img
            # cv2.imshow('image frame', frameImg)

            frame += frameImg

        except: 
            pass

        # frame = cv2.addWeighted(frame, 1.0, frameImg, 1.0, 0)
        
    
    else:
        w = frame.shape[1]
        h = frame.shape[0]
        try:
            if cXOther > imgl2 and cYOther > imgl2 and cXOther < w-imgl2 and cYOther < h -imgl2:
                cv2.circle(frame, (cXOther,cYOther-3), int(imgl2-2), (255, 255, 255), -1) 
                cv2.putText(frame, nicknameOther, (cXOther-80, cYOther-20), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
                cv2.putText(frame, 'is drawing...', (cXOther-100, cYOther+10), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
        except:
            pass

    return saveDrawing, saveDrawingOther

    
# no emoji                                      // state = 4, stateOther = 4
# user showing emoji, userOther not             // state = 5, stateOther = 4
# user not showing emoji, userOther is          // state = 4, stateOther = 5
# both showing emoji                            // state, stateOther = 5

# display emojis
def stage4(frame, emoji, emojiOther, cX, cY, imgl2, corners, 
           drawing, drawingOther, nickname, nicknameOther,
           cXOther, cYOther, saveDrawing, saveDrawingOther):
    
    saveDrawing, saveDrawingOther = append_drawing(frame, cX, cY, imgl2, corners, 
           drawing, drawingOther, nickname, nicknameOther,
           cXOther, cYOther, saveDrawing, saveDrawingOther)

    cv2.putText(frame, 'Draw something that reminds you of your childhood.', (50,50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
    cv2.putText(frame, "React to the other user's drawing!", (50,100),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 2)
    cv2.putText(frame, "Feel free to use gestures, facial expressions, or emojis.", (50,130),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 2)

    #1. If emoji exists, display emoji on offset center of screen(else, do nothing)
    # if emoji  '':

    if emoji != '':

        try:

            # print('emoji = ' + emoji)

            # Load the font file
            font_file = 'seguiemj.ttf'
            font_size = 150
            font = ImageFont.truetype(font_file, font_size)

            # Create a new RGBA image with a transparent background
            width, height = 200, 200
            image = Image.new('RGBA', (width, height), (0, 0, 0, 0))

            #Retrieve emoji text from backend
            text = emoji #text = drawing / drawingOther

            # Get the bounding box of the text
            bbox = font.getbbox(text) 

            # Create a new image with dimensions based on the bounding box
            image = Image.new("RGBA", (bbox[2] - bbox[0], bbox[3] - bbox[1]), (255, 255, 255, 0))

            # Draw the text on the image
            draw = ImageDraw.Draw(image)
            draw.text((0, -bbox[1]), text, font=font, embedded_color=True)

            # Save the image to disk
            bbox1 = (image.getbbox())
            image = image.crop(bbox1)
            image.save('emoji.png')

            #load and initialise emoji image in openCV
            emj = cv2.imread('emoji.png')

            # emoji_w = int(bbox1[2] - bbox1[0])
            # emoji_h = int(bbox1[3] - bbox1[1])

            emoji_w = int(emj.shape[1])
            emoji_h = int(emj.shape[0])

            w = frame.shape[1]
            h = frame.shape[0]

            # blank frame to place image on
            frameImg = np.zeros([h, w, 3], dtype=np.uint8)

            # get coordinates for center of frame
            centerX = int(frameImg.shape[1]/2) - 200
            centerY = int(frameImg.shape[0]/2)

            #draw a circle under the emoji area
            cv2.circle(frame, (centerX, centerY), int(emoji_h/2), (0, 0,0), -1)

            # #display emoji in the center of window
            frameImg[int(centerY- (emoji_h/2)):int(centerY+ (emoji_h/2)), int(centerX- (emoji_w/2)):int(centerX+ (emoji_w/2))] = emj

            frame += frameImg
        except:
            
            # blank frame to place image on
            frameImg = np.zeros([h, w, 3], dtype=np.uint8)

            # get coordinates for center of frame
            centerX = int(frameImg.shape[1]/2) - 200
            centerY = int(frameImg.shape[0]/2)

            # display text
            cv2.putText(frame, 'Emoji', (centerX -50, centerY - 20), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)
            cv2.putText(frame, 'not available', (centerX-100, centerY + 20), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)


    #2. If emojiOther exists, display other emoji on offset center of screen (mirrored)
    if emojiOther != '':

        try:
        
            # print('emojiOther = ' + emojiOther)

            # Load the font file
            font_file = 'seguiemj.ttf'
            font_size = 150
            font = ImageFont.truetype(font_file, font_size)

            # Create a new RGBA image with a transparent background
            width, height = 200, 200
            image = Image.new('RGBA', (width, height), (0, 0, 0, 0))

            #Retrieve emoji text from backend
            text = emojiOther #text = drawing / drawingOther

            # Get the bounding box of the text
            bbox = font.getbbox(text) 

            # Create a new image with dimensions based on the bounding box
            image = Image.new("RGBA", (bbox[2] - bbox[0], bbox[3] - bbox[1]), (255, 255, 255, 0))

            # Draw the text on the image
            draw = ImageDraw.Draw(image)
            draw.text((0, -bbox[1]), text, font=font, embedded_color=True)

            # Save the image to disk
            bbox1 = (image.getbbox())
            image = image.crop(bbox1)
            image.save('emojiOther.png')

            #load and initialise emoji image in openCV
            emj = cv2.imread('emojiOther.png')

            # emoji_w = int(bbox1[2] - bbox1[0])
            # emoji_h = int(bbox1[3] - bbox1[1])

            emoji_w = int(emj.shape[1])
            emoji_h = int(emj.shape[0])

            w = frame.shape[1]
            h = frame.shape[0]

            # blank frame to place image on
            frameImg = np.zeros([h, w, 3], dtype=np.uint8)

            # get coordinates for center of frame
            centerX = int(frameImg.shape[1]/2) + 200
            centerY = int(frameImg.shape[0]/2)

            #draw a circle under the emoji area
            cv2.circle(frame, (centerX, centerY), int(emoji_h/2), (0, 0,0), -1)

            # #display emoji in the center of window
            frameImg[int(centerY- (emoji_h/2)):int(centerY+ (emoji_h/2)), int(centerX- (emoji_w/2)):int(centerX+ (emoji_w/2))] = emj

            frame += frameImg
        except:
            # blank frame to place image on
            frameImg = np.zeros([h, w, 3], dtype=np.uint8)

            # get coordinates for center of frame
            centerX = int(frameImg.shape[1]/2) + 200
            centerY = int(frameImg.shape[0]/2)

            # display text
            cv2.putText(frame, 'Emoji', (centerX -50, centerY -20), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 1)
            cv2.putText(frame, 'not available', (centerX-100, centerY + 20), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 1)



    return saveDrawing, saveDrawingOther


# display thank you message
def stage5(frame):

    cv2.putText(frame, 'Thank you for playing!', (50,50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
    cv2.putText(frame, 'Check out the archive wall outside.', (50,100),
                cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
    
def getNicknameAndDrawings(getNickname, getDrawing, getDrawingOther, isDrawingReady, isDrawingReadyOther,
                           urlId, urlIdOther, nickname, nicknameOther, drawing, drawingOther,
                           description, descriptionOther):
    if getNickname:
        nicknameGetThread = Process(target=nickname_get_thread, args=(urlId, nickname, nicknameOther))
        nicknameGetThread.start()
        getNickname = False
    if isDrawingReady.value and getDrawing:
        drawingGetThread = Process(target=drawing_get_thread, args=(urlId, drawing, description))
        drawingGetThread.start()
        getDrawing = False
    if isDrawingReadyOther.value and getDrawingOther:
        drawingGetThreadOther = Process(target=drawing_get_thread, args=(urlIdOther, drawingOther, descriptionOther))
        drawingGetThreadOther.start()
        getDrawingOther = False

#loop that is running the aruco program
def aruco_thread(stage, urlId, urlIdOther, state, stateOther, nickname, nicknameOther, drawing, drawingOther, description,
                  descriptionOther, emoji, emojiOther, cXOther, cYOther, cX, cY, isDrawingReady, isDrawingReadyOther):
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

    scale =  1.1 #1.55 #1.35
    width = 100 #200
    height = 900 #800

    # for booth 1:
    # scale = 1.03
    # width = 160
    # height = 820

    # logitech
    # scale= 1.03
    # width = 0
    # height = 820
    

    # globals local to aruco thread
    getDrawing = True
    getDrawingOther = True
    getNickname = True
    saveDrawing = True
    saveDrawingOther = True

    #loop over frames from video stream
    while True:

        # if start_time + relativedelta(seconds=duration) > curr_time:
        if keyboard.is_pressed('p'):
            sys.stdin = open(0)
            scale = float(input('Enter scale, current {}: '.format(str(scale))) or str(scale))
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
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # print(w, h)


        #detect Aruco markers in the input frame
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters = arucoParams)

        # Detect the faces
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        #verify at least one aruco marker was detected
        imgl2=200

        if len(corners) > 0: #if marker is detected

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
                    cX.value = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY.value = int((topLeft[1] + bottomRight[1]) / 2.0)

                    frame = cv2.circle(frame, (cX.value,cY.value-3), int(imgl2-2), (255, 255, 255), -1) #display a white circle on aruco marker by default
                
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
        
        stage.value = getStage(state.value, stateOther.value)

        # stage.value = getStage(stateOther.value, state.value)

        # print('got drawing ' + drawing.value.decode('utf-8'))

        if stage.value == 1:
            stage1(frame, urlIdOther.value.decode('utf-8'))
        elif stage.value == 2:
            stage2(frame)
        elif stage.value ==3:
            if getNickname:
                nicknameGetThread = Process(target=nickname_get_thread, args=(urlId, nickname, nicknameOther))
                nicknameGetThread.start()
                getNickname = False
            if isDrawingReady.value and getDrawing:
                drawingGetThread = Process(target=drawing_get_thread, args=(urlId, drawing, description))
                drawingGetThread.start()
                getDrawing = False
            if isDrawingReadyOther.value and getDrawingOther:
                drawingGetThreadOther = Process(target=drawing_get_thread, args=(urlIdOther, drawingOther, descriptionOther))
                drawingGetThreadOther.start()
                getDrawingOther = False
            saveDrawing, saveDrawingOther = stage3(frame, cX.value, cY.value, imgl2, len(corners), drawing.value.decode('utf-8'), drawingOther.value.decode('utf-8'), 
                   nickname.value.decode('utf-8'), nicknameOther.value.decode('utf-8'),
                    cXOther.value, cYOther.value, saveDrawing, saveDrawingOther)
        elif stage.value == 4:
            if getNickname:
                nicknameGetThread = Process(target=nickname_get_thread, args=(urlId, nickname, nicknameOther))
                nicknameGetThread.start()
                getNickname = False
            if isDrawingReady.value and getDrawing:
                drawingGetThread = Process(target=drawing_get_thread, args=(urlId, drawing, description))
                drawingGetThread.start()
                getDrawing = False
            if isDrawingReadyOther.value and getDrawingOther:
                drawingGetThreadOther = Process(target=drawing_get_thread, args=(urlIdOther, drawingOther, descriptionOther))
                drawingGetThreadOther.start()
                getDrawingOther = False
            saveDrawing, saveDrawingOther = stage4(frame, emoji.value.decode('utf-8'), emojiOther.value.decode('utf-8'), cX.value, cY.value, imgl2, len(corners), drawing.value.decode('utf-8'), drawingOther.value.decode('utf-8'), 
                   nickname.value.decode('utf-8'), nicknameOther.value.decode('utf-8'),
                    cXOther.value, cYOther.value, saveDrawing, saveDrawingOther)
        elif stage.value == 5:
            stage5(frame)
            getDrawing = True
            getDrawingOther = True
            getNickname = True
            saveDrawing = True
            saveDrawingOther = True

        #show the output frame
        cv2.imshow("Say Hello", frame)

        key = cv2.waitKey(1) & 0xFF

        #if the 'q' key was pressed, break from the loop
        if key == ord('q'):
            break

    #cleanup
    cap.release()
    cv2.destroyAllWindows()
       

if __name__=='__main__':
    
    arucoThread = Process(target=aruco_thread, args=(stage, urlId, urlIdOther, state, stateOther, nickname, nicknameOther, drawing, drawingOther, description,
                  descriptionOther, emoji, emojiOther, cXOther, cYOther, cX, cY, isDrawingReady, isDrawingReadyOther))
    # talkerThread = Process(target=talker_thread, args=(stage, urlId, state, stateOther, nickname, nicknameOther, drawing, drawingOther, description,
    #               descriptionOther, emoji, emojiOther, cXOther, cYOther))
    talkerThreadLight = Process(target=talker_thread_light, args=(urlId, urlIdOther, state, stateOther, cXOther, cYOther, emoji, emojiOther, 
                                                                  isDrawingReady, isDrawingReadyOther, reset))
    senderThread = Process(target=sender_thread, args=(urlId, cX, cY))

    arucoThread.start()
    # talkerThread.start()
    talkerThreadLight.start()
    senderThread.start()
    # pool.close()
    # pool.join()
    arucoThread.join()
    # talkerThread.join()
    talkerThreadLight.join()
    senderThread.join()
    print('end')