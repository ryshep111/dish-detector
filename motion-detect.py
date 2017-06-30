import cv2
import numpy as np
import time

status = "clean"
inMotion = False
numFramesWithMotion = 0
numFramesWithoutMotion = 0
numFramesWithoutTrain = 0
minFramesRequiredForMotion = 5
minContourArea = 1000 #threshold for whether or not this is a large enough object to consider (may have to tweak this)
bgTrainRate = 100 #how many still frames before the bg is retrained to account for gradual lighting changes
stillFramesThreshold = 100 #how many consecutive still frames before motion is considered to have stopped
cam = cv2.VideoCapture(1)
# time.sleep(2)
winName = "Movement Indicator"
cv2.namedWindow(winName)
backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=150)
backgroundSubtractor.setShadowThreshold(0.2)
motionBackgroundSubtractor = cv2.createBackgroundSubtractorMOG2(history=minFramesRequiredForMotion, varThreshold=150, detectShadows=False)

def trainBGSubtractor(bgSubtractor, numFrames):
  for i in range(0, numFrames):
    bgSubtractor.apply(readImage(), learningRate=0.5)

def compareToBG(bgSubtractor, img):
  fg = bgSubtractor.apply(img, learningRate=0)
  thresh = cv2.threshold(fg, 254, 255, cv2.THRESH_BINARY)[1]
  return cv2.dilate(thresh, None, iterations=2)

def isDiff(diff):
  contours = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[1]
  for contour in contours:
    contourArea = cv2.contourArea(contour)
    if contourArea > minContourArea:
      return True
  return False

def readImage():
  img = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
  return cv2.GaussianBlur(img, (21, 21), 0)

readImage() # burn off first frame which seems to always come in as black
trainBGSubtractor(backgroundSubtractor, 10)
while True:
  trainBGSubtractor(motionBackgroundSubtractor, 1)
  nextImage = readImage()
  if isDiff(compareToBG(motionBackgroundSubtractor, nextImage)):
    numFramesWithMotion = numFramesWithMotion + 1
    if numFramesWithMotion > minFramesRequiredForMotion:
      inMotion = True
      print 'in motion'
    numFramesWithoutMotion = 0
  elif inMotion:
    numFramesWithMotion = 0
    numFramesWithoutMotion = numFramesWithoutMotion + 1
    if numFramesWithoutMotion > stillFramesThreshold: #motion stopped
      inMotion = False
      numFramesWithoutMotion = 0
      if isDiff(compareToBG(backgroundSubtractor, nextImage)): #sink has changed
        if status == "clean":
          status = "dirty"
          #do something funny
          cv2.imshow(winName, nextImage)
          print 'changed to dirty'
        else:
          print 'state is still dirty'
      elif status == "dirty": #someone cleaned up the sink!
        status = "clean"
        print 'changed to clean'
        #congratulate person on washing dishes?
  elif numFramesWithMotion > 0 and numFramesWithMotion <= minFramesRequiredForMotion:
    numFramesWithoutMotion = numFramesWithoutMotion + 1
    if numFramesWithoutMotion > stillFramesThreshold:
      numFramesWithMotion = 0
      numFramesWithoutMotion = 0
      if status == "clean": #environment may have changed on us, retrain bg
        trainBGSubtractor(backgroundSubtractor, 10)
        numFramesWithoutTrain = 0
        print 'training due to sudden change...'
  elif status == 'clean':
    numFramesWithoutTrain = numFramesWithoutTrain + 1
    if numFramesWithoutTrain > bgTrainRate:
      print 'training...'
      trainBGSubtractor(backgroundSubtractor, 10)
      numFramesWithoutTrain = 0
  key = cv2.waitKey(10)
  if key == 27:
    cv2.destroyWindow(winName)
    break