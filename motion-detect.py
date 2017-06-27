import cv2
import numpy as np
import time

staticFrame = None
prevImage = None
inMotion = False
status = "clean"
numFramesWithoutMotion = 0
cam = cv2.VideoCapture(0)
time.sleep(2)
winName = "Movement Indicator"
cv2.namedWindow(winName)

def diffImg(img1, img2):
  diff = cv2.absdiff(img2, img1)
  thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
  thresh = cv2.dilate(thresh, None, iterations=2)
  contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[1]
  for contour in contours:
    contourArea = cv2.contourArea(contour)
    if contourArea > 1000: #threshold for whether or not this is a large enough object to consider (may have to tweak this)
      return True
  return False

def readImage():
  img = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
  return cv2.GaussianBlur(img, (21, 21), 0)

if staticFrame is None:
  readImage() #burn off first frame which seems to always come in as black
  staticFrame = readImage()

while True:
  if prevImage is None:
    prevImage = readImage()
    continue
  nextImage = readImage()
  if diffImg(prevImage, nextImage):
    inMotion = True
    numFramesWithoutMotion = 0
  elif inMotion:
    numFramesWithoutMotion = numFramesWithoutMotion + 1
    if numFramesWithoutMotion > 100: #motion stopped
      inMotion = False
      numFramesWithoutMotion = 0
      if diffImg(staticFrame, nextImage): #sink has changed
        if status == "clean":
          status = "dirty"
          #do something funny
          cv2.imshow(winName, nextImage)
      elif status == "dirty": #someone cleaned up the sink!
        status = "clean"
        #congratulate person on washing dishes?
  prevImage = nextImage
  key = cv2.waitKey(10)
  if key == 27:
    cv2.destroyWindow(winName)
    break