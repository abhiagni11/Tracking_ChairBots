import numpy as np
import cv2
import math
import time

cap = cv2.VideoCapture('sample.mp4')
cap.set(3, 1920)
cap.set(4, 1080)
MAX_BOTS=10

FILE_OUTPUT = 'output_1.avi'
FILE_OUTPUT2 = 'output_2.avi'
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(FILE_OUTPUT,fourcc, 20.0, (1920,1080))
out2 = cv2.VideoWriter(FILE_OUTPUT2,fourcc, 20.0, (1920,1080))

# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
filename1 = ("CB01.txt")
filename2 = ("CB02.txt")
filename3 = ("CB03.txt")
with open(filename1, 'w') as f:
    f.write(filename1)
    f.write('\n')
with open(filename2, 'w') as f:
    f.write(filename2)
    f.write('\n')
with open(filename3, 'w') as f:
    f.write(filename3)
    f.write('\n')


def dist(x, y):
  return np.linalg.norm(y-x)

def run():
  global gray2
  while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 640, 480)
    # print (frame.shape)
    h, w, _ = frame.shape

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = frame
    #gray2 = gray
    res = cv2.aruco.detectMarkers(gray,dictionary)
    # print(res[0],res[1],len(res[2]))
    if len(res[0]) > 0:
      #print len(res[1])
      for (fids, index) in zip(res[0], res[1]):
        #print fids
        for pt in fids:
          try:
            if (int(index[0])==0):
              ll = ((pt[0] +pt[1] +pt[2] +pt[3])/4)
              cv2.circle(gray,(ll[0],ll[1]), 15, (0,0,255), -1)
              cv2.circle(gray2,(ll[0],ll[1]), 15, (0,0,255), -1)

              with open(filename1, 'a') as f:
                f.write(str(ll[0]))
                f.write('\t')
                f.write(str(ll[1]))
                f.write('\t')
                f.write(str(time.time()))
                f.write('\n')
              #cv2.circle(gray,(pt[0],pt[1]), 15, (0,0,255), -1)
            elif (int(index[0])==1):
              ll = ((pt[0] +pt[1] +pt[2] +pt[3])/4)
              cv2.circle(gray,(ll[0],ll[1]), 15, (0,255,0), -1)
              cv2.circle(gray2,(ll[0],ll[1]), 15, (0,255,0), -1)
              with open(filename2, 'a') as f:
                f.write(str(ll[0]))
                f.write('\t')
                f.write(str(ll[1]))
                f.write('\t')
                f.write(str(time.time()))
                f.write('\n')
              #cv2.circle(gray,(pt[0],pt[1]), 15, (0,0,255), -1)
            elif (int(index[0])==3):
              ll = ((pt[0] +pt[1] +pt[2] +pt[3])/4)
              cv2.circle(gray,(ll[0],ll[1]), 15, (255,255,0), -1)
              #cv2.circle(gray,(pt[0],pt[1]), 15, (0,0,255), -1)
            elif (int(index[0])==2):
              ll = ((pt[0] +pt[1] +pt[2] +pt[3])/4)
              cv2.circle(gray,(ll[0],ll[1]), 15, (255,0,0), -1)
              cv2.circle(gray2,(ll[0],ll[1]), 15, (255,0,0), -1)
              with open(filename3, 'a') as f:
                f.write(str(ll[0]))
                f.write('\t')
                f.write(str(ll[1]))
                f.write('\t')
                f.write(str(time.time()))
                f.write('\n')
              #cv2.circle(gray,(pt[0],pt[1]), 15, (0,0,255), -1)
          except IndexError:
            pass

    if len(res[0]) > 0:
      cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])
      #cv2.aruco.drawDetectedMarkers(gray2,res[0],res[1])
      #cv2.circle(gray,(447,63), 3, (0,0,255), -1)
    out.write(gray2)
    out2.write(gray)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      # When everything done, release the capture
      cap.release()
      out.release()
      cv2.destroyAllWindows()
      import sys
      sys.exit()

# for creating a blank image
def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

if __name__ == "__main__":
  global gray2
  gray2 = create_blank(1920, 1080, rgb_color=(0, 0, 0))
  # this might work?
  run()
