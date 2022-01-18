import cv2,sys,imutils, math
import numpy as np

# decide which quadrant ball is in
def quadrant_decision(quad_dict):
    cnt = 0
    quadrant = None
    for key, value in quad_dict.items():
      if value != 0:
        cnt += 1
        quadrant = str(key) 
    
    if(cnt == 0):
      return 'UnKnown'
    elif(cnt == 1):
      return quadrant

    return 'On Line' 
  
# resize image to percent
def resize_img(image, scale_percent=50):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
   
# find top 4 quadrants and center of each quadrants   
def find_quadrants_and_center(image):
      cntr_center= []
      # Grayscale
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      # smooth image to take out the noise
      gray = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)

      # Find Canny edges
      edged = cv2.Canny(gray, 1,200)

      # Taking a matrix of size 2 as the kernel for dilation
      kernel = np.ones((2,2), np.uint8)
      edged = cv2.dilate(edged, kernel, iterations=1)
          
      # Finding Contours
      # Use a copy of the image e.g. edged.copy()
      # since findContours alters the image
      _, contours, hierarchy = cv2.findContours(edged.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
      contour = sorted(contours, key=cv2.contourArea, reverse=True)

      # **** assuming only 4 quadrant ******   
      for i in range(4):
        M = cv2.moments(contour[i])
        if M['m00'] != 0:
          cntr_center.append((int(M['m10']/M['m00']), int(M['m01']/M['m00'])))
          
      return contour, cntr_center

# check if tennis ball is present and return number of contours
# here we use HSV color space to find tennis ball
 
def check_if_any_tennis_ball_in_img(img):
     # color space
    greenLower = (29, 86, 6)
    greenUpper = (64, 255, 255)  
    
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)  
    return cnts

#find 8 points on detected tennis ball
def find_points_on_ball(img,center, radius, no_points=8):
  pointx = []
  pointy = []
        
  # # get 8 point on the ball
  for i in range(no_points):
    angle = (i*(360/no_points)) * ( math.pi / 180 ); # Convert from Degrees to Radians
    pointx.append(int(center[0] + radius * math.cos(angle)))
    pointy.append(int(center[1] + radius * math.sin(angle)))
    cv2.circle(img, (pointx[i],pointy[i]), 2, (0, 0, 255), -1)
 
  return pointx,pointy         

def main():
  
  cap = cv2.VideoCapture('RollingTennisBall.mov')
  
  # Check if camera opened successfully
  if (cap.isOpened()== False):
    print("Error opening video stream or file")

  ball_detected = False
      
  #print("Number of Contours found = " + str(len(contours)))
  contour=None
  cntr_center = []
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, image = cap.read()
    if(not ret):
      sys.exit(0)
      
    # Grayscale
    scale_percent = 60 # percent of original size
    img = resize_img(image,scale_percent)

    if not ball_detected:
      cntr_center.clear()
      # find quadrants and centre
      contour, cntr_center = find_quadrants_and_center(img)

    center = None
    cnts = check_if_any_tennis_ball_in_img(img)
    # only proceed if at least one contour was found
    if len(cnts) > 0:
      ball_detected = True
      # find the largest contour in the mask, then use
      # # it to compute the minimum enclosing circle and
      # # centroid
      c = max(cnts, key=cv2.contourArea)
      ((x, y), radius) = cv2.minEnclosingCircle(c)
      M = cv2.moments(c)
      center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
      
      # only proceed if the radius meets a minimum size
      if radius > 40:
        
        # draw the circle and centroid on the frame
        cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        cv2.circle(img, center, 2, (0, 0, 255), -1)

        # find points on circle
        pointx, pointy= find_points_on_ball(img,center, radius, 8)
        
        quad_dict = {0: 0, 1: 0, 2: 0, 3: 0}
        # check if the points in any quadrant.
        for i in range(8):
          for j in range(4):
            cntHull = cv2.convexHull(contour[j], returnPoints=True)
            quad_dict[j] += 0 if cv2.pointPolygonTest(cntHull,(pointx[i],pointy[i]),False) <= 0 else 1
              
        img = cv2.putText(img, 'Ball Quadrant : ' + quadrant_decision(quad_dict), (400,40), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 255), 2, cv2.LINE_AA)  
        
    for i in range(4):    
      img = cv2.putText(img, str(i), cntr_center[i], cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 255), 2, cv2.LINE_AA)
  
    cv2.imshow('Tracking ball Quadrant', img)
    
    if cv2.waitKey(33) & 0xff == 27:
      cv2.destroyAllWindows()
      sys.exit(0)

if __name__ == "__main__":
  main()
