import cv2
import numpy as np





def get_centroid(x,y,w,h):
    cx=x + int(w/2)
    cy=y + int(h/2)
    return (cx,cy)
# web camera
cap = cv2.VideoCapture('./video.mp4')

# min width rect

min_w_rect=80
# initializing subtractor 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# line
line_pos=550
# offset allowable err b/w pixel
offset=6
# counter vehicle

counter=0

# initializing subtractor 
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG() 
  
while(1):
    ret, frame = cap.read()
  
    # applying on each frame
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(5,5),0)
    # applied segmentation
    fgmask = fgbg.apply(frame)
    
    fgmask=cv2.dilate(fgmask,np.ones((5,5)))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)  
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel) 
    contours,_ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
    cv2.line(frame, (5,line_pos), (1200,line_pos),(255,127,0),3)
    
    
    
    # drawing rectangles
    
    for c in contours:
        (x,y,w,h)=cv2.boundingRect(c)
        validate_counter = (w>=min_w_rect) and (h>=min_w_rect) 
        if validate_counter:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
  
            (cx,cy) =get_centroid(x,y,w,h)
            if cy<(offset+line_pos) and cy>(-offset+line_pos):
                counter+=1
                cv2.putText(frame, 'Vehicle No. ' + str(counter), (x, y), cv2.FONT_HERSHEY_SIMPLEX,  2,  (255, 0, 0), 2 , cv2.LINE_AA)
                cv2.line(frame, (5,line_pos), (1200,line_pos),(255,127,0),3)
    
            # Draw a circle of red color of thickness -1 px
            cv2.circle(frame, (cx,cy), 4, (255, 133, 233),   -1)
   
    # write on vehicle
           
    # Using cv2.putText() method
    cv2.putText(frame, 'Total Vehicle Count: ' + str(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,  2,  (255, 0, 0), 2 )
    cv2.imshow('frame', frame)
    # if key 13 is pressed (Enter is pressed)
    if cv2.waitKey(1)==13:
        break
print("Final: {}",counter)   
# destroy all widnows
# Python Opencv destroyAllWindows() function allows users to destroy or close all windows at any time after exiting the script. If you have multiple windows open at the same time and you want to close then you would use this function. It doesn’t take any parameters and doesn’t return anything. It is similar to destroyWindow() function but this function only destroys a specific window unlike destroyAllWindows().
cv2.destroyAllWindows()
# release video
cap.release()