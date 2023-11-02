#!/usr/bin/env python3 
import cv2 
import numpy as np
import argparse
import json

# Definition of onTrackBar function responsible for segmenting pixels
# in a given for R, G and B
def onTrackbar(image_rgb, min_B, min_G, min_R, max_B, max_G, max_R):

    # Creation of array with max and min of each channel BGR
    upper = np.array([max_B, max_G, max_R])
    lower = np.array([min_B, min_G, min_R])
    
	# Mask Creation
    image_rgb = cv2.inRange(image_rgb, lower, upper)
    
    #Showing changed Image
    cv2.imshow('TrackBar', image_rgb)

# nothing function, responsible to letting createTrackBar continue  
def nothing(x):
    pass
    

def main():

    # Create an object to read  
    # from camera 
    video = cv2.VideoCapture(0)

    # check if camera 
    # is opened previously or not 
    if (video.isOpened() == False):  
        print("Error reading video file") 

    size = (1000, 600) 
	
	#window name
    cv2.namedWindow('TrackBar')

	# creating trackbars for each min max channel
    cv2.createTrackbar('min B', 'TrackBar', 0, 255, nothing) 
    cv2.createTrackbar('max B', 'TrackBar', 0, 255, nothing)
    cv2.createTrackbar('min G', 'TrackBar', 0, 255, nothing) 
    cv2.createTrackbar('max G', 'TrackBar', 0, 255, nothing)
    cv2.createTrackbar('min R', 'TrackBar', 0, 255, nothing) 
    cv2.createTrackbar('max R', 'TrackBar', 0, 255, nothing)

    while(True):

        # load video
        ret, frame = video.read()

        # resizing video window
        frame = cv2.resize(frame, size) 

        #saving position of each trackbar
        max_BH = cv2.getTrackbarPos("max B", "TrackBar")
        min_BH = cv2.getTrackbarPos("min B", "TrackBar")
        max_GS = cv2.getTrackbarPos("max G", "TrackBar")
        min_GS = cv2.getTrackbarPos("min G", "TrackBar")
        max_RV = cv2.getTrackbarPos("max R", "TrackBar")
        min_RV = cv2.getTrackbarPos("min R", "TrackBar")
        
        # break if user press q (quit) or w to write values on file
        k = cv2.waitKey(1) & 0xFF
        if k == 113 or k == 119:
            if k == 113:
                print("q key pressed, exiting...")
            else:
                print("w key pressed, saving limits!")
            break
        
        if ret == True:
            
            # showing the video
            cv2.imshow('Frame', frame)

            # ontrackbar function call to update the frame
            onTrackbar(frame, min_BH, min_GS, min_RV, max_BH, max_GS, max_RV)
    
    # close all windows
    cv2.destroyAllWindows()

    # shut down video
    video.release()

    #if user press w, program must store R,G and B values on file
    if k == 119:
        # Dictionary to store min, max of R, G and B
        limits = { 'limits': {'B': {'max': max_BH, 'min': min_BH},
                'G': {'max': max_GS, 'min': min_GS},
                'R': {'max': max_RV, 'min': min_RV}}}
    	
        ### file creation and recording values ###
        file_name = 'limits.json'
        with open(file_name, 'w') as file_handle:
            print('writing dictionary limits to file ' + file_name)
            json.dump(limits, file_handle)
    
if __name__ == '__main__':
    main()