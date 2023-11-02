#!/usr/bin/env python3 
import argparse
import cv2
import numpy as np
import time

# Definition of onTrackBar function responsible for segmenting pixels
# in a given for R, G and B
# returns segmented image
def segmented(img, min_B, min_G, min_R, max_B, max_G, max_R):

    # Creation of array with max and min of each channel BGR
    upper = np.array([max_B, max_G, max_R])
    lower = np.array([min_B, min_G, min_R])
    
	# Mask Creation
    Segm = cv2.inRange(img, lower, upper)

    return Segm

def pencil(img, drawing, x, y):

    if drawing['pencil_on'] == True:
        # cv2.circle(image_rgb, (x, y), 3, (255,255,255), -1)
        cv2.line(img, (drawing['previous_x'], drawing['previous_y']), (x,y), drawing['color'], drawing['thickness']) 


    

def main():

    #### Initialization ####
    parser = argparse.ArgumentParser(description='Definition of test mode')
    parser.add_argument('-j', '--json', type=str, help='Full path to json file', required=False, 
                        default='limits.json')

    args = vars(parser.parse_args()) # creates a dictionary

    ##### Loading values of limits #####
    # reading file in dictionary args
    f = open(args['json'], "r")

    # defining variables num and i, for array of limits 
    num = []
    i = 0

    # 3 for cicles to check if a char is numeric
    # muitos ciclos for, tentar outra maneira !!!!
    # elevado custo computacional !!!
    for line in f:
        for word in line.split():
            n = ''
            for char in word:
                if char.isnumeric():
                    n += char
            if n != '':
                num.insert(i, int(n))
                i = i + 1

    # putting all the limits in the right place         
    max_B = num[0]
    min_B = num[1]
    max_G = num[2] 
    min_G = num[3]
    max_R = num[4]
    min_R = num[5]

    # Create an object to read  
    # from camera 
    video = cv2.VideoCapture(0)

    # check if camera 
    # is opened previously or not 
    if (video.isOpened() == False):  
        print("Error reading video file") 

    # windows size
    size = (1000, 600)

    color = (255,0,0)

    L = 2

    drawing = {'pencil_on': False, 'previous_x': 0, 'previous_y': 0, 'color': color, 'thickness' : L}

    #windows names
    cv2.namedWindow('Principal window')
    cv2.namedWindow('Painting Picture')
    cv2.namedWindow('Segmented Window')
    cv2.namedWindow('Object')

    # painting board creation
    tela = np.ones((size[1], size[0], 3), dtype = np.uint8)
    tela = 255*tela

    j = 0

    while(True):

        # load video
        ret, frame = video.read()

        # resizing video window
        frame = cv2.resize(frame, size) 
        
        # break if user press q (quit) or w to write values on file
        k = cv2.waitKey(1) & 0xFF
        if k == 113:
            print("q key pressed, exiting...")
            break
        elif k == 114:
            color = (0,0,255)
        elif k == 103:
            color = (0,255,0)
        elif k == 98:
            color = (255, 0, 0)
        elif k == 43:
            L += 1
        elif k == 45:
            if L >= 2:
                L -= 1
        elif k == 99:
            tela = np.ones((size[1], size[0], 3), dtype = np.uint8)
            tela = 255*tela
        elif k == 119:
            break

        if ret == True:

            Segm = segmented(frame, min_B, min_G, min_R, max_B, max_G, max_R)
            
            analysis = cv2.connectedComponentsWithStats(Segm, 4, cv2.CV_32S)

            (totalLabels, label_ids, values, centroid) = analysis

            # Initialize a new image to store  
            # all the output components 
            output = np.zeros(Segm.shape, dtype="uint8") 
            
            # Loop through each component 
            for i in range(1, totalLabels): 
    
                # Area of the component 
                area = values[i, cv2.CC_STAT_AREA]  
                
                if (area > 10000): 
                    componentMask = (label_ids == i).astype("uint8") * 255
                    output = cv2.bitwise_or(output, componentMask)
                    mask = output.astype(bool)
            
                    #Channel division
                    b, g, r = cv2.split(frame)

                    # Set color red in mask with channels
                    b[mask] = 0
                    g[mask] = 0
                    r[mask] = 255

                    #merged image
                    frame = cv2.merge((b,g,r))

                    (x, y) = centroid[i]
                    x = int(x)
                    y = int(y)
                    cv2.drawMarker(frame, (int(x), int(y)), color=[255, 0, 0], thickness=3, 
                                   markerType= cv2.MARKER_TILTED_CROSS, line_type=cv2.LINE_AA,
                                   markerSize=25)

                    if j == 0:
                        drawing['previous_x'] = x
                        drawing['previous_y'] = y

                    else:
                        drawing['thickness'] = L
                        drawing['pencil_on'] = True
                        drawing['color'] = color

                        pencil(tela, drawing, x, y)

                        drawing['previous_x'] = x
                        drawing['previous_y'] = y
                    
                    j = 1



            # showing the video
            cv2.imshow('Principal window', frame)

            #Showing changed Image
            cv2.imshow('Segmented Window', Segm)

            # showing painting board
            cv2.imshow('Painting Picture', tela)

            # showing painting board
            cv2.imshow('Object', output)
        
    # close all windows
    cv2.destroyAllWindows()

    # shut down video
    video.release()

    if k == 119:

        data = time.asctime( time.localtime(time.time()))
        data = data.replace(" ", "_")

        filename = 'drawing_' + data + '.png'
        cv2.imwrite(filename, tela)


if __name__ == '__main__':
    main()