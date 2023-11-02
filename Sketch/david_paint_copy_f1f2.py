#!/usr/bin/env python3 
import argparse
import cv2
import numpy as np
import time
import json
from colorama import Fore,Style
from pprint import pprint

#func 4 em tempo real era top, da avaliacao (racio de pixeis pintados)

# Definition of onTrackBar function responsible for segmenting pixels
# in a given for R, G and B
# returns segmented image


#argumentos de posição da função estes não mudam
def segmented(img, min_B, min_G, min_R, max_B, max_G, max_R):

    # Creation of array with max and min of each channel BGR
    upper = np.array([max_B, max_G, max_R])
    lower = np.array([min_B, min_G, min_R])
    
	# Mask Creation
    Segm = cv2.inRange(img, lower, upper)

    return Segm

def pencil(img, drawing_data, x, y):

    if drawing_data['pencil_on'] == True:
        # cv2.circle(image_rgb, (x, y), 3, (255,255,255), -1)
        cv2.line(img, (drawing_data['previous_x'], drawing_data['previous_y']), (x,y), drawing_data['color'], drawing_data['thickness']) 

def calculate_distance(x1,y1,x2,y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def main():

    #### Initialization ####
    parser = argparse.ArgumentParser(description='Definition of test mode')
    parser.add_argument('-j', '--json', type=str, help='Full path to json file', required=False, 
                        default='limits.json')
    parser.add_argument('-cl','--command_list',action="store_true",help="Gives a list of all the typing commands")
    parser.add_argument('-usp','--use_shake_prevention',action="store_true",help="Advanced mode that corrects painting imperfections")
    parser.add_argument('-vd', '--video_draw', help='drawing in window of camera', required=False, default='limits.json',action='store_true')
    args = vars(parser.parse_args()) # creates a dictionary

    #### Dictionaries  ###
    drawing_data = {'pencil_on': False, 'previous_x': 0, 'previous_y': 0, 'color': (0,0,0), 'thickness' : 2}
    with open(args['json'], "r") as json_file:
        limit_data = json.load(json_file)
    

    #Showing the list of commands in the beggining of the code 
    if args["command_list"]:
        command_list = {
        'r': "Red color pencil",
        'g': "Green color pencil",
        'b': "Blue color pencil",
        'e': "Eraser selected",
        'c': "Clearing canva",
        'w': "Save the image you just painted",
        'q': "Exits Program",
        '+': "Increases pencil thickness",
        '-': "Decreases pencil thickness",
        }   
        pprint(command_list)
    
    #####################podemos fazer isto com um argumento e fazer um colorama disto

    #Check if value in json file is numeric
    for channel in ['B', 'G', 'R']:
        for limit_type in ['min', 'max']:
            value = limit_data['limits'][channel][limit_type]
            if isinstance(value, (int, float)):
                limit_data['limits'][channel][limit_type] = value

    # Create an object to read  
    # from camera 
    video = cv2.VideoCapture(0)

    # check if camera 
    # is opened previously or not 
    if (video.isOpened() == False):  
        print("Error reading video file") 

    #windows names
    cv2.namedWindow('Principal window')
    cv2.namedWindow('Painting Picture')
    cv2.namedWindow('Segmented Window')
    cv2.namedWindow('Object')

    # painting board creation
    
    size = (1000, 600)
    tela = np.ones((size[1], size[0], 3), dtype = np.uint8)*255
    
    black_tela = np.zeros((size[1], size[0], 3), dtype = np.uint8)

    j = 0

    while(True):

        # load video
        ret, frame = video.read()

        # resizing video window
        frame = cv2.resize(frame, size) 
        
        # break if user press q (quit) or w to write values on file and drawing commands
        k = cv2.waitKey(1) & 0xFF
        if k == 113:
            print("q key pressed, exiting...")
            break
        elif k == 114:
            drawing_data["color"] = (0,0,255)
            print("Pencil color changed to "+ Fore.RED + "red" + Style.RESET_ALL) 
        elif k == 103:
            drawing_data["color"] = (0,255,0)
            print("Pencil color changed to " + Fore.GREEN + "green" + Style.RESET_ALL)
        elif k == 98:
            drawing_data["color"] = (255, 0, 0)
            print("Pencil color changed to " + Fore.BLUE + "blue" + Style.RESET_ALL)
        elif k == 101:
            drawing_data["color"] = (255,255,255)
            print("Eraser mode selected")
        elif k == 43:
            drawing_data["thickness"] += 1
            print("Incresing pencil thickness to ",drawing_data['thickness'])
        elif k == 45:
            if drawing_data["thickness"] >= 2:
                drawing_data["thickness"] -= 1
                print("Decreasing pencil thickness to ",drawing_data['thickness'])   
        elif k == 99:
            if args["video_draw"]:
                black_tela = np.zeros((size[1], size[0], 3), dtype = np.uint8)
                print("Clearing canva!")
            else:
                tela = np.ones((size[1], size[0], 3), dtype = np.uint8)
                tela = 255*tela
                print("Clearing canva!")
        elif k == 119:
            data = time.asctime( time.localtime(time.time()))
            data = data.replace(" ", "_")

            if args["video_draw"]:
                filename = 'drawing_' + data + '.png'
                cv2.imwrite(filename, black_tela)
                print(f"Image saved as {filename}")

            else:
                filename = 'drawing_' + data + '.png'
                cv2.imwrite(filename, tela)
                print(f"Image saved as {filename}")

        if ret == True:

            Segm = segmented(frame, limit_data['limits']['B']['min'], limit_data['limits']['G']['min'],
                             limit_data['limits']['R']['min'], limit_data['limits']['B']['max'], limit_data['limits']['G']['max'], limit_data['limits']['R']['max'])
            
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


                    if args['use_shake_prevention']:
                        if j == 0:
                            drawing_data['previous_x'] = x
                            drawing_data['previous_y'] = y

                        else:
                            distance = calculate_distance(x, y, drawing_data['previous_x'], drawing_data['previous_y'])
                            tolerable_distance = 100
                            if distance <= tolerable_distance:
                                drawing_data['thickness'] = drawing_data['thickness']
                                drawing_data['pencil_on'] = True
                                drawing_data['color'] = drawing_data['color']
                                if args['video_draw'] is True:
                                    drawing_data['color'] = (255,0,0)       
                                    pencil(black_tela, drawing_data, x, y)
                                else:
                                    pencil(tela, drawing_data, x, y)

                            drawing_data['previous_x'] = x
                            drawing_data['previous_y'] = y
                    else:
                        if j == 0:
                            drawing_data['previous_x'] = x
                            drawing_data['previous_y'] = y

                        else:
                            drawing_data['thickness'] = drawing_data['thickness']
                            drawing_data['pencil_on'] = True
                            drawing_data['color'] = drawing_data['color']

                            if args['video_draw'] is True:
                                drawing_data['color'] = (255,0,0)
                                pencil(black_tela, drawing_data, x, y)
                            else:
                                pencil(tela, drawing_data, x, y)

                            drawing_data['previous_x'] = x
                            drawing_data['previous_y'] = y
                                                    
                    j = 1

            if args['video_draw'] is True:
                # I want to put logo on top-left corner, So I create a ROI
                rows,cols,channels = frame.shape
                roi = frame[0:rows, 0:cols]

                # Now create a mask of logo and create its inverse mask also
                img2gray = cv2.cvtColor(black_tela,cv2.COLOR_BGR2GRAY)
                retg, mask2 = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask2)

                # Now black-out the area of logo in ROI
                img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

                # Take only region of logo from logo image.
                img2_fg = cv2.bitwise_and(black_tela,black_tela,mask = mask2)
                # Put logo in ROI and modify the main image
                dst = cv2.add(img1_bg,img2_fg)
                frame[0:rows, 0:cols ] = dst






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



if __name__ == '__main__':
    main()
