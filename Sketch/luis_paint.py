#!/usr/bin/env python3 
import argparse
import cv2
import numpy as np
from colorama import Fore,Back,Style
from datetime import datetime

# Definition of onTrackBar function responsible for segmenting pixels
# in a given for R, G and B
# returns segmented image
def segmented(img, lim_data):

    # Creation of array with max and min of each channel BGR
    upper = np.array([lim_data["max_B"],lim_data["max_G"], lim_data["max_R"]])
    lower = np.array([lim_data["min_B"], lim_data["min_G"], lim_data["min_R"]])
    
	# Mask Creation
    Segm = cv2.inRange(img, lower, upper)

    return Segm
    

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
    # num = []
    # i = 0

    ###########################
    #Dicionários


    lim_data = {"max_B":255,"min_B":0,"max_G":255,"min_G":0,"max_R":255,"min_R":0,"i":0}
    drawing_data={"previous_x":0,"previous_y":0,"color":(0,0,0),"thickness":1}


    # 3 for cicles to check if a char is numeric
    # muitos ciclos for, tentar outra maneira !!!!
    # elevado custo computacional !!!
    
    #ideia, na adquirição do j.son ser obrigatório ser numericos

    for line in f:
        for word in line.split():
            n = ''
            for char in word:
                if char.isnumeric():
                    n += char
            if n != '':
                lim_data["i"]=int(n)  #dict has no atribute to insert
                lim_data["i"]=lim_data["i"]+1

    # putting all the limits in the right place         
    
    # max_B = num[0]
    # min_B = num[1]
    # max_G = num[2] 
    # min_G = num[3]
    # max_R = num[4]
    # min_R = num[5]

    # Create an object to read  
    # from camera 
    video = cv2.VideoCapture(0)

    # check if camera 
    # is opened previously or not 
    if (video.isOpened() == False):  
        print("Error reading video file") 

    # windows size
    size = (1000, 600)

    #windows names
    cv2.namedWindow('Principal window')
    cv2.namedWindow('Painting Picture')
    cv2.namedWindow('Segmented Window')
    cv2.namedWindow('Object')

    # painting board creation
    tela = np.ones((size[1], size[0], 3), dtype = np.uint8)
    tela.fill(255)

    while(True):

        # load video
        ret, frame = video.read()

        # resizing video window
        frame = cv2.resize(frame, size) 
        
        #Comandos por teclado

        # break if user press q (quit) or w to write values on file
        k = cv2.waitKey(1) & 0xFF
        if k == 113 or k == 119:
            if k == 113:
                print("q key pressed, exiting...")
            else:
                print("w key pressed, saving limits!")
            break
        if k == 114: #r is pressed
            print("r key pressed, color changed to " + Fore.RED + "red" + Style.RESET_ALL)
            drawing_data["color"] = (0,0,255)
        if k == 103: #g is pressed
            print("g key pressed, color changed to " + Fore.GREEN + "green" + Style.RESET_ALL)
            drawing_data["color"] = (0,255,0)
        if k == 98: #b is pressed
            print("b key pressed, color changed to " + Fore.BLUE + "blue" +Style.RESET_ALL)
            drawing_data["color"] = (255,0,0)
        if k == 43: #"+" is pressed
            print("'x' key pressed, increasing line thickness")
            drawing_data["thickness"] = drawing_data["thickness"] + 1
        if k == 45: #"-" is pressed
            print("'-' key pressed, decreasing line thickness")
            drawing_data["thickness"] = drawing_data["thickness"] - 1
        if k == 99: #c is pressed, "tela" is refereshed
            cv2.imshow('Painting Picture', tela)
            tela.fill(255)                      #repintar a tela toda de branco
        if k == 119: # clicando no W, faz um save da imagem da tela e guarda num ficheiro png
            momento = datetime.now()
            data_hora = momento.strftime("%a_%b_%d_%H:%M:%S_%Y")
            ficheiro_save = f"painting_{data_hora}.png"
            cv2.imwrite(ficheiro_save, tela)
            print(f"Image saved as {ficheiro_save}")




        
        if ret == True:

            Segm = segmented(frame, lim_data)
            
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
                    cv2.drawMarker(frame, (int(x), int(y)), color=[255, 0, 0], thickness=3, 
                                   markerType= cv2.MARKER_TILTED_CROSS, line_type=cv2.LINE_AA,
                                   markerSize=25)

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
