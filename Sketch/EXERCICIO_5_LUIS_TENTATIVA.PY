#!/usr/bin/env python3 
import argparse
import cv2
import numpy as np
import time
import json
import random
from colorama import Fore,Style
from pprint import pprint

#func 5 em tempo real era top, da avaliacao (racio de pixeis pintados)

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
    parser.add_argument('-umm','--use_mirror_mode',action="store_true",help="With this mode, the camara and canva are mirrored, if that helps you")
    parser.add_argument('-np', '--numered_paint', help="Mode where you must paint according to the numbers", required=False, default='limits.json',action='store_true')
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

    # Board for mode numered paint
    if args['numered_paint'] is True:
        if args['use_mirror_mode']is True: #ambos os argumentos para numbered paint e mirror mode ligados
            tela = np.ones((size[1], size[0], 3), dtype = np.uint8)*255
            mirror_tela=cv2.flip(tela,1)
            #num_cols=int(input("Qual o numero de colunas desejado"))
            #num_rows=int(input("Qual o numero de linhas desejado"))

            cell_size_x = mirror_tela.shape[1] // 3 
            cell_size_y = mirror_tela.shape[0] // 2
            num_cols = 4    
            num_rows = 3

            zone_colors = {                    #In BGR
                                1: (0, 0, 255),  # Blue
                                2: (0, 255, 0),  # Green
                                3: (255, 0, 0)   # Red
                            }
        
            zones = np.zeros((num_rows, num_cols), dtype=int)
            print("Zone number 1 corresponds to the color: " + Fore.BLUE + "Blue" + Style.RESET_ALL)
            print("Zone number 2 corresponds to the color: " + Fore.GREEN + "Green" + Style.RESET_ALL)
            print("Zone number 3 corresponds to the color: " + Fore.RED + "Red" + Style.RESET_ALL)
            

            for row in range(num_rows):
                for col in range(num_cols):
                    zones[row, col] = random.randint(1, 3)

            def draw_cells(mirror_tela):
                for row in range(num_rows):
                    for col in range(num_cols):
                        x1 = col * cell_size_x
                        y1 = row * cell_size_y
                        x2 = x1 + cell_size_x
                        y2 = y1 + cell_size_y
                        cv2.rectangle(mirror_tela, (x1, y1), (x2, y2), (0, 0, 0), 2)
                        cell_number = zones[row, col]
                        # Calculates cell center
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        # Draw the number in the center of the cell
                        cv2.putText(mirror_tela, str(cell_number), (center_x - 10, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            draw_cells(mirror_tela)
        else:                                                                  #Caso o mirror mode não estiver ativo, segue normal
            tela = np.ones((size[1], size[0], 3), dtype = np.uint8)*255
            #num_cols=input("Qual o numero de colunas desejado")                         #tentar incrementar o numero de colunas
            #num_lines=input("Qual o numero de linhas desejado")

            cell_size_x = tela.shape[1] // 3 
            cell_size_y = tela.shape[0] // 2
            num_cols = 3
            num_rows = 2
        
            zones = np.zeros((num_rows, num_cols), dtype=int)

            zone_colors = {
                                1: (255, 0, 0),  # Red
                                2: (0, 255, 0),  # Green
                                3: (0, 0, 255)   # Blue
                            }
            
            print("Zone number 1 corresponds to the color: " + Fore.BLUE + "Blue" + Style.RESET_ALL)
            print("Zone number 2 corresponds to the color: " + Fore.GREEN + "Green" + Style.RESET_ALL)
            print("Zone number 3 corresponds to the color: " + Fore.RED + "Red" + Style.RESET_ALL)

            for row in range(num_rows):
                for col in range(num_cols):
                    zones[row, col] = random.randint(1, 3)

            def draw_cells(tela):
                for row in range(num_rows):
                    for col in range(num_cols):
                        x1 = col * cell_size_x
                        y1 = row * cell_size_y
                        x2 = x1 + cell_size_x
                        y2 = y1 + cell_size_y
                        cv2.rectangle(tela, (x1, y1), (x2, y2), (0, 0, 0), 2)
                        cell_number = zones[row, col]
                        # Calculates cell center
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        # Draw the number in the center of the cell
                        cv2.putText(tela, str(cell_number), (center_x - 10, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            draw_cells(tela)
    
     # Normal board
    else:
        tela = np.ones((size[1], size[0], 3), dtype = np.uint8)*255
        mirror_tela= cv2.flip(tela,1)


    num_sections=6


    black_tela = np.zeros((size[1], size[0], 3), dtype = np.uint8)
    mirror_black_tela = cv2.flip(black_tela,1)

    j = 0
    drawing_data['color'] = (255,0,0)
    #List for evaluation
    evaluation_results = []

    while(True):

        # load video
        ret, frame = video.read()

        # resizing video window
        frame = cv2.resize(frame, size) 
        mirror_frame= cv2.flip(frame,1)

        
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
            elif args["use_mirror_mode"]:
                mirror_tela = np.zeros((size[1], size[0], 3), dtype = np.uint8)
                print("Clearing canva!")
            else:
                tela = np.ones((size[1], size[0], 3), dtype = np.uint8)
                tela = 255*tela
                print("Clearing canva!") ##########problemas com o mirror
        elif k == 119:
            data = time.asctime( time.localtime(time.time()))
            data = data.replace(" ", "_")

            if args["video_draw"] is True:
                filename = 'drawing_' + data + '.png'
                cv2.imwrite(filename, saved_f)
                print(f"Image saved as {filename}")
            else:
                filename = 'drawing_' + data + '.png'
                cv2.imwrite(filename, tela)
                print(f"Image saved as {filename}")

        if ret == True:

            Segm = segmented(frame, limit_data['limits']['B']['min'], limit_data['limits']['G']['min'],
                             limit_data['limits']['R']['min'], limit_data['limits']['B']['max'], limit_data['limits']['G']['max'], limit_data['limits']['R']['max'])
            mirror_segm = cv2.flip(Segm,1)
            
            analysis = cv2.connectedComponentsWithStats(Segm, 4, cv2.CV_32S)
            mirror_analysis = cv2.connectedComponentsWithStats(mirror_segm, 4, cv2.CV_32S)


            (totalLabels, label_ids, values, centroid) = analysis
            (totalLabels, m_label_ids, values, m_centroid) = mirror_analysis

            # Initialize a new image to store  
            # all the output components 
            output = np.zeros(Segm.shape, dtype="uint8") 
            mirror_output = cv2.flip(output,1)
            
            
            # Loop through each component 
            for i in range(1, totalLabels): 
    
                # Area of the component 
                area = values[i, cv2.CC_STAT_AREA]  
                
                if args["use_mirror_mode"]:
                    if (area > 10000): 
                        m_componentMask = (m_label_ids == i).astype("uint8") * 255
                        mirror_output = cv2.bitwise_or(mirror_output, m_componentMask)
                        mirror_mask = mirror_output.astype(bool)
                    
                
                        #Channel division
                        mb, mg, mr = cv2.split(mirror_frame)


                        # Set color red in mask with channels
                        mb[mirror_mask] = 0
                        mg[mirror_mask] = 0
                        mr[mirror_mask] = 255
                
                        #merged image
                        mirror_frame = cv2.merge((mb,mg,mr))
                        #mirror_frame=cv2.flip(frame,1) ########

                        (x, y) = m_centroid[i]
                        x = int(x)
                        y = int(y)
                    
                        cv2.drawMarker(mirror_frame, (int(x),int(y)), color=[255, 0, 0], thickness=3, 
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
                                        pencil(mirror_black_tela, drawing_data, x, y)
                                    else:
                                        pencil(mirror_tela, drawing_data, x, y)

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
                                    pencil(mirror_black_tela, drawing_data, x, y)
                                else:
                                    pencil(mirror_tela, drawing_data, x, y)

                                drawing_data['previous_x'] = x
                                drawing_data['previous_y'] = y
                                                        
                        j = 1
                else:                                   #Se não se utilizar o mirror mode!!!
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
                        mirror_frame=cv2.flip(frame,1)

                        (x, y) = centroid[i]
                        x = int(x)
                        y = int(y)
                    
                        cv2.drawMarker(frame, (int(x),int(y)), color=[255, 0, 0], thickness=3, 
                                    markerType= cv2.MARKER_TILTED_CROSS, line_type=cv2.LINE_AA,
                                    markerSize=25)

                        if args["numered_paint"]:
                            if num_sections > 1:
                                zone_number = zones[y // cell_size_y, x // cell_size_x]
                                correct_color = zone_colors.get(zone_number)
                                print("teste1")
                                if correct_color:
                                    pixel_color = frame[y, x]
                                    is_correct_color = all(correct == pixel for correct, pixel in zip(correct_color, pixel_color))
                                    evaluation_results.append(is_correct_color)
                                    if is_correct_color:
                                        print(Fore.GREEN+"Acertar"+Style.RESET_ALL)
                                    else:
                                        print(Fore.RED+"Errar"+Style.RESET_ALL)





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
                                        #drawing_data['color'] = (255,0,0)       
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
                                    #drawing_data['color'] = (255,0,0)
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
                if args['use_mirror_mode'] is True:
                    img2gray = cv2.cvtColor(mirror_black_tela,cv2.COLOR_BGR2GRAY)
                    retg, mask2 = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask2)

                    # Now black-out the area of logo in ROI
                    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

                    # Take only region of logo from logo image.
                    img2_fg = cv2.bitwise_and(mirror_black_tela,mirror_black_tela,mask = mask2)
                    
                else:
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
                saved_f = frame.copy()


            if args["use_mirror_mode"]:
                cv2.imshow('Principal window', mirror_frame)
                cv2.imshow('Segmented Window', mirror_segm)
                cv2.imshow('Object', mirror_output)
                cv2.imshow('Painting Picture', mirror_tela)


            else:

                # showing the video
                cv2.imshow('Principal window', frame)

                #Showing changed Image
                cv2.imshow('Segmented Window', Segm)

                
                # showing painting board   
                cv2.imshow('Painting Picture', tela)


                # showing painting board
                cv2.imshow('Object', output)
        
       
    score = (evaluation_results.count(True) / len(evaluation_results)) * 100
    print(f"Evaluation Score: {score:.2f}%")
    
       
    # close all windows
    cv2.destroyAllWindows()

    # shut down video
    video.release()



if __name__ == '__main__':
    main()
