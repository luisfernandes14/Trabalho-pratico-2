#!/usr/bin/env python3
import cv2
import numpy as np

X = 0
Y = 0

X2 = 0
Y2 = 0
i = 0

def circles_or_squares(tela, x,y, x2, y2,color):

    cv2.ellipse(tela,(x,y),(x2,y2),0,0,360,color,2)

    cv2.imshow('drawing', tela)
    return tela

def mouse_click(event, x, y,  flags, param): 
    
    # check if left mouse button was clicked 
    if event == cv2.EVENT_LBUTTONDOWN:
        global X
        X = x
        global Y
        Y = y
        global i
        i += 1
        if i == 2:
            i = 0
    
    if i != 0:
        global X2
        X2 = x
        global Y2
        Y2 = y 

size = (1000, 600)
tela = (np.ones((size[1], size[0], 3), dtype = np.uint8))*255
tela2 = np.zeros((size[1], size[0], 3), dtype = np.uint8)
tela3 = tela

cv2.namedWindow("drawing")
# Calling mouse_click using setMouseCallback to print coordinates
cv2.setMouseCallback("drawing", mouse_click)
color = (255,0,0)
j = 0

while (True):
    if i == 1:
        
        tela = cv2.bitwise_or(tela3, tela)
        imgray = cv2.cvtColor(tela,cv2.COLOR_BGR2GRAY)
        rett3, mask2 = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY)
        cv2.imshow('teste', mask2)
        tela = circles_or_squares(tela, X, Y, X2, Y2,color)
        
        j = 1
           
    else:
        # I want to put logo on top-left corner, So I create a ROI
        rows,cols,channels = tela3.shape
        roi = tela3[0:rows, 0:cols]

        # Now create a mask of logo and create its inverse mask also
        img2gray = cv2.cvtColor(tela,cv2.COLOR_BGR2GRAY)
        retg, mask2 = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask2)

        #cv2.imshow('teste', mask2)

        # Now black-out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(tela,tela,mask = mask2)
        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg,img2_fg)
        tela3[0:rows, 0:cols ] = dst
        cv2.imshow('tela', tela3)
        if j == 1: color = (0,255,0)  

    if i == 2:
        print('hello')
        circles_or_squares(tela, X, Y, X2, Y2)
    
    
    

    k = cv2.waitKey(1) & 0xFF
    if k == 113:
        print("q key pressed, exiting...")
        break


