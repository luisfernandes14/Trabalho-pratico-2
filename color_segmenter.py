#!/usr/bin/env python3
import cv2
import numpy as np
from functools import partial



def onTrackbar(val,image):
     

    if image is not None:
        lower_b = cv2.getTrackbarPos('Min B', 'C/ Trackbar')
        upper_b = cv2.getTrackbarPos('Max B', 'C/ Trackbar')
        lower_g = cv2.getTrackbarPos('Min G', 'C/ Trackbar')
        upper_g = cv2.getTrackbarPos('Max G', 'C/ Trackbar')
        lower_r = cv2.getTrackbarPos('Min R', 'C/ Trackbar')
        upper_r = cv2.getTrackbarPos('Max R', 'C/ Trackbar')

        lower_bound = np.array([lower_b, lower_g, lower_r])
        upper_bound = np.array([upper_b, upper_g, upper_r])
        mask = cv2.inRange(image, lower_bound, upper_bound)

        result = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow("C/ Trackbar", result)

def main():
    # initial setup
    capture = cv2.VideoCapture(0)
    window_name = 'window'
    image = None
    val = 0
    cv2.namedWindow('C/ Trackbar', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('Min B', 'C/ Trackbar', 0, 255, partial(onTrackbar,image=image))
    cv2.createTrackbar('Max B', 'C/ Trackbar', 255, 255, partial(onTrackbar,image=image))
    cv2.createTrackbar('Min G', 'C/ Trackbar', 0, 255, partial(onTrackbar,image=image))
    cv2.createTrackbar('Max G', 'C/ Trackbar', 255, 255, partial(onTrackbar,image=image))
    cv2.createTrackbar('Min R', 'C/ Trackbar', 0, 255, partial(onTrackbar,image=image))
    cv2.createTrackbar('Max R', 'C/ Trackbar', 255, 255, partial(onTrackbar,image=image))
  
    while True:
        _, image = capture.read()  # get an image from the camera
        cv2.imshow('Original', image)
        onTrackbar(val,image)
        #print dos valores de RGB ao momento
        if cv2.waitKey(1) & 0xFF == ord('q'): # quit program
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
