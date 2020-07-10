from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time


video_capture = cv2.VideoCapture(0)

embedded = cv2.imread("./image/weather.png")

video_capture.set(cv2.CAP_PROP_FPS, 30)

width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

def ExtractionColor(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_min = np.array([30, 64, 0])
    hsv_max = np.array([90, 255, 255])
    
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    mask_process = cv2.bitwise_and(frame, frame, mask = mask)
    mask_not = cv2.bitwise_not(mask)
    frame_cut = cv2.bitwise_and(frame, frame, mask= mask_not)
    
    ChromaKey = cv2.bitwise_or(mask_process, frame_cut)
    
    return ChromaKey



def main():
    while True:
        ret, frame = video_capture.read()
        
        #ExtractionColor(frame)

        cv2.imshow("Video", ExtractionColor(frame))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()