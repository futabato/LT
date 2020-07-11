from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time


def ExtractionColor(frame, back_image):
    back_image = cv2.resize(back_image, dsize=(800, 800))
    frame = cv2.resize(frame, dsize=(800, 800))
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_min = np.array([30, 64, 0])
    hsv_max = np.array([90, 255, 255])
    
    src_image_mask = cv2.inRange(hsv, hsv_min, hsv_max)
    object_image = cv2.bitwise_and(back_image, back_image, mask = src_image_mask)
    extraction = cv2.bitwise_not(src_image_mask)
    embedded = cv2.bitwise_and(frame, frame, mask = extraction)
    ChromaKey = cv2.bitwise_or(object_image, embedded)
    
    return ChromaKey



def main():
    video_capture = cv2.VideoCapture(0)

    back_image = cv2.imread("./image/weather.png")

    video_capture.set(cv2.CAP_PROP_FPS, 30)
    
    while True:
        ret, frame = video_capture.read()
        
        cv2.imshow("ChromaKey", ExtractionColor(frame, back_image))
    
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()