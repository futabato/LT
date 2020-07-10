import cv2
import numpy as np

def main():
    image = cv2.imread("./image/gachapin.png")
    back = cv2.imread("./image/weather.png")
    back = cv2.resize(back, dsize=(800, 800))
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_min = np.array([30, 64, 0])
    hsv_max = np.array([90, 255, 255])
    
    image_mask = cv2.inRange(hsv, hsv_min, hsv_max)
    extraction = cv2.bitwise_not(image_mask)
    
    embedded = cv2.bitwise_and(back, back, mask = image_mask)

    frame = cv2.bitwise_and(back, back, mask = extraction)
    ChromaKey = cv2.bitwise_or(embedded, frame)
    
    #extraction = cv2.bitwise_and(image, back, mask = image_mask)
    #mask_process = cv2.bitwise_or(extraction, back)
    
    cv2.imshow("ChromaKey", extraction)

    while True:
        key = cv2.waitKey(1)&0xff
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()