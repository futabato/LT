import cv2
import numpy as np

def main():
    src_image = cv2.imread("./image/gachapin.png")
    back_image = cv2.imread("./image/weather.png")
    back_image = cv2.resize(back_image, dsize=(800, 800))
    
    # mask of green
    hsv = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    hsv_min = np.array([30, 60, 0])
    hsv_max = np.array([90, 255, 255])
    
    
    src_image_mask = cv2.inRange(hsv, hsv_min, hsv_max)
    object_image = cv2.bitwise_and(back_image, back_image, mask = src_image_mask)
    
    extraction = cv2.bitwise_not(src_image_mask)
    embedded = cv2.bitwise_and(src_image, src_image, mask = extraction)

    ChromaKey = cv2.bitwise_or(object_image, embedded)
    
    
    cv2.imshow("ChromaKey", extraction)

    while True:
        key = cv2.waitKey(1)&0xff
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()