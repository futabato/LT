import cv2
import numpy as np

def main():
    image = cv2.imread("./image/gachapin.png")
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_min = np.array([30, 64, 0])
    hsv_max = np.array([90, 255, 255])
    
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    mask_process = cv2.bitwise_and(image, image, mask = mask)
    
    cv2.imshow("ChromaKey", mask_process)

    while True:
        key = cv2.waitKey(1)&0xff
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()