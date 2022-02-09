import numpy as np
import cv2
from mss import mss
from PIL import Image
import time



mon = {'left': 0, 'top':100, 'width': 1200, 'height': 700}
currTime = time.time()
with mss() as sct:
    while True:
        screenShot = sct.grab(mon)
        img = Image.frombytes(
            'RGB', 
            (screenShot.width, screenShot.height), 
            screenShot.rgb, 
        )

        image = np.array(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        lower = np.array([105,175,150])
        upper = np.array([110,200,180])
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower, upper)


        res = cv2.bitwise_and(image,image, mask = mask)

        

        median = cv2.medianBlur(res,15)
        cv2.imshow('Median Blur',median)
        cv2.imshow('test', res)
        cv2.imshow('edges',mask)
        print(time.time() - currTime)
        currTime = time.time()

    

        if cv2.waitKey(33) & 0xFF in ( ord('q'), 27, ):
            cv2.destroyAllWindows()
            break