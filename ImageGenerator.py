'''
Created on Oct 1, 2017

@author: asad
'''

from darkflow.net.build import TFNet
import cv2


def main():
    options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}
    
    tfnet = TFNet(options)
    
    image = cv2.imread("./test_images/test1.jpg")
    result = tfnet.return_predict(image)
    
    #print(result)
    
    for box in result:
        print (box)
        x1 = box['topleft']['x']
        y1 = box['topleft']['y']
        x2 = box['bottomright']['x']
        y2 = box['bottomright']['y']
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),3)

    cv2.imwrite('out.png',image)

if __name__ == '__main__':
    main()