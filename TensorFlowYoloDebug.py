'''
Created on Oct 1, 2017

@author: Asad Zia
'''

from darkflow.net.build import TFNet
import cv2
import glob
import pickle
'''
YOLO provides multiple detectctions that might be interesting for self driving cars
like person bicycle bus truck botorbike traffic lights, pedestrian etc. 
'''     
def is_interesting(label):
    if (
        label == "car" or label == "truck" or
        label == "bicycle" or label == "bus" or
        label == "botorbike" or label == "person"
        ):
        return True
    
    return False
    
def main():
    
    dest_pickle = pickle.load( open("camera_cal/wide_dist_pickle.p", "rb"  ))
    mtx = dest_pickle["mtx"]
    dist = dest_pickle["dist"]
        
    options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}
    tfnet = TFNet(options)
    
    images = glob.glob( './test_images/test*.jpg' )
    
    for idx, fname in enumerate(images):
        image = cv2.imread(fname)
        
        image = cv2.undistort(image, mtx, dist, None, mtx)
        
        result = tfnet.return_predict(image)
        for box in result:
            print (box)
            x1 = box['topleft']['x']
            y1 = box['topleft']['y']
            x2 = box['bottomright']['x']
            y2 = box['bottomright']['y']
            label = box['label']
            confidence = box['confidence']
   
            if is_interesting(label):
                if confidence > .1:              
                    cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),3)
                    label = str(int(confidence*100))
                    cv2.putText(
                        image, label, (x1, y1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * 600, (0,255,0),
                        1)        
            
        cv2.imwrite('./output_images/out' + str(idx) + '.jpg',image)  

if __name__ == '__main__':
    main()