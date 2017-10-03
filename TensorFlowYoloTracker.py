'''
Created on Sep 23, 2017

@author: asad
'''
from darkflow.net.build import TFNet
import cv2
import pickle

class TensorFlowYoloTracker(object):
    
    '''
    class attributes
    '''
    keyframe_interval = 3
    min_confidence = 0.4
    box_color = (0,0,255)

    def __init__(self):
        '''
        Constructor
        '''
        #options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "bin/tiny-yolo-voc.weights", "threshold": 0.1, "gpu": 1.0}
        options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}
        
        self.tfnet = TFNet(options)
        self.frame_count = 0 
        self.tracker = cv2.MultiTracker_create()
     
    '''
    YOLO provides multiple detectctions that might be interesting for self driving cars
    like person bicycle bus truck botorbike traffic lights, pedestrian etc. 
    '''     
    def is_interesting(self, label):
        if (
            label == "car" or label == "truck" or
            label == "bicycle" or label == "bus" or
            label == "botorbike" or label == "person"
            ):
            return True
        
        return False
        
    # Expects undistored images
    def process_frame(self, image):
    
        result = self.tfnet.return_predict(image)
        
        bboxs = []
        if self.frame_count % TensorFlowYoloTracker.keyframe_interval == 0:           
            self.tracker = cv2.MultiTracker_create()
            for box in result:
                label = box['label']
                confidence = box['confidence']
       
                if self.is_interesting(label):
                    if confidence > TensorFlowYoloTracker.min_confidence:                            
                        x1 = box['topleft']['x']
                        y1 = box['topleft']['y']
                        x2 = box['bottomright']['x']
                        y2 = box['bottomright']['y']
                        bbox = (x1,y1,x2-x1,y2-y1)
                        self.tracker.add(cv2.TrackerMIL_create(), image, bbox)
                        bboxs.append(bbox)        
                
        else:
            _, bboxs = self.tracker.update(image)
                
        self.frame_count+= 1
         
        return bboxs
      
    def draw_boxes(self, image, boxes):
        for newbox in boxes:
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(image, p1, p2, TensorFlowYoloTracker.box_color,3)
