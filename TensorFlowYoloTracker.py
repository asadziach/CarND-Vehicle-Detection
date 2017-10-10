'''
Created on Sep 23, 2017

@author: Asad Zia
'''
from darkflow.net.build import TFNet
#from y2dk_wrapper import Y2dk

class TensorFlowYoloTracker(object):
    
    '''
    class attributes
    '''
    min_confidence = 0.4

    def __init__(self):
        '''
        Constructor
        '''
        #YOLO COCO
        options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}
        #Y2dk YOLO COCO
        #options = {"model": "model_data/yolo.h5", "threshold": 0.1, "anchors_path":"model_data/yolo_anchors.txt","classes_path":"model_data/coco_classes.txt"}
        self.tfnet = TFNet(options) # Change to Y2dk(options) for using Y2dk
        self.frame_count = 0 

        self.objects = []
        self.object_candidates = []
     
    '''
    YOLO provides multiple detectctions that might be interesting for self driving cars
    like person bicycle bus truck motorbike traffic lights, pedestrian etc. 
    '''     
    def is_interesting(self, label):
        if (
            label == "car" or label == "truck" or
            label == "bicycle" or label == "bus" or
            label == "motorbike" or label == "train" or
            label == "traffic light"
            ):
            return True
        
        return False
        
    # Expects undistored images
    def process_frame(self, image):
    
        result = self.tfnet.return_predict(image)
        
        bboxs = []
        info = [] 
              
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
                    
                    bboxs.append(bbox)
                    info.append((label,confidence))        
                
        self.frame_count+= 1
        return bboxs, info

