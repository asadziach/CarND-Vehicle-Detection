'''
Created on Sep 23, 2017

@author: asad
'''
from darkflow.net.build import TFNet
from y2dk_wrapper import Y2dk
import cv2

class TensorFlowYoloTracker(object):
    
    '''
    class attributes
    '''
    keyframe_interval = 3
    min_confidence = 0.4
    box_color = (255,128,128)
    line_color = (255,255,0)
    min_tracking_iou = 0.4
    remember_threshold = 10 #frames
    drawing_threshold = 5000 # pxels

    def __init__(self):
        '''
        Constructor
        '''
        #options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "bin/tiny-yolo-voc.weights", "threshold": 0.1, "gpu": 1.0}
        options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}
        #options = {"model": "model_data/yolo.h5", "threshold": 0.1, "anchors_path":"model_data/yolo_anchors.txt","classes_path":"model_data/coco_classes.txt"}
        self.tfnet = TFNet(options)
        self.frame_count = 0 
        self.tracker = cv2.MultiTracker_create()
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
                    self.tracker.add(cv2.TrackerTLD_create(), image, bbox)
                    bboxs.append(bbox)
                    info.append((label,confidence))        
                

        #_, bboxs = self.tracker.update(image)
        
        self.update_objects(bboxs, info)
                
        self.frame_count+= 1
         
    def draw_boxes(self, image):
        for obj in self.objects:
            newbox = obj.bbox
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            
            cv2.rectangle(image, p1, p2, TensorFlowYoloTracker.box_color,3)
            cv2.putText(
                image, obj.label, (p1[0], p1[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * 600, TensorFlowYoloTracker.box_color,
                1)
            #unreliable detections on apposite site of road
            #TODO comes from object tracker
        for newbox in self.object_candidates:
            p1 = (int(newbox[0]), int(newbox[1] + newbox[3]/2))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]/2))
            cv2.arrowedLine(image, p2, p1, TensorFlowYoloTracker.line_color,3)
                
        
    #Intersection over Union (IoU) TODO namedtuple
    def calculate_iou(self, boxA, boxB):
        x1 = boxA[0]
        y1 = boxA[1]
        x2 = x1 + boxA[2]
        y2 = y1 + boxA[3]
        x3 = boxB[0]
        y3 = boxB[1]
        x4 = x3 + boxB[2]
        y4 = y3 + boxB[3]
        
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(x1, x3)
        yA = max(y1, y3)
        xB = min(x2, x4)
        yB = min(y2, y4)
     
        # compute the area of intersection rectangle
        interArea = (xB - xA + 1) * (yB - yA + 1)
     
        # compute the area
        boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
        boxBArea = (boxB[2] + 1) * (boxB[3] + 1)
     
        # compute the intersection 
        iou = interArea / float(boxAArea + boxBArea - interArea)
     
        # return the intersection over union value
        return iou        
        
    def update_objects(self, boxes, info):
        self.object_candidates = []
        self.objects = []
        
        for i, box in enumerate(boxes):
            interArea = (box[2] + 1) * (box[3] + 1)
            #Small high velocity targets tend to be unrilable
            if interArea > TensorFlowYoloTracker.drawing_threshold:
                obj = tracked_object(box,info[i][0],info[i][1],self.frame_count)
                self.objects.append(obj)
            #Filter for apposite side of road
            #TODO value comes fromm lane detector
            elif box[0] < 500:
                self.object_candidates.append(box) 
                    
        # Delete objects have not seen for a while
        for obj in self.objects:
            gone_time = self.frame_count - obj.last_seen
            if gone_time > TensorFlowYoloTracker.remember_threshold:
                self.objects.remove(obj)
                 
class tracked_object:
    '''
    class attributes
    '''
    def __init__(self, bbox, label, score, last_seen):
        '''
        Constructor
        '''
        self.bbox = bbox
        self.label = label
        self.detection_score = score
        self.last_seen = last_seen
    
    #TODO add bbox setter and calc speed
