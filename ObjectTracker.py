'''
Created on Oct 8, 2017

@author: Asad Zia
'''
import cv2
from scipy import signal
import numpy as np

class ObjectTracker(object):
    '''
    class attributes
    '''
    #Primary Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    RED = (255, 0, 0)
    LEGO_BLUE = (0,50,150)
    LEGO_ORANGE = (255,150,40)
    VIOLET = (181, 126, 220)
    ORANGE = (255, 165, 0)
    GREEN = (0, 128, 0)
    GRAY = (128, 128, 128)
        
    box_color = (255,128,128)
    line_color = (255,255,0)
    min_tracking_iou = 0.4
    remember_threshold = 3    # frames
    drawing_threshold = 0     # pxels (disabled)
    corr_threshold = 18000000
    left_lane_threshold = 500 
    debug = False
    count = 0
        
    def __init__(self):
        '''
        Constructor
        '''
        self.frame_count = 0 
        self.objects = []
        self.object_candidates = []
        self.objects = []
        self.colorlist = [
                ObjectTracker.BLACK,
                ObjectTracker.WHITE,
                ObjectTracker.GRAY,
                ObjectTracker.GREEN,                    
                ObjectTracker.YELLOW,
                ObjectTracker.RED,
                ObjectTracker.BLUE,                
                ObjectTracker.VIOLET,
                ObjectTracker.ORANGE,
                ]
        
    # Each tracked objects gets a unique color     
    def allocate_color(self):
        ObjectTracker.count += 1
        size = len(self.colorlist)
        index = ObjectTracker.count%size
        return self.colorlist[index]
    
    # Return the color to the list to be recycled
    def free_color(self, color):
        pass
        
    def process_frame(self,image,bboxs,info):
        
        #Optically track all objects and predict expected bbox
        for obj in self.objects:
            if not obj.predict(image):
                #Remove object if visual tracking failed
                self.delete_tracked_obj(obj)
                
        if bboxs and info:
            self.update_objects(image,bboxs,info)
        
        self.frame_count += 1

                
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
    
        #Intersection over Union (IoU) TODO namedtuple
    def calculate_intersect(self, boxA, boxB):
        interAreaA = (boxA[2] + 1) * (boxA[3] + 1)
        interAreaB = (boxB[2] + 1) * (boxB[3] + 1)
        areaSmallerBox = min(interAreaA, interAreaB)
        
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
     
        # compute how much smaller box is in the big box
        result = interArea / areaSmallerBox
     
        return result 
    
    def draw_boxes(self, image):
        for obj in self.objects:
            newbox = obj.bbox
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            
            cv2.rectangle(image, p1, p2, obj.color,3)
            cv2.putText(
                image, str(obj.label), (p1[0], p1[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * 600, obj.color,
                1)
        #unreliable detections on apposite site of road
        for newbox in self.object_candidates:
            p1 = (int(newbox[0]), int(newbox[1] + newbox[3]/2))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]/2))
            cv2.arrowedLine(image, p2, p1, ObjectTracker.line_color,3)
                    
    def correlate(self,bboxA,bboxB,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        p1 = (int(bboxA[0]), int(bboxA[1]))
        p2 = (int(bboxA[0] + bboxA[2]), int(bboxA[1] + bboxA[3]))
        imgA = image[p1[1]:p2[1], p1[0]:p2[0]].astype(int)
        imgA -= int(imgA.mean())
        
        p1 = (int(bboxB[0]), int(bboxB[1]))
        p2 = (int(bboxB[0] + bboxB[2]), int(bboxB[1] + bboxB[3]))
        imgB = image[p1[1]:p2[1], p1[0]:p2[0]].astype(int)
        imgB -= int(imgB.mean())
        
        corr = signal.correlate2d(imgA,imgB)
                
        return corr.flat[np.argmax(corr)]
        
    def update_objects(self, image, boxes, info):

        self.objects.sort(key=lambda x: x.area, reverse=True)
        boxes.sort(key=lambda x: (x[2] + 1) * (x[3] + 1), reverse=True)
        
        self.object_candidates = []
        detect_boxes = boxes[:]

        for i, box in enumerate(boxes):
            interArea = (box[2] + 1) * (box[3] + 1)
            #Small high velocity targets tend to be unreliable
            if interArea < ObjectTracker.drawing_threshold:
                #Filter for apposite side of road
                if box[0] < ObjectTracker.left_lane_threshold:
                    self.object_candidates.append(box)
                    detect_boxes.remove(box) 
        
        big_boxes = detect_boxes[:]
        for detect_box, obj in zip(big_boxes, self.objects):
            iou = self.calculate_intersect(detect_box,obj.bbox)
            if iou > ObjectTracker.min_tracking_iou:
                #Found the object
                obj.update_location(image,detect_box,self.frame_count,iou)        
            else:
                self.create_tracked_obj(detect_box,info[i][0],info[i][1],image)
                
            detect_boxes.remove(detect_box)
                                
        #unmatched boxes
        for box in detect_boxes:
            self.create_tracked_obj(box,info[i][0],info[i][1],image)
            
        # Delete objects have not seen for a while
        for obj in self.objects:
            gone_time = self.frame_count - obj.last_seen
            if gone_time > ObjectTracker.remember_threshold:
                self.delete_tracked_obj(obj)
                
        if ObjectTracker.debug:
            for newbox in boxes:
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                
                cv2.rectangle(image, p1, p2, ObjectTracker.BLACK,3)
                cv2.putText(
                    image, obj.label, (p1[0], p1[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * 600, ObjectTracker.BLACK,
                    1)
     
    def create_tracked_obj(self, bbox, label, score, image):
        color = self.allocate_color()
        obj = Vehicle(bbox,label,score,self.frame_count,image,color)
        self.objects.append(obj)
        
    def delete_tracked_obj(self, obj):
        self.free_color(obj.color)
        self.objects.remove(obj)
                      
class Vehicle:
    '''
    class attributes
    '''
    def __init__(self, bbox, label, score, last_seen, image, color):
        '''
        Constructor
        '''
        self.bbox = bbox
        self.label = label
        self.score = score
        self.last_seen = last_seen
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(image, bbox)
        self.color = color
        self.area = (bbox[2] + 1) * (bbox[3] + 1)
    
    def update_location(self,image,bbox,framecount, score):
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(image, bbox)
        self.bbox = bbox
        self.last_seen = framecount
        self.detection_score = score
    
    def predict(self,image):
        ok, bbox = self.tracker.update(image)
        if ok:
            self.bbox = bbox
        return ok
    #TODO add bbox setter and calc speed
if __name__ == '__main__':
    pass
