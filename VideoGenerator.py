'''
Created on Sep 23, 2017

@author: asad
'''
from moviepy.editor import VideoFileClip
from darkflow.net.build import TFNet
import cv2

class VideoLaneProcessor(object):
    
    '''
    class attributes
    '''

    def __init__(self, camera_cal_pickle):
        '''
        Constructor
        '''
        #options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "bin/tiny-yolo-voc.weights", "threshold": 0.1, "gpu": 1.0}
        options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}
        
        self.tfnet = TFNet(options)
        self.frame_count = 0   
     

    # Video processing pipeline
    def process_image(self, image):
    
        result = self.tfnet.return_predict(image)
        
        #print(result)
        
        for box in result:
            label = box['label']
            confidence = box['confidence']
            '''
            YOLO provides multiple detectctions that might be interesting for self driving cars
            like person bicycle bus truck botorbike traffic lights etc. For this project I am
            just using car. 
            '''            
            if label == "car" and confidence > 0.5: 
                       
                x1 = box['topleft']['x']
                y1 = box['topleft']['y']
                x2 = box['bottomright']['x']
                y2 = box['bottomright']['y']
                cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),3)
                label = label + " " + str(int(confidence*100))
                cv2.putText(
                    image, label, (x1, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * 600, (0,255,0),
                    1)              
    
        return image  
        
def main():
    videoname = 'project_video'
    output = videoname + '_output.mp4'
    input  = videoname + '.mp4'
    
    clip = VideoFileClip(input)
    processor = VideoLaneProcessor("camera_cal/wide_dist_pickle.p")
    video_clip = clip.fl_image(processor.process_image)
    video_clip.write_videofile(output, audio=False)
    
if __name__ == '__main__':
    main()
