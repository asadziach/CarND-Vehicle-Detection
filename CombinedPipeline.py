'''
Created on Sep 23, 2017

@author: asad
'''
from moviepy.editor import VideoFileClip
from TensorFlowYoloTracker import TensorFlowYoloTracker
from ClassicLaneDetector import ClassicLaneDetector
import cv2
import pickle

class VideoPipeline(object):
    
    '''
    class attributes
    '''

    def __init__(self, camera_cal_pickle):
        '''
        Constructor
        '''
        dest_pickle = pickle.load( open(camera_cal_pickle, "rb"))
        self.mtx = dest_pickle["mtx"]
        self.dist = dest_pickle["dist"]
        
        self.car_detector = TensorFlowYoloTracker()
        self.lane_detector = ClassicLaneDetector()
            
    # Video processing pipeline
    def process_image(self, image):
    
        image = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        
        lane_info = self.lane_detector.process_frame(image)
        boxes = self.car_detector.process_frame(image)
        
        self.car_detector.draw_boxes(image,boxes)
        image = self.lane_detector.draw_lane_lines(image, lane_info)
        
        return image  
        
def main():
    videoname = 'test_video'
    output = videoname + '_output.mp4'
    input  = videoname + '.mp4'
    
    clip = VideoFileClip(input)
    processor = VideoPipeline("camera_cal/wide_dist_pickle.p")
    video_clip = clip.fl_image(processor.process_image)
    video_clip.write_videofile(output, audio=False)
    
if __name__ == '__main__':
    main()
