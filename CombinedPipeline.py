'''
Created on Sep 23, 2017

@author: asad
'''
from moviepy.editor import VideoFileClip
from TensorFlowYoloTracker import TensorFlowYoloTracker
from ClassicLaneDetector import ClassicLaneDetector
from ObjectTracker import ObjectTracker
import cv2
import pickle

class VideoPipeline(object):
    
    '''
    class attributes
    '''

    def __init__(self, camera_cal_pickle, track_lanes=True, track_objects=True):
        '''
        Constructor
        '''
        self.track_lanes = track_lanes
        self.track_objects = track_objects
        dest_pickle = pickle.load( open(camera_cal_pickle, "rb"))
        self.mtx = dest_pickle["mtx"]
        self.dist = dest_pickle["dist"]
        
        self.car_detector = TensorFlowYoloTracker()
        self.lane_detector = ClassicLaneDetector()
        self.obj_tracker = ObjectTracker()
            
    # Video processing pipeline
    def process_image(self, image):
    
        image = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        
        if self.track_lanes:
            lane_info = self.lane_detector.process_frame(image)
            
        if self.track_objects:
            bbox, info = self.car_detector.process_frame(image)
            self.obj_tracker.process_frame(image,bbox,info)
            self.obj_tracker.draw_boxes(image)
            
        if self.track_lanes and lane_info != None:
                image = self.lane_detector.draw_lane_lines(image, lane_info)
        
        return image  
        
def main():
    videoname = 'project_video'
    output = videoname + '_output.mp4'
    input  = videoname + '.mp4'
    
    clip = VideoFileClip(input)#subclip(4,7)
    processor = VideoPipeline("camera_cal/wide_dist_pickle.p", track_lanes=False)
    video_clip = clip.fl_image(processor.process_image)
    video_clip.write_videofile(output, audio=False)
    
if __name__ == '__main__':
    main()
