'''
Created on Sep 23, 2017

@author: Asad Zia
'''
from ClassicUtils import *

class ClassicLaneDetector(object):
    
    '''
    class attributes
    '''
    window_size = (50,100)    # Obtained empirically
    dpm = (3.7/700, 30/720)   # meters per pixel
    
    min_lane_distance = 425   # pixles 
    max_lane_distance = 550   # pixels, challenge vidoe does better with 500
    
    curve_tolerance  = 600
    bad_frame_threshold = 10
    lane_smoothing = 3
    drift_cotrol = 10
    margin = 25
    curve_radii_smoothing = 3
    
    def __init__(self):
        '''
        Constructor
        '''
        self.recent_lanes = []
        self.recent_centers = []
        self.recent_curve_radii = []
        self.needs_reset = True
        self.bad_frame_count = 0
        self.frame_count = 0   
     
    '''
        * Checking that they both lanes have similar curvature
        * Checking if they are separated by the right distance horizontally
        * Checking that they are roughly parallel     
    '''           
    def sanity_ok(self, window_centroids, curve_radii):
        left_center = window_centroids[:,0][0]
        right_center = window_centroids[:,1][0]
    
        lane_distance = (right_center - left_center)
        if (lane_distance < ClassicLaneDetector.min_lane_distance or 
            lane_distance > ClassicLaneDetector.max_lane_distance):
            return  False
        
        left_roc, right_roc = curve_radii
        diffence = abs(left_roc - right_roc)
        if diffence > ClassicLaneDetector.curve_tolerance:
            return  False
        return True
    
    # This expects undistored images
    def process_frame(self, image):
    
        preprocessed = create_binary_image(image)
        
        warped, m_inv = birds_eye_perspective(preprocessed)
        
        window_centroids = get_left_right_centroids(self.recent_centers, warped, 
                                                    ClassicLaneDetector.window_size, 
                                                    margin=ClassicLaneDetector.margin, 
                                                    hunt=self.needs_reset)
        
        lanes, yvals, camera_center = fit_lane_lines(image.shape[0], window_centroids, 
                                                     ClassicLaneDetector.window_size,
                                                     w_factor=0.1)
        
        curve_radii = radius_of_curvature(image.shape[0],ClassicLaneDetector.dpm,
                                                    window_centroids, yvals)
        
        if not self.sanity_ok(window_centroids, curve_radii):
            #bad frame
            if self.frame_count == 0:
                # Don't do anything if frist frame is bad
                return None
            lanes = self.recent_lanes[-1]
            curve_radii = self.recent_curve_radii[-1]
            self.bad_frame_count += 1         
        else: # Good frame
            self.recent_lanes.append(lanes)
            self.recent_curve_radii.append(curve_radii)
            self.needs_reset = False

            
        lanes = np.average(self.recent_lanes[-ClassicLaneDetector.lane_smoothing:], axis=0)
        curve_radii = np.average(self.recent_curve_radii[-ClassicLaneDetector.curve_radii_smoothing:], 
                                 axis=0)

        
        
        # Conditions for Reset
        if (self.frame_count < ClassicLaneDetector.lane_smoothing or 
            self.frame_count < ClassicLaneDetector.drift_cotrol        or
            self.bad_frame_count > ClassicLaneDetector.bad_frame_threshold):
            self.needs_reset = True
             
        self.frame_count += 1
        return (camera_center, curve_radii, m_inv, lanes)  
    
    def draw_lane_lines(self, image, lane_info):
        
        camera_center, curve_radii, m_inv, lanes = lane_info  
        result, _ = draw_lane_lines(image, m_inv, lanes, colors=([0,255,0],[0,255,0]))              
        annotate_results(result, camera_center, ClassicLaneDetector.dpm, curve_radii)
        return result
