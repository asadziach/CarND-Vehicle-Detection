'''
Created on Sep 18, 2017

@author: asad
'''
import numpy as np
import cv2
import glob
import pickle

# Thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

# Thresholds the V-channel of HSV
def hsv_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hls[:,:,2]
    binary_output = np.zeros_like(v_channel)
    binary_output[(v_channel > thresh[0]) & (v_channel <= thresh[1])] = 1
    return binary_output

# Takes an image, gradient orientation and threshold min / max values.
def abs_sobel_select(img, orient='x', thresh=(0, 255)):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient. Take absolute
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy
    binary_output = np.zeros_like(scaled_sobel)
    # Apply threshold
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

'''
 Returns magnitude of the gradientfor a given sobel kernel
 size and threshold
'''
def sobel_mag_select(img, sobel_kernel=3, sobel_mag_select=(0, 255)):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create copy
    binary_output = np.zeros_like(gradmag)
    # Apply threshold
    binary_output[(gradmag >= sobel_mag_select[0]) & (gradmag <= sobel_mag_select[1])] = 1
    return binary_output

# Threshold for a given range and Sobel kernel
def sobel_dir_select(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Create copy
    binary_output =  np.zeros_like(absgraddir)
    # Apply threshold
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

# Used to draw debug rectangles in sliding widows
def debug_rect_mask(width, height, image, center, level):
    output = np.zeros_like(image)
    output[int(image.shape[0]-(level+1)*height):int(image.shape[0]-level*height),
           max(0,int(center-width/2)):min(int(center+width/2),image.shape[1])] = 1
    return output

# Undo camera distortion
def undistort(image, mtx, dist):
    result = cv2.undistort(image, mtx, dist, None, mtx)
    return result

# Color a and gradient thresholding    
def create_binary_image(image):
    preprocessImage = np.zeros_like(image[:,:,0])
    gradx = abs_sobel_select(image, orient='x', thresh=(12,255))
    grady = abs_sobel_select(image, orient='y', thresh=(25,255))
    hls   = hls_select(image, thresh=(100,255))
    hsv   = hsv_select(image, thresh=(50,255))
    preprocessImage[(gradx==1) & (grady==1) | (hls==1) & (hsv==1) ] = 255    
    return preprocessImage

# Perspective transform of image.
def birds_eye_perspective(image):
    # Obtained empirically to map trapezoid to birds eye view
    bottom_width = 0.76
    bottom_trim = 0.935
    center_width = 0.08
    height_margin = 0.62
    
    img_size = (image.shape[1], image.shape[0])
            
    src_img = np.float32([[img_size[0]*(.5-center_width/2),img_size[1]*height_margin],
                      [img_size[0]*(.5+center_width/2),img_size[1]*height_margin],
                      [img_size[0]*(.5+bottom_width/2),img_size[1]*bottom_trim],
                      [img_size[0]*(.5-bottom_width/2),img_size[1]*bottom_trim]])
    offset = img_size[0]*.25
    dst_img = np.float32([[offset, 0], [img_size[0]-offset, 0], 
                      [img_size[0]-offset, img_size[1]],
                      [offset, img_size[1]]])
    
    M = cv2.getPerspectiveTransform(src_img, dst_img)
    Minv = cv2.getPerspectiveTransform(dst_img, src_img)
    wrapped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    
    return wrapped, Minv      
'''
 Convolution approach with the sliding window method
 recent_centers: send list of previous detections if you have. Othewise an empty list
 set hunt=True for looking for lines from scratch
 '''
def get_left_right_centroids(recent_centers, image, window_size, margin=100, smoothing=15, hunt=True):
    
    width = window_size[0]
    height = window_size[1]
    centroids = []
    window = np.ones(width)
    start_index = 0
    
    if hunt:
        #Left
        left_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
        left_center = np.argmax(np.convolve(window,left_sum)) - width/2
        #Right
        right_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
        right_center = np.argmax(np.convolve(window,right_sum)) - width/2 + int(image.shape[1]/2)
        
        centroids.append((left_center, right_center))
        
        start_index += 1 # This window is done, now start with the one above it
    else:
        left_center,right_center = recent_centers[0][-1]
    
    last_right = 0
    last_left = 0
    right_trend = 0
    left_trend = 0
    # Move up
    for level in range(start_index,(int)(image.shape[0]/height)):
       
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*height):
                                    int(image.shape[0]-level*height),:], axis=0)
        
        conv_signal = np.convolve(window, image_layer)
        
        offset = width/2
        
        left_min_index = int(max(left_center+offset-margin, 0))
        left_max_index = int(min(left_center+offset+margin, image.shape[1]))
        left_signal = conv_signal[left_min_index:left_max_index]
        if left_signal.any():
            left_center = np.argmax(left_signal)
            left_confidence = conv_signal[left_center + left_min_index] # signal strength
            left_center = left_center+left_min_index-offset
        else:
            left_confidence = 0
        
        right_min_index = int(max(right_center+offset-margin, 0))
        right_max_index = int(min(right_center+offset+margin, image.shape[1]))
        right_signal = conv_signal[right_min_index:right_max_index]
        if right_signal.any():
            right_center = np.argmax(right_signal)
            right_confidence = conv_signal[right_center + right_min_index] # signal strength
            right_center = right_center+right_min_index-offset
        else:
            right_confidence = 0
        
        # Drop window if it has no pixels (detection  failed)
        if (right_confidence == 0):
            right_center = last_right + right_trend
  
        if (left_confidence == 0):
            left_center = last_left + left_trend
                        
        right_trend = right_center - last_right 
        left_trend = left_center - last_left                    
        last_right =  right_center
        last_left = left_center
        
        centroids.append((left_center,right_center))
    
    recent_centers.append(centroids)
    
    # return result averaged over past centers.
    return np.average(recent_centers[-smoothing:], axis=0)

def draw_visual_debug(image, window_centroids, window_size):
    l_points = np.zeros_like(image)
    r_points = np.zeros_like(image)
    
    window_width = window_size[0] 
    window_height = window_size[1]
            
    for level in range(0,len(window_centroids)):
        l_mask = debug_rect_mask(window_width,window_height,image,window_centroids[level][0],level)
        r_mask = debug_rect_mask(window_width,window_height,image,window_centroids[level][1],level)
        l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
        r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
    
    # temp visual debug
    template = np.array(r_points+l_points,np.uint8) 
    zero_channel = np.zeros_like(template)  
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) 
    warpage = np.array(cv2.merge((image,image,image)),np.uint8) 
    return cv2.addWeighted(warpage, .7, template, 1, 0.0) 

# Fit their positions with a polynomial and find camera center
def fit_lane_lines(image_height, window_centroids, window_size, w_factor=1):
    
    left_x = window_centroids[:,0]
    right_x = window_centroids[:,1]
    
    yvals = np.arange(0, image_height)
    window_width = window_size[0] * w_factor
    window_height = window_size[1]        
    res_yvals = np.arange(image_height-window_height/2, 0, -window_height)
    left_fit = np.polyfit(res_yvals, left_x, 2)
    left_poly = np.poly1d(left_fit)
    left_fitx = np.int32(left_poly(yvals))

    right_fit = np.polyfit(res_yvals, right_x, 2)
    right_poly = np.poly1d(right_fit)
    right_fitx = np.int32(right_poly(yvals))
    
    # Fancy array magic to encapsulate list values
    left_lane = np.array(list(zip(np.concatenate((left_fitx - window_width/2,left_fitx[::-1]+window_width/2), axis=0),
                                  np.concatenate((yvals,yvals[::-1]),axis=0))))
    right_lane = np.array(list(zip(np.concatenate((right_fitx - window_width/2,right_fitx[::-1]+window_width/2), axis=0),
                                   np.concatenate((yvals,yvals[::-1]),axis=0))))
    centre_line = np.array(list(zip(np.concatenate((left_fitx + window_width/2,right_fitx[::-1]-window_width/2), axis=0),
                                   np.concatenate((yvals,yvals[::-1]),axis=0))))
    
    lanes = (left_lane, centre_line, right_lane)
    
    # Calculate camera center in pixels
    camera_center = (left_fitx[-1] + right_fitx[-1])/2 
    
    return lanes, res_yvals, camera_center 
'''
 Draws lane lines with optional whilte background to increase contrast.
 Perform inverse of birds eyeview on lanes and blend with image.
'''
def draw_lane_lines(image, m_inv, lanes, colors, draw_background=False):
    left_color, right_color = colors
    left, centre, right = lanes
    img_size = (image.shape[1], image.shape[0])
    
    fg = np.zeros_like(image)
    cv2.fillPoly(fg,np.int32([left]),left_color)
    cv2.fillPoly(fg,np.int32([right]),right_color)
    lane_lines_only = np.copy(fg)
    
    draw_driveable_surface(fg, centre)
    fg = cv2.warpPerspective(fg, m_inv, img_size, flags=cv2.INTER_LINEAR)
    
    if draw_background:
        bg = np.zeros_like(image)
        cv2.fillPoly(bg,np.int32([left]),[255,255,255])
        cv2.fillPoly(bg,np.int32([right]),[255,255,255])
        bg = cv2.warpPerspective(bg, m_inv, img_size, flags=cv2.INTER_LINEAR)
    
        base = cv2.addWeighted(image, 1.0, bg, -1.0, 0.0) 
        result = cv2.addWeighted(base, 1.0, fg, 0.7, 0.0) 
    else:
        result = cv2.addWeighted(image, 1.0, fg, 0.7, 0.0) 
            
    return result, lane_lines_only   

# Green area between lane lines
def draw_driveable_surface(image, centre):
    cv2.fillPoly(image,np.int32([centre]),color=[0,255,0])

# Blend on binary image for debuggings
def overlay_on_binary(base, overlay):
    warpage = np.array(cv2.merge((base,base,base)),np.uint8)
    return cv2.addWeighted(warpage, .7, overlay, 1., 0.0)

# Calculates radius of curvature       
def radius_of_curvature(image_height, dpm, window_centroids, res_yvals):

    left_x = window_centroids[:,0]
    curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*dpm[1], np.array(left_x,np.float32)*dpm[0], 2)
    left = ((1+(2*curve_fit_cr[0]*res_yvals[-1]*dpm[1] + curve_fit_cr[1])**2)**1.5) / np.absolute(2*curve_fit_cr[0])
    
    right_x = window_centroids[:,1]
    curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*dpm[1], np.array(right_x,np.float32)*dpm[0], 2)
    right = ((1+(2*curve_fit_cr[0]*res_yvals[-1]*dpm[1] + curve_fit_cr[1])**2)**1.5) / np.absolute(2*curve_fit_cr[0])    
    return (left,right) 

# Create final result 
def annotate_results(image, camera_center, dpm, curve_radii):
    
    curve_radius  = (curve_radii[0] + curve_radii[1])/2 # Take average of left and right
    center_diff = (camera_center-image.shape[1]/2)*dpm[0]
    side_pos = 'right' if center_diff <= 0 else'left'

    # Put the text on the resulting image
    cv2.putText(image, 'Radius of Curvature: '+ str(round(curve_radius,3))+'m',
                (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(image, 'Camera: '+str(abs(round(center_diff,3)))+'m ('+ side_pos +' of center)', 
                (50,100), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

def main():
    dest_pickle = pickle.load( open("camera_cal/wide_dist_pickle.p", "rb"  ))
    mtx = dest_pickle["mtx"]
    dist = dest_pickle["dist"]

    images = glob.glob( './test_images/test*.jpg' )
    
    for idx, fname in enumerate(images):
        image = cv2.imread(fname)
        image = undistort(image, mtx, dist)
        #Debug point
        cv2.imwrite('./output_images/undistorted' + str(idx) + '.jpg', image)
        
        preprocessed = create_binary_image(image)
        
        #Debug point
        cv2.imwrite('./output_images/preprocessed' + str(idx) + '.jpg', preprocessed)
                
        warped, m_inv = birds_eye_perspective(preprocessed)

        #Debug point
        cv2.imwrite('./output_images/warped' + str(idx) + '.jpg', warped)
        
        window_size = (50,80)     # Obtained empirically
        window_centroids = get_left_right_centroids([], warped, window_size, margin=25)
        
        #Debug point   
        tracked = draw_visual_debug(warped, window_centroids, window_size) 
        cv2.imwrite('./output_images/tracked' + str(idx) + '.jpg', tracked)
        
        lanes, yvals, camera_center = fit_lane_lines(image.shape[0], window_centroids, window_size)
        result, lane_lines_only = draw_lane_lines(image, m_inv, lanes, 
                                                  colors=([255,0,0],[0,0,255]), draw_background=True)
        
        #Debug point
        drawn = overlay_on_binary(warped, lane_lines_only)
        cv2.imwrite('./output_images/drawn' + str(idx) + '.jpg', drawn)
        
        dpm = (3.7/700, 30/720) # meters per pixel 
        curve_radii = radius_of_curvature(image.shape[0],dpm,window_centroids, yvals)
        
        annotate_results(result, camera_center, dpm, curve_radii)

        cv2.imwrite('./output_images/final' + str(idx) + '.jpg',result)   

        
if __name__ == '__main__':
    main()