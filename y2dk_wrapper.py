'''
Created on Oct 1, 2017

@author: Asad Zia

Adopted from https://github.com/allanzelener/YAD2K
'''
import cv2
import colorsys
import glob
import os
import random

import numpy as np
from keras import backend as K
from keras.models import load_model

from yad2k.models.keras_yolo import yolo_eval, yolo_head

class Y2dk(object):

    '''
    class attributes
    '''
    iou_threshold = .5
    def __init__(self, options):
        '''
        Constructor
        '''
        model_path = os.path.expanduser(options['model'])
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
        anchors_path = os.path.expanduser(options['anchors_path'])
        classes_path = os.path.expanduser(options['classes_path'])
    
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
    
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
    
        yolo_model = load_model(model_path)
    
        # Verify model, anchors, and classes are compatible
        num_classes = len(class_names)
        num_anchors = len(anchors)
        # TODO: Assumes dim ordering is channel last
        model_output_channels = yolo_model.layers[-1].output_shape[-1]
        assert model_output_channels == num_anchors * (num_classes + 5), \
            'Mismatch between model and given anchor and class sizes. ' \
            'Specify matching anchors and classes with --anchors_path and ' \
            '--classes_path flags.'
        print('{} model, anchors, and classes loaded.'.format(model_path))
    
        # Check if model is fully convolutional, assuming channel last order.
        model_image_size = yolo_model.layers[0].input_shape[1:3]
    
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(class_names), 1., 1.)
                      for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.
    
        # Generate output tensor targets for filtered bounding boxes.
        # TODO: Wrap these backend operations with Keras layers.
        yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
        input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(
            yolo_outputs,
            input_image_shape,
            score_threshold=options["threshold"],
            iou_threshold=Y2dk.iou_threshold)
        
        self.model_image_size = model_image_size
        self.yolo_state = (boxes, scores, classes, yolo_model, input_image_shape)
        self.class_names = class_names 
        self.colors = colors 
    
    def predict(self, image_data):
    
        boxes, scores, classes, yolo_model, input_image_shape = self.yolo_state 
    
        sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.
            

        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model.input: image_data,
                input_image_shape: [self.model_image_size[1], self.model_image_size[0]],
                K.learning_phase(): 0
            })
        #print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    
        #sess.close()
        
        return out_boxes, out_scores, out_classes

    def return_predict(self, input_image):
        
        image = np.copy(input_image)
        image = cv2.resize(image, self.model_image_size)
        image = image / 255.
        image = image[:,:,::-1]
        
        out_boxes, out_scores, out_classes = self.predict(image)
            
        boxesInfo = list()
        for box,score,label in zip(out_boxes, out_scores, out_classes):
            top, left, bottom, right = box
            #scale the boxes
            h_scale = input_image.shape[0]/self.model_image_size[0]
            w_scale = input_image.shape[1]/self.model_image_size[1] 
            top = int(top * h_scale)
            bottom = int(bottom * h_scale)
            left = int(left * w_scale)
            right = int(right * w_scale)
            #clip the boxes
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(input_image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(input_image.shape[1], np.floor(right + 0.5).astype('int32'))
        
            boxesInfo.append({
                "label": self.class_names[label],
                "confidence": score,
                "topleft": {
                    "y": top,
                    "x": left},
                "bottomright": {
                    "y": bottom,
                    "x": right}
            })
        return boxesInfo    
    
'''
YOLO provides multiple detectctions that might be interesting for self driving cars
like person bicycle bus truck botorbike traffic lights, pedestrian etc. 
'''     
def is_interesting(label):
    return True
    if (
        label == "car" or label == "truck" or
        label == "bicycle" or label == "bus" or
        label == "motorbike" or label == "person"
        ):
        return True
    
    return False
    
def resize_input(im):

    imsz = cv2.resize(im, (608, 608))
    imsz = imsz / 255.
    imsz = imsz[:,:,::-1]
    return imsz
    
def main():
    options = {"model": "model_data/yolo.h5", "threshold": 0.1, "anchors_path":"model_data/yolo_anchors.txt","classes_path":"model_data/coco_classes.txt"}
    model = Y2dk(options)
    
    images = glob.glob( './images/*.jpg' )
    
    for idx, fname in enumerate(images):
        print(fname)
        in_image = cv2.imread(fname)

        boxes = model.return_predict(in_image)
        
        #in_image = cv2.resize(in_image, (608, 608))       
        for box in boxes:
            x1 = box['topleft']['x']
            y1 = box['topleft']['y']
            x2 = box['bottomright']['x']
            y2 = box['bottomright']['y']
            label = box['label']
            confidence = box['confidence']
   
            if is_interesting(label):
                if confidence > .45:
                    color = (0,255,0)
                    if label == "car":
                        color = (255,128,128)
                    elif label == "bus":
                        color = (255,0,255)
                        
                    #in_image = cv2.resize(in_image, (608, 608))
                 
                    cv2.rectangle(in_image,(x1,y1),(x2,y2),color,2)
                    
                    label = str(label) + " " + str(int(confidence*100)) 
                    cv2.putText(
                        in_image, label, (x1, y1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * 600, color,
                        1)
                             
          
        cv2.imwrite('./images/out/' + str(idx) + '.jpg',in_image)  
                       
if __name__ == '__main__':
    main()


