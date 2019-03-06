from distutils.version import StrictVersion
from label_utils import label_map_util

import numpy as np
import tensorflow as tf
import argparse
import os
import time
import cv2

FLAGS = None

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


def load_image_and_labels(label_path,
                          image_path,
                          verbose=True,
                          types=['.jpg', '.png', '.jpeg']):
    # labels
    category_index = label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)
    
    image_paths = []
    if os.path.isfile(image_path):
        image_paths.append(image_path)
    else:
        for file_or_dir in os.listdir(image_path):
            file_path = os.path.join(image_path, file_or_dir)
            if os.path.isfile(file_path) and \
                    os.path.splitext(file_path)[1].lower() in types:
                image_paths.append(file_path)
    if verbose:
        print(image_paths)
    return category_index, image_paths


class TFDetector(object):
    def __init__(self, model_path, category_index):
        self.graph = self.create_graph(model_path)
        self.sess = self.create_session()
        self.category_index = category_index
    
    def create_graph(self, model_path):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.graph = detection_graph
        return self.graph
    
    def create_session(self):
        with self.graph.as_default():
            self.sess = tf.Session()
        return self.sess
    
    def detect(self, image, mark=False):
        with self.graph.as_default():
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            # Run inference
            start_time = time.time()
            output_dict = self.sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
            end_time = time.time()
            print("run time:", end_time - start_time)
            
            # all outputs are float32 numpy arrays, so convert types as appropriate
            num = int(output_dict['num_detections'][0])
            output_dict['num_detections'] = num
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)[:num]
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0][:num]
            output_dict['detection_scores'] = output_dict['detection_scores'][0][:num]
            
            if mark:
                for i in range(output_dict['num_detections']):
                    image_height, image_width = image.shape[:2]
                    
                    cls_id = output_dict['detection_classes'][i]
                    cls_name = self.category_index[cls_id]['name']
                    score = output_dict['detection_scores'][i]
                    box_ymin = int(output_dict['detection_boxes'][i][0] * image_height)
                    box_xmin = int(output_dict['detection_boxes'][i][1] * image_width)
                    box_ymax = int(output_dict['detection_boxes'][i][2] * image_height)
                    box_xmax = int(output_dict['detection_boxes'][i][3] * image_width)
                    cv2.rectangle(image, (box_xmin, box_ymin), (box_xmax, box_ymax), (0, 255, 0), 3)
                    text = "%s:%.2f" % (cls_name, score)
                    cv2.putText(image, text, (box_xmin, box_ymin - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                                (255, 0, 0))
                # show image
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow("img", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        return output_dict, image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str,
                        default='test_images/',
                        help='image path')
    parser.add_argument('--model_frozen', type=str,
                        default='TFMODEL/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb',
                        help='model path')
    parser.add_argument('--label_path', type=str,
                        default='label_utils/mscoco_label_map.pbtxt',
                        help='label path')
    FLAGS, unparsed = parser.parse_known_args()
    
    category_index, image_paths = load_image_and_labels(FLAGS.label_path, FLAGS.image_path)
    detector = TFDetector(FLAGS.model_frozen, category_index)
    
    for image_path in image_paths:
        # load image
        image = cv2.imread(image_path)
        # convert color space, try to remove this you'll see amazing result for image1
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # detection
        detector.detect(image, mark=True)
