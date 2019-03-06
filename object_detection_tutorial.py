#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import argparse
import sys
import os
import time
import cv2

from distutils.version import StrictVersion
from PIL import Image
from matplotlib import pyplot as plt

import rospy
from std_msgs.msg import String
from object_recognition_msgs.msg import *
from shape_msgs.msg import *
from geometry_msgs.msg import *

FLAGS = None

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from label_utils import label_map_util

def isimage(path):
  return os.path.splitext(path)[1].lower() in ['.jpg', '.png', '.jpeg']

class TFDetector:
  def __init__(self, model_path, label_path):
    self.image_tensor, self.tensor_dict, self.sess = self.init_model(model_path)
    self.category_index = label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)

  def create_graph(self, model_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(model_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    return detection_graph

  def init_model(self, model_frozen):
    # create graph
    graph = self.create_graph(model_frozen)

    with graph.as_default():
      sess = tf.Session()

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

      return image_tensor, tensor_dict, sess


  def detect(self, image_np):

    image_height, image_width, _ = image_np.shape
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Actual detection.
    start_time = time.time()
    output_dict = self.sess.run(self.tensor_dict, feed_dict={self.image_tensor: image_np_expanded})
    num = int(output_dict['num_detections'][0])
    output_dict['num_detections'] = num
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)[:num]
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0][:num]
    output_dict['detection_scores'] = output_dict['detection_scores'][0][:num]
    end_time = time.time()
    print("run time:", end_time - start_time)

    objects = []
    for i in range(output_dict['num_detections']):
      object = {}
      object['cls_id'] = output_dict['detection_classes'][i]
      object['cls_name'] = self.category_index[object['cls_id']]['name']
      object['score'] = output_dict['detection_scores'][i]
      object['box_ymin'] = int(output_dict['detection_boxes'][i][0]*image_height)
      object['box_xmin'] = int(output_dict['detection_boxes'][i][1]*image_width)
      object['box_ymax'] = int(output_dict['detection_boxes'][i][2]*image_height)
      object['box_xmax'] = int(output_dict['detection_boxes'][i][3]*image_width)
      objects.append(object)
    return objects

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def main():
  # init tf_detector
  tf_detector = TFDetector(FLAGS.model_frozen, FLAGS.label_path)

  # init ros_publisher
  rospy.init_node('talker', anonymous=True)
  pub = rospy.Publisher('chatter', RecognizedObjectArray, queue_size=10)
  rate = rospy.Rate(10) # 10hz

  image_paths = []
  if os.path.isfile(FLAGS.image_path):
    image_paths.append(FLAGS.image_path)
  else:
    for file_or_dir in os.listdir(FLAGS.image_path):
      file_path = os.path.join(FLAGS.image_path, file_or_dir)
      if os.path.isfile(file_path) and isimage(file_path):
        image_paths.append(file_path)
  print(image_paths)

  idx = 0
  while (not rospy.is_shutdown()) and idx < len(image_paths):
    image_path = image_paths[idx]

    # prepare data
    image = Image.open(image_path)
    #(image_width, image_height) = image.size
    image_np = load_image_into_numpy_array(image)

    # detect
    objects = tf_detector.detect(image_np)

    recognized_objects = RecognizedObjectArray()
    now = rospy.Time.now()
    recognized_objects.header.stamp = now
    recognized_objects.header.frame_id = str(idx)

    for object in objects:
      obj = RecognizedObject()
      obj.type.key = object['cls_name']
      obj.confidence = object['score']
      obj.bounding_contours = [Point(object['box_xmin'],object['box_ymin'],0), Point(object['box_xmax'],object['box_ymax'],0)]
      recognized_objects.objects.append(obj)

      cv2.rectangle(image_np, (object['box_xmin'],object['box_ymin']), (object['box_xmax'],object['box_ymax']), (0,255,0),3)
      text = "%s:%.2f" % (object['cls_name'], object['score'])
      cv2.putText(image_np, text, (object['box_xmin'],object['box_ymin']-4),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255,0,0))

    rospy.loginfo("%s %s" % (rospy.get_time(), image_path))
    idx += 1
    pub.publish(recognized_objects)
    rate.sleep()

    plt.figure(figsize=(12, 8)) # Size, in inches
    plt.imshow(image_np)
    plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--image_path', type=str,
                      default='/home/andy/selfdrivingcar/catkin_ws/src/object-detection/test_images/',
                      help='image path')
  parser.add_argument('--model_frozen', type=str,
                      default='/home/andy/selfdrivingcar/TFMODEL/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb',
                      help='model path')
  parser.add_argument('--label_path', type=str,
                      default='/home/andy/selfdrivingcar/catkin_ws/src/object-detection/label_utils/mscoco_label_map.pbtxt',
                      help='label path')
  FLAGS, unparsed = parser.parse_known_args()
  main()
