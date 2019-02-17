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

FLAGS = None

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from label_utils import label_map_util

def create_graph(model_path):
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_path, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
  return detection_graph

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
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
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})
      end_time = time.time()
      print("run time:", end_time - start_time)

      # all outputs are float32 numpy arrays, so convert types as appropriate
      num = int(output_dict['num_detections'][0])
      output_dict['num_detections'] = num
      output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)[:num]
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0][:num]
      output_dict['detection_scores'] = output_dict['detection_scores'][0][:num]

  return output_dict

def main():
  # labels
  category_index = label_map_util.create_category_index_from_labelmap(FLAGS.label_path, use_display_name=True)

  # create graph
  detection_graph = create_graph(FLAGS.model_frozen)

  # prepare data
  image = Image.open(FLAGS.image_path)
  (image_width, image_height) = image.size
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)

  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)

  for i in range(output_dict['num_detections']):
    cls_id = output_dict['detection_classes'][i]
    cls_name = category_index[cls_id]['name']
    score = output_dict['detection_scores'][i]
    box_ymin = int(output_dict['detection_boxes'][i][0]*image_height)
    box_xmin = int(output_dict['detection_boxes'][i][1]*image_width)
    box_ymax = int(output_dict['detection_boxes'][i][2]*image_height)
    box_xmax = int(output_dict['detection_boxes'][i][3]*image_width)
    cv2.rectangle(image_np, (box_xmin,box_ymin), (box_xmax,box_ymax), (0,255,0),3)
    text = "%s:%.2f" % (cls_name,score)
    cv2.putText(image_np, text, (box_xmin,box_ymin-4),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255,0,0))

  plt.figure(figsize=(12, 8)) # Size, in inches
  plt.imshow(image_np)
  plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--image_path', type=str,
                      default='/home/andy/selfdrivingcar/models/research/object_detection/test_images/image2.jpg',
                      help='image path')
  parser.add_argument('--model_frozen', type=str,
                      default='/home/andy/selfdrivingcar/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb',
                      help='model path')
  parser.add_argument('--label_path', type=str,
                      default='label_utils/mscoco_label_map.pbtxt',
                      help='label path')
  FLAGS, unparsed = parser.parse_known_args()
  main()
