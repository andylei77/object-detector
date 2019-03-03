# v0.1.0

## Update
- optimize session run time

## Usage

- download model from https://github.com/tensorflow/models/blob/v1.13.0/research/object_detection/g3doc/detection_model_zoo.md
- run `python object_detection_tutorial.py --model_frozen /PATH/TO/MODEL`

for example:
```
python object_detection_tutorial.py --model_frozen ../TFMODEL/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb
```

## ref:
- modify from tensorflow detection api demo(v1.13.0): https://github.com/tensorflow/models/blob/v1.13.0/research/object_detection/object_detection_tutorial.ipynb

v0.1.1
- optimization
- TFDetector class
