# Object Detection Tutorial
## v0.1.0

### Update

- draw bbox and label
- multi images

### Usage

- download model from https://github.com/tensorflow/models/blob/v1.13.0/research/object_detection/g3doc/detection_model_zoo.md
- run `python object_detection_tutorial.py --model_frozen /PATH/TO/MODEL`

for example:
```
python3 object_detection_tutorial.py --model_frozen ../ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb
```

### ref:
- modify from tensorflow detection api demo(v1.13.0): https://github.com/tensorflow/models/blob/v1.13.0/research/object_detection/object_detection_tutorial.ipynb

## v0.1.1
- optimization
- TFDetector class

### Update
- new TFDetector class with session as class member
- remove PIL, matplotlib dependencies
- reformat the code
- combine load image paths and labels in load_image_and_labels functions
- remove isimage function, add types parameter to handle more image types in the future
- remove main function 