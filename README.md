# Tensorflow2 object detection ros package

## References

- Install tensorflow2 and envirion setup : https://github.com/localryu/TIL/blob/master/Object_detection/tensorflow2_object_detection.md
- Training with custom dataset : https://github.com/localryu/TIL/blob/master/Object_detection/tensorflow2_object_detection_training_with%20_custom_dataset.md

## Version
    - tensorflow 2.3.0
    - CUDA : 10.1
    - CuDNN : 7.6.5
    - Python : 3.6

## Required
- Install python3 cv_bridge : https://github.com/localryu/TIL/blob/master/ROS/cv_bridge%20for%20python3.md

## How to use

### 1. Copy frozen_graph(floder)
  copy frozen_graph(floder) into tf2_object_detection_ros/src/object_detection/ folder

### 2. Modify file path
  modify files path in scripts/detector.py
  
    PATH_TO_MODEL_DIR = '/PATH/TO/INFERENCE_GRAPH/FLODER/inference_graph'
    PATH_TO_LABELS = '/PATH/TO/LABEL_FILE/label_map.pbtxt'
    PATH_TO_CKPT = "/PATH/TO/CHECKPOINT_FILE/inference_graph/checkpoint"
    
### 3. make
    cd catkin_ws
    catkin_make
    source devel/setup.bash
    source ~/catkin_build_ws/install/setup.bash (for using python3 cv_bridge)

### 4. run object_detection_node
  
    rosrun tf2_object_detection_ros detector.py
