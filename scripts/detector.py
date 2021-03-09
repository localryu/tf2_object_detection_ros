#!/usr/bin/env python3
import rospy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings
from cv_bridge.boost.cv_bridge_boost import getCvType
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


PATH_TO_MODEL_DIR = '/home/ryu/catkin_build_ws/src/tf2_object_detection/src/object_detection/efficientdet_d0_coco17_tpu-32'
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

PATH_TO_LABELS = '/home/ryu/catkin_build_ws/src/tf2_object_detection/src/object_detection/efficientdet_d0_coco17_tpu-32/mscoco_label_map.pbtxt'


# args = parser.parse_args()

class DETECTOR:
    def __init__(self):
        print("init")
        rospy.init_node('tf2_object_detector_node', anonymous=True)
        rospy.Subscriber('/d400/color/image_raw', Image, self.imageCB)
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/od_res", Image, queue_size=2)

    def imageCB(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # cv2.imshow("image",self.cv_image)
            # cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)

    def run(self):
        print("start load model")
        # self.test_image = cv2.imread('/home/ryu/models/research/object_detection/test_images/image1.jpg',cv2.IMREAD_COLOR)
        detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
        print("load model")
        while not rospy.is_shutdown():
            cv2.imshow("image",self.cv_image)

            input_tensor = tf.convert_to_tensor(self.cv_image)
            input_tensor = input_tensor[tf.newaxis, ...]
            detections = detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                            for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            image_np_with_detections = self.cv_image.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes'],
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=200,
                    min_score_thresh=.70,
                    agnostic_mode=False)

            cv2.imshow('object detection', image_np_with_detections)
            cv2.waitKey(1)
        else:
            cv2.destroyAllWindows()

            # self.infer_img = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)

            # self.image_pub.publish(self.bridge.cv2_to_imgmsg(mask_cv,"bgr8"))


if __name__ == '__main__':
    detector = DETECTOR()
    detector.run()
