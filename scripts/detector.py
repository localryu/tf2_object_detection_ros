#!/usr/bin/env python3
import rospy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder
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

# MODEL_TYPE = 'CKPT'
MODEL_TYPE = 'S_MODEL'

PATH_TO_MODEL_DIR = '/home/localryu/catkin_ws/src/tf2_object_detection_ros/src/object_detection/inference_graph'
PATH_TO_LABELS = '/home/localryu/catkin_ws/src/tf2_object_detection_ros/src/object_detection/inference_graph/label_map.pbtxt'

if(MODEL_TYPE == 'CKPT'):
    #from checkpoint
    PATH_TO_CKPT = "/home/localryu/catkin_ws/src/tf2_object_detection_ros/src/object_detection/inference_graph/checkpoint"
    PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"
else:
    # from model
    PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

# args = parser.parse_args()

class DETECTOR:
    def __init__(self):
        print("init")
        rospy.init_node('tf2_object_detector_node', anonymous=True)
        rospy.Subscriber('/camera/image_rect_color', Image, self.imageCB)
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/od_res", Image, queue_size=2)

    def imageCB(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            #self.cv_image = cv2.resize(self.cv_image_ori, dsize=(640, 480), interpolation=cv2.INTER_AREA)
            # cv2.imshow("image",self.cv_image)
            # cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)

    def run(self):
        print("start load model")
        # self.test_image = cv2.imread('/home/ryu/models/research/object_detection/test_images/image1.jpg',cv2.IMREAD_COLOR)
        if(MODEL_TYPE == 'CKPT'):
            #from checkpoint
            configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
            model_config = configs['model']
            detection_model = model_builder.build(model_config=model_config, is_training=False)

            ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
            ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()
            if(MODEL_TYPE == 'CKPT'):
                def detect_fn(image):
                    """Detect objects in image."""

                    image, shapes = detection_model.preprocess(image)
                    prediction_dict = detection_model.predict(image, shapes)
                    detections = detection_model.postprocess(prediction_dict, shapes)

                    return detections
        else:
            # from model
            detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
        print("load model")
        start_t = time.clock()
        while not rospy.is_shutdown():
            # cv2.imshow("image",self.cv_image)
            self.image_infer = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)

            if(MODEL_TYPE == 'CKPT'):
                input_tensor = tf.convert_to_tensor(np.expand_dims(self.image_infer, 0), dtype=tf.float32)
            else:
                input_tensor = tf.convert_to_tensor(self.image_infer)
                input_tensor = input_tensor[tf.newaxis, ...]
            detections = detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                            for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            image_np_with_detections = self.cv_image.copy()

            if(MODEL_TYPE == 'CKPT'):
                viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes']+1,
                        detections['detection_scores'],
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=200,
                        min_score_thresh=.80,
                        agnostic_mode=False)
            else:
                viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes'],
                        detections['detection_scores'],
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=200,
                        min_score_thresh=.80,
                        agnostic_mode=False)
            infer_time = time.clock() - start_t
            infer_time = round(1/infer_time,2)
            start_t = time.clock()
            cv2.putText(image_np_with_detections, str(infer_time), (10, 10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255, 0, 0), 2)
            # image_np_with_detections = cv2.resize(image_np_with_detections, dsize=(640, 480), interpolation=cv2.INTER_AREA)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(image_np_with_detections,"bgr8"))
        #     cv2.imshow('object detection', image_np_with_detections)
        #     cv2.waitKey(1)
        # else:
        #     cv2.destroyAllWindows()

            # self.infer_img = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)

            # self.image_pub.publish(self.bridge.cv2_to_imgmsg(mask_cv,"bgr8"))


if __name__ == '__main__':
    detector = DETECTOR()
    time.sleep(2)
    detector.run()
