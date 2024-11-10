"""
preAnnotations.py: Script for Label Studio ML to generate pre-annotations using YOLOv4.
Requirements: darknet's darknet.py and darknet_images.py

-> Make sure to configure paths, variables and `prediction` variable before running.
"""

import cv2
import sys

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import DATA_UNDEFINED_NAME, get_image_size, get_single_tag_keys

sys.path.insert(0, '/full/path/to/darknet')

import darknet, darknet_images

YOLO_CONFIG_FILE = r'/full/path/to/darknet/config_file.cfg'
YOLO_DATA_FILE = r'/full/path/to/darknet/obj.data'
YOLO_WEIGHTS_FILE = r'/full/path/to/darknet/weight_file.weights'
PRE_ANN_THRESHOLD = 0.50


class MyModel(LabelStudioMLBase):
    def __init__(self,
                 config_file=None,
                 data_file=None,
                 weights_file=None,
                 threshold=0.25,
                 **kwargs):
        # don't forget to initialize base class...
        super(MyModel, self).__init__(**kwargs)

        # Get config, data and weights files
        config_file = config_file or YOLO_CONFIG_FILE
        data_file = data_file or YOLO_DATA_FILE
        weights_file = weights_file or YOLO_WEIGHTS_FILE

        # Get variables from Label Studio configurations
        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config,
            'RectangleLabels',
            'Image')

        # Logs files locations
        print("Loading model configuration from: ", config_file)
        print("Loading data file from: ", data_file)
        print("Loading weights file from: ", weights_file)

        # Load darknet network
        self.model = darknet.load_network(config_file, data_file, weights_file)

        # Define threshold
        self.threshold = threshold

    def predict(self, tasks, **kwargs):
        # assert len(tasks) == 1
        task = tasks[0]

        img_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        img_path = self.get_local_path(img_url)
        img_shape = cv2.imread(img_path).shape

        # It's possible to add a threshold for the pre-annotations by adding a 3rd parameter to `pre_ann_call()`
        predictions, width_ratio, height_ratio = pre_ann_call(self.model, img_path, PRE_ANN_THRESHOLD)

        # Get annotation tag first, and extract from_name/to_name keys from the labeling config to make predictions
        # from_name, to_name = self.from_name, self.to_name

        # Instantiate list for pre-annotations
        result = []

        for label, confidence, bbox in predictions:
            # Skip step if there are no detections in image
            if not bbox:
                continue

            pixel_x, pixel_y, pixel_height, pixel_width = label_studio_bbox(bbox)

            # for each task, return classification results in the form of "choices" pre-annotations
            result.append({
                'from_name': self.from_name, 'to_name': self.to_name,
                'type': "rectanglelabels",
                'value': {
                    'rectanglelabels': [label],
                    'x': pixel_x, 'y': pixel_y,
                    'width': pixel_width, 'height': pixel_height
                },
                # 'score': confidence
            })

        # Add a name to key `model_version` and the score it achieved to `score` in order to keep a model's history
        predictions = [{'result': result, 'model_version': "preAnn_2023.09.29_v0400", 'score': 0.8404}]

        print(predictions)
        return predictions


def label_studio_bbox(bbox):
    """
    Converts YOLO bbox format to Label Studio format.

    `bbox`: YOLO bbox;

    In (416 / 2000) it's the ratio between the network's input layer (416 px)
    and the size of the original image (2000 px).
    """
    # Convert elements of YOLO bbox
    x, y, width, height = bbox

    # Convert to Label Studio units
    pixel_x = (x - width/2) * (416 / 2000)
    pixel_y = (y - height/2) * (416 / 2000)
    pixel_width = width * (416 / 2000)
    pixel_height = height * (416 / 2000)

    return pixel_x, pixel_y, pixel_height, pixel_width


# noinspection DuplicatedCode
def image_pre_annotations(img_path, network, class_names, thresh):
    # Based on `darknet_images.py`'s `image_detection()` method

    # Get network's width and ratio
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    # Loads image from path
    image = cv2.imread(img_path)

    # Get image ratios to convert bounding boxes back to proper size
    img_height, img_width, _ = image.shape
    width_ratio = img_width / width
    height_ratio = img_height / height

    # Convert image to RGB and resize for network input layer
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)

    return detections, width_ratio, height_ratio


def pre_ann_call(model, img_path, thresh=0.50):
    # Load network from given configs
    network, class_names, _ = model

    # Get detections and both width and height ratios relative of original image and resized for network input
    return image_pre_annotations(img_path, network, class_names, thresh=thresh)
