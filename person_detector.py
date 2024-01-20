import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import core.utils as utils
from core.yolov4 import filter_boxes
from config import AI_CONFIG

### Đưa frame vào xử lý, đầu ra sẽ là kết quả dự đoán và các thành phần cân thiết

# 1. Check GPU
physical_devices = tf.config.list_physical_devices('GPU')  # lấy danh sách GPU;

if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

else:
    print("No GPU")

# 2. Load model
m_interpreter = tf.lite.Interpreter(model_path=AI_CONFIG['weights'])
m_interpreter.allocate_tensors()
input_model = m_interpreter.get_input_details()
output_model = m_interpreter.get_output_details()


# 3. Dùng model để xử lý frame để đưa ra kết quả dự đoán
def model_inference(image_input, interpreter, input_details, output_details):
    interpreter.set_tensor(input_details[0]['index'], image_input)
    interpreter.invoke()
    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                    input_shape=tf.constant([AI_CONFIG['size'], AI_CONFIG['size']]))

    return boxes, pred_conf


def image_processing(frame, score_thred=AI_CONFIG['score']):
    # Preprocessing
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert BGR to RGB for 'PIL'

    image_data = cv2.resize(frame, (AI_CONFIG['size'], AI_CONFIG['size']))
    image_data = image_data / 255.

    image_list = []

    for i in range(1):
        image_list.append(image_data)
    image_list = np.asarray(image_list).astype(np.float32)

    """ Model inference 
    """
    boxes, pred_conf = model_inference(image_list, m_interpreter, input_model, output_model)

    """ bbox
    """
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=AI_CONFIG['iou'],
        score_threshold=score_thred
    )

    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

    return pred_bbox
