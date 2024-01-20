from person_detector import *
from motion_detection import *
import keyboard


def detection_event(e):
    key_name = e.name


def main():
    vid = cv2.VideoCapture('./data/persons_video.mp4')
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    total_frame_id = 0

    _, prev_frame = vid.read()
    while True:
        return_value, frame = vid.read()
        if return_value is None:
            raise ValueError("No image! Try with another video format")
        else:
            total_frame_id += 1
            pred_bbox = image_processing(frame)
            bboxes, conf_list = utils.get_list_pred_infos(frame, pred_bbox)

            indx, prev_frame = calculate_difference(prev_frame, frame, bboxes, motion_thres=0.3)

            m_bboxes, m_conf = get_human_motion_infos(indx, bboxes, conf_list)

            image = utils.draw_bboxes(frame, m_bboxes, m_conf)

            cv2.imshow("result", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
