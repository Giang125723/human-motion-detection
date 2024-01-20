from person_detector import *
from motion_detection import *
from datetime import datetime


# Các hàm cập nhật (check lại)
# Detection event
def detection_event(detection_flag):
    if detection_flag is False:
        detection_flag = True
        print('detection ON')
    else:
        detection_flag = False
        counter_flag = False
        acc_flag = False
        print('detection OFF')
    return detection_flag, counter_flag, acc_flag


def motion_det_event(motion_flag, frame):
    if motion_flag is False:
        motion_flag = True
        print('human motion detection ON')
        prev_frame = frame
    else:
        motion_flag = False
        print('human detection ON')
    return motion_flag, prev_frame


def counter_event(counter_flag):
    if counter_flag is False:
        counter_flag = True
        print('counter ON')
        c_frame_id = 0
    else:
        counter_flag = False
        print('counter OFF')
    return counter_flag, c_frame_id


def acc_event(acc_flag):
    if acc_flag is False:
        acc_flag = True
        a_frame_id = 0
        print('Calculate acc ON')
    else:
        acc_flag = False
        print('Calculate acc OFF')
    return acc_flag, a_frame_id


# Hàm lấy sự kiện từ bàn phím để check (chưa chạy)
# def handle_key_event(e):
#     global detection_flag, motion_flag, frame, counter_flag, a_frame_id, acc_flag, a_frame_id
#     key_name = e.name
#     print(f"Key {key_name} pressed")  # Hiển thị thông báo khi có sự kiện nhấn phím
#
#     # Cập nhật thông tin cho các hàm
#     if key_name == 'a':
#         detection_flag, counter_flag, acc_flag = detection_event(detection_flag)
#     elif key_name == 'b':
#         motion_flag, prev_frame = motion_det_event(motion_flag, frame)
#     elif key_name =='c':
#         counter_flag, c_frame_id = counter_event(counter_flag)
#     elif key_name == 'd':
#         acc_flag, a_frame_id = acc_event(acc_flag)


# Hàm chính để xử lý chung
def app_processing(video_path,
                   detection_flag=False, score_threshold=AI_CONFIG['score'],
                   motion_flag=False, motion_threshold=0.3,
                   counter_flag=False, count_sdist=10,
                   acc_flag=False, acc_sdist=2):
    """

    Parameters
    ----------
    video_path: video/rtsp/webcam
    detection_flag: cờ để đánh dấu cho Bộ phát hiện
    score_threshold: Tham số Ngưỡng phát hiện (0.25 --> 0.90, +- 0.01)
    motion_flag: cờ cho việc chọn Người/Người chuyển động (True)
    motion_threshold: Tham số ngưỡng xác định người chuyển động (0.1 --> 0.5, +- 0.05)
    counter_flag: cờ cho Bộ đếm người
    count_sdist: Khoảng cách lấy mẫu (s) (1 --> 30, +-1)
    acc_flag: cờ cho Bộ Tính điểm chính xác TB
    acc_sdist: Khoảng cách lấy mẫu (frame) (1 --> 100, +-1)

    Returns
    -------

    """

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    _, prev_frame = cap.read()
    total_frame_id = 0
    c_frame_id = 0
    a_frame_id = 0
    # Process first frame to get dimensions
    while True:
        return_value, frame = cap.read()
        if return_value is None:
            raise ValueError("No image! Try with another video format")
        else:
            total_frame_id += 1
            # Khi bật bất cứ 1 bộ nào
            if detection_flag or counter_flag or acc_flag:
                # Phát hiện người (human detection)
                pred_bbox = image_processing(frame, score_thred=score_threshold)
                bboxes_list, conf_list = utils.get_list_pred_infos(frame, pred_bbox)
                # Nếu bộ nhận diện người chuyển động
                if motion_flag:
                    indx, prev_frame = calculate_difference(prev_frame, frame, bboxes_list, motion_thres=motion_threshold)
                    m_bboxes, m_conf = get_human_motion_infos(indx, bboxes_list, conf_list)
                    bboxes_list, conf_list = m_bboxes, m_conf
                # Nếu bật Bộ đếm
                if counter_flag:
                    if c_frame_id % int(fps*count_sdist) == 0:
                        counter = len(conf_list)
                        print(str(datetime.now()) + ' frame thứ ' + str(total_frame_id) + ' có ' + str(
                            counter) + ' người')
                    c_frame_id += 1

                # Nếu bật Bộ tính điểm chính xác:
                if acc_flag:
                    if a_frame_id % acc_sdist == 0 and len(conf_list) != 0:
                        average = np.mean(conf_list)
                        # 
                        print(str(datetime.now()) + ' frame thứ ' + str(total_frame_id) + ' có độ chính xác TB ' + str(
                            average))
                    a_frame_id += 1
                # Show bằng OpenCV
                frame = utils.draw_bboxes(frame, bboxes_list, conf_list)
                cv2.imshow("result", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # Nếu không thì chỉ hiển thị video nguồn
            else:
                cv2.imshow("result", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Lắng nghe sự kiện nhấn phím
    # keyboard.hook(handle_key_event)
    app_processing('./data/persons_video.mp4')

