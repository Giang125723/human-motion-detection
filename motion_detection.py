import cv2
import numpy as np


# Hàm lấy danh sách các ảnh crop chứa người dự đoán
def get_list_human_detection(frame, bboxes_list):
    # Chuyển đổi frame gốc về gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)
    crop_list = []
    for i, bbox in enumerate(bboxes_list):
        xmin, ymin, xmax, ymax = bbox
        crop = gray[ymin:ymax, xmin: xmax]
        crop_list.append(crop)
    return crop_list


# Hàm để tính toán sự thay đổi giữa hai khung hình
def calculate_difference(prev_frame, current_frame, bboxes_list, motion_thres):
    """
    Parameters
    ----------
    prev_frame: frame lấy ngày trước
    current_frame: frame hiện tại
    bboxes_list: danh sách người được phát hiện tại frame_current
    motion_thres: ngưỡng phát hiện chuyển động

    Returns
    -------
    inds: danh sách các chỉ số của vùng người chuyển động
    Cập nhật prev_frame
    """
    inds = []
    if bboxes_list is not None:
        prev_list = get_list_human_detection(prev_frame, bboxes_list)
        current_list = get_list_human_detection(current_frame, bboxes_list)
        # Tính toán sự thay đổi giữa hai ảnh

        for i in range(len(bboxes_list)):
            w_diff = cv2.absdiff(prev_list[i], current_list[i])

            _, threshold_diff = cv2.threshold(w_diff, 30, 255, cv2.THRESH_BINARY)
            motion_ratio = np.count_nonzero(w_diff) / (current_list[i].shape[0] * current_list[i].shape[1])
            # Nếu vùng thỏa mãn ngưỡng thì cho vào danh sách đánh dấu
            if (motion_ratio) >= motion_thres:
                inds.append(i)

    prev_frame = current_frame
    return inds, prev_frame


# Lấy thông tin danh sách người chuyển động:
def get_human_motion_infos(indx, bboxes, conf_list):
    m_bboxes = []
    m_conf = []
    for i in range(len(conf_list)):
        for j in indx:          # indx: danh sách lấy các người chuyền động
            if i == j:
                m_bboxes.append(bboxes[i])
                m_conf.append(conf_list[i])
    return m_bboxes, m_conf
