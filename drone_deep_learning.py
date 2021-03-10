from djitellopy import Tello
import numpy as np
from FaceRecognition import *
import dlib


class MyDrone(Tello):
    def __init__(self, location):
        super().__init__()
        self.loc = location
        self.connect()
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 0
        self.streamoff()
        self.streamon()
        print(self.get_battery())
        print('Cuda Devices', dlib.cuda.get_num_devices())
        self.kwn_names, self.kwn_encoding = load_known_faces(self.loc)

    def get_fame(self, w, h):
        frame = self.get_frame_read()
        frame = frame.frame
        frame = cv2.resize(frame, (w, h))

        return frame

    def detect_face(self, img):
        curr_face_loc, curr_encode_img = load_encode_loc(img, 2)

        face_list = []
        face_area = []
        print('face loc', curr_face_loc)

        for (top, right, bottom, left), encode_img in zip(curr_face_loc, curr_encode_img):
            matches, face_dis = get_match_facedis(self.kwn_encoding, encode_img)
            idx = np.argmin(face_dis)

            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 2), 2)
            cv2.rectangle(img, (left, bottom - 25), (right, bottom), (0, 255, 2), cv2.FILLED)

            w = right - left
            h = bottom - top
            cx = left + w // 2
            cy = top + h // 2
            area = w * h
            if matches[idx]:
                name = self.kwn_names[idx].upper()
                cv2.putText(img, name, (left + 2, bottom - 2), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (255, 255,
                             255), 2)
            else:
                cv2.putText(img, 'Unknown', (left + 2, bottom - 2), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (255, 255,
                             255), 2)
            face_list.append([cx, cy])
            face_area.append(area)

        if len(face_area) != 0:
            i = face_area.index(max(face_area))

            return img, [face_list[i], face_area[i]]

        else:
            return img, [[0, 0], 0]

    def track_face(self, info, w, h, pid, p_error, p_up_dwn_error, p_for_back_error):
        error_yaw = info[0][0] - w // 2
        speed_yaw = pid[0] * error_yaw + pid[2] * (error_yaw - p_error)
        speed_yaw = int(np.clip(speed_yaw, -100, 100))

        error_up_dwn = info[0][1] - h // 2
        speed_up_dwn = pid[0] * error_up_dwn + pid[2] * (error_up_dwn - p_up_dwn_error)
        speed_up_dwn = int(np.clip(speed_up_dwn, -100, 100))

        error_for_back = info[1] - 7000
        speed_for_back = pid[0] * error_for_back + pid[2] * (error_for_back - p_for_back_error)
        speed_for_back = int(np.clip(speed_for_back, -30, 30))
        if info[0][0] != 0:
            self.yaw_velocity = speed_yaw
            self.up_down_velocity = - speed_up_dwn
            self.for_back_velocity = - speed_for_back


        else:
            self.for_back_velocity = 0
            self.left_right_velocity = 0
            self.up_down_velocity = 0
            self.yaw_velocity = 0

        if self.send_rc_control:
            self.send_rc_control(self.left_right_velocity,
                                 self.for_back_velocity,
                                 self.up_down_velocity,
                                 self.yaw_velocity)

        return error_yaw, error_up_dwn, error_for_back
