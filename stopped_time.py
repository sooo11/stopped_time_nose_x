import mediapipe as mp
import cv2
import time

# Import needed modules from osc4py3
from osc4py3.as_eventloop import *
from osc4py3 import oscbuildparse

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# Initialize variables to track time
true_start_time = None
total_true_duration = 0

def position_nose(landmarks):
    nose_tip = landmarks[mp_pose.PoseLandmark.NOSE]
    nose_tip_position = [nose_tip.x]
    print("Nose position:", nose_tip_position)
    check_nose_x_position(nose_tip.x)
    return nose_tip_position

def check_nose_x_position(nose_x):
    global true_start_time, total_true_duration
    current_time = time.time()

    if 0.4 <= nose_x <= 0.6:
        if true_start_time is None:
            true_start_time = current_time
        state = True
    else:
        if true_start_time is not None:
            duration = current_time - true_start_time
            total_true_duration = duration
            true_start_time = None
        state = False

    if state == False:
        duration = total_true_duration
    else:
        #duration = 0
        duration = current_time - true_start_time if true_start_time else 0
    
    print(state)
    #print("Total true duration:", total_true_duration, "seconds")
    print("Total true duration:", total_true_duration)

    
    # Send OSC message
    osc_process()
    osc_msg = oscbuildparse.OSCMessage("/nose", None, [nose_x, state, duration, total_true_duration])
    osc_send(osc_msg, "localhost")


def landmarks_to_keypoints(landmarks, definitions):
    keypoints = dict()
    for j, landmark in enumerate(landmarks):
        x = 1.0 - float(landmark.x)
        y = 1.0 - float(landmark.y)
        part = definitions(j).name
        keypoints[part] = [x, y]
    return keypoints

def detection_context(dev_id=0):
    cap = cv2.VideoCapture(dev_id)
    with mp_pose.Pose(smooth_landmarks=True) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = pose.process(image)

            if results.pose_landmarks:
                mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                nose_position = position_nose(results.pose_landmarks.landmark)
                
                keypoints = landmarks_to_keypoints(results.pose_landmarks.landmark, mp_pose.PoseLandmark)

                # for keypoint in keypoints:
                #     osc_process()
                #     osc_msg = oscbuildparse.OSCMessage("/keypoint/" + str(keypoint), None, keypoints[keypoint])
                #     osc_send(osc_msg, "localhost")

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.flip(image, 1)
            cv2.imshow('frame', image)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        osc_terminate()

if __name__ == '__main__':
    osc_startup()
    osc_udp_client("127.0.0.1", 9000, "localhost")
    detection_context()