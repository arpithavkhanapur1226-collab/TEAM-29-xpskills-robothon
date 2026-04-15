import cv2
import mediapipe as mp
import mujoco
import mujoco.viewer
import numpy as np
import time

# ================= LOAD =================
model = mujoco.MjModel.from_xml_path("model/humanoid.xml")
data  = mujoco.MjData(model)

mujoco.mj_resetData(model, data)
data.qpos[2] = 1.3

# ================= MEDIAPIPE =================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ================= MAIN =================
with mujoco.viewer.launch_passive(model, data) as viewer:

    viewer.cam.lookat = [0, 0, 1]
    viewer.cam.distance = 3
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -15

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        # ===== STABILITY =====
        data.qpos[0] = 0
        data.qpos[1] = 0
        data.qvel[:] *= 0.3

        data.ctrl[:] = 0

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            mp_draw.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            def P(i):
                return lm[i]

            # ================= ARMS (FIXED LOGIC) =================
            # Use vertical difference (more stable than angle)

            # LEFT ARM
            left_shoulder = P(11)
            left_wrist = P(15)

            left_move = (left_shoulder.y - left_wrist.y)

            data.ctrl[0] = np.clip(left_move * 3, -1, 1)   # shoulder
            data.ctrl[2] = np.clip(left_move * 3, -1, 1)   # elbow

            # RIGHT ARM
            right_shoulder = P(12)
            right_wrist = P(16)

            right_move = (right_shoulder.y - right_wrist.y)

            data.ctrl[1] = np.clip(right_move * 3, -1, 1)
            data.ctrl[3] = np.clip(right_move * 3, -1, 1)

            # ================= LEGS (FIXED LOGIC) =================
            # based on knee height difference

            # LEFT LEG
            left_hip = P(23)
            left_knee = P(25)

            left_leg_move = (left_hip.y - left_knee.y)

            data.ctrl[4] = np.clip(left_leg_move * 2, -0.5, 0.5)  # hip
            data.ctrl[6] = np.clip(left_leg_move * 3, -1, 1)      # knee

            # RIGHT LEG
            right_hip = P(24)
            right_knee = P(26)

            right_leg_move = (right_hip.y - right_knee.y)

            data.ctrl[5] = np.clip(right_leg_move * 2, -0.5, 0.5)
            data.ctrl[7] = np.clip(right_leg_move * 3, -1, 1)

        # ===== STEP =====
        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(0.04)

        cv2.imshow("MediaPipe Pose", frame)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()
