import cv2
import mediapipe as mp
import mujoco
import mujoco.viewer
import numpy as np

# Load MuJoCo model
model = mujoco.MjModel.from_xml_path("humanoid.xml")
data = mujoco.MjData(model)

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(1)

# Simple smoothing
prev_ctrl = np.zeros(model.nu)

def smooth_control(new_ctrl, alpha=0.2):
    global prev_ctrl
    smoothed = alpha * new_ctrl + (1 - alpha) * prev_ctrl
    prev_ctrl = smoothed
    return smoothed

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        ctrl = np.zeros(model.nu)

        # ================== STANDING CONTROL ==================
        # Keep robot upright (legs)
        ctrl[4] = -0.6   # left hip
        ctrl[5] = -0.6   # right hip
        ctrl[6] = 0.9    # left knee
        ctrl[7] = 0.9    # right knee

        # ================== POSE CONTROL ==================
        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark

            # Get landmarks
            l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_el = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
            r_el = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]

            # Convert pose to control signals
            ctrl[0] = (0.5 - l_sh.y) * 2      # left shoulder
            ctrl[1] = (0.5 - r_sh.y) * 2      # right shoulder

            ctrl[2] = (l_el.y - l_sh.y) * 2   # left elbow
            ctrl[3] = (r_el.y - r_sh.y) * 2   # right elbow

        # Smooth movement
        data.ctrl[:] = smooth_control(ctrl)

        # Step simulation
        mujoco.mj_step(model, data)

        # Show camera
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        viewer.sync()

cap.release()
cv2.destroyAllWindows()
