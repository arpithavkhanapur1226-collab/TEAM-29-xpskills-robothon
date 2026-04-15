# TEAM-29-xpskills-robothon
We implemented a humanoid robot control system using MediaPipe. A webcam captures human motion, detects body landmarks, and computes joint angles. These angles are mapped to the robot’s joints, enabling real-time imitation of human movements using a control system.

**XP_robot.zip (Your Humanoid / MuJoCo Model)**
Project Structure

## 📂 Project Structure

```
.
├── XP_robot/
│   ├── XP_robot_with_actuators.xml
│   ├── XP_robot_upper_body.xml
│   └── assets/
│
├── robot_pose/
│   ├── pose_control.py
│   ├── pd_control.py
│   └── configs/
│
└── README.md
```


**1. Robot Model (MuJoCo XML)**
XP_robot_with_actuators.xml
Full humanoid robot model
Contains joints and actuators
Used for simulation
Assets Folder

**Contains meshes and visual elements**
Control System
1.MediaPipe
Captures human pose using webcam
Extracts joint angles

2.pose_control.py
Main execution file
Maps human joint angles → robot joints

3.pd_control.py
Implements Proportional-Derivative (PD) controller
Ensures stability and smooth movement

**Workflow**
Webcam Input (MediaPipe)

        ↓
        
Joint Angle Extraction

        ↓
        
pose_control.py

        ↓
        
PD Controller

        ↓
        
MuJoCo Robot Simulation

