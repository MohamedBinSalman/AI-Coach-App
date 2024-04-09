# AI Coach App
## Overview


The Bicep Curl Counter project utilizes Mediapipe's Pose Estimation Model Blazepose to accurately detect and track key landmarks of the human body in a video.
Specifically, the 3D coordinates of the wrist, elbow, and shoulder keypoints are extracted to monitor the movement of the arms during bicep curls.

## Demo
The Following GIF shows how the project works:

![output demo](https://github.com/MohamedBinSalman/AI-Coach-App/blob/main/output/output.gif)



## Dependencies
- Python 3
- OpenCV (cv2)
- NumPy
- MediaPipe



### Installing Dependencies

You can install the required dependencies using the `requirements.txt` file provided:

```bash
pip install -r requirements.txt
```



## Usage

### Running with webcam

```bash
python app.py 
```

### Running without analysis

```bash
python app.py dataset/Bicep.mp4
```

### Running with analysis

```bash
python app.py dataset/Bicep.mp4 <correct_left_count> <correct_right_count> <incorrect_left_count> <incorrect_right_count>
```





## Features

- Real-time visualization of bicep curls count and correctness analysis.
- Supports both left and right bicep curl counts.
- Tracks angles of shoulder, elbow, and wrist for accurate counting.


## Project Details 

Check out the following link to see a walkthrough of the project, including a presentation and code explanation.
[Click here!](https://drive.google.com/file/d/1AwfLSHKh9_AAeYGBisp1oQ0UKX1m695i/view?usp=drivesdk)







