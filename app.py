import sys
import cv2
import numpy as np
import mediapipe as mp

def calc_angle(a, b, c):
    ''' 
    Calculate the angle between three points.
    
    Arguments:
        a,b,c -- Values (x,y,z, visibility) of the three points a, b, and c which will be used to calculate the
                vectors ab and bc where 'b' will be 'elbow', 'a' will be shoulder and 'c' will be wrist.
        
    Returns:
        theta : Angle in degrees between the lines joined by coordinates (a,b) and (b,c)
    '''
    a = np.array([a.x, a.y])    # Reduce 3D point to 2D
    b = np.array([b.x, b.y])    # Reduce 3D point to 2D
    c = np.array([c.x, c.y])    # Reduce 3D point to 2D

    ab = np.subtract(a, b)
    bc = np.subtract(b, c)
    
    theta = np.arccos(np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc)))     # A.B = |A||B|cos(x) where x is the angle b/w A and B
    theta = 180 - np.degrees(theta)    # Convert radians to degrees
    return np.round(theta, 2)

def count_bicep_curls(video_path, correct_left=0, correct_right=0, incorrect_left=0, incorrect_right=0):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    incorrect_left_flag = 'up'
    incorrct_left_count = 0
    correct_left_count = 0
    incorrect_left_info = []
    
    incorrect_right_flag = 'up'
    incorrct_right_count = 0
    correct_right_count = 0
    incorrect_right_info = []

    test = False
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            
            left_angle = calc_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calc_angle(right_shoulder, right_elbow, right_wrist)
            
            # Count left bicep curls
            if left_angle > 150:
                incorrect_left_flag = 'down'
            
            if left_angle < 150 and left_angle > 60 and test == True:
                incorrect_left_flag = 'up'

            if left_angle < 60 and incorrect_left_flag == 'up':
                incorrct_left_count += 1
                incorrect_left_info.append((incorrct_left_count, incorrect_left_flag, left_angle))
                incorrect_left_flag = 'down'
                test = True
                
                    
            if left_angle > 150:
                correct_left_flag='down'
            if left_angle < 60 and correct_left_flag == 'down':  
                correct_left_count += 1
                correct_left_flag = 'up'

            # Count right bicep curls
            if right_angle > 150:
                incorrect_right_flag = 'down'
            
            if right_angle < 150 and right_angle > 60 and test == True:
                incorrect_right_flag = 'up'

            if right_angle < 60 and incorrect_right_flag == 'up':
                incorrct_right_count += 1
                incorrect_right_info.append((incorrct_right_count, incorrect_right_flag, right_angle))
                incorrect_right_flag = 'down'
                test = True
                
                    
            if right_angle > 150:
                correct_right_flag='down'
            if right_angle < 60 and correct_right_flag == 'down':  
                correct_right_count += 1
                correct_right_flag = 'up'
                 
        except Exception as e:
            print(f"Error processing frame: {e}")

        cv2.rectangle(image, (0, 0), (150, 40), (0, 255, 0), -1)
        cv2.putText(image,'L     ' + str(correct_left_count),
                    (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.rectangle(image, (0, 50), (150, 90), (0, 255, 0), -1)
        cv2.putText(image,'R     ' + str(correct_right_count),
                    (4, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.rectangle(image, (0, 100), (150, 140), (0, 0, 255), -1)
        cv2.putText(image,'IL     ' + str(incorrct_left_count),
                    (1, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.rectangle(image, (0, 150), (150, 190), (0, 0, 255), -1)
        cv2.putText(image,'IR     ' + str(incorrct_right_count),
                    (0, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

       
        cv2.imshow('Bicep Curl Counter', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    total_correct_count = correct_left_count + correct_right_count
    total_incorrect_count = incorrct_left_count + incorrct_right_count

    correct_accuracy = (total_correct_count / (correct_left + correct_right)) * 100 if (correct_left + correct_right) != 0 else 100
    incorrect_accuracy = (total_incorrect_count / (incorrect_left + incorrect_right)) * 100 if (incorrect_left + incorrect_right) != 0 else 100

    

if __name__ == '__main__':
    if len(sys.argv) == 2:
        video_path = sys.argv[1]
        count_bicep_curls(video_path)
    elif len(sys.argv) == 6:
        video_path = sys.argv[1]
        correct_left = int(sys.argv[2])
        correct_right = int(sys.argv[3])
        incorrect_left = int(sys.argv[4])
        incorrect_right = int(sys.argv[5])
        count_bicep_curls(video_path, correct_left, correct_right, incorrect_left, incorrect_right)
    else:
        print("Usage:")
        print("For running without analysis: python app.py dataset/Bicep.mp4")
        print("For running with analysis: python app.py dataset/Bicep.mp4 <correct_left_count> <correct_right_count> <incorrect_left_count> <incorrect_right_count>")
