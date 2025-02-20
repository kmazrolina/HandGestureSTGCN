import pandas as pd
import os
import numpy as np
import mediapipe as mp
import cv2
import argparse


if(__name__ == "__main__"):

    parser = argparse.ArgumentParser(description="Args")
    parser.add_argument(
        "--data_path", type=str, default="data",
        help="Path to IPN Hand Dataset, containing annotations/ and frames/",
    )
    parser.add_argument(
        "--sequence_len", type=int, default=20,
        help="Num of frames in the time window",
    )
    parser.add_argument(
        "--offset", type=int, default=10,
        help="Step in number of frames for shifting time window",
    )
    parser.add_argument(
        "--classes", nargs='+', type=str, default=['B0A', 'G01', 'G03', 'D0X'],
        help="Choose classes to be recognized by the model. Options see https://gibranbenitez.github.io/IPN_Hand/",
    )
    args = parser.parse_args()
    # create global variables without the args prefix
    for attribute_name in vars(args).keys():
        globals()[attribute_name] = getattr(args, attribute_name)


    # Annotations contain 
    # - video ID
    # - gesture classes contained in the video
    # - start and end frames of gesture classes in the video
    annotations = pd.read_csv(f'{data_path}/annotations/Annot_List.txt')
    filtered_annot = annotations[annotations['label'].isin(classes)] 
    print(filtered_annot.head())
    print(filtered_annot['label'].unique())
    class_dict = {class_label:i for i, class_label in enumerate(classes)}
    print(class_dict)

    # Initialize Mediapipe Hand Tracker
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils


    hand_not_detected = 0
    num_vids = len(filtered_annot)
    graph_data = []
    labels_data = []

    step_save = 10

    # Iterate over gesture videos
    for vid_count, annot in enumerate(filtered_annot.itertuples()):
        video_id = annot.video
        class_end_frame = int(annot.t_end)
        class_start_frame = int(annot.t_start)
        
        
        # Iterate over time windows in given gesture
        for gesture_start_frame in range(class_start_frame, class_end_frame + 1, offset):
            # skip the last shorter window
            if (gesture_start_frame + sequence_len) > (class_end_frame + 1):
                break
            else:
                seq_data = []
                #iterate over frames in window
                for frame_num in range(gesture_start_frame, gesture_start_frame + sequence_len + 1):
                    frame_file = f"{data_path}/frames/{video_id}/{video_id}_{str(frame_num).zfill(6)}.jpg"
                    
                    # Read Frame
                    image = cv2.imread(frame_file)
                    if image is None:
                        print(f"Frame {frame_file} not found.")
                        continue
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Process Frame for Hand Keypoints
                    results = hands.process(image_rgb)
                    
                    # Extract Keypoints if a Hand is Detected
                    if results.multi_hand_landmarks:
                        keypoints = []
                        for hand_landmarks in results.multi_hand_landmarks:
                            for landmark in hand_landmarks.landmark:
                                keypoints.append([landmark.x, landmark.y, landmark.z])  # Collect x, y, z (normalized values)  # Collect x, y (normalized values)
                        seq_data.append(keypoints)
                    else:
                        hand_not_detected +=1
                if(len(seq_data) == sequence_len):
                    #only append homogenous data
                    graph_data.append(seq_data)
                    labels_data.append(class_dict[annot.label])
                    
        
        print(f'Processed videos: {round(((vid_count / num_vids) * 100), 2 )} %') 
        
        #save intermediate steps due to long execution time
        if (vid_count % step_save == 0): 
            np.save(f'{data_path}/graph_data.npy',np.array(graph_data))
            np.save(f'{data_path}/labels_data.npy',np.array(labels_data))


    # Save graph data to disk  
    np.save(f'{data_path}/graph_data.npy',np.array(graph_data))
    np.save(f'{data_path}/labels_data.npy',np.array(labels_data))

    
