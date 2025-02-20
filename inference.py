import cv2
import torch
import mediapipe as mp
import numpy as np
import argparse
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import Batch
from STGCN import STGCN
import threading
import queue

def parse_data4model(keypoints, edge_index, num_frames):
    """
    Parses mediapipe keypoints into torch geometric Data objects.
    Args:
        keypoints (list): List of keypoint sequences.
        edge_index (np.array): Edge index for graph construction.
        num_frames (int): Number of frames in the sequence.
    Returns:
        Batch: A PyTorch Geometric Batch object.
    """
    data_frames = []
    x_data = torch.tensor(keypoints, dtype=torch.float32) #convert keypoints to tensor

    for t in range(num_frames):
        # Create a Data object for each frame
        data_frame = Data(x=x_data[t], edge_index=torch.tensor(edge_index), y=torch.tensor(0).unsqueeze(0))
        data_frames.append(data_frame)

    return Batch.from_data_list(data_frames) #create a batch from the list of data objects.

def inference_thread(frame_queue, prediction_queue, model, device, classes, EDGE_INDEX, NUM_FRAMES):
    """
    Inference thread to process frames and make predictions.
    """
    while True:
        frames = frame_queue.get() #get frames from the queue.
        if frames is None:  # Signal to stop thread
            break
        graphs = parse_data4model(frames, EDGE_INDEX, NUM_FRAMES).to(device) #parse data and move to device.
        pred = model(graphs) #make prediction.
        probabilities = F.softmax(pred, dim=1) #get probabilities.
        confidence, predicted_class = torch.max(probabilities, dim=1) #get confidence and predicted class.
        prediction_queue.put((classes[predicted_class.item()], confidence.item() * 100)) #put prediction in queue.

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Args")
    parser.add_argument("--sequence_len", type=int, default=20, help="Num of frames in the time window")
    parser.add_argument("--classes", nargs='+', type=str, default=['Pointing with one finger', 'Click with one finger', 'Throw up', 'Non-gesture'], help="Choose classes to be recognized by the model. Options see https://gibranbenitez.github.io/IPN_Hand/")
    parser.add_argument("--ckpt_path", type=str, default="last.pth", help="Path to model checkpoint .pth file")
    args = parser.parse_args()
    for attribute_name in vars(args).keys():
        globals()[attribute_name] = getattr(args, attribute_name) #create global variables from parsed arguments.

    NUM_FRAMES = sequence_len
    NUM_JOINTS = 21
    NUM_FEATURES = 3
    NUM_CLASSES = len(classes)

    # Initialize Mediapipe Hand Tracker
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    connections = mp_hands.HAND_CONNECTIONS
    EDGE_INDEX = np.array(list(connections)).T #create edge index from mediapipe connections.

    # Load model from checkpoint
    model = STGCN(in_channels=NUM_FEATURES, out_channels=NUM_CLASSES, num_joints=NUM_JOINTS, num_frames=NUM_FRAMES)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    state_dict = torch.load(ckpt_path, map_location=device)["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval() #set model to evaluation mode.

    # Open the webcam
    cap = cv2.VideoCapture(0)
    while not cap.isOpened():
        print("Error: Could not open webcam.")

    # Create queues for frame and prediction communication
    frame_queue = queue.Queue()
    prediction_queue = queue.Queue()

    # Start the inference thread
    inference_thread_instance = threading.Thread(target=inference_thread, args=(frame_queue, prediction_queue, model, device, classes, EDGE_INDEX, NUM_FRAMES))
    inference_thread_instance.daemon = True #thread will close when the main thread exits.
    inference_thread_instance.start()

    current_sequence = []
    frame_count = 0
    predicted_class = "None"
    confidence = 0

    # Main loop for capturing and displaying frames
    while True:
        ret, frame = cap.read() #read frame.
        if not ret:
            print("Error: Could not read frame.")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert to RGB.
        results = hands.process(image_rgb) #process the frame with mediapipe.

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0] #consider only the first detected hand.
            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z])
            current_sequence.append(keypoints)
            frame_count += 1

        if frame_count == NUM_FRAMES:
            frame_queue.put(current_sequence) #put current sequence in the queue.
            current_sequence = []
            frame_count = 0

        try:
            prediction, conf = prediction_queue.get_nowait() #get prediction from the queue.
            predicted_class = prediction
            confidence = conf
        except queue.Empty:
            pass #if queue is empty, do nothing.

        # Display prediction on the frame
        cv2.putText(frame, f"Predicted: {predicted_class} {confidence:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): #quit if 'q' key is pressed.
            break

    frame_queue.put(None)  # Signal to stop the inference thread
    cap.release()
    cv2.destroyAllWindows()