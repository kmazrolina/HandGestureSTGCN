#!/usr/bin/env python
# coding: utf-8

# # Hand Gesture Classification 
# **using Spatio-Temporal Graph Convolutional Network**
# 
# 
# **Dataset: IPN Hand Gestures**
# https://gibranbenitez.github.io/IPN_Hand/

# ## Imports and config

# In[1]:


import torch
from torch_geometric.data import Data


# In[101]:


import pandas as pd
import os
import argparse
import numpy as np
import mediapipe as mp
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#check if run in jupyter notebook (interactive) or as python file
def is_interactive(): 
    import __main__ as main
    return not hasattr(main, '__file__')


# In[3]:


if is_interactive():
    jupyter_args = f"--model_name=IPN_Hand_STGCN \
    --classes B0A G01 G03 D0X "
    print(jupyter_args)
    jupyter_args = jupyter_args.split()


# In[4]:


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="IPN_Hand_STGCN",
    help="will save ckpt for model in train_logs/model_name",
)
parser.add_argument(
    "--data_path", type=str, default="data",
    help="Path to IPN Hand Dataset, containing annotations/ and frames/",
)
parser.add_argument(
    "--classes", nargs='+', type=str, default=['B0A', 'G01', 'G03', 'D0X'],
    help="Choose classes to be recognized by the model. Options see https://gibranbenitez.github.io/IPN_Hand/",
)


if is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()
# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)

# seed all random functions
seed=42
np.random.seed(seed)


# make ckpt output directory
os.makedirs(f"train_logs/{model_name}",exist_ok=True)


# ## Loading IPN Hand Gesture Dataset

# Annotations contain 
# - video ID
# - gesture classes contained in the video
# - start and end frames of gesture classes in the video

# In[5]:


annotations = pd.read_csv(f'{data_path}/annotations/Annot_List.txt')
filtered_annot = annotations[annotations['label'].isin(classes)] 
print(filtered_annot.head())
print(filtered_annot['label'].unique())


# In[88]:


class_dict = {class_label:i for i, class_label in enumerate(classes)}


# ### Extract Hand Keypoint Graphs from Dataframes

# #### Example Frame

# In[102]:


if is_interactive():

    example_frame_file = f"{data_path}/frames/1CM1_3_R_#228/1CM1_3_R_#228_000258.jpg"
    # Read the image using OpenCV
    example_frame = cv2.imread(example_frame_file)
    
    # Convert the image from BGR (OpenCV's default) to RGB (for Matplotlib)
    example_frame_rgb = cv2.cvtColor(example_frame, cv2.COLOR_BGR2RGB)
    
    # Display the image
    plt.imshow(example_frame_rgb)
    plt.axis('off')  # Optionally remove the axis for a cleaner display
    plt.show()


# Gesture sequences are interpolated to the same length

# In[47]:


MEAN_SEQ_LEN = int(filtered_annot['frames'].mean())

print(MEAN_SEQ_LEN)


# In[103]:


# Load preprocessed graph data
graph_data = np.load(os.path.join(data_path,'graph_data.npy'))
labels_data = np.load(os.path.join(data_path,'labels_data.npy'))


# In[48]:


# Access hand connections
mp_hands = mp.solutions.hands
connections = mp_hands.HAND_CONNECTIONS
EDGE_INDEX = np.array(list(connections)).T

# Print connections
print("Hand Connections:", connections)


# In[49]:


NUM_FRAMES = MEAN_SEQ_LEN
NUM_JOINTS = 21  # Number of joints in the hand
NUM_FEATURES = 2  # (x, y) positions for each joint
NUM_CLASSES = len(classes)  # hand geature classes


# In[ ]:


# Visualize the graph 
G = nx.Graph()
for i in range(NUM_JOINTS):
    G.add_node(i)
for edge in EDGE_INDEX.T:
    G.add_edge(edge[0].item(), edge[1].item())

# Visualizing the skeleton graph
plt.figure(figsize=(6, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=14)
plt.title('Hand Graph (Frame 1)')
plt.savefig(f"hand_graph.png", dpi=300)


# ### Parsing data into troch_geometric

# In[105]:


def parse_data4model(data, labels_data):
        
    all_data_frames = []  # Store data_frames for all batches
    
    for gesture_data, label in zip(data, labels_data):
    
        # Extract x_data from the batch
        x_data = torch.tensor(gesture_data, dtype=torch.float32).reshape(NUM_FRAMES, NUM_JOINTS, NUM_FEATURES)
    
        # Create data_frames for each time step
        data_frames = []
        for t in range(NUM_FRAMES):
            data_frame = Data(x=x_data[t], edge_index=torch.tensor(EDGE_INDEX), y=torch.tensor(label).unsqueeze(0))
            data_frames.append(data_frame)
    
        all_data_frames.append(data_frames)
    
    # Outputs
    print("Number of gesture vids:", len(all_data_frames))
    print("Number of frames in the first vid:", len(all_data_frames[0]))
    print("Shape of x_data in the first frame of the first batch:", all_data_frames[0][0].x.shape)
    return data_frames

data_frames = parse_data4model(graph_data, labels_data)
train_data_frames, test_data_frames = train_test_split(data_frames, test_size=0.2, random_state=42)


# # Create the model

# In[93]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

# Step 3: Define the ST-GCN Model
class STGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_joints, num_frames):
        super(STGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)  # First graph convolution layer
        self.conv2 = GCNConv(64, 128)  # Second graph convolution layer
        self.temporal_conv = nn.Conv2d(128, 128, kernel_size=(3, 1), padding=(1, 0))  # Temporal convolution
        self.fc = nn.Linear(128 * num_frames * num_joints, out_channels)  # Fully connected layer for classification
        self.num_frames = num_frames


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Graph convolution layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Reshape x to include the time dimension (num_frames)
        # We will repeat the graph features across time frames
        x = x.unsqueeze(0)  # Add batch dimension: shape becomes [1, NUM_JOINTS, 128]
        x = x.permute(0, 2, 1)  # Change shape to [1, 128, NUM_JOINTS] (channels, joints)
        x = x.repeat(1, 1, self.num_frames)  # Repeat across time frames: [1, 128, NUM_JOINTS, NUM_FRAMES]

        # Apply the temporal convolution (across time)
        x = x.unsqueeze(3)  # Add time_steps dimension: [1, 128, NUM_JOINTS, NUM_FRAMES, 1]
        x = F.relu(self.temporal_conv(x))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten all but the batch dimension

        # Output classification
        x = self.fc(x)
        return x
# Initialize the model, optimizer, and loss function
model = STGCN(in_channels=NUM_FEATURES, out_channels=NUM_CLASSES, num_joints=NUM_JOINTS, num_frames=NUM_FRAMES)

print(model)
model.to(device)


# # Training the ST-GCN

# In[ ]:


# Define a function to save the model checkpoint
def save_checkpoint(model, optimizer, epoch, loss, filename="last.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")


# In[ ]:


num_epochs = 200
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# In[ ]:


# Weights and Biases 
import wandb
wandb.init(
    # set the wandb project where this run will be logged
    project="Hand_Gesture_Recog",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "STGCN",
    "dataset": "IPN_Hand_Gesture",
    "epochs": num_epochs,
    }
)


# In[94]:


accuracy_scores = []
# Function to train the model
def train(model, data_frames, optimizer, criterion, epochs=200):
    model.train()
    for epoch in range(epochs):
        for data_frame in data_frames:
            optimizer.zero_grad()
            # Concatenate data frames along time dimension
            data_frame.to(device)
            out = model(data_frame)  # Only passing one frame for simplicity here (modify for sequence)
    
            # Compute loss and backpropagate
            loss = criterion(out, data_frame.y)
            loss.backward()
            optimizer.step()

            pred = out.argmax(dim=1)  # Get the predicted class
            correct = (pred == data_frame.y).sum()  # Compare with true label
            accuracy = correct / len(data_frame.y)
            accuracy_scores.append(accuracy.item())
    
        if epoch % 5 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            wandb.log({"train/loss": loss, "train/acc": accuracy})
            save_checkpoint(model, optimizer, epoch, loss,os.path.join('train_logs', model_name, 'last.pth' ))

    return np.mean(np.array(accuracy_scores))

# Train the model on the toy data
accuracy = train(model, train_data_frames, optimizer, criterion, epochs=num_epochs)
print(f'Accuracy on train data: {accuracy * 100:.2f}%')


# # Evaluation

# In[28]:


import numpy as np
def test(model, data_frames):
    model.eval()
    with torch.no_grad():
        accuracy_scores = []
        for data_frame in data_frames:
            data_frame.to(device)
            out = model(data_frame)  # Run the forward pass
            loss = criterion(out, data_frame.y)
            pred = out.argmax(dim=1)  # Get the predicted class
            correct = (pred == data_frame.y).sum()  # Compare with true label
            accuracy = correct / len(data_frame.y)
            accuracy_scores.append(accuracy.item())
            wandb.log({"test/acc": accuracy, "test/loss": loss})
        return np.mean(np.array(accuracy_scores))


# Test the model on the first frame (toy data)
accuracy = test(model, test_data_frames)
print(f'Accuracy on test data: {accuracy * 100:.2f}%')

wandb.finish()

