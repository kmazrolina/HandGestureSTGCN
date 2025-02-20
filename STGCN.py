import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class STGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_joints, num_frames):
        super(STGCN, self).__init__()
        self.num_frames = num_frames
        self.num_joints = num_joints
        # Define two graph convolution layers
        self.conv1 = GCNConv(in_channels, 64)  # First graph convolution layer
        self.conv2 = GCNConv(64, 128)  # Second graph convolution layer
        # Temporal convolution layer to capture the temporal dependencies
        self.temporal_conv = nn.Conv2d(128, 128, kernel_size=(3, 1), padding=(1, 0))  # Temporal convolution
        # Fully connected layer for classification
        self.fc = nn.Linear(128 * num_joints * num_frames, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
    
        # Graph convolution layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Reshape x to include the time dimension
        # After graph convolutions, we have x with shape [300, 128]
        # We need to reshape it into [1, num_frames, num_joints, 128]
        x = x.view(1, self.num_frames, self.num_joints, -1)  # Shape: [1, num_frames, num_joints, 128]
        
        # Now apply temporal convolution
        # Reshape to [batch_size, 128, num_joints, num_frames] to match expected input format for Conv2d
        x = x.permute(0, 3, 2, 1)  # Shape: [batch_size, 128, num_joints, num_frames]
        x = F.relu(self.temporal_conv(x))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten all but the batch dimension

        # Output classification
        x = self.fc(x)
        return x